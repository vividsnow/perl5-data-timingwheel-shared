/*
 * timingwheel.h -- Shared-memory hashed timing wheel for Linux
 *
 * O(1) timer scheduling: a single-level hashed timing wheel of num_slots buckets
 * over a fixed pool of timers.  Scheduling a timer to fire in `delay` ticks and
 * cancelling a timer are O(1); advancing the wheel one tick fires the timers due
 * in the current bucket.  Timers whose delay exceeds one full rotation carry a
 * "rounds" counter decremented once per rotation.  The wheel lives in a shared
 * mapping so several processes schedule into and advance one clock; a
 * write-preferring futex rwlock with reader-slot dead-process recovery guards
 * mutation.  Each timer carries an arbitrary 64-bit payload returned when it
 * fires.
 *
 * Layout: Header -> reader_slots[1024] -> slots[num_slots] -> timers[capacity]
 */

#ifndef TW_H
#define TW_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "timingwheel.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define TW_MAGIC        0x4C485754  /* TimingWheel */
#define TW_VERSION      1
#define TW_ERR_BUFLEN   256
#ifndef TW_READER_SLOTS
#define TW_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define TW_MIN_SLOTS    1
#define TW_MAX_SLOTS    0x1000000U    /* 2^24 wheel slots */
#define TW_MIN_CAP      1
#define TW_MAX_CAP      0x1000000U    /* 2^24 concurrent timers (index fits uint32, < TW_NIL) */
#define TW_NIL          0xFFFFFFFFU   /* empty list link / free-list terminator */

#define TW_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, TW_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} TwReaderSlot;

struct TwHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t num_slots;               /* 8   wheel resolution (buckets) */
    uint32_t capacity;                /* 12  max concurrent timers */
    uint64_t now;                     /* 16  absolute tick counter */
    uint32_t cur;                     /* 24  current wheel position (now % num_slots) */
    uint32_t free_head;               /* 28  free-list head (timer index) or TW_NIL */
    uint64_t count;                   /* 32  active timers */
    uint64_t slots_off;               /* 40  offset of the slot heads (num_slots uint32) */
    uint64_t timers_off;              /* 48  offset of the timer pool */
    uint64_t total_size;              /* 56 */
    uint64_t reader_slots_off;        /* 64 */
    uint32_t rwlock;                  /* 72 */
    uint32_t rwlock_waiters;          /* 76 */
    uint32_t rwlock_writers_waiting;  /* 80 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot */
    uint64_t stat_ops;                /* 88 */
    uint8_t  _pad[160];               /* 96..255 */
};
typedef struct TwHeader TwHeader;

_Static_assert(sizeof(TwHeader) == 256, "TwHeader must be 256 bytes");

/* One timer: a payload plus the number of full wheel rotations remaining before
 * it fires, linked into its bucket by a doubly-linked list (prev/next) for O(1)
 * cancellation.  state is 0 when the timer sits on the free list, 1 when active. */
typedef struct {
    uint64_t payload;
    uint64_t rounds;
    uint32_t prev;
    uint32_t next;
    uint32_t slot;
    uint32_t state;
} TwTimer;
_Static_assert(sizeof(TwTimer) == 32, "TwTimer must be 32 bytes");

/* ---- Process-local handle ---- */

typedef struct TwHandle {
    TwHeader     *hdr;
    TwReaderSlot *reader_slots;  /* TW_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      slots_off;     /* validated offsets, cached: never re-read from the peer-writable header */
    uint64_t      timers_off;
    uint32_t      num_slots;     /* cached */
    uint32_t      capacity;      /* cached */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* tw_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} TwHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define TW_RWLOCK_SPIN_LIMIT 32
#define TW_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void tw_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define TW_RWLOCK_WRITER_BIT 0x80000000U
#define TW_RWLOCK_PID_MASK   0x7FFFFFFFU
#define TW_RWLOCK_WR(pid)    (TW_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & TW_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int tw_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void tw_recover_stale_lock(TwHandle *h, uint32_t observed_rwlock) {
    TwHeader *hdr = h->hdr;
    uint32_t mypid = TW_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec tw_lock_timeout = { TW_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t tw_fork_gen = 1;
static pthread_once_t tw_atfork_once = PTHREAD_ONCE_INIT;
static void tw_on_fork_child(void) {
    __atomic_add_fetch(&tw_fork_gen, 1, __ATOMIC_RELAXED);
}
static void tw_atfork_init(void) {
    pthread_atfork(NULL, NULL, tw_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void tw_claim_reader_slot(TwHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&tw_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&tw_atfork_once, tw_atfork_init);
    /* Re-read after pthread_once: tw_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&tw_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % TW_READER_SLOTS;
    for (uint32_t i = 0; i < TW_READER_SLOTS; i++) {
        uint32_t s = (start + i) % TW_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and tw_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void tw_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  tw_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void tw_drain_dead_slot(TwHandle *h, uint32_t i, uint32_t pid) {
    TwHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    tw_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) tw_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void tw_recover_dead_readers(TwHandle *h) {
    if (!h->reader_slots) return;
    TwHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < TW_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (tw_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        tw_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < TW_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < TW_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && tw_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < TW_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || tw_pid_alive(p)) continue;
                tw_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void tw_recover_after_timeout(TwHandle *h) {
    TwHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= TW_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & TW_RWLOCK_PID_MASK;
        if (!tw_pid_alive(pid))
            tw_recover_stale_lock(h, val);
    } else {
        tw_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void tw_park_reader(TwHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void tw_unpark_reader(TwHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void tw_park_writer(TwHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void tw_unpark_writer(TwHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void tw_reader_enter(TwHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void tw_reader_leave(TwHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void tw_rwlock_rdlock(TwHandle *h) {
    tw_claim_reader_slot(h);
    TwHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    tw_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < TW_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < TW_RWLOCK_SPIN_LIMIT, 1)) {
            tw_rwlock_spin_pause();
            continue;
        }
        tw_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= TW_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &tw_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                tw_unpark_reader(h);
                tw_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        tw_unpark_reader(h);
        spin = 0;
    }
}

static inline void tw_rwlock_rdunlock(TwHandle *h) {
    TwHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    tw_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void tw_rwlock_wrlock(TwHandle *h) {
    tw_claim_reader_slot(h);  /* refresh cached_pid across fork */
    TwHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = TW_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < TW_RWLOCK_SPIN_LIMIT, 1)) {
            tw_rwlock_spin_pause();
            continue;
        }
        tw_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &tw_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                tw_unpark_writer(h);
                tw_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        tw_unpark_writer(h);
        spin = 0;
    }
}

static inline void tw_rwlock_wrunlock(TwHandle *h) {
    TwHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> slots[num_slots] -> timers[capacity]
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets.
 * Layout: Header -> reader_slots[1024] -> slots[num_slots] -> timers[capacity] */
typedef struct { uint64_t reader_slots, slots, timers, total; } TwLayout;

static inline TwLayout tw_layout_for(uint32_t num_slots, uint32_t capacity) {
    TwLayout L;
    L.reader_slots = sizeof(TwHeader);
    L.slots        = L.reader_slots + (uint64_t)TW_READER_SLOTS * sizeof(TwReaderSlot);
    L.slots        = (L.slots + 7) & ~(uint64_t)7;
    L.timers       = L.slots + (uint64_t)num_slots * sizeof(uint32_t);
    L.timers       = (L.timers + 7) & ~(uint64_t)7;
    L.total        = L.timers + (uint64_t)capacity * sizeof(TwTimer);
    return L;
}

static inline uint64_t tw_total_size(uint32_t num_slots, uint32_t capacity) {
    return tw_layout_for(num_slots, capacity).total;
}

static inline uint32_t *tw_slots(TwHandle *h) { return (uint32_t *)((char *)h->base + h->slots_off); }
static inline TwTimer  *tw_timer(TwHandle *h, uint64_t i) { return (TwTimer *)((char *)h->base + h->timers_off) + i; }

static inline void tw_init_header(void *base, uint32_t num_slots, uint32_t capacity, uint64_t total) {
    TwLayout L = tw_layout_for(num_slots, capacity);
    TwHeader *hdr = (TwHeader *)base;
    memset(base, 0, (size_t)L.total);
    uint32_t *slots = (uint32_t *)((char *)base + L.slots);
    for (uint32_t s = 0; s < num_slots; s++) slots[s] = TW_NIL;      /* all buckets empty */
    TwTimer *timers = (TwTimer *)((char *)base + L.timers);
    for (uint32_t i = 0; i < capacity; i++) {                        /* thread the free list */
        timers[i].next  = (i + 1 < capacity) ? (i + 1) : TW_NIL;
        timers[i].prev  = TW_NIL;
        timers[i].slot  = TW_NIL;
        timers[i].state = 0;
    }
    hdr->magic            = TW_MAGIC;
    hdr->version          = TW_VERSION;
    hdr->num_slots        = num_slots;
    hdr->capacity         = capacity;
    hdr->now              = 0;
    hdr->cur              = 0;
    hdr->free_head        = capacity ? 0 : TW_NIL;
    hdr->count            = 0;
    hdr->slots_off        = L.slots;
    hdr->timers_off       = L.timers;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* Layer B trusted bound: number of timers guaranteed within the real mapping.
 * Equals capacity for a valid wheel; every timer index from shared memory is
 * checked against it so a corrupt link can never drive an access out of bounds. */
static inline uint64_t tw_timers_max(TwHandle *h) {
    if (h->timers_off >= h->mmap_size) return 0;
    return (h->mmap_size - h->timers_off) / sizeof(TwTimer);
}
#define TW_TIMER_OK(h, i) ((uint32_t)(i) != TW_NIL && (uint64_t)(i) < (uint64_t)(h)->capacity)

static inline TwHandle *tw_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    TwHeader *hdr = (TwHeader *)base;
    TwHandle *h = (TwHandle *)calloc(1, sizeof(TwHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (TwReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    h->slots_off    = hdr->slots_off;   /* single validated read of each geometry field */
    h->timers_off   = hdr->timers_off;
    h->num_slots    = hdr->num_slots;
    h->capacity     = hdr->capacity;
    h->mmap_size    = map_size;
    /* Layer B: clamp the cached capacity to the number of timers that actually fit */
    {
        uint64_t fit = tw_timers_max(h);
        if ((uint64_t)h->capacity > fit) h->capacity = (uint32_t)fit;
    }
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by tw_create reopen and tw_open_fd). */
static inline int tw_validate_header(const TwHeader *hdr, uint64_t file_size) {
    if (hdr->magic != TW_MAGIC) return 0;
    if (hdr->version != TW_VERSION) return 0;
    if (hdr->num_slots < TW_MIN_SLOTS || hdr->num_slots > TW_MAX_SLOTS) return 0;
    if (hdr->capacity < TW_MIN_CAP || hdr->capacity > TW_MAX_CAP) return 0;
    if (hdr->cur >= hdr->num_slots) return 0;
    if (hdr->count > hdr->capacity) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != tw_total_size(hdr->num_slots, hdr->capacity)) return 0;
    TwLayout L = tw_layout_for(hdr->num_slots, hdr->capacity);
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->slots_off != L.slots) return 0;
    if (hdr->timers_off != L.timers) return 0;
    return 1;
}

/* validate the requested wheel resolution + timer capacity */
static int tw_validate_args(uint64_t num_slots, uint64_t capacity, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (num_slots < TW_MIN_SLOTS || num_slots > TW_MAX_SLOTS) { TW_ERR("num_slots must be between 1 and 2^24"); return 0; }
    if (capacity < TW_MIN_CAP || capacity > TW_MAX_CAP) { TW_ERR("capacity must be between 1 and 2^24"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via tw_validate_header. */
static int tw_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { TW_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        TW_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    TW_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static TwHandle *tw_create(const char *path, uint64_t num_slots, uint64_t capacity, mode_t mode, char *errbuf) {
    if (!tw_validate_args(num_slots, capacity, errbuf)) return NULL;

    uint64_t total = tw_total_size((uint32_t)num_slots, (uint32_t)capacity);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { TW_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = tw_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { TW_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { TW_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(TwHeader)) {
            TW_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            TW_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { TW_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!tw_validate_header((TwHeader *)base, (uint64_t)st.st_size)) {
                TW_ERR("invalid timing-wheel file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return tw_setup(base, map_size, path, -1);
        }
    }
    tw_init_header(base, (uint32_t)num_slots, (uint32_t)capacity, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return tw_setup(base, map_size, path, -1);
}

static TwHandle *tw_create_memfd(const char *name, uint64_t num_slots, uint64_t capacity, char *errbuf) {
    if (!tw_validate_args(num_slots, capacity, errbuf)) return NULL;

    uint64_t total = tw_total_size((uint32_t)num_slots, (uint32_t)capacity);
    int fd = memfd_create(name ? name : "timingwheel", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { TW_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        TW_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { TW_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    tw_init_header(base, (uint32_t)num_slots, (uint32_t)capacity, total);
    return tw_setup(base, (size_t)total, NULL, fd);
}

static TwHandle *tw_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { TW_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(TwHeader)) { TW_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { TW_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!tw_validate_header((TwHeader *)base, (uint64_t)st.st_size)) {
        TW_ERR("invalid timing-wheel table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { TW_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return tw_setup(base, ms, NULL, myfd);
}

static void tw_destroy(TwHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&tw_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int tw_msync(TwHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * Timing-wheel operations (callers hold the lock)
 *
 * A single-level hashed wheel of num_slots buckets.  A timer due in `delay`
 * ticks goes into bucket (cur + delay) % num_slots with rounds = (delay-1) /
 * num_slots full rotations to wait; each time the hand passes its bucket the
 * timer either fires (rounds == 0) or decrements a round.  Timers live in a
 * fixed pool with a free list; each bucket is a doubly-linked list (prev/next)
 * so cancellation is O(1).  All timer indices read from shared memory are bounds
 * checked, so a corrupt link can never drive an out-of-mapping access.
 * ================================================================ */

/* unlink timer `t` from its current bucket's doubly-linked list */
static void tw_unlink(TwHandle *h, uint32_t t) {
    TwTimer *tm = tw_timer(h, t);
    uint32_t pv = tm->prev, nx = tm->next;
    if (TW_TIMER_OK(h, pv)) tw_timer(h, pv)->next = nx;
    else if (tm->slot < h->num_slots) tw_slots(h)[tm->slot] = nx;   /* t was the head */
    if (TW_TIMER_OK(h, nx)) tw_timer(h, nx)->prev = pv;
}

/* return timer `t` to the free list (caller has already unlinked it) */
static void tw_free(TwHandle *h, uint32_t t) {
    TwTimer *tm = tw_timer(h, t);
    tm->state = 0;
    tm->prev  = TW_NIL;
    tm->slot  = TW_NIL;
    tm->next  = h->hdr->free_head;
    h->hdr->free_head = t;
}

/* schedule a timer to fire in `delay` ticks (>= 1); returns its id, or -1 if the
 * pool is full.  (caller holds the write lock) */
static int64_t tw_add_locked(TwHandle *h, uint64_t delay, uint64_t payload) {
    if (h->num_slots == 0 || h->capacity == 0) return -1;
    uint32_t t = h->hdr->free_head;
    if (!TW_TIMER_OK(h, t)) return -1;                     /* full (or corrupt free head) */
    if (tw_timer(h, t)->state != 0) return -1;             /* Layer B: free head points at an
                                                              active timer -- refuse, don't alias it */
    if (delay < 1) delay = 1;                              /* minimum effective delay is one tick */
    h->hdr->free_head = tw_timer(h, t)->next;              /* pop the free list */

    uint32_t slot = (uint32_t)(((uint64_t)h->hdr->cur + delay) % h->num_slots);
    uint64_t rounds = (delay - 1) / h->num_slots;

    TwTimer *tm = tw_timer(h, t);
    tm->payload = payload;
    tm->rounds  = rounds;
    tm->slot    = slot;
    tm->state   = 1;
    tm->prev    = TW_NIL;
    uint32_t head = tw_slots(h)[slot];                     /* prepend to the bucket */
    tm->next = head;
    if (TW_TIMER_OK(h, head)) tw_timer(h, head)->prev = t;
    tw_slots(h)[slot] = t;
    h->hdr->count++;
    return (int64_t)t;
}

/* cancel timer `t`; returns 1 if it was active and is now removed, else 0.
 * (caller holds the write lock) */
static int tw_cancel_locked(TwHandle *h, uint64_t t) {
    if (!TW_TIMER_OK(h, t)) return 0;
    TwTimer *tm = tw_timer(h, t);
    if (tm->state != 1) return 0;                          /* free / already fired */
    tw_unlink(h, (uint32_t)t);
    tw_free(h, (uint32_t)t);
    if (h->hdr->count) h->hdr->count--;
    return 1;
}

/* advance the wheel by `ticks`, collecting fired payloads into out[] (capped at
 * out_cap); returns the number fired.  (caller holds the write lock) */
static uint64_t tw_advance_locked(TwHandle *h, uint64_t ticks, uint64_t *out, uint64_t out_cap) {
    uint64_t fired = 0;
    if (h->num_slots == 0) return 0;
    for (uint64_t k = 0; k < ticks; k++) {
        h->hdr->now++;
        h->hdr->cur = (uint32_t)(((uint64_t)h->hdr->cur + 1) % h->num_slots);
        uint32_t t = tw_slots(h)[h->hdr->cur];
        uint64_t guard = 0;
        while (TW_TIMER_OK(h, t) && guard++ <= (uint64_t)h->capacity) {
            TwTimer *tm = tw_timer(h, t);
            if (tm->state != 1) break;                     /* Layer B: a corrupt link led into a
                                                              freed/free-list node -- stop the walk */
            uint32_t nx = tm->next;
            if (tm->rounds == 0) {                         /* due now: fire */
                tw_unlink(h, t);
                tw_free(h, t);
                if (h->hdr->count) h->hdr->count--;
                if (fired < out_cap) out[fired++] = tm->payload;   /* cap BOTH the write and the count */
            } else {
                tm->rounds--;                              /* one rotation closer */
            }
            t = nx;
        }
    }
    return fired;
}

/* reset to an empty wheel: rethread the free list, clear the buckets, reset time.
 * (caller holds the write lock) */
static inline void tw_clear_locked(TwHandle *h) {
    uint64_t ns = h->num_slots, cap = h->capacity;
    uint64_t smax = (h->slots_off < h->mmap_size) ? (h->mmap_size - h->slots_off) / sizeof(uint32_t) : 0;
    if (ns > smax) ns = smax;                              /* Layer B */
    uint64_t tmax = tw_timers_max(h);
    if (cap > tmax) cap = tmax;
    uint32_t *slots = tw_slots(h);
    for (uint64_t s = 0; s < ns; s++) slots[s] = TW_NIL;
    for (uint64_t i = 0; i < cap; i++) {
        TwTimer *tm = tw_timer(h, i);
        tm->next  = (i + 1 < cap) ? (uint32_t)(i + 1) : TW_NIL;
        tm->prev  = TW_NIL;
        tm->slot  = TW_NIL;
        tm->state = 0;
    }
    h->hdr->now = 0;
    h->hdr->cur = 0;
    h->hdr->count = 0;
    h->hdr->free_head = cap ? 0 : TW_NIL;
}

#endif /* TW_H */
