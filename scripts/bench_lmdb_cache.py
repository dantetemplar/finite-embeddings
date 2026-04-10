import shutil
import tempfile
import time
from multiprocessing import Process
from threading import Thread

import lmdb
import numpy as np

N = 50_000
DIM = 64
BATCH = 512
REPEATS = 2

THREADS = 4
PROCESSES = 4

# --------------------------------------------------
# Utils
# --------------------------------------------------


def random_key(i):
    return f"key_{i}".encode()


def random_vec():
    return np.random.rand(DIM).astype(np.float32)


def timer(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def split_range(n, parts):
    step = n // parts
    return [(i * step, (i + 1) * step if i < parts - 1 else n) for i in range(parts)]


# --------------------------------------------------
# Approaches
# --------------------------------------------------


def read_txn_and_write_txt_for_each_item(env, start, end):
    for i in range(start, end):
        k = random_key(i)

        with env.begin(write=False) as txn:
            val = txn.get(k)

        if val is None:
            with env.begin(write=True) as txn:
                txn.put(k, random_vec().tobytes())


def write_txn_for_each_item(env, start, end):
    for i in range(start, end):
        k = random_key(i)

        with env.begin(write=True) as txn:
            val = txn.get(k)
            if val is None:
                txn.put(k, random_vec().tobytes())


def read_one_write_buffered(env, start, end):
    buf = []

    for i in range(start, end):
        k = random_key(i)

        with env.begin(write=False) as txn:
            val = txn.get(k)

        if val is None:
            buf.append((k, random_vec()))

        if len(buf) >= BATCH:
            with env.begin(write=True) as txn:
                for k_, v_ in buf:
                    txn.put(k_, v_.tobytes())
            buf.clear()

    if buf:
        with env.begin(write=True) as txn:
            for k_, v_ in buf:
                txn.put(k_, v_.tobytes())


def read_chunk_write_buffered(env, start, end):
    buf = []

    for chunk_start in range(start, end, BATCH):
        chunk_end = min(chunk_start + BATCH, end)
        chunk = [random_key(i) for i in range(chunk_start, chunk_end)]

        with env.begin(write=False) as txn:
            for k in chunk:
                val = txn.get(k)
                if val is None:
                    buf.append((k, random_vec()))

        if len(buf) >= BATCH:
            with env.begin(write=True) as txn:
                for k_, v_ in buf:
                    txn.put(k_, v_.tobytes())
            buf.clear()

    if buf:
        with env.begin(write=True) as txn:
            for k_, v_ in buf:
                txn.put(k_, v_.tobytes())



APPROACHES = {
    "read_txn_and_write_txt_for_each_item": read_txn_and_write_txt_for_each_item,
    "write_txn_for_each_item": write_txn_for_each_item,
    "read_one_write_buffered": read_one_write_buffered,
    "read_chunk_write_buffered": read_chunk_write_buffered,
}

# --------------------------------------------------
# Modes
# --------------------------------------------------


def run_single(env, fn):
    fn(env, 0, N)


def run_threads(env, fn):
    threads = []
    for start, end in split_range(N, THREADS):
        t = Thread(target=fn, args=(env, start, end))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def proc_worker(path, cfg, fn, start, end):
    env = lmdb.open(path, **cfg)
    fn(env, start, end)
    env.close()


def run_processes(path, cfg, fn):
    procs = []
    for start, end in split_range(N, PROCESSES):
        p = Process(target=proc_worker, args=(path, cfg, fn, start, end))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


MODES = ["single", "threads", "processes"]


def is_valid_combo(cfg_name, mode):
    # lock=False disables LMDB locking and is only safe in this harness for single mode.
    if cfg_name == "nolock" and mode != "single":
        return False, "lock=False is unsafe for concurrent threads/processes"
    return True, ""

# --------------------------------------------------
# Configs
# --------------------------------------------------

MAP_SIZE = 1024 * 1024 * 1024 * 2

CONFIGS = {
    "default": {
        "map_size": MAP_SIZE,
    },
    "writemap": {
        "map_size": MAP_SIZE,
        "writemap": True,
    },
    "no_sync": {
        "map_size": MAP_SIZE,
        "sync": False,
    },
    "fast": {
        "map_size": MAP_SIZE,
        "writemap": True,
        "map_async": True,
        "sync": False,
        "readahead": False,
    },
    "nolock": {
        "map_size": MAP_SIZE,
        "lock": False,
    },
    "metasync_off": {
        "map_size": MAP_SIZE,
        "sync": False,
        "metasync": False,
    },
    "max_readers_high": {
        "map_size": MAP_SIZE,
        "max_readers": 1024,
    },
}

# --------------------------------------------------
# Runner
# --------------------------------------------------


def run_one(cfg_name, cfg, mode, app_name, fn):
    times = []

    for repeat_idx in range(REPEATS):
        path = tempfile.mkdtemp()

        env = lmdb.open(path, **cfg)

        if mode == "single":
            dt = timer(lambda: run_single(env, fn))

        elif mode == "threads":
            dt = timer(lambda: run_threads(env, fn))

        elif mode == "processes":
            env.close()
            dt = timer(lambda: run_processes(path, cfg, fn))

        else:
            raise ValueError

        times.append(dt)
        print(
            f"[{cfg_name}/{mode}/{app_name}] "
            f"repeat {repeat_idx + 1}/{REPEATS}: {dt:.3f}s"
        )

        if mode != "processes":
            env.close()

        shutil.rmtree(path)

    avg = sum(times) / len(times)
    ops = N / avg
    return avg, ops


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    print(f"N={N}, DIM={DIM}, BATCH={BATCH}")
    print(f"THREADS={THREADS}, PROCESSES={PROCESSES}, REPEATS={REPEATS}\n")

    results = []

    total_runs = len(CONFIGS) * len(MODES) * len(APPROACHES)
    run_idx = 0

    for cfg_name, cfg in CONFIGS.items():
        for mode in MODES:
            valid, reason = is_valid_combo(cfg_name, mode)
            if not valid:
                print(f"\n[skip] {cfg_name}/{mode}: {reason}")
                continue
            for app_name, fn in APPROACHES.items():
                run_idx += 1
                print(
                    f"\n[{run_idx}/{total_runs}] "
                    f"running {cfg_name}/{mode}/{app_name}..."
                )
                avg, ops = run_one(cfg_name, cfg, mode, app_name, fn)
                results.append((cfg_name, mode, app_name, avg, ops))
                print(
                    f"[{run_idx}/{total_runs}] done {cfg_name}/{mode}/{app_name}: "
                    f"avg={avg:.3f}s, ops={ops:.0f}/s"
                )

    # --------------------------------------------------
    # TABLE
    # --------------------------------------------------
    print("\n=== FULL TABLE ===")
    print(f"{'config':10s} | {'mode':10s} | {'approach':18s} | {'time':8s} | {'ops/sec':10s}")
    print("-" * 70)

    for cfg, mode, app, avg, ops in results:
        print(f"{cfg:10s} | {mode:10s} | {app:18s} | {avg:8.3f} | {ops:10.0f}")

    # --------------------------------------------------
    # SORTED
    # --------------------------------------------------
    print("\n=== SORTED BY SPEED ===")
    results.sort(key=lambda x: x[4], reverse=True)

    for cfg, mode, app, avg, ops in results:
        print(f"{ops:10.0f} ops/s | {cfg:10s} | {mode:10s} | {app:18s}")
