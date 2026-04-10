import argparse
import hashlib
import json
import string
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def random_text(rng: np.random.Generator, size: int = 192) -> str:
    chars = string.ascii_letters + string.digits + " .,;:-_/()[]{}"
    alphabet = np.fromiter(chars, dtype="<U1")
    return "".join(rng.choice(alphabet, size=size))


def build_dense_payload(
    model_id: str,
    truncate_dim: int | None,
    dense_prompt: str,
    dense_task: str | None,
    text: str,
) -> bytes:
    dim = 0 if truncate_dim is None else truncate_dim
    return b"".join(
        [
            b"dense\x00",
            model_id.encode("utf-8"),
            b"\x00",
            dim.to_bytes(4, "little", signed=False),
            b"\x00",
            json.dumps(
                {"dense_prompt": dense_prompt, "dense_task": dense_task},
                separators=(",", ":"),
            ).encode("utf-8"),
            b"\x00",
            text.encode("utf-8"),
        ]
    )


def dense_digest_json(
    factory: Callable[[], Any],
    model_id: str,
    truncate_dim: int | None,
    dense_prompt: str,
    dense_task: str | None,
    text: str,
) -> bytes:
    h = factory()
    h.update(b"dense\x00")
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    dim = 0 if truncate_dim is None else truncate_dim
    h.update(dim.to_bytes(4, "little", signed=False))
    h.update(b"\x00")
    h.update(
        json.dumps(
            {"dense_prompt": dense_prompt, "dense_task": dense_task},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.digest()


def dense_digest_tagged_fields(
    factory: Callable[[], Any],
    model_id: str,
    truncate_dim: int | None,
    dense_prompt: str,
    dense_task: str | None,
    text: str,
) -> bytes:
    h = factory()
    h.update(b"dense\x00")
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    dim = 0 if truncate_dim is None else truncate_dim
    h.update(dim.to_bytes(4, "little", signed=False))
    h.update(b"\x00")
    h.update(b"dense_prompt\x00")
    h.update(dense_prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(b"dense_task\x00")
    task_value = "" if dense_task is None else dense_task
    h.update(task_value.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.digest()


def build_sparse_payload(
    model_id: str,
    max_active_dims: int | None,
    pruning_ratio: float | None,
    sparse_task: str | None,
    text: str,
) -> bytes:
    return b"".join(
        [
            b"sparse\x00",
            model_id.encode("utf-8"),
            b"\x00",
            json.dumps(
                {
                    "mad": max_active_dims,
                    "pr": pruning_ratio,
                    "sparse_task": sparse_task,
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            b"\x00",
            text.encode("utf-8"),
        ]
    )


def build_bge_payload(prefix: bytes, model_id: str, text: str) -> bytes:
    return b"".join(
        [
            prefix,
            model_id.encode("utf-8"),
            b"\x00",
            text.encode("utf-8"),
        ]
    )


def make_dataset(n: int, rng: np.random.Generator) -> list[bytes]:
    payloads: list[bytes] = []
    for i in range(n):
        text = random_text(rng)
        model = f"model-{i % 7}"
        payloads.append(
            build_dense_payload(
                model_id=model,
                truncate_dim=[None, 256, 512, 1024][i % 4],
                dense_prompt=["", "query:", "doc:", "title:"][i % 4],
                dense_task=[None, "query", "document"][i % 3],
                text=text,
            )
        )
        payloads.append(
            build_sparse_payload(
                model_id=model,
                max_active_dims=[None, 64, 128, 256][i % 4],
                pruning_ratio=[None, 0.05, 0.1, 0.2][i % 4],
                sparse_task=[None, "query", "document"][i % 3],
                text=text,
            )
        )
        payloads.append(build_bge_payload(b"bge_dense\x00", model, text))
        payloads.append(build_bge_payload(b"bge_sparse\x00", model, text))
        payloads.append(build_bge_payload(b"bge_colbert\x00", model, text))
    return payloads


def make_dense_inputs(
    n: int, rng: np.random.Generator
) -> list[tuple[str, int | None, str, str | None, str]]:
    rows: list[tuple[str, int | None, str, str | None, str]] = []
    for i in range(n):
        rows.append(
            (
                f"model-{i % 7}",
                [None, 256, 512, 1024][i % 4],
                ["", "query:", "doc:", "title:"][i % 4],
                [None, "query", "document"][i % 3],
                random_text(rng),
            )
        )
    return rows


def bench_hash(
    factory: Callable[[], Any], payloads: list[bytes], repeats: int
) -> tuple[float, float]:
    best = float("inf")
    total_bytes = sum(len(p) for p in payloads)
    for _ in range(repeats):
        t0 = time.perf_counter()
        for payload in payloads:
            h = factory()
            h.update(payload)
            _ = h.digest()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    payloads_per_sec = len(payloads) / best
    mb_per_sec = (total_bytes / (1024 * 1024)) / best
    return payloads_per_sec, mb_per_sec


def bench_dense_key_strategy(
    factory: Callable[[], Any],
    dense_inputs: list[tuple[str, int | None, str, str | None, str]],
    repeats: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    best_json = float("inf")
    best_tagged = float("inf")
    total_text_bytes = sum(len(row[4].encode("utf-8")) for row in dense_inputs)

    for _ in range(repeats):
        t0 = time.perf_counter()
        for model_id, truncate_dim, dense_prompt, dense_task, text in dense_inputs:
            _ = dense_digest_json(
                factory, model_id, truncate_dim, dense_prompt, dense_task, text
            )
        dt_json = time.perf_counter() - t0
        best_json = min(best_json, dt_json)

        t1 = time.perf_counter()
        for model_id, truncate_dim, dense_prompt, dense_task, text in dense_inputs:
            _ = dense_digest_tagged_fields(
                factory, model_id, truncate_dim, dense_prompt, dense_task, text
            )
        dt_tagged = time.perf_counter() - t1
        best_tagged = min(best_tagged, dt_tagged)

    json_ops = len(dense_inputs) / best_json
    tagged_ops = len(dense_inputs) / best_tagged
    json_mb = (total_text_bytes / (1024 * 1024)) / best_json
    tagged_mb = (total_text_bytes / (1024 * 1024)) / best_tagged
    return (json_ops, json_mb), (tagged_ops, tagged_mb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark hash functions for LMDB cache key generation."
    )
    parser.add_argument(
        "--samples", type=int, default=100_000, help="Base sample count."
    )
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    payloads = make_dataset(args.samples, rng)
    dense_inputs = make_dense_inputs(args.samples, rng)

    hashers: list[tuple[str, Callable[[], Any]]] = [
        ("blake2b-32", lambda: hashlib.blake2b(digest_size=32)),
        ("blake2s-32", lambda: hashlib.blake2s(digest_size=32)),
        ("sha256", hashlib.sha256),
        ("sha3_256", hashlib.sha3_256),
        ("sha1", hashlib.sha1),
        ("md5", hashlib.md5),
    ]

    results: list[tuple[str, float, float]] = []
    for name, factory in hashers:
        payloads_per_sec, mb_per_sec = bench_hash(factory, payloads, args.repeats)
        results.append((name, payloads_per_sec, mb_per_sec))
        print(
            f"[done] {name:10s} | {payloads_per_sec:12.0f} payload/s | {mb_per_sec:10.2f} MiB/s"
        )

    results.sort(key=lambda row: row[1], reverse=True)
    print("\n=== Sorted by payload/s (higher is better) ===")
    print(f"{'hash':10s} | {'payload/s':>12s} | {'MiB/s':>10s}")
    print("-" * 40)
    for name, payloads_per_sec, mb_per_sec in results:
        print(f"{name:10s} | {payloads_per_sec:12.0f} | {mb_per_sec:10.2f}")

    print("\n=== Dense Key Strategy (blake2b-32) ===")
    (json_ops, json_mb), (tagged_ops, tagged_mb) = bench_dense_key_strategy(
        lambda: hashlib.blake2b(digest_size=32),
        dense_inputs,
        args.repeats,
    )
    print(f"{'strategy':26s} | {'ops/s':>12s} | {'MiB/s(text)':>12s}")
    print("-" * 58)
    print(f"{'json.dumps metadata':26s} | {json_ops:12.0f} | {json_mb:12.2f}")
    print(f"{'tagged h.update fields':26s} | {tagged_ops:12.0f} | {tagged_mb:12.2f}")


if __name__ == "__main__":
    main()
