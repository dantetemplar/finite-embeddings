from __future__ import annotations

import base64
import gzip
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import httpx
import lmdb
import numpy as np


class EmbedRequestDict(TypedDict):
    texts: list[str]
    dense_model_id: NotRequired[str | None]
    dense_truncate_dim: NotRequired[int | None]
    dense_prompt: NotRequired[str]
    dense_task: NotRequired[Literal["query", "document"] | None]
    sparse_model_id: NotRequired[str | None]
    sparse_max_active_dims: NotRequired[int | None]
    sparse_pruning_ratio: NotRequired[float | None]
    bge_model_id: NotRequired[str | None]
    sparse_task: NotRequired[Literal["query", "document"] | None]


class RerankRequestDict(TypedDict):
    reranker_model_id: str
    docs: list[str]
    query: NotRequired[str | None]
    queries: NotRequired[list[str] | None]


class DenseResponseDict(TypedDict):
    model_id: str
    shape: tuple[int, int]
    dtype: Literal["float32"]
    encoding: Literal["base64"]
    data: str


class SparseItemResponseDict(TypedDict):
    dim: int
    nnz: int
    indices_dtype: Literal["uint32"]
    values_dtype: Literal["float32"]
    indices: str
    values: str


class SparseResponseDict(TypedDict):
    model_id: str
    items: list[SparseItemResponseDict]
    encoding: Literal["base64"]


class ColbertItemResponseDict(TypedDict):
    shape: tuple[int, int]
    dtype: Literal["float32"]
    encoding: Literal["base64"]
    data: str


class BGEM3ResponseDict(TypedDict):
    model_id: str
    dense: DenseResponseDict
    sparse: SparseResponseDict
    colbert: list[ColbertItemResponseDict]


class EmbedResponseDict(TypedDict):
    texts_count: int
    dense: DenseResponseDict | None
    sparse: SparseResponseDict | None
    bgeM3: BGEM3ResponseDict | None


class ModelInfoDict(TypedDict):
    type: Literal["dense", "sparse", "reranker", "bgeM3"]
    id: str
    device: str
    dimensions: int | None
    dense_dimensions: NotRequired[int | None]
    sparse_dimensions: NotRequired[int | None]
    colbert_dimensions: NotRequired[int | None]
    batch_size: int | None


class ModelsResponseDict(TypedDict):
    models: list[ModelInfoDict]


class RerankResponseDict(TypedDict):
    model_id: str
    shape: tuple[int, int]
    scores: list[list[float]]


@dataclass(slots=True)
class DenseEmbeddings:
    model_id: str
    vectors: np.ndarray


@dataclass(slots=True)
class SparseEmbedding:
    dim: int
    indices: np.ndarray
    values: np.ndarray


@dataclass(slots=True)
class SparseEmbeddings:
    model_id: str
    items: list[SparseEmbedding]


@dataclass(slots=True)
class ParsedEmbedResponse:
    texts_count: int
    dense: DenseEmbeddings | None
    sparse: SparseEmbeddings | None
    bgeM3: "BGEM3Embeddings | None"


@dataclass(slots=True)
class BGEM3Embeddings:
    model_id: str
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    colbert: list[np.ndarray]


@dataclass(slots=True)
class ParsedRerankResponse:
    model_id: str
    shape: tuple[int, int]
    scores: list[list[float]]


@dataclass(slots=True)
class _EmbedCacheProgress:
    payload: EmbedRequestDict
    texts: list[str]
    misses: list[int]
    dense_model_id: str | None
    sparse_model_id: str | None
    bge_model_id: str | None
    dense_truncate_dim: int | None
    dense_prompt: str
    dense_task: Literal["query", "document"] | None
    sparse_max_active_dims: int | None
    sparse_pruning_ratio: float | None
    sparse_task: Literal["query", "document"] | None
    dense_vectors: list[np.ndarray | None] | None
    sparse_items: list[SparseEmbedding | None] | None
    bge_dense_vectors: list[np.ndarray | None] | None
    bge_sparse_items: list[SparseEmbedding | None] | None
    bge_colbert_items: list[np.ndarray | None] | None


def _decode_dense(dense: DenseResponseDict) -> DenseEmbeddings:
    raw = base64.b64decode(dense["data"])
    vectors = np.frombuffer(raw, dtype=np.float32)
    shape = tuple(dense["shape"])
    expected = shape[0] * shape[1]
    if vectors.size != expected:
        raise ValueError(
            f"Dense vector size mismatch: got {vectors.size}, expected {expected}."
        )
    return DenseEmbeddings(model_id=dense["model_id"], vectors=vectors.reshape(shape))


def _decode_sparse(sparse: SparseResponseDict) -> SparseEmbeddings:
    def _decode_sparse_item(item: SparseItemResponseDict) -> SparseEmbedding:
        indices = np.frombuffer(base64.b64decode(item["indices"]), dtype=np.uint32)
        values = np.frombuffer(base64.b64decode(item["values"]), dtype=np.float32)
        if indices.size != item["nnz"]:
            raise ValueError(
                f"Sparse indices size mismatch: got {indices.size}, expected {item['nnz']}."
            )
        if values.size != item["nnz"]:
            raise ValueError(
                f"Sparse values size mismatch: got {values.size}, expected {item['nnz']}."
            )
        return SparseEmbedding(dim=item["dim"], indices=indices, values=values)

    return SparseEmbeddings(
        model_id=sparse["model_id"],
        items=[_decode_sparse_item(item) for item in sparse["items"]],
    )


def _decode_colbert_item(item: ColbertItemResponseDict) -> np.ndarray:
    raw = base64.b64decode(item["data"])
    vectors = np.frombuffer(raw, dtype=np.float32)
    shape = tuple(item["shape"])
    expected = shape[0] * shape[1]
    if vectors.size != expected:
        raise ValueError(
            f"Colbert vector size mismatch: got {vectors.size}, expected {expected}."
        )
    return vectors.reshape(shape)


def _decode_bge_m3(bge_m3: BGEM3ResponseDict) -> BGEM3Embeddings:
    dense = _decode_dense(bge_m3["dense"])
    sparse = _decode_sparse(bge_m3["sparse"])
    colbert = [_decode_colbert_item(item) for item in bge_m3["colbert"]]
    return BGEM3Embeddings(
        model_id=bge_m3["model_id"], dense=dense, sparse=sparse, colbert=colbert
    )


def _normalize_dense_vectors_for_lmdb_cache(vectors: np.ndarray) -> np.ndarray:
    """Align with float16 LMDB storage so network and cache hits return the same dense values."""
    return np.asarray(vectors, dtype=np.float32).astype(np.float16).astype(np.float32)


def _normalize_colbert_vector_for_lmdb_cache(vectors: np.ndarray) -> np.ndarray:
    """Align colbert values with float16 LMDB storage for stable hit/miss behavior."""
    return np.asarray(vectors, dtype=np.float32).astype(np.float16).astype(np.float32)


class FiniteEmbeddingsClient:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
        use_cache: bool = False,
        cache_path: str | Path | None = None,
        cache_map_size: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        # Caller owns lifecycle and configuration (base_url, timeout, headers).
        self._client = client
        self._sync_client = sync_client
        if self._client is None and self._sync_client is None:
            raise ValueError(
                "Either client or sync_client must be provided, may be both."
            )
        self._use_cache = use_cache
        self._cache: lmdb.Environment | None = None
        if self._use_cache:
            cache_file = (
                Path(cache_path)
                if cache_path is not None
                else Path.home() / ".cache" / "finite-embeddings" / "client-cache.lmdb"
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache = lmdb.open(
                str(cache_file),
                map_size=cache_map_size,
                subdir=False,
                lock=True,
                readonly=False,
                readahead=True,
                meminit=False,
                max_dbs=1,
            )

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "client is not configured; pass client=httpx.AsyncClient(...) to FiniteEmbeddingsClient(...)."
            )
        return self._client

    @property
    def sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            raise RuntimeError(
                "sync_client is not configured; pass sync_client=httpx.Client(...) to FiniteEmbeddingsClient(...)."
            )
        return self._sync_client

    @staticmethod
    def _dense_cache_key(
        model_id: str,
        truncate_dim: int | None,
        dense_prompt: str,
        dense_task: Literal["query", "document"] | None,
        text: str,
    ) -> bytes:
        h = hashlib.blake2b(digest_size=32)
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

    @staticmethod
    def _sparse_cache_key(
        model_id: str,
        max_active_dims: int | None,
        pruning_ratio: float | None,
        sparse_task: Literal["query", "document"] | None,
        text: str,
    ) -> bytes:
        h = hashlib.blake2b(digest_size=32)
        h.update(b"sparse\x00")
        h.update(model_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(
            json.dumps(
                {
                    "mad": max_active_dims,
                    "pr": pruning_ratio,
                    "sparse_task": sparse_task,
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.digest()

    @staticmethod
    def _bge_dense_cache_key(model_id: str, text: str) -> bytes:
        h = hashlib.blake2b(digest_size=32)
        h.update(b"bge_dense\x00")
        h.update(model_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.digest()

    @staticmethod
    def _bge_sparse_cache_key(model_id: str, text: str) -> bytes:
        h = hashlib.blake2b(digest_size=32)
        h.update(b"bge_sparse\x00")
        h.update(model_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.digest()

    @staticmethod
    def _bge_colbert_cache_key(model_id: str, text: str) -> bytes:
        h = hashlib.blake2b(digest_size=32)
        h.update(b"bge_colbert\x00")
        h.update(model_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.digest()

    def _load_dense_vector(
        self, key: bytes, expected_dim: int | None
    ) -> np.ndarray | None:
        if self._cache is None:
            return None
        try:
            with self._cache.begin(write=False) as txn:
                raw = txn.get(key)
            if raw is None:
                return None
            if expected_dim is not None:
                vector = np.frombuffer(
                    raw, dtype=np.float16, count=expected_dim
                ).astype(np.float32)
                if vector.size != expected_dim:
                    return None
                return vector
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        except Exception:
            return None

    def _save_dense_vectors(self, items: list[tuple[bytes, np.ndarray]]) -> None:
        if self._cache is None or not items:
            return
        try:
            with self._cache.begin(write=True) as txn:
                for key, vector in items:
                    txn.put(key, np.asarray(vector, dtype=np.float16).tobytes())
        except Exception:
            return

    def _load_sparse_item(self, key: bytes) -> SparseEmbedding | None:
        if self._cache is None:
            return None
        try:
            with self._cache.begin(write=False) as txn:
                raw = txn.get(key)
            if raw is None or len(raw) < 8:
                return None
            dim, nnz = struct.unpack("<II", raw[:8])
            idx_end = 8 + (nnz * 4)
            val_end = idx_end + (nnz * 4)
            if len(raw) != val_end:
                return None
            indices = np.frombuffer(raw[8:idx_end], dtype=np.uint32).copy()
            values = np.frombuffer(raw[idx_end:val_end], dtype=np.float32).copy()
            return SparseEmbedding(dim=int(dim), indices=indices, values=values)
        except Exception:
            return None

    def _save_sparse_items(self, items: list[tuple[bytes, SparseEmbedding]]) -> None:
        if self._cache is None or not items:
            return
        try:
            with self._cache.begin(write=True) as txn:
                for key, item in items:
                    indices = np.asarray(item.indices, dtype=np.uint32)
                    values = np.asarray(item.values, dtype=np.float32)
                    if indices.size != values.size:
                        continue
                    header = struct.pack("<II", int(item.dim), int(indices.size))
                    txn.put(key, header + indices.tobytes() + values.tobytes())
        except Exception:
            return

    def _load_colbert_item(self, key: bytes) -> np.ndarray | None:
        if self._cache is None:
            return None
        try:
            with self._cache.begin(write=False) as txn:
                raw = txn.get(key)
            if raw is None or len(raw) < 8:
                return None
            rows, cols = struct.unpack("<II", raw[:8])
            expected_bytes = int(rows) * int(cols) * 2
            if len(raw) != (8 + expected_bytes):
                return None
            item = np.frombuffer(raw[8:], dtype=np.float16).astype(np.float32)
            if item.size != int(rows) * int(cols):
                return None
            return item.reshape((int(rows), int(cols)))
        except Exception:
            return None

    def _save_colbert_items(self, items: list[tuple[bytes, np.ndarray]]) -> None:
        if self._cache is None or not items:
            return
        try:
            with self._cache.begin(write=True) as txn:
                for key, item in items:
                    matrix = np.asarray(item, dtype=np.float32)
                    if matrix.ndim != 2:
                        continue
                    rows, cols = matrix.shape
                    header = struct.pack("<II", int(rows), int(cols))
                    txn.put(key, header + matrix.astype(np.float16).tobytes())
        except Exception:
            return

    async def models(self) -> ModelsResponseDict:
        response = await self.client.get("/models", headers={"Accept-Encoding": "gzip"})
        response.raise_for_status()
        return response.json()

    def models_sync(self) -> ModelsResponseDict:
        response = self.sync_client.get("/models", headers={"Accept-Encoding": "gzip"})
        response.raise_for_status()
        return response.json()

    async def _embed_remote(self, payload: EmbedRequestDict) -> ParsedEmbedResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = await self.client.post(
            "/embed",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        response.raise_for_status()
        raw = cast(EmbedResponseDict, response.json())
        return ParsedEmbedResponse(
            texts_count=raw["texts_count"],
            dense=_decode_dense(raw["dense"]) if raw["dense"] is not None else None,
            sparse=_decode_sparse(raw["sparse"]) if raw["sparse"] is not None else None,
            bgeM3=_decode_bge_m3(raw["bgeM3"]) if raw["bgeM3"] is not None else None,
        )

    def _embed_remote_sync(self, payload: EmbedRequestDict) -> ParsedEmbedResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self.sync_client.post(
            "/embed",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        response.raise_for_status()
        raw = cast(EmbedResponseDict, response.json())
        return ParsedEmbedResponse(
            texts_count=raw["texts_count"],
            dense=_decode_dense(raw["dense"]) if raw["dense"] is not None else None,
            sparse=_decode_sparse(raw["sparse"]) if raw["sparse"] is not None else None,
            bgeM3=_decode_bge_m3(raw["bgeM3"]) if raw["bgeM3"] is not None else None,
        )

    def _embed_cache_prepare(
        self, payload: EmbedRequestDict
    ) -> _EmbedCacheProgress | None:
        texts = payload.get("texts", [])
        dense_model_id = payload.get("dense_model_id")
        sparse_model_id = payload.get("sparse_model_id")
        bge_model_id = payload.get("bge_model_id")
        dense_truncate_dim = payload.get("dense_truncate_dim")
        dense_prompt = payload.get("dense_prompt") or ""
        dense_task = payload.get("dense_task")
        sparse_max_active_dims = payload.get("sparse_max_active_dims")
        sparse_pruning_ratio = payload.get("sparse_pruning_ratio")
        sparse_task = payload.get("sparse_task")
        if not texts or (
            dense_model_id is None and sparse_model_id is None and bge_model_id is None
        ):
            return None

        dense_vectors: list[np.ndarray | None] | None = (
            cast(list[np.ndarray | None], [None for _ in texts])
            if dense_model_id is not None
            else None
        )
        sparse_items: list[SparseEmbedding | None] | None = (
            cast(list[SparseEmbedding | None], [None for _ in texts])
            if sparse_model_id is not None
            else None
        )
        bge_dense_vectors: list[np.ndarray | None] | None = (
            cast(list[np.ndarray | None], [None for _ in texts])
            if bge_model_id is not None
            else None
        )
        bge_sparse_items: list[SparseEmbedding | None] | None = (
            cast(list[SparseEmbedding | None], [None for _ in texts])
            if bge_model_id is not None
            else None
        )
        bge_colbert_items: list[np.ndarray | None] | None = (
            cast(list[np.ndarray | None], [None for _ in texts])
            if bge_model_id is not None
            else None
        )

        misses: list[int] = []
        for idx, text in enumerate(texts):
            dense_hit = True
            sparse_hit = True
            bge_dense_hit = True
            bge_sparse_hit = True
            bge_colbert_hit = True
            if dense_model_id is not None and dense_vectors is not None:
                dense_key = self._dense_cache_key(
                    dense_model_id, dense_truncate_dim, dense_prompt, dense_task, text
                )
                cached_dense = self._load_dense_vector(dense_key, dense_truncate_dim)
                if cached_dense is None:
                    dense_hit = False
                else:
                    dense_vectors[idx] = cached_dense
            if sparse_model_id is not None and sparse_items is not None:
                sparse_key = self._sparse_cache_key(
                    sparse_model_id,
                    sparse_max_active_dims,
                    sparse_pruning_ratio,
                    sparse_task,
                    text,
                )
                cached_sparse = self._load_sparse_item(sparse_key)
                if cached_sparse is None:
                    sparse_hit = False
                else:
                    sparse_items[idx] = cached_sparse
            if bge_model_id is not None and bge_dense_vectors is not None:
                bge_dense_key = self._bge_dense_cache_key(bge_model_id, text)
                cached_bge_dense = self._load_dense_vector(
                    bge_dense_key, expected_dim=None
                )
                if cached_bge_dense is None:
                    bge_dense_hit = False
                else:
                    bge_dense_vectors[idx] = cached_bge_dense
            if bge_model_id is not None and bge_sparse_items is not None:
                bge_sparse_key = self._bge_sparse_cache_key(bge_model_id, text)
                cached_bge_sparse = self._load_sparse_item(bge_sparse_key)
                if cached_bge_sparse is None:
                    bge_sparse_hit = False
                else:
                    bge_sparse_items[idx] = cached_bge_sparse
            if bge_model_id is not None and bge_colbert_items is not None:
                bge_colbert_key = self._bge_colbert_cache_key(bge_model_id, text)
                cached_bge_colbert = self._load_colbert_item(bge_colbert_key)
                if cached_bge_colbert is None:
                    bge_colbert_hit = False
                else:
                    bge_colbert_items[idx] = cached_bge_colbert
            if (
                not dense_hit
                or not sparse_hit
                or not bge_dense_hit
                or not bge_sparse_hit
                or not bge_colbert_hit
            ):
                misses.append(idx)

        return _EmbedCacheProgress(
            payload=payload,
            texts=texts,
            misses=misses,
            dense_model_id=dense_model_id,
            sparse_model_id=sparse_model_id,
            bge_model_id=bge_model_id,
            dense_truncate_dim=dense_truncate_dim,
            dense_prompt=dense_prompt,
            dense_task=dense_task,
            sparse_max_active_dims=sparse_max_active_dims,
            sparse_pruning_ratio=sparse_pruning_ratio,
            sparse_task=sparse_task,
            dense_vectors=dense_vectors,
            sparse_items=sparse_items,
            bge_dense_vectors=bge_dense_vectors,
            bge_sparse_items=bge_sparse_items,
            bge_colbert_items=bge_colbert_items,
        )

    def _embed_cache_merge_remote(
        self, p: _EmbedCacheProgress, remote: ParsedEmbedResponse
    ) -> None:
        to_store_dense: list[tuple[bytes, np.ndarray]] = []
        to_store_sparse: list[tuple[bytes, SparseEmbedding]] = []
        to_store_bge_dense: list[tuple[bytes, np.ndarray]] = []
        to_store_bge_sparse: list[tuple[bytes, SparseEmbedding]] = []
        to_store_bge_colbert: list[tuple[bytes, np.ndarray]] = []
        for miss_row, idx in enumerate(p.misses):
            text = p.texts[idx]
            if p.dense_model_id is not None and p.dense_vectors is not None:
                if remote.dense is None:
                    raise ValueError(
                        "Dense cache expected dense embeddings in server response."
                    )
                vector = np.asarray(remote.dense.vectors[miss_row], dtype=np.float32)
                p.dense_vectors[idx] = vector
                dense_key = self._dense_cache_key(
                    p.dense_model_id,
                    p.dense_truncate_dim,
                    p.dense_prompt,
                    p.dense_task,
                    text,
                )
                to_store_dense.append((dense_key, vector))
            if p.sparse_model_id is not None and p.sparse_items is not None:
                if remote.sparse is None:
                    raise ValueError(
                        "Sparse cache expected sparse embeddings in server response."
                    )
                item = remote.sparse.items[miss_row]
                p.sparse_items[idx] = item
                sparse_key = self._sparse_cache_key(
                    p.sparse_model_id,
                    p.sparse_max_active_dims,
                    p.sparse_pruning_ratio,
                    p.sparse_task,
                    text,
                )
                to_store_sparse.append((sparse_key, item))
            if p.bge_model_id is not None:
                if remote.bgeM3 is None:
                    raise ValueError(
                        "BGE-M3 cache expected bgeM3 embeddings in server response."
                    )
                if (
                    p.bge_dense_vectors is None
                    or p.bge_sparse_items is None
                    or p.bge_colbert_items is None
                ):
                    raise ValueError("BGE-M3 cache merge failed: unresolved buffers.")
                bge_dense_vector = np.asarray(
                    remote.bgeM3.dense.vectors[miss_row], dtype=np.float32
                )
                p.bge_dense_vectors[idx] = bge_dense_vector
                bge_dense_key = self._bge_dense_cache_key(p.bge_model_id, text)
                to_store_bge_dense.append((bge_dense_key, bge_dense_vector))

                bge_sparse_item = remote.bgeM3.sparse.items[miss_row]
                p.bge_sparse_items[idx] = bge_sparse_item
                bge_sparse_key = self._bge_sparse_cache_key(p.bge_model_id, text)
                to_store_bge_sparse.append((bge_sparse_key, bge_sparse_item))

                bge_colbert_item = np.asarray(
                    remote.bgeM3.colbert[miss_row], dtype=np.float32
                )
                p.bge_colbert_items[idx] = bge_colbert_item
                bge_colbert_key = self._bge_colbert_cache_key(p.bge_model_id, text)
                to_store_bge_colbert.append((bge_colbert_key, bge_colbert_item))
        self._save_dense_vectors(to_store_dense)
        self._save_sparse_items(to_store_sparse)
        self._save_dense_vectors(to_store_bge_dense)
        self._save_sparse_items(to_store_bge_sparse)
        self._save_colbert_items(to_store_bge_colbert)

    def _embed_cache_finalize(self, p: _EmbedCacheProgress) -> ParsedEmbedResponse:
        merged_dense: DenseEmbeddings | None = None
        if p.dense_model_id is not None and p.dense_vectors is not None:
            if any(vector is None for vector in p.dense_vectors):
                raise ValueError("Dense cache merge failed: unresolved vectors remain.")
            merged_dense = DenseEmbeddings(
                model_id=p.dense_model_id,
                vectors=_normalize_dense_vectors_for_lmdb_cache(
                    np.vstack(p.dense_vectors)
                ),  # type: ignore[arg-type]
            )

        merged_sparse: SparseEmbeddings | None = None
        if p.sparse_model_id is not None and p.sparse_items is not None:
            if any(item is None for item in p.sparse_items):
                raise ValueError("Sparse cache merge failed: unresolved items remain.")
            merged_sparse = SparseEmbeddings(
                model_id=p.sparse_model_id,
                items=[item for item in p.sparse_items if item is not None],
            )

        merged_bge: BGEM3Embeddings | None = None
        if (
            p.bge_model_id is not None
            and p.bge_dense_vectors is not None
            and p.bge_sparse_items is not None
            and p.bge_colbert_items is not None
        ):
            if any(vector is None for vector in p.bge_dense_vectors):
                raise ValueError(
                    "BGE-M3 cache merge failed: unresolved dense vectors remain."
                )
            if any(item is None for item in p.bge_sparse_items):
                raise ValueError(
                    "BGE-M3 cache merge failed: unresolved sparse items remain."
                )
            if any(item is None for item in p.bge_colbert_items):
                raise ValueError(
                    "BGE-M3 cache merge failed: unresolved colbert items remain."
                )
            merged_bge = BGEM3Embeddings(
                model_id=p.bge_model_id,
                dense=DenseEmbeddings(
                    model_id=p.bge_model_id,
                    vectors=_normalize_dense_vectors_for_lmdb_cache(
                        np.vstack(p.bge_dense_vectors)
                    ),  # type: ignore[arg-type]
                ),
                sparse=SparseEmbeddings(
                    model_id=p.bge_model_id,
                    items=[item for item in p.bge_sparse_items if item is not None],
                ),
                colbert=[
                    _normalize_colbert_vector_for_lmdb_cache(item)
                    for item in p.bge_colbert_items
                    if item is not None
                ],
            )

        return ParsedEmbedResponse(
            texts_count=len(p.texts),
            dense=merged_dense,
            sparse=merged_sparse,
            bgeM3=merged_bge,
        )

    async def embed(
        self, payload: EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponse:
        should_use_cache = self._use_cache if use_cache is None else use_cache
        if not should_use_cache:
            return await self._embed_remote(payload)

        prepared = self._embed_cache_prepare(payload)
        if prepared is None:
            return await self._embed_remote(payload)

        if prepared.misses:
            miss_payload = cast(EmbedRequestDict, dict(prepared.payload))
            miss_payload["texts"] = [prepared.texts[idx] for idx in prepared.misses]
            remote = await self._embed_remote(miss_payload)
            if remote.texts_count != len(prepared.misses):
                raise ValueError(
                    "Cache merge failed: mismatch in misses and returned items."
                )
            self._embed_cache_merge_remote(prepared, remote)

        return self._embed_cache_finalize(prepared)

    def embed_sync(
        self, payload: EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponse:
        should_use_cache = self._use_cache if use_cache is None else use_cache
        if not should_use_cache:
            return self._embed_remote_sync(payload)

        prepared = self._embed_cache_prepare(payload)
        if prepared is None:
            return self._embed_remote_sync(payload)

        if prepared.misses:
            miss_payload = cast(EmbedRequestDict, dict(prepared.payload))
            miss_payload["texts"] = [prepared.texts[idx] for idx in prepared.misses]
            remote = self._embed_remote_sync(miss_payload)
            if remote.texts_count != len(prepared.misses):
                raise ValueError(
                    "Cache merge failed: mismatch in misses and returned items."
                )
            self._embed_cache_merge_remote(prepared, remote)

        return self._embed_cache_finalize(prepared)

    async def rerank(self, payload: RerankRequestDict) -> ParsedRerankResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = await self.client.post(
            "/rerank",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        response.raise_for_status()
        raw = cast(RerankResponseDict, response.json())
        shape_values = raw["shape"]
        if len(shape_values) != 2:
            raise ValueError(
                f"Rerank response shape must have 2 items, got: {shape_values!r}"
            )
        return ParsedRerankResponse(
            model_id=raw["model_id"],
            shape=(int(shape_values[0]), int(shape_values[1])),
            scores=raw["scores"],
        )

    def rerank_sync(self, payload: RerankRequestDict) -> ParsedRerankResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self.sync_client.post(
            "/rerank",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        response.raise_for_status()
        raw = cast(RerankResponseDict, response.json())
        shape_values = raw["shape"]
        if len(shape_values) != 2:
            raise ValueError(
                f"Rerank response shape must have 2 items, got: {shape_values!r}"
            )
        return ParsedRerankResponse(
            model_id=raw["model_id"],
            shape=(int(shape_values[0]), int(shape_values[1])),
            scores=raw["scores"],
        )
