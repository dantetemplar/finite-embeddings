from __future__ import annotations

import base64
import gzip
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, Required, TypedDict, TypeGuard, cast, overload

import httpx
import lmdb
import numpy as np
import numpy.typing as npt

type Float32Array = npt.NDArray[np.float32]
type Float16Array = npt.NDArray[np.float16]
type UInt32Array = npt.NDArray[np.uint32]


class EmbedRequestCommonDict(TypedDict):
    texts: list[str]
    dense_truncate_dim: NotRequired[int | None]
    dense_prompt: NotRequired[str]
    dense_task: NotRequired[Literal["query", "document"] | None]
    sparse_max_active_dims: NotRequired[int | None]
    sparse_pruning_ratio: NotRequired[float | None]
    sparse_task: NotRequired[Literal["query", "document"] | None]


class DenseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: NotRequired[None]
    bge_model_id: NotRequired[None]


class SparseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: NotRequired[None]
    sparse_model_id: Required[str]
    bge_model_id: NotRequired[None]


class BGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: NotRequired[None]
    sparse_model_id: NotRequired[None]
    bge_model_id: Required[str]


class DenseSparseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: Required[str]
    bge_model_id: NotRequired[None]


class DenseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: NotRequired[None]
    bge_model_id: Required[str]


class SparseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: NotRequired[None]
    sparse_model_id: Required[str]
    bge_model_id: Required[str]


class DenseSparseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: Required[str]
    bge_model_id: Required[str]


class _EmbedRequestImplDict(EmbedRequestCommonDict):
    dense_model_id: NotRequired[str | None]
    sparse_model_id: NotRequired[str | None]
    bge_model_id: NotRequired[str | None]


EmbedRequestPayload = (
    DenseEmbedRequestDict
    | SparseEmbedRequestDict
    | BGEM3EmbedRequestDict
    | DenseSparseEmbedRequestDict
    | DenseBGEM3EmbedRequestDict
    | SparseBGEM3EmbedRequestDict
    | DenseSparseBGEM3EmbedRequestDict
)

EmbedRequestAny = EmbedRequestPayload | _EmbedRequestImplDict


class EmbedOneRequestCommonDict(TypedDict):
    text: str
    dense_truncate_dim: NotRequired[int | None]
    dense_prompt: NotRequired[str]
    dense_task: NotRequired[Literal["query", "document"] | None]
    sparse_max_active_dims: NotRequired[int | None]
    sparse_pruning_ratio: NotRequired[float | None]
    sparse_task: NotRequired[Literal["query", "document"] | None]


class DenseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: NotRequired[None]
    bge_model_id: NotRequired[None]


class SparseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: NotRequired[None]
    sparse_model_id: Required[str]
    bge_model_id: NotRequired[None]


class BGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: NotRequired[None]
    sparse_model_id: NotRequired[None]
    bge_model_id: Required[str]


class DenseSparseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: Required[str]
    bge_model_id: NotRequired[None]


class DenseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: NotRequired[None]
    bge_model_id: Required[str]


class SparseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: NotRequired[None]
    sparse_model_id: Required[str]
    bge_model_id: Required[str]


class DenseSparseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: Required[str]
    sparse_model_id: Required[str]
    bge_model_id: Required[str]


class _EmbedOneRequestImplDict(EmbedOneRequestCommonDict):
    dense_model_id: NotRequired[str | None]
    sparse_model_id: NotRequired[str | None]
    bge_model_id: NotRequired[str | None]


EmbedOneRequestPayload = (
    DenseEmbedOneRequestDict
    | SparseEmbedOneRequestDict
    | BGEM3EmbedOneRequestDict
    | DenseSparseEmbedOneRequestDict
    | DenseBGEM3EmbedOneRequestDict
    | SparseBGEM3EmbedOneRequestDict
    | DenseSparseBGEM3EmbedOneRequestDict
)

EmbedOneRequestAny = EmbedOneRequestPayload | _EmbedOneRequestImplDict


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
    vectors: Float32Array


@dataclass(slots=True)
class SparseEmbedding:
    dim: int
    indices: UInt32Array
    values: Float32Array


@dataclass(slots=True)
class SparseEmbeddings:
    model_id: str
    items: list[SparseEmbedding]


@dataclass(slots=True)
class BGEM3Embeddings:
    model_id: str
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    colbert: list[Float32Array]


@dataclass(slots=True)
class ParsedEmbedResponseCommon:
    texts_count: int


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDense(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseSparse(ParsedEmbedResponseCommon):
    sparse: SparseEmbeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseBGEM3(ParsedEmbedResponseCommon):
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDenseSparse(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDenseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseSparseBGEM3(ParsedEmbedResponseCommon):
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDenseSparseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings


ParsedEmbedResponseVariant = (
    ParsedEmbedResponseDense
    | ParsedEmbedResponseSparse
    | ParsedEmbedResponseBGEM3
    | ParsedEmbedResponseDenseSparse
    | ParsedEmbedResponseDenseBGEM3
    | ParsedEmbedResponseSparseBGEM3
    | ParsedEmbedResponseDenseSparseBGEM3
)


@dataclass(slots=True)
class DenseEmbeddingVector:
    model_id: str
    vector: Float32Array


@dataclass(slots=True)
class SparseEmbeddingsOne:
    model_id: str
    item: SparseEmbedding


@dataclass(slots=True)
class BGEM3EmbeddingOne:
    model_id: str
    dense: DenseEmbeddingVector
    sparse: SparseEmbeddingsOne
    colbert: Float32Array


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDense:
    dense: DenseEmbeddingVector


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneSparse:
    sparse: SparseEmbeddingsOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneBGEM3:
    bgeM3: BGEM3EmbeddingOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDenseSparse:
    dense: DenseEmbeddingVector
    sparse: SparseEmbeddingsOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDenseBGEM3:
    dense: DenseEmbeddingVector
    bgeM3: BGEM3EmbeddingOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneSparseBGEM3:
    sparse: SparseEmbeddingsOne
    bgeM3: BGEM3EmbeddingOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDenseSparseBGEM3:
    dense: DenseEmbeddingVector
    sparse: SparseEmbeddingsOne
    bgeM3: BGEM3EmbeddingOne


ParsedEmbedOneVariant = (
    ParsedEmbedOneDense
    | ParsedEmbedOneSparse
    | ParsedEmbedOneBGEM3
    | ParsedEmbedOneDenseSparse
    | ParsedEmbedOneDenseBGEM3
    | ParsedEmbedOneSparseBGEM3
    | ParsedEmbedOneDenseSparseBGEM3
)


@dataclass(slots=True)
class ParsedRerankResponse:
    model_id: str
    shape: tuple[int, int]
    scores: list[list[float]]


@dataclass(slots=True)
class _EmbedCacheProgress:
    payload: EmbedRequestAny
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
    dense_vectors: list[Float32Array | None] | None
    sparse_items: list[SparseEmbedding | None] | None
    bge_dense_vectors: list[Float32Array | None] | None
    bge_sparse_items: list[SparseEmbedding | None] | None
    bge_colbert_items: list[Float32Array | None] | None


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


def _decode_colbert_item(item: ColbertItemResponseDict) -> Float32Array:
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


def _roundtrip_float32_float16_float32(vectors: np.ndarray) -> Float32Array:
    """Align with float16 LMDB storage so network and cache hits return the same vector values."""
    return np.asarray(vectors, dtype=np.float32).astype(np.float16).astype(np.float32)


def all_present[T](items: list[T | None] | None) -> TypeGuard[list[T]]:
    if items is None:
        return False
    for item in items:
        if item is None:
            return False
    return True


class FiniteEmbeddingsClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        aclient: httpx.AsyncClient | None = None,
        use_cache: bool = False,
        cache: lmdb.Environment | None = None,
        cache_path: str | Path | None = None,
        cache_map_size: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        # Caller owns lifecycle and configuration (base_url, timeout, headers).
        self._client = client
        self._aclient = aclient
        if self._client is None and self._aclient is None:
            raise ValueError("Either client or aclient must be provided, may be both.")
        self._cache = cache
        self._use_cache = use_cache or self._cache is not None
        if self._use_cache and self._cache is None:
            resolved_cache_dir = (
                Path(cache_path).expanduser()
                if cache_path is not None
                else Path.home() / ".cache" / "finite-embeddings" / "client-cache.lmdb"
            )
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = lmdb.open(str(resolved_cache_dir), map_size=cache_map_size)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError(
                "client is not configured; pass client=httpx.Client(...) to FiniteEmbeddingsClient(...)."
            )
        return self._client

    @property
    def aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            raise RuntimeError(
                "aclient is not configured; pass aclient=httpx.AsyncClient(...) to FiniteEmbeddingsClient(...)."
            )
        return self._aclient

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
        h.update(b"dense_prompt\x00")
        h.update(dense_prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(b"dense_task\x00")
        task_value = "" if dense_task is None else dense_task
        h.update(task_value.encode("utf-8"))
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
        h.update(b"mad\x00")
        if max_active_dims is not None:
            h.update(max_active_dims.to_bytes(4, "little", signed=False))
        h.update(b"\x00")
        h.update(b"pr\x00")
        if pruning_ratio is not None:
            h.update(str(pruning_ratio).encode("utf-8"))
        h.update(b"\x00")
        h.update(b"sparse_task\x00")
        task_value = "" if sparse_task is None else sparse_task
        h.update(task_value.encode("utf-8"))
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

    @staticmethod
    def _decode_dense_cache_value(
        raw: bytes | None, expected_dim: int | None
    ) -> np.ndarray | None:
        if raw is None:
            return None
        if expected_dim is not None:
            vector = np.frombuffer(raw, dtype=np.float16, count=expected_dim).astype(
                np.float32
            )
            if vector.size != expected_dim:
                return None
            return vector
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)

    @staticmethod
    def _decode_sparse_cache_value(raw: bytes | None) -> SparseEmbedding | None:
        if raw is None or len(raw) < 8:
            return None
        dim, nnz = struct.unpack("<II", raw[:8])
        idx_end = 8 + (nnz * 4)
        val_end = idx_end + (nnz * 4)
        if len(raw) != val_end:
            return None
        indices = cast(
            UInt32Array, np.frombuffer(raw[8:idx_end], dtype=np.uint32).copy()
        )
        values = cast(
            Float32Array, np.frombuffer(raw[idx_end:val_end], dtype=np.float32).copy()
        )
        return SparseEmbedding(dim=int(dim), indices=indices, values=values)

    @staticmethod
    def _decode_colbert_cache_value(raw: bytes | None) -> np.ndarray | None:
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

    def _load_dense_vectors(
        self, keys: list[bytes], expected_dim: int | None
    ) -> list[np.ndarray | None]:
        results: list[np.ndarray | None] = [None for _ in keys]
        if self._cache is None or not keys:
            return results
        try:
            with self._cache.begin(write=False) as txn:
                for idx, key in enumerate(keys):
                    raw = txn.get(key)
                    results[idx] = self._decode_dense_cache_value(
                        bytes(raw) if raw is not None else None, expected_dim
                    )
        except Exception:
            return [None for _ in keys]
        return results

    def _save_dense_vectors(self, items: list[tuple[bytes, np.ndarray]]) -> None:
        if self._cache is None or not items:
            return
        try:
            with self._cache.begin(write=True) as txn:
                for key, vector in items:
                    txn.put(key, np.asarray(vector, dtype=np.float16).tobytes())
        except Exception:
            return

    def _load_sparse_items(self, keys: list[bytes]) -> list[SparseEmbedding | None]:
        results: list[SparseEmbedding | None] = [None for _ in keys]
        if self._cache is None or not keys:
            return results
        try:
            with self._cache.begin(write=False) as txn:
                for idx, key in enumerate(keys):
                    raw = txn.get(key)
                    results[idx] = self._decode_sparse_cache_value(
                        bytes(raw) if raw is not None else None
                    )
        except Exception:
            return [None for _ in keys]
        return results

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

    def _load_colbert_items(self, keys: list[bytes]) -> list[np.ndarray | None]:
        results: list[np.ndarray | None] = [None for _ in keys]
        if self._cache is None or not keys:
            return results
        try:
            with self._cache.begin(write=False) as txn:
                for idx, key in enumerate(keys):
                    raw = txn.get(key)
                    results[idx] = self._decode_colbert_cache_value(
                        bytes(raw) if raw is not None else None
                    )
        except Exception:
            return [None for _ in keys]
        return results

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

    async def amodels(self) -> ModelsResponseDict:
        response = await self.aclient.get(
            "/models", headers={"Accept-Encoding": "gzip"}
        )
        response.raise_for_status()
        return response.json()

    def models(self) -> ModelsResponseDict:
        response = self.client.get("/models", headers={"Accept-Encoding": "gzip"})
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _build_parsed_embed_response(
        *,
        texts_count: int,
        dense: DenseEmbeddings | None = None,
        sparse: SparseEmbeddings | None = None,
        bge_m3: BGEM3Embeddings | None = None,
        payload: EmbedRequestAny,
        raw: EmbedResponseDict | None = None,
    ) -> ParsedEmbedResponseVariant:
        if raw is not None:
            texts_len = len(payload.get("texts", []))

            if payload.get("dense_model_id") is not None:
                if raw["dense"] is not None:
                    dense = _decode_dense(raw["dense"])
                    if len(dense.vectors) != texts_len:
                        raise ValueError(
                            "Dense remote embed failed: mismatch in texts count."
                        )
                else:
                    raise ValueError(
                        "Dense remote embed failed: unresolved vectors remain."
                    )

            if payload.get("sparse_model_id") is not None:
                if raw["sparse"] is not None:
                    sparse = _decode_sparse(raw["sparse"])
                    if len(sparse.items) != texts_len:
                        raise ValueError(
                            "Sparse remote embed failed: mismatch in texts count."
                        )
                else:
                    raise ValueError(
                        "Sparse remote embed failed: unresolved items remain."
                    )

            if payload.get("bge_model_id") is not None:
                if raw["bgeM3"] is not None:
                    bge_m3 = _decode_bge_m3(raw["bgeM3"])
                    if len(bge_m3.dense.vectors) != texts_len:
                        raise ValueError(
                            "BGE-M3 remote embed failed: mismatch in texts count."
                        )
                    if len(bge_m3.sparse.items) != texts_len:
                        raise ValueError(
                            "BGE-M3 remote embed failed: mismatch in texts count."
                        )
                    if len(bge_m3.colbert) != texts_len:
                        raise ValueError(
                            "BGE-M3 remote embed failed: mismatch in texts count."
                        )
                else:
                    raise ValueError(
                        "BGE-M3 remote embed failed: unresolved vectors remain."
                    )

        if dense is not None and sparse is not None and bge_m3 is not None:
            return ParsedEmbedResponseDenseSparseBGEM3(
                texts_count=texts_count, dense=dense, sparse=sparse, bgeM3=bge_m3
            )
        if dense is not None and sparse is not None:
            return ParsedEmbedResponseDenseSparse(
                texts_count=texts_count, dense=dense, sparse=sparse
            )
        if dense is not None and bge_m3 is not None:
            return ParsedEmbedResponseDenseBGEM3(
                texts_count=texts_count, dense=dense, bgeM3=bge_m3
            )
        if sparse is not None and bge_m3 is not None:
            return ParsedEmbedResponseSparseBGEM3(
                texts_count=texts_count, sparse=sparse, bgeM3=bge_m3
            )
        if dense is not None:
            return ParsedEmbedResponseDense(texts_count=texts_count, dense=dense)
        if sparse is not None:
            return ParsedEmbedResponseSparse(texts_count=texts_count, sparse=sparse)
        if bge_m3 is not None:
            return ParsedEmbedResponseBGEM3(texts_count=texts_count, bgeM3=bge_m3)
        raise ValueError("At least one of dense, sparse, or bgeM3 must be returned.")

    @staticmethod
    def _embed_one_payload_to_batch(payload: EmbedOneRequestAny) -> EmbedRequestAny:
        if "text" not in payload:
            raise ValueError("text must be provided.")
        text = payload["text"]
        batch = dict(cast(dict[str, object], payload))
        del batch["text"]
        batch["texts"] = [text]
        return cast(EmbedRequestAny, batch)

    @staticmethod
    def _dense_emb_to_vector(dense: DenseEmbeddings) -> DenseEmbeddingVector:
        if dense.vectors.shape[0] != 1:
            raise ValueError("embed_one requires batch size 1 for dense embeddings.")
        return DenseEmbeddingVector(
            model_id=dense.model_id,
            vector=np.asarray(dense.vectors[0], dtype=np.float32),
        )

    @staticmethod
    def _sparse_emb_to_one(sparse: SparseEmbeddings) -> SparseEmbeddingsOne:
        if len(sparse.items) != 1:
            raise ValueError("embed_one requires batch size 1 for sparse embeddings.")
        return SparseEmbeddingsOne(model_id=sparse.model_id, item=sparse.items[0])

    @staticmethod
    def _bge_m3_emb_to_one(bge: BGEM3Embeddings) -> BGEM3EmbeddingOne:
        if (
            bge.dense.vectors.shape[0] != 1
            or len(bge.sparse.items) != 1
            or len(bge.colbert) != 1
        ):
            raise ValueError("embed_one requires batch size 1 for BGE-M3 embeddings.")
        return BGEM3EmbeddingOne(
            model_id=bge.model_id,
            dense=DenseEmbeddingVector(
                model_id=bge.dense.model_id,
                vector=np.asarray(bge.dense.vectors[0], dtype=np.float32),
            ),
            sparse=SparseEmbeddingsOne(
                model_id=bge.sparse.model_id, item=bge.sparse.items[0]
            ),
            colbert=np.asarray(bge.colbert[0], dtype=np.float32),
        )

    @staticmethod
    def _parsed_embed_batch_to_one(
        response: ParsedEmbedResponseVariant,
    ) -> ParsedEmbedOneVariant:
        if response.texts_count != 1:
            raise ValueError("embed_one requires texts_count == 1.")
        if isinstance(response, ParsedEmbedResponseDenseSparseBGEM3):
            return ParsedEmbedOneDenseSparseBGEM3(
                dense=FiniteEmbeddingsClient._dense_emb_to_vector(response.dense),
                sparse=FiniteEmbeddingsClient._sparse_emb_to_one(response.sparse),
                bgeM3=FiniteEmbeddingsClient._bge_m3_emb_to_one(response.bgeM3),
            )
        if isinstance(response, ParsedEmbedResponseDenseSparse):
            return ParsedEmbedOneDenseSparse(
                dense=FiniteEmbeddingsClient._dense_emb_to_vector(response.dense),
                sparse=FiniteEmbeddingsClient._sparse_emb_to_one(response.sparse),
            )
        if isinstance(response, ParsedEmbedResponseDenseBGEM3):
            return ParsedEmbedOneDenseBGEM3(
                dense=FiniteEmbeddingsClient._dense_emb_to_vector(response.dense),
                bgeM3=FiniteEmbeddingsClient._bge_m3_emb_to_one(response.bgeM3),
            )
        if isinstance(response, ParsedEmbedResponseSparseBGEM3):
            return ParsedEmbedOneSparseBGEM3(
                sparse=FiniteEmbeddingsClient._sparse_emb_to_one(response.sparse),
                bgeM3=FiniteEmbeddingsClient._bge_m3_emb_to_one(response.bgeM3),
            )
        if isinstance(response, ParsedEmbedResponseDense):
            return ParsedEmbedOneDense(
                dense=FiniteEmbeddingsClient._dense_emb_to_vector(response.dense)
            )
        if isinstance(response, ParsedEmbedResponseSparse):
            return ParsedEmbedOneSparse(
                sparse=FiniteEmbeddingsClient._sparse_emb_to_one(response.sparse)
            )
        if isinstance(response, ParsedEmbedResponseBGEM3):
            return ParsedEmbedOneBGEM3(
                bgeM3=FiniteEmbeddingsClient._bge_m3_emb_to_one(response.bgeM3)
            )
        raise AssertionError("Unreachable embed response variant.")

    async def _embed_remote(
        self, payload: EmbedRequestAny
    ) -> ParsedEmbedResponseVariant:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = await self.aclient.post(
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
        return self._build_parsed_embed_response(
            texts_count=raw["texts_count"], payload=payload, raw=raw
        )

    def _embed_remote_sync(
        self, payload: EmbedRequestAny
    ) -> ParsedEmbedResponseVariant:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self.client.post(
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
        return self._build_parsed_embed_response(
            texts_count=raw["texts_count"], payload=payload, raw=raw
        )

    def _embed_cache_prepare(self, payload: EmbedRequestAny) -> _EmbedCacheProgress:
        texts = payload.get("texts", [])
        dense_model_id = payload.get("dense_model_id")
        dense_truncate_dim = payload.get("dense_truncate_dim")
        dense_prompt = payload.get("dense_prompt") or ""
        dense_task = payload.get("dense_task")

        sparse_model_id = payload.get("sparse_model_id")
        sparse_max_active_dims = payload.get("sparse_max_active_dims")
        sparse_pruning_ratio = payload.get("sparse_pruning_ratio")
        sparse_task = payload.get("sparse_task")

        bge_model_id = payload.get("bge_model_id")

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

        dense_keys: list[bytes] | None = None
        sparse_keys: list[bytes] | None = None
        bge_dense_keys: list[bytes] | None = None
        bge_sparse_keys: list[bytes] | None = None
        bge_colbert_keys: list[bytes] | None = None

        if dense_model_id is not None and dense_vectors is not None:
            dense_keys = [
                self._dense_cache_key(
                    dense_model_id, dense_truncate_dim, dense_prompt, dense_task, text
                )
                for text in texts
            ]
            dense_vectors[:] = self._load_dense_vectors(dense_keys, dense_truncate_dim)

        if sparse_model_id is not None and sparse_items is not None:
            sparse_keys = [
                self._sparse_cache_key(
                    sparse_model_id,
                    sparse_max_active_dims,
                    sparse_pruning_ratio,
                    sparse_task,
                    text,
                )
                for text in texts
            ]
            sparse_items[:] = self._load_sparse_items(sparse_keys)

        if bge_model_id is not None and bge_dense_vectors is not None:
            bge_dense_keys = [
                self._bge_dense_cache_key(bge_model_id, text) for text in texts
            ]
            bge_dense_vectors[:] = self._load_dense_vectors(
                bge_dense_keys, expected_dim=None
            )

        if bge_model_id is not None and bge_sparse_items is not None:
            bge_sparse_keys = [
                self._bge_sparse_cache_key(bge_model_id, text) for text in texts
            ]
            bge_sparse_items[:] = self._load_sparse_items(bge_sparse_keys)

        if bge_model_id is not None and bge_colbert_items is not None:
            bge_colbert_keys = [
                self._bge_colbert_cache_key(bge_model_id, text) for text in texts
            ]
            bge_colbert_items[:] = self._load_colbert_items(bge_colbert_keys)

        misses: list[int] = []
        for idx in range(len(texts)):
            dense_hit = dense_vectors is None or dense_vectors[idx] is not None
            sparse_hit = sparse_items is None or sparse_items[idx] is not None
            bge_dense_hit = (
                bge_dense_vectors is None or bge_dense_vectors[idx] is not None
            )
            bge_sparse_hit = (
                bge_sparse_items is None or bge_sparse_items[idx] is not None
            )
            bge_colbert_hit = (
                bge_colbert_items is None or bge_colbert_items[idx] is not None
            )
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
        self, p: _EmbedCacheProgress, remote: ParsedEmbedResponseVariant
    ) -> None:
        to_store_dense: list[tuple[bytes, np.ndarray]] = []
        to_store_sparse: list[tuple[bytes, SparseEmbedding]] = []
        to_store_bge_dense: list[tuple[bytes, np.ndarray]] = []
        to_store_bge_sparse: list[tuple[bytes, SparseEmbedding]] = []
        to_store_bge_colbert: list[tuple[bytes, np.ndarray]] = []
        for miss_row, idx in enumerate(p.misses):
            text = p.texts[idx]
            if p.dense_model_id is not None and p.dense_vectors is not None:
                if not isinstance(
                    remote,
                    (
                        ParsedEmbedResponseDense,
                        ParsedEmbedResponseDenseSparse,
                        ParsedEmbedResponseDenseBGEM3,
                        ParsedEmbedResponseDenseSparseBGEM3,
                    ),
                ):
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
                if not isinstance(
                    remote,
                    (
                        ParsedEmbedResponseSparse,
                        ParsedEmbedResponseDenseSparse,
                        ParsedEmbedResponseSparseBGEM3,
                        ParsedEmbedResponseDenseSparseBGEM3,
                    ),
                ):
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
                if not isinstance(
                    remote,
                    (
                        ParsedEmbedResponseBGEM3,
                        ParsedEmbedResponseDenseBGEM3,
                        ParsedEmbedResponseSparseBGEM3,
                        ParsedEmbedResponseDenseSparseBGEM3,
                    ),
                ):
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

    def _embed_cache_finalize(
        self, p: _EmbedCacheProgress
    ) -> ParsedEmbedResponseVariant:
        merged_dense: DenseEmbeddings | None = None
        if p.dense_model_id is not None:
            if all_present(p.dense_vectors):
                dense_vectors = _roundtrip_float32_float16_float32(
                    np.vstack(p.dense_vectors)
                )
            else:
                raise ValueError("Dense cache merge failed: unresolved vectors remain.")
            merged_dense = DenseEmbeddings(
                model_id=p.dense_model_id, vectors=dense_vectors
            )

        merged_sparse: SparseEmbeddings | None = None
        if p.sparse_model_id is not None:
            if all_present(p.sparse_items):
                sparse_items = p.sparse_items
            else:
                raise ValueError("Sparse cache merge failed: unresolved items remain.")
            merged_sparse = SparseEmbeddings(
                model_id=p.sparse_model_id, items=sparse_items
            )

        merged_bge: BGEM3Embeddings | None = None
        if p.bge_model_id is not None:
            if (
                all_present(p.bge_dense_vectors)
                and all_present(p.bge_sparse_items)
                and all_present(p.bge_colbert_items)
            ):
                bge_dense_vectors = _roundtrip_float32_float16_float32(
                    np.vstack(p.bge_dense_vectors)
                )
                bge_sparse_items = p.bge_sparse_items
                bge_colbert_items = [
                    _roundtrip_float32_float16_float32(item)
                    for item in p.bge_colbert_items
                ]
            else:
                raise ValueError(
                    "BGE-M3 cache merge failed: unresolved vectors remain."
                )
            merged_bge = BGEM3Embeddings(
                model_id=p.bge_model_id,
                dense=DenseEmbeddings(
                    model_id=p.bge_model_id, vectors=bge_dense_vectors
                ),
                sparse=SparseEmbeddings(
                    model_id=p.bge_model_id, items=bge_sparse_items
                ),
                colbert=bge_colbert_items,
            )

        return self._build_parsed_embed_response(
            texts_count=len(p.texts),
            dense=merged_dense,
            sparse=merged_sparse,
            bge_m3=merged_bge,
            payload=p.payload,
        )

    @overload
    async def aembed(
        self,
        payload: DenseSparseBGEM3EmbedRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedResponseDenseSparseBGEM3: ...

    @overload
    async def aembed(
        self, payload: DenseSparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseDenseSparse: ...

    @overload
    async def aembed(
        self, payload: DenseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseDenseBGEM3: ...

    @overload
    async def aembed(
        self, payload: SparseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseSparseBGEM3: ...

    @overload
    async def aembed(
        self, payload: DenseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseDense: ...

    @overload
    async def aembed(
        self, payload: SparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseSparse: ...

    @overload
    async def aembed(
        self, payload: BGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseBGEM3: ...

    async def aembed(
        self, payload: EmbedRequestPayload, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseVariant:
        if not payload.get("texts"):
            raise ValueError("At least one text must be provided.")

        if (
            not payload.get("dense_model_id")
            and not payload.get("sparse_model_id")
            and not payload.get("bge_model_id")
        ):
            raise ValueError("At least one model must be provided.")

        should_use_cache = self._use_cache if use_cache is None else use_cache
        if not should_use_cache:
            return await self._embed_remote(payload)

        prepared = self._embed_cache_prepare(payload)

        if prepared.misses:
            miss_payload = prepared.payload
            miss_payload["texts"] = [prepared.texts[idx] for idx in prepared.misses]
            remote = await self._embed_remote(miss_payload)
            if remote.texts_count != len(prepared.misses):
                raise ValueError(
                    "Cache merge failed: mismatch in misses and returned items."
                )
            self._embed_cache_merge_remote(prepared, remote)

        return self._embed_cache_finalize(prepared)

    @overload
    async def aembed_one(
        self,
        payload: DenseSparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDenseSparseBGEM3: ...

    @overload
    async def aembed_one(
        self,
        payload: DenseSparseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDenseSparse: ...

    @overload
    async def aembed_one(
        self,
        payload: DenseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDenseBGEM3: ...

    @overload
    async def aembed_one(
        self,
        payload: SparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneSparseBGEM3: ...

    @overload
    async def aembed_one(
        self,
        payload: DenseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDense: ...

    @overload
    async def aembed_one(
        self,
        payload: SparseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneSparse: ...

    @overload
    async def aembed_one(
        self,
        payload: BGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneBGEM3: ...

    async def aembed_one(
        self, payload: EmbedOneRequestPayload, *, use_cache: bool | None = None
    ) -> ParsedEmbedOneVariant:
        batch = self._embed_one_payload_to_batch(payload)
        parsed = await self.aembed(
            cast(EmbedRequestPayload, batch), use_cache=use_cache
        )
        return self._parsed_embed_batch_to_one(parsed)

    @overload
    def embed(
        self,
        payload: DenseSparseBGEM3EmbedRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedResponseDenseSparseBGEM3: ...

    @overload
    def embed(
        self, payload: DenseSparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseDenseSparse: ...

    @overload
    def embed(
        self, payload: DenseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseDenseBGEM3: ...

    @overload
    def embed(
        self, payload: SparseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseSparseBGEM3: ...

    @overload
    def embed(
        self, payload: DenseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseDense: ...

    @overload
    def embed(
        self, payload: SparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseSparse: ...

    @overload
    def embed(
        self, payload: BGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseBGEM3: ...

    def embed(
        self, payload: EmbedRequestPayload, *, use_cache: bool | None = None
    ) -> ParsedEmbedResponseVariant:
        if not payload.get("texts"):
            raise ValueError("At least one text must be provided.")
        if (
            not payload.get("dense_model_id")
            and not payload.get("sparse_model_id")
            and not payload.get("bge_model_id")
        ):
            raise ValueError("At least one model must be provided.")

        should_use_cache = self._use_cache if use_cache is None else use_cache
        if not should_use_cache:
            return self._embed_remote_sync(payload)

        prepared = self._embed_cache_prepare(payload)

        if prepared.misses:
            miss_payload = cast(EmbedRequestAny, dict(prepared.payload))
            miss_payload["texts"] = [prepared.texts[idx] for idx in prepared.misses]
            remote = self._embed_remote_sync(miss_payload)
            if remote.texts_count != len(prepared.misses):
                raise ValueError(
                    "Cache merge failed: mismatch in misses and returned items."
                )
            self._embed_cache_merge_remote(prepared, remote)

        return self._embed_cache_finalize(prepared)

    @overload
    def embed_one(
        self,
        payload: DenseSparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDenseSparseBGEM3: ...

    @overload
    def embed_one(
        self,
        payload: DenseSparseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDenseSparse: ...

    @overload
    def embed_one(
        self,
        payload: DenseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDenseBGEM3: ...

    @overload
    def embed_one(
        self,
        payload: SparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneSparseBGEM3: ...

    @overload
    def embed_one(
        self,
        payload: DenseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneDense: ...

    @overload
    def embed_one(
        self,
        payload: SparseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneSparse: ...

    @overload
    def embed_one(
        self,
        payload: BGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> ParsedEmbedOneBGEM3: ...

    def embed_one(
        self, payload: EmbedOneRequestPayload, *, use_cache: bool | None = None
    ) -> ParsedEmbedOneVariant:
        batch = self._embed_one_payload_to_batch(payload)
        parsed = self.embed(cast(EmbedRequestPayload, batch), use_cache=use_cache)
        return self._parsed_embed_batch_to_one(parsed)

    async def arerank(self, payload: RerankRequestDict) -> ParsedRerankResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = await self.aclient.post(
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

    def rerank(self, payload: RerankRequestDict) -> ParsedRerankResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self.client.post(
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
