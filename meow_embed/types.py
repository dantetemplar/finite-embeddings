from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, Required, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from typing_extensions import ReadOnly

type Float32Array = npt.NDArray[np.float32]
type Float16Array = npt.NDArray[np.float16]
type UInt32Array = npt.NDArray[np.uint32]


class EmbedRequestCommonDict(TypedDict):
    texts: ReadOnly[Sequence[str]]
    dense_truncate_dim: ReadOnly[NotRequired[int | None]]
    dense_prompt: ReadOnly[NotRequired[str]]
    dense_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]
    sparse_max_active_dims: ReadOnly[NotRequired[int | None]]
    sparse_pruning_ratio: ReadOnly[NotRequired[float | None]]
    sparse_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]


class DenseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[NotRequired[None]]


class SparseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class BGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class DenseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class SparseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


EmbedRequestPayload = (
    DenseEmbedRequestDict
    | SparseEmbedRequestDict
    | BGEM3EmbedRequestDict
    | DenseSparseEmbedRequestDict
    | DenseBGEM3EmbedRequestDict
    | SparseBGEM3EmbedRequestDict
    | DenseSparseBGEM3EmbedRequestDict
)


class EmbedOneRequestCommonDict(TypedDict):
    text: ReadOnly[Required[str]]

    dense_truncate_dim: ReadOnly[NotRequired[int | None]]
    dense_prompt: ReadOnly[NotRequired[str]]
    dense_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]

    sparse_max_active_dims: ReadOnly[NotRequired[int | None]]
    sparse_pruning_ratio: ReadOnly[NotRequired[float | None]]
    sparse_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]


class DenseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[NotRequired[None]]


class SparseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class BGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class DenseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class SparseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


EmbedOneRequestPayload = (
    DenseEmbedOneRequestDict
    | SparseEmbedOneRequestDict
    | BGEM3EmbedOneRequestDict
    | DenseSparseEmbedOneRequestDict
    | DenseBGEM3EmbedOneRequestDict
    | SparseBGEM3EmbedOneRequestDict
    | DenseSparseBGEM3EmbedOneRequestDict
)


class RerankRequestDict(TypedDict):
    reranker_model_id: ReadOnly[str]
    docs: ReadOnly[Sequence[str]]
    query: ReadOnly[NotRequired[str | None]]
    queries: ReadOnly[NotRequired[Sequence[str] | None]]


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
