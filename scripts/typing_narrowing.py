# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mypy",
# ]
# ///


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Required, TypedDict, Unpack, assert_type, overload

# ============================================================
# Mock domain models
# ============================================================


@dataclass(slots=True)
class DenseEmbeddings:
    model_id: str
    vectors: list[list[float]]


@dataclass(slots=True)
class SparseEmbedding:
    dim: int
    indices: list[int]
    values: list[float]


@dataclass(slots=True)
class SparseEmbeddings:
    model_id: str
    items: list[SparseEmbedding]


@dataclass(slots=True)
class BGEM3Embeddings:
    model_id: str
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    colbert: list[list[list[float]]]


# ============================================================
# Request types
# ============================================================


class EmbedRequestCommonDict(TypedDict, total=False):
    texts: list[str]

    dense_truncate_dim: int | None
    dense_prompt: str
    dense_task: Literal["query", "document"] | None

    sparse_max_active_dims: int | None
    sparse_pruning_ratio: float | None
    sparse_task: Literal["query", "document"] | None


class DenseEmbedRequestDict(EmbedRequestCommonDict, total=False):
    dense_model_id: Required[str]


class SparseEmbedRequestDict(EmbedRequestCommonDict, total=False):
    sparse_model_id: Required[str]


class BGEM3EmbedRequestDict(EmbedRequestCommonDict, total=False):
    bge_model_id: Required[str]


class DenseSparseEmbedRequestDict(EmbedRequestCommonDict, total=False):
    dense_model_id: Required[str]
    sparse_model_id: Required[str]


class DenseBGEM3EmbedRequestDict(EmbedRequestCommonDict, total=False):
    dense_model_id: Required[str]
    bge_model_id: Required[str]


class SparseBGEM3EmbedRequestDict(EmbedRequestCommonDict, total=False):
    sparse_model_id: Required[str]
    bge_model_id: Required[str]


class DenseSparseBGEM3EmbedRequestDict(EmbedRequestCommonDict, total=False):
    dense_model_id: Required[str]
    sparse_model_id: Required[str]
    bge_model_id: Required[str]


class _EmbedRequestImplDict(EmbedRequestCommonDict, total=False):
    dense_model_id: str
    sparse_model_id: str
    bge_model_id: str


# ============================================================
# Response types
# ============================================================


@dataclass(slots=True, frozen=True)
class ParsedEmbedResponseCommon:
    texts_count: int


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseDense(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings | None = None
    bgeM3: BGEM3Embeddings | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseSparse(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings | None = None
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings | None = None
    sparse: SparseEmbeddings | None = None
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseDenseSparse(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseSparseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings | None = None
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseDenseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings | None = None
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, frozen=True, kw_only=True)
class ParsedEmbedResponseDenseSparseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings


ParsedEmbedResponse = (
    ParsedEmbedResponseDense
    | ParsedEmbedResponseSparse
    | ParsedEmbedResponseBGEM3
    | ParsedEmbedResponseDenseSparse
    | ParsedEmbedResponseDenseBGEM3
    | ParsedEmbedResponseSparseBGEM3
    | ParsedEmbedResponseDenseSparseBGEM3
)


class FiniteEmbeddingsClient:
    def _make_dense(self, model_id: str, texts: list[str]) -> DenseEmbeddings:
        return DenseEmbeddings(
            model_id=model_id,
            vectors=[[0.1, 0.2, 0.3] for _ in texts],
        )

    def _make_sparse(self, model_id: str, texts: list[str]) -> SparseEmbeddings:
        return SparseEmbeddings(
            model_id=model_id,
            items=[
                SparseEmbedding(dim=8, indices=[1, 4, 6], values=[0.5, 0.2, 0.1])
                for _ in texts
            ],
        )

    def _make_bge(self, model_id: str, texts: list[str]) -> BGEM3Embeddings:
        return BGEM3Embeddings(
            model_id=model_id,
            dense=DenseEmbeddings(
                model_id=model_id,
                vectors=[[0.9, 0.8, 0.7] for _ in texts],
            ),
            sparse=SparseEmbeddings(
                model_id=model_id,
                items=[
                    SparseEmbedding(dim=16, indices=[2, 5], values=[0.4, 0.6])
                    for _ in texts
                ],
            ),
            colbert=[[[0.01, 0.02], [0.03, 0.04]] for _ in texts],
        )

    @overload
    def embed(
        self, **payload: Unpack[DenseEmbedRequestDict]
    ) -> ParsedEmbedResponseDense: ...

    @overload
    def embed(
        self, **payload: Unpack[SparseEmbedRequestDict]
    ) -> ParsedEmbedResponseSparse: ...

    @overload
    def embed(
        self, **payload: Unpack[BGEM3EmbedRequestDict]
    ) -> ParsedEmbedResponseBGEM3: ...

    @overload
    def embed(
        self, **payload: Unpack[DenseSparseEmbedRequestDict]
    ) -> ParsedEmbedResponseDenseSparse: ...

    @overload
    def embed(
        self, **payload: Unpack[DenseBGEM3EmbedRequestDict]
    ) -> ParsedEmbedResponseDenseBGEM3: ...

    @overload
    def embed(
        self, **payload: Unpack[SparseBGEM3EmbedRequestDict]
    ) -> ParsedEmbedResponseSparseBGEM3: ...

    @overload
    def embed(
        self, **payload: Unpack[DenseSparseBGEM3EmbedRequestDict]
    ) -> ParsedEmbedResponseDenseSparseBGEM3: ...

    def embed(
        self,
        **payload: Unpack[_EmbedRequestImplDict],
    ) -> ParsedEmbedResponse:

        if (
            "dense_model_id" not in payload
            and "sparse_model_id" not in payload
            and "bge_model_id" not in payload
        ):
            raise ValueError(
                "At least one of dense_model_id, sparse_model_id, or bge_model_id must be provided."
            )

        texts = payload.get("texts", [])
        texts_count = len(texts)

        dense_model_id = payload.get("dense_model_id")
        dense = (
            self._make_dense(dense_model_id, texts)
            if dense_model_id is not None
            else None
        )

        sparse_model_id = payload.get("sparse_model_id")
        sparse = (
            self._make_sparse(sparse_model_id, texts)
            if sparse_model_id is not None
            else None
        )

        bge_model_id = payload.get("bge_model_id")
        bge = self._make_bge(bge_model_id, texts) if bge_model_id is not None else None

        if "dense_model_id" in payload and dense is None:
            raise ValueError("Output is missing required key: dense")
        if "sparse_model_id" in payload and sparse is None:
            raise ValueError("Output is missing required key: sparse")
        if "bge_model_id" in payload and bge is None:
            raise ValueError("Output is missing required key: bgeM3")

        if dense is not None and sparse is not None and bge is not None:
            return ParsedEmbedResponseDenseSparseBGEM3(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        elif dense is not None and sparse is not None:
            return ParsedEmbedResponseDenseSparse(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        elif dense is not None and bge is not None:
            return ParsedEmbedResponseDenseBGEM3(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        elif sparse is not None and bge is not None:
            return ParsedEmbedResponseSparseBGEM3(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        elif dense is not None:
            return ParsedEmbedResponseDense(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        elif sparse is not None:
            return ParsedEmbedResponseSparse(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        elif bge is not None:
            return ParsedEmbedResponseBGEM3(
                texts_count=texts_count,
                dense=dense,
                sparse=sparse,
                bgeM3=bge,
            )
        else:
            raise ValueError("At least one of dense, sparse, or bge must be returned.")


# ============================================================
# Typing tests
# Run:
#   mypy this_file.py
# or:
#   pyright this_file.py
# ============================================================


def test_dense_only() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        dense_model_id="dense-v1",
        dense_prompt="",
    )
    assert_type(response, ParsedEmbedResponseDense)

    # Should type-check: dense is guaranteed.
    first = response.dense.vectors[0]
    print(first)

    # This should fail type checking, because sparse may be None.
    # EXPECT TYPE ERROR HERE
    print(response.sparse.items[0])
    print(response.bgeM3.colbert[0])


def test_sparse_only() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        sparse_model_id="sparse-v1",
    )
    assert_type(response, ParsedEmbedResponseSparse)

    # Should type-check: sparse is guaranteed.
    idx = response.sparse.items[0].indices
    print(idx)

    # Should fail type checking, because dense may be None.
    # EXPECT TYPE ERROR HERE
    print(response.dense.vectors[0])
    print(response.bgeM3.items[0])


def test_bge_only() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        bge_model_id="bge-m3",
    )
    assert_type(response, ParsedEmbedResponseBGEM3)

    # Should type-check: bgeM3 is guaranteed.
    colbert = response.bgeM3.colbert[0]
    print(colbert)

    # Should fail type checking, because top-level dense may be None.
    # EXPECT TYPE ERROR HERE
    print(response.dense.vectors[0])
    print(response.sparse.items[0])


def test_dense_sparse() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        dense_model_id="dense-v1",
        sparse_model_id="sparse-v1",
    )
    assert_type(response, ParsedEmbedResponseDenseSparse)

    print(response.dense.vectors[0])
    print(response.sparse.items[0])
    # EXPECT TYPE ERROR HERE
    print(response.bgeM3.colbert[0])


def test_dense_bge() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        dense_model_id="dense-v1",
        bge_model_id="bge-m3",
    )
    assert_type(response, ParsedEmbedResponseDenseBGEM3)

    print(response.dense.vectors[0])
    print(response.bgeM3.colbert[0])
    # EXPECT TYPE ERROR HERE
    print(response.sparse.items[0])


def test_sparse_bge() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        sparse_model_id="sparse-v1",
        bge_model_id="bge-m3",
    )
    assert_type(response, ParsedEmbedResponseSparseBGEM3)
    print(response.sparse.items[0])
    print(response.bgeM3.colbert[0])
    # EXPECT TYPE ERROR HERE
    print(response.dense.vectors[0])


def test_all() -> None:
    client = FiniteEmbeddingsClient()

    response = client.embed(
        texts=["hello"],
        dense_model_id="dense-v1",
        sparse_model_id="sparse-v1",
        bge_model_id="bge-m3",
    )
    assert_type(response, ParsedEmbedResponseDenseSparseBGEM3)
    print(response.dense.vectors[0])
    print(response.sparse.items[0])


if __name__ == "__main__":
    test_dense_only()
    test_sparse_only()
    test_bge_only()
    test_dense_sparse()
    test_dense_bge()
    test_sparse_bge()
    test_all()
