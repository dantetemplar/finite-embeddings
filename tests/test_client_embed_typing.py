"""Embed / aembed overload narrowing (``assert_type``); mypy/pyright clean."""

from __future__ import annotations

from typing import Any, assert_type, cast

import httpx
import numpy as np
import pytest

from meow_embed.client import (
    BGEM3Embeddings,
    BGEM3EmbedOneRequestDict,
    BGEM3EmbedRequestDict,
    DenseBGEM3EmbedOneRequestDict,
    DenseBGEM3EmbedRequestDict,
    DenseEmbedOneRequestDict,
    DenseEmbeddings,
    DenseEmbedRequestDict,
    DenseSparseBGEM3EmbedOneRequestDict,
    DenseSparseBGEM3EmbedRequestDict,
    DenseSparseEmbedOneRequestDict,
    DenseSparseEmbedRequestDict,
    EmbedRequestAny,
    MeowEmbedClient,
    ParsedEmbedOneBGEM3,
    ParsedEmbedOneDense,
    ParsedEmbedOneDenseBGEM3,
    ParsedEmbedOneDenseSparse,
    ParsedEmbedOneDenseSparseBGEM3,
    ParsedEmbedOneSparse,
    ParsedEmbedOneSparseBGEM3,
    ParsedEmbedResponseBGEM3,
    ParsedEmbedResponseDense,
    ParsedEmbedResponseDenseBGEM3,
    ParsedEmbedResponseDenseSparse,
    ParsedEmbedResponseDenseSparseBGEM3,
    ParsedEmbedResponseSparse,
    ParsedEmbedResponseSparseBGEM3,
    ParsedEmbedResponseVariant,
    SparseBGEM3EmbedOneRequestDict,
    SparseBGEM3EmbedRequestDict,
    SparseEmbedOneRequestDict,
    SparseEmbedding,
    SparseEmbeddings,
    SparseEmbedRequestDict,
)


def _fake_parsed_embed(payload: EmbedRequestAny) -> ParsedEmbedResponseVariant:
    texts = payload["texts"]
    n = len(texts)
    dense_id = payload.get("dense_model_id")
    sparse_id = payload.get("sparse_model_id")
    bge_id = payload.get("bge_model_id")

    dense: DenseEmbeddings | None = None
    if dense_id:
        dense = DenseEmbeddings(
            model_id=dense_id,
            vectors=np.zeros((n, 2), dtype=np.float32),
        )

    sparse: SparseEmbeddings | None = None
    if sparse_id:
        sparse = SparseEmbeddings(
            model_id=sparse_id,
            items=[
                SparseEmbedding(
                    dim=4,
                    indices=np.array([0], dtype=np.uint32),
                    values=np.array([1.0], dtype=np.float32),
                )
                for _ in texts
            ],
        )

    bge_m3: BGEM3Embeddings | None = None
    if bge_id:
        bge_m3 = BGEM3Embeddings(
            model_id=bge_id,
            dense=DenseEmbeddings(
                model_id=bge_id,
                vectors=np.zeros((n, 2), dtype=np.float32),
            ),
            sparse=SparseEmbeddings(
                model_id=bge_id,
                items=[
                    SparseEmbedding(
                        dim=4,
                        indices=np.array([0], dtype=np.uint32),
                        values=np.array([1.0], dtype=np.float32),
                    )
                    for _ in texts
                ],
            ),
            colbert=[np.zeros((2, 2), dtype=np.float32) for _ in texts],
        )

    return MeowEmbedClient._build_parsed_embed_response(
        texts_count=n,
        dense=dense,
        sparse=sparse,
        bge_m3=bge_m3,
        payload=payload,
        raw=None,
    )


class _EmbedHarnessClient(MeowEmbedClient):
    def _embed_remote_sync(
        self, payload: EmbedRequestAny
    ) -> ParsedEmbedResponseVariant:
        return _fake_parsed_embed(payload)


class _AEmbedHarnessClient(MeowEmbedClient):
    async def _embed_remote(
        self, payload: EmbedRequestAny
    ) -> ParsedEmbedResponseVariant:
        return _fake_parsed_embed(payload)


def _sync_harness() -> _EmbedHarnessClient:
    return _EmbedHarnessClient(
        client=httpx.Client(base_url="http://embed-harness.invalid")
    )


def _async_harness() -> _AEmbedHarnessClient:
    return _AEmbedHarnessClient(
        aclient=httpx.AsyncClient(base_url="http://embed-harness.invalid")
    )


def test_embed_dense_sync_types() -> None:
    client = _sync_harness()
    payload: DenseEmbedRequestDict = {"texts": ["a"], "dense_model_id": "d"}
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseDense)
    assert r.dense.vectors.shape == (1, 2)

    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedResponseDense" has no attribute "sparse"
    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedResponseDense" has no attribute "bgeM3"


def test_embed_sparse_sync_types() -> None:
    client = _sync_harness()
    payload: SparseEmbedRequestDict = {"texts": ["a"], "sparse_model_id": "s"}
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseSparse)
    assert r.sparse.items[0].dim == 4

    with pytest.raises(AttributeError):
        _ = r.dense  # E: "ParsedEmbedResponseSparse" has no attribute "dense"
    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedResponseSparse" has no attribute "bgeM3"


def test_embed_bge_sync_types() -> None:
    client = _sync_harness()
    payload: BGEM3EmbedRequestDict = {"texts": ["a"], "bge_model_id": "b"}
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseBGEM3)
    assert r.bgeM3.colbert[0].shape == (2, 2)

    with pytest.raises(AttributeError):
        _ = r.dense  # E: "ParsedEmbedResponseBGEM3" has no attribute "dense"
    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedResponseBGEM3" has no attribute "sparse"


def test_embed_dense_sparse_sync_types() -> None:
    client = _sync_harness()
    payload: DenseSparseEmbedRequestDict = {
        "texts": ["a"],
        "dense_model_id": "d",
        "sparse_model_id": "s",
    }
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseDenseSparse)
    assert r.dense.model_id == "d"
    assert r.sparse.model_id == "s"

    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedResponseDenseSparse" has no attribute "bgeM3"


def test_embed_dense_bge_sync_types() -> None:
    client = _sync_harness()
    payload: DenseBGEM3EmbedRequestDict = {
        "texts": ["a"],
        "dense_model_id": "d",
        "bge_model_id": "b",
    }
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseDenseBGEM3)

    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedResponseDenseBGEM3" has no attribute "sparse"


def test_embed_sparse_bge_sync_types() -> None:
    client = _sync_harness()
    payload: SparseBGEM3EmbedRequestDict = {
        "texts": ["a"],
        "sparse_model_id": "s",
        "bge_model_id": "b",
    }
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseSparseBGEM3)

    with pytest.raises(AttributeError):
        _ = r.dense  # E: "ParsedEmbedResponseSparseBGEM3" has no attribute "dense"


def test_embed_all_sync_types() -> None:
    client = _sync_harness()
    payload: DenseSparseBGEM3EmbedRequestDict = {
        "texts": ["a"],
        "dense_model_id": "d",
        "sparse_model_id": "s",
        "bge_model_id": "b",
    }
    r = client.embed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseDenseSparseBGEM3)


def test_embed_one_dense_sync_types() -> None:
    client = _sync_harness()
    payload: DenseEmbedOneRequestDict = {"text": "a", "dense_model_id": "d"}
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneDense)
    assert r.dense.vector.shape == (2,)

    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedOneDense" has no attribute "sparse"
    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedOneDense" has no attribute "bgeM3"


def test_embed_one_sparse_sync_types() -> None:
    client = _sync_harness()
    payload: SparseEmbedOneRequestDict = {"text": "a", "sparse_model_id": "s"}
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneSparse)
    assert r.sparse.item.dim == 4

    with pytest.raises(AttributeError):
        _ = r.dense  # E: "ParsedEmbedOneSparse" has no attribute "dense"
    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedOneSparse" has no attribute "bgeM3"


def test_embed_one_bge_sync_types() -> None:
    client = _sync_harness()
    payload: BGEM3EmbedOneRequestDict = {"text": "a", "bge_model_id": "b"}
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneBGEM3)
    assert r.bgeM3.colbert.shape == (2, 2)

    with pytest.raises(AttributeError):
        _ = r.dense  # E: "ParsedEmbedOneBGEM3" has no attribute "dense"
    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedOneBGEM3" has no attribute "sparse"


def test_embed_one_dense_sparse_sync_types() -> None:
    client = _sync_harness()
    payload: DenseSparseEmbedOneRequestDict = {
        "text": "a",
        "dense_model_id": "d",
        "sparse_model_id": "s",
    }
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneDenseSparse)
    assert r.dense.model_id == "d"
    assert r.sparse.model_id == "s"

    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedOneDenseSparse" has no attribute "bgeM3"


def test_embed_one_dense_bge_sync_types() -> None:
    client = _sync_harness()
    payload: DenseBGEM3EmbedOneRequestDict = {
        "text": "a",
        "dense_model_id": "d",
        "bge_model_id": "b",
    }
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneDenseBGEM3)

    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedOneDenseBGEM3" has no attribute "sparse"


def test_embed_one_sparse_bge_sync_types() -> None:
    client = _sync_harness()
    payload: SparseBGEM3EmbedOneRequestDict = {
        "text": "a",
        "sparse_model_id": "s",
        "bge_model_id": "b",
    }
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneSparseBGEM3)

    with pytest.raises(AttributeError):
        _ = r.dense  # E: "ParsedEmbedOneSparseBGEM3" has no attribute "dense"


def test_embed_one_all_sync_types() -> None:
    client = _sync_harness()
    payload: DenseSparseBGEM3EmbedOneRequestDict = {
        "text": "a",
        "dense_model_id": "d",
        "sparse_model_id": "s",
        "bge_model_id": "b",
    }
    r = client.embed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneDenseSparseBGEM3)


@pytest.mark.anyio
async def test_aembed_dense_types() -> None:
    client = _async_harness()
    payload: DenseEmbedRequestDict = {"texts": ["a"], "dense_model_id": "d"}
    r = await client.aembed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseDense)

    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedResponseDense" has no attribute "sparse"
    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedResponseDense" has no attribute "bgeM3"


@pytest.mark.anyio
async def test_aembed_dense_sparse_types() -> None:
    client = _async_harness()
    payload: DenseSparseEmbedRequestDict = {
        "texts": ["a"],
        "dense_model_id": "d",
        "sparse_model_id": "s",
    }
    r = await client.aembed(payload, use_cache=False)
    assert_type(r, ParsedEmbedResponseDenseSparse)

    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedResponseDenseSparse" has no attribute "bgeM3"


@pytest.mark.anyio
async def test_aembed_one_dense_types() -> None:
    client = _async_harness()
    payload: DenseEmbedOneRequestDict = {"text": "a", "dense_model_id": "d"}
    r = await client.aembed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneDense)

    with pytest.raises(AttributeError):
        _ = r.sparse  # E: "ParsedEmbedOneDense" has no attribute "sparse"
    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedOneDense" has no attribute "bgeM3"


@pytest.mark.anyio
async def test_aembed_one_dense_sparse_types() -> None:
    client = _async_harness()
    payload: DenseSparseEmbedOneRequestDict = {
        "text": "a",
        "dense_model_id": "d",
        "sparse_model_id": "s",
    }
    r = await client.aembed_one(payload, use_cache=False)
    assert_type(r, ParsedEmbedOneDenseSparse)

    with pytest.raises(AttributeError):
        _ = r.bgeM3  # E: "ParsedEmbedOneDenseSparse" has no attribute "bgeM3"


def test_embed_one_raises_when_text_missing() -> None:
    client = _sync_harness()
    with pytest.raises(ValueError, match="text must be provided"):
        client.embed_one(cast(Any, {"dense_model_id": "d"}), use_cache=False)


@pytest.mark.anyio
async def test_aembed_one_raises_when_text_missing() -> None:
    client = _async_harness()
    with pytest.raises(ValueError, match="text must be provided"):
        await client.aembed_one(cast(Any, {"dense_model_id": "d"}), use_cache=False)


def test_client_requires_http_client() -> None:
    with pytest.raises(ValueError, match="client or aclient"):
        MeowEmbedClient()


def test_embed_raises_when_texts_missing_or_empty() -> None:
    client = _sync_harness()
    with pytest.raises(ValueError, match="At least one text must be provided"):
        client.embed(cast(Any, {"dense_model_id": "d"}), use_cache=False)
    with pytest.raises(ValueError, match="At least one text must be provided"):
        client.embed(cast(Any, {"texts": [], "dense_model_id": "d"}), use_cache=False)


def test_embed_raises_when_no_model_ids() -> None:
    client = _sync_harness()
    with pytest.raises(ValueError, match="At least one model must be provided"):
        client.embed(cast(Any, {"texts": ["a"]}), use_cache=False)


@pytest.mark.anyio
async def test_aembed_raises_when_texts_empty() -> None:
    client = _async_harness()
    with pytest.raises(ValueError, match="At least one text must be provided"):
        await client.aembed(
            cast(Any, {"texts": [], "dense_model_id": "d"}), use_cache=False
        )


@pytest.mark.anyio
async def test_aembed_raises_when_no_models() -> None:
    client = _async_harness()
    with pytest.raises(ValueError, match="At least one model must be provided"):
        await client.aembed(cast(Any, {"texts": ["a"]}), use_cache=False)
