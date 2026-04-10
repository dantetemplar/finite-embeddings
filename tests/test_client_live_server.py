import os
import sys
from pathlib import Path

import httpx
import lmdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finite_embeddings.client import (
    EmbedRequestPayload,
    FiniteEmbeddingsClient,
    ParsedEmbedOneDenseSparse,
    ParsedEmbedResponseBGEM3,
    ParsedEmbedResponseDenseSparse,
)


@pytest.mark.anyio
async def test_client_models_and_embed_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(aclient=httpx_aclient)

        models = await client.amodels()
        assert "models" in models
        assert len(models["models"]) > 0

        available_model_ids = {model["id"] for model in models["models"]}
        dense_model_id = "sergeyzh/BERTA"
        sparse_model_id = (
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        )

        assert dense_model_id in available_model_ids
        assert sparse_model_id in available_model_ids

        result = await client.aembed(
            {
                "texts": ["hello world", "server integration test"],
                "dense_model_id": dense_model_id,
                "sparse_model_id": sparse_model_id,
            }
        )
        assert isinstance(result, ParsedEmbedResponseDenseSparse)

        assert result.texts_count == 2
        assert result.dense is not None
        assert result.dense.model_id == dense_model_id
        assert result.dense.vectors.shape[0] == 2
        assert result.dense.vectors.ndim == 2
        assert result.sparse is not None
        assert result.sparse.model_id == sparse_model_id
        assert len(result.sparse.items) == 2
        assert result.sparse.items[0].indices.size == result.sparse.items[0].values.size


@pytest.mark.anyio
async def test_client_aembed_one_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(aclient=httpx_aclient)

        models = await client.amodels()
        available_model_ids = {model["id"] for model in models["models"]}
        dense_model_id = "sergeyzh/BERTA"
        sparse_model_id = (
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        )

        assert dense_model_id in available_model_ids
        assert sparse_model_id in available_model_ids

        one = await client.aembed_one(
            {
                "text": "single string embed_one",
                "dense_model_id": dense_model_id,
                "sparse_model_id": sparse_model_id,
            }
        )
        assert isinstance(one, ParsedEmbedOneDenseSparse)
        assert one.dense.vector.ndim == 1
        assert one.sparse.item.indices.size == one.sparse.item.values.size


@pytest.mark.anyio
async def test_client_embed_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = (
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    )
    payload: EmbedRequestPayload = {
        "texts": ["cache me", "cache me too"],
        "dense_model_id": dense_model_id,
        "sparse_model_id": sparse_model_id,
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(
            aclient=httpx_aclient,
            use_cache=True,
            cache_path=tmp_path / "client-cache.lmdb",
        )

        first = await client.aembed(payload)
        assert isinstance(first, ParsedEmbedResponseDenseSparse)

        async def _fail_if_remote_called(payload_arg: object) -> object:
            raise AssertionError(
                f"Expected cache hit, but remote was called with: {payload_arg}"
            )

        monkeypatch.setattr(client, "_embed_remote", _fail_if_remote_called)
        second = await client.aembed(payload)
        assert isinstance(second, ParsedEmbedResponseDenseSparse)

        assert second.texts_count == first.texts_count
        assert second.dense is not None
        assert second.sparse is not None
        assert second.dense.vectors.shape == first.dense.vectors.shape
        assert np.array_equal(first.dense.vectors, second.dense.vectors)
        assert len(second.sparse.items) == len(first.sparse.items)
        for first_item, second_item in zip(
            first.sparse.items, second.sparse.items, strict=True
        ):
            assert first_item.dim == second_item.dim
            assert np.array_equal(first_item.indices, second_item.indices)
            assert np.array_equal(first_item.values, second_item.values)


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_client_accepts_external_lmdb_environment(
    tmp_path: Path, anyio_backend: str
) -> None:
    assert anyio_backend == "asyncio"
    cache_dir = tmp_path / "external-cache.lmdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_env = lmdb.open(str(cache_dir), map_size=2 * 1024 * 1024 * 1024)
    try:
        base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
            client = FiniteEmbeddingsClient(aclient=httpx_aclient, cache=cache_env)
            assert client._cache is cache_env
            assert client._use_cache is True
    finally:
        cache_env.close()


@pytest.mark.anyio
async def test_client_rerank_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    reranker_model_id = "BAAI/bge-reranker-v2-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(aclient=httpx_aclient)
        models = await client.amodels()
        available_model_ids = {
            model["id"] for model in models["models"] if model["type"] == "reranker"
        }
        if reranker_model_id not in available_model_ids:
            pytest.skip(f"{reranker_model_id} is not loaded on server")

        reranked = await client.arerank(
            {
                "reranker_model_id": reranker_model_id,
                "queries": ["what is panda?", "capital of france"],
                "docs": [
                    "The giant panda is a bear species endemic to China.",
                    "Paris is the capital city of France.",
                ],
            }
        )

    assert reranked.model_id == reranker_model_id
    assert reranked.shape == (2, 2)
    assert len(reranked.scores) == 2
    assert all(len(row) == 2 for row in reranked.scores)
    assert all(isinstance(score, float) for row in reranked.scores for score in row)


@pytest.mark.anyio
async def test_client_embed_bge_m3_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(aclient=httpx_aclient)
        models = await client.amodels()
        bge_models = [model for model in models["models"] if model["type"] == "bgeM3"]
        available_model_ids = {
            model["id"] for model in models["models"] if model["type"] == "bgeM3"
        }
        if bge_model_id not in available_model_ids:
            pytest.skip(f"{bge_model_id} is not loaded on server")
        bge_model_info = next(
            model for model in bge_models if model["id"] == bge_model_id
        )
        assert bge_model_info.get("dense_dimensions") is not None
        assert bge_model_info.get("sparse_dimensions") is not None
        assert bge_model_info.get("batch_size") is not None

        raw_response = await httpx_aclient.post(
            "/embed",
            json={
                "texts": ["What is BGE M3?", "Definition of BM25"],
                "bge_model_id": bge_model_id,
            },
        )
        raw_response.raise_for_status()
        raw_payload = raw_response.json()
        assert raw_payload["texts_count"] == 2
        assert raw_payload["bgeM3"]["model_id"] == bge_model_id
        assert raw_payload["bgeM3"]["dense"]["model_id"] == bge_model_id
        assert raw_payload["bgeM3"]["sparse"]["model_id"] == bge_model_id
        assert len(raw_payload["bgeM3"]["colbert"]) == 2

        parsed = await client.aembed(
            {
                "texts": ["What is BGE M3?", "Definition of BM25"],
                "bge_model_id": bge_model_id,
            }
        )
        assert isinstance(parsed, ParsedEmbedResponseBGEM3)
        assert parsed.texts_count == 2
        assert parsed.bgeM3 is not None
        assert parsed.bgeM3.model_id == bge_model_id
        assert parsed.bgeM3.dense.vectors.shape[0] == 2
        assert len(parsed.bgeM3.sparse.items) == 2
        assert len(parsed.bgeM3.colbert) == 2


@pytest.mark.anyio
async def test_client_embed_bge_m3_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    payload: EmbedRequestPayload = {
        "texts": ["cache bge one", "cache bge two"],
        "bge_model_id": bge_model_id,
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(
            aclient=httpx_aclient,
            use_cache=True,
            cache_path=tmp_path / "client-cache.lmdb",
        )
        models = await client.amodels()
        available_model_ids = {
            model["id"] for model in models["models"] if model["type"] == "bgeM3"
        }
        if bge_model_id not in available_model_ids:
            pytest.skip(f"{bge_model_id} is not loaded on server")

        first = await client.aembed(payload)
        assert isinstance(first, ParsedEmbedResponseBGEM3)

        async def _fail_if_remote_called(payload_arg: object) -> object:
            raise AssertionError(
                f"Expected cache hit, but remote was called with: {payload_arg}"
            )

        monkeypatch.setattr(client, "_embed_remote", _fail_if_remote_called)
        second = await client.aembed(payload)
        assert isinstance(second, ParsedEmbedResponseBGEM3)
        assert np.array_equal(first.bgeM3.dense.vectors, second.bgeM3.dense.vectors)
        assert len(first.bgeM3.sparse.items) == len(second.bgeM3.sparse.items)
        for first_item, second_item in zip(
            first.bgeM3.sparse.items, second.bgeM3.sparse.items, strict=True
        ):
            assert first_item.dim == second_item.dim
            assert np.array_equal(first_item.indices, second_item.indices)
            assert np.array_equal(first_item.values, second_item.values)
        assert len(first.bgeM3.colbert) == len(second.bgeM3.colbert)
        for first_row, second_row in zip(
            first.bgeM3.colbert, second.bgeM3.colbert, strict=True
        ):
            assert np.array_equal(first_row, second_row)


@pytest.mark.anyio
async def test_sync_api_requires_sync_client() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        client = FiniteEmbeddingsClient(aclient=httpx_aclient)
        with pytest.raises(RuntimeError, match="client is not configured"):
            client.models()


@pytest.mark.anyio
async def test_client_models_and_embed_sync_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = (
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            client = FiniteEmbeddingsClient(sync_httpx_aclient, httpx_aclient)

            models = client.models()
            assert "models" in models
            assert len(models["models"]) > 0

            available_model_ids = {model["id"] for model in models["models"]}
            assert dense_model_id in available_model_ids
            assert sparse_model_id in available_model_ids

            result = client.embed(
                {
                    "texts": ["hello world", "server integration test"],
                    "dense_model_id": dense_model_id,
                    "sparse_model_id": sparse_model_id,
                }
            )
            assert isinstance(result, ParsedEmbedResponseDenseSparse)

            assert result.texts_count == 2
            assert result.dense is not None
            assert result.dense.model_id == dense_model_id
            assert result.dense.vectors.shape[0] == 2
            assert result.dense.vectors.ndim == 2
            assert result.sparse is not None
            assert result.sparse.model_id == sparse_model_id
            assert len(result.sparse.items) == 2
            assert (
                result.sparse.items[0].indices.size
                == result.sparse.items[0].values.size
            )


@pytest.mark.anyio
async def test_client_embed_sync_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = (
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    )
    payload: EmbedRequestPayload = {
        "texts": ["cache me sync", "cache me sync too"],
        "dense_model_id": dense_model_id,
        "sparse_model_id": sparse_model_id,
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            client = FiniteEmbeddingsClient(
                sync_httpx_aclient,
                httpx_aclient,
                use_cache=True,
                cache_path=tmp_path / "client-cache-sync.lmdb",
            )

            first = client.embed(payload)
            assert isinstance(first, ParsedEmbedResponseDenseSparse)

            def _fail_if_remote_called(payload_arg: object) -> object:
                raise AssertionError(
                    f"Expected cache hit, but remote was called with: {payload_arg}"
                )

            monkeypatch.setattr(client, "_embed_remote_sync", _fail_if_remote_called)
            second = client.embed(payload)
            assert isinstance(second, ParsedEmbedResponseDenseSparse)

            assert second.texts_count == first.texts_count
            assert second.dense is not None
            assert second.sparse is not None
            assert second.dense.vectors.shape == first.dense.vectors.shape
            assert np.array_equal(first.dense.vectors, second.dense.vectors)
            assert len(second.sparse.items) == len(first.sparse.items)
            for first_item, second_item in zip(
                first.sparse.items, second.sparse.items, strict=True
            ):
                assert first_item.dim == second_item.dim
                assert np.array_equal(first_item.indices, second_item.indices)
                assert np.array_equal(first_item.values, second_item.values)


@pytest.mark.anyio
async def test_client_sync_rerank_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    reranker_model_id = "BAAI/bge-reranker-v2-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            client = FiniteEmbeddingsClient(sync_httpx_aclient, httpx_aclient)
            models = client.models()
            available_model_ids = {
                model["id"] for model in models["models"] if model["type"] == "reranker"
            }
            if reranker_model_id not in available_model_ids:
                pytest.skip(f"{reranker_model_id} is not loaded on server")

            reranked = client.rerank(
                {
                    "reranker_model_id": reranker_model_id,
                    "queries": ["what is panda?", "capital of france"],
                    "docs": [
                        "The giant panda is a bear species endemic to China.",
                        "Paris is the capital city of France.",
                    ],
                }
            )

    assert reranked.model_id == reranker_model_id
    assert reranked.shape == (2, 2)
    assert len(reranked.scores) == 2
    assert all(len(row) == 2 for row in reranked.scores)
    assert all(isinstance(score, float) for row in reranked.scores for score in row)


@pytest.mark.anyio
async def test_client_embed_bge_m3_sync_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            client = FiniteEmbeddingsClient(sync_httpx_aclient, httpx_aclient)
            models = client.models()
            available_model_ids = {
                model["id"] for model in models["models"] if model["type"] == "bgeM3"
            }
            if bge_model_id not in available_model_ids:
                pytest.skip(f"{bge_model_id} is not loaded on server")

            parsed = client.embed(
                {
                    "texts": ["What is BGE M3?", "Definition of BM25"],
                    "bge_model_id": bge_model_id,
                }
            )
            assert isinstance(parsed, ParsedEmbedResponseBGEM3)
            assert parsed.texts_count == 2
            assert parsed.bgeM3 is not None
            assert parsed.bgeM3.model_id == bge_model_id
            assert parsed.bgeM3.dense.vectors.shape[0] == 2
            assert len(parsed.bgeM3.sparse.items) == 2
            assert len(parsed.bgeM3.colbert) == 2


@pytest.mark.anyio
async def test_client_embed_bge_m3_sync_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    payload: EmbedRequestPayload = {
        "texts": ["cache bge sync one", "cache bge sync two"],
        "bge_model_id": bge_model_id,
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            client = FiniteEmbeddingsClient(
                sync_httpx_aclient,
                httpx_aclient,
                use_cache=True,
                cache_path=tmp_path / "client-cache-bge-sync.lmdb",
            )
            models = client.models()
            available_model_ids = {
                model["id"] for model in models["models"] if model["type"] == "bgeM3"
            }
            if bge_model_id not in available_model_ids:
                pytest.skip(f"{bge_model_id} is not loaded on server")

            first = client.embed(payload)
            assert isinstance(first, ParsedEmbedResponseBGEM3)

            def _fail_if_remote_called(payload_arg: object) -> object:
                raise AssertionError(
                    f"Expected cache hit, but remote was called with: {payload_arg}"
                )

            monkeypatch.setattr(client, "_embed_remote_sync", _fail_if_remote_called)
            second = client.embed(payload)
            assert isinstance(second, ParsedEmbedResponseBGEM3)
            assert np.array_equal(first.bgeM3.dense.vectors, second.bgeM3.dense.vectors)
            assert len(first.bgeM3.sparse.items) == len(second.bgeM3.sparse.items)
            for first_item, second_item in zip(
                first.bgeM3.sparse.items, second.bgeM3.sparse.items, strict=True
            ):
                assert first_item.dim == second_item.dim
                assert np.array_equal(first_item.indices, second_item.indices)
                assert np.array_equal(first_item.values, second_item.values)
            assert len(first.bgeM3.colbert) == len(second.bgeM3.colbert)
            for first_row, second_row in zip(
                first.bgeM3.colbert, second.bgeM3.colbert, strict=True
            ):
                assert np.array_equal(first_row, second_row)
