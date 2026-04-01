import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finite_embeddings.client import EmbedRequestDict, FiniteEmbeddingsClient


@pytest.mark.anyio
async def test_client_models_and_embed_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as http_client:
        client = FiniteEmbeddingsClient(http_client)

        models = await client.models()
        assert "models" in models
        assert len(models["models"]) > 0

        available_model_ids = {model["id"] for model in models["models"]}
        dense_model_id = "sergeyzh/BERTA"
        sparse_model_id = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"

        assert dense_model_id in available_model_ids
        assert sparse_model_id in available_model_ids

        result = await client.embed(
            {
                "texts": ["hello world", "server integration test"],
                "dense_model_id": dense_model_id,
                "sparse_model_id": sparse_model_id,
            }
        )

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
async def test_client_embed_cache_hit_skips_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    payload: EmbedRequestDict = {
        "texts": ["cache me", "cache me too"],
        "dense_model_id": dense_model_id,
        "sparse_model_id": sparse_model_id,
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as http_client:
        client = FiniteEmbeddingsClient(
            http_client,
            use_cache=True,
            cache_path=tmp_path / "client-cache.lmdb",
        )

        first = await client.embed(payload)
        assert first.dense is not None
        assert first.sparse is not None

        async def _fail_if_remote_called(payload_arg: object) -> object:
            raise AssertionError(f"Expected cache hit, but remote was called with: {payload_arg}")

        monkeypatch.setattr(client, "_embed_remote", _fail_if_remote_called)
        second = await client.embed(payload)

        assert second.texts_count == first.texts_count
        assert second.dense is not None
        assert second.sparse is not None
        assert second.dense.vectors.shape == first.dense.vectors.shape
        expected_cached_dense = first.dense.vectors.astype(np.float16).astype(np.float32)
        assert np.array_equal(second.dense.vectors, expected_cached_dense)
        assert len(second.sparse.items) == len(first.sparse.items)
        for first_item, second_item in zip(first.sparse.items, second.sparse.items, strict=True):
            assert first_item.dim == second_item.dim
            assert np.array_equal(first_item.indices, second_item.indices)
            assert np.allclose(first_item.values, second_item.values)


@pytest.mark.anyio
async def test_client_rerank_live_server() -> None:
    base_url = os.getenv("FINITE_EMBEDDINGS_BASE_URL", "http://127.0.0.1:8067")
    reranker_model_id = "BAAI/bge-reranker-v2-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as http_client:
        client = FiniteEmbeddingsClient(http_client)
        models = await client.models()
        available_model_ids = {model["id"] for model in models["models"] if model["type"] == "reranker"}
        assert reranker_model_id in available_model_ids

        reranked = await client.rerank(
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
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as http_client:
        client = FiniteEmbeddingsClient(http_client)
        models = await client.models()
        bge_models = [model for model in models["models"] if model["type"] == "bgeM3"]
        available_model_ids = {model["id"] for model in models["models"] if model["type"] == "bgeM3"}
        if bge_model_id not in available_model_ids:
            pytest.skip(f"{bge_model_id} is not loaded on server")
        bge_model_info = next(model for model in bge_models if model["id"] == bge_model_id)
        assert bge_model_info.get("dense_dimensions") is not None
        assert bge_model_info.get("sparse_dimensions") is not None
        assert bge_model_info.get("batch_size") is not None

        raw_response = await http_client.post(
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

        parsed = await client.embed(
            {
                "texts": ["What is BGE M3?", "Definition of BM25"],
                "bge_model_id": bge_model_id,
            }
        )
        assert parsed.texts_count == 2
        assert parsed.bgeM3 is not None
        assert parsed.bgeM3.model_id == bge_model_id
        assert parsed.bgeM3.dense.vectors.shape[0] == 2
        assert len(parsed.bgeM3.sparse.items) == 2
        assert len(parsed.bgeM3.colbert) == 2
