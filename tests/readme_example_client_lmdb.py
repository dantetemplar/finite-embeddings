import asyncio
import tempfile
from pathlib import Path

import httpx
import numpy as np

from finite_embeddings.client import FiniteEmbeddingsClient


def numpy_info(array: np.ndarray) -> str:
    return f"[ndarray] shape={array.shape}, dtype={array.dtype}"


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "embed-cache.lmdb"
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8067", timeout=30.0
        ) as http_client:
            client = FiniteEmbeddingsClient(
                http_client,
                use_cache=True,
                cache_path=cache_path,
            )
            models = await client.models()
            payload = {
                "texts": ["hello world", "one request for both outputs"],
                "dense_model_id": "sergeyzh/BERTA",
                "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
            }
            first = await client.embed(payload)
            second = await client.embed(payload)
            assert first.texts_count == second.texts_count
            assert first.dense is not None and second.dense is not None
            assert np.array_equal(first.dense.vectors, second.dense.vectors)
            assert first.sparse is not None and second.sparse is not None
            for a, b in zip(first.sparse.items, second.sparse.items, strict=True):
                assert np.array_equal(a.indices, b.indices)
                assert np.array_equal(a.values, b.values)
            print("LMDB cache path:", cache_path)
            print(
                "dense (from cache on second call):", numpy_info(second.dense.vectors)
            )

            bge_model_id = "BAAI/bge-m3"
            available_bge_ids = {
                model["id"] for model in models["models"] if model["type"] == "bgeM3"
            }
            if bge_model_id in available_bge_ids:
                bge_payload = {
                    "texts": ["cache bge one", "cache bge two"],
                    "bge_model_id": bge_model_id,
                }
                first_bge = await client.embed(bge_payload)
                second_bge = await client.embed(bge_payload)
                assert first_bge.bgeM3 is not None and second_bge.bgeM3 is not None
                assert np.array_equal(
                    first_bge.bgeM3.dense.vectors, second_bge.bgeM3.dense.vectors
                )
                print(
                    "bge dense (from cache on second call):",
                    numpy_info(second_bge.bgeM3.dense.vectors),
                )


asyncio.run(main())
