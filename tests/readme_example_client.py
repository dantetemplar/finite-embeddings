import httpx
import numpy as np

from finite_embeddings.client import FiniteEmbeddingsClient


def numpy_info(array: np.ndarray) -> str:
    return f"[ndarray] shape={array.shape}, dtype={array.dtype}"


client = FiniteEmbeddingsClient(
    client=httpx.Client(base_url="http://127.0.0.1:8067"),
    aclient=httpx.AsyncClient(base_url="http://127.0.0.1:8067"),  # NOTE: async version
)
models = client.models()  # NOTE: async version is await client.amodels()
print(models)
# { 'models': [
#   {
#     'batch_size': 32,
#     'dimensions': 768,
#     'id': 'sergeyzh/BERTA',
#     'type': 'dense'},
#   { 'batch_size': 32,
#     'dimensions': 105879,
#     'id': 'opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1',
#     'type': 'sparse'},
#   { 'batch_size': 128,
#     'dimensions': None,
#     'id': 'BAAI/bge-reranker-v2-m3',
#     'type': 'reranker'},
#   { 'batch_size': 256,
#     'dimensions': None,
#     'id': 'BAAI/bge-m3',
#     'type': 'bgeM3'}]}

result = client.embed(
    {
        "texts": ["hello world", "one request for both outputs"],
        "dense_model_id": "sergeyzh/BERTA",
        "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        "bge_model_id": "BAAI/bge-m3",
    }
)  # NOTE: async version is await client.aembed(...)
print(f"texts_count: {result.texts_count}")
print("dense")
print(f"    .model_id: {result.dense.model_id}")
print(f"    .vectors: {numpy_info(result.dense.vectors)}")
# dense
# .model_id: sergeyzh/BERTA
# .vectors: [ndarray] shape=(2, 768), dtype=float32

print("sparse")
print(f"    .model_id: {result.sparse.model_id}")
print(f"    .items: total {len(result.sparse.items)} items")
print(f"        .[0].indices: {numpy_info(result.sparse.items[0].indices)}")
print(f"        .[0].values: {numpy_info(result.sparse.items[0].values)}")
# sparse
# .model_id: opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1
# .items: total 2 items
#     .[0].indices: [ndarray] shape=(226,), dtype=uint32
#     .[0].values: [ndarray] shape=(226,), dtype=float32

print("bgeM3")
print(f"    .model_id: {result.bgeM3.model_id}")
print(f"    .dense.vectors: {numpy_info(result.bgeM3.dense.vectors)}")
print(f"    .sparse.items: total {len(result.bgeM3.sparse.items)} items")
print(f"        .[0].indices: {numpy_info(result.bgeM3.sparse.items[0].indices)}")
print(f"        .[0].values: {numpy_info(result.bgeM3.sparse.items[0].values)}")
print(f"    .colbert: total {len(result.bgeM3.colbert)} items")
print(f"        .[0]: {numpy_info(result.bgeM3.colbert[0])}")
# bgeM3
# .model_id: BAAI/bge-m3
# .dense.vectors: [ndarray] shape=(2, 1024), dtype=float32
# .sparse.items: total 2 items
#     .[0].indices: [ndarray] shape=(3,), dtype=uint32
#     .[0].values: [ndarray] shape=(3,), dtype=float32
# .colbert: total 2 items
#     .[0]: [ndarray] shape=(4, 1024), dtype=float32
