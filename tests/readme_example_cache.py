import tempfile
from pathlib import Path

import httpx

from finite_embeddings.client import DenseSparseEmbedRequestDict, FiniteEmbeddingsClient

with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "embed-cache.lmdb"
    client = FiniteEmbeddingsClient(
        httpx.Client(base_url="http://127.0.0.1:8067"),
        use_cache=True,
        cache_path=cache_path,
    )
    payload: DenseSparseEmbedRequestDict = {
        "texts": ["hello world", "cache demo"],
        "dense_model_id": "sergeyzh/BERTA",
        "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    }
    client.embed(payload)  # fills LMDB on miss
    client.embed(payload)  # reads from LMDB
