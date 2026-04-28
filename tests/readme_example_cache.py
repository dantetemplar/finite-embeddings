import tempfile
from pathlib import Path

import httpx

from meow_embed import EmbedCache, MeowEmbedClient
from meow_embed.types import DenseSparseEmbedRequestDict

with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "embed-cache.lmdb"
    cache = EmbedCache.open(cache_path)
    try:
        meow = MeowEmbedClient(
            httpx.Client(base_url="http://127.0.0.1:8067"),
            cache=cache,
        )
        payload: DenseSparseEmbedRequestDict = {
            "texts": ["hello world", "cache demo"],
            "dense_model_id": "sergeyzh/BERTA",
            "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        }
        meow.embed(payload)  # fills LMDB on miss
        meow.embed(payload)  # reads from LMDB
    finally:
        cache.close()
