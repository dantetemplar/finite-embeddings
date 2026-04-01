# finite-embeddings
> FastAPI embedding server and Python client for dense and sparse text embeddings

## What is it?

`finite-embeddings` is an HTTP API for text embeddings.

It can return **dense + sparse + BGE-M3 (dense, sparse, colbert) in one request**, with efficient transport:
- vectors are encoded as `base64`
- request body can be `gzip`-compressed
- response body can be `gzip`-compressed

## Why?

There is already [Infinity](https://github.com/michaelfeil/infinity), but this project exists because:
- sparse embeddings are needed
- BGE-M3 colbert and sparse embeddings are needed
- gzip request handling is needed end-to-end
- one round-trip for dense + sparse is needed
- patching upstream behavior was too complex
- newer `sentence-transformers` versions are required
- caching on client side is needed

## How to use

### 1) Start server

Install with server extras from git:

```bash
uv tool install "git+https://github.com/dantetemplar/finite-embeddings.git[server]"
```

Run server and load models:

```bash
finite-embeddings \
  --SentenceTransformer "sergeyzh/BERTA" \
  --SparseEncoder "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1" \
  --BGEM3FlagModel "BAAI/bge-m3" '{"use_fp16": true}' \
  --FlagReranker "BAAI/bge-reranker-v2-m3" '{"use_fp16": true}' \
  --host 0.0.0.0
```

...wait until server is ready (Uvicorn running on http://0.0.0.0:8067)...

### 2) Use Python client

Install client extras from git:

```bash
uv add "git+https://github.com/dantetemplar/finite-embeddings.git[client]"
```

Example:

```python
import asyncio

import httpx
import numpy as np

from finite_embeddings.client import FiniteEmbeddingsClient


def numpy_info(array: np.ndarray) -> str:
    return f"[ndarray] shape={array.shape}, dtype={array.dtype}"


async def main() -> None:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8067", timeout=30.0) as http_client:
        client = FiniteEmbeddingsClient(http_client)
        models = await client.models()
        print(models)
        # { 'models': [
        #   {
        #     'batch_size': 32,
        #     'colbert_dimensions': None,
        #     'dense_dimensions': 768,
        #     'dimensions': 768,
        #     'id': 'sergeyzh/BERTA',
        #     'sparse_dimensions': None,
        #     'type': 'dense'},
        #   { 'batch_size': 32,
        #     'colbert_dimensions': None,
        #     'dense_dimensions': None,
        #     'dimensions': 105879,
        #     'id': 'opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1',
        #     'sparse_dimensions': 105879,
        #     'type': 'sparse'},
        #   { 'batch_size': 128,
        #     'colbert_dimensions': None,
        #     'dense_dimensions': None,
        #     'dimensions': None,
        #     'id': 'BAAI/bge-reranker-v2-m3',
        #     'sparse_dimensions': None,
        #     'type': 'reranker'},
        #   { 'batch_size': 256,
        #     'colbert_dimensions': 1024,
        #     'dense_dimensions': 1024,
        #     'dimensions': None,
        #     'id': 'BAAI/bge-m3',
        #     'sparse_dimensions': 250002,
        #     'type': 'bgeM3'}]}

        result = await client.embed(
            {
                "texts": ["hello world", "one request for both outputs"],
                "dense_model_id": "sergeyzh/BERTA",
                "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
                "bge_model_id": "BAAI/bge-m3",
            }
        )
        print(f"texts_count: {result.texts_count}")
        if result.dense is not None:
            print("dense")
            print(f"    .model_id: {result.dense.model_id}")
            print(f"    .vectors: {numpy_info(result.dense.vectors)}")
            # dense
            # .model_id: sergeyzh/BERTA
            # .vectors: [ndarray] shape=(2, 768), dtype=float32

        if result.sparse is not None:
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


        if result.bgeM3 is not None:
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

asyncio.run(main())
```

### Client-side LMDB cache

The client can persist **per-text** dense, sparse, and BGE-M3 (dense+sparse+colbert) vectors in LMDB. Keys are derived from model id and request options (`dense_truncate_dim`, `dense_prompt`, `dense_task`, `sparse_max_active_dims`, `sparse_pruning_ratio`, and `sparse_task` when set) where applicable. Enable it with `use_cache=True` and optionally set `cache_path` (default: `~/.cache/finite-embeddings/client-cache.lmdb`) and `cache_map_size` (LMDB map size in bytes, default 2 GiB).

```python
import asyncio
import tempfile
from pathlib import Path

import httpx

from finite_embeddings.client import FiniteEmbeddingsClient


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "embed-cache.lmdb"
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8067", timeout=30.0) as http_client:
            client = FiniteEmbeddingsClient(
                http_client,
                use_cache=True,
                cache_path=cache_path,
            )
            payload = {
                "texts": ["hello world", "cache demo"],
                "dense_model_id": "sergeyzh/BERTA",
                "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
            }
            await client.embed(payload)  # fills LMDB on miss
            await client.embed(payload)  # reads from LMDB


asyncio.run(main())
```

### curl examples

#### Check loaded models

```bash
curl -sS "http://127.0.0.1:8067/models"
```

#### JSON request (dense + sparse in one call)

```bash
curl -sS -X POST "http://127.0.0.1:8067/embed" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "dense_model_id": "sergeyzh/BERTA",
    "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    "texts": ["hello", "world"],
    "dense_truncate_dim": null,
    "sparse_max_active_dims": null,
    "sparse_pruning_ratio": null
  }'
```

#### JSON request (rerank: query vs docs)

```bash
curl -sS -X POST "http://127.0.0.1:8067/rerank" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "reranker_model_id": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "docs": ["hi", "The giant panda is a bear species endemic to China."]
  }'
```

#### JSON request (rerank: queries vs docs bulk)

```bash
curl -sS -X POST "http://127.0.0.1:8067/rerank" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "reranker_model_id": "BAAI/bge-reranker-v2-m3",
    "queries": ["what is panda?", "capital of france"],
    "docs": ["The giant panda is a bear species endemic to China.", "Paris is the capital city of France."]
  }'
```

#### Gzipped request + gzipped response

```bash
cat <<'JSON' | gzip -c | curl -sS -X POST "http://127.0.0.1:8067/embed" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -H "Content-Encoding: gzip" \
  -H "Accept-Encoding: gzip" \
  --compressed \
  --data-binary @-
{
  "dense_model_id": "sergeyzh/BERTA",
  "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
  "texts": ["hello", "world"]
}
JSON
```

#### JSON request (BGE-M3 dense+sparse+colbert)

```bash
curl -sS -X POST "http://127.0.0.1:8067/embed" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "bge_model_id": "BAAI/bge-m3",
    "texts": ["What is BGE M3?", "Definition of BM25"]
  }'
```
