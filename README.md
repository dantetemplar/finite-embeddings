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

# if you don't have git installed you can use the zip file
uv tool install "https://github.com/dantetemplar/finite-embeddings/archive/refs/heads/main.zip#egg=finite-embeddings[server]"
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

`finite-embeddings` handles only model flags (`--SentenceTransformer`, `--SparseEncoder`, `--FlagReranker`, `--BGEM3FlagModel`).
All other flags are passed directly to Uvicorn CLI parsing.

Defaults (if not provided): `--host 0.0.0.0 --port 8067`.

Example with extra Uvicorn options:

```bash
finite-embeddings \
  --SentenceTransformer "sergeyzh/BERTA" \
  --reload \
  --log-level debug \
  --proxy-headers \
  --forwarded-allow-ips="*"
```

### 2) Use Python client

Install client extras from git:

```bash
uv add "git+https://github.com/dantetemplar/finite-embeddings.git[client]"

# if you don't have git installed you can use the zip file
uv add "https://github.com/dantetemplar/finite-embeddings/archive/refs/heads/main.zip#egg=finite-embeddings[client]"
```

Example:

```python
import httpx
import numpy as np

from finite_embeddings.client import FiniteEmbeddingsClient


def numpy_info(array: np.ndarray) -> str:
    return f"[ndarray] shape={array.shape}, dtype={array.dtype}"


client = FiniteEmbeddingsClient(
    client=httpx.Client(base_url="http://127.0.0.1:8067"),
    aclient=httpx.AsyncClient(base_url="http://127.0.0.1:8067"), # NOTE: async version
)
models = client.models() # NOTE: async version is await client.amodels()
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
) # NOTE: async version is await client.aembed(...)
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
```

### Client-side LMDB cache

Enable with `use_cache=True`. Embeddings are keyed per text and model options. Set `cache_path` to an LMDB directory (default `~/.cache/finite-embeddings/client-cache.lmdb`); optional `cache_map_size` sets the map size in bytes (default 2 GiB).

```python
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


## Development

Install as editable package with the `all` extra and the `dev` dependency group (pytest, mypy, pyright, …):

```bash
uv pip install -e ".[all]" --group dev --extra-index-url https://download.pytorch.org/whl/cu126
```

Run pytest:
```bash
uv run -m pytest
```

Run mypy tests:

```bash
uv run -m mypy_test tests/
```
