import argparse
import json
import sys
from typing import Any, Literal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastAPI and load dense/sparse models on startup.",
        epilog=(
            "Examples:\n"
            "  finite-embeddings --SentenceTransformer sergeyzh/BERTA --SparseEncoder opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1\n"
            '  finite-embeddings --SparseEncoder opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 {"device":"cuda","max_active_dims":256}\n'
            '  finite-embeddings --FlagReranker BAAI/bge-reranker-v2-m3 {"use_fp16":true}\n'
            '  finite-embeddings --BGEM3FlagModel BAAI/bge-m3 {"use_fp16":true}\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8067)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--SentenceTransformer",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a dense model; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--SparseEncoder",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a sparse model; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--FlagReranker",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a reranker model; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--BGEM3FlagModel",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a BGE M3 model; optional next token can be JSON kwargs for this model.",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_model_specs(
    argv: list[str],
) -> list[tuple[Literal["dense", "sparse", "reranker", "bgeM3"], str, dict[str, Any]]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--reload", action="store_true")
    _, extras = parser.parse_known_args(argv)

    models: list[
        tuple[Literal["dense", "sparse", "reranker", "bgeM3"], str, dict[str, Any]]
    ] = []
    idx = 0
    while idx < len(extras):
        token = extras[idx]
        if token in (
            "--SentenceTransformer",
            "--SparseEncoder",
            "--FlagReranker",
            "--BGEM3FlagModel",
        ):
            if idx + 1 >= len(extras):
                raise SystemExit(f"Missing value for {token}.")
            model_id = extras[idx + 1]
            if model_id.startswith("--"):
                raise SystemExit(f"Missing value for {token}.")
            model_type: Literal["dense", "sparse", "reranker", "bgeM3"]
            if token == "--SentenceTransformer":
                model_type = "dense"
            elif token == "--SparseEncoder":
                model_type = "sparse"
            elif token == "--FlagReranker":
                model_type = "reranker"
            else:
                model_type = "bgeM3"
            kwargs: dict[str, Any] = {}
            next_idx = idx + 2
            if next_idx < len(extras):
                maybe_kwargs = extras[next_idx]
                if not maybe_kwargs.startswith("--"):
                    try:
                        parsed_kwargs = json.loads(maybe_kwargs)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(
                            f"Invalid JSON kwargs for {token} model {model_id!r}: {exc}"
                        ) from exc
                    if not isinstance(parsed_kwargs, dict):
                        raise SystemExit(
                            f"JSON kwargs for {token} model {model_id!r} must decode to an object."
                        )
                    kwargs = parsed_kwargs
                    next_idx += 1
            models.append((model_type, model_id, kwargs))
            idx = next_idx
            continue
        raise SystemExit(f"Unknown argument: {token}")
    return models


def main() -> None:
    args = parse_args()
    model_specs = parse_model_specs(sys.argv[1:])

    # Lazy import: avoid loading server stack on `--help`.
    from finite_embeddings import server

    model_config = server.ModelConfig(
        models=[
            server.ModelInstanceConfig(
                type=model_type, model_id=model_id, kwargs=kwargs
            )
            for model_type, model_id, kwargs in model_specs
        ]
    )
    server.run_server(
        model_config=model_config, host=args.host, port=args.port, reload=args.reload
    )
