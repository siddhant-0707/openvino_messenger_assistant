#!/usr/bin/env python3
"""
Benchmark runner for OpenVINO Messenger Assistant

Measures:
- Model load time (LLM, Embeddings, Reranker)
- Retrieval latency and Recall@K (using keywords as weak labels)
- Generation latency and chars/sec (approx throughput)

Usage:
  python benchmarks/run_benchmarks.py \
    --device AUTO \
    --embedding-type text_embedding_pipeline \
    --queries benchmarks/sample_queries.json \
    --k 5 \
    --runs 3 \
    --output benchmarks/results

Results are printed to stdout and saved as JSON in the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).parent.parent

# Ensure src/ and examples/ on path
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def load_backend_module():
    """Import the examples backend lazily to avoid unnecessary side-effects."""
    # Import after sys.path is set
    import telegram_rag_gradio as backend  # type: ignore
    return backend


def time_call(fn, *args, **kwargs) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def ensure_vector_store(backend) -> bool:
    vector_dir: Path = backend.vector_store_path
    return vector_dir.exists() and any(vector_dir.iterdir())


def benchmark_model_load(backend, device: str, embedding_type: str, llm_model: str | None, embedding_model: str | None, rerank_model: str | None) -> Dict[str, float]:
    # Use existing reload_models helper for precise control
    # Fallbacks to current defaults if not provided
    language = backend.DEFAULT_LANGUAGE
    if llm_model is None:
        llm_model = backend.DEFAULT_LLM_MODEL
    if embedding_model is None:
        embedding_model = backend.DEFAULT_EMBEDDING_MODEL
    if rerank_model is None:
        rerank_model = backend.DEFAULT_RERANK_MODEL

    t_load, (model_info_text, status_msg) = time_call(
        backend.reload_models,
        language,
        llm_model,
        embedding_model,
        rerank_model,
        "int4",
        device,
        embedding_type,
    )

    # Best-effort parse sub-times if printed; otherwise lump into total
    return {
        "total_model_load_s": round(t_load, 4),
    }


def load_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "queries file must be a list"
    return data


def retrieval_metrics(backend, queries: List[Dict[str, Any]], k: int, runs: int) -> Dict[str, Any]:
    if backend.retriever is None:
        return {"error": "retriever not initialized"}

    latencies: List[float] = []
    hits = 0
    total = 0

    for _ in range(runs):
        for q in queries:
            query = q.get("query", "").strip()
            if not query:
                continue
            expected_keywords: List[str] = [s.lower() for s in q.get("keywords", [])]

            t, docs = time_call(backend.retriever.invoke, query)
            # backend.retriever returns a list of docs; limit top-k
            docs = docs[:k] if isinstance(docs, list) else []

            latencies.append(t)
            total += 1

            if not expected_keywords:
                continue

            blob = "\n".join([getattr(d, "page_content", "") for d in docs]).lower()
            if any(kw in blob for kw in expected_keywords):
                hits += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    recall_at_k = (hits / total) if total else 0.0

    return {
        "count": total,
        "avg_latency_s": round(avg_latency, 4),
        "p95_latency_s": round(p95_latency, 4),
        "recall_at_k": round(recall_at_k, 4),
    }


def generation_metrics(backend, queries: List[Dict[str, Any]], runs: int, num_context: int) -> Dict[str, Any]:
    if backend.llm is None or backend.rag_chain is None:
        return {"error": "llm or rag_chain not initialized"}

    latencies: List[float] = []
    char_counts: List[int] = []

    for _ in range(runs):
        for q in queries:
            question = q.get("query", "").strip()
            if not question:
                continue

            t, answer = time_call(
                backend.answer_question,
                question,
                "",  # no channel filter
                0.7,
                num_context,
                False,
                1.1,
            )

            latencies.append(t)
            char_counts.append(len(str(answer)))

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0.0
    chars_per_sec = (avg_chars / avg_latency) if avg_latency > 0 else 0.0

    return {
        "count": len(latencies),
        "avg_latency_s": round(avg_latency, 4),
        "p95_latency_s": round(p95_latency, 4),
        "avg_chars": int(avg_chars),
        "throughput_chars_per_sec": round(chars_per_sec, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for OpenVINO Messenger Assistant")
    parser.add_argument("--device", default="AUTO", help="OpenVINO device (CPU/GPU.x/NPU/AUTO)")
    parser.add_argument("--embedding-type", default="text_embedding_pipeline", choices=["text_embedding_pipeline", "openvino_genai", "legacy"], help="Embedding implementation")
    parser.add_argument("--queries", default=str(REPO_ROOT / "benchmarks" / "sample_queries.json"), help="Path to sample queries JSON")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics")
    parser.add_argument("--runs", type=int, default=3, help="Number of passes over query set")
    parser.add_argument("--num-context", type=int, default=5, help="Number of context docs for generation")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "results"), help="Directory to write results JSON")
    parser.add_argument("--llm-model", default=None, help="Override LLM model id")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model id")
    parser.add_argument("--rerank-model", default=None, help="Override reranker model id")
    args = parser.parse_args()

    backend = load_backend_module()

    # Model load timing
    load_stats = benchmark_model_load(
        backend,
        device=args.device,
        embedding_type=args.embedding_type,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
    )

    # Ensure RAG is initialized
    if not ensure_vector_store(backend):
        print("❌ Vector store not found. Run message processing first.")
        print("Hint: python examples/telegram_rag_gradio.py → Process Messages tab")
        sys.exit(1)

    backend.initialize_rag()

    # Load queries
    queries_path = Path(args.queries)
    queries = load_queries(queries_path)

    # Retrieval stats
    retr_stats = retrieval_metrics(backend, queries, args.k, args.runs)

    # Generation stats
    gen_stats = generation_metrics(backend, queries, args.runs, args.num_context)

    # Assemble result object
    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": args.device,
        "embedding_type": args.embedding_type,
        "k": args.k,
        "runs": args.runs,
        "num_context": args.num_context,
        "model_load": load_stats,
        "retrieval": retr_stats,
        "generation": gen_stats,
    }

    # Print concise summary
    print("\n=== Benchmark Summary ===")
    print(json.dumps(result, indent=2))

    # Persist to disk
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bench_{int(time.time())}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()


