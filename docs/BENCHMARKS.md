## Benchmarks

This document explains how we benchmark the OpenVINO Messenger Assistant and how to reproduce results.

### What we measure

- Model load time: end-to-end time to load Embedding, Reranker, and LLM
- Retrieval: latency and Recall@K (weak labels via keyword match)
- Generation: latency and throughput (chars/sec)

### Quick start

1) Prepare data and vector store

```bash
python examples/telegram_rag_gradio.py
# Download messages → Process Messages (creates data/telegram_vector_store)
```

2) Run benchmarks

```bash
python benchmarks/run_benchmarks.py \
  --device AUTO \
  --embedding-type text_embedding_pipeline \
  --queries benchmarks/sample_queries.json \
  --k 5 \
  --runs 3 \
  --output benchmarks/results
```

Output is printed and saved to `benchmarks/results/bench_<timestamp>.json`.

### Parameters

- `--device`: CPU, GPU.x, NPU, or AUTO
- `--embedding-type`: text_embedding_pipeline | openvino_genai | legacy
- `--queries`: JSON file with [ { query, keywords[] } ]
- `--k`: Top-K for Recall@K
- `--runs`: Passes over the query set
- `--num-context`: Docs provided to the LLM

### Methodology notes

- Recall@K uses naive keyword match over retrieved docs; it is a weak proxy for topical relevance
- Generation throughput is chars/sec based on final answer length
- Measurements include Python overhead, reflecting real user-perceived latency

### Tips

- For GPUs/NPUs, prefer INT4 LLM and INT8 embeddings/reranker
- Use `--device CPU` for most stable runs
- Increase `--runs` for more stable p95 latency


