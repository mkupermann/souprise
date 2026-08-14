<p align="center">
  <img src="docs/assets/hero.svg" alt="Souprise — offline RAG for business data" width="1000">
</p>

<div align="center">

[![CI](https://github.com/mkupermann/souprise/actions/workflows/ci.yml/badge.svg)](https://github.com/mkupermann/souprise/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-4c6ef5.svg)](#platform-support)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

> Ask questions about your invoices, orders, customers, and KPIs. Retrieval and generation run on your own hardware; the default configuration with a local `model_path` operates entirely offline and never transmits business data.

---

## What is Souprise?

Souprise is a Retrieval-Augmented Generation (RAG) pipeline for business data that must stay on-premises: it retrieves the records relevant to a question, and a language model on your own machine answers from those records. You get grounded answers over your ERP and CRM data without sending a byte of it to a cloud service.

Five terms are enough to read the rest of this README:

- **HDC (Hyperdimensional Computing)** — text represented as very long binary vectors (here: 10,000 bits); similar texts receive similar bit patterns, so search is inexpensive bitwise arithmetic instead of model inference.
- **MLX** — Apple's machine-learning framework for M-series processors; PyTorch covers every other machine.
- **Soup** — an open-source CLI for fine-tuning LLMs with LoRA, on MLX or PyTorch.
- **LoRA** — fine-tuning via small adapter matrices instead of the full model; `r` and `alpha` control adapter size and strength.
- **Alpaca format** — a simple JSONL layout for training examples: `instruction`, `input`, `output`.

Souprise replaces the usual embedding-model-plus-vector-database stack with HDC: no neural network runs at indexing or search time, results are fully deterministic, and the index occupies exactly **1,250 bytes per entry**. Every record becomes a 10,000-bit binary hypervector; similarity search is a vectorized XOR + popcount over the packed index, implemented in this repository with NumPy alone and verified by the test suite.

Retrieval feeds a **local LLM**, optionally fine-tuned on your domain with [Soup](https://github.com/MakazhanAlpamys/Soup). Like Soup, Souprise is infrastructure-neutral: the generation backend is auto-detected — MLX on Apple Silicon, PyTorch on NVIDIA CUDA, AMD ROCm, or plain CPU — and the same code and commands work unchanged on Linux, macOS, and Windows.

**What touches the network, exactly.** The built-in retriever operates only on local data; there is no network path in its code. `quickstart()` downloads a public base model once from Hugging Face on first run (public models download without an API key); pass a local `model_path` instead and there is no download and no connection at all, including air-gapped environments. Fine-tuning with Soup likewise runs on your own hardware. Business data is never transmitted anywhere at any time. These guarantees cover the built-in components — if you inject a custom `BaseRetriever` or `BaseGenerator`, its network behavior is yours to vouch for.

## Design Principles

| Principle | Implementation |
|---|---|
| Data sovereignty | All indexing, retrieval, and generation run locally; no telemetry, no API keys at runtime |
| Infrastructure-neutral | Linux, macOS, and Windows; backend auto-detection selects MLX or PyTorch per machine |
| Verifiable claims | The HDC retriever ships in this repository; storage and search behavior are covered by tests, and every `RAGResult` reports its own latencies |
| Lean core | Three dependencies (NumPy, Typer, Rich); backends and optional engines install via extras |
| Pluggable architecture | `BaseRetriever` and `BaseGenerator` interfaces; any implementation can be injected |

## Platform Support

<p align="center">
  <img src="docs/assets/platform_matrix.svg" alt="Platform support matrix" width="1000">
</p>

`RAGConfig(backend="auto")` — the default — resolves to MLX where available and PyTorch everywhere else, and picks a matching small base model unless one is specified. Any Hugging Face causal language model works as `model_path`: Mistral, Llama, Qwen, Phi, or your own fine-tuned checkpoint; the defaults are chosen only for small download size.

## Architecture

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Souprise architecture" width="1000">
</p>

### Query lifecycle

<p align="center">
  <img src="docs/assets/query_flow.svg" alt="Query lifecycle" width="1000">
</p>

Performance figures are deliberately not published in this README. Every `RAGResult` carries `retrieval_latency`, `generation_latency`, and `total_latency`, and `benchmarks/retrieval_bench.py` measures index build time and query latency on your hardware — so the numbers you plan with are your own.

## Retrieval at Scale

The built-in retriever is designed to work well past 10,000 records:

- **Exact, chunked search** — Hamming distances are computed in fixed-size chunks, so the temporary buffers are bounded by `chunk_rows` × 1,250 bytes (about 80 MB at the default of 65,536 rows) regardless of corpus size, while results remain exact. Chunked and unchunked search are verified equivalent by the test suite.
- **O(n) top-k selection** — `argpartition` instead of a full sort.
- **Hardware popcount** — uses `np.bitwise_count` on NumPy 2.x, with a lookup-table fallback on older versions.
- **Incremental indexing** — `add(entries)` appends new records without re-encoding the existing index.
- **Linear, predictable storage** — 1,250 bytes per entry: 100,000 records occupy 125 MB of index.

```bash
# Measure on your own machine — any corpus size
python benchmarks/retrieval_bench.py --n 100000 --queries 50
```

For corpora where an exact linear scan is no longer acceptable, the optional [JuiceHDC](https://github.com/mkupermann/JuiceHDC) engine can be swapped in (`pip install -e ".[juicehdc]"`, `RAGConfig(retriever="juicehdc")`), and an HD-NSW approximate index is on the roadmap for v1.0.

| | `SimpleHDCRetriever` (default) | `HDCRetriever` (JuiceHDC) |
|---|---|---|
| Installation | None — included, NumPy only | `pip install -e ".[juicehdc]"` |
| Encoding | Token bundling, deterministic BLAKE2b-seeded hypervectors | JuiceHDC `CortexEncoder` with typed tokens and character n-grams |
| Search | Exact XOR + popcount, chunked, O(n) top-k | Managed by JuiceHDC |
| Intended use | Default for any corpus; air-gapped installs | Advanced encoding, larger corpora |

Any object implementing `BaseRetriever` (`index`, `search`, `clear`) can be assigned to `rag.retriever` directly; the same applies to `BaseGenerator` for custom inference backends.

## Installation

```bash
git clone https://github.com/mkupermann/souprise.git
cd souprise

pip install -e .                # core: retrieval, data generators, CLI
```

| Target | Command |
|---|---|
| Apple Silicon inference | `pip install -e ".[mlx]"` |
| CUDA / ROCm / CPU inference | `pip install -e ".[torch]"` |
| Fine-tuning with Soup | `pip install -e ".[finetune]"` plus `soup-cli[mlx]` (Apple Silicon) or `soup-cli[train]` (CUDA/CPU) |
| JuiceHDC retrieval engine | `pip install -e ".[juicehdc]"` |
| Development (tests, lint) | `pip install -e ".[dev]"` |

## Quick Start

The smallest possible start — retrieval only, no model, no download:

```python
from souprise import SimpleHDCRetriever

retriever = SimpleHDCRetriever()
retriever.index([
    {"id": "INV-001", "text": "Invoice ACME Corp amount 12400 status overdue"},
    {"id": "INV-002", "text": "Invoice Globex amount 800 status paid"},
])
print(retriever.search("overdue invoice ACME", k=1)[0].title)   # INV-001
```

The full pipeline with a local LLM:

```python
from souprise import quickstart

# Indexes 10,000 synthetic business records and loads a model.
# Backend and default model are auto-detected for this machine; the base
# model is downloaded once from Hugging Face on first run — pass
# model_path="./your_local_model" for a fully offline start.
rag = quickstart(n_data=10_000)

result = rag.query("What are the open invoices for Customer_0123?")

print(result.answer)
print(f"retrieval : {result.retrieval_latency * 1000:7.2f} ms")
print(f"generation: {result.generation_latency * 1000:7.2f} ms")
print(f"total     : {result.total_latency * 1000:7.2f} ms")
```

Bring your own data with three fields per record:

```python
from souprise import SoupriseRAG, RAGConfig

rag = SoupriseRAG(RAGConfig(retriever="simple", model_path="./souprise_model"))
rag.index_from_entries([
    {"id": "INV-2025-001",
     "text": "Invoice ACME Corp\nAmount: $12,400\nStatus: overdue\nDue: 15 Mar 2025",
     "metadata": {"tags": ["invoice", "overdue"]}},
    # ... your ERP/CRM exports
])
rag.load_model()
```

### Command line

```bash
# Generate 10,000 synthetic business Q&A pairs (Alpaca format)
souprise train generate --output-path data/business_training.jsonl --n 10000

# Create a Soup fine-tuning configuration (backend auto-detected), then train
souprise train create-config
soup train --config soup_config.yaml

# Chat with your data — same command on macOS, Linux, or Windows
souprise chat --model ./souprise_model
```

## Synthetic Business Data

Souprise ships generators for six ERP/CRM entity types — seeded, reproducible, and containing no real customer information. They support fine-tuning experiments and retrieval testing before real data is connected.

| Entity | Generated fields |
|---|---|
| Invoice | customer, amount, status (paid / open / overdue / cancelled), region, department, due date |
| Order | customer, product, quantity, unit price, total, fulfillment status |
| Customer | annual revenue, segment (A / B / C / Enterprise), region, contact, open tickets |
| Product | stock, price, margin, 30-day sales, trend |
| KPI | department, metric, quarterly value versus target, status |
| Budget | department, allocated, spent, remaining, utilization |

Default category mix at `seed=42`: invoices 30 %, orders 25 %, customer profiles 20 %, products 10 %, KPIs 8 %, budgets 7 %. Each entry converts to three formats: retrieval (`to_retrieval_format`), plain dict (`to_dict`), or Alpaca (`to_alpaca_format`) for fine-tuning.

## Fine-Tuning Workflow

1. `souprise train generate` — synthetic Q&A pairs in Alpaca JSONL (or your own data in the same format).
2. `souprise train create-config` — writes `soup_config.yaml`; defaults: LoRA `r=16, alpha=32, dropout=0.05`, 4-bit quantization, 3 epochs, learning rate `2e-5`, 10 % validation split.
3. `soup train --config soup_config.yaml` — fine-tunes the base model locally, producing `./souprise_model`.
4. `souprise chat --model ./souprise_model` — RAG over your data with the tuned model, fully offline.

## CLI Reference

| Command | Purpose |
|---|---|
| `souprise chat` | Interactive RAG chat session |
| `souprise chat query "<question>"` | One-shot question |
| `souprise train generate` | Generate Alpaca-format training data |
| `souprise train create-config` | Write a Soup fine-tuning configuration |
| `souprise train all` | Data and configuration in one step |
| `souprise index` | Manage the HDC index |
| `souprise info` | Show installed backends and versions |
| `souprise version` | Show version |

## Configuration

```python
from souprise import RAGConfig

config = RAGConfig(
    retrieval_k=5,                  # top-k records fed into the prompt
    model_path="./souprise_model",  # local path or any Hugging Face causal LM, e.g.
                                    # "mistralai/Mistral-7B-Instruct-v0.3",
                                    # "meta-llama/Llama-3.2-1B-Instruct",
                                    # "Qwen/Qwen2.5-1.5B-Instruct"
    backend="auto",                 # "auto" | "mlx" (Apple Silicon) | "torch" (CUDA/ROCm/CPU)
    retriever="auto",               # "auto" | "simple" | "juicehdc"
    max_tokens=256,
    temperature=0.7,
)
```

The out-of-the-box defaults are chosen purely for download size, not vendor preference — override `model_path` with any vendor's model, local or from the Hub.

## Testing

The test suite runs against the core install alone — no model downloads, no optional dependencies — and covers the pipeline end to end:

- `tests/test_hdc_retriever.py` — the documented storage claim (10,000 bits = 1,250 bytes per entry), deterministic encoding, retrieval relevance, a 25,000-record scale test, chunked-search equivalence, and incremental `add()`.
- `tests/test_pipeline.py` — the complete query path (retrieval, context building, generation, latency instrumentation) with the built-in retriever and a stub generator, backend resolution, and custom-retriever injection.
- `tests/test_data_generators.py` — reproducibility and format guarantees of the data generators.

```bash
pip install -e ".[dev]"
pytest tests/
ruff check souprise/ tests/
```

Continuous integration runs the suite on Python 3.10 through 3.13 across Linux, macOS, and Windows on every push and pull request.

## Project Status

Alpha (v0.1.0). The pipeline, CLI, built-in retriever, and data generators are implemented and tested; persistence and connectors are in progress.

| Version | Scope |
|---|---|
| v0.1 (current) | Built-in HDC retrieval (tested to 25k+ records), auto-detected MLX/PyTorch generation, CLI, synthetic data generators, cross-platform CI, retrieval benchmark |
| v0.2 | Persistent HDC storage (SQLite), Postgres connector, REST API, multi-turn chat, standardized benchmark suite |
| v0.3 | SAP and DATEV integration, Excel/CSV importers, authentication and RBAC, multi-tenant |
| v1.0 | Production deployment options, observability, HD-NSW approximate index for very large corpora |

## Ecosystem

Souprise integrates with, but does not require, two sibling projects:

- [JuiceHDC](https://github.com/mkupermann/JuiceHDC) — Apache-2.0 — optional HDC retrieval engine (the built-in retriever covers the default path)
- [Soup](https://github.com/MakazhanAlpamys/Soup) — Apache-2.0 — LoRA fine-tuning for MLX and PyTorch
- [MLX](https://github.com/ml-explore/mlx) — Apache-2.0 — Apple Silicon ML framework

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). License: Apache-2.0 ([LICENSE](LICENSE), third-party attributions in [NOTICE](NOTICE)).

## Citation

```bibtex
@misc{souprise2026,
  author = {Michael Kupermann},
  title  = {Souprise: Offline RAG for Business Data},
  year   = {2026},
  url    = {https://github.com/mkupermann/souprise}
}
```

## Support

[Issues](https://github.com/mkupermann/souprise/issues) · [Discussions](https://github.com/mkupermann/souprise/discussions) · michael@kupermann.com
