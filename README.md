<p align="center">
  <img src="docs/assets/hero.svg" alt="Souprise — private business AI, end to end" width="1000">
</p>

<div align="center">

[![CI](https://github.com/mkupermann/souprise/actions/workflows/ci.yml/badge.svg)](https://github.com/mkupermann/souprise/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-4c6ef5.svg)](#platform-support)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

> Generate training data, fine-tune a domain model, index your records, then ask questions about invoices, orders, customers and KPIs. It all runs on your hardware. Point `model_path` at a local folder and nothing ever touches the network.

<p align="center">
  <img src="docs/assets/demo.gif" alt="souprise demo, recorded live. System info, training data generation, Soup config, a 10,000-record retrieval benchmark, and a grounded answer from a local model." width="1000"><br>
  <sub>Recorded live on an Apple M-series laptop, nothing cut. The whole stack is loaded (MLX, JuiceHDC, Soup), 10,000 records index in 8 s with a 3.6 ms median query, and a local Qwen 0.5B answers a grounded question in 1.4 s. One machine's numbers, not a promise. Run <code>benchmarks/retrieval_bench.py</code> and get your own.</sub>
</p>

---

## What is Souprise?

Souprise is a toolkit for building business AI that stays in the building. Retrieval-augmented generation is the runtime path, but the repo covers the whole route from raw data to grounded answers. Four parts, each useful on its own.

| Component | What it does | Standalone use |
|---|---|---|
| Synthetic data generators | Seeded, reproducible ERP/CRM records and Alpaca-format Q&A pairs | Fine-tuning experiments and retrieval benchmarks before real data is connected |
| Fine-tuning workflow | Turns a base model into your domain model. Data generation, Soup/LoRA configuration, training orchestration | Produce a business-tuned LLM for any downstream use |
| HDC retrieval engine | Deterministic 10,000-bit hypervector search, NumPy only | A search library for your own applications, no LLM required |
| RAG runtime + CLI | Retrieves relevant records and has your local model answer from them, fully instrumented | Grounded question answering over ERP and CRM data |

You need five terms for the rest of this page.

- **HDC (Hyperdimensional Computing).** Text becomes a very long binary vector, 10,000 bits here. Similar texts get similar bit patterns, so search is cheap bitwise math instead of model inference.
- **MLX.** Apple's ML framework for M-series chips. PyTorch covers every other machine.
- **Soup.** An open-source CLI that fine-tunes LLMs with LoRA, on MLX or PyTorch.
- **LoRA.** Fine-tuning through small adapter matrices instead of the whole model. `r` and `alpha` set adapter size and strength.
- **Alpaca format.** A plain JSONL layout for training examples with `instruction`, `input` and `output` fields.

Most RAG stacks run an embedding model plus a vector database. Souprise skips both. No neural network runs at indexing or search time, results are deterministic, and the index occupies exactly **1,250 bytes per entry**. Every record becomes a 10,000-bit hypervector. Search is a vectorized XOR plus popcount over the packed index, written in plain NumPy in this repository and pinned down by the test suite.

The retrieved records go to a local LLM, which you can fine-tune on your domain with [Soup](https://github.com/MakazhanAlpamys/Soup). And like Soup, Souprise doesn't care what machine you own. It detects MLX on Apple Silicon and falls back to PyTorch on NVIDIA CUDA, AMD ROCm or plain CPU. Same code, same commands, on Linux, macOS and Windows.

**What touches the network, exactly.** The built-in retriever has no network path, full stop. `quickstart()` pulls a public base model from Hugging Face once, no API key needed. Point `model_path` at a local folder instead and there's no download and no connection at all, air gap included. Fine-tuning with Soup runs on your hardware too. Your business data is never sent anywhere, at any time. One caveat, and it's yours to own. If you inject a custom `BaseRetriever` or `BaseGenerator`, nobody can vouch for its network behavior but you.

## What Your Teams Can Ask

Souprise answers from your records, so the useful questions are the ones your teams already ask every day.

| Team | Questions that work against the shipped data model | Backed by |
|---|---|---|
| Sales | Which invoices for Customer_0123 are overdue? What did a customer order last quarter, and for how much? Which A-segment customers in the EU haven't been contacted since March? | Invoices, orders, customer profiles |
| Marketing | Which products are trending down despite high stock? Which segments carry the most annual revenue per region? How is the marketing budget tracking against its allocation? | Products, segments, KPIs, budgets |
| Service | Which enterprise customers sit on more than five open tickets? What's the fulfillment status of a customer's last order? Which departments miss their satisfaction targets? | Customer profiles, orders, KPIs |

Two honest notes on this. First, the shipped generators produce synthetic data, so you can try all of these questions in the demo before any real record is involved. Second, connecting real data works today through `souprise index build` with CSV, Excel, JSONL or a PostgreSQL query (see [Persistent Indexes and Connectors](#persistent-indexes-and-connectors)). Native SAP and DATEV integration is a roadmap item (v0.3), not a current feature.

## Design Principles

| Principle | Implementation |
|---|---|
| Data sovereignty | Indexing, retrieval and generation run locally. No telemetry, no API keys at runtime |
| Infrastructure-neutral | Linux, macOS and Windows. Backend auto-detection picks MLX or PyTorch per machine |
| Verifiable claims | The HDC retriever ships in this repository, its storage and search behavior are covered by tests, and every `RAGResult` reports its own latencies |
| Lean core | Three dependencies (NumPy, Typer, Rich). Backends and optional engines install via extras |
| Pluggable architecture | `BaseRetriever` and `BaseGenerator` interfaces. Any implementation can be injected |

## Platform Support

<p align="center">
  <img src="docs/assets/platform_matrix.svg" alt="Platform support matrix" width="1000">
</p>

`backend="auto"` is the default and does what you'd expect. MLX where it exists, PyTorch everywhere else, plus a matching small base model unless you name one. Any Hugging Face causal LM works as `model_path`. Mistral, Llama, Qwen, Phi, or your own fine-tuned checkpoint. The shipped defaults were picked for download size, nothing else.

## Architecture

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Souprise architecture" width="1000">
</p>

### Query lifecycle

<p align="center">
  <img src="docs/assets/query_flow.svg" alt="Query lifecycle" width="1000">
</p>

You won't find performance claims in this README, and that's on purpose. Every `RAGResult` carries its own retrieval, generation and total latency, and `benchmarks/retrieval_bench.py` measures index build and query time where it counts, on your machine. Plan with your numbers, not ours.

## Retrieval at Scale

The built-in retriever doesn't stop at 10,000 records.

- **Exact, chunked search.** Hamming distances are computed in fixed-size blocks, so temporary memory stays near 80 MB at the default `chunk_rows` of 65,536, no matter how big the corpus gets. The tests prove chunked and unchunked search return identical results.
- **Top-k by argpartition.** O(n) selection, no full sort.
- **Hardware popcount** on NumPy 2.x, lookup-table fallback on older versions.
- **Incremental indexing.** `add(entries)` appends new records without re-encoding what's already there.
- **Linear storage.** 1,250 bytes per entry, so 100,000 records need 125 MB of index.

<p align="center">
  <img src="docs/assets/scale.gif" alt="Retrieval benchmark at 10,000 and then 1,000,000 records on the same machine" width="1000"><br>
  <sub>Same laptop, same exact search, two corpus sizes. 10,000 records index in 8 s and answer in 3.8 ms. One million records build 1.25 GB of index in just under 8 minutes and answer in 371 ms median, with 20 of 20 self-retrieval hits. The recording pauses during the long build, nothing else is cut. One machine's numbers, not a promise.</sub>
</p>

```bash
# Measure on your own machine, any corpus size
python benchmarks/retrieval_bench.py --n 100000 --queries 50
```

When an exact linear scan stops being fast enough for you, swap in the optional [JuiceHDC](https://github.com/mkupermann/JuiceHDC) engine with `pip install -e ".[juicehdc]"` and `RAGConfig(retriever="juicehdc")`. An HD-NSW approximate index is planned for v1.0.

| | `SimpleHDCRetriever` (default) | `HDCRetriever` (JuiceHDC) |
|---|---|---|
| Installation | None. Included, NumPy only | `pip install -e ".[juicehdc]"` |
| Encoding | Token bundling, deterministic BLAKE2b-seeded hypervectors | JuiceHDC `CortexEncoder` with typed tokens and character n-grams |
| Search | Exact XOR + popcount, chunked, O(n) top-k | Managed by JuiceHDC |
| Intended use | Default for any corpus, air-gapped installs | Advanced encoding, larger corpora |

Anything that implements `BaseRetriever` (`index`, `search`, `clear`) can be assigned to `rag.retriever` directly. Same for `BaseGenerator` and custom inference backends.

## Persistent Indexes and Connectors

Encoding happens once, not on every start. `souprise index build` writes a self-contained SQLite file with the entries and the packed hypervector matrix, and loading it back takes seconds because nothing gets re-encoded. Your data comes in through four doors.

```bash
# From a CSV export (delimiter is sniffed, works with ; and tab too)
souprise index build --from-csv invoices.csv --id-column invoice_id --tag-columns status

# From an Excel workbook
souprise index build --from-xlsx export.xlsx --sheet Invoices --id-column invoice_id

# Straight from PostgreSQL
souprise index build --from-postgres postgresql://user@localhost/erp \
    --query "SELECT id, customer, amount, status FROM invoices" --id-column id

# Inspect and search the index without loading any model
souprise index info --path souprise_index.db
souprise index query "overdue invoices ACME" --path souprise_index.db

# Chat against the persistent index
souprise chat --index souprise_index.db --model ./souprise_model
```

The same works in code through `souprise.data.importers` (`load_csv`, `load_excel`, `load_jsonl`, `load_postgres`) and `SimpleHDCRetriever.save(path)` / `SimpleHDCRetriever.load(path)`. Rows are rendered as plain "Column: value" text, the same shape the synthetic generators produce, so a model fine-tuned on the synthetic data feels at home with your real records. The Postgres path is covered by a round-trip test against a real database in CI.

### Daily updates, no retraining

New data does not mean new training. The facts live in the index, not in the model's weights, so appending today's records makes them answerable immediately. `souprise index add` loads the existing index, encodes only the delta (around 2,000 entries per second), and saves. A thousand new invoices land in about a second.

```bash
# Nightly job: append today's records to the standing index
souprise index add --path souprise_index.db \
    --from-postgres postgresql://user@localhost/erp \
    --query "SELECT id, customer, amount, status FROM invoices WHERE created_at::date = CURRENT_DATE" \
    --id-column id

# Or from a daily export
souprise index add --path souprise_index.db --from-csv todays_invoices.csv --id-column invoice_id
```

The fine-tuned model only needs retraining when the shape of your data changes, new record types or new fields, not when new rows arrive. Day-to-day freshness is an index append, and the append is covered by the test suite.

<p align="center">
  <img src="docs/assets/daily.gif" alt="Daily update proof: build a 10,000-record index, append three new invoices from a CSV in a quarter second, search finds them immediately, and the local model answers with the correct amount" width="1000"><br>
  <sub>Recorded live. A 10,000-record index gets today's three invoices appended in 0.24 s, the new record is the top hit immediately, and the local model answers with the correct status and amount. No training happened anywhere in this clip.</sub>
</p>

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
| Excel importer | `pip install -e ".[excel]"` |
| PostgreSQL connector | `pip install -e ".[postgres]"` |
| Development (tests, lint) | `pip install -e ".[dev]"` |

## Quick Start

The smallest possible start. Retrieval only, no model, no download.

```python
from souprise import SimpleHDCRetriever

retriever = SimpleHDCRetriever()
retriever.index([
    {"id": "INV-001", "text": "Invoice ACME Corp amount 12400 status overdue"},
    {"id": "INV-002", "text": "Invoice Globex amount 800 status paid"},
])
print(retriever.search("overdue invoice ACME", k=1)[0].title)   # INV-001
```

The full pipeline with a local LLM.

```python
from souprise import quickstart

# Indexes 10,000 synthetic business records and loads a model.
# Backend and default model are auto-detected for this machine. The base
# model is downloaded once from Hugging Face on first run. Pass
# model_path="./your_local_model" for a fully offline start.
rag = quickstart(n_data=10_000)

result = rag.query("What are the open invoices for Customer_0123?")

print(result.answer)
print(f"retrieval : {result.retrieval_latency * 1000:7.2f} ms")
print(f"generation: {result.generation_latency * 1000:7.2f} ms")
print(f"total     : {result.total_latency * 1000:7.2f} ms")
```

Bring your own data. Three fields per record.

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

# Chat with your data. Same command on macOS, Linux or Windows
souprise chat --model ./souprise_model
```

<p align="center">
  <img src="docs/assets/chat.gif" alt="Interactive souprise chat session answering two questions about overdue invoices and a customer profile" width="1000"><br>
  <sub>An interactive session over 2,000 records, recorded live. Retrieval takes about 35 ms per question, the local 0.5B model answers in about 1.3 s, and every answer names the records it came from. Ctrl+C ends it. No session data leaves the machine.</sub>
</p>

## Synthetic Business Data

Six ERP/CRM entity types ship as generators. Seeded, reproducible, and containing no real customer information. Use them to test fine-tuning and retrieval before any real data gets involved.

| Entity | Generated fields |
|---|---|
| Invoice | customer, amount, status (paid / open / overdue / cancelled), region, department, due date |
| Order | customer, product, quantity, unit price, total, fulfillment status |
| Customer | annual revenue, segment (A / B / C / Enterprise), region, contact, open tickets |
| Product | stock, price, margin, 30-day sales, trend |
| KPI | department, metric, quarterly value versus target, status |
| Budget | department, allocated, spent, remaining, utilization |

The default mix at `seed=42` is 30 % invoices, 25 % orders, 20 % customer profiles, 10 % products, 8 % KPIs and 7 % budgets. Each entry converts to three formats. Retrieval (`to_retrieval_format`), plain dict (`to_dict`), or Alpaca (`to_alpaca_format`) for fine-tuning.

## Fine-Tuning Workflow

1. `souprise train generate` writes synthetic Q&A pairs as Alpaca JSONL. Your own data works in the same format.
2. `souprise train create-config` writes `soup_config.yaml`. Defaults are LoRA `r=16, alpha=32, dropout=0.05`, 4-bit quantization, 3 epochs, learning rate `2e-5`, 10 % validation split.
3. `soup train --config soup_config.yaml` fine-tunes the base model locally and produces `./souprise_model`.
4. `souprise chat --model ./souprise_model` runs RAG over your data with the tuned model, fully offline.

## CLI Reference

| Command | Purpose |
|---|---|
| `souprise chat` | Interactive RAG chat session |
| `souprise chat query "<question>"` | One-shot question |
| `souprise train generate` | Generate Alpaca-format training data |
| `souprise train create-config` | Write a Soup fine-tuning configuration |
| `souprise train all` | Data and configuration in one step |
| `souprise index build` | Build a persistent index from CSV, Excel, JSONL, PostgreSQL, or synthetic data |
| `souprise index add` | Append new records to an existing index, encoding only the delta |
| `souprise index info` | Show statistics of a persistent index |
| `souprise index query "<question>"` | Search a persistent index without loading an LLM |
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

## Testing

The suite runs against the core install alone. No model downloads, no optional extras. It still covers the pipeline end to end.

- `tests/test_hdc_retriever.py` checks the storage claim (10,000 bits are 1,250 bytes per entry), deterministic encoding, retrieval relevance, a 25,000-record scale test, chunked-search equivalence and incremental `add()`.
- `tests/test_pipeline.py` runs the whole query path with the built-in retriever and a stub generator, plus backend resolution and custom-retriever injection.
- `tests/test_data_generators.py` covers reproducibility and format guarantees.
- `tests/test_persistence_importers.py` covers the save/load round trip with identical scores, the CSV, Excel and JSONL importers end to end, and a real-PostgreSQL round trip (runs wherever `SOUPRISE_TEST_PG_DSN` points at a database; CI provides one).

```bash
pip install -e ".[dev]"
pytest tests/
ruff check souprise/ tests/
```

CI runs the suite on Python 3.10 through 3.13 across Linux, macOS and Windows, on every push and pull request.

## Project Status

Alpha, v0.2.0. Pipeline, CLI, built-in retriever, data generators, persistent indexes and the CSV/Excel/JSONL/Postgres doors are built and tested.

| Version | Scope |
|---|---|
| v0.1 | Built-in HDC retrieval (tested to 25k+ records), auto-detected MLX/PyTorch generation, CLI, synthetic data generators, cross-platform CI, retrieval benchmark |
| v0.2 (current) | Persistent SQLite indexes (`souprise index build/info/query`, `--index` in chat), CSV/Excel/JSONL importers, PostgreSQL connector with a real-database CI test |
| v0.2.x | REST API, multi-turn chat, standardized benchmark suite |
| v0.3 | SAP and DATEV integration, authentication and RBAC, multi-tenant |
| v1.0 | Production deployment options, observability, HD-NSW approximate index for very large corpora |

## Ecosystem

Souprise plays well with two sibling projects but needs neither.

- [JuiceHDC](https://github.com/mkupermann/JuiceHDC), Apache-2.0. Optional HDC retrieval engine. The built-in retriever covers the default path.
- [Soup](https://github.com/MakazhanAlpamys/Soup), Apache-2.0. LoRA fine-tuning for MLX and PyTorch.
- [MLX](https://github.com/ml-explore/mlx), Apache-2.0. Apple Silicon ML framework.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Licensed Apache-2.0 ([LICENSE](LICENSE), third-party attributions in [NOTICE](NOTICE)).

## Citation

```bibtex
@misc{souprise2026,
  author = {Michael Kupermann},
  title  = {Souprise: Private Business AI Toolkit},
  year   = {2026},
  url    = {https://github.com/mkupermann/souprise}
}
```

## Support

[Issues](https://github.com/mkupermann/souprise/issues) · [Discussions](https://github.com/mkupermann/souprise/discussions) · michael@kupermann.com
