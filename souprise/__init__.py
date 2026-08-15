"""Souprise: private business AI toolkit.

Covers the full path from data to grounded answers, entirely on-premises:
synthetic ERP/CRM data generation, a Soup/LoRA fine-tuning workflow,
HDC (Hyperdimensional Computing) retrieval, and an offline RAG runtime.

Main components:
- souprise.core: RAG pipeline (retrieval + generation)
- souprise.data: Synthetic data generators
- souprise.cli: Command-line interface

Example usage:
    from souprise import quickstart
    rag = quickstart(n_data=10000, model_path="mlx-community/Phi-2-4bit")
    result = rag.query("What are the open invoices?")
    print(result.answer)

CLI usage:
    souprise chat --model mlx-community/Phi-2-4bit
    souprise train generate --n 5000
    souprise index info

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

__version__ = "0.2.0"

from souprise.core.hdc import SimpleHDCRetriever
from souprise.core.pipeline import (
    GenerationResult,
    RAGConfig,
    RAGResult,
    RetrievalResult,
    SoupriseRAG,
    quickstart,
)

__all__ = [
    "__version__",
    "SoupriseRAG",
    "RAGConfig",
    "RetrievalResult",
    "GenerationResult",
    "RAGResult",
    "SimpleHDCRetriever",
    "quickstart",
]
