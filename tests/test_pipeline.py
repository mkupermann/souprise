"""End-to-end tests for the RAG pipeline.

Runs the full query path — HDC retrieval, context building, generation,
latency instrumentation — using the built-in retriever and a fake generator,
so the pipeline is verified without downloading a model.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import pytest

from souprise import RAGConfig, SimpleHDCRetriever, SoupriseRAG
from souprise.core.pipeline import (
    DEFAULT_MODELS,
    BaseGenerator,
    GenerationResult,
    resolve_backend,
)


class EchoGenerator(BaseGenerator):
    """Fake generator that records the prompt and echoes a fixed answer."""

    def __init__(self):
        self.last_prompt = None
        self._loaded = False

    def load(self, model_path: str) -> None:
        self._loaded = True

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        self.last_prompt = prompt
        return GenerationResult(text="ECHO_ANSWER", latency=0.0, tokens_generated=1)


def build_pipeline():
    rag = SoupriseRAG(RAGConfig(retriever="simple", retrieval_k=3, answer_mode="generative"))
    rag.generator = EchoGenerator()
    rag.index_from_business_data(n=300, seed=42)
    return rag


class TestEndToEnd:
    def test_query_runs_full_path(self):
        rag = build_pipeline()
        result = rag.query("Which invoices are overdue?")

        assert result.answer == "ECHO_ANSWER"
        assert len(result.retrieval_results) == 3
        assert result.retrieval_latency > 0
        assert result.total_latency >= result.retrieval_latency

    def test_prompt_contains_retrieved_context_and_question(self):
        rag = build_pipeline()
        question = "What is the status of Customer_0001?"
        result = rag.query(question)

        prompt = rag.generator.last_prompt
        assert question in prompt
        for retrieved in result.retrieval_results:
            assert retrieved.content in prompt

    def test_retriever_selection_simple(self):
        rag = SoupriseRAG(RAGConfig(retriever="simple"))
        assert isinstance(rag._get_retriever(), SimpleHDCRetriever)

    def test_retriever_selection_auto_falls_back(self):
        """Without JuiceHDC installed, "auto" resolves to the built-in retriever."""
        try:
            import cortex  # noqa: F401
            pytest.skip("JuiceHDC installed; auto resolves to juicehdc here")
        except ImportError:
            pass
        rag = SoupriseRAG(RAGConfig(retriever="auto"))
        assert isinstance(rag._get_retriever(), SimpleHDCRetriever)

    def test_unknown_retriever_raises(self):
        rag = SoupriseRAG(RAGConfig(retriever="nonsense"))
        with pytest.raises(ValueError):
            rag._get_retriever()

    def test_custom_retriever_injection(self):
        """Any BaseRetriever implementation can be plugged in directly."""
        rag = SoupriseRAG(RAGConfig())
        custom = SimpleHDCRetriever()
        rag.retriever = custom
        rag.index_from_entries([{"id": "X", "text": "custom entry text"}])
        assert rag._get_retriever() is custom

    def test_backend_auto_resolves_to_available_backend(self):
        """"auto" resolves to mlx or torch depending on the platform."""
        backend = resolve_backend("auto")
        assert backend in ("mlx", "torch")
        assert backend in DEFAULT_MODELS

    def test_backend_explicit_passthrough(self):
        assert resolve_backend("mlx") == "mlx"
        assert resolve_backend("torch") == "torch"

    def test_backend_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_backend("tensorflow")

    def test_juicehdc_retriever_end_to_end(self):
        """Real JuiceHDC integration; runs only where cortex-hdc is installed."""
        pytest.importorskip("cortex")
        from souprise.core.pipeline import HDCRetriever

        retriever = HDCRetriever()
        retriever.index([
            {"id": "INV-001", "text": "Invoice ACME Corp amount 12400 status overdue",
             "metadata": {"tags": ["invoice"]}},
            {"id": "PRD-001", "text": "Product AB stock 500 trend rising",
             "metadata": {"tags": ["product"]}},
        ])
        results = retriever.search("overdue invoice ACME", k=2)
        assert results[0].title == "INV-001"
        assert 0.0 <= results[0].score <= 1.0
        retriever.clear()
        with pytest.raises(RuntimeError):
            retriever.search("anything")

    def test_chat_uses_latest_user_message(self):
        rag = build_pipeline()
        answer = rag.chat([
            {"role": "user", "content": "First question about budgets"},
            {"role": "assistant", "content": "Some answer"},
            {"role": "user", "content": "What are the open invoices?"},
        ])
        assert answer == "ECHO_ANSWER"
        assert "What are the open invoices?" in rag.generator.last_prompt

    def test_clear_resets_pipeline(self):
        rag = build_pipeline()
        rag.clear()
        assert rag.retriever is None
        assert rag.generator is None
