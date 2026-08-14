"""RAG pipeline combining HDC retrieval with LLM generation.

This module provides the core Souprise functionality:
- HDC (Hyperdimensional Computing) retrieval via JuiceHDC
- LLM generation via Soup/MLX
- Combined RAG pipeline

All code is original and does not copy from JuiceHDC or Soup,
but integrates with them via their public APIs.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    title: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    text: str
    latency: float = 0.0  # seconds
    tokens_generated: int = 0


@dataclass
class RAGResult:
    """Combined result from RAG query."""
    question: str
    answer: str
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    retrieval_latency: float = 0.0  # seconds
    generation_latency: float = 0.0  # seconds
    total_latency: float = 0.0  # seconds


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    retrieval_k: int = 5  # Number of results to retrieve
    model_path: str = "./souprise_model"  # Path to fine-tuned model
    backend: str = "mlx"  # "mlx" for Apple Silicon, "torch" for CUDA
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class BaseRetriever:
    """Abstract base class for retrieval systems."""
    
    def index(self, entries: List[Dict[str, Any]]) -> None:
        """Index a list of entries for retrieval."""
        raise NotImplementedError
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for top-k results matching the query."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear the index."""
        raise NotImplementedError


class BaseGenerator:
    """Abstract base class for LLM generators."""
    
    def load(self, model_path: str) -> None:
        """Load a model from the given path."""
        raise NotImplementedError
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from a prompt."""
        raise NotImplementedError


class HDCRetriever(BaseRetriever):
    """HDC-based retriever using JuiceHDC.
    
    This is a wrapper around the JuiceHDC library (cortex-hdc).
    Requires: pip install git+https://github.com/mkupermann/JuiceHDC.git
    """
    
    def __init__(self):
        self._store = None
        self._engine = None
        self._initialized = False
    
    def index(self, entries: List[Dict[str, Any]]) -> None:
        """Index entries using JuiceHDC.
        
        Args:
            entries: List of dicts with 'id', 'text', and optionally 'metadata'.
        """
        try:
            from cortex.store import KnowledgeStore
            from cortex.encoder import CortexEncoder
            from cortex.engine import HDCEngine
        except ImportError as e:
            raise ImportError(
                "JuiceHDC not installed. Install with: "
                "pip install git+https://github.com/mkupermann/JuiceHDC.git"
            ) from e
        
        # Initialize store and encoder
        store = KnowledgeStore()
        encoder = CortexEncoder()
        
        # Index each entry
        for entry in entries:
            entry_id = entry.get("id", str(hash(entry["text"])))
            store.add(entry_id, entry["text"], metadata=entry.get("metadata", {}))
        
        # Initialize engine
        self._store = store
        self._engine = HDCEngine(store, encoder)
        self._initialized = True
        logger.info(f"Indexed {len(entries)} entries with HDC")
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for top-k results.
        
        Args:
            query: The search query.
            k: Number of results to return.
        
        Returns:
            List of RetrievalResult objects.
        """
        if not self._initialized:
            raise RuntimeError("Retriever not initialized. Call index() first.")
        
        start_time = time.time()
        results = self._engine.search(query, k=k)
        latency = time.time() - start_time
        
        logger.debug(f"Retrieved {len(results)} results in {latency*1000:.2f}ms")
        
        return [
            RetrievalResult(
                title=result.entry_id,
                content=result.text,
                score=result.score,
                metadata=result.metadata or {}
            )
            for result in results
        ]
    
    def clear(self) -> None:
        """Clear the index."""
        self._store = None
        self._engine = None
        self._initialized = False


class MLXGenerator(BaseGenerator):
    """LLM generator using MLX backend (Apple Silicon).
    
    Requires: pip install "soup-cli[mlx]"
    """
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def load(self, model_path: str) -> None:
        """Load a model using MLX.
        
        Args:
            model_path: Path to the model directory or HuggingFace ID.
        """
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "MLX not installed. Install with: pip install mlx mlx-lm"
            ) from e
        
        self._model, self._tokenizer = load(model_path)
        self._loaded = True
        logger.info(f"Loaded MLX model from {model_path}")
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters (max_tokens, temperature, etc.)
        
        Returns:
            GenerationResult with text and latency.
        """
        if not self._loaded:
            raise RuntimeError("Generator not loaded. Call load() first.")
        
        from mlx_lm import generate
        
        start_time = time.time()
        
        # Set defaults
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        latency = time.time() - start_time
        
        logger.debug(f"Generated {len(text.split())} tokens in {latency*1000:.2f}ms")
        
        return GenerationResult(
            text=text,
            latency=latency,
            tokens_generated=len(text.split())
        )


class TorchGenerator(BaseGenerator):
    """LLM generator using PyTorch backend (CUDA/CPU).
    
    Requires: pip install "soup-cli[train]"
    """
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def load(self, model_path: str) -> None:
        """Load a model using Transformers.
        
        Args:
            model_path: Path to the model directory or HuggingFace ID.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            ) from e
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._loaded = True
        logger.info(f"Loaded PyTorch model from {model_path}")
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
        
        Returns:
            GenerationResult with text and latency.
        """
        if not self._loaded:
            raise RuntimeError("Generator not loaded. Call load() first.")
        
        import torch
        
        start_time = time.time()
        
        # Set defaults
        max_new_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        latency = time.time() - start_time
        
        return GenerationResult(
            text=text,
            latency=latency,
            tokens_generated=len(text.split())
        )


class SoupriseRAG:
    """Main RAG pipeline combining retrieval and generation.
    
    This is the core class for Souprise, providing a simple interface
    for RAG operations with HDC retrieval and LLM generation.
    
    Example usage:
        rag = SoupriseRAG(config=RAGConfig(backend="mlx"))
        rag.index_from_entries(entries)
        result = rag.query("What are the open invoices?")
        print(result.answer)
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG pipeline.
        
        Args:
            config: Configuration for the pipeline. Uses defaults if None.
        """
        self.config = config or RAGConfig()
        self.retriever: Optional[BaseRetriever] = None
        self.generator: Optional[BaseGenerator] = None
        self._entries: List[Dict[str, Any]] = []
    
    def _get_retriever(self) -> BaseRetriever:
        """Get or create the retriever based on config."""
        if self.retriever is None:
            # Default to HDC retriever
            self.retriever = HDCRetriever()
        return self.retriever
    
    def _get_generator(self) -> BaseGenerator:
        """Get or create the generator based on config."""
        if self.generator is None:
            if self.config.backend == "mlx":
                self.generator = MLXGenerator()
            else:
                self.generator = TorchGenerator()
        return self.generator
    
    def index_from_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Index a list of entries for retrieval.
        
        Args:
            entries: List of dicts with 'id', 'text', and optionally 'metadata'.
        """
        self._entries = entries
        retriever = self._get_retriever()
        retriever.index(entries)
    
    def index_from_business_data(self, n: int = 10000, seed: int = 42) -> None:
        """Generate and index synthetic business data.
        
        Args:
            n: Number of entries to generate.
            seed: Random seed for reproducibility.
        """
        from souprise.data.generators.business import generate_business_data
        
        entries = generate_business_data(n=n, seed=seed)
        indexed_entries = [
            {
                "id": entry.title,
                "text": f"{entry.title}\n{entry.content}",
                "metadata": {"tags": entry.tags}
            }
            for entry in entries
        ]
        self.index_from_entries(indexed_entries)
        self._entries = indexed_entries
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the LLM model.
        
        Args:
            model_path: Path to the model. Uses config.model_path if None.
        """
        path = model_path or self.config.model_path
        generator = self._get_generator()
        generator.load(path)
    
    def query(self, question: str, k: Optional[int] = None) -> RAGResult:
        """Execute a RAG query: retrieval + generation.
        
        Args:
            question: The user's question.
            k: Number of retrieval results. Uses config.retrieval_k if None.
        
        Returns:
            RAGResult with answer, retrieval results, and latencies.
        """
        k = k or self.config.retrieval_k
        retriever = self._get_retriever()
        generator = self._get_generator()
        
        # Measure retrieval latency
        retrieval_start = time.time()
        retrieval_results = retriever.search(question, k=k)
        retrieval_latency = time.time() - retrieval_start
        
        # Build context
        context = "\n\n".join(
            f"--- {r.title} ---\n{r.content}"
            for r in retrieval_results
        )
        
        # Build prompt
        prompt = f"""CONTEXT:
{context}

QUESTION: {question}
ANSWER (based only on the context):"""
        
        # Measure generation latency
        generation_start = time.time()
        gen_result = generator.generate(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        generation_latency = time.time() - generation_start
        
        total_latency = retrieval_latency + generation_latency
        
        return RAGResult(
            question=question,
            answer=gen_result.text,
            retrieval_results=retrieval_results,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            total_latency=total_latency
        )
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat interface for multi-turn conversations.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Example: [{"role": "user", "content": "Hello"}]
        
        Returns:
            Assistant's response.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Get the latest user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg["content"]
                break
        
        if user_message is None:
            raise ValueError("No user message found")
        
        # Execute RAG query
        result = self.query(user_message)
        return result.answer
    
    def clear(self) -> None:
        """Clear the index and unload the model."""
        if self.retriever:
            self.retriever.clear()
        self.retriever = None
        self.generator = None
        self._entries = []


# Convenience function for quick use
def quickstart(
    n_data: int = 10000,
    model_path: str = "mlx-community/Phi-2-4bit",
    backend: str = "mlx",
) -> SoupriseRAG:
    """Quickstart function to create a pre-configured RAG pipeline.
    
    Generates synthetic business data and loads a model.
    
    Args:
        n_data: Number of synthetic data entries to generate.
        model_path: Path or HuggingFace ID for the LLM.
        backend: "mlx" for Apple Silicon, "torch" for CUDA.
    
    Returns:
        Configured SoupriseRAG instance ready for queries.
    
    Example:
        rag = quickstart(n_data=5000, model_path="mlx-community/Phi-2-4bit")
        result = rag.query("What are the open invoices?")
        print(result.answer)
    """
    config = RAGConfig(
        retrieval_k=5,
        model_path=model_path,
        backend=backend,
        max_tokens=256,
        temperature=0.7
    )
    
    rag = SoupriseRAG(config=config)
    rag.index_from_business_data(n=n_data, seed=42)
    rag.load_model()
    
    return rag
