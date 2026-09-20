import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    Result: 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Base Interface ────────────────────────────────────────────────────────────
class BaseEmbeddingService(ABC):
    """Every embedding provider implements this contract."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string → returns a vector (list of floats)."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings at once (batched — faster & cheaper)."""
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Document]) -> list[tuple[Document, list[float]]]:
        """
        Embed a list of Document chunks.
        Returns list of (Document, vector) pairs.
        """
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embed_documents(texts)
        return list(zip(chunks, vectors))

    def find_most_similar(
        self,
        query: str,
        chunks: list[Document],
    ) -> tuple[Document, float]:
        """
        Return the single most similar chunk to the query.
        Uses cosine similarity to rank all chunks.
        """
        query_vec = self.embed_text(query)
        chunk_embeddings = self.embed_chunks(chunks)
        best_doc, best_score = max(
            [(doc, cosine_similarity(query_vec, vec)) for doc, vec in chunk_embeddings],
            key=lambda x: x[1]
        )
        return best_doc, best_score


# ── HuggingFace Embeddings (FREE) ─────────────────────────────────────────────
class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """
    Uses sentence-transformers locally — completely free, no API calls.
    Model: all-MiniLM-L6-v2 → produces 384-dimension vectors.
    First run downloads ~90MB model and caches it.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading HuggingFace model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        # verbose=False → no progress bar for single text
        vector = self.model.encode(
            text.replace("\n", " "),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t.replace("\n", " ") for t in texts]
        # show_progress_bar=False → removes noisy "Batches: 100%|███" lines
        vectors = self.model.encode(
            cleaned,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


# ── OpenAI Embeddings ─────────────────────────────────────────────────────────
class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    Uses OpenAI text-embedding-3-small — best quality/cost ratio.
    Requires OPENAI_API_KEY in .env
    1536 dimensions (vs HuggingFace's 384)
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        logger.info(f"OpenAI embedding model: {self.model}")

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text.replace("\n", " "),
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # OpenAI supports batching — all texts in ONE API call
        cleaned = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned,
        )
        return [item.embedding for item in response.data]


# ── Factory ───────────────────────────────────────────────────────────────────
def get_embedding_service(provider: str = "huggingface") -> BaseEmbeddingService:
    """
    Factory — returns the right embedding service.
    provider: "openai" | "huggingface"
    """
    if provider == "openai":
        return OpenAIEmbeddingService()
    elif provider == "huggingface":
        return HuggingFaceEmbeddingService()
    raise ValueError(f"Unknown embedding provider: '{provider}'. Use 'openai' or 'huggingface'.")


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # WARNING only — hides httpx noise
    import sys
    sys.path.insert(0, os.getcwd())

    # Load service ONCE — reuse in all tests (avoid reloading model 5 times)
    print("\n" + "="*60)
    print("Loading HuggingFace model (cached after first run)...")
    print("="*60)
    service = HuggingFaceEmbeddingService()
    print("Model ready.\n")

    # ── TEST 1: Single text embedding ────────────────────────────────────────
    print("="*60)
    print("TEST 1: Single text embedding")
    print("="*60)
    vector = service.embed_text("What is your refund policy?")
    print(f"Input:             'What is your refund policy?'")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values:    {[round(v, 4) for v in vector[:5]]}")
    print(f"Value range:       min={round(min(vector),4)}, max={round(max(vector),4)}")

    # ── TEST 2: Cosine Similarity ────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Cosine Similarity — same meaning, different words")
    print("="*60)

    query = "What is your refund policy?"
    query_vec = service.embed_text(query)
    print(f"Query: '{query}'\n")

    comparisons = [
        ("SAME MEANING",    "How do I get my money back?"),
        ("RELATED",         "Can I cancel my subscription?"),
        ("UNRELATED",       "What is the weather in Mumbai today?"),
    ]

    for label, sentence in comparisons:
        vec = service.embed_text(sentence)
        score = cosine_similarity(query_vec, vec)
        filled = int(score * 20)
        bar = "█" * max(filled, 0)
        print(f"  [{label}]")
        print(f"  Text:  '{sentence}'")
        print(f"  Score: {score:.4f}  {bar}")
        print()

    # ── TEST 3: Embed Day 9 chunks (FIXED — CONVERSATIONAL = 3 clean chunks) ─
    print("="*60)
    print("TEST 3: Embed Day 9 chunks (3 clean chunks this time)")
    print("="*60)

    from app.knowledge_base.splitters.text_splitter import (
        DocumentChunkingService, ContentType
    )

    sample_text = """
Refund Policy
We offer a 30-day money-back guarantee on all plans.
To request a refund, contact support@crmplatform.com.
Refunds are processed within 5-7 business days.

Pricing Plans
Starter Plan: Rs.999/month for up to 3 users and 500 leads.
Growth Plan: Rs.2,999/month for up to 15 users with AI features.
Enterprise Plan: Custom pricing with dedicated support.

Support Hours
Available Monday to Saturday, 9am to 7pm IST.
Average response time is under 2 hours during business hours.
"""

    # FIX: CONVERSATIONAL (300 chars) → 3 clean separate chunks
    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(sample_text, source="company_faq")
    print(f"Total chunks created: {len(chunks)}")
    print()

    chunk_embeddings = service.embed_chunks(chunks)

    for i, (doc, vec) in enumerate(chunk_embeddings):
        print(f"Chunk {i}:")
        print(f"  Content:    '{doc.page_content[:70].strip()}...'")
        print(f"  Dimensions: {len(vec)}")
        print(f"  First 3:    {[round(v, 4) for v in vec[:3]]}")
        print()

    # ── TEST 4: Semantic Search ───────────────────────────────────────────────
    print("="*60)
    print("TEST 4: Semantic Search — query vs all chunks")
    print("="*60)

    queries = [
        "How do I get a refund?",
        "What is the price of the starter plan?",
        "When is support available?",
    ]

    for query in queries:
        query_vec = service.embed_text(query)
        results = []
        for doc, vec in chunk_embeddings:
            score = cosine_similarity(query_vec, vec)
            results.append((score, doc.page_content))
        results.sort(key=lambda x: x[0], reverse=True)

        best_score, best_content = results[0]
        print(f"Query:    '{query}'")
        print(f"Match:    '{best_content[:70].strip()}...'")
        print(f"Score:    {best_score:.4f}")
        print()

    # ── TEST 5: find_most_similar() ───────────────────────────────────────────
    print("="*60)
    print("TEST 5: find_most_similar() — Mini Assignment")
    print("="*60)

    test_queries = [
        "How do I get a refund?",
        "What does the Growth plan cost?",
        "What time does support close?",
    ]

    for query in test_queries:
        best_doc, best_score = service.find_most_similar(query, chunks)
        print(f"Query:   '{query}'")
        print(f"Score:   {best_score:.4f}")
        print(f"Content: '{best_doc.page_content[:80].strip()}...'")
        print()

    # ── TEST 6: Provider comparison ───────────────────────────────────────────
    print("="*60)
    print("TEST 6: HuggingFace vs OpenAI — dimensions comparison")
    print("="*60)
    print(f"HuggingFace (all-MiniLM-L6-v2): {len(vector)} dimensions  — FREE, local")
    print(f"OpenAI (text-embedding-3-small): 1536 dimensions — $0.02/million tokens")
    print(f"OpenAI (text-embedding-3-large): 3072 dimensions — $0.13/million tokens")
    print()
    print("Rule: Use HuggingFace for dev/testing, OpenAI for production.")
    print("Rule: ALWAYS use the same model for ingestion AND query.")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE ✅")
    print("="*60)
    print("Day 10 DONE — Embedding Service fully working!")
    print("Fix applied: CONVERSATIONAL chunking → 3 clean separate chunks")
    print("Next → Day 11: ChromaDB — store these vectors permanently")
    print("="*60)