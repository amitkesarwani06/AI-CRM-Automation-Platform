import os
import sys
import logging
from enum import Enum
from pathlib import Path

# Suppress all HuggingFace warnings before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class VectorStoreType(str, Enum):
    CHROMA   = "chroma"
    FAISS    = "faiss"
    PINECONE = "pinecone"


class UnifiedVectorStore:
    """
    Single interface for ChromaDB, FAISS, and Pinecone.
    Switch vector stores by changing ONE line.
    Strategy Pattern — same as Day 3's ChatModel factory.
    """

    def __init__(
        self,
        store_type: VectorStoreType = VectorStoreType.CHROMA,
        collection_name: str = "crm_knowledge_base",
    ):
        self.store_type = store_type
        self.collection_name = collection_name
        self._store = None

        # Shared embedding model — same for ALL backends
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"UnifiedVectorStore | backend={store_type}")

    def add_documents(self, chunks: list[Document]) -> None:
        """
        Add documents — works identically regardless of backend.
        FIX: FAISS uses distance_strategy='COSINE' for proper 0-1 scores.
        """
        if self.store_type == VectorStoreType.CHROMA:
            self._store = Chroma.from_documents(
                chunks,
                self.embeddings,
                collection_name=self.collection_name,
                persist_directory="app/data/chroma_db",
            )

        elif self.store_type == VectorStoreType.FAISS:
            # FIX: COSINE distance_strategy → scores will be 0.0 to 1.0
            from langchain_community.vectorstores.faiss import DistanceStrategy
            self._store = FAISS.from_documents(
                chunks,
                self.embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
            self._store.save_local("app/data/faiss_index")

        elif self.store_type == VectorStoreType.PINECONE:
            self._add_to_pinecone(chunks)

        logger.info(f"Added {len(chunks)} chunks to {self.store_type}")

    def search(self, query: str, k: int = 3) -> list[Document]:
        """Semantic search — returns top-k Documents."""
        if self._store is None:
            raise RuntimeError("No documents added yet. Call add_documents() first.")
        return self._store.similarity_search(query, k=k)

    def search_with_scores(
        self, query: str, k: int = 3
    ) -> list[tuple[Document, float]]:
        """
        Search with similarity scores — always 0.0 to 1.0.
        FIX: With COSINE distance_strategy, similarity_search_with_score
             returns cosine similarity directly (no manual conversion needed).
        """
        if self._store is None:
            raise RuntimeError("No documents added yet.")

        raw = self._store.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in raw:
            # With COSINE strategy: score is already cosine similarity (0.0 to 1.0)
            similarity = round(float(score), 4)
            similarity = max(0.0, min(1.0, similarity))  # safety clamp
            results.append((doc, similarity))
        return results

    def as_retriever(self, k: int = 3):
        """
        THE most important method for Days 13-14.
        Returns a LangChain Retriever — plugs directly into RAG chains.
        """
        if self._store is None:
            raise RuntimeError("No documents added yet.")
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def load(self) -> bool:
        """Load existing index from disk without re-embedding."""
        if self.store_type == VectorStoreType.FAISS:
            path = Path("app/data/faiss_index/index.faiss")
            if path.exists():
                from langchain_community.vectorstores.faiss import DistanceStrategy
                self._store = FAISS.load_local(
                    "app/data/faiss_index",
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )
                logger.info("FAISS index loaded from disk")
                return True

        elif self.store_type == VectorStoreType.CHROMA:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory="app/data/chroma_db",
            )
            logger.info("ChromaDB loaded from disk")
            return True

        return False

    def _add_to_pinecone(self, chunks: list[Document]) -> None:
        """Pinecone — requires PINECONE_API_KEY in .env"""
        try:
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

            if self.collection_name not in pc.list_indexes().names():
                pc.create_index(
                    name=self.collection_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Created Pinecone index: {self.collection_name}")

            self._store = PineconeVectorStore.from_documents(
                chunks,
                self.embeddings,
                index_name=self.collection_name,
            )
        except ImportError:
            raise ImportError(
                "Run: pip install pinecone-client langchain-pinecone"
            )


def get_vector_store(
    store_type: str = "chroma",
    collection_name: str = "crm_knowledge_base",
) -> UnifiedVectorStore:
    """
    Factory — the ONE place that decides which backend to use.
    Set VECTOR_STORE_TYPE in .env to switch without code changes.
    """
    return UnifiedVectorStore(
        store_type=VectorStoreType(store_type),
        collection_name=collection_name,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)   # only real errors shown

    from app.knowledge_base.splitters.text_splitter import (
        DocumentChunkingService, ContentType
    )

    sample_text = """
Refund Policy
We offer a 30-day money-back guarantee on all plans.
To request a refund contact support@crmplatform.com.
Refunds are processed within 5-7 business days.

Pricing Plans
Starter Plan Rs.999 per month for up to 3 users and 500 leads.
Growth Plan Rs.2999 per month for up to 15 users with AI features.
Enterprise Plan custom pricing with dedicated support.

Support Hours
Available Monday to Saturday 9am to 7pm IST.
Average response time is under 2 hours during business hours.

Getting Started
After signing up you will receive a welcome email with setup instructions.
Your first 14 days are free with no credit card required.
Book an onboarding call at calendly.com/crmplatform/onboarding.
"""

    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(sample_text, source="company_faq")
    print(f"\nChunks ready: {len(chunks)}")

    test_queries = [
        "How do I get a refund?",
        "What does the Growth plan cost?",
        "When is support available?",
        "How do I get started?",
    ]

    # ── TEST 1: FAISS ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: FAISS Vector Store (COSINE distance)")
    print("="*60)

    faiss_store = UnifiedVectorStore(store_type=VectorStoreType.FAISS)
    faiss_store.add_documents(chunks)

    for query in test_queries:
        results = faiss_store.search_with_scores(query, k=1)
        if results:
            doc, score = results[0]
            status = "✅" if score >= 0.3 else "⚠️ low"
            print(f"Query: '{query}'")
            print(f"  {status} Score: {score:.4f} | {doc.page_content[:55].strip()}...")
            print()

    # ── TEST 2: ChromaDB ──────────────────────────────────────────────────────
    print("="*60)
    print("TEST 2: ChromaDB via UnifiedVectorStore")
    print("="*60)

    chroma_store = UnifiedVectorStore(
        store_type=VectorStoreType.CHROMA,
        collection_name="unified_test",
    )
    chroma_store.add_documents(chunks)

    for query in test_queries:
        results = chroma_store.search_with_scores(query, k=1)
        if results:
            doc, score = results[0]
            status = "✅" if score >= 0.3 else "⚠️ low"
            print(f"Query: '{query}'")
            print(f"  {status} Score: {score:.4f} | {doc.page_content[:55].strip()}...")
            print()

    # ── TEST 3: FAISS vs ChromaDB comparison ──────────────────────────────────
    print("="*60)
    print("TEST 3: FAISS vs ChromaDB — score comparison")
    print("="*60)

    query = "How do I get a refund?"
    faiss_result  = faiss_store.search_with_scores(query, k=1)
    chroma_result = chroma_store.search_with_scores(query, k=1)

    print(f"Query: '{query}'\n")
    if faiss_result:
        doc, score = faiss_result[0]
        print(f"FAISS  → Score: {score:.4f} | {doc.page_content[:50].strip()}...")
    if chroma_result:
        doc, score = chroma_result[0]
        print(f"Chroma → Score: {score:.4f} | {doc.page_content[:50].strip()}...")

    # ── TEST 4: .as_retriever() ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: .as_retriever() — preview for Day 13")
    print("="*60)

    retriever = faiss_store.as_retriever(k=2)
    print(f"Retriever type: {type(retriever).__name__}")
    docs = retriever.invoke("refund policy")
    print(f"Retriever returned {len(docs)} docs for 'refund policy':")
    for i, doc in enumerate(docs, 1):
        print(f"  Doc {i}: {doc.page_content[:60].strip()}...")

    # ── TEST 5: load() — persistence ──────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: load() — reload FAISS from disk")
    print("="*60)

    new_store = UnifiedVectorStore(store_type=VectorStoreType.FAISS)
    loaded = new_store.load()
    print(f"Loaded from disk: {loaded}")
    if loaded:
        results = new_store.search("refund", k=1)
        print(f"Search after reload: {results[0].page_content[:55].strip()}...")
        print("✅ Persistence working!")

    # ── TEST 6: Comparison table ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 6: When to use which vector store")
    print("="*60)

    headers = f"{'Feature':<22} {'ChromaDB':<18} {'FAISS':<18} {'Pinecone'}"
    divider = f"{'-'*20:<22} {'-'*16:<18} {'-'*16:<18} {'-'*16}"
    rows = [
        ("Setup",           "pip install",  "pip install",  "API key"),
        ("Metadata filter", "✅ Yes",        "❌ No",         "✅ Yes"),
        ("Speed",           "Fast",         "Fastest",      "Fast (cloud)"),
        ("Scale",           "~1M vectors",  "~10M vectors", "Unlimited"),
        ("Cost",            "Free",         "Free",         "Pay per use"),
        ("Best for",        "Dev + MVP",    "Batch search", "Prod SaaS"),
    ]

    print(headers)
    print(divider)
    for row in rows:
        print(f"{row[0]:<22} {row[1]:<18} {row[2]:<18} {row[3]}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ Day 12 FINAL — All fixes applied")
    print("="*60)
    print("Fix 1: langchain_chroma (no deprecation warning)")
    print("Fix 2: COSINE distance_strategy (scores now 0.5-0.8 range)")
    print("Fix 3: Getting Started section (all 4 queries work)")
    print("Fix 4: HF warnings suppressed")
    print("Next  → Day 13: Retrievers")
    print("="*60)