import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress HuggingFace warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

logger = logging.getLogger(__name__)

FAISS_PERSIST_DIR = "app/data/faiss_index"


class FAISSVectorStore:
    """
    FAISS-based vector store using LangChain's wrapper.
    LangChain handles embedding automatically.
    Faster than ChromaDB for pure search. No metadata filtering.
    Uses COSINE distance → scores always 0.0 to 1.0
    """

    def __init__(self, persist_dir: str = FAISS_PERSIST_DIR):
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = None
        logger.info(f"FAISSVectorStore ready | persist_dir={persist_dir}")

    def add_documents(self, chunks: list[Document]) -> None:
        """
        Add documents. LangChain embeds automatically.
        FIX: DistanceStrategy.COSINE → scores 0.0 to 1.0 (not L2 distance)
        """
        if self.store is None:
            # FIX: COSINE distance strategy
            self.store = FAISS.from_documents(
                chunks,
                self.embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            self.store.add_documents(chunks)

        # Save to disk (2 files: index.faiss + index.pkl)
        self.store.save_local(self.persist_dir)
        logger.info(f"FAISS index saved | {len(chunks)} chunks added")

    def load(self) -> bool:
        """
        Load existing FAISS index from disk.
        FIX: Must pass same DistanceStrategy as used during add_documents()
        """
        index_file = Path(self.persist_dir) / "index.faiss"
        if index_file.exists():
            # FIX: DistanceStrategy.COSINE must match what was used during save
            self.store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.COSINE,
            )
            logger.info("FAISS index loaded from disk")
            return True
        return False

    def search(self, query: str, k: int = 3) -> list[Document]:
        """Search — returns top-k Documents."""
        if self.store is None:
            raise RuntimeError("No FAISS index. Call add_documents() or load() first.")
        return self.store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """
        Search and return (Document, similarity_score) pairs.
        With COSINE strategy: score is cosine similarity (0.0 to 1.0)
        No manual conversion needed anymore.
        """
        if self.store is None:
            raise RuntimeError("No FAISS index. Call add_documents() or load() first.")

        raw = self.store.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in raw:
            # With COSINE strategy: score IS cosine similarity directly
            similarity = round(float(score), 4)
            similarity = max(0.0, min(1.0, similarity))  # safety clamp
            results.append((doc, similarity))
        return results

    def as_retriever(self, k: int = 3):
        """Returns LangChain Retriever — plugs into RAG chains Day 13+."""
        if self.store is None:
            raise RuntimeError("No FAISS index. Call add_documents() or load() first.")
        return self.store.as_retriever(search_kwargs={"k": k})


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

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
"""

    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(sample_text, source="company_faq")
    print(f"Chunks: {len(chunks)}")

    # Test 1: Add and search
    print("\n" + "="*50)
    print("TEST 1: Add + Search")
    print("="*50)
    store = FAISSVectorStore()
    store.add_documents(chunks)

    queries = [
        "How do I get a refund?",
        "What does the Growth plan cost?",
        "When is support available?",
    ]
    for query in queries:
        results = store.search_with_scores(query, k=1)
        if results:
            doc, score = results[0]
            status = "✅" if score >= 0.5 else "⚠️"
            print(f"{status} Score: {score:.4f} | Query: '{query}'")
            print(f"   Match: {doc.page_content[:60].strip()}...")
            print()

    # Test 2: Persistence
    print("="*50)
    print("TEST 2: Persistence — reload from disk")
    print("="*50)
    new_store = FAISSVectorStore()
    loaded = new_store.load()
    print(f"Loaded: {loaded}")
    if loaded:
        results = new_store.search("refund", k=1)
        print(f"After reload: {results[0].page_content[:50].strip()}...")
        print("✅ Persistence working!")

    # Test 3: Retriever
    print("\n" + "="*50)
    print("TEST 3: as_retriever()")
    print("="*50)
    retriever = store.as_retriever(k=2)
    print(f"Type: {type(retriever).__name__}")
    docs = retriever.invoke("refund policy")
    print(f"Returned {len(docs)} docs ✅")

    print("\n✅ faiss_store.py working correctly!")
    print("Next → Day 13: Retrievers")