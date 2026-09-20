import os
import logging
import uuid
from pathlib import Path
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = "app/data/chroma_db"

class ChromaVectorStore:
    """
    Persistent vector store using ChromaDB.
    Stores chunk text + embeddings + metadata.
    Survives server restarts — embed once, search forever.
    """

    def __init__(
        self,
        collection_name: str = "crm_knowledge_base",
        persist_dir: str = CHROMA_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # Create storage directory if it doesn't exist
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # PersistentClient → saves to disk (NOT in-memory)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get existing collection or create a new one
        # hnsw:space = "cosine" → use cosine distance for similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB ready | collection='{collection_name}' | "
            f"documents={self.collection.count()} | path={persist_dir}"
        )

    def add_chunks(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Store chunks + embeddings. Returns list of assigned IDs."""
        if not chunks:
            logger.warning("No chunks to add.")
            return []

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) mismatch."
            )

        # Generate a unique ID for each chunk
        ids = [str(uuid.uuid4()) for _ in chunks]

        texts = [chunk.page_content for chunk in chunks]

        # Clean metadata — ChromaDB only allows str, int, float, bool
        metadatas = []
        for chunk in chunks:
            clean_meta = {}
            for k, v in chunk.metadata.items():
                clean_meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
            metadatas.append(clean_meta)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            f"Added {len(chunks)} chunks. Total stored: {self.collection.count()}"
        )
        return ids

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Search for most similar chunks to a query vector.
        where: optional metadata filter e.g. {"source": "company_faq"}
        score: 1.0 = perfect match, 0.0 = unrelated
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity_score = round(1 - distance, 4)   # convert distance → similarity
            formatted.append({
                "id":       results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": round(distance, 4),
                "score":    similarity_score,
            })

        return formatted

    def count(self) -> int:
        """Total chunks stored."""
        return self.collection.count()

    def get_all(self) -> list[dict]:
        """Retrieve all stored chunks — useful for debugging."""
        results = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Collection '{self.collection_name}' deleted.")

    def reset(self) -> None:
        """Clear all documents and recreate empty collection."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' reset (empty).")

    def collection_info(self) -> dict:
        """Return summary info about this collection."""
        return {
            "name":        self.collection_name,
            "count":       self.collection.count(),
            "persist_dir": self.persist_dir,
        }

    def search_with_context(self, query_embedding: list[float], n_results: int = 3) -> str:
        """
        Mini Assignment — returns results as a formatted context string
        ready to be injected into an LLM prompt (Day 14 will use this).
        """
        results = self.search(query_embedding, n_results)
        context = ""
        for i, r in enumerate(results, 1):
            context += f"[Source {i}: {r['metadata'].get('source', 'unknown')}]\n"
            context += f"{r['text']}\n\n"
        return context.strip()

# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, os.getcwd())

    from app.knowledge_base.splitters.text_splitter import DocumentChunkingService, ContentType
    from app.knowledge_base.embeddings.embedding_service import HuggingFaceEmbeddingService

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

Getting Started
After signing up you will receive a welcome email with setup instructions.
Your first 14 days are free with no credit card required.
"""

    print("\n" + "="*60)
    print("DAY 11 — ChromaDB Vector Store Tests")
    print("="*60)

    # STEP 1: Chunk + embed
    print("\n[STEP 1] Chunking + embedding...")
    embedding_service = HuggingFaceEmbeddingService()
    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(sample_text, source="company_faq")
    chunk_embeddings = embedding_service.embed_chunks(chunks)
    embeddings_only = [vec for _, vec in chunk_embeddings]
    print(f"Chunks: {len(chunks)} | Dims: {len(embeddings_only[0])}")

    # STEP 2: Store in ChromaDB
    print("\n[STEP 2] Storing in ChromaDB...")
    store = ChromaVectorStore(collection_name="test_crm_kb")
    store.reset()
    ids = store.add_chunks(chunks, embeddings_only)
    print(f"Stored: {store.count()} chunks")

    # STEP 3: Semantic Search
    print("\n" + "="*60)
    print("[STEP 3] Semantic Search Tests")
    print("="*60)
    queries = [
        "How do I get a refund?",
        "What is the price of the Growth plan?",
        "When is customer support available?",
        "How do I get started?",
    ]
    for query in queries:
        query_vec = embedding_service.embed_text(query)
        results = store.search(query_vec, n_results=1)
        if results:
            r = results[0]
            print(f"\nQuery: '{query}'")
            print(f"  Score: {r['score']:.4f} | {r['text'][:70].strip()}...")

    # STEP 4: Metadata filter
    print("\n" + "="*60)
    print("[STEP 4] Metadata Filter Test")
    print("="*60)
    extra_chunk_text = "WhatsApp Integration\nConnect your WhatsApp Business account in Settings > Integrations."
    extra_chunks = DocumentChunkingService(
        content_type=ContentType.CONVERSATIONAL
    ).split_text(extra_chunk_text, source="whatsapp_guide")
    extra_embeddings = [embedding_service.embed_text(c.page_content) for c in extra_chunks]
    store.add_chunks(extra_chunks, extra_embeddings)
    print(f"Total chunks now: {store.count()}")

    query_vec = embedding_service.embed_text("pricing")
    filtered = store.search(query_vec, n_results=2, where={"source": "company_faq"})
    print(f"\nFiltered (source=company_faq only):")
    for r in filtered:
        print(f"  Score: {r['score']:.4f} | {r['text'][:60].strip()}...")

    # STEP 5: Persistence test
    print("\n" + "="*60)
    print("[STEP 5] Persistence Test — reload from disk")
    print("="*60)
    store2 = ChromaVectorStore(collection_name="test_crm_kb")
    count = store2.count()
    print(f"Reloaded — chunks found: {count}")
    print("✅ Vectors survived restart!" if count > 0 else "❌ Persistence failed")

    # STEP 6: search_with_context (Mini Assignment)
    print("\n" + "="*60)
    print("[STEP 6] search_with_context() — Mini Assignment")
    print("="*60)
    query_vec = embedding_service.embed_text("refund")
    context = store2.search_with_context(query_vec, n_results=2)
    print("Context string for LLM prompt:\n")
    print(context)

    # STEP 7: Collection info
    print("\n" + "="*60)
    print("[STEP 7] Collection Summary")
    print("="*60)
    for k, v in store2.collection_info().items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("✅ Day 11 COMPLETE — ChromaDB working!")
    print("Next → Day 12: FAISS + Pinecone")
    print("="*60)
