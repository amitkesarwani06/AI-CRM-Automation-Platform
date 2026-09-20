import os
import sys
import logging
from enum import Enum
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


# ── Retriever Types ───────────────────────────────────────────────────────────
class RetrieverType(str, Enum):
    SIMILARITY  = "similarity"   # basic top-k (default)
    MMR         = "mmr"          # diverse results
    COMPRESSION = "compression"  # extract only relevant sentence


# ── Main Service ──────────────────────────────────────────────────────────────
class RetrieverService:
    """
    Builds and manages different types of retrievers.

    Usage:
        store = UnifiedVectorStore(...)
        store.add_documents(chunks)

        service = RetrieverService(store)
        docs = service.retrieve("How do I get a refund?")
    """

    def __init__(self, vector_store):
        """
        vector_store: UnifiedVectorStore instance from Day 12
        """
        self.vector_store = vector_store
        logger.info("RetrieverService initialized")

    # ── Get Retriever Object ──────────────────────────────────────────────────
    def get_retriever(
        self,
        retriever_type: RetrieverType = RetrieverType.SIMILARITY,
        k: int = 3,
    ) -> BaseRetriever:
        """
        Returns a LangChain Retriever object.
        This object plugs directly into LCEL chains (Day 14).

        retriever_type: SIMILARITY / MMR / COMPRESSION
        k: number of documents to return
        """
        if retriever_type == RetrieverType.SIMILARITY:
            return self._similarity_retriever(k)

        elif retriever_type == RetrieverType.MMR:
            return self._mmr_retriever(k)

        elif retriever_type == RetrieverType.COMPRESSION:
            return self._compression_retriever(k)

        raise ValueError(f"Unknown retriever type: {retriever_type}")

    # ── Similarity Retriever ──────────────────────────────────────────────────
    def _similarity_retriever(self, k: int) -> BaseRetriever:
        """
        Basic similarity retriever — most common, fastest.
        Returns top-k most similar documents.

        When to use:
        - Most RAG use cases
        - When speed matters
        - Default choice
        """
        return self.vector_store.as_retriever(k=k)

    # ── MMR Retriever ─────────────────────────────────────────────────────────
    def _mmr_retriever(self, k: int) -> BaseRetriever:
        """
        MMR = Maximal Marginal Relevance
        Returns diverse results — avoids 3 chunks saying the same thing.

        How it works:
        1. Get top 20 candidates by similarity
        2. Pick k that are most DIFFERENT from each other
        3. Result = relevant + diverse

        When to use:
        - Knowledge base has overlapping content
        - Want variety in retrieved chunks
        - Large knowledge bases
        """
        if self.vector_store._store is None:
            raise RuntimeError("No documents in vector store.")

        return self.vector_store._store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 20,       # candidates to consider
                "lambda_mult": 0.5,  # 0=max diversity, 1=max similarity
            }
        )

    # ── Compression Retriever ─────────────────────────────────────────────────
    def _compression_retriever(self, k: int) -> BaseRetriever:
        """
        Contextual Compression Retriever.
        Retrieves chunks → then extracts ONLY the relevant sentence.

        Example:
        Full chunk:  "We offer refunds within 30 days.
                      Our support team is available 9am-7pm..."
        Compressed:  "We offer refunds within 30 days."

        When to use:
        - Large chunks with mixed content
        - Want to reduce token usage
        - Need very precise answers

        Note: costs one extra LLM call per chunk retrieved.
        Requires OPENAI_API_KEY in .env
        """
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor
            from langchain_openai import ChatOpenAI

            base_retriever = self._similarity_retriever(k * 2)
            compressor = LLMChainExtractor.from_llm(
                ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            )
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")

    # ── Convenience Methods ───────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        retriever_type: RetrieverType = RetrieverType.SIMILARITY,
        k: int = 3,
    ) -> list[Document]:
        """
        Shortcut — retrieve documents directly without calling get_retriever().

        Usage:
            docs = service.retrieve("How do I get a refund?")
        """
        retriever = self.get_retriever(retriever_type, k)
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} docs for: '{query[:50]}'")
        return docs

    def retrieve_with_display(
        self,
        query: str,
        k: int = 3,
    ) -> list[Document]:
        """
        Retrieve + print formatted results.
        Use during development to inspect what LLM will receive.

        Usage:
            service.retrieve_with_display("refund policy")
        """
        docs = self.retrieve(query, k=k)
        print(f"\nQuery: '{query}'")
        print(f"Retrieved {len(docs)} chunks:\n")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            print(f"  [{i}] Source: {source}")
            print(f"       Content: {doc.page_content[:120].strip()}...")
            print()
        return docs

    def build_context_string(self, docs: list[Document]) -> str:
        """
        Convert retrieved documents into a formatted context string.
        This string gets injected into the LLM prompt in Day 14.

        Output format:
        [Source 1: company_faq]
        Refund Policy...

        [Source 2: company_faq]
        Support Hours...

        When to use:
        - Always — before passing context to LLM
        - Day 14 RAG chain uses this automatically
        """
        if not docs:
            return "No relevant information found in the knowledge base."

        context = ""
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context += f"[Source {i}: {source}]\n"
            context += f"{doc.page_content.strip()}\n\n"
        return context.strip()

    def search_with_filter(
        self,
        query: str,
        source: str,
        k: int = 3,
    ) -> list[Document]:
        """
        Mini Assignment — search only within a specific source.

        Example:
            # SupportAgent: search only FAQ docs
            docs = service.search_with_filter("refund", source="company_faq")

            # SalesAgent: search only product docs
            docs = service.search_with_filter("pricing", source="product_catalog")

        How it works:
        1. Retrieve k*2 docs (more candidates)
        2. Filter by source metadata
        3. Return top k filtered results
        """
        docs = self.retrieve(query, k=k * 2)
        filtered = [
            d for d in docs
            if d.metadata.get("source") == source
        ]
        logger.info(
            f"search_with_filter: {len(filtered)} docs matched "
            f"source='{source}' from {len(docs)} retrieved"
        )
        return filtered[:k]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    from app.knowledge_base.splitters.text_splitter import (
        DocumentChunkingService, ContentType
    )
    from app.knowledge_base.vector_store.vector_store_factory import (
        UnifiedVectorStore, VectorStoreType
    )

    # ── Sample Data ───────────────────────────────────────────────────────────
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

    print("\n" + "="*60)
    print("DAY 13 — RetrieverService Tests")
    print("="*60)

    # Setup
    print("\n[SETUP] Preparing vector store...")
    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(sample_text, source="company_faq")
    print(f"Chunks created: {len(chunks)}")

    store = UnifiedVectorStore(
        store_type=VectorStoreType.FAISS,
        collection_name="day13_test",
    )
    store.add_documents(chunks)
    print("Vector store ready.\n")

    # Initialize
    service = RetrieverService(vector_store=store)

    # ── TEST 1: Basic Similarity Retriever ────────────────────────────────────
    print("="*60)
    print("TEST 1: Basic Similarity Retriever")
    print("="*60)

    retriever = service.get_retriever(RetrieverType.SIMILARITY, k=1)
    print(f"Retriever type: {type(retriever).__name__}\n")

    test_queries = [
        "How do I get a refund?",
        "What is the pricing?",
        "When is support available?",
        "How do I get started?",
    ]

    for query in test_queries:
        docs = retriever.invoke(query)
        if docs:
            print(f"Query:  '{query}'")
            print(f"Result: '{docs[0].page_content[:65].strip()}...'")
            print()

    # ── TEST 2: retrieve_with_display() ───────────────────────────────────────
    print("="*60)
    print("TEST 2: retrieve_with_display()")
    print("="*60)
    service.retrieve_with_display("How do I get a refund?", k=2)

    # ── TEST 3: build_context_string() ────────────────────────────────────────
    print("="*60)
    print("TEST 3: build_context_string() — what LLM sees tomorrow")
    print("="*60)
    docs = service.retrieve("How do I get a refund?", k=2)
    context = service.build_context_string(docs)
    print("Context string:\n")
    print(context)

    # ── TEST 4: Retriever as Runnable (LCEL) ──────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Retriever as Runnable — LCEL")
    print("="*60)

    retriever = service.get_retriever(RetrieverType.SIMILARITY, k=2)

    print("1. invoke() — single query:")
    docs = retriever.invoke("refund policy")
    print(f"   → {len(docs)} documents returned")

    print("\n2. batch() — multiple queries at once:")
    batch_results = retriever.batch([
        "refund policy",
        "pricing plans",
        "support hours",
    ])
    for i, result in enumerate(batch_results):
        print(f"   Query {i+1} → {len(result)} docs | "
              f"'{result[0].page_content[:40].strip()}...'")

    # ── TEST 5: MMR Retriever ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: MMR Retriever — diverse results")
    print("="*60)
    try:
        mmr_retriever = service.get_retriever(RetrieverType.MMR, k=2)
        docs = mmr_retriever.invoke("CRM platform information")
        print(f"MMR returned {len(docs)} diverse docs:")
        for i, doc in enumerate(docs, 1):
            print(f"  Doc {i}: '{doc.page_content[:60].strip()}...'")
    except Exception as e:
        print(f"MMR note: {e}")

    # ── TEST 6: search_with_filter() — Mini Assignment ─────────────────────────
    print("\n" + "="*60)
    print("TEST 6: search_with_filter() — Mini Assignment")
    print("="*60)

    # Add a second source for testing
    extra_chunks = DocumentChunkingService(
        content_type=ContentType.CONVERSATIONAL
    ).split_text(
        "WhatsApp Integration\n"
        "Connect your WhatsApp Business account from Settings > Integrations.\n"
        "Send automated messages to leads directly from the CRM.",
        source="whatsapp_guide"
    )
    store.add_documents(extra_chunks)

    print("Searching ONLY in company_faq source:")
    faq_docs = service.search_with_filter("refund", source="company_faq", k=2)
    for doc in faq_docs:
        print(f"  Source: {doc.metadata.get('source')} | "
              f"'{doc.page_content[:55].strip()}...'")

    print("\nSearching ONLY in whatsapp_guide source:")
    wa_docs = service.search_with_filter("WhatsApp", source="whatsapp_guide", k=2)
    for doc in wa_docs:
        print(f"  Source: {doc.metadata.get('source')} | "
              f"'{doc.page_content[:55].strip()}...'")

    # ── Final Pipeline ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FULL WEEK 2 PIPELINE STATUS")
    print("="*60)
    print("✅ Day 8  — Document Loaders")
    print("✅ Day 9  — Text Splitters")
    print("✅ Day 10 — Embedding Models")
    print("✅ Day 11 — ChromaDB")
    print("✅ Day 12 — FAISS + Pinecone + Factory")
    print("✅ Day 13 — Retriever Service")
    print("⬜ Day 14 — Full RAG Chain ← TOMORROW 🎯")
    print("="*60)
    print("Next → Day 14: Full RAG Chain — Knowledge Base LIVE!")
    print("="*60)