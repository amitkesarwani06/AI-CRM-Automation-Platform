import os
import logging
from app.knowledge_base.rag.rag_chain import RAGChain
from app.knowledge_base.retriever.retriever_service import (
    RetrieverService, RetrieverType
)
from app.knowledge_base.vector_store.vector_store_factory import (
    UnifiedVectorStore, VectorStoreType
)
from app.knowledge_base.splitters.text_splitter import (
    DocumentChunkingService, ContentType
)

logger = logging.getLogger(__name__)


class KnowledgeBaseAgent:
    """
    CRM Knowledge Base Agent.

    All-in-one: ingest text → build RAG chain → answer questions.

    Usage:
        agent = KnowledgeBaseAgent()
        agent.ingest(company_text, source="company_faq")
        answer = agent.ask("How do I get a refund?")

    Week 3 will add tools and autonomous decision-making on top of this.
    """

    def __init__(
        self,
        store_type: VectorStoreType = VectorStoreType.FAISS,
        model: str = "llama-3.1-8b-instant", 
    ):
        self.store = UnifiedVectorStore(
            store_type=store_type,
            collection_name="crm_knowledge_base",
        )
        self.chunker = DocumentChunkingService(
            content_type=ContentType.CONVERSATIONAL
        )
        self.retriever_service = None
        self.rag_chain = None
        self.model = model
        self._is_ready = False
        logger.info("KnowledgeBaseAgent initialized")

    # ── Ingest ────────────────────────────────────────────────────────────────
    def ingest(self, text: str, source: str = "knowledge_base") -> int:
        """
        Ingest raw text into the knowledge base.
        Chunks → embeds → stores → builds RAG chain automatically.
        Returns number of chunks created.
        """
        chunks = self.chunker.split_text(text, source=source)
        self.store.add_documents(chunks)

        # Build retriever + RAG chain after ingestion
        self.retriever_service = RetrieverService(self.store)
        retriever = self.retriever_service.get_retriever(
            RetrieverType.SIMILARITY, k=3
        )
        self.rag_chain = RAGChain(retriever, model=self.model)
        self._is_ready = True

        logger.info(f"Ingested {len(chunks)} chunks from '{source}'")
        return len(chunks)

    # ── Ask ───────────────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        """
        Ask a question — get grounded answer from knowledge base.
        Simple one-line interface for agents and tools.
        """
        if not self._is_ready:
            return "Knowledge base is empty. Please call ingest() first."
        return self.rag_chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """
        Ask a question — get answer + source chunks used.
        Use when you want to show citations to the user.

        Returns:
            {
                "answer":  "You can get a refund within 30 days...",
                "sources": [Document(...), Document(...)],
                "context": "[Source 1: company_faq]\nRefund Policy..."
            }
        """
        if not self._is_ready:
            return {"answer": "Knowledge base empty. Call ingest() first.", "sources": []}
        return self.rag_chain.invoke_with_sources(question)

    def stream_answer(self, question: str) -> None:
        """
        Stream answer token by token — for chat UI.
        Tokens appear one at a time as LLM generates them.
        """
        if not self._is_ready:
            print("Knowledge base is empty. Call ingest() first.")
            return
        self.rag_chain.stream(question)

    # ── Mini Assignment: Multi-turn Chat ──────────────────────────────────────
    def chat(self, question: str, history: list = []) -> str:
        """
        Multi-turn RAG chat — remembers previous questions.
        Combines Day 7 Memory concept with today's RAG.

        history: list of (question, answer) tuples from previous turns
        Only uses last 3 turns to keep context short.

        Usage:
            history = []
            answer1 = agent.chat("What is your refund policy?", history)
            history.append(("What is your refund policy?", answer1))
            answer2 = agent.chat("How long does it take?", history)
        """
        context_question = question
        if history:
            history_text = "\n".join(
                [f"Q: {h[0]}\nA: {h[1]}" for h in history[-3:]]
            )
            context_question = (
                f"Previous conversation:\n{history_text}\n\nNew question: {question}"
            )
        return self.ask(context_question)

    # ── Utility ───────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        """True if knowledge base has been ingested and RAG chain is built."""
        return self._is_ready
