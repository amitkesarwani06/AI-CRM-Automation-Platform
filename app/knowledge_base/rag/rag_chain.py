import os
import sys
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from langchain_core.documents import Document

load_dotenv()

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


# ── RAG System Prompt ─────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful CRM assistant for our platform.

Answer the user's question using ONLY the context provided below.

RULES:
1. Answer ONLY from the context — never use outside knowledge.
2. If the answer is not in the context, say exactly:
   "I don't have that information in our knowledge base.
    Please contact support@crmplatform.com for help."
3. Keep answers concise — 2 to 4 sentences maximum.
4. Always mention the source when possible.
5. Never make up prices, dates, or contact details.

CONTEXT:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ── Format Documents ──────────────────────────────────────────────────────────
def format_docs(docs: list[Document]) -> str:
    """
    Convert list of Documents into a formatted context string.
    This string gets inserted into {context} in the RAG prompt.

    Output example:
    [Source 1: company_faq]
    Refund Policy
    We offer a 30-day money-back guarantee...

    [Source 2: company_faq]
    Support Hours
    Available Monday to Saturday...
    """
    if not docs:
        return "No relevant information found in the knowledge base."

    formatted = ""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted += f"[Source {i}: {source}]\n"
        formatted += f"{doc.page_content.strip()}\n\n"
    return formatted.strip()


# ── RAG Chain ─────────────────────────────────────────────────────────────────
class RAGChain:
    """
    Full RAG pipeline:
    question → retrieve → format → prompt → LLM → answer

    LCEL Chain:
    RunnableParallel runs TWO branches at once:
      Branch 1: retriever fetches docs → format_docs converts to string
      Branch 2: RunnablePassthrough keeps original question unchanged

    Then:
      → RAG_PROMPT fills {context} and {question}
      → LLM generates grounded answer
      → StrOutputParser extracts plain string

    Usage:
        rag = RAGChain(retriever)
        answer = rag.invoke("How do I get a refund?")
    """

    def __init__(self, retriever, model: str = "gpt-4o-mini"):
        self.retriever = retriever
        self.model = model

        # LLM — temperature=0 for deterministic factual answers
        self.llm = ChatGroq(
            model=model, 
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        # THE FULL RAG CHAIN — read left to right
        self.chain = (
            RunnableParallel(
                # Branch 1: retrieve docs + format as context string
                context=self.retriever | format_docs,
                # Branch 2: pass question through unchanged
                question=RunnablePassthrough(),
            )
            | RAG_PROMPT        # fills {context} and {question}
            | self.llm          # generates answer → AIMessage
            | StrOutputParser() # extracts .content string
        )

        logger.info(f"RAGChain ready | model={model}")

    def invoke(self, question: str) -> str:
        """
        Ask a question — get grounded answer from knowledge base.

        Usage:
            answer = rag.invoke("How do I get a refund?")
            print(answer)
        """
        logger.info(f"RAG query: '{question[:60]}'")
        return self.chain.invoke(question)

    def invoke_with_sources(self, question: str) -> dict:
        """
        Ask a question — get answer AND source chunks used.

        Returns dict:
            {
                "answer":  "You can get a refund within 30 days...",
                "sources": [Document(...), Document(...)],
                "context": "[Source 1: company_faq]\nRefund Policy..."
            }

        Use when:
            - Building UI that shows citations
            - Debugging why an answer is wrong
            - Verifying which chunks were retrieved
        """
        # Get source docs separately for inspection
        source_docs = self.retriever.invoke(question)
        context = format_docs(source_docs)

        # Get answer from chain
        answer = self.chain.invoke(question)

        return {
            "answer":  answer,
            "sources": source_docs,
            "context": context,
        }

    def stream(self, question: str) -> None:
        """
        Stream answer token by token — real-time response.

        Use when:
            - Chat UI where user sees tokens appear live
            - Long answers where waiting feels slow

        Usage:
            print("A: ", end="")
            rag.stream("What plans do you offer?")
        """
        for chunk in self.chain.stream(question):
            print(chunk, end="", flush=True)
        print()  # newline after stream ends

    def batch(self, questions: list[str]) -> list[str]:
        """
        Answer multiple questions in parallel — faster than loop.

        Usage:
            answers = rag.batch(["question1", "question2", "question3"])

        Returns list of answers in same order as questions.
        """
        return self.chain.batch(questions)

    def ainvoke(self, question: str):
        """
        Async version — use in FastAPI endpoints (Week 4).
        Never block the event loop with sync calls.

        Usage:
            answer = await rag.ainvoke("How do I get a refund?")
        """
        return self.chain.ainvoke(question)


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    from app.knowledge_base.splitters.text_splitter import (
        DocumentChunkingService, ContentType
    )
    from app.knowledge_base.vector_store.vector_store_factory import (
        UnifiedVectorStore, VectorStoreType
    )
    from app.knowledge_base.retriever.retriever_service import (
        RetrieverService, RetrieverType
    )

    # ── Company Knowledge Base ────────────────────────────────────────────────
    company_knowledge = """
Refund Policy
We offer a 30-day money-back guarantee on all plans.
To request a refund contact support@crmplatform.com.
Refunds are processed within 5-7 business days.
No questions asked for cancellations within the first 14 days.

Pricing Plans
Starter Plan Rs.999 per month for up to 3 users and 500 leads.
Growth Plan Rs.2999 per month for up to 15 users with AI features.
Enterprise Plan has custom pricing with dedicated support and unlimited users.
Annual billing gives 20 percent discount on all plans.

Support Hours
Available Monday to Saturday 9am to 7pm IST.
Average response time is under 2 hours during business hours.
Emergency support is available 24x7 for Enterprise customers.
Contact us at support@crmplatform.com or call 1800-123-4567.

Getting Started
After signing up you will receive a welcome email with setup instructions.
Your first 14 days are free with no credit card required.
Book an onboarding call at calendly.com/crmplatform/onboarding.
Import your existing leads from CSV in Settings then Import Data.

Features
Lead management with AI scoring and automatic follow-up reminders.
Email automation with customizable templates and scheduling.
WhatsApp integration for direct messaging to leads.
Analytics dashboard with real-time sales reports.
"""

    # ── Setup Pipeline ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DAY 14 — Full RAG Chain")
    print("Week 2 Capstone — Knowledge Base LIVE!")
    print("="*60)

    print("\n[SETUP] Building full RAG pipeline...")

    # Step 1: Chunk
    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(company_knowledge, source="company_faq")
    print(f"Step 1 ✅ Chunks created: {len(chunks)}")

    # Step 2: Store in FAISS
    store = UnifiedVectorStore(
        store_type=VectorStoreType.FAISS,
        collection_name="crm_kb_day14",
    )
    store.add_documents(chunks)
    print(f"Step 2 ✅ Stored in FAISS vector store")

    # Step 3: Retriever
    retriever_service = RetrieverService(store)
    retriever = retriever_service.get_retriever(RetrieverType.SIMILARITY, k=3)
    print(f"Step 3 ✅ Retriever ready: {type(retriever).__name__}")

    # Step 4: RAG Chain
    rag = RAGChain(retriever, model="llama-3.1-8b-instant")
    print(f"Step 4 ✅ RAG Chain ready")
    print("\nPipeline: Chunk → FAISS → Retriever → RAG Chain ✅\n")

    # ── TEST 1: Basic Q&A ─────────────────────────────────────────────────────
    print("="*60)
    print("TEST 1: Basic Knowledge Base Q&A")
    print("="*60)

    questions = [
        "How do I get a refund?",
        "What does the Growth plan cost?",
        "When is customer support available?",
        "How do I import my leads?",
        "Does the plan include WhatsApp?",
        "Is there a free trial?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.invoke(question)
        print(f"A: {answer}")
        print("-" * 50)

    # ── TEST 2: Answer with Sources ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Answer with Sources (invoke_with_sources)")
    print("="*60)

    result = rag.invoke_with_sources("What is the refund policy?")
    print(f"Answer:\n{result['answer']}")
    print(f"\nSources used ({len(result['sources'])} chunks):")
    for i, doc in enumerate(result['sources'], 1):
        src = doc.metadata.get('source', 'unknown')
        print(f"  [{i}] {src} → {doc.page_content[:60].strip()}...")
    print(f"\nFull context sent to LLM ({len(result['context'])} chars)")

    # ── TEST 3: Hallucination Prevention ─────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Hallucination Prevention")
    print("="*60)

    out_of_scope = [
        "What is the CEO's name?",
        "What is the stock price?",
        "Who founded the company?",
    ]

    for question in out_of_scope:
        print(f"\nQ: {question}")
        answer = rag.invoke(question)
        print(f"A: {answer}")
        has_guard = (
            "don't have" in answer.lower() or
            "not in" in answer.lower() or
            "knowledge base" in answer.lower() or
            "contact" in answer.lower()
        )
        status = "✅ Hallucination prevented!" if has_guard else "⚠️ Check prompt"
        print(f"   {status}")

    # ── TEST 4: Streaming ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Streaming Response (token by token)")
    print("="*60)

    print("Q: What plans do you offer?\n")
    print("A: ", end="")
    rag.stream("What plans do you offer?")

    # ── TEST 5: Batch ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: Batch — 3 questions in parallel")
    print("="*60)

    batch_questions = [
        "What is the refund policy?",
        "What are the support hours?",
        "Is there a free trial?",
    ]

    print("Sending 3 questions in parallel...\n")
    answers = rag.batch(batch_questions)

    for q, a in zip(batch_questions, answers):
        print(f"Q: {q}")
        print(f"A: {a[:120]}...")
        print()

    # ── TEST 6: format_docs() Inspection ─────────────────────────────────────
    print("="*60)
    print("TEST 6: format_docs() — what LLM actually sees")
    print("="*60)

    docs = retriever.invoke("refund policy")
    context = format_docs(docs)
    print(f"Retrieved {len(docs)} chunks")
    print(f"Context string ({len(context)} chars):\n")
    print(context)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("WEEK 2 COMPLETE!")
    print("="*60)
    print("✅ Day 8  — Document Loaders")
    print("✅ Day 9  — Text Splitters")
    print("✅ Day 10 — Embedding Models")
    print("✅ Day 11 — ChromaDB")
    print("✅ Day 12 — FAISS + Pinecone + Factory")
    print("✅ Day 13 — Retriever Service")
    print("✅ Day 14 — Full RAG Chain ← COMPLETE TODAY 🎉")
    print()
    print("RAG Pipeline:")
    print("  Text → Chunks → Vectors → Store → Retrieve → LLM → Answer")
    print()
    print("Next → Week 3: Tools + Agents + ReAct")
    print("  Agents that ACT — not just answer!")
    print("="*60)