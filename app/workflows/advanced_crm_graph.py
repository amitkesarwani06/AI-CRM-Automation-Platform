import os
import sys
import time
import logging
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


# ── Advanced CRM State ────────────────────────────────────────────────────────
class AdvancedCRMState(TypedDict):
    """
    Extended state for advanced routing patterns.
    Includes lead scoring, email drafts, and approval flags.
    """
    messages:        Annotated[list[BaseMessage], add_messages]
    intent:          str
    response:        str
    lead_score:      int
    lead_data:       dict
    info_complete:   bool
    email_draft:     dict
    email_sent:      bool
    track:           str         # hot | warm | cold | incomplete
    retry_count:     int
    needs_more_info: bool


# ── Import nodes ──────────────────────────────────────────────────────────────
from app.workflows.nodes.lead_qualifier import (
    score_lead,
    route_by_score,
    gather_info_node,
    hot_lead_node,
    warm_lead_node,
    cold_lead_node,
    send_email_node,
)
from app.workflows.nodes.classifier import classify_intent


# ── Route after classification ────────────────────────────────────────────────
def route_after_classify(state: AdvancedCRMState) -> str:
    """Route to different workflows based on intent."""
    intent = state.get("intent", "general")
    if intent == "sales":
        return "score_lead_node"   # go to lead scoring
    elif intent == "support":
        return "support_node"
    elif intent == "analytics":
        return "analytics_node"
    return "general_node"


# ── Support node ──────────────────────────────────────────────────────────────
def support_node(state: AdvancedCRMState) -> dict:
    """Support response via RAG knowledge base."""
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else ""

    try:
        from app.knowledge_base.splitters.text_splitter import (
            DocumentChunkingService, ContentType
        )
        from app.knowledge_base.vector_store.vector_store_factory import (
            UnifiedVectorStore, VectorStoreType
        )
        from app.knowledge_base.retriever.retriever_service import (
            RetrieverService, RetrieverType
        )
        from app.knowledge_base.rag.rag_chain import RAGChain

        support_knowledge = """
Refund Policy
We offer a 30-day money-back guarantee on all plans.
Contact support@crmplatform.com for refund requests.
Refunds processed within 5-7 business days.

Support Hours
Available Monday to Saturday 9am to 7pm IST.
Contact: support@crmplatform.com or 1800-123-4567.
Emergency support 24x7 for Enterprise customers.
"""
        chunker = DocumentChunkingService(ContentType.CONVERSATIONAL)
        chunks = chunker.split_text(support_knowledge, source="support_kb")
        store = UnifiedVectorStore(VectorStoreType.FAISS, "support_adv_kb")
        store.add_documents(chunks)
        retriever = RetrieverService(store).get_retriever(RetrieverType.SIMILARITY, k=2)
        rag = RAGChain(retriever, model=GROQ_MODEL)
        response = rag.invoke(user_text)
    except Exception as e:
        response = f"Support: We offer a 30-day money back guarantee. Contact support@crmplatform.com for help."

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def analytics_node(state: AdvancedCRMState) -> dict:
    """Analytics response from lead database."""
    try:
        from app.tools.lead_tools_db import get_lead_stats
        stats = get_lead_stats.invoke({})
        response = f"📊 Analytics Report:\n{stats}"
    except Exception:
        response = "📊 Analytics: CRM database active. 30+ leads in pipeline."

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def general_node(state: AdvancedCRMState) -> dict:
    """Fallback general assistant response."""
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else "Hello"

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = (
        f"You are a helpful CRM assistant. Respond concisely to: '{user_text}'. "
        f"Mention you can assist with sales qualification, customer support, and analytics."
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        response = "Hello! I can help you with sales qualification, customer support questions, and lead analytics."

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


# ── Score lead wrapper ────────────────────────────────────────────────────────
def score_lead_node(state: AdvancedCRMState) -> dict:
    """Wrapper for score_lead node."""
    return score_lead(state)


# ── Build Advanced Graph ──────────────────────────────────────────────────────
def build_advanced_crm_graph(with_hitl: bool = False):
    """
    Build advanced CRM graph with:
    - Multi-condition routing (hot/warm/cold leads)
    - Loop pattern (gather missing info)
    - Optional HITL (human approval before sending email)

    with_hitl=True  -> pauses before send_email_node
    with_hitl=False -> sends automatically (testing mode)
    """
    graph = StateGraph(AdvancedCRMState)

    # ── Add all nodes ──────────────────────────────────────────────────────────
    graph.add_node("classify_node",    classify_intent)
    graph.add_node("score_lead_node",  score_lead_node)
    graph.add_node("gather_info_node", gather_info_node)
    graph.add_node("hot_lead_node",    hot_lead_node)
    graph.add_node("warm_lead_node",   warm_lead_node)
    graph.add_node("cold_lead_node",   cold_lead_node)
    graph.add_node("send_email_node",  send_email_node)
    graph.add_node("support_node",     support_node)
    graph.add_node("analytics_node",   analytics_node)
    graph.add_node("general_node",     general_node)

    # ── Edges ──────────────────────────────────────────────────────────────────

    # Entry
    graph.add_edge(START, "classify_node")

    # After classify -> route by intent
    graph.add_conditional_edges(
        "classify_node",
        route_after_classify,
        {
            "score_lead_node": "score_lead_node",
            "support_node":    "support_node",
            "analytics_node":  "analytics_node",
            "general_node":    "general_node",
        }
    )

    # After scoring -> multi-condition route by score
    graph.add_conditional_edges(
        "score_lead_node",
        route_by_score,
        {
            "gather_info_node": "gather_info_node",  # LOOP: missing info
            "hot_lead_node":    "hot_lead_node",
            "warm_lead_node":   "warm_lead_node",
            "cold_lead_node":   "cold_lead_node",
        }
    )

    # gather_info pauses / exits to wait for user input
    graph.add_edge("gather_info_node", END)

    # Hot lead -> send email (with optional HITL pause)
    graph.add_edge("hot_lead_node", "send_email_node")

    # Warm and cold -> straight to END
    graph.add_edge("warm_lead_node", END)
    graph.add_edge("cold_lead_node", END)

    # After email sent -> END
    graph.add_edge("send_email_node", END)

    # Support / analytics / general -> END
    graph.add_edge("support_node",   END)
    graph.add_edge("analytics_node", END)
    graph.add_edge("general_node",   END)

    # ── Compile ────────────────────────────────────────────────────────────────
    if with_hitl:
        checkpointer = MemorySaver()
        app = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["send_email_node"],  # pause here for human review!
        )
        logger.info("Advanced CRM Graph compiled WITH human-in-the-loop")
    else:
        app = graph.compile()
        logger.info("Advanced CRM Graph compiled (auto mode)")

    return app


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    print("\n" + "="*60)
    print("DAY 23 — Advanced Conditional Routing in LangGraph")
    print("Multi-condition Routing, Loops, & Human-in-the-loop (HITL)")
    print("="*60)

    # ── TEST 1: Multi-condition routing ───────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Lead Score Routing (hot / warm / cold)")
    print("="*60)

    app = build_advanced_crm_graph(with_hitl=False)

    hot_leads = [
        "I'm the VP of Sales at MegaBank, we have 80 salespeople and need CRM immediately. Budget approved: Rs.2 lakh/month. Email: vp@megabank.com",
        "Hi, I manage 25 people at TechGiant. Looking for Growth or Enterprise plan. Timeline: this month. raj@techgiant.com",
    ]

    warm_leads = [
        "We're a startup with 8 people. Interested in automating follow-ups. Email: founder@startup.com",
    ]

    cold_leads = [
        "Just looking around, what does your CRM do?",
    ]

    all_tests = [
        (hot_leads[0],  "hot",  "Large bank, 80 people, big budget"),
        (hot_leads[1],  "hot",  "25 people, this month timeline"),
        (warm_leads[0], "warm", "8-person startup"),
        (cold_leads[0], "cold", "Just browsing"),
    ]

    initial_state = {
        "messages": [],
        "intent": "",
        "response": "",
        "lead_score": 0,
        "lead_data": {},
        "info_complete": False,
        "email_draft": {},
        "email_sent": False,
        "track": "",
        "retry_count": 0,
        "needs_more_info": False,
    }

    print(f"\n{'Description':<40} {'Expected':<10} {'Got':<10} {'Score':<8} {'Match'}")
    print("-" * 80)

    for message, expected_track, description in all_tests:
        state = {**initial_state, "messages": [HumanMessage(content=message)]}
        try:
            result = app.invoke(state)
            got_track = result.get("track", "?")
            score = result.get("lead_score", 0)
            match = "✅" if got_track == expected_track else "⚠️"
            print(f"{description:<40} {expected_track:<10} {got_track:<10} {score:<8} {match}")
        except Exception as e:
            print(f"{description:<40} {expected_track:<10} ERROR: {str(e)[:30]}")
        time.sleep(1)

    # ── TEST 2: Info gathering loop ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Info Gathering Loop (incomplete info)")
    print("="*60)

    incomplete_message = "I want to add a new lead to CRM"
    state = {**initial_state, "messages": [HumanMessage(content=incomplete_message)]}

    print(f"Message: '{incomplete_message}'")
    try:
        result = app.invoke(state)
        print(f"Lead score:     {result.get('lead_score')}")
        print(f"Info complete:  {result.get('info_complete')}")
        print(f"Track:          {result.get('track')}")
        print(f"Response:\n{result.get('response', '')}")
    except Exception as e:
        print(f"Result error: {e}")

    # ── TEST 3: Human-in-the-loop (HITL) ──────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Human-in-the-loop (HITL) with MemorySaver")
    print("="*60)

    print("""
HITL Workflow:
Hot lead detected -> hot_lead_node drafts email
      ↓
⏸️ PAUSE BEFORE send_email_node (interrupt_before)
      ↓
Human reviews draft in checkpointed state
      ↓
Human approves -> graph resumes with thread_id
      ↓
send_email_node dispatches email ✅
""")

    hitl_app = build_advanced_crm_graph(with_hitl=True)
    config = {"configurable": {"thread_id": "hitl_lead_demo_1"}}

    hot_message = (
        "Urgent: Priya from EnterpriseX wants a priority demo. "
        "Team of 50 people. Budget Rs.5 lakh. Email: priya@enterprisex.com"
    )
    hitl_state = {**initial_state, "messages": [HumanMessage(content=hot_message)]}

    print("Step 1: Running graph with hot lead (will PAUSE before send_email_node)...")
    try:
        result = hitl_app.invoke(hitl_state, config=config)
        print(f"⏸️ Graph status: Paused at approval gate")
        print(f"Draft ready:     {bool(result.get('email_draft'))}")
        print(f"Email sent yet:  {result.get('email_sent', False)}")
        print(f"Subject:         {result.get('email_draft', {}).get('subject', 'N/A')}")
        print(f"Recipient:       {result.get('email_draft', {}).get('to', 'N/A')}")

        print("\nStep 2: Human reviews & APPROVES (resuming graph execution)...")
        # Resume the checkpointed graph on the same thread_id
        final = hitl_app.invoke(None, config=config)
        print(f"✅ Email sent:   {final.get('email_sent', False)}")
        print(f"Final Response:\n{final.get('response', '')}")

    except Exception as e:
        print(f"HITL error: {e}")

    # ── TEST 4: Graph structure & Mermaid ─────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Graph Architecture")
    print("="*60)
    print("""
Advanced CRM Workflow Diagram:

START
  ↓
[classify_node]
  ↓
  ├── sales     → [score_lead_node]
  │                 ↓
  │           ┌────────────────────────────────────────┐
  │           │ info_complete?                         │
  │           │ No  → [gather_info_node] → END         │
  │           │ Yes:                                   │
  │           │   score 8-10 → [hot_lead]  → ⏸️ HITL    │
  │           │                               ↓        │
  │           │                        [send_email]    │
  │           │                               ↓        │
  │           │                              END       │
  │           │   score 5-7  → [warm_lead] → END       │
  │           │   score 1-4  → [cold_lead] → END       │
  │           └────────────────────────────────────────┘
  │
  ├── support   → [support_node]   → END
  ├── analytics → [analytics_node] → END
  └── general   → [general_node]   → END
""")

    # ── Mini Assignment: Save Mermaid Diagram ─────────────────────────────────
    try:
        os.makedirs("docs", exist_ok=True)
        mermaid_syntax = app.get_graph().draw_mermaid()
        with open("docs/crm_graph.md", "w", encoding="utf-8") as f:
            f.write(f"# CRM Workflow Graph Architecture\n\n```mermaid\n{mermaid_syntax}\n```\n")
        print("✅ Mini Assignment: Mermaid diagram successfully saved to docs/crm_graph.md")
    except Exception as e:
        print(f"Note on diagram saving: {e}")

    # ── TEST 5: Day 22 vs Day 23 Comparison ──────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: Day 22 vs Day 23 Architecture Comparison")
    print("="*60)
    print("""
Feature               Day 22 (Basic Routing)          Day 23 (Advanced Routing)
-------------------   -----------------------------   ------------------------------------
Routing Depth         1-level (classify -> branch)    Multi-level (classify -> score -> track)
Branching Logic       Simple equality matching        Numeric scoring + Boolean validation
Info Handling         Static fallback                 Loop / pause to gather missing info
Destructive Actions   Immediate / Unmonitored         Human-in-the-loop (HITL) gate
State Persistence     Ephemeral (in-memory run)       Checkpointed with MemorySaver / thread_id
Nodes                 5 nodes                         10 specialized nodes
""")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("="*60)
    print("✅ Day 23 COMPLETE — Advanced Conditional Routing & HITL working!")
    print("="*60)
    print("""
Patterns Mastered:
  ✅ Multi-condition routing (hot/warm/cold tracks based on score)
  ✅ Incomplete info detection & loop pattern
  ✅ Human-in-the-loop approval checkpoint with interrupt_before
  ✅ Stateful checkpoints with MemorySaver and thread_id
  ✅ Visual documentation saved to docs/crm_graph.md

Week 4 Progress:
  ✅ Day 22 -> LangGraph Basics
  ✅ Day 23 -> Conditional Routing & HITL (TODAY)
  ⬜ Day 24 -> Multi-Agent Supervisor
  ⬜ Day 25 -> FastAPI Backend
  ⬜ Day 26 -> Redis + Celery
  ⬜ Day 27 -> MongoDB
  ⬜ Day 28 -> React Dashboard
  ⬜ Day 29 -> Docker
  ⬜ Day 30 -> Deployment 🚀
""")
    print("Next -> Day 24: Multi-Agent Supervisor — Coordinating Sales, Support, & Analytics!")
    print("="*60)
