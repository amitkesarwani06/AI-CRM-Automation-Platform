import os
import sys
import logging
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


# ── State Definition ──────────────────────────────────────────────────────────
class CRMState(TypedDict):
    """
    Shared state that flows through every node in the graph.

    Every node reads from this state and returns an updated version.
    LangGraph merges the returned dict back into the state automatically.

    messages: full conversation history
              Annotated[list, add_messages] -> LangGraph auto-appends
              new messages instead of replacing the list

    intent:   classified intent (sales/support/analytics/general/escalation)

    response: final response to show the user
    """
    messages:  Annotated[list[BaseMessage], add_messages]
    intent:    str
    response:  str


# ── Node Functions ────────────────────────────────────────────────────────────
def classify_node(state: CRMState) -> dict:
    """
    Node: Classify user intent.
    Reads latest message -> sets intent in state.
    """
    from app.workflows.nodes.classifier import classify_intent
    return classify_intent(state)


def sales_node(state: CRMState) -> dict:
    """
    Node: Handle sales-related requests.
    Uses SalesAgent (Maya) from Day 21.
    """
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else ""

    try:
        from app.agents.sales_agent import SalesAgent
        maya = SalesAgent(verbose=False)
        response = maya.chat(user_text)
    except Exception as e:
        response = f"Sales Agent: I can help with leads and pricing. ({str(e)[:100]})"

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def support_node(state: CRMState) -> dict:
    """
    Node: Handle support-related requests.
    Uses RAG knowledge base for accurate answers.
    """
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

Getting Started
First 14 days free, no credit card required.
Import leads from CSV in Settings then Import Data.
"""
        chunker = DocumentChunkingService(ContentType.CONVERSATIONAL)
        chunks = chunker.split_text(support_knowledge, source="support_kb")
        store = UnifiedVectorStore(VectorStoreType.FAISS, "support_graph_kb")
        store.add_documents(chunks)
        retriever = RetrieverService(store).get_retriever(RetrieverType.SIMILARITY, k=2)
        rag = RAGChain(retriever, model=GROQ_MODEL)
        response = rag.invoke(user_text)

    except Exception as e:
        response = (
            f"Support: For help with this, please contact "
            f"support@crmplatform.com. ({str(e)[:50]})"
        )

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def analytics_node(state: CRMState) -> dict:
    """
    Node: Handle analytics requests.
    Returns lead statistics from database.
    """
    try:
        from app.tools.lead_tools_db import get_lead_stats
        stats = get_lead_stats.invoke({})
        response = f"📊 Analytics Report:\n{stats}"
    except Exception as e:
        response = (
            "Analytics: I can show you pipeline stats, lead counts, "
            f"and performance metrics. (DB: {str(e)[:50]})"
        )

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def general_node(state: CRMState) -> dict:
    """
    Node: Handle general/unclear requests.
    Friendly fallback with capability overview.
    """
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else ""

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = (
        f"You are a friendly CRM assistant. "
        f"Respond helpfully to: '{user_text}'\n\n"
        f"You can help with: sales (leads, pricing, demos), "
        f"support (refunds, technical help), and analytics (reports, stats). "
        f"Keep response under 3 sentences."
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        response = "Hello! I am your CRM Assistant. How can I help you with sales, support, or pipeline analytics today?"

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


# ── Route Function ────────────────────────────────────────────────────────────
def route_by_intent(state: CRMState) -> str:
    """
    Conditional edge function.
    Called by LangGraph after classify_node.
    Returns the NAME of the next node to run.
    """
    intent = state.get("intent", "general")
    routes = {
        "sales":     "sales_node",
        "support":   "support_node",
        "analytics": "analytics_node",
        "general":   "general_node",
    }
    return routes.get(intent, "general_node")


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_crm_graph():
    """
    Build and compile the CRM routing graph.

    Graph structure:
    START -> classify_node -> [conditional] -> sales_node     -> END
                                           -> support_node   -> END
                                           -> analytics_node -> END
                                           -> general_node   -> END
    """
    graph = StateGraph(CRMState)

    # Add nodes
    graph.add_node("classify_node",   classify_node)
    graph.add_node("sales_node",      sales_node)
    graph.add_node("support_node",    support_node)
    graph.add_node("analytics_node",  analytics_node)
    graph.add_node("general_node",    general_node)

    # Entry point
    graph.add_edge(START, "classify_node")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_node",    # from this node
        route_by_intent,    # call this function to decide
        {                   # map return value -> next node
            "sales_node":      "sales_node",
            "support_node":    "support_node",
            "analytics_node":  "analytics_node",
            "general_node":    "general_node",
        }
    )

    # All terminal nodes lead to END
    graph.add_edge("sales_node",      END)
    graph.add_edge("support_node",    END)
    graph.add_edge("analytics_node",  END)
    graph.add_edge("general_node",    END)

    # Compile
    app = graph.compile()
    logger.info("CRM Graph compiled successfully")
    return app


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    print("\n" + "="*60)
    print("DAY 22 — LangGraph: State, Nodes, Edges")
    print("Week 4 Begins!")
    print("="*60)

    # Build graph
    print("\n[SETUP] Building CRM Graph...")
    crm_app = build_crm_graph()
    print("Graph compiled! ✅\n")

    # ── TEST 1: Graph Structure ───────────────────────────────────────────────
    print("="*60)
    print("TEST 1: Graph Structure")
    print("="*60)
    print("""
CRM Routing Graph:

START
  ↓
[classify_node] ← reads message, sets intent
  ↓
  ├── intent=sales     → [sales_node]     → END
  ├── intent=support   → [support_node]   → END
  ├── intent=analytics → [analytics_node] → END
  └── intent=general   → [general_node]   → END
""")

    # ── TEST 2: Intent Classification ────────────────────────────────────────
    print("="*60)
    print("TEST 2: Routing Different Message Types")
    print("="*60)

    test_messages = [
        ("I have a new lead from TechCorp", "sales"),
        ("How do I get a refund?", "support"),
        ("Show me our pipeline statistics", "analytics"),
        ("Hello, how are you?", "general"),
        ("I want to add Rahul as a prospect", "sales"),
        ("What are your support hours?", "support"),
        ("How many leads do we have?", "analytics"),
    ]

    print(f"{'Message':<45} {'Expected':<12} {'Got':<12} {'Match'}")
    print("-" * 80)

    correct = 0
    from app.workflows.nodes.classifier import classify_intent
    for message, expected_intent in test_messages:
        state = {"messages": [HumanMessage(content=message)], "intent": "", "response": ""}
        result = classify_intent(state)
        got_intent = result.get("intent", "?")
        match = "✅" if got_intent == expected_intent else "⚠️"
        if got_intent == expected_intent:
            correct += 1
        print(f"{message:<45} {expected_intent:<12} {got_intent:<12} {match}")
        time.sleep(1)

    print(f"\nAccuracy: {correct}/{len(test_messages)} ({correct*100//len(test_messages)}%)")

    # ── TEST 3: Full Graph Run ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Full Graph — end to end")
    print("="*60)

    full_test_cases = [
        "What is your refund policy?",
        "Show me the lead statistics",
        "Hello, what can you help me with?",
    ]

    for message in full_test_cases:
        print(f"\nMessage: '{message}'")
        try:
            result = crm_app.invoke({
                "messages": [HumanMessage(content=message)],
                "intent": "",
                "response": "",
            })
            print(f"Intent:   {result.get('intent', '?')}")
            print(f"Response:\n{result.get('response', '')}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)
        time.sleep(1)

    # ── TEST 4: State Flow ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: State Flow — what happens inside")
    print("="*60)
    print("""
State starts as:
  {
    "messages": [HumanMessage("What is your refund policy?")],
    "intent":   "",
    "response": ""
  }

After classify_node:
  {
    "messages": [HumanMessage(...)],
    "intent":   "support",    ← UPDATED
    "response": ""
  }

Conditional edge reads intent="support" → routes to support_node

After support_node:
  {
    "messages": [HumanMessage(...), AIMessage("30-day refund...")],
    "intent":   "support",
    "response": "We offer a 30-day money-back guarantee..."  ← UPDATED
  }

Graph returns final state → response shown to user
""")

    # ── TEST 5: LangGraph vs ReAct ────────────────────────────────────────────
    print("="*60)
    print("TEST 5: LangGraph vs ReAct — key differences")
    print("="*60)
    print("""
ReAct Agent (Week 3):         LangGraph (Week 4):
─────────────────────         ──────────────────
One agent                     Multiple specialized nodes
Linear loop                   Graph with branches
No routing                    Conditional routing
No state schema               TypedDict state schema
Hard to visualize             Visualizable graph
Hard to add humans            Easy human-in-the-loop
Tool-based control flow       Graph-based control flow
""")

    # ── TEST 6: Mini Assignment (Mermaid Diagram) ─────────────────────────────
    print("="*60)
    print("TEST 6: Mini Assignment — Graph Mermaid Diagram")
    print("="*60)
    try:
        mermaid_syntax = crm_app.get_graph().draw_mermaid()
        print(mermaid_syntax)
    except Exception as e:
        print(f"Mermaid generation note: {e}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ Day 22 COMPLETE — LangGraph basics working!")
    print("="*60)
    print("""
Key concepts learned:
  State     -> TypedDict shared across all nodes
  Nodes     -> functions that transform state
  Edges     -> connections (simple + conditional)
  START     -> graph entry point
  END       -> graph exit point
  compile() -> turns graph into runnable app

Week 4 Progress:
  ✅ Day 22 -> LangGraph basics (State, Nodes, Edges)
  ⬜ Day 23 -> Conditional Routing (advanced)
  ⬜ Day 24 -> Multi-Agent + Supervisor
  ⬜ Day 25 -> FastAPI Backend
  ⬜ Day 28 -> React Dashboard
  ⬜ Day 29 -> Docker
  ⬜ Day 30 -> Deployment
""")
    print("Next -> Day 23: Conditional Routing — advanced graph patterns!")
    print("="*60)
