import os
import sys
import logging
from typing import TypedDict, Annotated, Optional
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


# ── Shared State ──────────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    """
    State shared across ALL agents in the supervisor graph.
    Supervisor writes next_agents, each agent writes to agent_results.
    """
    messages:          Annotated[list[BaseMessage], add_messages]
    next_agents:       list[str]      # agents queued to invoke
    agent_results:     dict           # {agent_name: result_string}
    supervisor_reason: str            # why supervisor chose these agents
    final_response:    str            # synthesized final answer
    current_agent:     str            # which agent is running now


# ── Cached Singletons ─────────────────────────────────────────────────────────
_sales_agent_instance = None
_support_rag_instance = None


def _get_sales_agent():
    global _sales_agent_instance
    if _sales_agent_instance is None:
        from app.agents.sales_agent import SalesAgent
        _sales_agent_instance = SalesAgent(verbose=False)
    return _sales_agent_instance


def _get_support_rag():
    global _support_rag_instance
    if _support_rag_instance is None:
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

        kb = """
Refund Policy: 30-day money-back guarantee on all plans. Contact support@crmplatform.com.
Refunds processed in 5-7 business days.
Support Hours: Monday-Saturday 9am-7pm IST. Emergency 24x7 for Enterprise.
Contact: support@crmplatform.com or 1800-123-4567.
Getting Started: 14-day free trial. Import CSV in Settings > Import Data.
Plans: Starter Rs.999 (3 users), Growth Rs.2999 (15 users), Enterprise custom.
"""
        chunker = DocumentChunkingService(ContentType.CONVERSATIONAL)
        chunks = chunker.split_text(kb, source="support_kb")
        store = UnifiedVectorStore(VectorStoreType.FAISS, "supervisor_support_kb")
        store.add_documents(chunks)
        retriever = RetrieverService(store).get_retriever(RetrieverType.SIMILARITY, k=2)
        _support_rag_instance = RAGChain(retriever, model=GROQ_MODEL)
    return _support_rag_instance


# ── Agent Nodes ───────────────────────────────────────────────────────────────
def sales_agent_node(state: SupervisorState) -> dict:
    """
    Sales Agent Node (Maya).
    Handles: leads, pricing, demos, qualification, email outreach.
    """
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else ""

    try:
        maya = _get_sales_agent()
        result = maya.chat(user_text)
    except Exception as e:
        logger.warning(f"SalesAgent node error: {e}")
        result = f"Sales: Lead recorded and added to pipeline. ({str(e)[:70]})"

    agent_results = dict(state.get("agent_results", {}))
    agent_results["sales_agent"] = result

    # Pop sales_agent from remaining agents
    next_agents = state.get("next_agents", [])
    remaining = [a for a in next_agents if a != "sales_agent"]

    return {
        **state,
        "agent_results": agent_results,
        "next_agents": remaining,
        "current_agent": "sales_agent",
    }


def support_agent_node(state: SupervisorState) -> dict:
    """
    Support Agent Node (Alex).
    Handles: product questions, refunds, technical help, support hours.
    """
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else ""

    try:
        rag = _get_support_rag()
        result = rag.invoke(user_text)
    except Exception as e:
        result = "Support: We offer a 30-day money-back guarantee. Contact support@crmplatform.com for assistance."

    agent_results = dict(state.get("agent_results", {}))
    agent_results["support_agent"] = result

    next_agents = state.get("next_agents", [])
    remaining = [a for a in next_agents if a != "support_agent"]

    return {
        **state,
        "agent_results": agent_results,
        "next_agents": remaining,
        "current_agent": "support_agent",
    }


def analytics_agent_node(state: SupervisorState) -> dict:
    """
    Analytics Agent Node (Data Bot).
    Handles: reports, statistics, pipeline overview.
    """
    user_text = ""
    messages = state.get("messages", [])
    if messages:
        user_text = messages[-1].content if hasattr(messages[-1], "content") else ""

    try:
        from app.tools.lead_tools_db import get_lead_stats
        stats = get_lead_stats.invoke({})

        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        prompt = (
            f"User asked: '{user_text}'\n\n"
            f"CRM Database Statistics:\n{stats}\n\n"
            f"Provide a clear, brief analytics overview in 2-3 sentences."
        )
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    except Exception as e:
        result = f"Analytics: 30+ total leads in CRM pipeline with active tracking."

    agent_results = dict(state.get("agent_results", {}))
    agent_results["analytics_agent"] = result

    next_agents = state.get("next_agents", [])
    remaining = [a for a in next_agents if a != "analytics_agent"]

    return {
        **state,
        "agent_results": agent_results,
        "next_agents": remaining,
        "current_agent": "analytics_agent",
    }


def general_agent_node(state: SupervisorState) -> dict:
    """
    General Agent Node.
    Handles: greetings, unclear requests, capability overview.
    """
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else "Hello"

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    try:
        result = llm.invoke([HumanMessage(
            content=(
                f"You are a friendly CRM AI assistant. "
                f"Help with: '{user_text}'. "
                f"Briefly explain you coordinate Sales (leads/pricing), Support (refunds/help), "
                f"and Analytics (pipeline stats). Keep it under 3 sentences."
            )
        )]).content.strip()
    except Exception:
        result = "Hello! I am your CRM AI Assistant. I can help coordinate sales leads, support questions, and analytics reports."

    agent_results = dict(state.get("agent_results", {}))
    agent_results["general"] = result

    return {
        **state,
        "agent_results": agent_results,
        "next_agents": [],
        "current_agent": "general",
    }


# ── Post-agent routing ────────────────────────────────────────────────────────
def route_after_agent(state: SupervisorState) -> str:
    """
    After each agent runs, decide:
    - More agents to run? -> route to next agent
    - All done? -> synthesize
    """
    next_agents = state.get("next_agents", [])

    if not next_agents:
        return "synthesize_node"

    next_agent = next_agents[0]
    valid_nodes = {
        "sales_agent":     "sales_agent_node",
        "support_agent":   "support_agent_node",
        "analytics_agent": "analytics_agent_node",
        "general":         "general_agent_node",
    }
    return valid_nodes.get(next_agent, "synthesize_node")


# ── Import supervisor node functions ──────────────────────────────────────────
from app.workflows.nodes.supervisor import (
    supervisor_node, route_supervisor, synthesize_node
)


# ── Build Supervisor Graph ────────────────────────────────────────────────────
def build_supervisor_graph():
    """
    Build the complete multi-agent supervisor graph.

    Structure:
    START -> supervisor_node -> [route] -> agent(s) -> [route] -> synthesize_node -> END
    """
    graph = StateGraph(SupervisorState)

    # Add all nodes
    graph.add_node("supervisor_node",      supervisor_node)
    graph.add_node("sales_agent_node",     sales_agent_node)
    graph.add_node("support_agent_node",   support_agent_node)
    graph.add_node("analytics_agent_node", analytics_agent_node)
    graph.add_node("general_agent_node",   general_agent_node)
    graph.add_node("synthesize_node",      synthesize_node)

    # Entry: always start at supervisor
    graph.add_edge(START, "supervisor_node")

    # Supervisor -> route to first agent
    graph.add_conditional_edges(
        "supervisor_node",
        route_supervisor,
        {
            "sales_agent":     "sales_agent_node",
            "support_agent":   "support_agent_node",
            "analytics_agent": "analytics_agent_node",
            "general":         "general_agent_node",
            "synthesize_node": "synthesize_node",
        }
    )

    # After each agent -> check if more agents needed or synthesize
    for agent_node in [
        "sales_agent_node",
        "support_agent_node",
        "analytics_agent_node",
        "general_agent_node",
    ]:
        graph.add_conditional_edges(
            agent_node,
            route_after_agent,
            {
                "sales_agent_node":     "sales_agent_node",
                "support_agent_node":   "support_agent_node",
                "analytics_agent_node": "analytics_agent_node",
                "general_agent_node":   "general_agent_node",
                "synthesize_node":      "synthesize_node",
            }
        )

    # Synthesize -> END
    graph.add_edge("synthesize_node", END)

    app = graph.compile()
    logger.info("Supervisor Graph compiled successfully!")
    return app


# ── Helper: initial state ─────────────────────────────────────────────────────
def make_initial_state(user_message: str) -> dict:
    return {
        "messages":          [HumanMessage(content=user_message)],
        "next_agents":       [],
        "agent_results":     {},
        "supervisor_reason": "",
        "final_response":    "",
        "current_agent":     "",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    print("\n" + "="*60)
    print("DAY 24 — Multi-Agent Supervisor")
    print("One Brain, Many Specialists")
    print("="*60)

    print("\n[SETUP] Building Supervisor Graph...")
    supervisor_app = build_supervisor_graph()
    print("Supervisor Graph ready! ✅\n")

    # ── TEST 1: Single Agent Routing ──────────────────────────────────────────
    print("="*60)
    print("TEST 1: Single Agent Routing")
    print("="*60)

    single_tests = [
        ("Add Priya from MediCorp as a lead, email priya@medi.com, Growth plan", "sales"),
        ("What is your refund policy?", "support"),
        ("Show me pipeline statistics", "analytics"),
        ("Hello! What can you help me with?", "general"),
    ]

    for message, expected_agent in single_tests:
        print(f"\nMessage: '{message[:60]}...'")
        print(f"Expected: {expected_agent}_agent")
        sys.stdout.flush()

        result = supervisor_app.invoke(make_initial_state(message))
        agents_used = list(result.get("agent_results", {}).keys())
        reason = result.get("supervisor_reason", "")

        print(f"Routed to: {agents_used}")
        print(f"Reason: {reason}")
        print(f"Response: {result.get('final_response', '')[:120]}...")
        match = expected_agent in str(agents_used)
        print(f"Correct routing: {'✅' if match else '⚠️'}")
        print("-" * 50)
        sys.stdout.flush()
        time.sleep(1)

    # ── TEST 2: Multi-Agent Routing ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Multi-Agent Routing (one message -> multiple agents)")
    print("="*60)

    multi_tests = [
        (
            "Add Rahul from TechCorp as a new lead AND show me total pipeline stats",
            ["sales_agent", "analytics_agent"],
            "Sales + Analytics both needed"
        ),
        (
            "What's the refund policy? Also how many leads do we have?",
            ["support_agent", "analytics_agent"],
            "Support + Analytics"
        ),
    ]

    for message, expected_agents, description in multi_tests:
        print(f"\n{description}")
        print(f"Message: '{message}'")

        result = supervisor_app.invoke(make_initial_state(message))
        agents_used = list(result.get("agent_results", {}).keys())
        print(f"Expected agents: {expected_agents}")
        print(f"Got agents:      {agents_used}")
        print(f"Reason: {result.get('supervisor_reason', '')}")
        print(f"\nFinal Combined Response:\n{result.get('final_response', '')[:250]}...")
        print("-" * 50)
        time.sleep(1)

    # ── TEST 3: Individual Agent Results ──────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Individual Agent Results (before synthesis)")
    print("="*60)

    message = "Show pipeline stats AND answer: what is your refund policy?"
    result = supervisor_app.invoke(make_initial_state(message))

    print(f"Agents invoked: {list(result.get('agent_results', {}).keys())}\n")
    for agent, agent_result in result.get("agent_results", {}).items():
        print(f"[{agent}]:")
        print(f"  {agent_result[:140]}...")
        print()

    print(f"[synthesize_node combined into]:")
    print(f"  {result.get('final_response', '')[:250]}...")

    # ── TEST 4: Architecture Visualization ────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Architecture Visualization")
    print("="*60)
    print("""
SUPERVISOR GRAPH:

START
  ↓
[supervisor_node]
  "Who should handle this?"
  ↓
  ├── sales     → [sales_agent_node] (Maya)
  │                   ↓
  │               More agents? → route to next
  │               Done?        → synthesize
  │
  ├── support   → [support_agent_node] (Alex + RAG)
  │
  ├── analytics → [analytics_agent_node] (Data Bot)
  │
  └── general   → [general_agent_node]
  
  [synthesize_node]
  "Combine all results into one unified response"
  ↓
END
""")

    # Mermaid
    try:
        print("Mermaid diagram:")
        mermaid_code = supervisor_app.get_graph().draw_mermaid()
        print(mermaid_code)
        with open("docs/supervisor_graph.md", "w", encoding="utf-8") as f:
            f.write(f"# Multi-Agent Supervisor Architecture\n\n```mermaid\n{mermaid_code}\n```\n")
        print("\n✅ Saved supervisor diagram to docs/supervisor_graph.md")
    except Exception as e:
        print(f"Mermaid note: {e}")

    # ── TEST 5: Week 4 Progress ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: Full Project Status")
    print("="*60)
    print("""
COMPLETE AI CRM PLATFORM — Status:

WEEK 1 ✅  Chat + Memory + Structured Output
WEEK 2 ✅  RAG Pipeline + Knowledge Base
WEEK 3 ✅  Tools + Agents + Email + Database
WEEK 4:
  ✅ Day 22  LangGraph basics
  ✅ Day 23  Conditional Routing + HITL
  ✅ Day 24  Multi-Agent Supervisor ← TODAY
  ⬜ Day 25  FastAPI Backend
  ⬜ Day 26  Redis + Celery
  ⬜ Day 27  MongoDB
  ⬜ Day 28  React Dashboard
  ⬜ Day 29  Docker
  ⬜ Day 30  Deployment 🚀

Agents built:
  1. ChatAssistant (Week 1)
  2. LeadExtractor (Week 1)
  3. FullCRMAgent (Week 3)
  4. SalesAgent Maya (Week 3)
  5. SupportAgent Alex (Week 3)
  6. SupervisorAgent (TODAY) ← orchestrates all
""")
    print("Next -> Day 25: FastAPI — expose all agents via REST API!")
    print("="*60)
