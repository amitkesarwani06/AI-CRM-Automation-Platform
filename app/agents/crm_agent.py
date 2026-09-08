import os
import sys
import logging
import threading

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-20b"


# ── Build RAG Tool ────────────────────────────────────────────────────────────
def build_rag_tool() -> BaseTool:
    """
    Builds the Week 2 RAG pipeline and wraps it as a LangChain tool.
    Agent decides WHEN to call this — vs answering from memory or other tools.
    Built ONCE here (closure) — not rebuilt on every tool call.
    """
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

    company_knowledge = """
Refund Policy
We offer a 30-day money-back guarantee on all plans.
To request a refund contact support@crmplatform.com.
Refunds are processed within 5-7 business days.
No questions asked for cancellations within first 14 days.

Pricing Plans
Starter Plan Rs.999 per month for up to 3 users and 500 leads.
Growth Plan Rs.2999 per month for up to 15 users with AI features.
Enterprise Plan has custom pricing with dedicated support and unlimited users.
Annual billing gives 20 percent discount on all plans.

Support Hours
Available Monday to Saturday 9am to 7pm IST.
Average response time under 2 hours during business hours.
Emergency support 24x7 for Enterprise customers.
Contact support@crmplatform.com or call 1800-123-4567.

Getting Started
First 14 days are free with no credit card required.
Import leads from CSV in Settings then Import Data.
Book onboarding at calendly.com/crmplatform/onboarding.

Features
Lead management with AI scoring and follow-up reminders.
Email automation with customizable templates.
WhatsApp integration for direct messaging to leads.
Analytics dashboard with real-time sales reports.
"""

    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks = chunker.split_text(company_knowledge, source="company_kb")
    store = UnifiedVectorStore(store_type=VectorStoreType.FAISS, collection_name="crm_agent_kb")
    store.add_documents(chunks)
    retriever = RetrieverService(store).get_retriever(RetrieverType.SIMILARITY, k=3)
    rag_chain = RAGChain(retriever, model=GROQ_MODEL)

    class KBSearchInput(BaseModel):
        query: str = Field(description="Question to search in the CRM knowledge base")

    @tool("search_crm_knowledge_base", args_schema=KBSearchInput)
    def search_crm_knowledge_base(query: str) -> str:
        """
        Search the CRM knowledge base for accurate information about:
        - Refund and cancellation policy
        - Pricing plans and features
        - Support hours and contact details
        - Getting started guides
        - Product features

        Use this tool for ANY product or policy question.
        Always prefer this over guessing from memory.
        """
        try:
            return rag_chain.invoke(query)
        except Exception as e:
            return f"KB search error: {str(e)}"

    return search_crm_knowledge_base


# ── Full CRM Agent ────────────────────────────────────────────────────────────
class FullCRMAgent:
    """
    Week 3 Capstone — Complete AI CRM Agent.

    Combines:
      Week 1 Memory  → remembers conversation history across turns
      Week 2 RAG     → answers from company knowledge base
      Week 3 ReAct   → autonomous Thought -> Action -> Observation loop

    Usage:
        agent = FullCRMAgent()
        response = agent.chat("Add Rahul from TechCorp as a lead")
        response = agent.chat("What company does he work at?")  # remembers!
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        max_iterations: int = 8,
        verbose: bool = True,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose

        # ── Memory: simple list of (human, AI) turns ──────────────────────────
        self._history: list[dict] = []   # {"role": "human"/"ai", "content": str}

        # ── LLM with tools bound ──────────────────────────────────────────────
        from app.tools import ALL_CRM_TOOLS
        rag_tool = build_rag_tool()

        # Replace basic search_knowledge_base with RAG-powered version
        self.tools = [rag_tool] + [
            t for t in ALL_CRM_TOOLS
            if t.name != "search_knowledge_base"
        ]
        self.tool_map = {t.name: t for t in self.tools}

        llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.llm_with_tools = llm.bind_tools(self.tools)

        logger.info(
            f"FullCRMAgent 'Aria' ready | "
            f"model={model} | tools={len(self.tools)} | memory=ON"
        )

    # ── Memory helpers ────────────────────────────────────────────────────────
    def _build_system_prompt(self) -> str:
        """Build system prompt with current conversation history injected."""
        history_text = ""
        if self._history:
            lines = []
            for turn in self._history[-6:]:   # last 6 turns to keep context short
                prefix = "User" if turn["role"] == "human" else "Aria"
                lines.append(f"{prefix}: {turn['content'][:200]}")
            history_text = "\n".join(lines)

        return f"""You are Aria, an AI CRM assistant for our sales platform.

You help sales teams by:
- Managing leads (create, find, update)
- Answering product questions from our knowledge base
- Providing pricing information
- Tracking conversation context

CONVERSATION HISTORY (use this to remember earlier context):
{history_text if history_text else "No prior conversation yet."}

Think step by step. Use tools when needed.
When answering from conversation history, you do NOT need to call a tool.
Be concise and professional.
"""

    def _save_turn(self, human_msg: str, ai_response: str) -> None:
        """Save turn to memory."""
        self._history.append({"role": "human", "content": human_msg})
        self._history.append({"role": "ai", "content": ai_response})

    # ── Core ReAct loop ───────────────────────────────────────────────────────
    def _react_loop(self, user_message: str) -> tuple[str, list[dict]]:
        """
        ReAct loop with memory-injected system prompt.
        Each turn gets updated conversation history in the system message.
        """
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=user_message),
        ]
        steps = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n> Iteration {iteration + 1}")

            ai_message = self.llm_with_tools.invoke(messages)
            messages.append(ai_message)

            # No tool calls → Final Answer
            if not ai_message.tool_calls:
                if self.verbose:
                    print(f"> Final Answer (after {iteration + 1} iteration(s))")
                return ai_message.content, steps

            # Execute tools
            if self.verbose:
                print(f"  Tools: {[tc['name'] for tc in ai_message.tool_calls]}")

            for tool_call in ai_message.tool_calls:
                tool_name    = tool_call["name"]
                tool_args    = tool_call["args"]
                tool_call_id = tool_call["id"]

                if tool_name in self.tool_map:
                    try:
                        tool_result = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Error in {tool_name}: {str(e)}"
                else:
                    tool_result = f"Tool '{tool_name}' not found."

                if self.verbose:
                    print(f"  [{tool_name}] -> {str(tool_result)[:80]}...")

                steps.append({
                    "tool":   tool_name,
                    "input":  tool_args,
                    "output": str(tool_result),
                })

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )

        return "Agent reached max iterations without completing the task.", steps

    # ── Public chat methods ───────────────────────────────────────────────────
    def chat(self, message: str) -> str:
        """
        Multi-turn chat with memory.
        Agent remembers everything said in this session.

        Usage:
            answer = agent.chat("Add Rahul from TechCorp as a lead")
            answer = agent.chat("What company is he from?")  # remembers!
        """
        if self.verbose:
            print(f"\n> Aria thinking: '{message[:60]}'")
        try:
            answer, _ = self._react_loop(message)
            self._save_turn(message, answer)
            return answer
        except Exception as e:
            logger.error(f"Agent error: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            self._save_turn(message, error_msg)
            return error_msg

    def chat_verbose(self, message: str) -> dict:
        """Chat and return full tool call details for debugging."""
        if self.verbose:
            print(f"\n> Aria thinking: '{message[:60]}'")
        try:
            answer, steps = self._react_loop(message)
            self._save_turn(message, answer)
            return {
                "message":    message,
                "answer":     answer,
                "tools_used": [s["tool"] for s in steps],
                "iterations": len(steps),
                "memory_turns": len(self._history),
                "steps":      steps,
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return {
                "message":    message,
                "answer":     error_msg,
                "tools_used": [],
                "iterations": 0,
                "memory_turns": len(self._history),
                "steps":      [],
            }

    # ── Memory methods ────────────────────────────────────────────────────────
    def get_memory(self) -> str:
        """Return current conversation history as formatted string."""
        if not self._history:
            return "No conversation history yet."
        lines = []
        for turn in self._history:
            prefix = "User" if turn["role"] == "human" else "Aria"
            lines.append(f"{prefix}: {turn['content'][:200]}")
        return "\n".join(lines)

    def reset_memory(self) -> None:
        """Clear conversation history — fresh start."""
        self._history = []
        logger.info("Memory cleared")
        print("Memory cleared — fresh start!")

    def get_tool_list(self) -> list[str]:
        """Return list of available tool names."""
        return [t.name for t in self.tools]

    # ── Mini Assignment: session summary ──────────────────────────────────────
    def summarize_session(self) -> str:
        """
        Ask the LLM to summarize what was accomplished this session.
        Useful for sales reps to read at the start of their next session.
        """
        history = self.get_memory()
        if history == "No conversation history yet.":
            return "No conversation history to summarize."

        from langchain_groq import ChatGroq
        llm = ChatGroq(model=self.model, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        summary = llm.invoke(
            f"Summarize what was accomplished in this CRM session in 3-5 bullet points:\n\n{history}"
        )
        return summary.content


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    print("\n" + "="*60)
    print("DAY 18 — Full CRM Agent (Week 3 Capstone)")
    print("Memory + RAG + ReAct Tools Combined")
    print("="*60)

    print("\n[SETUP] Building FullCRMAgent 'Aria'...")
    agent = FullCRMAgent(model=GROQ_MODEL, max_iterations=8, verbose=True)
    print(f"Tools: {agent.get_tool_list()}\n")

    # ── TEST 1: RAG question ──────────────────────────────────────────────────
    print("="*60)
    print("TEST 1: Knowledge Base Question (RAG)")
    print("="*60)

    result = agent.chat_verbose("What is your refund policy?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")

    # ── TEST 2: Create lead ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Create Lead (CRM Tool)")
    print("="*60)

    result = agent.chat_verbose(
        "Add a new lead: Rahul Sharma from TechCorp Solutions, "
        "email rahul@techcorp.com, phone 9823456710, interested in Growth plan"
    )
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {result['tools_used']}")

    # ── TEST 3: Memory test ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Memory — agent remembers Rahul")
    print("="*60)

    result = agent.chat_verbose("What company does Rahul work at?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {result['tools_used']} (should be empty — from memory!)")
    print(f"Memory turns: {result['memory_turns']}")

    # ── TEST 4: RAG + Action combined ────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: RAG + Action combined")
    print("="*60)

    result = agent.chat_verbose(
        "Rahul asked about the Growth plan pricing. "
        "Tell me the details and update his status to contacted."
    )
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")

    # ── TEST 5: Multi-turn conversation ───────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: Full Multi-Turn Conversation (6 turns)")
    print("="*60)

    conversation = [
        "Good morning! I have a new lead from yesterday's conference.",
        "Her name is Priya Mehta from StartupXYZ, email priya@startupxyz.com",
        "She's interested in the Starter plan. What does it include?",
        "Great, add her as a lead with those details.",
        "What leads do we have now?",
        "Update Priya's status to qualified.",
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\nTurn {i}: {message}")
        answer = agent.chat(message)
        print(f"Aria: {answer[:150]}...")

    # ── TEST 6: Memory inspection ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 6: Memory contents")
    print("="*60)

    memory = agent.get_memory()
    print(f"Total memory turns: {len(agent._history)}")
    print(f"\nFirst 300 chars:\n{memory[:300]}...")

    # ── TEST 7: Session summary (Mini Assignment) ─────────────────────────────
    print("\n" + "="*60)
    print("TEST 7: Session Summary (Mini Assignment)")
    print("="*60)

    summary = agent.summarize_session()
    print(f"Session Summary:\n{summary}")

    # ── TEST 8: Memory reset ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 8: Reset memory — fresh start")
    print("="*60)

    agent.reset_memory()
    result = agent.chat_verbose("What company does Rahul work at?")
    print(f"After reset answer: {result['answer'][:100]}...")
    print(f"Tools used: {result['tools_used']} (should use get_lead now — memory cleared)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("WEEK 3 COMPLETE — Full CRM Agent!")
    print("="*60)
    print("""
Day 15: 7 Custom CRM Tools
Day 16: Tool Binding + Calling
Day 17: ReAct Agent (autonomous loop)
Day 18: Full CRM Agent = Memory + RAG + ReAct  TODAY

Agent capabilities:
  Remembers conversation history
  Answers from knowledge base (RAG)
  Creates and manages leads
  Updates lead status
  Multi-turn autonomous conversations
""")
    print("Day 15 to Day 18 complete!")
    print("Next -> Day 19: Email Automation Tool")
    print("="*60)
