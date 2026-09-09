import os
import sys
import logging

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-20b"


# ── Build Sales Knowledge Base Tool ──────────────────────────────────────────
def _build_sales_kb_tool() -> BaseTool:
    """
    RAG tool built once (closure) for the Sales Agent.
    Contains pricing, features, and plan qualification info.
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
Pricing Plans
Starter Plan Rs.999 per month for up to 3 users and 500 leads, basic analytics.
Growth Plan Rs.2999 per month for up to 15 users with AI features including
lead scoring, email automation, and WhatsApp integration, unlimited leads.
Enterprise Plan has custom pricing with dedicated support, unlimited users,
custom integrations, and 24x7 emergency support.
Annual billing gives 20 percent discount on all plans.

Team Size to Plan Guide
1 to 3 users recommend Starter Plan at Rs.999 per month.
4 to 15 users recommend Growth Plan at Rs.2999 per month.
More than 15 users recommend Enterprise Plan with custom pricing.

Features
Lead management with AI scoring and automatic follow-up reminders.
Email automation with customizable templates and scheduling.
WhatsApp integration for direct messaging to leads.
Analytics dashboard with real-time sales reports.
Free trial of 14 days with no credit card required.
Import leads from CSV in Settings then Import Data.
Onboarding call available at calendly.com/crmplatform/onboarding.

Refund Policy
30-day money-back guarantee on all plans. Contact support@crmplatform.com.
Refunds processed in 5-7 business days.
"""

    chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
    chunks  = chunker.split_text(company_knowledge, source="sales_kb")
    store   = UnifiedVectorStore(store_type=VectorStoreType.FAISS, collection_name="sales_agent_kb")
    store.add_documents(chunks)
    retriever = RetrieverService(store).get_retriever(RetrieverType.SIMILARITY, k=3)
    rag_chain = RAGChain(retriever, model=GROQ_MODEL)

    class SalesKBInput(BaseModel):
        query: str = Field(description="Question about CRM features, pricing, or plans")

    @tool("search_sales_knowledge", args_schema=SalesKBInput)
    def search_sales_knowledge(query: str) -> str:
        """
        Search CRM product knowledge for sales conversations.

        Use when prospect asks about:
        - Plan features and differences
        - Pricing and discounts
        - What the product can do
        - Free trial or onboarding

        Always use this before quoting prices or features.
        """
        try:
            return rag_chain.invoke(query)
        except Exception as e:
            return f"KB search error: {str(e)}"

    return search_sales_knowledge


# ── Sales Agent System Prompt ─────────────────────────────────────────────────
def _build_system_prompt(history: list[dict]) -> str:
    """Build Maya's system prompt with conversation history injected."""
    history_text = ""
    if history:
        lines = []
        for turn in history[-6:]:
            prefix = "Sales Rep" if turn["role"] == "human" else "Maya"
            lines.append(f"{prefix}: {turn['content'][:200]}")
        history_text = "\n".join(lines)

    return f"""You are Maya, an expert AI Sales Assistant for our CRM platform.

YOUR MISSION:
Help sales teams qualify prospects, manage leads, and close deals faster.

YOUR PERSONALITY:
- Confident and results-oriented
- Always thinking about next steps to move deals forward
- Data-driven: use team size to recommend the right plan
- Proactive: suggest actions the rep might not think of

CONVERSATION HISTORY (use this to remember earlier context):
{history_text if history_text else "No prior conversation yet."}

PLAN QUALIFICATION GUIDE:
  1-3 users            -> Starter Plan (Rs.999/month)
  4-15 users           -> Growth Plan (Rs.2,999/month)
  15+ users            -> Enterprise Plan (custom pricing)
  Annual billing       -> 20% discount on any plan

SALES WORKFLOW:
1. QUALIFY  - Understand team size, budget, needs
2. MATCH    - Identify right plan using the guide above
3. CREATE   - Add them as lead in CRM database
4. EDUCATE  - Share relevant product info from knowledge base
5. EMAIL    - Send personalized follow-up
6. ADVANCE  - Always end with a concrete next step

IMPORTANT: Always end your final response with:
"Recommended Next Step: [specific action with timeline]"
"""


# ── Sales Agent ───────────────────────────────────────────────────────────────
class SalesAgent:
    """
    Maya — Specialized AI Sales Assistant.

    Focused on:
    - Lead qualification and creation (DB-backed)
    - Pipeline management
    - Personalized email outreach
    - Deal advancement with concrete next steps

    Difference from FullCRMAgent (Day 18):
      FullCRMAgent: general-purpose, 12 tools, generic prompt
      SalesAgent:   sales-focused, 10 curated tools, sales workflow prompt

    Week 4: This agent will be orchestrated by SupervisorAgent.
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        self.model       = model
        self.max_iterations = max_iterations
        self.verbose     = verbose

        # ── Memory ────────────────────────────────────────────────────────────
        self._history: list[dict] = []

        # ── Tools (curated for sales) ─────────────────────────────────────────
        from app.tools.lead_tools_db import (
            create_lead_db, get_lead_db, update_lead_db, get_lead_stats,
        )
        from app.tools.email_tools import draft_email, send_email, email_lead
        from app.tools.utility_tools import get_current_datetime, calculate_discount

        sales_kb = _build_sales_kb_tool()

        # 10 sales-focused tools (NOT all 12 — specialized!)
        self.tools = [
            sales_kb,            # product knowledge for sales conversations
            create_lead_db,      # add qualified leads to DB
            get_lead_db,         # look up existing leads
            update_lead_db,      # advance lead status
            get_lead_stats,      # pipeline overview
            draft_email,         # write personalized follow-up
            send_email,          # send email to prospect
            email_lead,          # draft + send in one step
            calculate_discount,  # show annual pricing savings
            get_current_datetime,
        ]
        self.tool_map = {t.name: t for t in self.tools}

        # ── LLM ───────────────────────────────────────────────────────────────
        llm = ChatGroq(
            model=model,
            temperature=0.2,   # slight creativity for warm sales tone
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.llm_with_tools = llm.bind_tools(self.tools)

        logger.info(
            f"SalesAgent 'Maya' ready | "
            f"model={model} | tools={len(self.tools)}"
        )

    # ── ReAct loop ────────────────────────────────────────────────────────────
    def _react_loop(self, user_message: str) -> tuple[str, list[dict]]:
        """Manual ReAct loop with sales-focused system prompt."""
        messages = [
            SystemMessage(content=_build_system_prompt(self._history)),
            HumanMessage(content=user_message),
        ]
        steps = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n  [Iter {iteration + 1}]")

            ai_message = self.llm_with_tools.invoke(messages)
            messages.append(ai_message)

            if not ai_message.tool_calls:
                if self.verbose:
                    print(f"  -> Final Answer")
                return ai_message.content, steps

            if self.verbose:
                print(f"  -> Tools: {[tc['name'] for tc in ai_message.tool_calls]}")

            for tc in ai_message.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id   = tc["id"]

                if tool_name in self.tool_map:
                    try:
                        result = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Error in {tool_name}: {e}"
                else:
                    result = f"Tool '{tool_name}' not found."

                if self.verbose:
                    print(f"     [{tool_name}] -> {str(result)[:80]}...")

                steps.append({"tool": tool_name, "input": tool_args, "output": str(result)})
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return "Reached max iterations.", steps

    # ── Public API ────────────────────────────────────────────────────────────
    def chat(self, message: str) -> str:
        """
        Multi-turn sales conversation with memory.
        Maya remembers context across the session.
        """
        if self.verbose:
            print(f"\n> Maya processing: '{message[:60]}'")
        try:
            answer, _ = self._react_loop(message)
            self._history.append({"role": "human",  "content": message})
            self._history.append({"role": "ai",     "content": answer})
            return answer
        except Exception as e:
            logger.error(f"SalesAgent error: {e}")
            return f"Error: {str(e)}"

    def chat_verbose(self, message: str) -> dict:
        """Chat and return full tool call details."""
        if self.verbose:
            print(f"\n> Maya processing: '{message[:60]}'")
        try:
            answer, steps = self._react_loop(message)
            self._history.append({"role": "human",  "content": message})
            self._history.append({"role": "ai",     "content": answer})
            return {
                "message":     message,
                "answer":      answer,
                "tools_used":  [s["tool"] for s in steps],
                "iterations":  len(steps),
                "memory_turns": len(self._history),
            }
        except Exception as e:
            return {
                "message": message, "answer": f"Error: {str(e)}",
                "tools_used": [], "iterations": 0, "memory_turns": 0,
            }

    def qualify_lead(self, conversation: str) -> dict:
        """
        Qualify a lead from a free-form conversation description.
        Runs full workflow: qualify → create → educate → email → next steps.
        """
        prompt = (
            f"Based on this conversation, qualify the lead and complete the sales workflow:\n\n"
            f"{conversation}\n\n"
            f"Complete ALL these steps:\n"
            f"1. Determine the right plan based on team size\n"
            f"2. Create the lead in CRM database\n"
            f"3. Search product knowledge for relevant info\n"
            f"4. Send a personalized follow-up email\n"
            f"5. Provide concrete next steps with timeline"
        )
        return self.chat_verbose(prompt)

    def get_pipeline_summary(self) -> str:
        """Get current sales pipeline overview."""
        return self.chat(
            "Give me a summary of our current sales pipeline — "
            "total leads, breakdown by status, and what needs immediate attention."
        )

    def reset_memory(self) -> None:
        """Clear conversation history — fresh session."""
        self._history = []
        print("Maya's memory cleared — fresh session!")

    def get_tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


# ── Mini Assignment: Support Agent ────────────────────────────────────────────
class SupportAgent:
    """
    Alex — AI Support Assistant. (Week 3 Mini Assignment)

    Focused on:
    - Answering product/policy questions from KB
    - Looking up customer info
    - Sending support emails
    - NOT selling (that's Maya's job)

    Persona: Empathetic, solution-focused, calm under pressure.
    Week 4: SupervisorAgent routes to Alex for support issues.
    """

    def __init__(self, model: str = GROQ_MODEL, verbose: bool = True):
        self.verbose = verbose
        self._history: list[dict] = []

        from app.tools.lead_tools_db import get_lead_db
        from app.tools.email_tools import email_lead, draft_email
        from app.tools.utility_tools import get_current_datetime
        from app.tools.knowledge_base_tool import search_knowledge_base, get_pricing

        # Support gets fewer, support-specific tools
        self.tools = [
            search_knowledge_base,  # answer policy/product questions
            get_pricing,            # pricing info only
            get_lead_db,            # look up customer info
            email_lead,             # send support email
            draft_email,            # write empathetic response
            get_current_datetime,
        ]
        self.tool_map = {t.name: t for t in self.tools}

        llm = ChatGroq(
            model=model, temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.llm_with_tools = llm.bind_tools(self.tools)

    def _build_system_prompt(self) -> str:
        history_text = ""
        if self._history:
            lines = [
                f"{'Customer' if t['role'] == 'human' else 'Alex'}: {t['content'][:150]}"
                for t in self._history[-4:]
            ]
            history_text = "\n".join(lines)

        return f"""You are Alex, an empathetic AI Support Assistant for our CRM platform.

YOUR MISSION: Resolve customer issues quickly and leave them feeling heard and helped.

YOUR PERSONALITY:
- Empathetic and patient — always acknowledge the customer's frustration first
- Solution-focused — give clear, actionable answers
- Honest — if you don't know, say so and offer to escalate
- NOT sales-focused — don't push upgrades or new features

CONVERSATION HISTORY:
{history_text if history_text else "No prior conversation yet."}

Always:
1. Acknowledge the issue first ("I understand this is frustrating...")
2. Search the knowledge base for accurate answers
3. Give a clear solution or next steps
4. Offer follow-up if needed
"""

    def chat(self, message: str) -> str:
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=message),
        ]
        for _ in range(6):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            if not ai_msg.tool_calls:
                answer = ai_msg.content
                self._history.append({"role": "human", "content": message})
                self._history.append({"role": "ai",    "content": answer})
                return answer
            for tc in ai_msg.tool_calls:
                result = self.tool_map.get(tc["name"], lambda x: "Tool not found").invoke(tc["args"]) \
                    if tc["name"] in self.tool_map else "Tool not found"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return "Unable to resolve — please contact support@crmplatform.com"


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    print("\n" + "="*60)
    print("DAY 21 — Sales Agent: Maya")
    print("Week 3 Final Capstone")
    print("="*60)

    print("\n[SETUP] Building SalesAgent Maya...")
    maya = SalesAgent(model=GROQ_MODEL, max_iterations=10, verbose=True)
    print(f"\nMaya ready | {len(maya.tools)} sales tools:")
    for t in maya.get_tool_names():
        print(f"  - {t}")

    # ── TEST 1: Full Lead Qualification Flow ──────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Full Lead Qualification")
    print("="*60)

    import time

    result = maya.chat_verbose(
        "I just got off a call with Rahul Sharma from TechCorp Solutions. "
        "He manages a sales team of 12 people and is frustrated that "
        "they lose leads because follow-ups happen too late. "
        "Budget is around Rs.3000-4000 per month. "
        "Email: rahul@techcorp.com, Phone: 9823456710. "
        "He seemed very interested when I mentioned AI features."
    )

    print(f"\nTools used: {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nMaya's Response:\n{result['answer']}")

    # ── TEST 2: Product Question ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Prospect Asks Product Question")
    print("="*60)
    time.sleep(2)

    result = maya.chat_verbose(
        "Rahul is asking what exactly is included in the Growth plan. "
        "He specifically wants to know about AI features and if there's a trial."
    )
    print(f"Tools used: {result['tools_used']}")
    print(f"\nMaya:\n{result['answer']}")

    # ── TEST 3: Memory Test ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Memory — Maya remembers Rahul from Turn 1")
    print("="*60)
    time.sleep(2)

    result = maya.chat_verbose(
        "Rahul replied to our email. He wants the annual pricing with discount."
    )
    print(f"Tools used: {result['tools_used']}")
    print(f"Memory turns: {result['memory_turns']}")
    print(f"\nMaya:\n{result['answer']}")

    # ── TEST 4: Pipeline Summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Sales Pipeline Overview")
    print("="*60)
    time.sleep(2)

    maya.chat("Add Priya Mehta from StartupXYZ, 3 users, starter plan. Email: priya@startupxyz.com")
    time.sleep(1)
    maya.chat("Add Amit Kumar from InfoSys, 50+ users, enterprise plan. Email: amit@infosys.com")
    time.sleep(1)

    summary = maya.get_pipeline_summary()
    print(f"\nPipeline Summary:\n{summary}")

    # ── TEST 5: Full qualify_lead() workflow ──────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: qualify_lead() — Full Automated Workflow")
    print("="*60)
    time.sleep(3)

    maya.reset_memory()
    result = maya.qualify_lead(
        "Met Divya Singh at a conference. Company: MediTech Solutions, "
        "team of 8 people. Email: divya@meditech.com. "
        "Needs CRM automation, specifically email follow-ups. Budget flexible."
    )
    print(f"\nTools used: {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nFull Sales Report:\n{result['answer']}")

    # ── TEST 6: Support Agent (Mini Assignment) ───────────────────────────────
    print("\n" + "="*60)
    print("TEST 6: SupportAgent Alex (Mini Assignment)")
    print("="*60)
    time.sleep(2)

    print("[SETUP] Building SupportAgent Alex...")
    alex = SupportAgent(model=GROQ_MODEL, verbose=False)
    print(f"Alex ready | {len(alex.tools)} support tools")
    print(f"Tools: {[t.name for t in alex.tools]}\n")

    support_q = "I was charged after cancelling. What is your refund policy?"
    print(f"Customer: {support_q}")
    answer = alex.chat(support_q)
    print(f"Alex: {answer}")

    # ── TEST 7: Agent Comparison ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 7: Specialized vs General — Key Differences")
    print("="*60)
    print("""
General FullCRMAgent (Day 18):    SalesAgent Maya (Day 21):
------------------------------    -------------------------
12 tools (everything)             10 sales-focused tools
Generic CRM prompt                Sales workflow + qualification guide
temperature=0                     temperature=0.2 (warmer tone)
No next-steps guidance            Always ends with Recommended Next Step
No plan qualification guide       1-3 / 4-15 / 15+ users -> plan

SupportAgent Alex (Mini):
  6 support-focused tools
  Empathetic, solution-focused prompt
  NOT selling — resolving issues
""")

    # ── Week 3 Summary ────────────────────────────────────────────────────────
    print("="*60)
    print("WEEK 3 COMPLETE!")
    print("="*60)
    print("""
Day 15: Custom Tools       - 7 CRM tools with @tool decorator
Day 16: Tool Binding       - LLM decides which tool to use
Day 17: ReAct Agent        - Thought -> Action -> Observation loop
Day 18: Full CRM Agent     - Memory + RAG + ReAct combined
Day 19: Email Automation   - draft_email, send_email, email_lead
Day 20: Database Layer     - SQLAlchemy + SQLite/PostgreSQL
Day 21: Sales Agent Maya   - Specialized sales workflow agent  TODAY

Total tools built:  12
Total agents built: 6  (ChatAssistant, LeadExtractor, ToolCallingChain,
                         FullCRMAgent, SalesAgent, SupportAgent)

Next -> Week 4: LangGraph + FastAPI + React Dashboard + Docker!
""")
    print("="*60)
