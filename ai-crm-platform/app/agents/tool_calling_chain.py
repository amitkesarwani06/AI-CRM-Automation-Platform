import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


# ── CRM System Prompt ─────────────────────────────────────────────────────────
CRM_SYSTEM_PROMPT = """You are an AI CRM assistant for our sales platform.

You help sales teams by:
- Creating and managing leads
- Answering product questions
- Looking up customer information
- Providing pricing information

Always be concise and professional.
When you use a tool, briefly explain what you did and the result.
If you cannot help with something, say so clearly.
"""


class ToolCallingChain:
    """
    Connects LLM with CRM tools.

    Flow:
    User message
        -> LLM decides which tool to call
        -> Tool executes
        -> Result sent back to LLM
        -> LLM gives final answer

    This is the foundation for Day 17's ReAct Agent.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        model: str = "openai/gpt-oss-20b",
    ):
        self.tools = tools
        self.model = model

        # Tool lookup map: name -> tool object
        self.tool_map = {tool.name: tool for tool in tools}

        # Tool call history for Mini Assignment
        self._history: list[dict] = []

        # Groq LLM with tools bound
        llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.llm_with_tools = llm.bind_tools(tools)

        logger.info(
            f"ToolCallingChain ready | "
            f"model={model} | "
            f"tools={[t.name for t in tools]}"
        )

    # ── Core invoke ───────────────────────────────────────────────────────────
    def invoke(self, user_message: str) -> str:
        """
        Process a user message — LLM decides which tool to call.
        Returns the final LLM response after tool execution.
        """
        messages = [
            SystemMessage(content=CRM_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Step 1: LLM decides which tool to call (if any)
        ai_message = self.llm_with_tools.invoke(messages)
        messages.append(ai_message)

        # Step 2: No tool needed — LLM answered directly
        if not ai_message.tool_calls:
            logger.info("No tool call — LLM answered directly")
            return ai_message.content

        # Step 3: Execute each tool the LLM requested
        logger.info(f"Tool calls: {[tc['name'] for tc in ai_message.tool_calls]}")

        for tool_call in ai_message.tool_calls:
            tool_name    = tool_call["name"]
            tool_args    = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name in self.tool_map:
                try:
                    tool_result = self.tool_map[tool_name].invoke(tool_args)
                    logger.info(f"Tool '{tool_name}' executed OK")
                except Exception as e:
                    tool_result = f"Error executing {tool_name}: {str(e)}"
                    logger.error(f"Tool '{tool_name}' failed: {e}")
            else:
                tool_result = f"Tool '{tool_name}' not found."
                logger.warning(f"Unknown tool: {tool_name}")

            # Log to history (Mini Assignment)
            self._history.append({
                "tool":      tool_name,
                "args":      tool_args,
                "result":    str(tool_result)[:100],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

            # Step 4: Add tool result — tool_call_id MUST match
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

        # Step 5: LLM formulates final response with tool results
        final_response = self.llm_with_tools.invoke(messages)
        return final_response.content

    # ── Verbose invoke ────────────────────────────────────────────────────────
    def invoke_verbose(self, user_message: str) -> dict:
        """
        Same as invoke() but returns full details for debugging:
        - which tools were called
        - what args were used
        - what results came back
        - final answer
        """
        messages = [
            SystemMessage(content=CRM_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        ai_message = self.llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls_made = []

        if not ai_message.tool_calls:
            return {
                "question":   user_message,
                "tool_calls": [],
                "answer":     ai_message.content,
            }

        for tool_call in ai_message.tool_calls:
            tool_name    = tool_call["name"]
            tool_args    = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name in self.tool_map:
                try:
                    tool_result = self.tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
            else:
                tool_result = f"Tool '{tool_name}' not found."

            tool_calls_made.append({
                "tool":   tool_name,
                "args":   tool_args,
                "result": str(tool_result),
            })

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

        final_response = self.llm_with_tools.invoke(messages)

        return {
            "question":   user_message,
            "tool_calls": tool_calls_made,
            "answer":     final_response.content,
        }

    # ── Mini Assignment: history ──────────────────────────────────────────────
    def get_tool_call_history(self) -> list[dict]:
        """
        Returns all tool calls made so far with timestamps.
        Foundation for audit logging in production.
        """
        return self._history

    def print_history(self) -> None:
        """Print tool call history in a readable format."""
        if not self._history:
            print("No tool calls made yet.")
            return
        print(f"\nTool Call History ({len(self._history)} calls):")
        print("-" * 60)
        for i, entry in enumerate(self._history, 1):
            print(f"[{entry['timestamp']}] {i}. {entry['tool']}")
            print(f"   Args:   {entry['args']}")
            print(f"   Result: {entry['result'][:70]}...")
            print()

    # ── Interactive chat loop ─────────────────────────────────────────────────
    def chat(self) -> None:
        """
        Interactive chat loop with tool calling.
        Type 'quit' to exit | 'tools' to list tools | 'history' to see logs.
        """
        print("\n" + "="*60)
        print("CRM AI Assistant — Tool Calling Mode")
        print("Type 'quit' | 'tools' | 'history'")
        print("="*60 + "\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "tools":
                print("\nAvailable tools:")
                for i, t in enumerate(self.tools, 1):
                    print(f"  {i}. {t.name}")
                print()
                continue
            if user_input.lower() == "history":
                self.print_history()
                continue

            response = self.invoke(user_input)
            print(f"Aria: {response}\n")


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    from app.tools import ALL_CRM_TOOLS

    print("\n" + "="*60)
    print("DAY 16 — Tool Binding + Tool Calling")
    print("LLM (Groq) decides which tool to use!")
    print("="*60)

    chain = ToolCallingChain(tools=ALL_CRM_TOOLS)
    print(f"\nLLM bound with {len(ALL_CRM_TOOLS)} tools: {[t.name for t in ALL_CRM_TOOLS]}\n")

    # ── TEST 1: Tools visible to LLM ─────────────────────────────────────────
    print("="*60)
    print("TEST 1: Tools bound to LLM")
    print("="*60)
    for i, t in enumerate(ALL_CRM_TOOLS, 1):
        print(f"  {i}. {t.name}")

    # ── TEST 2: Direct tool call baseline ────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Direct tool call (no LLM) — baseline")
    print("="*60)
    from app.tools.lead_tools import create_lead
    print(create_lead.invoke({"name": "Test Lead", "company": "Test Corp"}))

    # ── TEST 3: LLM decides tool ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: LLM decides which tool to call")
    print("="*60)

    test_cases = [
        ("Add new lead: Rahul Sharma, TechCorp, email rahul@techcorp.com, Growth plan", "create_lead"),
        ("What are your pricing plans?",                                                "get_pricing"),
        ("Find the lead named Rahul",                                                   "get_lead"),
        ("What is your refund policy?",                                                 "search_knowledge_base"),
        ("What is today's date?",                                                       "get_current_datetime"),
    ]

    for i, (message, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{message[:55]}...'")
        print(f"Expected: {expected}")
        result = chain.invoke_verbose(message)
        called = [tc["tool"] for tc in result["tool_calls"]]
        correct = expected in called
        print(f"Called:   {called if called else 'None (direct)'}")
        print(f"Status:   {'OK' if correct else 'CHECK'}")
        print(f"Answer:   {result['answer'][:100]}...")

    # ── TEST 4: Verbose output ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Verbose — full tool call details")
    print("="*60)

    result = chain.invoke_verbose(
        "Add Priya Mehta from StartupXYZ as a lead. "
        "Interested in Starter plan. Phone: 9876543210"
    )
    print(f"Question: {result['question']}")
    print(f"\nTool calls: {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"\n  Tool:   {tc['tool']}")
        print(f"  Args:   {tc['args']}")
        print(f"  Result: {tc['result'][:80]}...")
    print(f"\nFinal Answer: {result['answer']}")

    # ── TEST 5: No tool needed ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: No tool needed — direct answers")
    print("="*60)

    for q in ["Hello! What can you help me with?", "What is a CRM system?", "Thank you!"]:
        result = chain.invoke_verbose(q)
        count = len(result["tool_calls"])
        print(f"Q: {q}")
        print(f"   Tools: {count} ({'direct answer' if count == 0 else 'used tool'})")
        print(f"   A: {result['answer'][:80]}...")
        print()

    # ── TEST 6: Multiple tools ────────────────────────────────────────────────
    print("="*60)
    print("TEST 6: Complex request — multiple tool calls")
    print("="*60)

    req = (
        "Create a lead for Amit Kumar from InfoSys, email amit@infosys.com, "
        "Enterprise plan. Also show me the Enterprise pricing."
    )
    print(f"Request: {req}\n")
    result = chain.invoke_verbose(req)
    print(f"Tools called: {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"  - {tc['tool']} -> {tc['result'][:50]}...")
    print(f"\nAnswer:\n{result['answer']}")

    # ── TEST 7: Comparison table ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 7: Tool vs Direct — comparison table")
    print("="*60)
    print(f"{'Question':<45} {'Tool':<25} {'Type'}")
    print("-" * 80)

    for q in [
        "Add lead: Raj from MegaCorp",
        "What is the Growth plan price?",
        "Hello how are you?",
        "Find lead named Priya",
        "What is today's date?",
        "What is 2 + 2?",
    ]:
        result = chain.invoke_verbose(q)
        tools  = [tc["tool"] for tc in result["tool_calls"]]
        t_str  = tools[0] if tools else "none"
        a_type = "Tool" if tools else "Direct"
        print(f"{q:<45} {t_str:<25} {a_type}")

    # ── TEST 8: Tool call history ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 8: Tool call history (Mini Assignment)")
    print("="*60)
    chain.print_history()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("="*60)
    print("Day 16 COMPLETE — Tool Binding + Tool Calling!")
    print("="*60)
    print("""
Tool Binding:   llm.bind_tools([tools])  LLM knows what tools exist
Tool Calling:   LLM returns tool_calls   your code executes them
ToolMessage:    result sent back         LLM formulates final answer

Day 15: Tools created (7 CRM tools)
Day 16: Tools connected to LLM  TODAY
Day 17: ReAct Agent loops automatically
""")
    print("Next -> Day 17: ReAct Agent!")
    print("="*60)
