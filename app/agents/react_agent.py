import os
import sys
import logging
import threading

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')
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
REACT_SYSTEM_PROMPT = """You are an AI CRM assistant with access to tools.

You help sales teams by managing leads, answering product questions,
and automating CRM workflows.

Think step by step. Use as many tools as needed to complete the task.
Be concise and professional in your final answer.
"""


# ── CRM ReAct Agent ───────────────────────────────────────────────────────────
class CRMReActAgent:
    """
    Autonomous CRM Agent using ReAct pattern (Reason + Act loop).

    ReAct Loop:
    Thought -> Action -> Observation -> Thought -> ... -> Final Answer

    Difference from Day 16 ToolCallingChain:
      Day 16: ONE round of tool calls, manual execution
      Day 17: LOOPS until all tool calls are resolved (multi-step chains)

    Usage:
        agent = CRMReActAgent(tools=ALL_CRM_TOOLS)
        answer = agent.run("Find Rahul and update status to qualified")
    """

    def __init__(
        self,
        tools: list[BaseTool],
        model: str = "openai/gpt-oss-20b",
        max_iterations: int = 8,
        verbose: bool = True,
    ):
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Tool lookup: name -> tool object
        self.tool_map = {t.name: t for t in tools}

        # Groq LLM with tools bound
        llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.llm_with_tools = llm.bind_tools(tools)

        logger.info(
            f"CRMReActAgent ready | model={model} | "
            f"tools={len(tools)} | max_iter={max_iterations}"
        )

    # ── Core ReAct loop ───────────────────────────────────────────────────────
    def _run_loop(self, task: str) -> tuple[str, list[dict]]:
        """
        Internal ReAct loop:
        1. LLM decides which tools to call
        2. Execute tools, add results to conversation
        3. LLM decides next tools OR gives Final Answer
        4. Repeat until no more tool calls OR max_iterations reached

        Returns (final_answer, steps_taken)
        """
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=task),
        ]
        steps = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n> Iteration {iteration + 1}")

            # LLM reasons and decides tool calls
            ai_message = self.llm_with_tools.invoke(messages)
            messages.append(ai_message)

            # No more tool calls -> Final Answer
            if not ai_message.tool_calls:
                if self.verbose:
                    print(f"> Final Answer reached after {iteration + 1} iteration(s)")
                return ai_message.content, steps

            # Execute all tool calls from this iteration
            if self.verbose:
                print(f"  Invoking: {[tc['name'] for tc in ai_message.tool_calls]}")

            for tool_call in ai_message.tool_calls:
                tool_name    = tool_call["name"]
                tool_args    = tool_call["args"]
                tool_call_id = tool_call["id"]

                # Execute tool
                if tool_name in self.tool_map:
                    try:
                        tool_result = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Error in {tool_name}: {str(e)}"
                        logger.error(f"Tool error: {e}")
                else:
                    tool_result = f"Tool '{tool_name}' not found."

                if self.verbose:
                    print(f"  [{tool_name}] -> {str(tool_result)[:80]}...")

                # Record step
                steps.append({
                    "tool":   tool_name,
                    "input":  tool_args,
                    "output": str(tool_result),
                })

                # Add tool result to conversation (MUST match tool_call_id)
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return "Agent reached maximum iterations without completing the task.", steps

    # ── Public methods ────────────────────────────────────────────────────────
    def run(self, task: str) -> str:
        """
        Run task autonomously — loops until complete.

        Usage:
            answer = agent.run("Add Rahul as lead then check pricing")
        """
        if self.verbose:
            print(f"\n> Entering ReAct loop for: '{task[:60]}'")
        try:
            answer, _ = self._run_loop(task)
            return answer
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Agent error: {str(e)}"

    def run_verbose(self, task: str) -> dict:
        """
        Run task and return full details:
        - final answer
        - each step (tool, input, output)
        - total iteration count
        """
        try:
            answer, steps = self._run_loop(task)
            return {
                "task":       task,
                "answer":     answer,
                "iterations": len(steps),
                "steps":      steps,
            }
        except Exception as e:
            return {
                "task":       task,
                "answer":     f"Error: {str(e)}",
                "iterations": 0,
                "steps":      [],
            }

    # ── Mini Assignment: timeout ───────────────────────────────────────────────
    def run_with_timeout(self, task: str, timeout_seconds: int = 30) -> str:
        """
        Run task with a wall-clock timeout.
        Prevents hung agents from running forever.
        """
        result = {"answer": f"Timeout — agent took longer than {timeout_seconds}s."}

        def target():
            result["answer"] = self.run(task)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout_seconds)
        return result["answer"]

    # ── Interactive chat ───────────────────────────────────────────────────────
    def chat(self) -> None:
        """
        Interactive chat with the ReAct agent.
        Type 'quit' | 'tools' | 'help'
        """
        print("\n" + "="*60)
        print("CRM ReAct Agent — Autonomous Mode")
        print("Agent thinks step by step and uses tools automatically.")
        print("Type 'quit' | 'tools' | 'help'")
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
            if user_input.lower() == "help":
                print("\nExample tasks:")
                print("  - Add Rahul from TechCorp as a lead")
                print("  - Find lead named Priya and update status to contacted")
                print("  - What is the refund policy?")
                print("  - Show all leads and get Growth plan pricing\n")
                continue

            print("\nAgent thinking...\n")
            answer = self.run(user_input)
            print(f"\nAnswer: {answer}\n")
            print("-" * 50)


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    from app.tools import ALL_CRM_TOOLS

    print("\n" + "="*60)
    print("DAY 17 — ReAct Agent")
    print("Autonomous Decision Making Loop")
    print("="*60)

    agent = CRMReActAgent(
        tools=ALL_CRM_TOOLS,
        model="openai/gpt-oss-20b",
        max_iterations=8,
        verbose=True,
    )
    print(f"\nAgent ready | {len(ALL_CRM_TOOLS)} tools | max_iterations=8\n")

    # ── TEST 1: Single step ───────────────────────────────────────────────────
    print("="*60)
    print("TEST 1: Single step — create one lead")
    print("="*60)

    result = agent.run_verbose(
        "Add Rahul Sharma from TechCorp, email rahul@techcorp.com, Growth plan"
    )
    print(f"\nIterations: {result['iterations']}")
    print(f"Tools used: {[s['tool'] for s in result['steps']]}")
    print(f"Answer: {result['answer']}")

    # ── TEST 2: Multi-step ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Multi-step — create lead THEN check pricing")
    print("="*60)

    result = agent.run_verbose(
        "Add Priya Mehta from StartupXYZ, Starter plan. "
        "Then tell me the Starter plan pricing."
    )
    print(f"\nIterations: {result['iterations']}")
    print(f"Tools used: {[s['tool'] for s in result['steps']]}")
    print(f"Answer: {result['answer']}")

    # ── TEST 3: Find and update ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Find lead THEN update status")
    print("="*60)

    result = agent.run_verbose(
        "Find the lead named Rahul and update status to contacted. "
        "Note: Called Monday, very interested in demo."
    )
    print(f"\nIterations: {result['iterations']}")
    print(f"Tools used: {[s['tool'] for s in result['steps']]}")
    for step in result['steps']:
        print(f"  [{step['tool']}]")
        print(f"    Input:  {step['input']}")
        print(f"    Output: {str(step['output'])[:80]}...")
    print(f"\nAnswer: {result['answer']}")

    # ── TEST 4: Knowledge + lead combined ─────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Knowledge question + lead creation together")
    print("="*60)

    result = agent.run_verbose(
        "What is the refund policy? Also add Amit Kumar from InfoSys, Enterprise plan."
    )
    print(f"\nIterations: {result['iterations']}")
    print(f"Tools used: {[s['tool'] for s in result['steps']]}")
    print(f"Answer: {result['answer'][:250]}...")

    # ── TEST 5: No tool needed ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: General question — no tool needed")
    print("="*60)

    result = agent.run_verbose("What is a CRM system and why do businesses use it?")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools used: {[s['tool'] for s in result['steps']]}")
    print(f"Answer: {result['answer'][:150]}...")

    # ── TEST 6: All leads summary ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 6: Get all leads summary")
    print("="*60)

    result = agent.run_verbose(
        "Show me all the leads we have created so far with their status."
    )
    print(f"Iterations: {result['iterations']}")
    print(f"Tools used: {[s['tool'] for s in result['steps']]}")
    print(f"Answer:\n{result['answer']}")

    # ── TEST 7: Day 16 vs Day 17 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 7: Day 16 vs Day 17 — key difference")
    print("="*60)
    print("""
Day 16 — ToolCallingChain:
  User -> LLM -> tool calls -> answer  (one round)
  Good for: simple, single-step tasks

Day 17 — ReAct Agent:
  User -> LLM -> tools -> observe -> LLM -> tools -> ... -> answer (loop)
  Good for: complex, multi-step tasks that need chained reasoning

Real difference:
  Day 16: "Add Rahul as lead"             -> 1 round
  Day 17: "Find Rahul, update status,     -> loops until all done
           then check his plan pricing"
""")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("="*60)
    print("Day 17 COMPLETE — ReAct Agent working!")
    print("="*60)
    print("""
ReAct Loop:
  Thought -> Action -> Observation
  Thought -> Action -> Observation
  Thought -> Final Answer

Week 3 Progress:
  Day 15  Custom Tools (7 CRM tools)
  Day 16  Tool Binding + Calling
  Day 17  ReAct Agent (autonomous loop)  TODAY
  Day 18  Full CRM Agent
""")
    print("Next -> Day 18: Full CRM Agent — all tools + RAG + memory!")
    print("="*60)
