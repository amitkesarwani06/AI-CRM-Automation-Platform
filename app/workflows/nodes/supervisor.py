import os
import sys
import json
import re
import logging
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

SUPERVISOR_SYSTEM = """You are the Supervisor Orchestrator of an AI CRM system.

Your team:
- sales_agent:     Maya — handles new leads, prospect creation, pricing, plans, demos, deal qualification
- support_agent:   Alex — handles product questions, refund policy, technical help, support hours
- analytics_agent: Data Bot — handles pipeline statistics, reports, lead counts, performance metrics
- general:         You handle directly — greetings, generic chit-chat, unclear requests

Your job: Read the user's message and decide which agent(s) should handle it.

Rules:
1. Choose ONE agent for simple single-intent requests.
2. Choose MULTIPLE agents in order of execution for multi-intent requests (e.g., ["sales_agent", "analytics_agent"] or ["support_agent", "analytics_agent"]).
3. For greetings or unclear requests, choose ["general"] and provide a direct_response.
4. Always return ONLY valid JSON with no markdown formatting and no extra text.

Return ONLY this JSON format:
{
  "agents": ["sales_agent"] or ["support_agent"] or ["analytics_agent"] or ["sales_agent", "analytics_agent"],
  "reason": "one sentence explaining why these agents were selected",
  "direct_response": null or "friendly greeting response if handling general directly"
}
"""


def log_routing_decision(user_message: str, agents_chosen: list, reason: str):
    """Mini Assignment: Log supervisor routing decisions for monitoring."""
    try:
        os.makedirs("logs", exist_ok=True)
        log_entry = {
            "timestamp": str(datetime.now()),
            "message": user_message[:120],
            "agents": agents_chosen,
            "reason": reason,
        }
        with open("logs/supervisor_decisions.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Error logging routing decision: {e}")


def supervisor_node(state: dict) -> dict:
    """
    Supervisor Node: Reads message -> decides routing plan.
    Sets state["next_agents"], state["supervisor_reason"], and logs decision.
    """
    messages = state.get("messages", [])
    if not messages:
        return {**state, "next_agents": ["general"], "supervisor_reason": "No message provided"}

    latest = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    try:
        response = llm.invoke([
            HumanMessage(content=f"{SUPERVISOR_SYSTEM}\n\nUser message: \"{latest}\"")
        ])
        content = response.content.strip()

        if "```" in content:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
            else:
                content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)
        next_agents = data.get("agents", ["general"])
        reason = data.get("reason", "Routing decided by supervisor")
        direct = data.get("direct_response")

    except Exception as e:
        logger.warning(f"Supervisor parsing fallback: {e}")
        # Rule-based fallback
        text_lower = latest.lower()
        next_agents = []
        if any(w in text_lower for w in ["lead", "pricing", "plan", "demo", "prospect", "add priya", "add rahul"]):
            next_agents.append("sales_agent")
        if any(w in text_lower for w in ["refund", "support", "hours", "help", "cancel"]):
            next_agents.append("support_agent")
        if any(w in text_lower for w in ["stats", "statistics", "pipeline", "report", "how many leads"]):
            next_agents.append("analytics_agent")

        if not next_agents:
            next_agents = ["general"]
            reason = "General query or greeting"
            direct = "Hello! I am the CRM Supervisor Assistant. I can coordinate our Sales, Support, and Analytics agents to help you."
        else:
            reason = f"Routing determined by keywords: {', '.join(next_agents)}"
            direct = None

    # Log routing decision (Mini Assignment)
    log_routing_decision(latest, next_agents, reason)
    logger.info(f"Supervisor routing: {next_agents} | Reason: {reason}")

    if direct and next_agents == ["general"]:
        return {
            **state,
            "next_agents": [],
            "supervisor_reason": reason,
            "final_response": direct,
            "messages": [AIMessage(content=direct)],
            "agent_results": {"general": direct},
        }

    return {
        **state,
        "next_agents": next_agents,
        "supervisor_reason": reason,
        "agent_results": {},
    }


def route_supervisor(state: dict) -> str:
    """
    Conditional edge after supervisor_node.
    Returns the first agent to execute, or synthesize_node if none.
    """
    next_agents = state.get("next_agents", [])
    if not next_agents:
        return "synthesize_node"
    return next_agents[0]


def synthesize_node(state: dict) -> dict:
    """
    Synthesis Node: Combines results from multiple specialist agents
    into one coherent, professional response.
    """
    agent_results = state.get("agent_results", {})
    final_response = state.get("final_response", "")

    if final_response:
        return state

    if not agent_results:
        msg = "I've reviewed your request but no specialist output was produced."
        return {**state, "final_response": msg, "messages": [AIMessage(content=msg)]}

    if len(agent_results) == 1:
        agent_name = list(agent_results.keys())[0]
        res = agent_results[agent_name]
        return {**state, "final_response": res, "messages": [AIMessage(content=res)]}

    # Multiple agent results — synthesize via LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    results_text = "\n\n".join([
        f"[{agent.upper()}]:\n{result}"
        for agent, result in agent_results.items()
    ])

    synthesis_prompt = (
        f"You are the CRM Supervisor. Combine these specialist agent reports into ONE seamless, "
        f"concise, and professional response for the user:\n\n"
        f"{results_text}\n\n"
        f"Rules:\n"
        f"- Present a unified answer directly to the user.\n"
        f"- Do not use technical jargon like 'Agent 1 returned' or 'Agent 2 reported'.\n"
        f"- Keep the combined response clear and under 150 words."
    )

    try:
        combined = llm.invoke([HumanMessage(content=synthesis_prompt)]).content.strip()
    except Exception as e:
        logger.warning(f"Synthesis fallback: {e}")
        combined = "\n\n".join([f"{res}" for res in agent_results.values()])

    return {
        **state,
        "final_response": combined,
        "messages": [AIMessage(content=combined)],
    }
