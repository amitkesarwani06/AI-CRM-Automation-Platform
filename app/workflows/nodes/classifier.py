import os
import sys
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def classify_intent(state: dict) -> dict:
    """
    Node 1: Classify user intent.
    Reads the latest message and determines if it's:
    - sales:     new lead, pricing, demo request
    - support:   product help, refund, technical issue
    - analytics: reports, statistics, pipeline overview
    - general:   greetings, unclear requests

    Returns updated state with 'intent' field set.
    """
    messages = state.get("messages", [])
    if not messages:
        return {**state, "intent": "general"}

    # Get latest user message
    latest = messages[-1]
    user_text = latest.content if hasattr(latest, "content") else str(latest)

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    classification_prompt = f"""Classify this CRM message into ONE category:

Message: "{user_text}"

Categories:
- sales: new lead, prospect, pricing inquiry, demo request, deal closing
- support: refund, technical issue, how-to, complaint, product help
- analytics: reports, statistics, pipeline numbers, performance metrics
- general: greeting, unclear, doesn't fit above

Reply with ONLY the category word. Nothing else."""

    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        intent = response.content.strip().lower()
    except Exception as e:
        logger.error(f"Classification error: {e}")
        intent = "general"

    # Validate intent
    valid_intents = ["sales", "support", "analytics", "general"]
    if intent not in valid_intents:
        for valid in valid_intents:
            if valid in intent:
                intent = valid
                break
        else:
            intent = "general"

    logger.info(f"Intent classified: '{intent}' for: '{user_text[:50]}'")
    return {**state, "intent": intent}


def route_by_intent(state: dict) -> str:
    """
    Conditional edge function.
    Reads 'intent' from state → returns next node name.
    LangGraph calls this to decide where to route.
    """
    intent = state.get("intent", "general")
    routes = {
        "sales":     "sales_node",
        "support":   "support_node",
        "analytics": "analytics_node",
        "general":   "general_node",
    }
    next_node = routes.get(intent, "general_node")
    logger.info(f"Routing to: {next_node}")
    return next_node
