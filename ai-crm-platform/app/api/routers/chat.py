import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.api.dependencies import verify_api_key

router = APIRouter(prefix="/chat", tags=["Chat & Agents"])
logger = logging.getLogger(__name__)

# ── Cached Singletons ─────────────────────────────────────────────────────────
_supervisor_app = None
_kb_agent = None


def get_supervisor():
    """Build and cache supervisor graph once at startup."""
    global _supervisor_app
    if _supervisor_app is None:
        from app.workflows.supervisor_graph import build_supervisor_graph
        _supervisor_app = build_supervisor_graph()
        logger.info("Supervisor graph initialized and cached")
    return _supervisor_app


def get_kb_agent():
    """Build and cache KnowledgeBaseAgent once."""
    global _kb_agent
    if _kb_agent is None:
        from app.knowledge_base.rag.knowledge_base_agent import KnowledgeBaseAgent
        _kb_agent = KnowledgeBaseAgent()
        company_knowledge = """
Refund Policy: 30-day money-back guarantee. Contact support@crmplatform.com.
Pricing: Starter Rs.999/month (3 users), Growth Rs.2999/month (15 users), Enterprise custom.
Support: Monday-Saturday 9am-7pm IST. Emergency 24x7 for Enterprise.
Features: Lead management, email automation, WhatsApp integration, analytics.
Free trial: 14 days, no credit card required.
"""
        _kb_agent.ingest(company_knowledge, source="company_kb")
        logger.info("KnowledgeBaseAgent initialized and ingested company KB")
    return _kb_agent


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Chat with AI CRM",
    description="Send a message to the AI CRM supervisor. It routes to Sales, Support, Analytics, or General agents automatically."
)
async def chat(
    request: ChatRequest,
    auth: bool = Depends(verify_api_key),
):
    """
    Main chat endpoint — routes to Sales, Support, or Analytics agent.

    Examples:
    - "Add Rahul from TechCorp as a lead"
    - "What is your refund policy?"
    - "Show me pipeline statistics"
    - "Add a lead AND show stats"
    """
    try:
        supervisor = get_supervisor()

        initial_state = {
            "messages":          [HumanMessage(content=request.message)],
            "next_agents":       [],
            "agent_results":     {},
            "supervisor_reason": "",
            "final_response":    "",
            "current_agent":     "",
        }

        result = await supervisor.ainvoke(initial_state)

        agents_used = list(result.get("agent_results", {}).keys())
        intent = result.get("intent", "")
        raw_response = result.get("final_response") or "I couldn't process your request."
        if isinstance(raw_response, list):
            parts = []
            for item in raw_response:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            response_text = " ".join(parts)
        else:
            response_text = str(raw_response)

        logger.info(
            f"Chat processed: '{request.message[:50]}' -> "
            f"agents={agents_used} | intent={intent}"
        )

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            agents_used=agents_used,
            intent=intent,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)[:200]}"
        )


@router.post(
    "/rag",
    response_model=ChatResponse,
    summary="Ask Knowledge Base",
    description="Ask questions directly answered from company knowledge base documents."
)
async def ask_knowledge_base(
    request: ChatRequest,
    auth: bool = Depends(verify_api_key),
):
    """Direct RAG endpoint — bypasses multi-agent supervisor, queries KB directly."""
    try:
        agent = get_kb_agent()
        answer = agent.ask(request.message)

        return ChatResponse(
            response=answer,
            session_id=request.session_id,
            agents_used=["knowledge_base"],
            intent="support",
        )

    except Exception as e:
        logger.error(f"Knowledge Base error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
