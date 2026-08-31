import sys
import os
sys.path.insert(0, os.getcwd())

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SearchKBInput(BaseModel):
    query: str = Field(
        description="The question or topic to search for in the knowledge base"
    )
    k: int = Field(default=3, description="Number of results to return")


class GetPricingInput(BaseModel):
    plan_name: str = Field(
        default="all",
        description="Plan name: 'starter', 'growth', 'enterprise', or 'all'"
    )


@tool("get_pricing", args_schema=GetPricingInput)
def get_pricing(plan_name: str = "all") -> str:
    """
    Get CRM platform pricing information.
    Use when a user asks about cost, plans, or pricing.

    Available plans: starter, growth, enterprise, all
    """
    pricing = {
        "starter":    "Starter Plan: Rs.999/month - up to 3 users, 500 leads, basic analytics.",
        "growth":     "Growth Plan: Rs.2,999/month - up to 15 users, unlimited leads, AI features.",
        "enterprise": "Enterprise Plan: Custom pricing - unlimited users, dedicated support.",
    }

    if plan_name.lower() == "all":
        result = "CRM Platform Pricing:\n"
        for plan, details in pricing.items():
            result += f"  - {details}\n"
        result += "\nAnnual billing: 20% discount on all plans."
        return result

    plan = plan_name.lower()
    if plan in pricing:
        return pricing[plan]
    return f"Plan '{plan_name}' not found. Available: starter, growth, enterprise, all"


@tool("search_knowledge_base", args_schema=SearchKBInput)
def search_knowledge_base(query: str, k: int = 3) -> str:
    """
    Search the CRM knowledge base for information about:
    - Refund policy
    - Pricing plans
    - Support hours
    - Getting started guides
    - Product features

    Use this tool whenever a user asks a question about the product,
    company policies, or how to use the CRM platform.
    """
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

        company_knowledge = """
Refund Policy
We offer a 30-day money-back guarantee on all plans.
To request a refund contact support@crmplatform.com.
Refunds are processed within 5-7 business days.

Pricing Plans
Starter Plan Rs.999 per month for up to 3 users and 500 leads.
Growth Plan Rs.2999 per month for up to 15 users with AI features.
Enterprise Plan has custom pricing with dedicated support.
Annual billing gives 20 percent discount on all plans.

Support Hours
Available Monday to Saturday 9am to 7pm IST.
Average response time is under 2 hours.
Contact support@crmplatform.com or call 1800-123-4567.

Getting Started
First 14 days are free with no credit card required.
Import leads from CSV in Settings then Import Data.

Features
Lead management with AI scoring and follow-up reminders.
Email automation with templates.
WhatsApp integration for direct messaging.
Analytics dashboard with real-time reports.
"""
        chunker = DocumentChunkingService(content_type=ContentType.CONVERSATIONAL)
        chunks = chunker.split_text(company_knowledge, source="company_faq")
        store = UnifiedVectorStore(
            store_type=VectorStoreType.FAISS,
            collection_name="tool_kb",
        )
        store.add_documents(chunks)
        retriever = RetrieverService(store).get_retriever(RetrieverType.SIMILARITY, k=k)
        rag = RAGChain(retriever)
        return rag.invoke(query)

    except Exception as e:
        return f"Knowledge base search failed: {str(e)}"
