import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv

# Fix Windows terminal emoji / unicode encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown Lifespan ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    logger.info("🚀 AI CRM Platform API starting...")

    # Initialize database tables
    from app.database.connection import init_db
    init_db()
    logger.info("✅ Database initialized")

    yield  # Application runs while yielded

    logger.info("👋 AI CRM Platform API shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI CRM Automation Platform API",
    description="""
## AI-Powered CRM Backend

Built with **LangChain**, **LangGraph Multi-Agent Supervisor**, and **FastAPI**.

### Features
- 🤖 **Multi-Agent Chat**: Routes dynamically to Sales, Support, Analytics, or General agents.
- 📋 **Lead Management**: Full CRUD persistence with SQLite / PostgreSQL.
- 🔍 **Knowledge Base RAG**: Fast contextual Q&A from ingested company docs.
- 🎯 **Lead Extraction**: AI-powered structured extraction from unstructured text.
- 📊 **Pipeline Analytics**: Real-time funnel counts and status tracking.

### Authentication
    All protected endpoints require the `X-API-Key` header.
    Click the **🔒 Authorize** button (top right) and paste your API key once to authenticate all endpoints.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
# Required for React Dashboard (Day 28) and external frontends to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routers.chat import router as chat_router
from app.api.routers.leads import router as leads_router
from app.api.routers.tasks import router as tasks_router
from app.api.routers.history import router as history_router


app.include_router(chat_router)
app.include_router(leads_router)
app.include_router(tasks_router)
app.include_router(history_router)


# ── Global OpenAPI Security (Authorize 🔒 button in Swagger) ──────────────────
def custom_openapi():
    """Inject ApiKeyAuth security scheme so Swagger shows the Authorize button."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Define the security scheme (API key in X-API-Key header)
    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Paste your API key here. Get it from the `.env` file (API_KEY).",
        }
    }

    # Apply security globally to ALL endpoints
    schema["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# ── Health & Info Endpoints ───────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Welcome and API documentation link."""
    return {
        "status": "running",
        "service": "AI CRM Automation Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check verifying database, mongo, and service readiness."""
    from app.database.connection import engine
    from app.database.mongo_connection import ping_mongo
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    mongo_status = "connected" if ping_mongo() else "not connected"

    return {
        "status": "healthy",
        "database": db_status,
        "mongodb": mongo_status,
        "ai_model": os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")),
        "auth_enabled": os.getenv("API_KEY", "crm-dev-key-2024") != "none",
    }



# ── Direct Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
