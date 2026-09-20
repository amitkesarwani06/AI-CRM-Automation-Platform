# AI CRM Platform - Backend

An AI-powered CRM automation platform built with Python, FastAPI, LangChain, LangGraph, MongoDB, and Celery.

## Prerequisites
- Python 3.10+
- MongoDB (Atlas or local)
- Redis (for Celery tasks)

## Setup

1. Create virtual environment: 
   ```bash
   python -m venv .venv
   ```
2. Activate it: 
   ```bash
   # Windows
   .\.venv\Scripts\Activate.ps1
   # Mac/Linux
   source .venv/bin/activate
   ```
3. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
4. Copy environment template and fill in your keys:
   ```bash
   cp .env.example .env
   ```

## Running the Application

### 1. Start the API Server
```bash
python -m uvicorn app.api.main:app --reload --port 8000
```
API Documentation will be available at: http://localhost:8000/docs

### 2. Start the Celery Worker (for background tasks)
```bash
python -m celery -A app.tasks.celery_app worker --loglevel=info
```

## Architecture
- **API**: FastAPI (`app/api/`)
- **Agents**: LangChain & LangGraph (`app/agents/`, `app/workflows/`)
- **Database**: SQLAlchemy (SQLite) + PyMongo (MongoDB) (`app/database/`)
- **Tasks**: Celery + Redis (`app/tasks/`)
