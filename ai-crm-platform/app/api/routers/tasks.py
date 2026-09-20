import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from celery.result import AsyncResult
from app.tasks.celery_app import celery_app
from app.api.dependencies import verify_api_key

router = APIRouter(prefix="/tasks", tags=["Background Tasks"])
logger = logging.getLogger(__name__)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class BulkEmailRequest(BaseModel):
    lead_ids: list[int]
    email_type: str = "follow_up"
    context: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    progress: Optional[dict] = None


# ── Status Polling Endpoint ───────────────────────────────────────────────────
@router.get("/{task_id}", response_model=TaskStatusResponse, summary="Check task status")
def get_task_status(task_id: str, auth: bool = Depends(verify_api_key)):
    """Check task state: PENDING -> STARTED -> PROGRESS -> SUCCESS/FAILURE."""
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return TaskStatusResponse(
            task_id=task_id,
            status="PENDING",
            progress={"status": "Task queued, waiting in Redis..."}
        )
    elif result.state == "PROGRESS":
        return TaskStatusResponse(task_id=task_id, status="PROGRESS", progress=result.info)
    elif result.state == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status="SUCCESS", result=result.result)
    elif result.state == "FAILURE":
        return TaskStatusResponse(task_id=task_id, status="FAILURE", result={"error": str(result.info)})

    return TaskStatusResponse(task_id=task_id, status=result.state)


# ── Bulk Email Trigger ────────────────────────────────────────────────────────
@router.post("/email/bulk", response_model=TaskResponse, status_code=202, summary="Send bulk emails")
def send_bulk_emails(request: BulkEmailRequest, auth: bool = Depends(verify_api_key)):
    from app.tasks.email_tasks import send_bulk_email_task
    task = send_bulk_email_task.delay(
        lead_ids=request.lead_ids,
        email_type=request.email_type,
        context=request.context,
    )
    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Queued emails to {len(request.lead_ids)} leads. Poll /tasks/{task.id} for progress.",
    )


# ── Single Email Trigger ──────────────────────────────────────────────────────
@router.post("/email/single", response_model=TaskResponse, status_code=202, summary="Send single email")
def send_single_email(
    to_email: str,
    subject: str,
    body: str,
    lead_id: Optional[int] = None,
    auth: bool = Depends(verify_api_key),
):
    from app.tasks.email_tasks import send_single_email_task
    task = send_single_email_task.delay(to_email=to_email, subject=subject, body=body, lead_id=lead_id)
    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Email queued for {to_email}. Poll /tasks/{task.id}.",
    )


# ── Scheduled Email (Mini Assignment) ────────────────────────────────────────
@router.post("/email/scheduled", response_model=TaskResponse, status_code=202, summary="Schedule email with delay")
def send_scheduled_email(
    to_email: str,
    subject: str,
    body: str,
    delay_minutes: int = 60,
    auth: bool = Depends(verify_api_key),
):
    """Mini Assignment: Schedule email using Celery countdown."""
    from app.tasks.email_tasks import send_single_email_task
    task = send_single_email_task.apply_async(
        args=[to_email, subject, body],
        countdown=delay_minutes * 60,
    )
    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Email scheduled to send in {delay_minutes} minutes. Task ID: {task.id}",
    )


# ── CSV Import ────────────────────────────────────────────────────────────────
@router.post("/leads/import", response_model=TaskResponse, status_code=202, summary="Import leads from CSV")
async def import_leads_csv(file: UploadFile = File(...), auth: bool = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    content = await file.read()
    csv_text = content.decode("utf-8")

    from app.tasks.lead_tasks import import_leads_csv_task
    task = import_leads_csv_task.delay(csv_content=csv_text, source=file.filename)
    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"CSV '{file.filename}' queued for import. Poll /tasks/{task.id}.",
    )


# ── Score All Leads ───────────────────────────────────────────────────────────
@router.post("/leads/score-all", response_model=TaskResponse, status_code=202, summary="Re-score all leads")
def score_all_leads(auth: bool = Depends(verify_api_key)):
    from app.tasks.lead_tasks import score_all_leads_task
    task = score_all_leads_task.delay()
    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Batch scoring queued. Poll /tasks/{task.id}.",
    )
