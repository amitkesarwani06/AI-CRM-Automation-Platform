import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Redis URL — used for both broker (queue) and backend (result storage)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "crm_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.email_tasks",
        "app.tasks.lead_tasks",
    ]
)

# Configuration settings
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,          # Results kept in Redis for 1 hour
    task_track_started=True,      # Track STARTED state
    worker_prefetch_multiplier=1, # Worker takes 1 task at a time
    task_acks_late=True,          # Acknowledge task only after completion
)

if __name__ == "__main__":
    celery_app.start()
