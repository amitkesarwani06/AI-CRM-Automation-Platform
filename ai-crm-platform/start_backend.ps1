# AI CRM Platform — Backend Startup Script
# Run this from the ai-crm-platform directory

Write-Host "Starting AI CRM Platform Backend..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"
Write-Host "Virtual environment activated" -ForegroundColor Green
Write-Host "Starting FastAPI on http://localhost:8000" -ForegroundColor Yellow
Write-Host "Swagger docs at http://localhost:8000/docs" -ForegroundColor Yellow
python -m uvicorn app.api.main:app --reload --port 8000
