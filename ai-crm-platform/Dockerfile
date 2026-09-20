FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user (security best practice)
RUN useradd -m -u 1000 crmuser && chown -R crmuser:crmuser /app
USER crmuser

# Expose port
EXPOSE 8000

# Default command: run FastAPI
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
