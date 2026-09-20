# Makefile — convenience commands
.PHONY: up down build logs clean

# Start all services
up:
	docker-compose up -d

# Start with logs visible
up-logs:
	docker-compose up

# Development mode with hot reload
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop all
down:
	docker-compose down

# Rebuild images
build:
	docker-compose build --no-cache

# View logs
logs:
	docker-compose logs -f

# API logs only
logs-api:
	docker-compose logs -f api

# Worker logs only
logs-worker:
	docker-compose logs -f worker

# Status
ps:
	docker-compose ps

# Shell into API container
shell-api:
	docker-compose exec api bash

# Shell into PostgreSQL
shell-db:
	docker-compose exec postgres psql -U crm_user -d crm_db

# Remove everything including volumes (CAREFUL — deletes data!)
clean:
	docker-compose down -v --remove-orphans
