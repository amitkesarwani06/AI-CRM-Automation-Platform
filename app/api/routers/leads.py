import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.api.schemas.lead import (
    LeadCreate,
    LeadUpdate,
    LeadResponse,
    LeadListResponse,
    LeadExtractRequest,
)
from app.api.dependencies import verify_api_key, get_db
from app.database.lead_repository import LeadRepository

router = APIRouter(prefix="/leads", tags=["Lead Management"])
logger = logging.getLogger(__name__)

# Cached LeadExtractor instance
_lead_extractor = None


def get_lead_extractor():
    global _lead_extractor
    if _lead_extractor is None:
        from app.agents.lead_extractor import LeadExtractor
        _lead_extractor = LeadExtractor()
    return _lead_extractor


# ── Create Lead ───────────────────────────────────────────────────────────────
@router.post(
    "/",
    response_model=LeadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new lead"
)
def create_lead(
    lead: LeadCreate,
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """Create a new lead in the CRM database."""
    try:
        repo = LeadRepository(db)
        new_lead = repo.create(
            name=lead.name,
            email=str(lead.email) if lead.email else None,
            phone=lead.phone,
            company=lead.company,
            plan_interest=lead.plan_interest,
            notes=lead.notes,
        )
        if lead.city:
            new_lead.city = lead.city
            db.commit()
            db.refresh(new_lead)

        logger.info(f"Lead created via API: #{new_lead.id} {new_lead.name}")
        return LeadResponse(**new_lead.to_dict())
    except Exception as e:
        logger.error(f"Error creating lead: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── List Leads ────────────────────────────────────────────────────────────────
@router.get(
    "/",
    response_model=LeadListResponse,
    summary="List all leads"
)
def list_leads(
    status: Optional[str] = Query(default=None, description="Filter by status (e.g. new, contacted, qualified)"),
    limit: int = Query(default=20, ge=1, le=100, description="Max number of leads to return"),
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """Get all leads with optional status filter and limit pagination."""
    try:
        repo = LeadRepository(db)
        if status:
            leads = repo.get_by_status(status)
        else:
            leads = repo.get_all(limit=limit)
        total = repo.count()
        return LeadListResponse(
            leads=[LeadResponse(**l.to_dict()) for l in leads],
            total=total,
        )
    except Exception as e:
        logger.error(f"Error listing leads: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Pipeline Stats Overview (Declared before /{lead_id} to avoid collision) ───
@router.get(
    "/stats/overview",
    summary="Get lead statistics"
)
def get_stats(
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """Get pipeline statistics — total leads and count by status."""
    try:
        repo = LeadRepository(db)
        return repo.get_stats()
    except Exception as e:
        logger.error(f"Error fetching stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Mini Assignment: Extract Lead from Text (Declared before /{lead_id}) ───────
@router.post(
    "/extract",
    response_model=LeadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Extract lead from text and save"
)
def extract_lead(
    request: LeadExtractRequest,
    save: bool = Query(default=True, description="Save extracted lead into database"),
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """
    Day 25 Mini Assignment:
    Accepts raw unstructured text (email, transcript, meeting note),
    runs Day 6's Pydantic LeadExtractor to parse structured fields,
    and saves the lead directly into the CRM database.
    """
    try:
        extractor = get_lead_extractor()
        lead_info = extractor.extract(request.text)

        if not lead_info or not lead_info.name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract valid lead information from the provided text."
            )

        repo = LeadRepository(db)
        new_lead = repo.create(
            name=lead_info.name,
            email=str(lead_info.email) if lead_info.email else None,
            phone=lead_info.phone,
            company=lead_info.company,
            plan_interest=None,
            notes=f"Extracted Intent: {lead_info.intent}",
        )
        # Apply score and city
        new_lead.score = lead_info.score
        new_lead.city = lead_info.city
        if hasattr(lead_info.status, "value"):
            new_lead.status = lead_info.status.value
        db.commit()
        db.refresh(new_lead)

        logger.info(f"Lead extracted and created via API: #{new_lead.id} {new_lead.name} (Score: {new_lead.score})")
        return LeadResponse(**new_lead.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during lead extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Get Lead by ID ────────────────────────────────────────────────────────────
@router.get(
    "/{lead_id}",
    response_model=LeadResponse,
    summary="Get a lead by ID"
)
def get_lead(
    lead_id: int,
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """Get a single lead by its database ID."""
    repo = LeadRepository(db)
    lead = repo.get_by_id(lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead #{lead_id} not found"
        )
    return LeadResponse(**lead.to_dict())


# ── Update Lead ───────────────────────────────────────────────────────────────
@router.patch(
    "/{lead_id}",
    response_model=LeadResponse,
    summary="Update a lead"
)
def update_lead(
    lead_id: int,
    update: LeadUpdate,
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """Update lead status, notes, score, or plan interest."""
    repo = LeadRepository(db)
    lead = repo.update(
        lead_id,
        status=update.status,
        notes=update.notes,
        plan_interest=update.plan_interest,
        score=update.score,
    )
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead #{lead_id} not found"
        )
    if update.city is not None:
        lead.city = update.city
        db.commit()
        db.refresh(lead)

    return LeadResponse(**lead.to_dict())


# ── Delete Lead ───────────────────────────────────────────────────────────────
@router.delete(
    "/{lead_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a lead"
)
def delete_lead(
    lead_id: int,
    db: Session = Depends(get_db),
    auth: bool = Depends(verify_api_key),
):
    """Delete a lead from the CRM database."""
    repo = LeadRepository(db)
    deleted = repo.delete(lead_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead #{lead_id} not found"
        )
    return None
