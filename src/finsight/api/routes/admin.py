"""Admin endpoints (ingest triggers, eval runs) — expanded in later steps."""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

logger = structlog.get_logger()
router = APIRouter()


class StatusResponse(BaseModel):
    status: str


@router.get("/status", response_model=StatusResponse)
async def admin_status() -> StatusResponse:
    return StatusResponse(status="ok")
