"""POST /query endpoint with SSE streaming — wired in Step 5."""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = structlog.get_logger()
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, str]] = []
    session_id: str | None = None


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Placeholder — full agent graph wired in Step 5."""
    logger.info("query.received", question_len=len(request.question))
    return QueryResponse(
        answer="Agent graph not yet wired. Coming in Step 5.",
        session_id=request.session_id,
    )
