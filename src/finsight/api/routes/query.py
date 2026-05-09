"""POST /query endpoint with SSE streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finsight.agents.graph import run_query
from finsight.config import get_settings
from finsight.llm.guardrails import append_disclaimer, apply_all

logger = structlog.get_logger()
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    stream: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, str]] = []
    session_id: str | None = None


@router.post("/query", response_model=None)
async def query(
    request_body: QueryRequest, http_request: Request
) -> StreamingResponse | QueryResponse:
    """Run multi-agent graph and return answer. Streams SSE when stream=True."""
    settings = get_settings()

    # Guardrails on input
    safe_question = apply_all(request_body.question)

    if request_body.stream:
        return StreamingResponse(
            _sse_stream(safe_question, settings, request_body.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path
    final_state = await run_query(safe_question, settings, request_body.session_id)
    answer = append_disclaimer(final_state.final_answer or final_state.draft_answer)
    return QueryResponse(
        answer=answer,
        citations=final_state.citations,
        session_id=request_body.session_id,
    )


async def _sse_stream(
    question: str,
    settings: object,
    session_id: str | None,
) -> AsyncIterator[str]:
    from finsight.config import Settings

    assert isinstance(settings, Settings)
    try:
        final_state = await run_query(question, settings, session_id)
        answer = final_state.final_answer or final_state.draft_answer

        # Stream answer in chunks
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            token = answer[i : i + chunk_size]
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Final event with citations
        from finsight.llm.guardrails import FINANCIAL_DISCLAIMER

        yield f"data: {json.dumps({'token': FINANCIAL_DISCLAIMER})}\n\n"
        yield f"data: {json.dumps({'done': True, 'citations': final_state.citations})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.error("query.stream_error", error=str(exc))
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
