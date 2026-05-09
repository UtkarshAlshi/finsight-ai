"""Research agent: hybrid retrieval → LLM synthesis → citations."""

from __future__ import annotations

import structlog

from finsight.agents.state import GraphState
from finsight.config import Settings
from finsight.llm.router import LLMRouter, TaskComplexity
from finsight.rag.retriever import HybridRetriever

logger = structlog.get_logger()

_SYSTEM_PROMPT = """You are a financial research analyst. Answer the user's question
using ONLY the provided context excerpts. Each excerpt includes metadata (ticker,
section, filing date). Cite the source of each claim inline using [ticker:section] format.
If the context is insufficient, say so explicitly — do not hallucinate.
Be concise and precise. Output plain text, not markdown."""


async def research_node(state: GraphState, settings: Settings) -> dict[str, object]:
    """LangGraph node: retrieve context and synthesise a grounded draft answer."""
    question = state.question

    # Retrieve
    retriever = HybridRetriever(settings)
    chunks = await retriever.retrieve(question)
    logger.info("research.retrieved", chunks=len(chunks))

    # Build context string for the LLM
    context_parts: list[str] = []
    for chunk in chunks:
        ticker = chunk.get("ticker", "?")
        section = chunk.get("section", "?")
        date = chunk.get("filing_date", "?")
        text = chunk.get("text", "")
        context_parts.append(f"[{ticker}:{section} ({date})]\n{text}")
    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

    # Generate
    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.HIGH)
    draft = await client.generate(prompt, system=_SYSTEM_PROMPT)

    # Build citation list from top chunks
    citations = [
        {
            "ticker": str(c.get("ticker", "")),
            "section": str(c.get("section", "")),
            "filing_date": str(c.get("filing_date", "")),
            "accession_number": str(c.get("accession_number", "")),
            "excerpt": str(c.get("text", ""))[:200],
        }
        for c in chunks[:5]
    ]

    logger.info("research.draft_generated", draft_len=len(draft))
    return {
        "retrieved_context": chunks,
        "draft_answer": draft,
        "citations": citations,
    }
