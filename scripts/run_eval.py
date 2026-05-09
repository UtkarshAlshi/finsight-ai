"""Run full evaluation suite and write markdown report to docs/eval_reports/."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog
import typer

logger = structlog.get_logger()
app = typer.Typer()


@app.command()
def main(
    output_dir: Path = typer.Option(Path("docs/eval_reports"), "--output", "-o"),
    dataset: Path = typer.Option(
        Path("src/finsight/eval/datasets/eval_questions.json"), "--dataset", "-d"
    ),
) -> None:
    """Run RAGAS + LLM-judge eval and write markdown report."""
    asyncio.run(_run(output_dir, dataset))


async def _run(output_dir: Path, dataset: Path) -> None:
    from finsight.config import get_settings
    from finsight.eval.llm_judge import run_llm_judge
    from finsight.eval.ragas_runner import run_ragas

    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("eval.start", dataset=str(dataset))

    ragas_metrics = await run_ragas(dataset, settings)
    judge_metrics = await run_llm_judge(dataset, settings)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_{timestamp}.md"

    report = _build_report(ragas_metrics, judge_metrics, timestamp)
    report_path.write_text(report)

    logger.info("eval.complete", report=str(report_path))
    print(f"\nReport written to: {report_path}")


def _build_report(ragas: dict, judge: dict, timestamp: str) -> str:
    return f"""# Eval Report — {timestamp}

## RAGAS Metrics

| Metric | Score |
|---|---|
| Faithfulness | {ragas.get('faithfulness', 'N/A'):.3f} |
| Answer Relevancy | {ragas.get('answer_relevancy', 'N/A'):.3f} |
| Context Precision | {ragas.get('context_precision', 'N/A'):.3f} |
| Context Recall | {ragas.get('context_recall', 'N/A'):.3f} |

## LLM Judge Metrics

| Metric | Score |
|---|---|
| Correctness | {judge.get('correctness', 'N/A'):.3f} |
| Completeness | {judge.get('completeness', 'N/A'):.3f} |
| Citation Quality | {judge.get('citation_quality', 'N/A'):.3f} |
"""


if __name__ == "__main__":
    app()
