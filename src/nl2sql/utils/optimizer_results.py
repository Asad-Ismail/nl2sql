"""Unified results saving for all optimizers."""

import json
from pathlib import Path

from nl2sql.utils.util import generate_markdown_report
from .optimizer_eval import EvaluationResult


def save_evaluation_results(
    result: EvaluationResult,
    output_dir: str,
    title: str,
    model_name: str,
    dataset_name: str,
    artifacts: dict = None,
) -> None:
    """
    Save evaluation results in standardized format.

    Creates:
    - results.jsonl - Individual example results
    - metrics.json - Aggregated metrics
    - detailed_results.json - Full results with all details
    - report.md - Human-readable report
    - Additional artifacts (prompts, checkpoints, etc.)

    Parameters
    ----------
    result : EvaluationResult
        Evaluation result to save
    output_dir : str
        Directory to save results
    title : str
        Title for the report
    model_name : str
        Name of the model evaluated
    dataset_name : str
        Name of the dataset used
    artifacts : dict, optional
        Additional artifacts to save (prompts, checkpoints, etc.)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save individual results as JSONL
    results_file = out_path / "results.jsonl"
    with open(results_file, "w") as f:
        for r in result.results:
            f.write(json.dumps(r) + "\n")

    # Save metrics
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(result.metrics, f, indent=2)

    # Save detailed results
    detailed_file = out_path / "detailed_results.json"
    with open(detailed_file, "w") as f:
        json.dump(
            {
                "results": result.results,
                "complexity_metrics": result.complexity_metrics,
                "token_stats": result.token_stats,
            },
            f,
            indent=2,
        )

    # Generate markdown report
    generate_markdown_report(
        metrics=result.metrics,
        output_dir=str(out_path),
        title=title,
        model_name=model_name,
        dataset_name=dataset_name,
    )

    # Save additional artifacts
    if artifacts:
        for name, content in artifacts.items():
            artifact_file = out_path / name
            if isinstance(content, dict):
                artifact_file = artifact_file.with_suffix(".json")
                with open(artifact_file, "w") as f:
                    json.dump(content, f, indent=2)
            else:
                artifact_file.write_text(content)
