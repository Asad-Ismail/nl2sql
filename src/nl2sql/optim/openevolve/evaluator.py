"""
OpenEvolve evaluator for NL2SQL prompt evolution.

- The "program" being evolved is a TEXT FILE containing a system prompt.
  OpenEvolve will write candidates to disk and pass the path here.

- Fitness is execution-based: percentage of examples where predicted SQL executes
  and matches the gold query's result (execution match).
"""

from __future__ import annotations

import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from datasets import load_dataset

# Unified LLM provider and utilities
from nl2sql.llm import get_llm
from nl2sql.llm.base import LLMMessage
from nl2sql.utils import load_optimizer_data, evaluate_sql_predictions, save_evaluation_results, BASELINE_SYSTEM_PROMPT
from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    compare_results,
    extract_sql_from_text,
    get_db_path,
    TokenStats,
    extract_token_usage,
    print_token_statistics,
    generate_optimizer_markdown_report,
)

# Optional: OpenEvolve structured result (newer versions).
try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
except Exception:
    EvaluationResult = None  # type: ignore


SCHEMAS = None


@dataclass
class ExResult:
    db_id: str
    question: str
    pred_sql: str
    gold_sql: str
    pred_success: bool
    gold_success: bool
    match: bool
    pred_error: str


def _predict_sql(student_engine, system_prompt: str, schema: str, question: str) -> str:
    """Predict SQL from question using student engine."""
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    raw = student_engine(prompt, system_prompt)
    return extract_sql_from_text(raw)


class OpenEvolveModelWrapper:
    """Wrapper to make OpenEvolve student engine compatible with unified evaluation."""

    def __init__(self, llm, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.token_stats = TokenStats()

    def forward(self, prompt: str):
        """Forward method compatible with evaluate_sql_predictions."""
        # Use generate() instead of generate_text() to get token usage
        response = self.llm.generate(
            messages=[
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=prompt)
            ],
            max_tokens=512,
            temperature=0.0,
        )

        # Track token usage
        usage = extract_token_usage(response.usage)
        self.token_stats.add(usage["prompt_tokens"], usage["completion_tokens"])

        # Return a simple object with .value attribute for compatibility
        class Response:
            def __init__(self, content):
                self.value = content
        return Response(response.content)


def evaluate(program_path: str) -> Any:
    """
    OpenEvolve entry point. `program_path` points to the evolved candidate (text prompt).
    Returns either EvaluationResult (if available) or a dict with a `score` key.
    """
    # Load .env files from current directory AND project root
    load_dotenv()  # Current directory
    load_dotenv(Path(__file__).parent.parent.parent / ".env")  # Project root

    try:
        prompt_path = Path(program_path)
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            system_prompt = BASELINE_SYSTEM_PROMPT

        # Student model access
        student_model_name = os.getenv("STUDENT_MODEL", "codellama_7b")
        llm = get_llm(student_model_name)

        # Load data using unified data loader
        train, _, dev = load_optimizer_data(format="dict")

        # Fitness sampling
        stage1_n = int(os.getenv("FITNESS_SAMPLES", "40"))
        eval_subset = train[:stage1_n]

        # Create model wrapper for unified evaluation (pass llm directly)
        model_wrapper = OpenEvolveModelWrapper(llm, system_prompt)

        # Evaluate using unified evaluation function
        from nl2sql.utils.optimizer_eval import EvaluationResult as UnifiedEvalResult

        eval_result = evaluate_sql_predictions(model_wrapper, eval_subset, model_type="openevolve")

        # Add token statistics
        eval_result.token_stats = model_wrapper.token_stats.to_dict()

        # Extract metrics
        metrics = eval_result.metrics
        score = metrics.get("result_match_pct", 0.0) / 100.0

        # Build output metrics for OpenEvolve
        out_metrics = {
            "score": score,
            "exec_match_pct": metrics.get("result_match_pct", 0.0),
            "combined_score": metrics.get("result_match_pct", 0.0),
            "valid_pct": metrics.get("valid_sql_pct", 0.0),
            "prompt_char_len": float(len(system_prompt)),
        }

        # Save results if OUTPUT_DIR is set
        output_dir = os.getenv("OUTPUT_DIR")
        if output_dir:
            print(f"\nSaving results to {output_dir}...")
            artifacts = {"best_system_prompt.txt": system_prompt}

            save_evaluation_results(
                result=eval_result,
                output_dir=output_dir,
                title="OpenEvolve Optimization Results",
                model_name=student_model_name,
                dataset_name="Spider (sampled)",
                artifacts=artifacts,
            )

            # Print console output and generate detailed markdown report with token breakdown
            print(f"\n{'='*60}")
            print("OPTIMIZATION RESULTS")
            print(f"{'='*60}\n")

            print("OPENEVOLVE")
            print(f"  Valid SQL: {eval_result.metrics['valid_sql_count']}/{eval_result.metrics['total_examples']} ({eval_result.metrics['valid_sql_pct']:.1f}%)")
            print(f"  Results Match Gold: {eval_result.metrics['result_match_count']}/{eval_result.metrics['total_examples']} ({eval_result.metrics['result_match_pct']:.1f}%)")
            print()

            # Print token statistics (OpenEvolve has no teacher model)
            print_token_statistics(
                student_tokens=eval_result.token_stats,
                teacher_tokens=None,  # OpenEvolve has no teacher model
                title="TOKEN STATISTICS"
            )

            # Generate detailed markdown report
            generate_optimizer_markdown_report(
                metrics=eval_result.metrics,
                student_tokens=eval_result.token_stats,
                teacher_tokens=None,  # OpenEvolve has no teacher model
                output_dir=output_dir,
                title="OpenEvolve Optimization Results",
                model_name=student_model_name,
                dataset_name="Spider (sampled)",
                optimizer_name="OpenEvolve",
            )

            logger.info(f"Detailed report saved to: {output_dir}/evaluation_report.md")

        if EvaluationResult is not None:
            return EvaluationResult(metrics=out_metrics)

        return out_metrics

    except Exception:
        tb = traceback.format_exc()
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"score": 0.0, "crash": 1.0}, artifacts={"traceback": tb})
        return {"score": 0.0, "crash": 1.0, "traceback": tb}
