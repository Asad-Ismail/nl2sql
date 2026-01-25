"""Unified evaluation interface for all optimizers."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from nl2sql.utils.util import (
    execute_sql,
    compare_results,
    extract_sql_from_text,
    get_db_path,
    calculate_metrics,
    categorize_sql_complexity,
)
from .optimizer_data import SCHEMAS
from .prompts import BASELINE_USER_PROMPT_TEMPLATE


@dataclass
class EvaluationResult:
    """Standardized evaluation result."""

    results: List[Dict[str, Any]]  # Individual example results
    metrics: Dict[str, float]  # Aggregated metrics
    complexity_metrics: Optional[Dict[str, Dict]] = None  # Per-complexity breakdown
    token_stats: Optional[Dict[str, int]] = None  # Token usage


def evaluate_sql_predictions(
    model,
    dataset: List[Dict],
    model_type: str = "textgrad",
) -> EvaluationResult:
    """
    Unified evaluation function for all optimizers.

    Parameters
    ----------
    model : object
        The model to evaluate (varies by optimizer type)
    dataset : list
        List of evaluation examples
    model_type : str
        Type of model ("textgrad", "dspy", "openevolve")

    Returns
    -------
    EvaluationResult
        Standardized evaluation results
    """
    results = []
    complexity_metrics = {}

    for ex in dataset:
        # Get prediction based on model type
        if model_type == "textgrad":
            schema = SCHEMAS.get(ex.get("db_id", ""), "")
            prompt = BASELINE_USER_PROMPT_TEMPLATE.format(schema=schema, question=ex.get('question', ''))
            prediction = model.forward(prompt)
            pred_sql = extract_sql_from_text(prediction.value)
        elif model_type == "dspy":
            prediction = model(db_schema=ex.db_schema, question=ex.question)
            pred_sql = extract_sql_from_text(prediction.sql)
        elif model_type == "openevolve":
            # OpenEvolve: model has forward() method returning object with .value
            schema = SCHEMAS.get(ex.get("db_id", ""), "")
            prompt = BASELINE_USER_PROMPT_TEMPLATE.format(schema=schema, question=ex.get('question', ''))
            prediction = model.forward(prompt)
            pred_sql = extract_sql_from_text(prediction.value)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Execute and compare
        gold_sql = ex.get("sql", "")
        db_id = ex.get("db_id", "")
        db_path = get_db_path(db_id)

        gold_success, _, gold_res = execute_sql(gold_sql, db_path)
        pred_success, error, pred_res = execute_sql(pred_sql, db_path)

        match = False
        if gold_success and pred_success:
            match, _ = compare_results(pred_res, gold_res)

        # Categorize complexity
        categories = categorize_sql_complexity(gold_sql)

        results.append(
            {
                "question": ex.get("question", ""),
                "db_id": db_id,
                "generated_sql": pred_sql,
                "gold_sql": gold_sql,
                "is_valid": pred_success,
                "results_match": match,
                "complexity": categories,
            }
        )

        # Track complexity metrics
        for cat in categories:
            if cat not in complexity_metrics:
                complexity_metrics[cat] = {"total": 0, "valid": 0, "matched": 0}
            complexity_metrics[cat]["total"] += 1
            if pred_success:
                complexity_metrics[cat]["valid"] += 1
            if match:
                complexity_metrics[cat]["matched"] += 1

    metrics = calculate_metrics(results)

    return EvaluationResult(
        results=results,
        metrics=metrics,
        complexity_metrics=complexity_metrics,
    )
