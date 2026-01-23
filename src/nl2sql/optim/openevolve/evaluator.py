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

from datasets import load_dataset

# Reuse your project utilities (same ones you used in TextGrad).
# If these imports fail, fix PYTHONPATH or run from your repo root.
from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    compare_results,
    extract_sql_from_text,
    get_db_path,
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


def _load_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (train_split, dev_split).
    We use train for fast fitness sampling, dev for occasional reporting.
    """
    train = load_dataset(
        "AsadIsmail/nl2sql-deduplicated",
        data_files="spider_clean.jsonl",
        split="train",
    )
    dev = load_dataset(
        "AsadIsmail/nl2sql-deduplicated",
        data_files="spider_dev_clean.jsonl",
        split="train",
    )
    return list(train), list(dev)


def _ensure_schemas_loaded():
    global SCHEMAS
    if SCHEMAS is None:
        SCHEMAS = load_schemas()


def _predict_sql(student_engine, system_prompt: str, schema: str, question: str) -> str:
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    raw = student_engine(prompt, system_prompt)
    return extract_sql_from_text(raw)


def _evaluate_on_examples(
    student_engine,
    system_prompt: str,
    examples: List[Dict[str, Any]],
    max_examples: int,
    seed: int,
) -> Tuple[Dict[str, float], List[ExResult]]:
    _ensure_schemas_loaded()
    rng = random.Random(seed)
    if max_examples < len(examples):
        examples = rng.sample(examples, max_examples)

    results: List[ExResult] = []
    n = 0
    n_valid = 0
    n_match = 0
    n_gold_fail = 0

    for ex in examples:
        n += 1
        db_id = ex["db_id"]
        schema = SCHEMAS.get(db_id, "")
        q = ex["question"]
        gold_sql = ex["sql"]

        pred_sql = ""
        pred_success = False
        pred_error = ""
        pred_res = None

        try:
            pred_sql = _predict_sql(student_engine, system_prompt, schema, q)
            db_path = get_db_path(db_id)
            pred_success, pred_error, pred_res = execute_sql(pred_sql, db_path)
            gold_success, gold_error, gold_res = execute_sql(gold_sql, db_path)
            if not gold_success:
                n_gold_fail += 1
                # Skip gold failures from fitness so you don't optimize toward broken labels
                results.append(
                    ExResult(
                        db_id=db_id,
                        question=q,
                        pred_sql=pred_sql,
                        gold_sql=gold_sql,
                        pred_success=pred_success,
                        gold_success=False,
                        match=False,
                        pred_error=f"Gold failed: {gold_error}",
                    )
                )
                continue

            match = False
            if pred_success:
                n_valid += 1
                match, _ = compare_results(pred_res, gold_res)
                if match:
                    n_match += 1

            results.append(
                ExResult(
                    db_id=db_id,
                    question=q,
                    pred_sql=pred_sql,
                    gold_sql=gold_sql,
                    pred_success=pred_success,
                    gold_success=True,
                    match=match,
                    pred_error=pred_error,
                )
            )
        except Exception as e:
            results.append(
                ExResult(
                    db_id=db_id,
                    question=q,
                    pred_sql=pred_sql or "SELECT 'EVAL_EXCEPTION';",
                    gold_sql=gold_sql,
                    pred_success=False,
                    gold_success=True,
                    match=False,
                    pred_error=f"Exception: {e}",
                )
            )

    denom = max(1, (n - n_gold_fail))
    metrics = {
        "n": float(n),
        "n_gold_failed": float(n_gold_fail),
        "valid_pct": 100.0 * (n_valid / denom),
        "exec_match_pct": 100.0 * (n_match / denom),
    }
    return metrics, results


def _format_failure_artifacts(ex_results: List[ExResult], k: int = 5) -> str:
    fails = [r for r in ex_results if r.gold_success and (not r.match)]
    fails = fails[:k]
    blocks = []
    for r in fails:
        blocks.append(
            "\n".join(
                [
                    f"DB: {r.db_id}",
                    f"Q: {r.question}",
                    "Pred SQL:",
                    r.pred_sql,
                    "Gold SQL:",
                    r.gold_sql,
                    f"Pred exec ok: {r.pred_success}",
                    f"Pred error: {r.pred_error}",
                    f"Match: {r.match}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks) if blocks else "No failures in sampled set."


def evaluate(program_path: str) -> Any:
    """
    OpenEvolve entry point. `program_path` points to the evolved candidate (text prompt).
    Returns either EvaluationResult (if available) or a dict with a `score` key.
    """
    try:
        prompt_path = Path(program_path)
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            system_prompt = "Convert natural language to SQL. Output only the query."

        # ---- Student model access ----
        # We assume you're using an OpenAI-compatible local server (vLLM / SGLang / etc)
        # like in your TextGrad script.
        from openai import OpenAI

        api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        student_model = os.getenv("STUDENT_MODEL", "TheBloke/CodeLlama-7B-Instruct-AWQ")
        client = OpenAI(base_url=api_base, api_key=os.getenv("OPENAI_API_KEY", "dummy"), timeout=300.0)

        def student_engine(user_prompt: str, sys_prompt: str) -> str:
            resp = client.chat.completions.create(
                model=student_model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=int(os.getenv("STUDENT_MAX_TOKENS", "512")),
            )
            return resp.choices[0].message.content or ""

        train, _dev = _load_data()

        # Fitness sampling (keep it small; OpenEvolve runs many iterations)
        stage1_n = int(os.getenv("FITNESS_SAMPLES", "40"))
        seed = int(os.getenv("EVAL_SEED", "42"))

        metrics, ex_results = _evaluate_on_examples(
            student_engine=student_engine,
            system_prompt=system_prompt,
            examples=train,
            max_examples=stage1_n,
            seed=seed,
        )

        # Score in [0,1] (OpenEvolve expects higher=better)
        score = metrics["exec_match_pct"] / 100.0

        # Artifacts to support LLM feedback in OpenEvolve (when enabled in config).
        artifacts = {
            "summary": (
                f"exec_match_pct={metrics['exec_match_pct']:.2f} "
                f"valid_pct={metrics['valid_pct']:.2f} "
                f"n={int(metrics['n'])} gold_failed={int(metrics['n_gold_failed'])}"
            ),
            "failures": _format_failure_artifacts(ex_results, k=5),
            "prompt_char_len": len(system_prompt),
        }

        out_metrics = {
            "score": score,
            "exec_match_pct": metrics["exec_match_pct"],
            "combined_score": metrics["exec_match_pct"],
            "valid_pct": metrics["valid_pct"],
            "prompt_char_len": float(len(system_prompt)),
        }

        if EvaluationResult is not None:
            return EvaluationResult(metrics=out_metrics, artifacts=artifacts)

        out_metrics["artifacts"] = artifacts
        return out_metrics

    except Exception:
        tb = traceback.format_exc()
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"score": 0.0, "crash": 1.0}, artifacts={"traceback": tb})
        return {"score": 0.0, "crash": 1.0, "traceback": tb}
