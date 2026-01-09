"""
DSPy Optimizer for NL2SQL
Configurable version with YAML config support.
"""

import argparse
import logging
import os
from collections import defaultdict

import dspy
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from dspy.utils.callback import BaseCallback
from tqdm import tqdm

from nl2sql.utils.util import (
    calculate_metrics,
    categorize_sql_complexity,
    compare_results,
    execute_sql,
    extract_sql_from_text,
    generate_markdown_report,
    get_db_path,
    load_schemas,
)

from .config import DSPyOptimizerConfig, cli_args_to_config_overrides, load_config
from .optimizers import OptimizerRegistry

logger = logging.getLogger(__name__)
SCHEMAS = load_schemas()


class LLMLogger(BaseCallback):
    """Callback to log LLM calls for debugging."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def on_lm_start(self, call_id, instance, inputs):
        if not self.enabled:
            return
        model_name = getattr(instance, "model", "Unknown Model")
        prompt = inputs.get("prompt") or inputs.get("messages")
        print(f"\n{'='*20} [LM CALL START] {'='*20}")
        print(f"MODEL:   {model_name}")
        print(f"CALL ID: {call_id}")
        print(f"PROMPT:  {str(prompt)}")

    def on_lm_end(self, call_id, outputs, exception, **kwargs):
        if not self.enabled:
            return
        instance = kwargs.get("instance")
        model_name = getattr(instance, "model", "Unknown Model")
        print(f"\n{'-'*20} [LM CALL END] {'-'*20}")
        print(f"MODEL:   {model_name}")
        print(f"CALL ID: {call_id}")
        if exception:
            print(f"ERROR:   {exception}")
        else:
            print(f"OUTPUT:  {outputs}")
        print(f"{'='*57}\n")


class TextToSQL(dspy.Signature):
    """Convert natural language to SQL given the schema. Return only the SQL query."""

    db_schema = dspy.InputField(desc="The Schema of the database tables")
    question = dspy.InputField(desc="The user's question to answer")
    sql = dspy.OutputField(desc="The executable SQL query string")


class SQLModule(dspy.Module):
    """DSPy module for Text-to-SQL generation."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.prog = dspy.ChainOfThought(TextToSQL) if use_cot else dspy.Predict(TextToSQL)

    def forward(self, db_schema, question):
        return self.prog(db_schema=db_schema, question=question)


def sql_metric(example, prediction, trace=None) -> float:
    """DSPy metric for optimization based on execution accuracy."""
    if not hasattr(prediction, "sql") or not prediction.sql:
        return 0.0

    pred_sql = extract_sql_from_text(prediction.sql)
    db_path = get_db_path(example.db_id)

    gold_success, _, gold_results = execute_sql(example.sql, db_path)
    pred_success, _, pred_results = execute_sql(pred_sql, db_path)

    if not gold_success or not pred_success:
        return 0.0

    match, _ = compare_results(pred_results, gold_results)
    return 1.0 if match else 0.0


def evaluate_module(module, devset, desc="Evaluating"):
    """Comprehensive evaluation with complexity tracking."""
    results = []
    complexity_metrics = defaultdict(lambda: {"total": 0, "valid": 0, "matched": 0})

    for example in tqdm(devset, desc=desc):
        prediction = module(db_schema=example.db_schema, question=example.question)
        pred_sql = (
            extract_sql_from_text(prediction.sql)
            if hasattr(prediction, "sql") and prediction.sql
            else ""
        )

        db_path = get_db_path(example.db_id)
        gold_success, _, gold_results = execute_sql(example.sql, db_path)
        pred_success, _, pred_results = execute_sql(pred_sql, db_path)

        results_match = False
        if gold_success and pred_success:
            results_match, _ = compare_results(pred_results, gold_results)

        categories = categorize_sql_complexity(example.sql)

        results.append(
            {
                "question": example.question,
                "db_id": example.db_id,
                "generated_sql": pred_sql,
                "gold_sql": example.sql,
                "is_valid": pred_success,
                "results_match": results_match,
                "complexity": categories,
            }
        )

        for cat in categories:
            complexity_metrics[cat]["total"] += 1
            if pred_success:
                complexity_metrics[cat]["valid"] += 1
            if results_match:
                complexity_metrics[cat]["matched"] += 1

    return results, calculate_metrics(results), complexity_metrics


def setup_logging(config: DSPyOptimizerConfig):
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    llm_logger = LLMLogger(enabled=config.logging.enable_llm_logger)
    dspy.configure(callbacks=[llm_logger])


def setup_lms(config: DSPyOptimizerConfig):
    """Initialize Student and Teacher LMs from config."""
    student_cfg = config.models.student
    student_lm = dspy.LM(
        model=student_cfg.name,
        api_base=student_cfg.api_base,
        api_key=student_cfg.api_key,
        max_tokens=student_cfg.max_tokens,
        temperature=student_cfg.temperature,
    )
    dspy.configure(lm=student_lm)

    teacher_lm = None
    if config.models.teacher.enabled:
        load_dotenv(find_dotenv())
        teacher_cfg = config.models.teacher
        teacher_lm = dspy.LM(
            model=teacher_cfg.name,
            api_key=teacher_cfg.api_key or os.getenv("NVIDIA_API_KEY"),
            api_base=teacher_cfg.api_base,
            temperature=teacher_cfg.temperature,
        )

    return student_lm, teacher_lm


def load_and_prep_data(config: DSPyOptimizerConfig):
    """Load and prepare data from config."""
    data_cfg = config.data

    full_data = load_dataset(
        data_cfg.dataset.name,
        data_files=data_cfg.dataset.train_file,
        split="train",
    )
    dev_raw = load_dataset(
        data_cfg.dataset.name,
        data_files=data_cfg.dataset.dev_file,
        split="train",
    )

    if data_cfg.preprocessing.shuffle:
        shuffled = full_data.shuffle(seed=data_cfg.preprocessing.seed)
    else:
        shuffled = full_data

    def to_dspy(subset):
        return [
            dspy.Example(
                db_schema=SCHEMAS.get(i["db_id"], f"Database: {i['db_id']}"),
                question=i["question"],
                sql=i["sql"],
                db_id=i["db_id"],
            ).with_inputs("db_schema", "question")
            for i in subset
        ]

    train_size = data_cfg.splits.train_size
    val_size = data_cfg.splits.val_size

    trainset = to_dspy(shuffled.select(range(0, train_size)))
    valset = to_dspy(shuffled.select(range(train_size, train_size + val_size)))
    devset = to_dspy(dev_raw)

    return trainset, valset, devset


def run_optimization(config: DSPyOptimizerConfig):
    """
    Run the full optimization pipeline with given config.

    Parameters
    ----------
    config : DSPyOptimizerConfig
        Validated configuration

    Returns
    -------
    tuple
        (compiled_module, optimized_metrics)
    """
    setup_logging(config)
    student_lm, teacher_lm = setup_lms(config)
    trainset, valset, devset = load_and_prep_data(config)

    b_metrics = None
    if config.evaluation.run_baseline:
        logger.info("Running Baseline Evaluation...")
        baseline = SQLModule(use_cot=config.module.use_cot)
        _, b_metrics, _ = evaluate_module(baseline, devset, "Baseline")
        logger.info(f"Baseline Result Match: {b_metrics['result_match_pct']:.2f}%")

    logger.info(f"Starting Optimization with {config.optimizer.name}...")

    optimizer_cls = OptimizerRegistry.get(config.optimizer.name)
    optimizer_wrapper = optimizer_cls(metric=sql_metric, config=config.optimizer)

    teacher_module = None
    if config.models.teacher.enabled and teacher_lm:
        teacher_module = SQLModule(use_cot=True)
        for p in teacher_module.predictors():
            p.lm = teacher_lm

    compiled = optimizer_wrapper.compile(
        student=SQLModule(use_cot=config.module.use_cot),
        teacher=teacher_module,
        trainset=trainset,
        valset=valset,
    )

    logger.info("Running Optimized Evaluation...")
    _, o_metrics, _ = evaluate_module(compiled, devset, "Optimized")

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    compiled.save(os.path.join(output_dir, "model.json"))

    generate_markdown_report(
        metrics=o_metrics,
        output_dir=output_dir,
        title=f"Optimization: {config.optimizer.name}",
        model_name=config.models.student.name,
        dataset_name="Spider Cleaned",
    )

    if b_metrics:
        improvement = o_metrics["result_match_pct"] - b_metrics["result_match_pct"]
        logger.info(f"Optimization Complete. Improvement: {improvement:+.2f}%")
    else:
        logger.info(
            f"Optimization Complete. Result Match: {o_metrics['result_match_pct']:.2f}%"
        )

    return compiled, o_metrics


def main():
    parser = argparse.ArgumentParser(description="DSPy NL2SQL Optimizer (Configurable)")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=OptimizerRegistry.list_available(),
    )
    parser.add_argument("--use_cot", action="store_true", default=None)
    parser.add_argument("--no_cot", action="store_true", help="Disable chain-of-thought")
    parser.add_argument("--use_teacher", action="store_true", default=None)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--val_size", type=int, default=None)

    args = parser.parse_args()

    if args.no_cot:
        args.use_cot = False

    cli_overrides = cli_args_to_config_overrides(args)
    config = load_config(config_path=args.config, cli_overrides=cli_overrides)

    run_optimization(config)


if __name__ == "__main__":
    main()
