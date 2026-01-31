"""TextGrad optimizer for NL2SQL with unified configuration and utilities."""

import argparse
import copy
import logging
import random
import textgrad as tg
from pathlib import Path

# Unified configuration and utilities
from nl2sql.utils import (
    load_optimizer_config,
    cli_args_to_dict,
    load_optimizer_data,
    evaluate_sql_predictions,
    save_evaluation_results,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_USER_PROMPT_TEMPLATE,
)
from nl2sql.llm import get_llm
from nl2sql.llm.base import LLMMessage
from nl2sql.utils.util import (
    extract_sql_from_text,
    execute_sql,
    compare_results,
    get_db_path,
    TokenStats,
    extract_token_usage,
    print_token_statistics,
    generate_optimizer_markdown_report,
)

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TaskEngine:
    """Student model engine using unified LLM provider."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.token_stats = TokenStats()

    def __call__(self, prompt: str, system_prompt: str) -> str:
        # Use generate() instead of generate_text() to get token usage
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=prompt)
        ]
        response = self.llm.generate(
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )

        # Track token usage
        usage = extract_token_usage(response.usage)
        self.token_stats.add(usage["prompt_tokens"], usage["completion_tokens"])

        return response.content or "SELECT 'Empty Response Error';"


class NVIDIAGradientEngine:
    """Teacher engine using unified LLM provider."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.token_stats = TokenStats()

    def __call__(self, prompt: str, **kwargs) -> str:
        # Use generate() instead of generate_text() to get token usage
        response = self.llm.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=2048,
            temperature=0.7,
        )

        # Track token usage
        usage = extract_token_usage(response.usage)
        self.token_stats.add(usage["prompt_tokens"], usage["completion_tokens"])

        return response.content


class SQLModule(tg.Variable):
    """TextGrad SQL module."""

    def __init__(self, system_prompt: tg.Variable, engine: TaskEngine):
        super().__init__(system_prompt.value, role_description="SQL Generation Module")
        self.system_prompt = system_prompt
        self.engine = engine

    def forward(self, question_str: str) -> tg.Variable:
        response_text = self.engine(question_str, self.system_prompt.value)
        return tg.Variable(
            response_text, role_description="model_prediction", predecessors=[self.system_prompt]
        )


def summarize(res, k=3):
    """Summarize results for loss input."""
    try:
        return res[:k]
    except Exception:
        return str(res)[:500]


def main():
    parser = argparse.ArgumentParser(description="TextGrad optimizer for NL2SQL")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")

    # Optional CLI overrides (use dot notation for nested keys)
    parser.add_argument("--data.train_size", type=int, dest="data_train_size")
    parser.add_argument("--data.val_size", type=int, dest="data_val_size")
    parser.add_argument("--student_model", type=str)
    parser.add_argument("--teacher_model", type=str)
    parser.add_argument("--train.epochs", type=int, dest="train_epochs")
    parser.add_argument("--train.batch_size", type=int, dest="train_batch_size")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    # Load configuration with CLI overrides
    cli_overrides = cli_args_to_dict(args)
    config = load_optimizer_config(args.config, cli_overrides)

    logger.info(f"Configuration loaded: {config.experiment.name}")
    logger.info(f"Output directory: {config.experiment.output_dir}")

    # Load data using unified data loader
    logger.info("Loading datasets...")
    train_set, val_set, test_set = load_optimizer_data(
        dataset_name=config.data.dataset_name,
        train_file=config.data.train_file,
        dev_file=config.data.dev_file,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        shuffle_seed=config.data.shuffle_seed,
        format="dict",
    )
    logger.info(f"Loaded {len(train_set)} train, {len(val_set)} val, {len(test_set)} test examples")

    # Setup engines
    task_engine = TaskEngine(config.models.student_model)
    if config.models.teacher_model:
        eval_engine = NVIDIAGradientEngine(config.models.teacher_model)
        tg.set_backward_engine(eval_engine)

    # Setup TextGrad variables and optimizer
    initial_prompt = BASELINE_SYSTEM_PROMPT
    system_prompt = tg.Variable(
        initial_prompt, requires_grad=True, role_description="system prompt"
    )
    model = SQLModule(system_prompt, task_engine)

    # Setup optimizer
    optimizer = tg.TGD(
        parameters=[system_prompt],
        verbose=1,
        gradient_memory=10,
        engine=eval_engine if config.models.teacher_model else task_engine,
        constraints=[
            "Output must be a single SYSTEM prompt.",
            "Must instruct: output ONLY SQL query, no explanation.",
        ],
    )

    # Setup loss function
    loss_fn = tg.TextLoss(
        """You are a functional SQL evaluator.
    1. The 'Gold SQL' is the absolute ground truth. NEVER critique it or suggest changes to it.
    2. Ignore efficiency differences (like Subqueries vs Joins) unless they cause a data mismatch.
    3. If 'Execution Match' is False, identify exactly what logic in the Student's SQL caused the mismatch.
    4. Provide feedback on how to update the 'System Prompt' to ensure the student follows the Gold SQL's logic exactly."""
    )

    # Baseline evaluation
    logger.info("Running baseline evaluation...")
    from nl2sql.utils.optimizer_eval import SCHEMAS

    baseline_result = evaluate_sql_predictions(model, test_set[:100], model_type="textgrad")
    logger.info(f"Baseline result match: {baseline_result.metrics['result_match_pct']:.2f}%")

    # Initialize best prompt
    best_val_result = evaluate_sql_predictions(model, val_set, model_type="textgrad")
    best_val_acc = best_val_result.metrics["result_match_pct"]
    best_prompt = initial_prompt
    logger.info(f"Baseline validation accuracy: {best_val_acc:.2f}%")

    # Optimization loop
    for epoch in range(config.training.epochs):
        logger.info(f"--- Epoch {epoch+1} ---")
        random.shuffle(train_set)

        for step in range(0, len(train_set), config.training.batch_size):
            batch = train_set[step : step + config.training.batch_size]
            optimizer.zero_grad()
            losses = []

            for ex in batch:
                schema = SCHEMAS.get(ex["db_id"], "")
                prompt = BASELINE_USER_PROMPT_TEMPLATE.format(schema=schema, question=ex['question'])
                prediction = model.forward(prompt)
                pred_sql = extract_sql_from_text(prediction.value)

                db_path = get_db_path(ex["db_id"])
                pred_success, error, pred_res = execute_sql(pred_sql, db_path)
                gold_success, _, gold_res = execute_sql(ex["sql"], db_path)

                if not gold_success:
                    continue

                match = False
                if gold_success and pred_success:
                    match, _ = compare_results(pred_res, gold_res)

                evidence = f"""DB: {ex['db_id']}
Schema:
{schema}

Question: {ex['question']}

Student SQL:
{pred_sql}

Gold SQL:
{ex['sql']}

Student exec ok: {pred_success}
Student error: {error}

Gold exec ok: {gold_success}

Student result sample: {summarize(pred_res)}
Gold result sample: {summarize(gold_res)}

Match: {match}""".strip()

                if (not pred_success) or (not match):
                    loss_input = tg.Variable(
                        evidence,
                        predecessors=[prediction, system_prompt],
                        role_description="sql_execution_outcome",
                    )
                    losses.append(loss_fn(loss_input))

            if not losses:
                continue
            tg.sum(losses).backward()
            optimizer.step()

            old = system_prompt.value
            new = system_prompt.value
            if new != old:
                logger.info(f"Prompt updated (len {len(old)} -> {len(new)})")

        # Validation checkpoint
        val_result = evaluate_sql_predictions(model, val_set, model_type="textgrad")
        val_acc = val_result.metrics["result_match_pct"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_prompt = copy.deepcopy(system_prompt.value)
            logger.info(f"Performance improved! Accuracy: {best_val_acc:.2f}%")
            logger.info(f"Best Prompt: {best_prompt}")
        else:
            logger.warning(f"Accuracy dropped to {val_acc:.2f}%. Reverting.")
            system_prompt.set_value(best_prompt)

    # Final evaluation and save results
    system_prompt.set_value(best_prompt)
    logger.info("Running final evaluation...")

    final_result = evaluate_sql_predictions(model, test_set, model_type="textgrad")
    logger.info(f"Final result match: {final_result.metrics['result_match_pct']:.2f}%")

    # Add token statistics
    final_result.token_stats = task_engine.token_stats.to_dict()
    if config.models.teacher_model:
        final_result.token_stats["teacher_tokens"] = eval_engine.token_stats.to_dict()

    # Save results using unified results saver
    artifacts = {"best_system_prompt.txt": best_prompt}

    save_evaluation_results(
        result=final_result,
        output_dir=config.experiment.output_dir,
        title="TextGrad Optimization Results",
        model_name=config.models.student_model,
        dataset_name=config.data.dataset_name,
        artifacts=artifacts,
    )

    logger.info(f"Results saved to {config.experiment.output_dir}")

    # Print console output and generate detailed markdown report with token breakdown
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}\n")

    print("TEXTGRAD")
    print(f"  Valid SQL: {final_result.metrics['valid_sql_count']}/{final_result.metrics['total_examples']} ({final_result.metrics['valid_sql_pct']:.1f}%)")
    print(f"  Results Match Gold: {final_result.metrics['result_match_count']}/{final_result.metrics['total_examples']} ({final_result.metrics['result_match_pct']:.1f}%)")
    print()

    # Extract student and teacher tokens
    student_tokens = final_result.token_stats
    teacher_tokens = final_result.token_stats.get("teacher_tokens") if config.models.teacher_model else None

    # Print token statistics
    print_token_statistics(
        student_tokens=student_tokens,
        teacher_tokens=teacher_tokens,
        title="TOKEN STATISTICS"
    )

    # Generate detailed markdown report with token breakdown
    generate_optimizer_markdown_report(
        metrics=final_result.metrics,
        student_tokens=student_tokens,
        teacher_tokens=teacher_tokens,
        output_dir=config.experiment.output_dir,
        title="TextGrad Optimization Results",
        model_name=config.models.student_model,
        dataset_name=config.data.dataset_name,
        optimizer_name="TextGrad",
    )

    logger.info(f"Detailed report saved to: {config.experiment.output_dir}/evaluation_report.md")


if __name__ == "__main__":
    main()
