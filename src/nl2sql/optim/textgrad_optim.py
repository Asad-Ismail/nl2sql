import argparse
import os
import json
import logging
import random
import textgrad as tg
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv

# Shared utility imports
from nl2sql.utils.util import (
    load_schemas, execute_sql, compare_results, extract_sql_from_text,
    get_db_path, categorize_sql_complexity, calculate_metrics,
    save_evaluation_results, generate_markdown_report
)

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SCHEMAS = load_schemas()

# ==============================================================================
# Native Logger
# ==============================================================================

class LLMLogger:
    """Simple logger to track student and teacher calls."""
    def on_lm_start(self, role: str, model: str, prompt: str):
        print(f"\n{'='*20} [{role.upper()} CALL START] {'='*20}")
        print(f"MODEL: {model}")
        print(f"PROMPT SNIPPET: {str(prompt)}\n")

    def on_lm_end(self, role: str, output: str):
        print(f"\n{'-'*20} [{role.upper()} CALL END] {'-'*20}")
        print(f"OUTPUT: {output}\n")

llm_logger = LLMLogger()

# ==============================================================================
# Engines & Graph Wrappers
# ==============================================================================

class TaskEngine:
    def __init__(self, model, base_url):
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model

    def __call__(self, prompt: str, system_prompt: str) -> str:
        llm_logger.on_lm_start("student", self.model, f"System: {system_prompt}\nUser: {prompt}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        llm_logger.on_lm_end("student", content)
        return content

class NVIDIAGradientEngine:
    """Stable wrapper for NVIDIA/Teacher API for TextGrad."""
    def __init__(self, model, api_key):
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        self.model = model

    def __call__(self, prompt: str, **kwargs) -> str:
        llm_logger.on_lm_start("teacher", self.model, prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        llm_logger.on_lm_end("teacher", content)
        return content

class SQLModule(tg.Variable):
    def __init__(self, system_prompt: tg.Variable, engine: TaskEngine):
        super().__init__(system_prompt.value, role_description="SQL Generation Module")
        self.system_prompt = system_prompt
        self.engine = engine

    def forward(self, question_str: str) -> tg.Variable:
        response_text = self.engine(question_str, self.system_prompt.value)
        return tg.Variable(
            response_text,
            role_description="model_prediction",
            predecessors=[self.system_prompt]
        )

# ==============================================================================
# Optimization Helpers
# ==============================================================================

def evaluate(model, dataset, desc="Evaluating"):
    results = []
    for ex in tqdm(dataset, desc=desc, leave=False):
        schema = SCHEMAS.get(ex['db_id'], "No Schema")
        prompt = f"Schema:\n{schema}\n\nQuestion: {ex['question']}\nSQL:"
        prediction = model.forward(prompt)
        pred_sql = extract_sql_from_text(prediction.value)
        db_path = get_db_path(ex['db_id'])
        success, _, res = execute_sql(pred_sql, db_path)
        _, _, gold_res = execute_sql(ex['sql'], db_path)
        match, _ = compare_results(res, gold_res)
        results.append({"is_valid": success, "results_match": match})
    return calculate_metrics(results)

# ==============================================================================
# Main Optimizer
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=30)
    parser.add_argument("--num_val", type=int, default=20)
    parser.add_argument("--student_model", type=str, default="TheBloke/CodeLlama-7B-Instruct-AWQ")
    parser.add_argument("--eval_model", type=str, default="meta/llama-3.1-70b-instruct")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/textgrad_v3")
    args = parser.parse_args()

    load_dotenv(find_dotenv())
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    logger.info("Loading Datasets...")
    full_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
    dev_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")
    shuffled = full_data.shuffle(seed=42)
    
    train_set = [x for x in shuffled.select(range(0, args.num_train))]
    val_set = [x for x in shuffled.select(range(args.num_train, args.num_train + args.num_val))]
    test_set = [x for x in dev_data]

    # 2. Setup Engines
    task_engine = TaskEngine(args.student_model, args.api_base)
    eval_engine = NVIDIAGradientEngine(args.eval_model, os.getenv("NVIDIA_API_KEY"))
    tg.set_backward_engine(eval_engine)

    # 3. Variables & Optimizer
    initial_prompt = "Convert natural language to SQL. Output only the query."
    system_prompt = tg.Variable(initial_prompt, requires_grad=True, role_description="system prompt")
    model = SQLModule(system_prompt, task_engine)
    optimizer = tg.TGD(parameters=[system_prompt])

    loss_fn = tg.TextLoss("""You are a functional SQL evaluator. 
    1. The 'Gold SQL' is the absolute ground truth. NEVER critique it or suggest changes to it.
    2. Ignore efficiency differences (like Subqueries vs Joins) unless they cause a data mismatch.
    3. If 'Execution Match' is False, identify exactly what logic in the Student's SQL caused the mismatch.
    4. Provide feedback on how to update the 'System Prompt' to ensure the student follows the Gold SQL's logic exactly.""")

    best_val_acc = 0.0
    best_prompt = initial_prompt

    # 4. Optimization Loop
    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1} ---")
        random.shuffle(train_set)
        
        for step in range(0, len(train_set), args.batch_size):
            batch = train_set[step : step + args.batch_size]
            optimizer.zero_grad()
            losses = []
            
            for ex in batch:
                schema = SCHEMAS.get(ex['db_id'], "")
                prediction = model.forward(f"Schema:\n{schema}\nQuestion: {ex['question']}")
                pred_sql = extract_sql_from_text(prediction.value)
                
                # Execution Evidence
                success, error, res = execute_sql(pred_sql, get_db_path(ex['db_id']))
                match, _ = compare_results(res, execute_sql(ex['sql'], get_db_path(ex['db_id']))[2])
                
                evidence = f"Pred: {pred_sql}\nGold: {ex['sql']}\nMatch: {match}\nError: {error}"
                loss_input = tg.Variable(evidence, predecessors=[prediction], role_description="outcome")
                losses.append(loss_fn(loss_input))
            
            tg.sum(losses).backward()
            optimizer.step()

        # Checkpoint Revert Logic
        val_metrics = evaluate(model, val_set, desc="Val Check")
        if val_metrics['result_match_pct'] >= best_val_acc:
            best_val_acc = val_metrics['result_match_pct']
            best_prompt = copy.deepcopy(system_prompt.value)
            logger.info(f"Performance improved! Accuracy: {best_val_acc}%")
            logger.info(f"Best Prompt so far: {system_prompt}%")
        else:
            logger.warning(f"Accuracy dropped to {val_metrics['result_match_pct']}%. Reverting.")
            system_prompt.set_value(best_prompt)

    #  Final Report
    system_prompt.set_value(best_prompt)
    prompt_file = Path(args.output_dir) / "best_system_prompt.txt"
    prompt_file.write_text(best_prompt)
    logger.info(f"Best prompt saved to {prompt_file}")
    final_metrics = evaluate(model, test_set, desc="Final Test")
    generate_markdown_report(metrics=final_metrics, output_dir=args.output_dir, title="TextGrad SQL Results")

if __name__ == "__main__":
    main()