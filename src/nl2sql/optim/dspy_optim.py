"""
DSPy Optimizer for NL2SQL
Enhanced version with CLI support for different models, optimizers, and teacher settings.
"""

import dspy
import os
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from dspy.utils.callback import BaseCallback
# utility imports
from nl2sql.utils.util import (
    load_schemas, execute_sql, compare_results, extract_sql_from_text,
    get_db_path, categorize_sql_complexity, calculate_metrics,
    save_evaluation_results, save_evaluation_summary, generate_markdown_report
)

# Configuration and Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SCHEMAS = load_schemas()

class LLMLogger(BaseCallback):
    def on_lm_start(self, call_id, instance, inputs):
        # Identify which model is being called
        model_name = getattr(instance, 'model', 'Unknown Model')
        prompt = inputs.get('prompt') or inputs.get('messages')
        
        print(f"\n{'='*20} [LM CALL START] {'='*20}")
        print(f"MODEL:   {model_name}")
        print(f"CALL ID: {call_id}")
        # Optionally truncate the prompt if it's too long for the console
        print(f"PROMPT:  {str(prompt)}") 

    def on_lm_end(self, call_id, outputs, exception, **kwargs):
        # We use **kwargs here to catch 'instance' and any other metadata
        # Access the instance from kwargs to identify the model
        instance = kwargs.get('instance')
        model_name = getattr(instance, 'model', 'Unknown Model')
        
        print(f"\n{'-'*20} [LM CALL END] {'-'*20}")
        print(f"MODEL:   {model_name}")
        print(f"CALL ID: {call_id}")
        
        if exception:
            print(f"ERROR:   {exception}")
        else:
            # outputs is usually a list of completions
            print(f"OUTPUT:  {outputs}")
        print(f"{'='*57}\n")


logger_cb = LLMLogger()
dspy.configure(callbacks=[logger_cb])

# ==============================================================================
# Model & Module Definitions
# ==============================================================================

class TextToSQL(dspy.Signature):
    """Convert natural language to SQL given the schema. Return only the SQL query."""
    db_schema = dspy.InputField(desc="The Schema of the database tables")
    question = dspy.InputField(desc="The user's question to answer")
    sql = dspy.OutputField(desc="The executable SQL query string")

class SQLModule(dspy.Module):
    def __init__(self, use_cot=True):
        super().__init__()
        self.prog = dspy.ChainOfThought(TextToSQL) if use_cot else dspy.Predict(TextToSQL)

    def forward(self, db_schema, question):
        return self.prog(db_schema=db_schema, question=question)

# ==============================================================================
# Metric & Evaluation
# ==============================================================================

def sql_metric(example, prediction, trace=None):
    """DSPy metric for optimization."""
    if not hasattr(prediction, 'sql') or not prediction.sql:
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
    complexity_metrics = defaultdict(lambda: {'total': 0, 'valid': 0, 'matched': 0})
    
    for example in tqdm(devset, desc=desc):
        prediction = module(db_schema=example.db_schema, question=example.question)
        pred_sql = extract_sql_from_text(prediction.sql) if hasattr(prediction, 'sql') and prediction.sql else ""
        
        db_path = get_db_path(example.db_id)
        gold_success, _, gold_results = execute_sql(example.sql, db_path)
        pred_success, _, pred_results = execute_sql(pred_sql, db_path)
        
        results_match = False
        if gold_success and pred_success:
            results_match, _ = compare_results(pred_results, gold_results)
        
        categories = categorize_sql_complexity(example.sql)
        
        results.append({
            "question": example.question,
            "db_id": example.db_id,
            "generated_sql": pred_sql,
            "gold_sql": example.sql,
            "is_valid": pred_success,
            "results_match": results_match,
            "complexity": categories
        })
        
        for cat in categories:
            complexity_metrics[cat]['total'] += 1
            if pred_success: complexity_metrics[cat]['valid'] += 1
            if results_match: complexity_metrics[cat]['matched'] += 1
            
    return results, calculate_metrics(results), complexity_metrics

# ==============================================================================
# Optimization Runner
# ==============================================================================

def get_optimizer(name, **kwargs):
    """Factory for DSPy optimizers."""
    if name == "BootstrapFewShot":
        return dspy.teleprompt.BootstrapFewShot(metric=sql_metric, **kwargs)
    elif name == "BootstrapFewShotWithRandomSearch":
        return dspy.teleprompt.BootstrapFewShotWithRandomSearch(metric=sql_metric, **kwargs)
    elif name == "MIPRO":
        return dspy.teleprompt.MIPROv2(metric=sql_metric, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def setup_lms(args):
    """Initialize Student and Teacher LMs based on CLI args."""
    # Student LM
    student_lm = dspy.LM(
        model=args.student_model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=0.0
    )
    dspy.configure(lm=student_lm)
    
    teacher_lm = None
    if args.use_teacher:
        load_dotenv(find_dotenv())
        teacher_key = os.getenv("NVIDIA_API_KEY") or args.api_key
        teacher_lm = dspy.LM(
            model=args.teacher_model,
            api_key=teacher_key,
            api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.7
        )
    return student_lm, teacher_lm

# ==============================================================================
# Data Loading
# ==============================================================================

def load_and_prep_data(train_size=3000, val_size=2000):
    full_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
    dev_raw = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")
    
    shuffled = full_data.shuffle(seed=42)
    
    def to_dspy(subset):
        return [dspy.Example(
            db_schema=SCHEMAS.get(i['db_id'], f"Database: {i['db_id']}"),
            question=i['question'], sql=i['sql'], db_id=i['db_id']
        ).with_inputs('db_schema', 'question') for i in subset]

    trainset = to_dspy(shuffled.select(range(0, train_size)))
    valset = to_dspy(shuffled.select(range(train_size, train_size + val_size)))
    devset = to_dspy(dev_raw)
    
    return trainset, valset, devset

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="DSPy NL2SQL Optimizer")
    parser.add_argument("--student_model", type=str, default="openai/TheBloke/CodeLlama-7B-Instruct-AWQ")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="dummy")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="BootstrapFewShotWithRandomSearch")
    parser.add_argument("--use_cot", action="store_true", default=True)
    parser.add_argument("--use_teacher", action="store_true")
    parser.add_argument("--teacher_model", type=str, default="moonshotai/kimi-k2-thinking")
    parser.add_argument("--output_dir", type=str, default="results/dspy_optimized_run")
    args = parser.parse_args()

    #  Setup
    student_lm, teacher_lm = setup_lms(args)
    trainset, valset, devset = load_and_prep_data()
    
    #  Baseline
    logger.info("Running Baseline Evaluation...")
    baseline = SQLModule(use_cot=args.use_cot)
    b_res, b_metrics, b_complexity = evaluate_module(baseline, devset, "Baseline")

    #  Optimization
    logger.info(f"Starting Optimization with {args.optimizer}...")
    optimizer = get_optimizer(
        args.optimizer, 
        max_bootstrapped_demos=2, 
        max_labeled_demos=2, 
        num_candidate_programs=5
    )
    
    teacher_module = None
    if args.use_teacher and teacher_lm:
        teacher_module = SQLModule(use_cot=True)
        for p in teacher_module.predictors():
            p.lm = teacher_lm

    compiled = optimizer.compile(
        student=SQLModule(use_cot=args.use_cot),
        teacher=teacher_module,
        trainset=trainset,
        valset=valset
    )

    # Final Evaluation
    logger.info("Running Optimized Evaluation...")
    o_res, o_metrics, o_complexity = evaluate_module(compiled, devset, "Optimized")

    #  Reporting & Saving
    os.makedirs(args.output_dir, exist_ok=True)
    compiled.save(os.path.join(args.output_dir, "model.json"))
    
    # Use utility function for the final MD report
    generate_markdown_report(
        metrics=o_metrics,
        output_dir=args.output_dir,
        title=f"Optimization: {args.optimizer}",
        model_name=args.student_model,
        dataset_name="Spider Cleaned"
    )
    
    logger.info(f"Optimization Complete. Improvement: {o_metrics['result_match_pct'] - b_metrics['result_match_pct']:+.2f}%")

if __name__ == "__main__":
    main()