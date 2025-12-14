"""
DSPy Optimizer for NL2SQL - Simple Version
"""
import dspy
import os
import sqlite3
import logging
import json
from pathlib import Path
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from datasets import load_dataset
from tqdm import tqdm
from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    print_comparison,
    compare_results,
    extract_sql_from_text,
    get_db_path,
    categorize_sql_complexity,
    calculate_metrics,
    save_evaluation_results,
    save_evaluation_summary,
    generate_markdown_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load schemas once using shared utility
SCHEMAS = load_schemas()

# ==============================================================================
# LM Setup
# ==============================================================================
lm = dspy.LM(
    model="openai/TheBloke/CodeLlama-7B-Instruct-AWQ",
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    max_tokens=500,
    temperature=0.0
)
dspy.configure(lm=lm)

# ==============================================================================
# Module
# ==============================================================================
class TextToSQL(dspy.Signature):
    """Convert natural language to SQL."""
    db_schema = dspy.InputField()
    question = dspy.InputField()
    sql = dspy.OutputField()

class SQLModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(TextToSQL)

    def forward(self, db_schema, question):
        return self.prog(db_schema=db_schema, question=question)

# ==============================================================================
# Metric
# ==============================================================================

def metric(example, prediction, trace=None):
    """Evaluate prediction using shared utility functions"""
    if not hasattr(prediction, 'sql') or not prediction.sql:
        return 0.0
    
    # Clean the generated SQL using shared utility
    pred_sql = extract_sql_from_text(prediction.sql)
    
    # Execute both queries using shared utilities
    db_path = get_db_path(example.db_id)
    
    gold_success, gold_error, gold_results = execute_sql(example.sql, db_path)
    pred_success, pred_error, pred_results = execute_sql(pred_sql, db_path)
    
    if not gold_success or not pred_success:
        print("❌ Execution failed")
        return 0.0
    
    # Compare using shared utility
    results_match, feedback = compare_results(pred_results, gold_results)
    
    if results_match:
        print("✅ Match!")
        return 1.0
    else:
        print("❌ Mismatch")
        return 0.0

# ==============================================================================
# Data
# ==============================================================================


def convert_to_dspy(dataset_split):
    """
    Just converts a dataset list/split to DSPy examples.
    No shuffling, no limiting. We do that beforehand.
    """
    examples = []
    for item in dataset_split:
        db_id = item['db_id']
        schema = SCHEMAS.get(db_id, f"Database: {db_id}")
        
        examples.append(dspy.Example(
            db_schema=schema,
            question=item['question'],
            sql=item['sql'],
            db_id=db_id
        ).with_inputs('db_schema', 'question'))
    return examples


def generate_optimization_report(baseline_results, baseline_metrics, baseline_complexity,
                                 optimized_results, optimized_metrics, optimized_complexity,
                                 output_dir="results/dspy_optimized"):
    """Generate comprehensive optimization report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    save_evaluation_results(baseline_results, output_dir, prefix="baseline_results")
    save_evaluation_results(optimized_results, output_dir, prefix="optimized_results")
    
    # Save summaries
    save_evaluation_summary(baseline_metrics, output_dir, 
                           additional_data={"complexity_metrics": dict(baseline_complexity)})
    
    # Build comparison sections
    comparison_table = """| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
"""
    
    comparison_table += f"| Valid SQL | {baseline_metrics['valid_sql_pct']:.1f}% | {optimized_metrics['valid_sql_pct']:.1f}% | "
    comparison_table += f"{optimized_metrics['valid_sql_pct'] - baseline_metrics['valid_sql_pct']:+.1f}% |\n"
    
    comparison_table += f"| Results Match | {baseline_metrics['result_match_pct']:.1f}% | {optimized_metrics['result_match_pct']:.1f}% | "
    comparison_table += f"{optimized_metrics['result_match_pct'] - baseline_metrics['result_match_pct']:+.1f}% |\n"
    
    comparison_table += f"| Total Examples | {baseline_metrics['total_examples']} | {optimized_metrics['total_examples']} | - |\n"
    
    # Complexity breakdown
    complexity_section = "### Baseline (Before Optimization)\n\n"
    for category, stats in sorted(baseline_complexity.items()):
        if stats['total'] > 0:
            complexity_section += f"**{category.upper().replace('_', ' ')}**: "
            complexity_section += f"Valid: {stats['valid']}/{stats['total']} ({100*stats['valid']/stats['total']:.1f}%), "
            complexity_section += f"Match: {stats['matched']}/{stats['total']} ({100*stats['matched']/stats['total']:.1f}%)\n\n"
    
    complexity_section += "\n### Optimized (After Optimization)\n\n"
    for category, stats in sorted(optimized_complexity.items()):
        if stats['total'] > 0:
            complexity_section += f"**{category.upper().replace('_', ' ')}**: "
            complexity_section += f"Valid: {stats['valid']}/{stats['total']} ({100*stats['valid']/stats['total']:.1f}%), "
            complexity_section += f"Match: {stats['matched']}/{stats['total']} ({100*stats['matched']/stats['total']:.1f}%)\n\n"
    
    # Generate report
    sections = {
        "Performance Comparison": comparison_table,
        "Performance by SQL Complexity": complexity_section,
        "Notes": """- **Baseline**: Zero-shot predictions without optimization
- **Optimized**: After DSPy BootstrapFewShot optimization
- **Metric**: Result match percentage (execution results identical to gold standard)
- **Optimization**: Uses few-shot examples selected automatically by DSPy"""
    }
    
    report_path = generate_markdown_report(
        metrics={},
        output_dir=output_dir,
        title="DSPy Optimization Results",
        model_name="CodeLlama-7B-Instruct (via vLLM)",
        dataset_name="Spider (Train subset for optimization, Dev for evaluation)",
        additional_sections=sections
    )
    
    logger.info(f"\n✅ Optimization report saved to: {report_path}")
    return report_path


def load_data():
    # 1. Load the raw dataset
    full_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
    dev_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")

    # 2. SHUFFLE ONCE GLOBALLY
    # This ensures the order is fixed before we start slicing
    shuffled_data = full_data.shuffle(seed=42)

    # 3. Create disjoint slices using indices
    # Train: 0 to 2000
    train_slice = shuffled_data.select(range(0, 2000))
    
    # Opt Val: 2000 to 2100 (Guaranteed to be different from 0-2000)
    val_slice = shuffled_data.select(range(2000, 2100))

    print(f"Split sizes -> Train: {len(train_slice)}, Val: {len(val_slice)}")

    # 4. Convert to DSPy
    trainset = convert_to_dspy(train_slice)
    valset = convert_to_dspy(val_slice)
    devset = convert_to_dspy(dev_data) # Keep dev set full

    return trainset, valset, devset

    

def evaluate(module, devset, print_every=10):
    """Evaluate module with detailed metrics and complexity tracking"""
    from collections import defaultdict
    
    results = []
    complexity_metrics = defaultdict(lambda: {'total': 0, 'valid': 0, 'matched': 0})
    
    for i, example in enumerate(tqdm(devset, desc="Evaluating")):
        prediction = module(db_schema=example.db_schema, question=example.question)
        
        # Extract SQL
        pred_sql = extract_sql_from_text(prediction.sql) if hasattr(prediction, 'sql') and prediction.sql else ""
        
        # Execute and compare
        db_path = get_db_path(example.db_id)
        gold_success, gold_error, gold_results = execute_sql(example.sql, db_path)
        pred_success, pred_error, pred_results = execute_sql(pred_sql, db_path)
        
        results_match = False
        if gold_success and pred_success:
            results_match, _ = compare_results(pred_results, gold_results)
        
        # Categorize complexity
        complexity_categories = categorize_sql_complexity(example.sql)
        
        # Store result
        result = {
            "question": example.question,
            "db_id": example.db_id,
            "generated_sql": pred_sql,
            "gold_sql": example.sql,
            "is_valid": pred_success,
            "results_match": results_match,
            "complexity": complexity_categories
        }
        results.append(result)
        
        # Update complexity metrics
        for category in complexity_categories:
            complexity_metrics[category]['total'] += 1
            if pred_success:
                complexity_metrics[category]['valid'] += 1
            if results_match:
                complexity_metrics[category]['matched'] += 1
        
        # Print progress
        if (i + 1) % print_every == 0:
            matched = sum(1 for r in results if r['results_match'])
            valid = sum(1 for r in results if r['is_valid'])
            print(f"\nProgress: Valid {valid}/{len(results)} ({100*valid/len(results):.1f}%), Match {matched}/{len(results)} ({100*matched/len(results):.1f}%)")
    
    # Calculate overall metrics
    metrics = calculate_metrics(results)
    
    return results, metrics, complexity_metrics


# ==============================================================================
# Main
# ==============================================================================
def main():
    logger.info("Loading schemas...")
    logger.info(f"Loaded {len(SCHEMAS)} database schemas")
    
    logger.info("Loading data...")
    trainset, valset, devset = load_data()
    logger.info(f"Loaded {len(trainset)} train, {len(valset)} val, {len(devset)} test")
    
    # Evaluate baseline (before optimization)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING BASELINE (Before Optimization)")
    logger.info("="*80)
    baseline = SQLModule()
    baseline_results, baseline_metrics, baseline_complexity = evaluate(baseline, devset)
    
    logger.info(f"\nBaseline Results:")
    logger.info(f"  Valid SQL: {baseline_metrics['valid_sql_count']}/{baseline_metrics['total_examples']} ({baseline_metrics['valid_sql_pct']:.1f}%)")
    logger.info(f"  Results Match: {baseline_metrics['result_match_count']}/{baseline_metrics['total_examples']} ({baseline_metrics['result_match_pct']:.1f}%)")
    
    # Run optimization
    logger.info("\n" + "="*80)
    logger.info("RUNNING OPTIMIZATION")
    logger.info("="*80)
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=3,
        num_candidate_programs=5
    )
    
    compiled = optimizer.compile(SQLModule(), trainset=trainset, valset=valset)
    
    # Evaluate optimized model (after optimization)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING OPTIMIZED MODEL (After Optimization)")
    logger.info("="*80)
    optimized_results, optimized_metrics, optimized_complexity = evaluate(compiled, devset)
    
    logger.info(f"\nOptimized Results:")
    logger.info(f"  Valid SQL: {optimized_metrics['valid_sql_count']}/{optimized_metrics['total_examples']} ({optimized_metrics['valid_sql_pct']:.1f}%)")
    logger.info(f"  Results Match: {optimized_metrics['result_match_count']}/{optimized_metrics['total_examples']} ({optimized_metrics['result_match_pct']:.1f}%)")
    
    # Print final comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)
    logger.info(f"Valid SQL:      {baseline_metrics['valid_sql_pct']:.1f}% → {optimized_metrics['valid_sql_pct']:.1f}% ({optimized_metrics['valid_sql_pct'] - baseline_metrics['valid_sql_pct']:+.1f}%)")
    logger.info(f"Results Match:  {baseline_metrics['result_match_pct']:.1f}% → {optimized_metrics['result_match_pct']:.1f}% ({optimized_metrics['result_match_pct'] - baseline_metrics['result_match_pct']:+.1f}%)")
    logger.info("="*80)
    
    # Save optimized model
    logger.info("\nSaving optimized model...")
    os.makedirs("models/dspy_optimized", exist_ok=True)
    compiled.save("models/dspy_optimized/nl2sql.json")
    logger.info("✅ Model saved to: models/dspy_optimized/nl2sql.json")
    
    # Generate comprehensive report
    logger.info("\nGenerating optimization report...")
    generate_optimization_report(
        baseline_results, baseline_metrics, baseline_complexity,
        optimized_results, optimized_metrics, optimized_complexity
    )
    
    logger.info("\n✅ Optimization complete!")
    

if __name__ == "__main__":
    main()