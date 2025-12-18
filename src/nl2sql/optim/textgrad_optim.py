"""
TextGrad Prompt Optimizer for NL2SQL

Uses TextGrad library to optimize system prompts for text-to-SQL generation
through gradient-based prompt engineering. Evaluates on Spider dev set with
complexity tracking and detailed reporting.

Architecture (following TextGrad best practices):
- Task Engine: vLLM (CodeLlama) - the model being optimized for production use
- Eval Engine: NVIDIA API (Qwen/Llama models) - generates high-quality textual gradients
  
This dual-engine approach is recommended because:
1. Evaluation needs strong reasoning to provide useful feedback
2. You optimize the cheaper/faster model you'll actually deploy
3. More cost-effective than using powerful models only for gradients

Set NVIDIA_API_KEY environment variable for the evaluation engine.
"""
import argparse
import concurrent.futures
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import textgrad as tg
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    compare_results,
    extract_sql_from_text,
    get_db_path,
    categorize_sql_complexity,
    calculate_metrics,
    save_evaluation_results,
    save_evaluation_summary,
    generate_markdown_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load schemas once using shared utility
SCHEMAS = load_schemas()

# Default system prompt - matches baseline.py zero-shot for fair comparison
DEFAULT_SYSTEM_PROMPT = """### Task: Convert the following natural language question to a SQL query.

Given a database schema and a question, generate the SQL query that answers the question.
Output only the SQL query without explanations."""


class TextGradVLLMEngine:
    """Wrapper for vLLM to work with TextGrad (Task/Generation Engine)"""
    
    def __init__(self, model: str = "TheBloke/CodeLlama-7B-Instruct-AWQ", 
                 base_url: str = "http://localhost:8000/v1",
                 max_tokens: int = 1024,
                 temperature: float = 0.1):
        """
        Initialize vLLM engine for TextGrad.
        
        Parameters:
            model: Model name on vLLM server
            base_url: vLLM server URL
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy"
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate text using vLLM.
        
        Parameters:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    def __call__(self, input_var):
        """TextGrad-compatible call interface"""
        if hasattr(input_var, 'value'):
            return self.generate(input_var.value)
        return self.generate(str(input_var))


class NVIDIATextGradEngine:
    """NVIDIA API engine for TextGrad (Evaluation/Gradient Engine)"""
    
    def __init__(
        self,
        model: str = "meta/llama-3.1-70b-instruct",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_tokens: int = 12024,
        temperature: float = 0.7
    ):
        """
        Initialize NVIDIA API engine for TextGrad evaluation.
        
        Parameters:
            model: NVIDIA model name (e.g., meta/llama-3.1-70b-instruct, 
                   qwen/qwen3-next-80b-a3b-thinking, nvidia/llama-3.1-nemotron-70b-instruct)
            api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
            base_url: NVIDIA API base URL
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable or api_key parameter required")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"Initialized NVIDIA engine with model: {model}")

    @sleep_and_retry              
    @limits(calls=38, period=60)
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using NVIDIA API.
        
        Parameters:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"NVIDIA API error: {e}")
            raise
    
    def __call__(self, *args, **kwargs):
        """
        TextGrad-compatible call interface.
        
        Supports both direct string input and Variable objects.
        """
        # Handle Variable objects
        if args and hasattr(args[0], 'value'):
            prompt = args[0].value
        elif args:
            prompt = str(args[0])
        else:
            prompt = kwargs.get('prompt', '')
        
        system_prompt = kwargs.get('system_prompt', None)
        return self.generate(prompt, system_prompt)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)


def evaluate_single_example(
    example: Dict[str, Any],
    model,
    system_prompt: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate a single example using shared utilities.
    
    Parameters:
        example: Dict with 'question', 'sql', 'db_id', 'context'
        model: TextGrad model
        system_prompt: Current system prompt (unused but kept for signature consistency)
        
    Returns:
        Tuple of (success, result_dict)
    """
    question = example["question"]
    gold_sql = example["sql"]
    db_id = example["db_id"]
    
    # Get schema using shared utility - matches baseline.py and DSPy exactly
    schema = SCHEMAS.get(db_id, f"Database: {db_id}")
    
    # Build prompt with schema - matches baseline.py format exactly
    prompt = f"""### Database Schema:
{schema}

### Question: {question}

### SQL Query:
"""
    
    # Generate SQL using TextGrad Variable
    x = tg.Variable(prompt, requires_grad=False, role_description="SQL generation prompt")
    response = model(x)
    pred_sql = extract_sql_from_text(response.value if hasattr(response, 'value') else response)
    
    # Execute both queries
    db_path = get_db_path(db_id)
    gold_success, gold_error, gold_results = execute_sql(gold_sql, db_path)
    pred_success, pred_error, pred_results = execute_sql(pred_sql, db_path)
    
    # Get complexity from gold SQL
    complexity = categorize_sql_complexity(gold_sql)
    # Handle both string and list returns
    if isinstance(complexity, list):
        complexity = complexity[0] if complexity else "UNKNOWN"
    
    # Compare results
    match = False
    if gold_success and pred_success:
        match, _ = compare_results(pred_results, gold_results)
    
    result = {
        "question": question,
        "gold_sql": gold_sql,
        "predicted_sql": pred_sql,
        "db_id": db_id,
        "execution_success": pred_success,
        "execution_error": pred_error if not pred_success else None,
        "results_match": match,
        "complexity": complexity
    }
    
    return match, result


def evaluate_dataset(
    dataset: List[Dict],
    model,
    system_prompt: str,
    max_samples: int = None,
    desc: str = "Evaluating"
) -> Tuple[List[Dict], Dict[str, float], Dict[str, Any]]:
    """
    Evaluate full dataset using shared utilities.
    
    Parameters:
        dataset: List of examples
        model: TextGrad model
        system_prompt: Current system prompt (unused but kept for signature consistency)
        max_samples: Maximum samples to evaluate
        desc: Progress bar description
        
    Returns:
        Tuple of (results, metrics, complexity_metrics)
    """
    if max_samples:
        dataset = dataset[:max_samples]
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_single_example, example, model, system_prompt): example 
                   for example in dataset}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            match, result = future.result()
            results.append(result)
    
    # Calculate metrics using shared utility
    metrics = calculate_metrics(results)
    
    # Aggregate complexity metrics from individual results
    complexity_counts = {}
    for result in results:
        complexity = result.get("complexity", "UNKNOWN")
        # Handle list complexity values
        if isinstance(complexity, list):
            complexity = complexity[0] if complexity else "UNKNOWN"
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    complexity_metrics = {
        "by_complexity": complexity_counts,
        "total": len(results)
    }
    
    return results, metrics, complexity_metrics


class TextGradSQL:
    """Simple wrapper for SQL generation with system prompt"""
    
    def __init__(self, system_prompt: tg.Variable, engine):
        """
        Initialize SQL generation wrapper.
        
        Parameters:
            system_prompt: TextGrad Variable containing system prompt
            engine: vLLM engine
        """
        self.system_prompt = system_prompt
        self.engine = engine
    
    def __call__(self, x: tg.Variable) -> tg.Variable:
        """
        Generate SQL from prompt.
        
        Parameters:
            x: Input prompt variable
            
        Returns:
            Generated SQL as Variable
        """
        response = self.engine.generate(x.value, self.system_prompt.value)
        return tg.Variable(
            response,
            requires_grad=True,
            role_description="generated SQL query",
            predecessors=[self.system_prompt, x]
        )


def run_validation_revert(
    system_prompt: tg.Variable,
    results_history: Dict,
    model,
    val_set: List[Dict],
    max_samples: int = 50
):
    """
    Run validation and revert prompt if performance degrades.
    
    Parameters:
        system_prompt: Current system prompt variable
        results_history: Dictionary tracking optimization progress
        model: TextGrad model
        val_set: Validation dataset
        max_samples: Max validation samples
    """
    _, val_metrics, _ = evaluate_dataset(
        val_set,
        model,
        system_prompt.value,
        max_samples=max_samples,
        desc="Validating"
    )
    
    val_performance = val_metrics["valid_sql_pct"]
    previous_performance = results_history["validation_acc"][-1]
    
    logger.info(f"Validation: {val_performance:.1f}% | Previous: {previous_performance:.1f}%")
    
    if val_performance < previous_performance:
        logger.info(f"Performance degraded - reverting prompt")
        previous_prompt = results_history["prompts"][-1]
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance
    
    results_history["validation_acc"].append(val_performance)


def optimize_prompt(
    train_set: List[Dict],
    val_set: List[Dict],
    test_set: List[Dict],
    num_epochs: int = 3,
    steps_per_epoch: int = 4,
    batch_size: int = 3,
    learning_rate: float = 0.1,
    val_samples: int = 50,
    output_dir: str = "results/textgrad",
    eval_model: str = "meta/llama-3.1-70b-instruct"
):
    """
    Optimize system prompt using TextGrad.
    
    Parameters:
        train_set: Training examples
        val_set: Validation examples
        test_set: Test examples
        num_epochs: Number of optimization epochs
        steps_per_epoch: Training steps per epoch
        batch_size: Batch size for training
        learning_rate: Learning rate (unused - TextGrad uses textual gradients)
        val_samples: Validation samples per step
        output_dir: Output directory for results
        eval_model: NVIDIA model for evaluation/gradients
    """
    set_seed(42)
    
    # Initialize engines following TextGrad best practice:
    # Task engine: vLLM CodeLlama (model being optimized for production)
    # Eval engine: NVIDIA API powerful model (generates textual gradients)
    logger.info("Initializing task engine: vLLM CodeLlama")
    task_engine = TextGradVLLMEngine()
    
    logger.info(f"Initializing evaluation engine: NVIDIA {eval_model}")
    eval_engine = NVIDIATextGradEngine(model=eval_model)
    
    # Set backward engine for TextGrad
    tg.set_backward_engine(eval_engine, override=True)
    
    # Initialize TextGrad components
    system_prompt = tg.Variable(
        DEFAULT_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="system prompt for SQL generation"
    )
    
    model = TextGradSQL(system_prompt, task_engine)
    optimizer = tg.TextualGradientDescent(
        engine=eval_engine,
        parameters=[system_prompt]
    )
    
    # Track optimization progress
    results_history = {
        "test_acc": [],
        "validation_acc": [],
        "prompts": [system_prompt.value],
        "test_results": []
    }
    
    # Initial evaluation
    '''
    logger.info("Running initial evaluation...")
    test_results, test_metrics, test_complexity = evaluate_dataset(
        test_set, model, system_prompt.value, desc="Initial Test"
    )
    val_results, val_metrics, _ = evaluate_dataset(
        val_set, model, system_prompt.value, max_samples=val_samples, desc="Initial Val"
    )
    
    results_history["test_acc"].append(test_metrics["valid_sql_pct"])
    results_history["validation_acc"].append(val_metrics["valid_sql_pct"])
    results_history["test_results"].append({
        "results": test_results,
        "metrics": test_metrics,
        "complexity": test_complexity
    })
    
    logger.info(f"Initial Test: {test_metrics['valid_sql_pct']:.1f}%")
    logger.info(f"Initial Val: {val_metrics['valid_sql_pct']:.1f}%")
    '''

    eval_instruction = """You are a senior SQL instructor optimizing a student AI.

1. Analyze the 'Execution Evidence'. If it says 'MISMATCH' or 'Error', the SQL is WRONG.
2. Compare the 'Generated SQL' against the 'Gold SQL' (Ground Truth).
3. Identify exactly what logic is missing (e.g., missing JOIN, wrong GROUP BY, incorrect WHERE condition).
4. Provide feedback that focuses on how the System Prompt should be changed to prevent this specific error.
"""

    # TextLoss wraps the eval engine
    loss_fn = tg.TextLoss(eval_instruction, engine=eval_engine)
        
    # Optimization loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle training data
        random.shuffle(train_set)
        
        for step in range(steps_per_epoch):
            logger.info(f"Step {step + 1}/{steps_per_epoch}")
            
            optimizer.zero_grad()
            
            # Sample batch
            batch_start = step * batch_size
            batch = train_set[batch_start:batch_start + batch_size]
            
            # Accumulate gradients
            losses = []
            for example in batch:
                question = example["question"]
                gold_sql = example["sql"]
                db_id = example["db_id"]
                
                # Get schema using shared utility - matches baseline.py and DSPy exactly
                schema = SCHEMAS.get(db_id, f"Database: {db_id}")
                
                # Build prompt - matches baseline.py format
                prompt = f"""### Database Schema:
{schema}

### Question: {question}

### SQL Query:
"""
                
                x = tg.Variable(prompt, requires_grad=False, role_description="SQL generation prompt")
                prediction = model(x)
                pred_sql = extract_sql_from_text(prediction.value if hasattr(prediction, 'value') else prediction)
                db_path = get_db_path(db_id)
                pred_success, pred_error, pred_results = execute_sql(pred_sql, db_path)
                gold_success, gold_error, gold_results = execute_sql(gold_sql, db_path)
                
                if not pred_success:
                    status = f"Execution Error: {pred_error}"
                    match_str = "N/A"
                else:
                    status = f"Execution Successful"
                    match, _ = compare_results(pred_results, gold_results)
                    match_str = "MATCH" if match else "MISMATCH (Data differs from Gold SQL)"

                # Build the Context (Evidence only)
                # We don't tell it "You are wrong". We show it the Gold SQL and let it realize it's wrong.
                eval_context_str = f"""
### Question
{question}

### Generated SQL
{pred_sql}

### Gold SQL (Ground Truth)
{gold_sql}

### Execution Evidence
Status: {status}
Result Comparison: {match_str}
"""

                #Connect the Graph
                # This links the evidence back to the generated prediction
                eval_input = tg.Variable(
                    eval_context_str,
                    predecessors=[prediction],
                    role_description="evaluation context with ground truth"
                )
                # TextLoss calls the teacher model!
                loss = loss_fn(eval_input)
                losses.append(loss)
            
            # Backward pass
            total_loss = tg.sum(losses)
            total_loss.backward()
            
            # Update prompt
            optimizer.step()
            
            # Validate and revert if needed
            #run_validation_revert(
            #    system_prompt,
            #    results_history,
            #    model,
            #    val_set,
            #    max_samples=val_samples
            #)
            
            # Test evaluation
            test_results, test_metrics, test_complexity = evaluate_dataset(
                test_set, model, system_prompt.value, desc=f"Test (E{epoch+1}S{step+1})"
            )
            
            results_history["test_acc"].append(test_metrics["valid_sql_pct"])
            results_history["prompts"].append(system_prompt.value)
            results_history["test_results"].append({
                "results": test_results,
                "metrics": test_metrics,
                "complexity": test_complexity
            })
            
            logger.info(f"Test: {test_metrics['valid_sql_pct']:.1f}%")
            logger.info(f"Current prompt length: {len(system_prompt.value)} chars")
    
    # Save final results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save optimization history
    with open(output_path / "optimization_history.json", "w") as f:
        json.dump(results_history, f, indent=2, default=str)
    
    # Generate before/after report
    generate_optimization_report(
        results_history["test_results"][0],
        results_history["test_results"][-1],
        results_history["prompts"][0],
        results_history["prompts"][-1],
        output_path
    )
    
    logger.info(f"\nOptimization complete!")
    logger.info(f"Initial: {results_history['test_acc'][0]:.1f}%")
    logger.info(f"Final: {results_history['test_acc'][-1]:.1f}%")
    logger.info(f"Improvement: {results_history['test_acc'][-1] - results_history['test_acc'][0]:.1f}%")
    logger.info(f"Results saved to {output_path}")


def generate_optimization_report(
    before_results: Dict,
    after_results: Dict,
    before_prompt: str,
    after_prompt: str,
    output_dir: Path
):
    """
    Generate before/after optimization comparison report using shared utilities.
    
    Matches DSPy optimizer output format for fair comparison.
    
    Parameters:
        before_results: Initial evaluation results
        after_results: Final evaluation results
        before_prompt: Initial system prompt
        after_prompt: Optimized system prompt
        output_dir: Output directory
    """
    # Save detailed results using shared utility
    save_evaluation_results(
        before_results["results"],
        output_dir / "before_optimization_results.jsonl"
    )
    save_evaluation_results(
        after_results["results"],
        output_dir / "after_optimization_results.jsonl"
    )
    
    # Generate comparison report using shared utility
    comparison_table = "## Performance Comparison\n\n"
    comparison_table += "| Metric | Before | After | Change |\n"
    comparison_table += "|--------|--------|-------|--------|\n"
    comparison_table += f"| Valid SQL % | {before_results['metrics']['valid_sql_pct']:.1f}% | "
    comparison_table += f"{after_results['metrics']['valid_sql_pct']:.1f}% | "
    comparison_table += f"{after_results['metrics']['valid_sql_pct'] - before_results['metrics']['valid_sql_pct']:+.1f}% |\n"
    comparison_table += f"| Results Match % | {before_results['metrics']['result_match_pct']:.1f}% | "
    comparison_table += f"{after_results['metrics']['result_match_pct']:.1f}% | "
    comparison_table += f"{after_results['metrics']['result_match_pct'] - before_results['metrics']['result_match_pct']:+.1f}% |\n"
    comparison_table += f"| Total Examples | {before_results['metrics']['total_examples']} | "
    comparison_table += f"{after_results['metrics']['total_examples']} | - |\n"
    
    # Build complexity breakdown (matches DSPy format)
    complexity_section = "## Performance by SQL Complexity\n\n"
    complexity_section += "### Before Optimization\n\n"
    for complexity, count in sorted(before_results["complexity"]["by_complexity"].items()):
        if count > 0:
            # Calculate stats from results
            cat_results = [r for r in before_results["results"] if r.get("complexity") == complexity]
            valid = sum(1 for r in cat_results if r.get("execution_success"))
            matched = sum(1 for r in cat_results if r.get("results_match"))
            complexity_section += f"**{complexity.upper().replace('_', ' ')}**: "
            complexity_section += f"Valid: {valid}/{count} ({100*valid/count:.1f}%), "
            complexity_section += f"Match: {matched}/{count} ({100*matched/count:.1f}%)\n\n"
    
    complexity_section += "\n### After Optimization\n\n"
    for complexity, count in sorted(after_results["complexity"]["by_complexity"].items()):
        if count > 0:
            # Calculate stats from results
            cat_results = [r for r in after_results["results"] if r.get("complexity") == complexity]
            valid = sum(1 for r in cat_results if r.get("execution_success"))
            matched = sum(1 for r in cat_results if r.get("results_match"))
            complexity_section += f"**{complexity.upper().replace('_', ' ')}**: "
            complexity_section += f"Valid: {valid}/{count} ({100*valid/count:.1f}%), "
            complexity_section += f"Match: {matched}/{count} ({100*matched/count:.1f}%)\n\n"
    
    # Prompt comparison section
    prompt_section = f"""## System Prompts

### Initial System Prompt

```
{before_prompt}
```

### Optimized System Prompt

```
{after_prompt}
```
"""
    
    # Generate report using shared utility
    sections = {
        "Performance Comparison": comparison_table,
        "Performance by SQL Complexity": complexity_section,
        "System Prompts": prompt_section,
    }
    
    report_path = generate_markdown_report(
        metrics={},
        output_dir=output_dir,
        title="TextGrad Optimization Results",
        model_name="CodeLlama-7B-Instruct (via vLLM)",
        dataset_name="Spider (Train subset for optimization, Dev for evaluation)",
        additional_sections=sections
    )
    
    logger.info(f"Comparison report saved to: {report_path}")


def main():
    """Main entry point for TextGrad optimization"""
    parser = argparse.ArgumentParser(
        description="TextGrad Prompt Optimization for NL2SQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--num-train", type=int, default=100, help="Number of training examples")
    parser.add_argument("--num-val", type=int, default=100, help="Number of validation examples")
    parser.add_argument("--num-test", type=int, default=None, help="Number of test examples (default: all)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=4, help="Training steps per epoch")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (not used by TextGrad)")
    parser.add_argument("--val-samples", type=int, default=50, help="Validation samples per step")
    parser.add_argument("--eval-model", type=str, default="nvidia/nemotron-3-nano-30b-a3b", 
                       help="NVIDIA model for evaluation/gradients")
    parser.add_argument("--output-dir", type=str, default="results/textgrad", help="Output directory")
    args = parser.parse_args()
    
    # Load data exactly like DSPy: train from spider_clean, test from spider_dev_clean
    logger.info("Loading datasets...")
    
    # Training data from spider_clean.jsonl (same as DSPy)
    train_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
    train_data = train_data.shuffle(seed=42)
    
    # Evaluation data from spider_dev_clean.jsonl (full dev set, same as DSPy)
    test_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")
    
    # Convert training data to list and split into train/val
    train_examples = []
    for item in train_data:
        train_examples.append({
            "question": item["question"],
            "sql": item["sql"],
            "db_id": item["db_id"],
            "context": item.get("context", "")
        })
    
    train_set = train_examples[:args.num_train]
    val_set = train_examples[args.num_train:args.num_train + args.num_val]
    
    # Convert test data to list (use full dev set for evaluation like DSPy)
    test_set = []
    for item in test_data:
        test_set.append({
            "question": item["question"],
            "sql": item["sql"],
            "db_id": item["db_id"],
            "context": item.get("context", "")
        })
    
    # Optional: limit test set size
    if args.num_test:
        test_set = test_set[:args.num_test]
    
    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # Run optimization
    logger.info("="*60)
    logger.info("TextGrad Configuration:")
    logger.info(f"  Task Engine: vLLM CodeLlama (http://localhost:8000)")
    logger.info(f"  Eval Engine: NVIDIA {args.eval_model}")
    logger.info(f"  Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    logger.info("="*60)
    
    optimize_prompt(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_samples=args.val_samples,
        output_dir=args.output_dir,
        eval_model=args.eval_model
    )


if __name__ == "__main__":
    main()
