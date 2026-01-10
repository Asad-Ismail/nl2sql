"""
Baseline Evaluation for NL2SQL on Spider Dataset (using vLLM)

Evaluates 3 standard baseline approaches:
1. Zero-shot: Single LLM call without examples
2. Few-shot: With 2 similar examples
3. Self-correction: Generate → Fix execution errors → LLM validation → Result comparison
Usage:
    python baseline_improved.py --model codellama/CodeLlama-7b-hf
    python baseline_improved.py --model meta-llama/Llama-2-7b-hf --num-samples 50
    python baseline_improved.py --vllm-url http://localhost:8000/v1
"""

import os
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import time
from datasets import load_dataset
from ratelimit import limits, sleep_and_retry
from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    print_comparison,
    compare_results,
    calculate_metrics,
    print_metrics,
    extract_sql_from_text,
    categorize_sql_complexity,
    get_db_path,
    save_evaluation_results,
    generate_markdown_report,
)


class SemanticValidator:
    """Validate SQL queries using LLM and result comparison"""

    def __init__(self, generate_func):
        """
        Args:
            generate_func: Function to generate text from LLM
        """
        self.generate_func = generate_func

    def ask_llm_validation(self, question: str, sql: str, schema: str) -> str:
        """
        Ask LLM if the SQL correctly answers the question

        Returns:
            LLM's assessment (yes/no with brief explanation)
        """
        prompt = f"""-- Database Schema
{schema}

-- Question: {question}
-- SQL Query: {sql}

Does this SQL query correctly answer the question? Answer with 'Yes' or 'No' followed by a brief explanation.
Answer:"""

        response = self.generate_func(prompt, max_new_tokens=100)
        return response.strip()

    @staticmethod
    def compare_results(generated_results: List, gold_results: List) -> Tuple[bool, str]:
        """Use shared compare_results function from utils"""
        return compare_results(generated_results, gold_results)


class SpiderEvaluator:
    """Evaluate baseline approaches on Spider dataset using vLLM"""

    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        vllm_url: str = "http://localhost:8000/v1",
    ):
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.client = None
        self.semantic_validator = None
        self.schemas = load_schemas()

    def load_dataset_from_hf(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load Spider evaluation dataset from HuggingFace

        Args:
            num_samples: Number of samples to load (None for all)

        Returns:
            List of evaluation examples
        """
        print(f"\n{'='*60}")
        print("Loading Spider Dev Dataset from HuggingFace")
        print(f"{'='*60}\n")

        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train"
            )

            print(f"✓ Loaded {len(dataset):,} examples from HuggingFace")

            # Convert to list of dicts
            data = []
            for item in dataset:
                data.append(
                    {
                        "question": item["question"],
                        "query": item.get("sql", ""),  # Use 'sql' field from cleaned dataset
                        "db_id": item.get("db_id", ""),
                        "context": item.get("context", ""),
                    }
                )
            if num_samples and num_samples < len(data):
                print(f"  Subsampling to {num_samples} examples")
                data = data[:num_samples]

            print(f"  Using {len(data):,} examples for evaluation\n")

            return data

        except Exception as e:
            print(f"❌ Error loading from HuggingFace: {e}")
            print("\nFalling back to local file if available...")

            # Fallback to local file
            local_file = "nl2sql_data/eval/spider_dev.jsonl"
            if os.path.exists(local_file):
                print(f"✓ Loading from local file: {local_file}")
                with open(local_file) as f:
                    data = [json.loads(line) for line in f]

                if num_samples:
                    data = data[:num_samples]

                return data
            else:
                raise FileNotFoundError(
                    f"Could not load from HuggingFace or local file. "
                    f"Please run: python src/nl2sql/data/download_all_datasets.py"
                )

    def load_model(self):
        """Connect to vLLM server via OpenAI API"""
        print(f"\n{'='*60}")
        print(f"Connecting to server: {self.vllm_url}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}\n")

        # Initialize OpenAI client pointing to vLLM server
        key = "sk-local"

        print(f"{self.vllm_url}")
        self.client = OpenAI(api_key=key, base_url=self.vllm_url)  # vLLM doesn't require API key
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "SELECT"}],
                max_tokens=20,
                temperature=0.1,
            )
            print("✓ Successfully connected to server\n")
        except Exception as e:
            print(f"{key}")
            print(f"❌ Error connecting to vLLM server: {e}")
            print(f"\nMake sure vLLM server is running:")
            print(f"  vllm serve {self.model_name} --host 0.0.0.0 --port 8000\n")
            raise

        # Initialize semantic validator with generate function
        self.semantic_validator = SemanticValidator(self.generate_sql)

        print("Client initialized successfully\n")

    @sleep_and_retry
    @limits(calls=35, period=60)
    def generate_sql(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate SQL from prompt using vLLM via OpenAI API"""
        try:
            # Wrap the raw prompt in the user role
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                stop=["\n\n", "###"],  # Uncomment if needed
            )

            # Access content via message.content, not text
            generated = response.choices[0].message.content

            sql = self._extract_sql(generated)
            return sql.strip()

        except Exception as e:
            print(f"Error generating SQL: {e}")
            return ""

    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from generated text using shared utility"""
        return extract_sql_from_text(text)

    # ================================================================
    # Baseline 1: Zero-shot
    # ================================================================

    def zero_shot(self, question: str, schema: str, db_path: str, gold_sql: str = None) -> Dict:
        """Baseline 1: Single LLM call without examples"""

        prompt = f"""### Task: Convert the following natural language question to a SQL query.  Give only SQL Query as Output

### Database Schema:
{schema}

### Question: {question}

### SQL Query:
"""

        start_time = time.time()
        sql = self.generate_sql(prompt, max_new_tokens=250)
        inference_time = time.time() - start_time

        # Execute generated SQL
        is_valid, error, results = execute_sql(sql, db_path)

        # Execute gold SQL for comparison
        gold_results = None
        results_match = False
        if gold_sql:
            gold_valid, gold_error, gold_results = execute_sql(gold_sql, db_path)
            if is_valid and gold_valid:
                results_match, _ = self.semantic_validator.compare_results(results, gold_results)

        return {
            "method": "zero_shot",
            "sql": sql,
            "is_valid": is_valid,
            "error": error,
            "results": results if is_valid else None,
            "gold_results": gold_results,
            "results_match": results_match,
            "inference_time": inference_time,
            "num_attempts": 1,
        }

    # ================================================================
    # Baseline 2: Few-shot
    # ================================================================

    def few_shot(
        self, question: str, schema: str, db_path: str, examples: List[Dict], gold_sql: str = None
    ) -> Dict:
        """Baseline 2: Few-shot with similar examples"""

        prompt = "### Task: Convert natural language questions to SQL queries.  Give only SQL Query as Output \n\n"
        prompt += "### Examples:\n\n"

        # Add 2-3 examples
        for i, ex in enumerate(examples[:2], 1):
            prompt += f"**Example {i}:**\n"
            prompt += f"Schema: {ex.get('context', 'N/A')}\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"SQL: {ex.get('query', ex.get('sql', ''))}\n\n"

        prompt += f"### Now convert this question:\n"
        prompt += f"Schema:\n{schema}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += f"SQL Query:\n"

        start_time = time.time()
        sql = self.generate_sql(prompt, max_new_tokens=250)
        inference_time = time.time() - start_time

        # Execute generated SQL
        is_valid, error, results = execute_sql(sql, db_path)

        # Execute gold SQL for comparison
        gold_results = None
        results_match = False
        if gold_sql:
            gold_valid, gold_error, gold_results = execute_sql(gold_sql, db_path)
            if is_valid and gold_valid:
                results_match, _ = self.semantic_validator.compare_results(results, gold_results)

        return {
            "method": "few_shot",
            "sql": sql,
            "is_valid": is_valid,
            "error": error,
            "results": results if is_valid else None,
            "gold_results": gold_results,
            "results_match": results_match,
            "inference_time": inference_time,
            "num_attempts": 1,
        }

    # ================================================================
    # Baseline 3: Self-correction with LLM Validation and Result Comparison
    # ================================================================

    def self_correction(
        self, question: str, schema: str, db_path: str, gold_sql: str, max_attempts: int = 3
    ) -> Dict:
        """
        Baseline 3: Iterative improvement with LLM semantic validation

        Process:
        1. Generate SQL and fix execution errors
        2. Once valid, ask LLM if it correctly answers the question
        3. If LLM says no, retry with LLM feedback
        4. At the end, compare with gold standard for evaluation only

        Total attempts: up to max_attempts (default 3)
        Each attempt includes: generation → execution → LLM validation → retry if needed
        """

        attempts = []
        total_start = time.time()

        # Execute gold SQL once for final evaluation (not used in feedback loop)
        gold_valid, gold_error, gold_results = execute_sql(gold_sql, db_path)

        # Track the best attempt based on LLM validation
        best_attempt = None
        best_results = None

        prompt = f"""### Task: Convert the following natural language question to a SQL query. Give only SQL Query as Output

### Database Schema:
{schema}

### Question: {question}

### SQL Query:
"""
        for attempt_num in range(max_attempts):
            # Generate SQL
            sql = self.generate_sql(prompt, max_new_tokens=1024)
            ## handle empty sql
            if not sql or not sql.strip():
                attempt_record = {
                    "attempt_number": attempt_num + 1,
                    "sql": sql,
                    "is_valid": False,
                    "execution_error": "Generated SQL is empty",
                    "llm_validation": None,
                }
                attempts.append(attempt_record)

                # Add feedback for next attempt
                prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
                prompt += f"\n-- Error: Generated empty SQL"
                prompt += f"\n-- Please generate a valid SQL query:\n-- SQL:\n"
                continue

            # Try to execute
            is_valid, error, results = execute_sql(sql, db_path)

            # Initialize attempt record
            attempt_record = {
                "attempt_number": attempt_num + 1,
                "sql": sql,
                "is_valid": is_valid,
                "execution_error": error,
                "llm_validation": None,
            }

            # If execution failed, add error feedback and continue to next attempt
            if not is_valid:
                attempts.append(attempt_record)

                # Add execution error feedback for next attempt
                prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
                prompt += f"\n-- SQL: {sql}"
                prompt += f"\n-- Execution error: {error}"
                prompt += f"\n-- Fix the error and try again:\n"
                continue

            # Execution succeeded - now do LLM semantic validation
            llm_validation = self.semantic_validator.ask_llm_validation(question, sql, schema)
            attempt_record["llm_validation"] = llm_validation

            attempts.append(attempt_record)

            # Update best attempt if this is the first valid one or LLM says it's correct
            llm_says_correct = llm_validation and llm_validation.lower().startswith("yes")

            if best_attempt is None or llm_says_correct:
                best_attempt = attempt_record
                best_results = results

            # If LLM says it's correct, we're done
            if llm_says_correct:
                break

            # If this is the last attempt, we're done (no more retries)
            if attempt_num >= max_attempts - 1:
                break

            # LLM says it's not correct - add feedback for next attempt
            prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
            prompt += f"\n-- SQL: {sql}"
            prompt += f"\n-- LLM Feedback: {llm_validation[:200]}..."
            prompt += f"\n\n-- Generate improved SQL:"
            prompt += f"\n-- SQL:"  # Clear signal to generate here

        # Use best attempt for final results (fallback to last if no valid attempts)
        final_attempt = best_attempt if best_attempt else attempts[-1]
        final_results = best_results
        total_time = time.time() - total_start

        print(attempts)
        # NOW compare with gold results for evaluation purposes only
        results_match_gold = False
        results_feedback = None
        if final_attempt["is_valid"] and gold_valid and final_results is not None:
            results_match_gold, results_feedback = self.semantic_validator.compare_results(
                final_results, gold_results
            )
        elif not final_attempt["is_valid"]:
            results_feedback = "Query failed to execute"
        elif not gold_valid:
            results_feedback = "Cannot compare - gold query failed"

        return {
            "method": "self_correction",
            "sql": final_attempt["sql"],
            "is_valid": final_attempt["is_valid"],
            "error": final_attempt.get("execution_error"),
            "llm_validation": final_attempt.get("llm_validation"),
            "results_match_gold": results_match_gold,  # Evaluation metric only
            "results_feedback": results_feedback,  # Evaluation metric only
            "results": final_results,
            "gold_results": gold_results if gold_valid else None,
            "results_match": results_match_gold,  # For consistency with other methods
            "inference_time": total_time,
            "num_attempts": len(attempts),
            "all_attempts": attempts,
        }

    # ================================================================
    # Main Evaluation
    # ================================================================

    def evaluate(
        self, output_dir: str = "results/baseline", num_samples: int = None, print_every: int = 10
    ):
        """
        Evaluate all baseline methods on Spider dataset from HuggingFace

        Args:
            output_dir: Where to save results
            num_samples: Number of samples to evaluate (None = all)
            print_every: Print intermediate results every N examples
        """

        print(f"\n{'='*60}")
        print("Baseline Evaluation on Spider Dataset (vLLM)")
        print(f"{'='*60}\n")

        # Load data from HuggingFace
        data = self.load_dataset_from_hf(num_samples=num_samples)

        print(f"Evaluating on {len(data)} examples\n")
        print(f"Printing intermediate results every {print_every} examples\n")

        # Load model
        self.load_model()

        # Results storage
        results = {"zero_shot": [], "few_shot": [], "self_correction": []}

        # Running statistics
        stats = {
            "zero_shot": {"valid": 0, "matched": 0, "total": 0},
            "few_shot": {"valid": 0, "matched": 0, "total": 0},
            "self_correction": {"valid": 0, "matched": 0, "total": 0},
        }

        # Complexity metrics tracking
        from collections import defaultdict

        complexity_metrics = {
            "zero_shot": defaultdict(lambda: {"total": 0, "valid": 0, "matched": 0}),
            "few_shot": defaultdict(lambda: {"total": 0, "valid": 0, "matched": 0}),
            "self_correction": defaultdict(lambda: {"total": 0, "valid": 0, "matched": 0}),
        }

        # Evaluate each method
        for method_name in ["zero_shot", "few_shot", "self_correction"]:
            print(f"\n{'='*60}")
            print(f"Method: {method_name.upper().replace('_', ' ')}")
            print(f"{'='*60}\n")

            for i, item in enumerate(tqdm(data, desc=method_name)):
                question = item["question"]
                db_id = item.get("db_id", "")
                gold_sql = item.get("query", item.get("sql", ""))

                # Get actual schema from tables.json or use provided context
                schema = self.schemas.get(db_id, item.get("context", f"Database: {db_id}"))

                # Database path
                db_path = f"database/spider_data/database/{db_id}/{db_id}.sqlite"

                if not os.path.exists(db_path):
                    print(f"\n⚠️  Warning: Database does not exist at {db_path}")
                    continue

                try:
                    if method_name == "zero_shot":
                        result = self.zero_shot(question, schema, db_path, gold_sql)
                    elif method_name == "few_shot":
                        # Use previous examples as few-shot examples
                        examples = data[max(0, i - 5) : i] if i > 0 else data[1:4]
                        result = self.few_shot(question, schema, db_path, examples, gold_sql)
                    else:  # self_correction with semantic feedback
                        result = self.self_correction(question, schema, db_path, gold_sql)

                    # Categorize complexity
                    complexity_categories = categorize_sql_complexity(gold_sql)

                    result["question"] = question
                    result["gold_sql"] = gold_sql
                    result["db_id"] = db_id
                    result["complexity"] = complexity_categories
                    results[method_name].append(result)

                    # Update stats
                    stats[method_name]["total"] += 1
                    if result.get("is_valid", False):
                        stats[method_name]["valid"] += 1
                    if result.get("results_match", False):
                        stats[method_name]["matched"] += 1

                    # Update complexity metrics
                    for category in complexity_categories:
                        complexity_metrics[method_name][category]["total"] += 1
                        if result.get("is_valid", False):
                            complexity_metrics[method_name][category]["valid"] += 1
                        if result.get("results_match", False):
                            complexity_metrics[method_name][category]["matched"] += 1

                    # Print intermediate results
                    if (i + 1) % print_every == 0:
                        print_comparison(
                            i,
                            question,
                            result["sql"],
                            gold_sql,
                            result.get("results"),
                            result.get("gold_results"),
                            result.get("results_match", False),
                            result.get("is_valid", False),
                        )

                        # Print running statistics
                        s = stats[method_name]
                        print(f" Running Stats ({method_name}) after {s['total']} examples:")
                        print(
                            f"   Valid SQL: {s['valid']}/{s['total']} ({100*s['valid']/s['total']:.1f}%)"
                        )
                        print(
                            f"   Results Match: {s['matched']}/{s['total']} ({100*s['matched']/s['total']:.1f}%)"
                        )
                        print()

                except Exception as e:
                    print(f"\n❌ Error on example {i}: {e}")
                    results[method_name].append(
                        {
                            "method": method_name,
                            "question": question,
                            "gold_sql": gold_sql,
                            "db_id": db_id,
                            "error": str(e),
                            "is_valid": False,
                            "results_match": False,
                        }
                    )

        # Save results and generate report
        self._save_results(results, output_dir)
        self._generate_report(results, output_dir, complexity_metrics)

        return results

    def _save_results(self, results: Dict, output_dir: str):
        """Save results to JSONL files using shared utility"""
        save_evaluation_results(results, output_dir)
        print(f"\n Saved results to: {output_dir}/")

    def _generate_report(self, results: Dict, output_dir: str, complexity_metrics: Dict):
        """Generate comparison report with complexity breakdown"""

        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}\n")

        # Calculate metrics for each method
        all_metrics = {}
        for method, data in results.items():
            metrics = calculate_metrics(data)
            metrics["avg_attempts"] = (
                sum(r.get("num_attempts", 1) for r in data) / len(data) if data else 0
            )
            all_metrics[method] = metrics

            # Print to console
            print(f"{method.upper()}")
            print(
                f"  Valid SQL: {metrics['valid_sql_count']}/{metrics['total_examples']} ({metrics['valid_sql_pct']:.1f}%)"
            )
            print(
                f"  Results Match Gold: {metrics['result_match_count']}/{metrics['total_examples']} ({metrics['result_match_pct']:.1f}%)"
            )
            print(f"  Avg time: {metrics['avg_inference_time']:.2f}s")
            print(f"  Avg attempts: {metrics['avg_attempts']:.1f}")
            print()

        # Build summary table
        summary_table = "| Method | Valid SQL % | Results Match % | Avg Time (s) | Avg Attempts |\n"
        summary_table += "|--------|-------------|-----------------|--------------|-------------|\n"

        for method, metrics in all_metrics.items():
            summary_table += f"| {method.replace('_', ' ').title()} | "
            summary_table += f"{metrics['valid_sql_pct']:.1f}% | "
            summary_table += f"{metrics['result_match_pct']:.1f}% | "
            summary_table += f"{metrics['avg_inference_time']:.2f} | "
            summary_table += f"{metrics['avg_attempts']:.1f} |\n"

        # Build complexity breakdown for each method
        complexity_section = ""
        for method in ["zero_shot", "few_shot", "self_correction"]:
            if method in complexity_metrics:
                complexity_section += f"\n### {method.replace('_', ' ').title()}\n\n"
                for category, stats in sorted(complexity_metrics[method].items()):
                    if stats["total"] > 0:
                        complexity_section += f"**{category.upper().replace('_', ' ')}**: "
                        complexity_section += f"Valid: {stats['valid']}/{stats['total']} ({100*stats['valid']/stats['total']:.1f}%), "
                        complexity_section += f"Match: {stats['matched']}/{stats['total']} ({100*stats['matched']/stats['total']:.1f}%)\n\n"

        # Generate report using shared utility
        sections = {
            "Summary": summary_table,
            "Performance by SQL Complexity": complexity_section,
        }

        report_path = generate_markdown_report(
            metrics={},
            output_dir=output_dir,
            title="Baseline Evaluation Results",
            model_name=self.model_name,
            dataset_name="Spider Dev (HuggingFace: AsadIsmail/nl2sql-deduplicated)",
            additional_sections=sections,
        )

        print(f"{'='*60}")
        print(f" Report saved to: {report_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation on Spider (vLLM)")
    parser.add_argument(
        "--model",
        type=str,
        default="TheBloke/CodeLlama-7B-Instruct-AWQ",
        help="Model name (must match vLLM server)",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1", help="vLLM server URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_nematron3",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print intermediate results every N examples (default: 10)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION WITH vLLM")
    print("=" * 60)
    print(f"\nMake sure vLLM server is running:")
    print(f"  vllm serve {args.model} --host 0.0.0.0 --port 8000")
    print(f"\nConnecting to: {args.vllm_url}")
    print(f"Model: {args.model}")
    print("=" * 60 + "\n")

    # Run evaluation
    evaluator = SpiderEvaluator(model_name=args.model, vllm_url=args.vllm_url)
    results = evaluator.evaluate(
        output_dir=args.output, num_samples=args.num_samples, print_every=args.print_every
    )

    print("\n Baseline evaluation complete!")
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
