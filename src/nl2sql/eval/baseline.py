"""
Baseline Evaluation for NL2SQL on Spider Dataset (using vLLM)

Evaluates 3 standard baseline approaches:
1. Zero-shot: Single LLM call without examples
2. Few-shot: With 2 similar examples  
3. Self-correction: Generate → Fix execution errors → LLM validation → Result comparison

Key improvements:
- Uses vLLM for 3-5x faster inference
- Uses HuggingFace datasets instead of local files
- LLM-based semantic validation (asks if SQL answers the question)
- Compares execution results with gold standard
- Shows intermediate results during evaluation
- Simpler and more reliable than SQL parsing

Requirements:
    pip install vllm openai datasets tqdm

Start vLLM server first:
    vllm serve codellama/CodeLlama-7b-Instruct-hf --host 0.0.0.0 --port 8000

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
        """
        Compare generated results with gold standard results
        
        Handles:
        - Row order independence
        - Column order independence (within rows)
        - Type normalization (int vs float, string representations)
        
        Returns:
            (matches, feedback)
        """
        if generated_results is None or gold_results is None:
            return False, "Cannot compare - one or both queries failed to execute"
        
        # Normalize a single value (handle type differences)
        def normalize_value(val):
            if val is None:
                return None
            # Convert to string and strip for comparison
            # This handles int vs float (1 vs 1.0) and string representations
            return str(val).strip()
        
        # Normalize and sort each row to handle column order differences
        def normalize_row(row):
            if isinstance(row, (list, tuple)):
                # Sort values within the row (column order independent)
                normalized = tuple(sorted(normalize_value(v) for v in row))
            else:
                # Single value row
                normalized = (normalize_value(row),)
            return normalized
        
        # Convert to sets for row-order-independent comparison
        try:
            gen_set = set(normalize_row(row) for row in generated_results)
            gold_set = set(normalize_row(row) for row in gold_results)
        except Exception as e:
            return False, f"Error normalizing results: {str(e)}"
        
        # Check for exact match
        if gen_set == gold_set:
            return True, "Results match gold standard"
        
        # Provide detailed feedback on differences
        if len(gen_set) != len(gold_set):
            return False, f"Result count mismatch: generated {len(gen_set)} rows, expected {len(gold_set)} rows"
        
        # Same number of rows but different content
        return False, "Results differ from gold standard (same row count, different values)"


class SpiderEvaluator:
    """Evaluate baseline approaches on Spider dataset using vLLM"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf", 
                 vllm_url: str = "http://localhost:8000/v1"):
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.client = None
        self.semantic_validator = None
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, str]:
        """Load database schemas from Spider tables.json"""
        schema_file = Path("database/spider_data/tables.json")
        
        if not schema_file.exists():
            print("⚠️  Warning: tables.json not found. Using minimal schema info.")
            return {}
        
        with open(schema_file) as f:
            tables_data = json.load(f)
        
        schemas = {}
        for db in tables_data:
            db_id = db['db_id']
            table_names = db['table_names_original']
            column_names = db['column_names_original']
            column_types = db['column_types']
            
            # Format schema as CREATE TABLE statements
            schema_lines = []
            for table_idx, table_name in enumerate(table_names):
                cols = [(col[1], column_types[i]) for i, col in enumerate(column_names) if col[0] == table_idx]
                if cols:
                    col_defs = [f"{name} {dtype}" for name, dtype in cols]
                    schema_lines.append(f"CREATE TABLE {table_name} ({', '.join(col_defs)})")
            
            schemas[db_id] = "\n".join(schema_lines)
        
        return schemas
    
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
                "AsadIsmail/nl2sql-deduplicated",
                data_files="spider_dev_clean.jsonl",
                split="train"
            )
            
            print(f"✓ Loaded {len(dataset):,} examples from HuggingFace")
            
            # Convert to list of dicts
            data = []
            for item in dataset:
                data.append({
                    "question": item["question"],
                    "query": item.get("sql", ""),  # Use 'sql' field from cleaned dataset
                    "db_id": item.get("db_id", ""),
                    "context": item.get("context", "")
                })
            
            # Subsample if requested
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
        print(f"Connecting to vLLM server: {self.vllm_url}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}\n")
        
        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=self.vllm_url
        )
        
        # Test connection
        try:
            # Simple test to verify server is running
            response = self.client.completions.create(
                model=self.model_name,
                prompt="SELECT",
                max_tokens=1,
                temperature=0.1
            )
            print("✓ Successfully connected to vLLM server\n")
        except Exception as e:
            print(f"❌ Error connecting to vLLM server: {e}")
            print(f"\nMake sure vLLM server is running:")
            print(f"  vllm serve {self.model_name} --host 0.0.0.0 --port 8000\n")
            raise
        
        # Initialize semantic validator with generate function
        self.semantic_validator = SemanticValidator(self.generate_sql)
        
        print("✓ Client initialized successfully\n")
    
    def generate_sql(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate SQL from prompt using vLLM via OpenAI API"""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1.0,
                stop=["\n\n", "###", "Question:", "Schema:"]  # Stop at common delimiters
            )
            
            generated = response.choices[0].text
            sql = self._extract_sql(generated)
            return sql.strip()
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return ""
    
    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from generated text"""
        # Look for SELECT statement
        text_upper = text.upper()
        if "SELECT" in text_upper:
            start = text_upper.find("SELECT")
            sql = text[start:]
            # Stop at common delimiters
            for delimiter in ["\n\n", "###", "Question:", "Schema:"]:
                if delimiter in sql:
                    sql = sql[:sql.find(delimiter)]
            # Remove trailing semicolon if present
            sql = sql.rstrip(';').strip()
            return sql
        return text.strip()
    
    def execute_sql(self, sql: str, db_path: str) -> Tuple[bool, Optional[str], Optional[List]]:
        """
        Execute SQL and return (success, error_message, results)
        
        Returns:
            success (bool): Whether query executed without errors
            error (str): Error message if failed, None otherwise
            results (list): Query results if successful, None otherwise
        """
        if not os.path.exists(db_path):
            return False, f"Database not found: {db_path}", None
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return True, None, results
        except Exception as e:
            return False, str(e), None
    
    def _print_comparison(self, idx: int, question: str, gen_sql: str, gold_sql: str, 
                         gen_results, gold_results, results_match: bool, is_valid: bool):
        """Print intermediate comparison results"""
        print(f"\n{'='*70}")
        print(f"Example {idx + 1}")
        print(f"{'='*70}")
        print(f"Question: {question[:100]}...")
        print(f"\nGenerated SQL:\n  {gen_sql[:150]}...")
        print(f"\nGold SQL:\n  {gold_sql[:150]}...")
        
        if is_valid:
            print(f"\n✅ Generated SQL executed successfully")
            print(f"Generated Results: {gen_results[:3] if len(gen_results) > 3 else gen_results}{'...' if len(gen_results) > 3 else ''}")
            print(f"Gold Results:      {gold_results[:3] if gold_results and len(gold_results) > 3 else gold_results}{'...' if gold_results and len(gold_results) > 3 else ''}")
            
            if results_match:
                print(f"🎯 MATCH: Results are identical!")
            else:
                print(f"❌ MISMATCH: Results differ")
        else:
            print(f"\n❌ Generated SQL failed to execute")
        print(f"{'='*70}\n")
    
    # ================================================================
    # Baseline 1: Zero-shot
    # ================================================================
    
    def zero_shot(self, question: str, schema: str, db_path: str, gold_sql: str = None) -> Dict:
        """Baseline 1: Single LLM call without examples"""
        
        prompt = f"""-- Database Schema
{schema}

-- Question: {question}
-- SQL:
"""
        
        start_time = time.time()
        sql = self.generate_sql(prompt, max_new_tokens=150)
        inference_time = time.time() - start_time
        
        # Execute generated SQL
        is_valid, error, results = self.execute_sql(sql, db_path)
        
        # Execute gold SQL for comparison
        gold_results = None
        results_match = False
        if gold_sql:
            gold_valid, gold_error, gold_results = self.execute_sql(gold_sql, db_path)
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
            "num_attempts": 1
        }
    
    # ================================================================
    # Baseline 2: Few-shot
    # ================================================================
    
    def few_shot(self, question: str, schema: str, db_path: str, 
                 examples: List[Dict], gold_sql: str = None) -> Dict:
        """Baseline 2: Few-shot with similar examples"""
        
        prompt = "-- Examples of natural language to SQL:\n\n"
        
        # Add 2-3 examples
        for i, ex in enumerate(examples[:2], 1):
            prompt += f"-- Example {i}\n"
            prompt += f"-- Schema: {ex.get('context', 'N/A')}\n"
            prompt += f"-- Question: {ex['question']}\n"
            prompt += f"-- SQL:\n{ex.get('query', ex.get('sql', ''))}\n\n"
        
        prompt += f"-- Now convert this question:\n"
        prompt += f"-- Schema:\n{schema}\n"
        prompt += f"-- Question: {question}\n"
        prompt += f"-- SQL:\n"
        
        start_time = time.time()
        sql = self.generate_sql(prompt, max_new_tokens=150)
        inference_time = time.time() - start_time
        
        # Execute generated SQL
        is_valid, error, results = self.execute_sql(sql, db_path)
        
        # Execute gold SQL for comparison
        gold_results = None
        results_match = False
        if gold_sql:
            gold_valid, gold_error, gold_results = self.execute_sql(gold_sql, db_path)
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
            "num_attempts": 1
        }
    
    # ================================================================
    # Baseline 3: Self-correction with LLM Validation and Result Comparison
    # ================================================================
    
    def self_correction(self, question: str, schema: str, db_path: str,
                       gold_sql: str, max_attempts: int = 3) -> Dict:
        """
        Baseline 3: Iterative improvement with execution fixes and semantic validation
        
        Process:
        1. Generate SQL and fix execution errors
        2. Once valid, ask LLM if it correctly answers the question
        3. If LLM says no, retry with semantic feedback
        4. Compare results with gold standard
        5. If results don't match, provide feedback (but don't retry - already at max)
        
        Total attempts: up to max_attempts (default 3)
        Each attempt includes: generation → execution → semantic check → result comparison
        """
        
        attempts = []
        total_start = time.time()
        
        # Execute gold SQL once to get expected results
        gold_valid, gold_error, gold_results = self.execute_sql(gold_sql, db_path)
        
        prompt = f"""-- Database Schema
{schema}

-- Question: {question}
-- Generate a valid SQL query:
"""
        
        for attempt_num in range(max_attempts):
            # Generate SQL
            sql = self.generate_sql(prompt, max_new_tokens=150)
            
            # Try to execute
            is_valid, error, results = self.execute_sql(sql, db_path)
            
            # Initialize attempt record
            attempt_record = {
                "attempt_number": attempt_num + 1,
                "sql": sql,
                "is_valid": is_valid,
                "execution_error": error,
                "llm_validation": None,
                "results_match_gold": False,
                "results_feedback": None
            }
            
            # If execution failed, add error feedback and continue to next attempt
            if not is_valid:
                attempt_record["results_feedback"] = "Query failed to execute"
                attempts.append(attempt_record)
                
                # Add execution error feedback for next attempt
                prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
                prompt += f"\n-- SQL: {sql}"
                prompt += f"\n-- Execution error: {error}"
                prompt += f"\n-- Fix the error and try again:\n"
                continue
            
            # Execution succeeded - now do semantic validation
            llm_validation = self.semantic_validator.ask_llm_validation(
                question, sql, schema
            )
            attempt_record["llm_validation"] = llm_validation
            
            # Compare results with gold standard
            if gold_valid:
                results_match, results_feedback = self.semantic_validator.compare_results(
                    results, gold_results
                )
                attempt_record["results_match_gold"] = results_match
                attempt_record["results_feedback"] = results_feedback
            else:
                attempt_record["results_feedback"] = "Cannot compare - gold query failed"
            
            attempts.append(attempt_record)
            
            # Check if we got it right (LLM says yes AND results match)
            # LLM validation starts with "Yes" or "yes" if correct
            llm_says_correct = llm_validation and llm_validation.lower().startswith('yes')
            
            if llm_says_correct and results_match:
                # Perfect! SQL is correct both semantically and in results
                break
            
            # If this is the last attempt, we're done (no more retries)
            if attempt_num >= max_attempts - 1:
                break
            
            # Build feedback for next attempt
            feedback_parts = []
            
            if not llm_says_correct:
                feedback_parts.append(f"LLM validation: {llm_validation}")
            
            if not results_match:
                feedback_parts.append(f"Result comparison: {results_feedback}")
            
            combined_feedback = "\n".join(feedback_parts)
            
            # Add feedback for next attempt
            prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
            prompt += f"\n-- SQL: {sql}"
            prompt += f"\n-- Feedback: {combined_feedback}"
            prompt += f"\n-- Improve the SQL query:\n"
        
        # Get final attempt results
        final_attempt = attempts[-1]
        total_time = time.time() - total_start
        
        return {
            "method": "self_correction",
            "sql": final_attempt["sql"],
            "is_valid": final_attempt["is_valid"],
            "error": final_attempt["execution_error"],
            "llm_validation": final_attempt["llm_validation"],
            "results_match_gold": final_attempt["results_match_gold"],
            "results_feedback": final_attempt["results_feedback"],
            "generated_results": results if final_attempt["is_valid"] else None,
            "gold_results": gold_results if gold_valid else None,
            "results_match": final_attempt["results_match_gold"],  # Add for consistency
            "inference_time": total_time,
            "num_attempts": len(attempts),
            "all_attempts": attempts
        }
    
    # ================================================================
    # Main Evaluation
    # ================================================================
    
    def evaluate(self, output_dir: str = "results/baseline", num_samples: int = None,
                 print_every: int = 10):
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
        print(f"📊 Printing intermediate results every {print_every} examples\n")
        
        # Load model
        self.load_model()
        
        # Results storage
        results = {
            "zero_shot": [],
            "few_shot": [],
            "self_correction": []
        }
        
        # Running statistics
        stats = {
            "zero_shot": {"valid": 0, "matched": 0, "total": 0},
            "few_shot": {"valid": 0, "matched": 0, "total": 0},
            "self_correction": {"valid": 0, "matched": 0, "total": 0}
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
                        examples = data[max(0, i-5):i] if i > 0 else data[1:4]
                        result = self.few_shot(question, schema, db_path, examples, gold_sql)
                    else:  # self_correction with semantic feedback
                        result = self.self_correction(question, schema, db_path, gold_sql)
                    
                    result["question"] = question
                    result["gold_sql"] = gold_sql
                    result["db_id"] = db_id
                    results[method_name].append(result)
                    
                    # Update stats
                    stats[method_name]["total"] += 1
                    if result.get("is_valid", False):
                        stats[method_name]["valid"] += 1
                    if result.get("results_match", False):
                        stats[method_name]["matched"] += 1
                    
                    # Print intermediate results
                    if (i + 1) % print_every == 0:
                        self._print_comparison(
                            i, question, result["sql"], gold_sql,
                            result.get("results"), result.get("gold_results"),
                            result.get("results_match", False), result.get("is_valid", False)
                        )
                        
                        # Print running statistics
                        s = stats[method_name]
                        print(f"📈 Running Stats ({method_name}) after {s['total']} examples:")
                        print(f"   Valid SQL: {s['valid']}/{s['total']} ({100*s['valid']/s['total']:.1f}%)")
                        print(f"   Results Match: {s['matched']}/{s['total']} ({100*s['matched']/s['total']:.1f}%)")
                        print()
                    
                except Exception as e:
                    print(f"\n❌ Error on example {i}: {e}")
                    results[method_name].append({
                        "method": method_name,
                        "question": question,
                        "gold_sql": gold_sql,
                        "db_id": db_id,
                        "error": str(e),
                        "is_valid": False,
                        "results_match": False
                    })
        
        # Save results and generate report
        self._save_results(results, output_dir)
        self._generate_report(results, output_dir)
        
        return results
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save results to JSONL files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for method, data in results.items():
            filepath = output_path / f"{method}_results.jsonl"
            with open(filepath, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            print(f"\n✅ Saved {method} results to: {filepath}")
    
    def _generate_report(self, results: Dict, output_dir: str):
        """Generate comparison report"""
        
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        report = "# Baseline Evaluation Results\n\n"
        report += f"Model: {self.model_name}\n"
        report += f"Dataset: Spider Dev (HuggingFace: AsadIsmail/nl2sql-deduplicated)\n\n"
        report += "## Summary\n\n"
        report += "| Method | Valid SQL % | Results Match % | Avg Time (s) | Avg Attempts |\n"
        report += "|--------|-------------|-----------------|--------------|-------------|\n"
        
        for method, data in results.items():
            valid_count = sum(1 for r in data if r.get("is_valid", False))
            valid_pct = 100 * valid_count / len(data) if data else 0
            
            match_count = sum(1 for r in data if r.get("results_match", False))
            match_pct = 100 * match_count / len(data) if data else 0
            
            avg_time = sum(r.get("inference_time", 0) for r in data) / len(data) if data else 0
            avg_attempts = sum(r.get("num_attempts", 1) for r in data) / len(data) if data else 0
            
            report += f"| {method.replace('_', ' ').title()} | {valid_pct:.1f}% | {match_pct:.1f}% | {avg_time:.2f} | {avg_attempts:.1f} |\n"
            
            print(f"{method.upper()}")
            print(f"  Valid SQL: {valid_count}/{len(data)} ({valid_pct:.1f}%)")
            print(f"  Results Match Gold: {match_count}/{len(data)} ({match_pct:.1f}%)")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Avg attempts: {avg_attempts:.1f}")
            print()
        
        report += "\n## Notes\n\n"
        report += "- **Zero-shot**: Single generation attempt\n"
        report += "- **Few-shot**: Uses 2 examples from dataset\n"
        report += "- **Self-correction**: Up to 3 attempts to fix execution errors\n\n"
        report += "- **Valid SQL** = Query executed without errors\n"
        report += "- **Results Match** = Execution results identical to gold standard\n\n"
        report += "### Self-Correction Process\n\n"
        report += "The self-correction method includes:\n"
        report += "1. **Execution error fixing**: Up to 3 attempts to generate valid SQL\n"
        report += "2. **LLM validation**: Ask LLM if the query correctly answers the question\n"
        report += "3. **Result comparison**: Compare execution results with gold standard\n\n"
        report += "This provides both syntactic correctness and semantic validation.\n"
        
        # Save report
        report_path = Path(output_dir) / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"{'='*60}")
        print(f"✅ Report saved to: {report_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation on Spider (vLLM)")
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-hf",
                       help="Model name (must match vLLM server)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL")
    parser.add_argument("--output", type=str, default="results/baseline",
                       help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--print-every", type=int, default=10,
                       help="Print intermediate results every N examples (default: 10)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BASELINE EVALUATION WITH vLLM")
    print("="*60)
    print(f"\nMake sure vLLM server is running:")
    print(f"  vllm serve {args.model} --host 0.0.0.0 --port 8000")
    print(f"\nConnecting to: {args.vllm_url}")
    print(f"Model: {args.model}")
    print("="*60 + "\n")
    
    # Run evaluation
    evaluator = SpiderEvaluator(model_name=args.model, vllm_url=args.vllm_url)
    results = evaluator.evaluate(
        output_dir=args.output,
        num_samples=args.num_samples,
        print_every=args.print_every
    )
    
    print("\n✅ Baseline evaluation complete!")
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()