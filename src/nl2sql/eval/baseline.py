"""
Baseline Evaluation for NL2SQL on Spider Dataset

Evaluates 3 standard baseline approaches:
1. Zero-shot: Single LLM call without examples
2. Few-shot: With 3 similar examples  
3. Self-correction: Generate → Execute → Fix errors (up to 3 attempts)

Usage:
    python src/nl2sql/eval/baseline.py --model codellama/CodeLlama-7b-hf
    python src/nl2sql/eval/baseline.py --model meta-llama/Llama-2-7b-hf --num-samples 50
"""

import os
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time


class SpiderEvaluator:
    """Evaluate baseline approaches on Spider dataset"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model for inference"""
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_name}")
        print(f"{'='*60}\n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✓ Model loaded successfully\n")
    
    def generate_sql(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate SQL from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL from response (remove the prompt)
        sql = generated[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        sql = self._extract_sql(sql)
        return sql.strip()
    
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
    
    # ================================================================
    # Baseline 1: Zero-shot
    # ================================================================
    
    def zero_shot(self, question: str, schema: str, db_path: str) -> Dict:
        """Baseline 1: Single LLM call without examples"""
        
        prompt = f"""-- Database Schema
{schema}

-- Question: {question}
-- SQL:
SELECT"""
        
        start_time = time.time()
        sql = "SELECT" + self.generate_sql(prompt, max_new_tokens=150)
        inference_time = time.time() - start_time
        
        is_valid, error, results = self.execute_sql(sql, db_path)
        
        return {
            "method": "zero_shot",
            "sql": sql,
            "is_valid": is_valid,
            "error": error,
            "results": results if is_valid else None,
            "inference_time": inference_time,
            "num_attempts": 1
        }
    
    # ================================================================
    # Baseline 2: Few-shot
    # ================================================================
    
    def few_shot(self, question: str, schema: str, db_path: str, 
                 examples: List[Dict]) -> Dict:
        """Baseline 2: Few-shot with similar examples"""
        
        prompt = "-- Examples of natural language to SQL:\n\n"
        
        # Add 2-3 examples
        for i, ex in enumerate(examples[:2], 1):
            prompt += f"-- Example {i}\n"
            prompt += f"-- Schema: {ex.get('schema', 'N/A')}\n"
            prompt += f"-- Question: {ex['question']}\n"
            prompt += f"-- SQL:\n{ex['sql']}\n\n"
        
        prompt += f"-- Now convert this question:\n"
        prompt += f"-- Schema:\n{schema}\n"
        prompt += f"-- Question: {question}\n"
        prompt += f"-- SQL:\nSELECT"
        
        start_time = time.time()
        sql = "SELECT" + self.generate_sql(prompt, max_new_tokens=150)
        inference_time = time.time() - start_time
        
        is_valid, error, results = self.execute_sql(sql, db_path)
        
        return {
            "method": "few_shot",
            "sql": sql,
            "is_valid": is_valid,
            "error": error,
            "results": results if is_valid else None,
            "inference_time": inference_time,
            "num_attempts": 1
        }
    
    # ================================================================
    # Baseline 3: Self-correction
    # ================================================================
    
    def self_correction(self, question: str, schema: str, db_path: str,
                       max_attempts: int = 3) -> Dict:
        """Baseline 3: Generate, execute, fix errors iteratively"""
        
        attempts = []
        total_start = time.time()
        
        prompt = f"""-- Database Schema
                {schema}
                -- Question: {question}
                -- Generate a valid SQL query:
                SELECT"""
        
        for attempt in range(max_attempts):
            sql = "SELECT" + self.generate_sql(prompt, max_new_tokens=150)
            is_valid, error, results = self.execute_sql(sql, db_path)
            
            attempts.append({
                "sql": sql,
                "is_valid": is_valid,
                "error": error
            })
            
            if is_valid:
                break
            
            # Add error feedback for next attempt
            prompt += f"\n\n-- Previous SQL had error: {error}\n"
            prompt += f"-- Previous SQL: {sql}\n"
            prompt += f"-- Fix the SQL query:\nSELECT"
        
        total_time = time.time() - total_start
        
        return {
            "method": "self_correction",
            "sql": attempts[-1]["sql"],
            "is_valid": attempts[-1]["is_valid"],
            "error": attempts[-1]["error"],
            "results": results if attempts[-1]["is_valid"] else None,
            "inference_time": total_time,
            "num_attempts": len(attempts),
            "all_attempts": attempts
        }
    
    # ================================================================
    # Main Evaluation
    # ================================================================
    
    def evaluate(self, data_path: str, output_dir: str = "results/baseline",
                num_samples: int = None):
        """
        Evaluate all baseline methods on Spider dataset
        
        Args:
            data_path: Path to Spider JSONL file
            output_dir: Where to save results
            num_samples: Number of samples to evaluate (None = all)
        """
        
        print(f"\n{'='*60}")
        print("Baseline Evaluation on Spider Dataset")
        print(f"{'='*60}\n")
        
        # Load data
        print(f"Loading data from: {data_path}")
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
        
        if num_samples:
            data = data[:num_samples]
        
        print(f"Evaluating on {len(data)} examples\n")
        
        # Load model
        self.load_model()
        
        # Results storage
        results = {
            "zero_shot": [],
            "few_shot": [],
            "self_correction": []
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
                
                # Get schema (simplified - you may need to load actual schema)
                schema = item.get("context", f"Database: {db_id}")
                
                # Database path
                db_path = f"nl2sql_data/database/spider_data/database/{db_id}/{db_id}.sqlite"
                
                try:
                    if method_name == "zero_shot":
                        result = self.zero_shot(question, schema, db_path)
                    elif method_name == "few_shot":
                        # Use previous examples as few-shot examples
                        examples = data[max(0, i-5):i] if i > 0 else data[1:4]
                        result = self.few_shot(question, schema, db_path, examples)
                    else:  # self_correction
                        result = self.self_correction(question, schema, db_path)
                    
                    result["question"] = question
                    result["gold_sql"] = gold_sql
                    result["db_id"] = db_id
                    results[method_name].append(result)
                    
                except Exception as e:
                    print(f"\n⚠️ Error on example {i}: {e}")
                    results[method_name].append({
                        "method": method_name,
                        "question": question,
                        "gold_sql": gold_sql,
                        "db_id": db_id,
                        "error": str(e),
                        "is_valid": False
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
            print(f"\n✓ Saved {method} results to: {filepath}")
    
    def _generate_report(self, results: Dict, output_dir: str):
        """Generate comparison report"""
        
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        report = "# Baseline Evaluation Results\n\n"
        report += f"Model: {self.model_name}\n\n"
        report += "## Summary\n\n"
        report += "| Method | Valid SQL % | Avg Time (s) | Avg Attempts |\n"
        report += "|--------|-------------|--------------|-------------|\n"
        
        for method, data in results.items():
            valid_count = sum(1 for r in data if r.get("is_valid", False))
            valid_pct = 100 * valid_count / len(data) if data else 0
            avg_time = sum(r.get("inference_time", 0) for r in data) / len(data) if data else 0
            avg_attempts = sum(r.get("num_attempts", 1) for r in data) / len(data) if data else 0
            
            report += f"| {method.replace('_', ' ').title()} | {valid_pct:.1f}% | {avg_time:.2f} | {avg_attempts:.1f} |\n"
            
            print(f"{method.upper()}")
            print(f"  Valid SQL: {valid_count}/{len(data)} ({valid_pct:.1f}%)")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Avg attempts: {avg_attempts:.1f}")
            print()
        
        report += "\n## Notes\n\n"
        report += "- **Zero-shot**: Single generation attempt\n"
        report += "- **Few-shot**: Uses 2 examples from training set\n"
        report += "- **Self-correction**: Up to 3 attempts with error feedback\n\n"
        report += "Valid SQL = Query executed without errors (not necessarily correct results)\n"
        
        # Save report
        report_path = Path(output_dir) / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"{'='*60}")
        print(f"✓ Report saved to: {report_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation on Spider")
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-hf",
                       help="Model to evaluate")
    parser.add_argument("--data", type=str, default="nl2sql_data/eval/spider_dev.jsonl",
                       help="Path to Spider dataset")
    parser.add_argument("--output", type=str, default="results/baseline",
                       help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("\nPlease download Spider dataset first:")
        print("  python src/nl2sql/data/download_all_datasets.py")
        return
    
    # Run evaluation
    evaluator = SpiderEvaluator(model_name=args.model)
    results = evaluator.evaluate(
        data_path=args.data,
        output_dir=args.output,
        num_samples=args.num_samples
    )
    
    print("\n✅ Baseline evaluation complete!")
    print(f"\nResults saved to: {args.output}/")
    print("\nNext steps:")
    print("  1. Review results in results/baseline/")
    print("  2. Train your model: python src/nl2sql/train/train_curriculum_lora.py")
    print("  3. Compare with baselines")


if __name__ == "__main__":
    main()
