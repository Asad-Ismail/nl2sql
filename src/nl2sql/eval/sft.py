"""
Run Evaluation of SFT finetune model on sql dataset
Requirements:
    pip install unsloth datasets tqdm pandas matplotlib seaborn

Usage:
    python sft.py --model models/nl2sql-unsloth/final --num-samples 100
    python sft.py --model models/nl2sql-unsloth/checkpoint-10800 --print-prompts
"""

import os
import json
import sqlite3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    compare_results,
    calculate_metrics,
    print_metrics,
    print_comparison,
    categorize_sql_complexity,
    extract_sql_from_text,
    get_db_path,
    save_evaluation_results,
    save_evaluation_summary,
    generate_markdown_report,
    SQL_PATTERNS
)

class FinetunedModelEvaluator:
    """Evaluate fine-tuned NL2SQL model on Spider dataset"""
    
    def __init__(self, model_path: str, database_dir: str = "database/spider_data/database"):
        self.model_path = model_path
        self.database_dir = database_dir
        self.model = None
        self.tokenizer = None
        
        # Load schemas from tables.json using shared utility
        print(f"\n{'='*60}")
        print(f"Loading Database Schemas from tables.json")
        print(f"{'='*60}\n")
        self.schemas = load_schemas()
        print(f"✓ Loaded schemas for {len(self.schemas)} databases\n") 
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load fine-tuned model"""
        print(f"\n{'='*60}")
        print(f"Loading Fine-tuned Model: {self.model_path}")
        print(f"{'='*60}\n")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        print("Model loaded successfully\n")
    
    def format_prompt(self, schema: str, question: str) -> str:
        """
        Format input prompt EXACTLY matching training template
        Must match custom_alpaca_template from training EXACTLY
        """
        instruction = f"Database Schema:\n{schema}\n\nQuestion: {question}"
        
        # EXACT same format as training (note the leading spaces before ###)
        prompt = f""" ### Instruction: Generate a SQL query to answer the given question based on the provided database schema
    {instruction}
    
     ### Response:
    """
        return prompt
    
    def generate_sql(self, schema: str, question: str, 
                    temperature: float = 0.1, max_new_tokens: int = 512,
                    print_prompt: bool = False) -> str:
        """Generate SQL from natural language"""
        
        prompt = self.format_prompt(schema, question)
        if print_prompt:
            print("\n" + "="*100)
            print("PROMPT SENT TO MODEL:")
            print("="*100)
            print(prompt)
            print("="*100 + "\n")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract SQL
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in full_output:
            sql = full_output.split("### Response:")[-1].strip()
        else:
            sql = full_output[len(prompt):].strip()
        
        # Remove trailing semicolon
        sql = sql.rstrip(';').strip()
        
        return sql
    
    @staticmethod
    def categorize_sql_complexity(sql: str) -> List[str]:
        """Categorize SQL query complexity using shared utility"""
        return categorize_sql_complexity(sql)
    
    def load_dataset_from_hf(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Load Spider dev dataset from HuggingFace"""
        print(f"\n{'='*60}")
        print("Loading Spider Dev Dataset from HuggingFace")
        print(f"{'='*60}\n")
        
        try:
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
                    "query": item.get("sql", ""),
                    "db_id": item.get("db_id", ""),
                    "context": item.get("context", "")  # This might be minimal/empty
                })
            
            # Subsample if requested
            if num_samples and num_samples < len(data):
                print(f"  Subsampling to {num_samples} examples")
                data = data[:num_samples]
            
            print(f"  Using {len(data):,} examples for evaluation\n")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading from HuggingFace: {e}")
            raise
    
    def evaluate(self, output_dir: str = "results/finetuned", num_samples: int = None,
                print_every: int = 10, print_prompts: bool = False):
        """
        Evaluate fine-tuned model on Spider dataset
        
        Args:
            output_dir: Where to save results
            num_samples: Number of samples to evaluate (None = all)
            print_every: Print intermediate results every N examples
            print_prompts: Print prompts being sent to model (for debugging)
        """
        
        print(f"\n{'='*60}")
        print("Fine-tuned Model Evaluation on Spider Dataset")
        print(f"{'='*60}\n")
        
        # Load data
        data = self.load_dataset_from_hf(num_samples=num_samples)
        
        print(f"Evaluating on {len(data)} examples")
        print(f" Printing intermediate results every {print_every} examples")
        if print_prompts:
            print(f"🔍 Printing ALL prompts sent to model")
        print()
        
        # Results storage
        results = []
        
        # Overall metrics
        metrics = {
            "total": 0,
            "valid": 0,
            "matched": 0,
            "exact_match": 0
        }
        
        # Metrics by complexity
        complexity_metrics = defaultdict(lambda: {
            'total': 0, 'valid': 0, 'matched': 0, 'exact_match': 0
        })
        
        # Error analysis
        error_patterns = defaultdict(int)
        
        for i, item in enumerate(tqdm(data, desc="Evaluating")):
            question = item["question"]
            db_id = item.get("db_id", "")
            gold_sql = item.get("query", "")

            schema = self.schemas.get(db_id, item.get("context", f"Database: {db_id}"))
            
            if not schema or schema.startswith("Database:"):
                print(f"\n⚠️  Warning: No proper schema found  will probably hallucinate for db_id={db_id}, skipping...")
                continue
            
            # Database path using shared utility
            db_path = get_db_path(db_id, self.database_dir)
            
            if not os.path.exists(db_path):
                print(f"\n⚠️  Warning: Database not found: {db_path}")
                continue
            
            try:
                # Generate SQL (print prompt on first example or if requested)
                should_print = print_prompts or (i == 0)
                generated_sql = self.generate_sql(schema, question, print_prompt=should_print)
                
                # Execute generated SQL using shared utility
                gen_valid, gen_error, gen_results = execute_sql(
                    generated_sql, db_path
                )
                
                # Execute gold SQL using shared utility
                gold_valid, gold_error, gold_results = execute_sql(
                    gold_sql, db_path
                )
                
                # Compare results using shared utility
                results_match = False
                comparison_feedback = ""
                
                if gen_valid and gold_valid:
                    results_match, comparison_feedback = compare_results(
                        gen_results, gold_results
                    )
                elif not gen_valid:
                    comparison_feedback = f"Generated SQL failed: {gen_error}"
                    # Track error type
                    if "no such table" in str(gen_error).lower():
                        error_patterns["table_name_error"] += 1
                    elif "no such column" in str(gen_error).lower():
                        error_patterns["column_name_error"] += 1
                    elif "syntax" in str(gen_error).lower():
                        error_patterns["syntax_error"] += 1
                    else:
                        error_patterns["other_error"] += 1
                
                # Check exact match
                exact_match = generated_sql.strip().lower() == gold_sql.strip().lower()
                
                # Categorize complexity
                complexity_categories = self.categorize_sql_complexity(gold_sql)
                
                # Store result
                result = {
                    "question": question,
                    "db_id": db_id,
                    "generated_sql": generated_sql,
                    "gold_sql": gold_sql,
                    "is_valid": gen_valid,
                    "error": gen_error,
                    "results_match": results_match,
                    "exact_match": exact_match,
                    "generated_results": gen_results if gen_valid else None,
                    "gold_results": gold_results if gold_valid else None,
                    "comparison_feedback": comparison_feedback,
                    "complexity": complexity_categories
                }
                results.append(result)
                
                # Update overall metrics
                metrics["total"] += 1
                if gen_valid:
                    metrics["valid"] += 1
                if results_match:
                    metrics["matched"] += 1
                if exact_match:
                    metrics["exact_match"] += 1
                
                # Update complexity metrics
                for category in complexity_categories:
                    complexity_metrics[category]["total"] += 1
                    if gen_valid:
                        complexity_metrics[category]["valid"] += 1
                    if results_match:
                        complexity_metrics[category]["matched"] += 1
                    if exact_match:
                        complexity_metrics[category]["exact_match"] += 1
                
                # Print intermediate results
                if (i + 1) % print_every == 0:
                    self._print_comparison(i, result, gen_results, gold_results)
                    
                    # Print running statistics
                    print(f" Running Stats after {metrics['total']} examples:")
                    print(f"   Valid SQL: {metrics['valid']}/{metrics['total']} ({100*metrics['valid']/metrics['total']:.1f}%)")
                    print(f"   Results Match: {metrics['matched']}/{metrics['total']} ({100*metrics['matched']/metrics['total']:.1f}%)")
                    print(f"   Exact Match: {metrics['exact_match']}/{metrics['total']} ({100*metrics['exact_match']/metrics['total']:.1f}%)")
                    print()
                
            except Exception as e:
                print(f"\n❌ Error on example {i}: {e}")
                results.append({
                    "question": question,
                    "db_id": db_id,
                    "error": str(e),
                    "is_valid": False,
                    "results_match": False
                })
        
        # Save and generate report
        self._save_results(results, metrics, complexity_metrics, error_patterns, output_dir)
        self._generate_report(metrics, complexity_metrics, error_patterns, output_dir)
        
        return results, metrics
    
    def _print_comparison(self, idx: int, result: Dict, gen_results, gold_results):
        """Print intermediate comparison results using shared utility"""
        print_comparison(
            idx=idx,
            question=result['question'],
            gen_sql=result['generated_sql'],
            gold_sql=result['gold_sql'],
            gen_results=gen_results,
            gold_results=gold_results,
            results_match=result['results_match'],
            is_valid=result['is_valid']
        )
    
    def _save_results(self, results, metrics, complexity_metrics, error_patterns, output_dir):
        """Save results to files using shared utilities"""
        # Save detailed results using shared utility
        save_evaluation_results(results, output_dir, prefix="detailed_results")
        
        # Save summary using shared utility
        save_evaluation_summary(
            metrics,
            output_dir,
            additional_data={
                "complexity_metrics": dict(complexity_metrics),
                "error_patterns": dict(error_patterns)
            }
        )
        
        print(f"\n Saved results to: {output_dir}/")
    
    def _generate_report(self, metrics, complexity_metrics, error_patterns, output_dir):
        """Generate evaluation report - preserves exact output format"""
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}\n")
        
        # Print to console (unchanged)
        print(f"Overall Metrics:")
        print(f"  Total: {metrics['total']}")
        print(f"  Valid SQL: {metrics['valid']} ({100*metrics['valid']/metrics['total']:.2f}%)")
        print(f"  Results Match: {metrics['matched']} ({100*metrics['matched']/metrics['total']:.2f}%)")
        print(f"  Exact Match: {metrics['exact_match']} ({100*metrics['exact_match']/metrics['total']:.2f}%)\n")
        
        # Build custom sections for report
        overall_section = f"""- Total Samples: {metrics['total']}
- Valid SQL: {metrics['valid']} ({100*metrics['valid']/metrics['total']:.2f}%)
- Results Match Gold: {metrics['matched']} ({100*metrics['matched']/metrics['total']:.2f}%)
- Exact Match: {metrics['exact_match']} ({100*metrics['exact_match']/metrics['total']:.2f}%)"""
        
        # Complexity breakdown
        complexity_section = ""
        for category, stats in sorted(complexity_metrics.items()):
            if stats['total'] > 0:
                complexity_section += f"\n### {category.upper().replace('_', ' ')}\n"
                complexity_section += f"- Total: {stats['total']}\n"
                complexity_section += f"- Valid: {stats['valid']} ({100*stats['valid']/stats['total']:.2f}%)\n"
                complexity_section += f"- Results Match: {stats['matched']} ({100*stats['matched']/stats['total']:.2f}%)\n"
        
        # Error patterns
        error_section = ""
        if error_patterns:
            for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                error_section += f"- {error_type}: {count} ({100*count/metrics['total']:.2f}%)\n"
        
        # Use shared utility to generate report with custom sections
        sections = {
            "Overall Metrics": overall_section,
            "Performance by SQL Complexity": complexity_section,
        }
        if error_section:
            sections["Error Analysis"] = error_section
        
        report_path = generate_markdown_report(
            metrics={},  # Empty since we're using custom sections
            output_dir=output_dir,
            title="Fine-tuned Model Evaluation Results",
            model_name=self.model_path,
            dataset_name="Spider Dev (HuggingFace)",
            additional_sections=sections
        )
        
        print(f"{'='*60}")
        print(f"✅ Report saved to: {report_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned NL2SQL Model")
    parser.add_argument("--model", type=str, default="models/nl2sql-unsloth/checkpoint-10800",
                       help="Path to fine-tuned model")
    parser.add_argument("--database-dir", type=str, default="database/spider_data/database",
                       help="Path to Spider database directory")
    parser.add_argument("--output", type=str, default="results/finetuned",
                       help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--print-every", type=int, default=1,
                       help="Print intermediate results every N examples")
    parser.add_argument("--print-prompts", action="store_true",default=True,
                       help="Print all prompts sent to model (for debugging)")
    
    args = parser.parse_args()
    
    # Check database directory
    if not os.path.exists(args.database_dir):
        print(f"\n❌ Error: Database directory not found: {args.database_dir}")
        print("Please download Spider databases first!")
        return
    
    # Check tables.json
    tables_json = Path("database/spider_data/tables.json")
    if not tables_json.exists():
        print(f"\n❌ Error: tables.json not found at: {tables_json}")
        print("This file is required to load database schemas!")
        return
    
    # Run evaluation
    evaluator = FinetunedModelEvaluator(
        model_path=args.model,
        database_dir=args.database_dir
    )
    
    results, metrics = evaluator.evaluate(
        output_dir=args.output,
        num_samples=args.num_samples,
        print_every=args.print_every,
        print_prompts=args.print_prompts
    )
    
    print("\n FineTune Evaluation complete!")
    print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()