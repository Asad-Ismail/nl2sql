"""
Baseline Evaluation for NL2SQL on Spider Dataset (using unified LLM provider system)

Evaluates 3 standard baseline approaches:
1. Zero-shot: Single LLM call without examples
2. Few-shot: With 2 similar examples
3. Self-correction: Generate → Fix execution errors → LLM validation → Result comparison

Usage:
    python baseline.py --model codellama_7b
    python baseline.py --model claude_sonnet --num-samples 50
    python baseline.py --model llama_70b_nvidia --config-path path/to/providers.yaml
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import re
from nl2sql.llm.factory import get_llm
from nl2sql.llm.base import LLMMessage
from nl2sql.utils.prompts import ZERO_SHOT_PROMPT, BASELINE_SYSTEM_PROMPT, BASELINE_USER_PROMPT_TEMPLATE
from nl2sql.utils.util import (
    load_schemas,
    execute_sql,
    print_comparison,
    compare_results,
    calculate_metrics,
    extract_sql_from_text,
    categorize_sql_complexity,
    save_evaluation_results,
    generate_markdown_report,
    TokenStats,
    extract_token_usage,
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


class BM25Retriever:
    """Retrieve similar examples using BM25 ranking"""

    def __init__(self, corpus: List[Dict]):
        """
        Initialize BM25 retriever with a corpus of examples

        Args:
            corpus: List of examples with 'question' field
        """
        self.corpus = corpus
        # Tokenize questions for BM25
        tokenized_corpus = [self._tokenize(ex["question"]) for ex in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        # Keep alphanumeric characters and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Split on whitespace
        tokens = text.split()
        return tokens

    def retrieve(self, query: str, k: int = 2, exclude_indices: Optional[List[int]] = None) -> List[Dict]:
        """
        Retrieve top-k most similar examples using BM25

        Args:
            query: Question to find similar examples for
            k: Number of examples to retrieve
            exclude_indices: Optional list of indices to exclude (e.g., current example)

        Returns:
            List of k most similar examples
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Filter out excluded indices
        if exclude_indices:
            top_indices = [i for i in top_indices if i not in exclude_indices]

        # Return top-k examples
        return [self.corpus[i] for i in top_indices[:k]]


class SemanticRetriever:
    """Retrieve similar examples using sentence transformer embeddings"""

    def __init__(self, corpus: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic retriever with sentence transformer

        Args:
            corpus: List of examples with 'question' field
            model_name: Name of sentence transformer model
        """
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.model = SentenceTransformer(model_name)

        # Pre-compute embeddings for all questions
        print(f"Computing embeddings for {len(corpus)} examples using {model_name}...")
        self.embeddings = self.model.encode(
            [ex["question"] for ex in corpus],
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print("✓ Embeddings computed\n")

    def retrieve(self, query: str, k: int = 2, exclude_indices: Optional[List[int]] = None) -> List[Dict]:
        """
        Retrieve top-k most similar examples using cosine similarity

        Args:
            query: Question to find similar examples for
            k: Number of examples to retrieve
            exclude_indices: Optional list of indices to exclude

        Returns:
            List of k most similar examples
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Compute query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

        # Filter out excluded indices
        if exclude_indices:
            top_indices = [i for i in top_indices if i not in exclude_indices]

        # Return top-k examples
        return [self.corpus[i] for i in top_indices[:k]]


class SpiderEvaluator:
    """Evaluate baseline approaches on Spider dataset using unified LLM provider system"""

    def __init__(
        self,
        model_name: str = "codellama_7b",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator with a model from providers.yaml

        Args:
            model_name: Model name from providers.yaml (e.g., "codellama_7b", "claude_sonnet")
            config_path: Optional path to custom providers.yaml configuration file
        """
        self.model_name = model_name
        self.config_path = config_path
        self.llm = None
        self.semantic_validator = None
        self.schemas = load_schemas()
        self.token_stats = TokenStats()
        # Per-method token stats
        self.method_token_stats = {
            "zero_shot": TokenStats(),
            "few_shot": TokenStats(),
            "self_correction": TokenStats(),
        }

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
                    "Could not load from HuggingFace or local file. "
                    "Please run: python src/nl2sql/data/download_all_datasets.py"
                )

    def load_model(self):
        """Load LLM provider using unified factory"""
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_name}")
        print(f"{'='*60}\n")

        try:
            # Get provider from unified factory
            self.llm = get_llm(self.model_name, config_path=self.config_path)

            # Test connection
            test_response = self.llm.generate_text("SELECT", max_tokens=20)
            print("✓ Successfully connected to provider\n")
            print(f"Provider: {self.llm.provider_type}")
            print(f"Model: {self.llm.model}\n")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nAvailable models from providers.yaml:")
            from nl2sql.llm.factory import LLMFactory
            factory = LLMFactory(self.config_path)
            for model in factory.list_models():
                print(f"  - {model}")
            raise

        # Initialize semantic validator
        self.semantic_validator = SemanticValidator(self.generate_sql)

    def generate_sql(
        self, user_prompt: str, system_prompt: str = None, max_new_tokens: int = 1024, method: str = None
    ) -> str:
        """Generate SQL using unified LLM provider

        Args:
            user_prompt: The user prompt (schema, question, etc.)
            system_prompt: Optional system prompt (task instruction)
            max_new_tokens: Maximum tokens to generate
            method: Method name for per-method token tracking ("zero_shot", "few_shot", "self_correction")
        """
        try:
            # Use LLMFactory provider (rate limiting handled automatically)
            # Need to use generate() instead of generate_text() to get usage info
            messages = []
            if system_prompt:
                messages.append(LLMMessage(role="system", content=system_prompt))
            messages.append(LLMMessage(role="user", content=user_prompt))
            response = self.llm.generate(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
                stop=["\n\n", "###"],
            )

            # Track token usage (both overall and per-method)
            usage = extract_token_usage(response.usage)
            self.token_stats.add(usage["prompt_tokens"], usage["completion_tokens"])
            if method and method in self.method_token_stats:
                self.method_token_stats[method].add(usage["prompt_tokens"], usage["completion_tokens"])

            sql = self._extract_sql(response.content)
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

        user_prompt = BASELINE_USER_PROMPT_TEMPLATE.format(schema=schema, question=question)

        start_time = time.time()
        sql = self.generate_sql(user_prompt, system_prompt=BASELINE_SYSTEM_PROMPT, max_new_tokens=250, method="zero_shot")
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

        # System prompt: task instruction
        system_prompt = "### Task: Convert natural language questions to SQL queries. Give only SQL Query as Output"

        # User prompt: examples + current question
        user_prompt = "### Examples:\n\n"

        # Add 2-3 examples
        for i, ex in enumerate(examples[:2], 1):
            user_prompt += f"**Example {i}:**\n"
            user_prompt += f"Schema: {ex.get('context', 'N/A')}\n"
            user_prompt += f"Question: {ex['question']}\n"
            user_prompt += f"SQL: {ex.get('query', ex.get('sql', ''))}\n\n"

        user_prompt += "### Now convert this question:\n"
        user_prompt += f"Schema:\n{schema}\n\n"
        user_prompt += f"Question: {question}\n\n"
        user_prompt += "SQL Query:\n"

        start_time = time.time()
        sql = self.generate_sql(user_prompt, system_prompt=system_prompt, max_new_tokens=250, method="few_shot")
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

        user_prompt = BASELINE_USER_PROMPT_TEMPLATE.format(schema=schema, question=question)

        for attempt_num in range(max_attempts):
            # Generate SQL
            sql = self.generate_sql(user_prompt, system_prompt=BASELINE_SYSTEM_PROMPT, max_new_tokens=1024, method="self_correction")
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
                user_prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
                user_prompt += "\n-- Error: Generated empty SQL"
                user_prompt += "\n-- Please generate a valid SQL query:\n-- SQL:\n"
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
                user_prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
                user_prompt += f"\n-- SQL: {sql}"
                user_prompt += f"\n-- Execution error: {error}"
                user_prompt += "\n-- Fix the error and try again:\n"
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
            user_prompt += f"\n\n-- Previous attempt {attempt_num + 1}:"
            user_prompt += f"\n-- SQL: {sql}"
            user_prompt += f"\n-- LLM Feedback: {llm_validation[:200]}..."
            user_prompt += "\n\n-- Generate improved SQL:"
            user_prompt += "\n-- SQL:"  # Clear signal to generate here

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
        self,
        output_dir: str = "results/baseline",
        num_samples: int = None,
        print_every: int = 10,
        methods: List[str] = None,
        retriever_type: str = "bm25",
    ):
        """
        Evaluate baseline methods on Spider dataset from HuggingFace

        Args:
            output_dir: Where to save results
            num_samples: Number of samples to evaluate (None = all)
            print_every: Print intermediate results every N examples
            methods: List of methods to run (None = all three)
            retriever_type: Type of retriever for few-shot ('bm25' or 'semantic')
        """

        # Default to all methods if not specified
        if methods is None:
            methods = ["zero_shot", "few_shot", "self_correction"]

        print(f"\n{'='*60}")
        print("Baseline Evaluation on Spider Dataset (vLLM)")
        print(f"{'='*60}\n")
        print(f"Methods to run: {', '.join(methods)}\n")

        # Load data from HuggingFace
        data = self.load_dataset_from_hf(num_samples=num_samples)

        print(f"Evaluating on {len(data)} examples\n")
        print(f"Printing intermediate results every {print_every} examples\n")

        # Initialize retriever only if few-shot is enabled
        if "few_shot" in methods:
            print(f"Initializing {retriever_type.upper()} retriever for few-shot example selection...")
            if retriever_type == "semantic":
                retriever = SemanticRetriever(data)
            else:
                retriever = BM25Retriever(data)
            print(f"✓ {retriever_type.upper()} retriever ready\n")
        else:
            retriever = None

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
        for method_name in methods:
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
                        # Use selected retriever to retrieve most similar examples
                        examples = retriever.retrieve(question, k=2, exclude_indices=[i])
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

        # Add token statistics to results (both overall and per-method)
        results["token_stats"] = self.token_stats.to_dict()
        results["per_method_token_stats"] = {
            method: stats.to_dict() for method, stats in self.method_token_stats.items()
        }

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
            # Skip non-list items (e.g., token_stats)
            if not isinstance(data, list):
                continue

            metrics = calculate_metrics(data)
            all_metrics[method] = metrics

            # Add token stats to metrics if available
            if "per_method_token_stats" in results and method in results["per_method_token_stats"]:
                metrics["total_tokens"] = results["per_method_token_stats"][method]["total_tokens"]
                metrics["total_calls"] = results["per_method_token_stats"][method]["total_calls"]

            # Print to console
            print(f"{method.upper()}")
            print(
                f"  Valid SQL: {metrics['valid_sql_count']}/{metrics['total_examples']} ({metrics['valid_sql_pct']:.1f}%)"
            )
            print(
                f"  Results Match Gold: {metrics['result_match_count']}/{metrics['total_examples']} ({metrics['result_match_pct']:.1f}%)"
            )
            print()

        # Print overall token statistics
        if "token_stats" in results:
            ts = results["token_stats"]
            print(f"{'='*60}")
            print("OVERALL TOKEN STATISTICS")
            print(f"{'='*60}")
            print(f"  Total LLM Calls:        {ts['total_calls']:,}")
            print(f"  Total Prompt Tokens:    {ts['total_prompt_tokens']:,}")
            print(f"  Total Completion Tokens: {ts['total_completion_tokens']:,}")
            print(f"  Total Tokens:            {ts['total_tokens']:,}")
            print(f"{'='*60}\n")

        # Build summary table with token stats
        summary_table = "| Method | Valid SQL % | Results Match % | Total Tokens | LLM Calls |\n"
        summary_table += "|--------|-------------|-----------------|--------------|----------|\n"

        for method, metrics in all_metrics.items():
            summary_table += f"| {method.replace('_', ' ').title()} | "
            summary_table += f"{metrics['valid_sql_pct']:.1f}% | "
            summary_table += f"{metrics['result_match_pct']:.1f}% | "

            # Add token stats
            if "per_method_token_stats" in results and method in results["per_method_token_stats"]:
                ts = results["per_method_token_stats"][method]
                summary_table += f"{ts['total_tokens']:,} | "
                summary_table += f"{ts['total_calls']:,} |\n"
            else:
                summary_table += "| N/A | N/A |\n"

        # Print per-method token statistics to console
        if "per_method_token_stats" in results:
            print(f"{'='*60}")
            print("PER-METHOD TOKEN STATISTICS")
            print(f"{'='*60}")
            for method, ts in results["per_method_token_stats"].items():
                print(f"{method.upper().replace('_', ' ')}:")
                print(f"  LLM Calls:        {ts['total_calls']:,}")
                print(f"  Total Tokens:      {ts['total_tokens']:,}")
                print(f"  Prompt Tokens:     {ts['total_prompt_tokens']:,}")
                print(f"  Completion Tokens: {ts['total_completion_tokens']:,}")
                print()
            print(f"{'='*60}\n")

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
    parser = argparse.ArgumentParser(
        description="Baseline Evaluation on Spider (using unified LLM provider system)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codellama_7b",
        help="Model name from providers.yaml (e.g., codellama_7b, claude_sonnet, llama_70b_nvidia)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to custom providers.yaml configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: results/baseline_<model_name>)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print intermediate results every N examples (default: 10)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["zero_shot", "few_shot", "self_correction"],
        choices=["zero_shot", "few_shot", "self_correction", "all"],
        help="Which baseline methods to run (default: all)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "semantic"],
        help="Retriever type for few-shot learning (default: bm25)",
    )

    args = parser.parse_args()

    # Handle "all" shortcut for methods
    if "all" in args.methods:
        methods_to_run = ["zero_shot", "few_shot", "self_correction"]
    else:
        methods_to_run = args.methods

    print(f"\nMethods to run: {', '.join(methods_to_run)}")
    print(f"Retriever type: {args.retriever.upper()}")

    # Generate output directory based on model name if not specified
    if args.output is None:
        args.output = f"results/baseline_{args.model}"

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION WITH UNIFIED LLM PROVIDER SYSTEM")
    print("=" * 60)
    print(f"\nModel: {args.model}")
    if args.config_path:
        print(f"Config: {args.config_path}")
    print("=" * 60 + "\n")

    # Run evaluation
    evaluator = SpiderEvaluator(model_name=args.model, config_path=args.config_path)
    evaluator.evaluate(
        output_dir=args.output,
        num_samples=args.num_samples,
        print_every=args.print_every,
        methods=methods_to_run,
        retriever_type=args.retriever,
    )

    print("\n✓ Baseline evaluation complete!")
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
