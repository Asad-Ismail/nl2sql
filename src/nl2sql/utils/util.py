from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# Schema Loading
# ==============================================================================


def load_schemas(tables_path: str = "database/spider_data/tables.json") -> Dict[str, str]:
    """
    Load database schemas from Spider tables.json

    Args:
        tables_path: Path to Spider tables.json file

    Returns:
        Dict mapping db_id to formatted schema string (CREATE TABLE statements)
    """
    schema_file = Path(tables_path)

    if not schema_file.exists():
        logger.error(f"tables.json not found at {tables_path}!")
        return {}

    with open(schema_file) as f:
        tables_data = json.load(f)

    schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]

        # Format schema as CREATE TABLE statements
        schema_lines = []
        for table_idx, table_name in enumerate(table_names):
            cols = [
                (col[1], column_types[i])
                for i, col in enumerate(column_names)
                if col[0] == table_idx
            ]
            if cols:
                col_defs = [f"{name} {dtype}" for name, dtype in cols]
                schema_lines.append(f"CREATE TABLE {table_name} ({', '.join(col_defs)})")

        schemas[db_id] = "\n".join(schema_lines)

    return schemas


# ==============================================================================
# SQL Execution
# ==============================================================================


def execute_sql(sql: str, db_path: str) -> Tuple[bool, Optional[str], Optional[List]]:
    """
    Execute SQL query on SQLite database

    Args:
        sql: SQL query string to execute
        db_path: Path to SQLite database file

    Returns:
        Tuple of (success, error_message, results):
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


def get_db_path(db_id: str, database_dir: str = "database/spider_data/database") -> str:
    """
    Get path to Spider database file

    Args:
        db_id: Database identifier
        database_dir: Root directory containing Spider databases

    Returns:
        Full path to .sqlite file
    """
    return os.path.join(database_dir, db_id, f"{db_id}.sqlite")


# ==============================================================================
# Result Comparison
# ==============================================================================


def normalize_value(val: Any) -> Optional[str]:
    """
    Normalize a single value for comparison

    Handles type differences (int vs float, string representations)

    Args:
        val: Value to normalize

    Returns:
        Normalized string representation or None
    """
    if val is None:
        return None
    return str(val).strip().lower()


def normalize_row(row: Any) -> Tuple:
    """
    Normalize and sort a row to handle column order differences

    Args:
        row: Database row (tuple, list, or single value)

    Returns:
        Normalized and sorted tuple
    """
    if isinstance(row, (list, tuple)):
        # Sort values within the row (column order independent)
        normalized = tuple(sorted(normalize_value(v) for v in row))
    else:
        # Single value row
        normalized = (normalize_value(row),)
    return normalized


def compare_results(
    generated_results: Optional[List], gold_results: Optional[List]
) -> Tuple[bool, str]:
    """
    Compare generated results with gold standard results

    Handles:
    - Row order independence
    - Column order independence (within rows)
    - Type normalization (int vs float, string representations)

    Args:
        generated_results: Results from generated SQL query
        gold_results: Results from gold standard SQL query

    Returns:
        Tuple of (matches, feedback):
            matches (bool): Whether results match
            feedback (str): Explanation of match/mismatch
    """
    if generated_results is None or gold_results is None:
        return False, "Cannot compare - one or both queries failed to execute"

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
        return (
            False,
            f"Result count mismatch: generated {len(gen_set)} rows, expected {len(gold_set)} rows",
        )

    # Same number of rows but different content
    return False, "Results differ from gold standard (same row count, different values)"


# ==============================================================================
# Metrics Calculation
# ==============================================================================


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from results

    Args:
        results: List of evaluation result dictionaries

    Returns:
        Dictionary containing aggregated metrics:
            - valid_sql_count: Number of queries that executed successfully
            - valid_sql_pct: Percentage of valid SQL queries
            - result_match_count: Number of results matching gold standard
            - result_match_pct: Percentage of results matching gold standard
            - avg_inference_time: Average inference time in seconds
            - total_examples: Total number of examples evaluated
    """
    if not results:
        return {
            "valid_sql_count": 0,
            "valid_sql_pct": 0.0,
            "result_match_count": 0,
            "result_match_pct": 0.0,
            "avg_inference_time": 0.0,
            "total_examples": 0,
        }

    valid_sql_count = sum(1 for r in results if r.get("is_valid", False))
    result_match_count = sum(1 for r in results if r.get("results_match", False))
    total_inference_time = sum(r.get("inference_time", 0.0) for r in results)
    total_examples = len(results)

    return {
        "valid_sql_count": valid_sql_count,
        "valid_sql_pct": (valid_sql_count / total_examples * 100) if total_examples > 0 else 0.0,
        "result_match_count": result_match_count,
        "result_match_pct": (
            (result_match_count / total_examples * 100) if total_examples > 0 else 0.0
        ),
        "avg_inference_time": (
            (total_inference_time / total_examples) if total_examples > 0 else 0.0
        ),
        "total_examples": total_examples,
    }


def print_metrics(metrics: Dict[str, Any], method_name: str = ""):
    """
    Print formatted metrics

    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        method_name: Name of evaluation method (for display)
    """
    title = f"{method_name} Results" if method_name else "Results"
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total Examples:           {metrics['total_examples']:,}")
    print(
        f"Valid SQL:                {metrics['valid_sql_count']:,} ({metrics['valid_sql_pct']:.1f}%)"
    )
    print(
        f"Results Match Gold:       {metrics['result_match_count']:,} ({metrics['result_match_pct']:.1f}%)"
    )
    print(f"Avg Inference Time:       {metrics['avg_inference_time']:.2f}s")
    print(f"{'='*60}\n")


# ==============================================================================
# Display/Debugging
# ==============================================================================


def print_comparison(
    idx: int,
    question: str,
    gen_sql: str,
    gold_sql: str,
    gen_results: Optional[List],
    gold_results: Optional[List],
    results_match: bool,
    is_valid: bool,
) -> None:
    """
    Print intermediate comparison results for debugging

    Args:
        idx: Example index
        question: Natural language question
        gen_sql: Generated SQL query
        gold_sql: Gold standard SQL query
        gen_results: Results from generated SQL
        gold_results: Results from gold SQL
        results_match: Whether results match
        is_valid: Whether generated SQL executed successfully
    """
    print(f"\n{'='*70}")
    print(f"Example {idx + 1}")
    print(f"{'='*70}")
    print(f"Question: {question}")
    print(f"\nGenerated SQL:\n  {gen_sql}")
    print(f"{'='*70}")
    print(f"\nGold SQL:\n  {gold_sql}")
    print(f"{'='*70}")

    if is_valid:
        print("\n✅ Generated SQL executed successfully")

        # Safe printing with None checks
        if gen_results and len(gen_results) > 3:
            print(f"Generated Results: {gen_results[:3]}...")
        else:
            print(f"Generated Results: {gen_results}")

        if gold_results is not None and len(gold_results) > 3:
            print(f"Gold Results:      {gold_results[:3]}...")
        elif gold_results is not None:
            print(f"Gold Results:      {gold_results}")
        else:
            print("Gold Results:      None (failed to execute)")

        if results_match:
            print("🎯 MATCH: Results are identical!")
        else:
            print("❌ MISMATCH: Results differ")
    else:
        print("\n❌ Generated SQL failed to execute")
    print(f"{'='*70}\n")


# ==============================================================================
# SQL Pattern Analysis
# ==============================================================================

SQL_PATTERNS = {
    "simple_select": r"^\s*SELECT\s+.*\s+FROM\s+\w+\s*;?\s*$",
    "with_where": r"WHERE",
    "with_join": r"JOIN",
    "with_aggregation": r"(COUNT|SUM|AVG|MIN|MAX|GROUP BY)",
    "with_subquery": r"SELECT.*SELECT",
    "with_union": r"UNION",
    "with_order_limit": r"(ORDER BY|LIMIT)",
}


def categorize_sql_complexity(sql: str) -> List[str]:
    """
    Categorize SQL query complexity based on patterns

    Args:
        sql: SQL query string

    Returns:
        List of complexity categories found in the query
    """
    categories = []
    sql_upper = sql.upper()

    for category, pattern in SQL_PATTERNS.items():
        if re.search(pattern, sql_upper, re.IGNORECASE | re.DOTALL):
            categories.append(category)

    if not categories:
        categories.append("simple_select")

    return categories


# ==============================================================================
# SQL Extraction
# ==============================================================================


def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL query from generated text

    Handles common patterns:
    - SQL wrapped in markdown code blocks
    - SQL mixed with other text
    - Multiple SQL statements (returns first SELECT)

    Args:
        text: Generated text that may contain SQL

    Returns:
        Extracted and cleaned SQL query
    """
    # Remove markdown code blocks
    text = text.replace("```sql", "").replace("```", "").strip()

    # Remove leading brackets (DSPy artifact)
    if text.startswith("]]"):
        text = text[2:].strip()

    text = text.lstrip("\n").strip()

    # Look for SELECT statement
    text_upper = text.upper()
    if "SELECT" in text_upper:
        start = text_upper.find("SELECT")
        sql = text[start:]

        # Stop at common delimiters
        for delimiter in ["\n\n", "###", "Question:", "Schema:", "--"]:
            if delimiter in sql:
                pos = sql.find(delimiter)
                if pos > 0:  # Make sure we don't cut at position 0
                    sql = sql[:pos]

        # Remove trailing semicolon if present
        sql = sql.rstrip(";").strip()
        return sql

    return text.strip()


# ==============================================================================
# Result Saving and Reporting
# ==============================================================================


def save_evaluation_results(results: Any, output_dir: str, prefix: str = "results") -> None:
    """
    Save evaluation results to JSONL file

    Args:
        results: Results to save (list of dicts, dict of lists, etc.)
        output_dir: Directory to save results
        prefix: Filename prefix for saved files
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle different result formats
    if isinstance(results, dict):
        # Dict of lists (e.g., {"zero_shot": [...], "few_shot": [...]})
        for method, data in results.items():
            filepath = output_path / f"{method}_results.jsonl"
            with open(filepath, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
    elif isinstance(results, list):
        # Single list of results
        filepath = output_path / f"{prefix}.jsonl"
        with open(filepath, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError(f"Unsupported results type: {type(results)}")


def save_evaluation_summary(
    metrics: Dict, output_dir: str, additional_data: Optional[Dict] = None
) -> None:
    """
    Save evaluation summary as JSON

    Args:
        metrics: Metrics dictionary
        output_dir: Directory to save summary
        additional_data: Additional data to include (complexity metrics, error patterns, etc.)
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {"overall_metrics": metrics}

    if additional_data:
        summary.update(additional_data)

    filepath = output_path / "summary.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)


def generate_markdown_report(
    metrics: Dict[str, Any],
    output_dir: str,
    filename: str = "evaluation_report.md",
    title: str = "Evaluation Results",
    model_name: str = "",
    dataset_name: str = "",
    additional_sections: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate markdown evaluation report

    Args:
        metrics: Dictionary of metrics (can be nested for multiple methods)
        output_dir: Directory to save report
        title: Report title
        model_name: Model name to include in report
        dataset_name: Dataset name to include in report
        additional_sections: Additional markdown sections to include

    Returns:
        Path to saved report file
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = f"# {title}\n\n"

    if model_name:
        report += f"**Model:** {model_name}\n\n"
    if dataset_name:
        report += f"**Dataset:** {dataset_name}\n\n"

    report += "## Summary\n\n"

    # Check if metrics is a single dict or dict of dicts (multiple methods)
    if isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
        # Multiple methods (e.g., baseline with zero_shot, few_shot, etc.)
        report += "| Method | Valid SQL % | Results Match % | Avg Time (s) |\n"
        report += "|--------|-------------|-----------------|-------------|\n"

        for method_name, method_metrics in metrics.items():
            if isinstance(method_metrics, dict) and "valid_sql_pct" in method_metrics:
                report += f"| {method_name.replace('_', ' ').title()} | "
                report += f"{method_metrics['valid_sql_pct']:.1f}% | "
                report += f"{method_metrics['result_match_pct']:.1f}% | "
                report += f"{method_metrics.get('avg_inference_time', 0):.2f} |\n"
    else:
        # Single method metrics
        if "total_examples" in metrics:
            report += f"- **Total Examples:** {metrics['total_examples']}\n"
            report += (
                f"- **Valid SQL:** {metrics['valid_sql_count']} ({metrics['valid_sql_pct']:.2f}%)\n"
            )
            report += f"- **Results Match Gold:** {metrics['result_match_count']} ({metrics['result_match_pct']:.2f}%)\n"
            if "avg_inference_time" in metrics:
                report += f"- **Avg Inference Time:** {metrics['avg_inference_time']:.2f}s\n"

    # Add additional sections
    if additional_sections:
        for section_title, section_content in additional_sections.items():
            report += f"\n## {section_title}\n\n"
            report += section_content + "\n"

    # Save report
    report_path = output_path / filename
    with open(report_path, "w") as f:
        f.write(report)

    return str(report_path)
