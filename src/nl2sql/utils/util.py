from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


def load_schemas(tables_path:str="database/spider_data/tables.json"):
    """Load database schemas from Spider tables.json"""
    schema_file = Path(tables_path)
    
    if not schema_file.exists():
        logger.error("tables.json not found!")
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


def execute_sql(sql: str, db_path: str) -> Tuple[bool, Optional[str], Optional[List]]:
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

def print_comparison( idx: int, question: str, gen_sql: str, gold_sql: str, gen_results, gold_results, results_match: bool, is_valid: bool):
        """Print intermediate comparison results"""
        print(f"\n{'='*70}")
        print(f"Example {idx + 1}")
        print(f"{'='*70}")
        print(f"Question: {question}")
        print(f"\nGenerated SQL:\n  {gen_sql}")  
        print(f"{'='*70}")
        print(f"\nGold SQL:\n  {gold_sql}") 
        print(f"{'='*70}")
        
        if is_valid:
            print(f"\n✅ Generated SQL executed successfully")
            
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
                print(f"Gold Results:      None (failed to execute)")
            
            if results_match:
                print(f"🎯 MATCH: Results are identical!")
            else:
                print(f"❌ MISMATCH: Results differ")
        else:
            print(f"\n❌ Generated SQL failed to execute")
        print(f"{'='*70}\n")


SQL_PATTERNS = {
    'simple_select': r'^\s*SELECT\s+.*\s+FROM\s+\w+\s*;?\s*$',
    'with_where': r'WHERE',
    'with_join': r'JOIN',
    'with_aggregation': r'(COUNT|SUM|AVG|MIN|MAX|GROUP BY)',
    'with_subquery': r'SELECT.*SELECT',
    'with_union': r'UNION',
    'with_order_limit': r'(ORDER BY|LIMIT)',
}
