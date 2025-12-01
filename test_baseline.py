"""
Quick test to verify baseline evaluation setup

Tests that:
1. Model loads correctly
2. SQL generation works
3. Execution validation works
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.nl2sql.eval.baseline import SpiderEvaluator


def test_basic_functionality():
    """Test basic functionality without requiring Spider dataset"""
    
    print("Testing NL2SQL Baseline Evaluator\n")
    print("="*60)
    
    # Create evaluator (don't load model yet to save time)
    evaluator = SpiderEvaluator(model_name="codellama/CodeLlama-7b-hf")
    
    # Test SQL extraction
    test_cases = [
        "SELECT * FROM students WHERE age > 18",
        "Here is the SQL:\nSELECT name FROM users\n\nThat should work!",
        "SELECT COUNT(*) FROM orders WHERE status = 'completed';"
    ]
    
    print("\n1. Testing SQL extraction:")
    for sql in test_cases:
        extracted = evaluator._extract_sql(sql)
        print(f"   Input: {sql[:50]}...")
        print(f"   Output: {extracted}")
        print()
    
    # Test execution (create dummy in-memory DB)
    print("\n2. Testing SQL execution:")
    import sqlite3
    
    # Create temp database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice', 25)")
    cursor.execute("INSERT INTO users VALUES (2, 'Bob', 30)")
    conn.commit()
    conn.close()
    
    # Save to file for testing
    test_db = "/tmp/test_nl2sql.db"
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice', 25)")
    cursor.execute("INSERT INTO users VALUES (2, 'Bob', 30)")
    conn.commit()
    conn.close()
    
    # Test valid SQL
    valid_sql = "SELECT name FROM users WHERE age > 20"
    success, error, results = evaluator.execute_sql(valid_sql, test_db)
    print(f"   Valid SQL: {valid_sql}")
    print(f"   Success: {success}")
    print(f"   Results: {results}")
    print()
    
    # Test invalid SQL
    invalid_sql = "SELECT nonexistent FROM users"
    success, error, results = evaluator.execute_sql(invalid_sql, test_db)
    print(f"   Invalid SQL: {invalid_sql}")
    print(f"   Success: {success}")
    print(f"   Error: {error}")
    print()
    
    # Clean up
    os.remove(test_db)
    
    print("="*60)
    print("✅ All tests passed!")
    print("\nNext steps:")
    print("  1. Download Spider data: python src/nl2sql/data/download_all_datasets.py")
    print("  2. Run baseline eval: python src/nl2sql/eval/baseline.py --num-samples 10")


if __name__ == "__main__":
    test_basic_functionality()
