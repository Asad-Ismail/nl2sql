---
name: sql-evaluation
description: SQL evaluation patterns for NL2SQL. Use when evaluating SQL queries, executing SQLite, comparing results, or extracting SQL from LLM output. Covers validation, execution, and result comparison.
---

# SQL Evaluation Patterns

## When to Use
- Evaluating generated SQL queries
- Executing SQL against SQLite databases
- Comparing query results with gold standards
- Extracting SQL from LLM text output
- Validating SQL syntax and execution

## Core Patterns

### SQL Extraction from LLM Output

Use `extract_sql_from_text()` from `src/nl2sql/utils/util.py`:

```python
from nl2sql.utils.util import extract_sql_from_text

llm_output = "Here's the SQL:\n```sql\nSELECT * FROM users;\n```"
sql = extract_sql_from_text(llm_output)
# Returns: "SELECT * FROM users"
```

**What it handles:**
- Markdown code blocks (` ```sql `, ` ``` `)
- DSPy artifacts (leading `]]`)
- Multiple SQL statements (returns first SELECT)
- Delimiters: `\n\n`, `###`, `Question:`, `Schema:`, `--`

### SQLite Execution

Use `execute_sql()` from `src/nl2sql/utils/util.py`:

```python
from nl2sql.utils.util import execute_sql, get_db_path

db_path = get_db_path("concert_singer", "database/spider_data/database")
success, error, results = execute_sql("SELECT * FROM singer", db_path)

if success:
    print(results)  # List of tuples
else:
    print(f"Error: {error}")
```

**Returns:**
- `success` (bool): Query executed without errors
- `error` (str|None): Error message if failed
- `results` (list|None): Query results as list of tuples

### Result Comparison

Use `compare_results()` for row/column-order-independent comparison:

```python
from nl2sql.utils.util import compare_results

matches, feedback = compare_results(generated_results, gold_results)
```

**What it handles:**
- Row order independence
- Column order independence (within rows)
- Type normalization (int vs float, string representations)
- None handling

### Schema Loading

Use `load_schemas()` for Spider database schemas:

```python
from nl2sql.utils.util import load_schemas

schemas = load_schemas("database/spider_data/tables.json")
schema = schemas.get("concert_singer")
# Returns: "CREATE TABLE singer (...)\nCREATE TABLE concert (...)"
```

## Critical Gotchas

### SQL Extraction Fragility
Generated text may include markdown, comments, or artifacts:
- Always use `extract_sql_from_text()`, don't manually parse
- Look for "SELECT" anchor to find query start
- Stop at common delimiters to capture only the SQL

### Schema Context Truncation
SQaLe schemas can exceed token limits:
```python
# Truncate large schemas to prevent token overflow
schema = schema[:1000]  # First 1000 chars
```

### Database Path Construction
Use `get_db_path()` helper:
```python
db_path = get_db_path(db_id, "database/spider_data/database")
# Returns: database/spider_data/database/{db_id}/{db_id}.sqlite
```

### Result Normalization
Comparison automatically handles:
- Type differences: `1` vs `1.0` vs `"1"` → all match
- Column order: `(a, b)` vs `(b, a)` → sorted before compare
- Row order: Sets compared, not lists

## Metrics Calculation

Use `calculate_metrics()` and `print_metrics()`:

```python
from nl2sql.utils.util import calculate_metrics, print_metrics

results = [
    {"is_valid": True, "results_match": True, "inference_time": 2.5},
    {"is_valid": False, "results_match": False, "inference_time": 1.2},
    # ...
]

metrics = calculate_metrics(results)
print_metrics(metrics, method_name="Zero-Shot")
```

**Output:**
```
============================================================
Zero-Shot Results
============================================================
Total Examples:           1,034
Valid SQL:                524 (50.7%)
Results Match Gold:       312 (30.2%)
Avg Inference Time:       2.34s
============================================================
```

## Integration

- **Related skill:** `data-processing` - For dataset field handling
- **Related skill:** `prompt-patterns` - For SQL generation prompts
- **Utility module:** `src/nl2sql/utils/util.py` - All evaluation functions

## Anti-Patterns

### What NOT to Do

**Don't manually parse SQL from LLM output:**
```python
# BAD: Fragile regex parsing
sql = re.search(r'SELECT.*;', llm_output, re.DOTALL).group(0)

# GOOD: Use utility function
sql = extract_sql_from_text(llm_output)
```

**Don't compare results directly:**
```python
# BAD: Fails on order/type differences
if gen_results == gold_results:

# GOOD: Normalizes for order and types
matches, _ = compare_results(gen_results, gold_results)
```

**Don't hardcode database paths:**
```python
# BAD: Brittle path construction
db_path = f"database/spider_data/database/{db_id}/{db_id}.sqlite"

# GOOD: Use helper function
db_path = get_db_path(db_id)
```
