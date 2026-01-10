---
name: data-processing
description: Data processing patterns for NL2SQL datasets. Use when handling datasets, loading from HuggingFace, or processing JSONL files. Covers field handling, schema truncation, dataset loading, and augmentation patterns.
---

# Data Processing Patterns for NL2SQL

## When to Use
- Loading datasets from HuggingFace or local files
- Processing dataset field inconsistencies
- Handling schema context truncation
- Creating data augmentations
- Saving datasets in JSONL format

## Core Patterns

### Standard Dataset Format

NL2SQL uses consistent JSONL format across all datasets:

```json
{
  "dataset": "spider",
  "question": "Find the name of the singer who released the most albums.",
  "sql": "SELECT T1.name FROM singer AS T1 ORDER BY T1.album_count DESC LIMIT 1",
  "db_id": "concert_singer",
  "context": "CREATE TABLE singer (name TEXT, album_count INTEGER)"
}
```

**Field definitions:**
- `dataset` (str): Source dataset name
- `question` (str): Natural language question
- `sql` (str): Ground truth SQL query
- `db_id` (str): Database identifier (optional for some datasets)
- `context` (str): Database schema as CREATE TABLE statements

### Loading from HuggingFace

Standard pattern for loading HuggingFace datasets:

```python
from datasets import load_dataset

# Load dataset by name
dataset = load_dataset("xlangai/spider", split="train")

# Load with data_files (for JSONL)
dataset = load_dataset(
    "json",
    data_files="nl2sql_data/train/spider_train.jsonl",
    split="train"
)

# Load from HuggingFace Hub with specific file
dataset = load_dataset(
    "AsadIsmail/nl2sql-deduplicated",
    data_files="spider_dev_clean.jsonl",
    split="train"
)
```

**Important:** Handle deprecated loading scripts:
```python
# OLD (deprecated, fails):
dataset = load_dataset("wikisql")

# NEW (use json loader):
dataset = load_dataset("json", data_files="data/wikisql.jsonl", split="train")
```

### Field Name Handling

Different datasets use different field names - always use `.get()` with fallbacks:

```python
# SQL field variability
sql = ex.get("sql", ex.get("query", ex.get("text", "")))

# Question field variability
question = ex.get("question", ex.get("sql_prompt", ex.get("text", "")))

# Schema/context variability
context = ex.get("context", ex.get("schema", ""))

# Database ID variability
db_id = ex.get("db_id", ex.get("database_id", ""))
```

**Why this matters:**
- Spider uses: `question`, `query` (not `sql`)
- WikiSQL uses: `sql_prompt` (not `question`)
- SQaLe uses: `schema` (not `context`)
- Gretel uses: `text` for SQL

### Schema Context Truncation

Large schemas (SQaLe) exceed token limits - truncate to 1000 chars:

```python
# Truncate very long schemas to prevent token overflow
schema = ex.get("schema", "")
context = schema[:1000] if schema else ""

# In training prompt creation
def create_prompt(example: Dict) -> str:
    schema = example.get('context', '').strip()
    question = example.get('question', '').strip()

    # Truncate if needed
    if len(schema) > 1000:
        schema = schema[:1000]

    prompt = f"""### Instruction:
Generate a SQL query for the given question.

Database Schema:
{schema}

Question: {question}

### Response:
"""
    return prompt
```

### Saving to JSONL

Standard JSONL save function:

```python
import json
from pathlib import Path

def save_to_jsonl(data: list, path: str) -> int:
    """
    Save list of dicts to JSONL format.

    Args:
        data: List of dictionaries to save
        path: Output file path

    Returns:
        Number of records saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return len(data)

# Usage
data = [
    {"question": "...", "sql": "...", "dataset": "spider"},
    {"question": "...", "sql": "...", "dataset": "spider"},
]
count = save_to_jsonl(data, "nl2sql_data/train/spider_train.jsonl")
print(f"Saved {count:,} examples")
```

### Loading from JSONL

Standard JSONL load function:

```python
import json
from pathlib import Path

def load_jsonl(path: str) -> list:
    """
    Load JSONL file into list of dicts.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Usage
examples = load_jsonl("nl2sql_data/train/spider_train.jsonl")
print(f"Loaded {len(examples):,} examples")
```

### Dataset Filtering and Validation

Filter out invalid examples:

```python
def validate_example(ex: Dict) -> bool:
    """
    Check if example has required fields.

    Args:
        ex: Example dictionary

    Returns:
        True if valid, False otherwise
    """
    question = ex.get("question", ex.get("sql_prompt", "")).strip()
    sql = ex.get("sql", ex.get("query", "")).strip()

    # Must have both question and SQL
    if not question or not sql:
        return False

    # Basic SQL validation
    if "SELECT" not in sql.upper():
        return False

    return True

# Filter dataset
valid_data = [ex for ex in raw_data if validate_example(ex)]
print(f"Filtered to {len(valid_data):,} valid examples (from {len(raw_data):,})")
```

## Dataset-Specific Patterns

### Spider Dataset

```python
# Spider has schema info in separate tables.json file
from nl2sql.utils.util import load_schemas, get_db_path

# Load schemas
schemas = load_schemas("database/spider_data/tables.json")

# Get schema for specific database
db_id = "concert_singer"
schema = schemas.get(db_id, "")
db_path = get_db_path(db_id, "database/spider_data/database")
```

### SQaLe Dataset

```python
# SQaLe has embedded schemas (22,989 unique schemas)
sqale = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")

data = []
for ex in sqale:
    question = ex.get("question", "")
    sql = ex.get("query", ex.get("sql", ""))
    schema = ex.get("schema", "")

    # CRITICAL: Truncate large schemas
    if len(schema) > 1000:
        schema = schema[:1000]

    data.append({
        "dataset": "sqale",
        "question": question,
        "sql": sql,
        "db_id": "",
        "context": schema
    })
```

## Data Augmentation Patterns

### Question Paraphrasing

Generate multiple questions for same SQL:

```python
# Using LLM to paraphrase questions
def augment_question_paraphrase(example: Dict, n_variations: int = 2) -> List[Dict]:
    """Create n variations of the question with same SQL"""
    base_question = example["question"]
    base_sql = example["sql"]
    base_schema = example["context"]

    variations = []
    for _ in range(n_variations):
        paraphrased = llm_paraphrase(base_question)
        variations.append({
            "question": paraphrased,
            "sql": base_sql,
            "context": base_schema,
            "dataset": example["dataset"] + "_paraphrase"
        })

    return variations
```

### Schema Shuffling

Change table/column names while preserving logic:

```python
import random

def augment_schema_shuffle(example: Dict) -> Dict:
    """Shuffle schema names while keeping SQL structure"""
    # This would use a schema-aware transformation
    # Preserving relationships between tables
    shuffled = example.copy()

    # Apply name transformations to both schema and SQL
    # (requires SQL parsing)

    return shuffled
```

### Query Augmentation

Add LIMIT/ORDER BY to existing queries:

```python
def augment_query_additions(example: Dict) -> Dict:
    """Add LIMIT or ORDER BY to existing SQL"""
    sql = example["sql"].rstrip(";")

    # Randomly add LIMIT if not present
    if "LIMIT" not in sql.upper() and random.random() < 0.3:
        limit_val = random.choice([1, 5, 10, 100])
        sql += f" LIMIT {limit_val}"

    # Randomly add ORDER BY if not present
    if "ORDER BY" not in sql.upper() and random.random() < 0.2:
        # Extract first column from SELECT (simplified)
        sql = sql.replace("SELECT", f"SELECT * ORDER BY 1")  # Simplified

    return {**example, "sql": sql}
```

## Critical Gotchas

### HuggingFace Cache Location

Datasets cached at `~/.cache/huggingface/datasets/` - can be large:
```bash
# Check cache size
du -sh ~/.cache/huggingface/datasets/

# Clear cache if needed
rm -rf ~/.cache/huggingface/datasets/
```

### Field Name Consistency

Always use `.get()` with fallbacks - field names vary:
```python
# BAD: Assumes 'sql' field exists
sql = ex["sql"]

# GOOD: Handles multiple field names
sql = ex.get("sql", ex.get("query", ""))
```

### Dataset SQL Dialect

Ensure SQL dialect consistency (standard SQL only):
- **Include:** Spider, SQaLe, Gretel, SQL-Context, Know-SQL
- **Exclude:** DuckDB (uses PRAGMA, different dialect)

### Memory Management

Full 752K dataset at 512 tokens = ~48GB RAM:
```python
# Use subsampling for development
if config.max_samples and len(dataset) > config.max_samples:
    indices = np.random.choice(len(dataset), config.max_samples, replace=False)
    dataset = dataset.select(indices)
```

## Integration

- **Related skill:** `sql-evaluation` - For executing SQL during validation
- **Related skill:** `nl2sql-training` - For dataset configuration and weighting
- **Data downloader:** `src/nl2sql/data/download_all_datasets.py` - Full download pipeline
- **Augmentation script:** `src/nl2sql/data/synthetic_augmentation.py` - Data augmentation

## Anti-Patterns

### What NOT to Do

**Don't assume field names:**
```python
# BAD: Breaks on Spider which uses 'query'
sql = ex["sql"]

# GOOD: Handles variations
sql = ex.get("sql", ex.get("query", ""))
```

**Don't forget schema truncation:**
```python
# BAD: SQaLe schemas exceed token limits
context = ex["schema"]

# GOOD: Truncate to prevent overflow
context = ex.get("schema", "")[:1000]
```

**Don't mix SQL dialects:**
```python
# BAD: DuckDB uses different SQL syntax
datasets = ["spider", "duckdb", "sqale"]  # Mixed dialects

# GOOD: Consistent SQL flavor
datasets = ["spider", "sqale", "gretel"]  # All standard SQL
```
