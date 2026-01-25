"""
Push cleaned NL2SQL dataset to HuggingFace Hub.

Uploads:
- Training datasets (4 separate files for weighted sampling)
- Evaluation dataset (Spider dev)
- Configuration with recommended weights
- Statistics and metadata
"""

import json
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm

# Configuration
REPO_ID = "AsadIsmail/nl2sql-deduplicated"
LOCAL_DIR = Path("nl2sql_data/unsloth")


def create_readme():
    """Generate README for the dataset"""

    # Load stats
    with open(LOCAL_DIR / "preparation_stats.json") as f:
        stats = json.load(f)

    readme = f"""---
license: cc-by-4.0
task_categories:
  - text2text-generation
  - text-generation
language:
  - en
tags:
  - sql
  - text-to-sql
  - database
  - query-generation
  - spider
  - sqale
size_categories:
  - 100K<n<1M
pretty_name: NL2SQL Deduplicated Training Dataset
---

# NL2SQL Deduplicated Training Dataset

A curated and deduplicated Text-to-SQL training dataset with {stats['valid_unique']:,} unique examples from 4 high-quality sources.

## Dataset Summary

- **Total Examples**: {stats['valid_unique']:,} unique question-SQL pairs
- **Sources**: Spider, SQaLe, Gretel Synthetic, SQL-Create-Context
- **Deduplication Strategy**: Input-only (question-based) with conflict resolution via quality priority
- **Conflicts Resolved**: {stats['conflicts_resolved']:,} cases where same question had different SQL
- **SQL Dialect**: Standard SQL (SQLite/MySQL/PostgreSQL compatible)
- **Evaluation Set**: 1,034 Spider dev examples (20 unseen databases)

## Key Features

### 1. Schema Enrichment
- All Spider examples enriched with full CREATE TABLE schemas from tables.json
- Consistent schema format across all datasets (100% coverage)
- Models receive complete table/column information during training
- Matches evaluation format for zero train-test distribution mismatch

### 2. Conflict Resolution
- Uses question-only deduplication to prevent conflicting labels
- When same question appears with different SQL, keeps highest-quality version based on:
  - Spider (Priority 5) - Benchmark quality
  - SQaLe (Priority 4) - Real-world schemas
  - Gretel (Priority 3) - Synthetic quality
  - SQL-Context (Priority 2) - Schema-aware

This prevents gradient confusion where model sees same input with different outputs.

### 3. SQL Dialect Validation
- All queries validated for standard SQL compatibility
- Rejected: DuckDB-specific (PRAGMA, DESCRIBE), PostgreSQL-specific (RETURNING, ILIKE, ::)
- {stats['invalid_sql']:,} invalid queries filtered out

### 4. Separated for Weighted Sampling

Each source is in a separate file to enable weighted training:

| Dataset | Examples | Recommended Weight | Reason |
|---------|----------|-------------------|---------|
| Spider | {stats['per_dataset']['spider']['kept']:,} | 0.50 | Benchmark quality, complex multi-table queries |
| SQaLe | {stats['per_dataset']['sqale']['kept']:,} | 0.30 | 22,989 real-world schemas |
| Gretel | {stats['per_dataset']['gretel']['kept']:,} | 0.15 | High-quality synthetic diversity |
| SQL-Context | {stats['per_dataset']['sql_context']['kept']:,} | 0.03 | Schema-aware supplementary |

## Dataset Structure

```
nl2sql-deduplicated/
├── spider_clean.jsonl          # 6,956 examples
├── sqale_clean.jsonl           # 502,837 examples
├── gretel_clean.jsonl          # 99,013 examples
├── sql_context_clean.jsonl     # 74,209 examples
├── spider_dev_clean.jsonl      # 1,034 eval examples
├── unsloth_config.json         # Training configuration
└── preparation_stats.json      # Deduplication statistics
```

## Usage

### Load All Training Data

```python
from datasets import load_dataset

# Load all training data combined (683K examples)
dataset = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="*_clean.jsonl", split="train")

print(f"Total examples: {{len(dataset):,}}")
```

### Load Individual Sources (for Weighted Sampling)

```python
from datasets import load_dataset

# Load separately for weighted training
spider = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
sqale = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="sqale_clean.jsonl", split="train")
gretel = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="gretel_clean.jsonl", split="train")
sql_context = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="sql_context_clean.jsonl", split="train")

# Load eval set
eval_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")
```

## Data Format

Each example is a JSON object with:

```json
{{
  "dataset": "spider",
  "question": "How many singers do we have?",
  "sql": "SELECT COUNT(*) FROM singer",
  "db_id": "concert_singer",
  "context": "CREATE TABLE stadium (Stadium_ID number, ...)",
  "source_dataset": "spider"
}}
```

### Fields:
- `question` (str): Natural language question
- `sql` (str): Target SQL query (standard SQL)
- `db_id` (str): Database identifier (for Spider examples)
- `context` (str): Full CREATE TABLE schemas with columns/types
- `dataset` (str): Original source dataset
- `source_dataset` (str): Dataset kept after deduplication

## Statistics

### Overall
- **Total Loaded**: {stats['total_loaded']:,} raw examples
- **Spider Enriched with Schemas**: {stats.get('spider_enriched', 'N/A'):,} examples
- **Invalid SQL**: {stats['invalid_sql']:,} ({100*stats['invalid_sql']/stats['total_loaded']:.1f}%)
- **Duplicates**: {stats['duplicates']:,} ({100*stats['duplicates']/stats['total_loaded']:.1f}%)
- **Conflicts Resolved**: {stats['conflicts_resolved']:,} same question, different SQL
- **Final Unique**: {stats['valid_unique']:,} ({100*stats['valid_unique']/stats['total_loaded']:.1f}% retained)

### Per-Dataset Contribution

```
Dataset       Loaded    Invalid  Duplicates  Kept     Retention
Spider        7,000     4        40          6,956    99.4%
SQaLe         517,676   13,732   1,107       502,837  97.1%
Gretel        100,000   774      213         99,013   99.0%
SQL-Context   78,577    93       4,275       74,209   94.4%
Know-SQL      49,456    49       49,407      0        0.0% (all duplicates)
```

## Evaluation

Use `spider_dev_clean.jsonl` (1,034 examples) for validation:
- 20 unseen databases (zero overlap with training)
- Tests generalization to new schemas
- Standard benchmark for Text-to-SQL

### Expected Performance Targets:
- Baseline (zero-shot): 40-50% valid SQL
- After training: 70-85% valid SQL (on execution)
- State-of-the-art: >80% exact match

## Methodology

### Deduplication Process
1. Load datasets in priority order (Spider → SQaLe → Gretel → SQL-Context → Know-SQL)
2. Normalize questions: Remove punctuation, lowercase, whitespace
3. Hash question only (not SQL) to catch conflicting labels
4. On conflict: Keep SQL from highest-priority dataset
5. SQL validation: Parse with sqlparse, reject non-standard dialects

### Why Input-Only Deduplication?

**Problem**: Same question with different SQL → Gradient confusion during training

Example conflict:
- Spider: "Count users" → `SELECT COUNT(*) FROM users`
- Gretel: "Count users" → `SELECT COUNT(id) FROM users`

**Solution**: Hash question only, keep Spider's version (higher priority)

This resolved {stats['conflicts_resolved']:,} conflicts that would have hurt model convergence.

## Source Datasets

1. **Spider** (xlangai/spider)
   - 7,000 training examples, 1,034 dev examples
   - 200 databases with complex multi-table queries
   - Benchmark standard for Text-to-SQL

2. **SQaLe** (trl-lab/SQaLe-text-to-SQL-dataset)
   - 517,676 examples across 22,989 real schemas
   - Grounded in real-world database diversity

3. **Gretel Synthetic** (gretelai/synthetic_text_to_sql)
   - 100,000 high-quality synthetic examples
   - Diverse SQL patterns and complexity

4. **SQL-Create-Context** (b-mc2/sql-create-context)
   - 78,577 schema-aware examples
   - Context-rich queries

## License

CC-BY-4.0 (inherits from source datasets)

## Citation

If you use this dataset, please cite the original sources:

```bibtex
@inproceedings{{yu2018spider,
  title={{Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task}},
  author={{Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others}},
  booktitle={{EMNLP}},
  year={{2018}}
}}
```

## Full Documentation

For complete details on dataset creation, verification, and reproducibility, see: `data/data.md` in the repository.
"""

    return readme


def main():
    print("=" * 70)
    print("Pushing NL2SQL Dataset to HuggingFace Hub")
    print("=" * 70)

    # Initialize API
    api = HfApi()

    # Create README
    print("\n[1/3] Generating README...")
    readme_content = create_readme()
    readme_path = LOCAL_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  [OK] README created: {readme_path}")

    # Get all files to upload
    files_to_upload = [
        "spider_clean.jsonl",
        "sqale_clean.jsonl",
        "gretel_clean.jsonl",
        "sql_context_clean.jsonl",
        "spider_dev_clean.jsonl",
        "unsloth_config.json",
        "preparation_stats.json",
        "README.md",
    ]

    print("\n[2/3] Verifying files...")
    for filename in files_to_upload:
        filepath = LOCAL_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {filename:.<40} {size_mb:>6.1f} MB")
        else:
            print(f"  [MISSING] {filename}")
            return

    print(f"\n[3/3] Uploading to {REPO_ID}...")

    try:
        # Upload each file with progress
        for filename in tqdm(files_to_upload, desc="Uploading files"):
            filepath = LOCAL_DIR / filename
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="dataset",
            )

        print(f"\n{'='*70}")
        print("Dataset uploaded successfully!")
        print(f"{'='*70}")
        print(f"\nView at: https://huggingface.co/datasets/{REPO_ID}")
        print(f"\nQuick start:")
        print("  from datasets import load_dataset")
        print(f'  dataset = load_dataset("{REPO_ID}", data_files="*_clean.jsonl", split="train")')
        print("  print(len(dataset))  # 683,015 examples")

    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check you're logged in: huggingface-cli whoami")
        print("2. Verify repo exists")
        print("3. Check write permissions")


if __name__ == "__main__":
    main()
