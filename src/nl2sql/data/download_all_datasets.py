"""
NL2SQL Dataset Downloader
Downloads Text-to-SQL datasets from HuggingFace.

Downloads:
- Spider: 7,000 training examples
- SQaLe: 517,676 examples with 22,989 real schemas
- Gretel: 100,000 synthetic examples
- SQL-Context: 78,577 schema-aware examples
- Know-SQL: 49,456 educational examples
- Spider Dev: 1,034 evaluation examples
"""

import os
import json
from datasets import load_dataset

os.makedirs("nl2sql_data", exist_ok=True)
os.makedirs("nl2sql_data/train", exist_ok=True)
os.makedirs("nl2sql_data/eval", exist_ok=True)


def save_to_jsonl(data, path):
    """Save list of dicts to JSONL format."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return len(data)


print("=" * 70)
print("NL2SQL Dataset Downloader")
print("=" * 70)

datasets_info = []
total_train = 0
total_eval = 0

# ========================================
# TRAINING DATASETS
# ========================================

print("\n[1/5] Downloading Spider Training Set...")
try:
    spider_train = load_dataset("xlangai/spider", split="train")

    data = []
    for ex in spider_train:
        data.append(
            {
                "dataset": "spider",
                "question": ex["question"],
                "sql": ex["query"],
                "db_id": ex.get("db_id", ""),
                "context": f"Database: {ex.get('db_id', '')}",
            }
        )

    count = save_to_jsonl(data, "nl2sql_data/train/spider_train.jsonl")
    print(f"  [OK] Spider Train: {count:,} examples")
    datasets_info.append(("Spider Train", count, "train"))
    total_train += count
except Exception as e:
    print(f"  [FAILED] {str(e)[:80]}")

print("\n[2/5] Downloading SQaLe Dataset...")
try:
    sqale = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")

    data = []
    print(f"  Processing {len(sqale):,} examples...")

    for ex in sqale:
        question = ex.get("question", "")
        sql = ex.get("query", ex.get("sql", ""))
        schema = ex.get("schema", "")

        if question and sql:
            data.append(
                {
                    "dataset": "sqale",
                    "question": question,
                    "sql": sql,
                    "db_id": "",
                    "context": schema[:1000] if schema else "",
                }
            )

    count = save_to_jsonl(data, "nl2sql_data/train/sqale.jsonl")
    print(f"  [OK] SQaLe: {count:,} examples")
    datasets_info.append(("SQaLe", count, "train"))
    total_train += count
except Exception as e:
    print(f"  [FAILED] {str(e)[:80]}")

print("\n[3/5] Downloading Gretel Synthetic...")
try:
    gretel = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    data = []
    for ex in gretel:
        data.append(
            {
                "dataset": "gretel-synthetic",
                "question": ex["sql_prompt"],
                "sql": ex["sql"],
                "db_id": "",
                "context": ex.get("sql_context", ""),
            }
        )

    count = save_to_jsonl(data, "nl2sql_data/train/gretel_train.jsonl")
    print(f"  [OK] Gretel Synthetic: {count:,} examples")
    datasets_info.append(("Gretel Synthetic", count, "train"))
    total_train += count
except Exception as e:
    print(f"  [FAILED] {str(e)[:80]}")

print("\n[4/5] Downloading SQL-Create-Context...")
try:
    sql_ctx = load_dataset("b-mc2/sql-create-context", split="train")

    data = []
    for ex in sql_ctx:
        data.append(
            {
                "dataset": "sql-context",
                "question": ex["question"],
                "sql": ex["answer"],
                "db_id": "",
                "context": ex.get("context", ""),
            }
        )

    count = save_to_jsonl(data, "nl2sql_data/train/sql_context_train.jsonl")
    print(f"  [OK] SQL-Context: {count:,} examples")
    datasets_info.append(("SQL-Context", count, "train"))
    total_train += count
except Exception as e:
    print(f"  [FAILED] {str(e)[:80]}")

print("\n[5/5] Downloading Know-SQL...")
try:
    know_sql = load_dataset("knowrohit07/know_sql", split="validation")

    data = []
    for ex in know_sql:
        data.append(
            {
                "dataset": "know-sql",
                "question": ex.get("question", ""),
                "sql": ex.get("answer", ""),
                "db_id": "",
                "context": ex.get("context", ""),
            }
        )

    count = save_to_jsonl(data, "nl2sql_data/train/know_sql.jsonl")
    print(f"  [OK] Know-SQL: {count:,} examples")
    datasets_info.append(("Know-SQL", count, "train"))
    total_train += count
except Exception as e:
    print(f"  [FAILED] {str(e)[:80]}")


# ========================================
# EVALUATION DATASETS
# ========================================

print("\n[1/1] Downloading Spider Dev Set...")
try:
    spider_dev = load_dataset("xlangai/spider", split="validation")

    data = []
    for ex in spider_dev:
        data.append(
            {
                "dataset": "spider",
                "question": ex["question"],
                "sql": ex["query"],
                "db_id": ex.get("db_id", ""),
                "context": f"Database: {ex.get('db_id', '')}",
            }
        )

    count = save_to_jsonl(data, "nl2sql_data/eval/spider_dev.jsonl")
    print(f"  [OK] Spider Dev: {count:,} examples")
    datasets_info.append(("Spider Dev", count, "eval"))
    total_eval += count
except Exception as e:
    print(f"  [FAILED] {str(e)[:80]}")


# ========================================
# CREATE COMBINED TRAINING FILE
# ========================================

print("\nCreating combined training file...")
combined_path = "nl2sql_data/all_train.jsonl"
with open(combined_path, "w") as outf:
    for fname in os.listdir("nl2sql_data/train"):
        if fname.endswith(".jsonl"):
            fpath = os.path.join("nl2sql_data/train", fname)
            with open(fpath) as inf:
                outf.write(inf.read())

print(f"  [OK] Combined: {combined_path}")


# ========================================
# SUMMARY
# ========================================

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("\nTraining datasets:")
for name, count, split in datasets_info:
    if split == "train":
        print(f"  {name}: {count:,} examples")
print(f"  TOTAL: {total_train:,} examples")

print("\nEvaluation datasets:")
for name, count, split in datasets_info:
    if split == "eval":
        print(f"  {name}: {count:,} examples")
print(f"  TOTAL: {total_eval:,} examples")

print("\nFiles created:")
print("  nl2sql_data/train/*.jsonl")
print("  nl2sql_data/all_train.jsonl")
print("  nl2sql_data/eval/spider_dev.jsonl")
