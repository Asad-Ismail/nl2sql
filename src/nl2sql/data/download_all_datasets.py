"""
Optimized NL2SQL Dataset Downloader
Downloads curated Text-to-SQL datasets for maximum diversity and SQL consistency.

Strategy:
1. Use FULL SQaLe dataset (517K) - grounded in 22,989 real schemas
2. Add high-quality complementary datasets (Spider, Gretel, SQL-Context, Know-SQL)
3. Remove DuckDB (different SQL dialect) and Clinton (potential redundancy)
4. Total: ~751K examples with consistent SQL flavor
5. Evaluate on: Spider dev set (standard benchmark)

This approach maximizes schema diversity while maintaining SQL consistency.
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


print("="*70)
print("Optimized NL2SQL Dataset Downloader")
print("Strategy: Maximum diversity + SQL consistency")
print("="*70)

datasets_info = []
total_train = 0
total_eval = 0

# ========================================
# TRAINING DATASETS
# ========================================

print("\n📥 DOWNLOADING OPTIMIZED TRAINING DATASETS\n")

# ----------------------------
# 1. Spider TRAIN split (Benchmark Quality)
# ----------------------------
print("[1/5] Spider Training Set (Benchmark Standard)...")
try:
    spider_train = load_dataset("xlangai/spider", split="train")
    
    data = []
    for ex in spider_train:
        data.append({
            "dataset": "spider",
            "question": ex["question"],
            "sql": ex["query"],
            "db_id": ex.get("db_id", ""),
            "context": f"Database: {ex.get('db_id', '')}"
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/spider_train.jsonl")
    print(f"  ✓ Spider Train: {count:,} examples (complex multi-table queries)")
    datasets_info.append(("Spider Train", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 2. SQaLe - FULL DATASET (517K examples, 22,989 real schemas)
# ----------------------------
print("\n[2/5] SQaLe Full Dataset (Real Schema Grounding)...")
try:
    sqale = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
    
    data = []
    print(f"  Processing FULL dataset: {len(sqale):,} examples...")
    
    for ex in sqale:
        question = ex.get("question", "")
        sql = ex.get("query", ex.get("sql", ""))
        schema = ex.get("schema", "")
        
        if question and sql:
            data.append({
                "dataset": "sqale",
                "question": question,
                "sql": sql,
                "db_id": "",
                "context": schema[:1000] if schema else ""  # Truncate very long schemas
            })
    
    count = save_to_jsonl(data, "nl2sql_data/train/sqale.jsonl")
    print(f"  ✓ SQaLe: {count:,} examples (22,989 real schemas)")
    datasets_info.append(("SQaLe", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 3. Gretel Synthetic (High Quality)
# ----------------------------
print("\n[3/5] Gretel Synthetic Text-to-SQL...")
try:
    gretel = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    
    data = []
    for ex in gretel:
        data.append({
            "dataset": "gretel-synthetic",
            "question": ex["sql_prompt"],
            "sql": ex["sql"],
            "db_id": "",
            "context": ex.get("sql_context", "")
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/gretel_train.jsonl")
    print(f"  ✓ Gretel Synthetic: {count:,} examples (diverse synthetic)")
    datasets_info.append(("Gretel Synthetic", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 4. SQL-Create-Context (Schema Aware)
# ----------------------------
print("\n[4/5] SQL-Create-Context...")
try:
    sql_ctx = load_dataset("b-mc2/sql-create-context", split="train")
    
    data = []
    for ex in sql_ctx:
        data.append({
            "dataset": "sql-context",
            "question": ex["question"],
            "sql": ex["answer"],
            "db_id": "",
            "context": ex.get("context", "")
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/sql_context_train.jsonl")
    print(f"  ✓ SQL-Context: {count:,} examples (with schema context)")
    datasets_info.append(("SQL-Context", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 5. Know-SQL (Educational Breadth)
# ----------------------------
print("\n[5/5] Know-SQL...")
try:
    know_sql = load_dataset("knowrohit07/know_sql", split="validation")
    
    data = []
    for ex in know_sql:
        data.append({
            "dataset": "know-sql",
            "question": ex.get("question", ex.get("input", "")),
            "sql": ex.get("query", ex.get("output", "")),
            "db_id": "",
            "context": ex.get("context", "")
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/know_sql.jsonl")
    print(f"  ✓ Know-SQL: {count:,} examples (educational variety)")
    datasets_info.append(("Know-SQL", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")


# ========================================
# EVALUATION DATASETS
# ========================================

print("\n\n📊 DOWNLOADING EVALUATION DATASETS\n")

# ----------------------------
# Spider DEV set (standard benchmark)
# ----------------------------
print("[1/1] Spider Dev Set (for evaluation)...")
try:
    spider_dev = load_dataset("xlangai/spider", split="validation")
    
    data = []
    for ex in spider_dev:
        data.append({
            "dataset": "spider",
            "question": ex["question"],
            "sql": ex["query"],
            "db_id": ex.get("db_id", ""),
            "context": f"Database: {ex.get('db_id', '')}"
        })
    
    count = save_to_jsonl(data, "nl2sql_data/eval/spider_dev.jsonl")
    print(f"  ✓ Spider Dev: {count:,} examples")
    datasets_info.append(("Spider Dev", count, "eval"))
    total_eval += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")


# ========================================
# CREATE COMBINED TRAINING FILE
# ========================================

print("\n\n🔗 Creating combined training file...")
combined_path = "nl2sql_data/all_train.jsonl"
with open(combined_path, "w") as outf:
    for fname in os.listdir("nl2sql_data/train"):
        if fname.endswith(".jsonl"):
            fpath = os.path.join("nl2sql_data/train", fname)
            with open(fpath) as inf:
                outf.write(inf.read())

print(f"  ✓ Combined training data: {combined_path}")


# ========================================
# SUMMARY
# ========================================

print("\n" + "="*70)
print("✅ DOWNLOAD COMPLETE")
print("="*70)

print("\n📊 Training Datasets (Optimized Selection):")
for name, count, split in datasets_info:
    if split == "train":
        print(f"  • {name:.<40} {count:>8,} examples")
print(f"  {'TOTAL TRAINING':.>40} {total_train:>8,} examples")

print("\n📈 Evaluation Datasets:")
for name, count, split in datasets_info:
    if split == "eval":
        print(f"  • {name:.<40} {count:>8,} examples")
print(f"  {'TOTAL EVALUATION':.>40} {total_eval:>8,} examples")

print("\n📁 Files Created:")
print(f"  • Training data: nl2sql_data/train/*.jsonl")
print(f"  • Combined train: nl2sql_data/all_train.jsonl")
print(f"  • Evaluation: nl2sql_data/eval/spider_dev.jsonl")

print("\n✅ Dataset Optimization Applied:")
print("  ✓ Removed DuckDB (different SQL dialect)")
print("  ✓ Removed Clinton (potential redundancy)")
print("  ✓ Using FULL SQaLe (517K examples, 22,989 real schemas)")
print("  ✓ Consistent SQL flavor (SQLite/PostgreSQL/MySQL standard)")
print("  ✓ Maximum schema diversity")

print("\n🚀 Next Steps:")
print("  1. Explore data: head nl2sql_data/all_train.jsonl")
print("  2. Train with LoRA on all_train.jsonl")
print("  3. Evaluate on spider_dev.jsonl")
print("  4. Compare results with Spider leaderboard!")

print("\n💡 Recommended Training Setup:")
print("  • Model: CodeLlama-7b, Mistral-7b, or DeepSeek-Coder")
print("  • Method: LoRA (r=16, alpha=32)")
print("  • Batch size: 4-8 depending on GPU")
print("  • Epochs: 2-3 (large dataset)")
print("  • Evaluation metric: Exact Match on Spider dev")
print("  • Expected performance: High generalization due to schema diversity")
