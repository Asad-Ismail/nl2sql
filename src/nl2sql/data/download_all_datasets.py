"""
Comprehensive NL2SQL Dataset Downloader
Downloads all major Text-to-SQL datasets for training and evaluates on Spider dev/test.

Strategy:
1. Train on: Spider train + WikiSQL + Gretel + Clinton + other synthetic data
2. Evaluate on: Spider dev set (standard benchmark)
3. Fine-tune with LoRA for efficiency

This multi-dataset approach has been shown to improve generalization.
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

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
print("NL2SQL Comprehensive Dataset Downloader")
print("Training Strategy: Multi-dataset training -> Spider evaluation")
print("="*70)

datasets_info = []
total_train = 0
total_eval = 0

# ========================================
# TRAINING DATASETS
# ========================================

print("\n📥 DOWNLOADING TRAINING DATASETS\n")

# ----------------------------
# 1. Spider TRAIN split
# ----------------------------
print("[1/6] Spider Training Set...")
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
    print(f"  ✓ Spider Train: {count:,} examples")
    datasets_info.append(("Spider Train", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 2. WikiSQL (large single-table dataset)
# ----------------------------
print("\n[2/6] WikiSQL Training Set...")
try:
    wikisql = load_dataset("wikisql", split="train", trust_remote_code=True)
    
    data = []
    # Sample WikiSQL to avoid imbalance (it's 80K+ examples)
    sample_size = min(20000, len(wikisql))
    print(f"  Sampling {sample_size:,} from {len(wikisql):,} examples...")
    
    for i, ex in enumerate(wikisql):
        if i >= sample_size:
            break
            
        # Reconstruct SQL
        agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        col_idx = ex["sql"]["sel"]
        col_name = ex["table"]["header"][col_idx]
        agg = ex["sql"]["agg"]
        
        sql_parts = []
        if agg > 0:
            sql_parts.append(f"SELECT {agg_ops[agg]}({col_name})")
        else:
            sql_parts.append(f"SELECT {col_name}")
        
        sql_parts.append(f"FROM {ex['table']['id']}")
        
        if ex["sql"]["conds"]:
            where_parts = []
            ops = ['=', '>', '<', 'OP']
            for cond_col, cond_op, cond_val in ex["sql"]["conds"]:
                cond_col_name = ex["table"]["header"][cond_col]
                where_parts.append(f"{cond_col_name} {ops[cond_op]} '{cond_val}'")
            sql_parts.append("WHERE " + " AND ".join(where_parts))
        
        data.append({
            "dataset": "wikisql",
            "question": ex["question"],
            "sql": " ".join(sql_parts),
            "db_id": ex["table"]["id"],
            "context": f"Table: {ex['table']['id']}"
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/wikisql_train.jsonl")
    print(f"  ✓ WikiSQL: {count:,} examples")
    datasets_info.append(("WikiSQL", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 3. Gretel Synthetic
# ----------------------------
print("\n[3/6] Gretel Synthetic Text-to-SQL...")
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
    print(f"  ✓ Gretel Synthetic: {count:,} examples")
    datasets_info.append(("Gretel Synthetic", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 4. Clinton Text-to-SQL v1
# ----------------------------
print("\n[4/6] Clinton Text-to-SQL...")
try:
    clinton = load_dataset("Clinton/Text-to-sql-v1", split="train")
    
    data = []
    for ex in clinton:
        data.append({
            "dataset": "clinton",
            "question": ex["instruction"],
            "sql": ex["response"],
            "db_id": "",
            "context": ex.get("input", "")
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/clinton_train.jsonl")
    print(f"  ✓ Clinton: {count:,} examples")
    datasets_info.append(("Clinton", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 5. SQL-Create-Context
# ----------------------------
print("\n[5/6] SQL-Create-Context...")
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
    print(f"  ✓ SQL-Context: {count:,} examples")
    datasets_info.append(("SQL-Context", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 6. Additional free datasets
# ----------------------------
print("\n[6/10] Know-SQL...")
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
    print(f"  ✓ Know-SQL: {count:,} examples")
    datasets_info.append(("Know-SQL", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 7. SEDE (Stack Exchange Data Explorer queries)
# ----------------------------
print("\n[7/10] SEDE (Stack Exchange SQL)...")
try:
    sede = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    
    # Filter for SQL-related entries
    data = []
    for ex in sede:
        if "sql" in str(ex).lower() or "query" in str(ex).lower():
            if isinstance(ex.get("query"), str):
                data.append({
                    "dataset": "sede",
                    "question": ex.get("question", ""),
                    "sql": ex.get("query", ""),
                    "db_id": "",
                    "context": ""
                })
    
    if data:
        count = save_to_jsonl(data, "nl2sql_data/train/sede.jsonl")
        print(f"  ✓ SEDE: {count:,} examples")
        datasets_info.append(("SEDE", count, "train"))
        total_train += count
    else:
        print(f"  ⊘ No SQL examples found")
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 8. Spider Realistic (with typos/ambiguity)
# ----------------------------
print("\n[8/10] Spider-Realistic...")
try:
    spider_real = load_dataset("xlangai/spider-realistic", split="train")
    
    data = []
    for ex in spider_real:
        data.append({
            "dataset": "spider-realistic",
            "question": ex["question"],
            "sql": ex["query"],
            "db_id": ex.get("db_id", ""),
            "context": f"Database: {ex.get('db_id', '')}"
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/spider_realistic.jsonl")
    print(f"  ✓ Spider-Realistic: {count:,} examples")
    datasets_info.append(("Spider-Realistic", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 9. CoSQL (conversational SQL)
# ----------------------------
print("\n[9/10] CoSQL (conversational)...")
try:
    cosql = load_dataset("cosql", split="train")
    
    data = []
    for ex in cosql:
        # CoSQL has conversation history - flatten it
        if isinstance(ex.get("final_question"), str):
            data.append({
                "dataset": "cosql",
                "question": ex["final_question"],
                "sql": ex.get("query", ""),
                "db_id": ex.get("db_id", ""),
                "context": f"Database: {ex.get('db_id', '')}"
            })
    
    count = save_to_jsonl(data, "nl2sql_data/train/cosql.jsonl")
    print(f"  ✓ CoSQL: {count:,} examples")
    datasets_info.append(("CoSQL", count, "train"))
    total_train += count
except Exception as e:
    print(f"  ✗ Failed: {str(e)[:80]}")

# ----------------------------
# 10. Squall (compositional)
# ----------------------------
print("\n[10/10] Squall...")
try:
    squall = load_dataset("squall", split="train")
    
    data = []
    for ex in squall:
        data.append({
            "dataset": "squall",
            "question": ex.get("nl", ex.get("question", "")),
            "sql": ex.get("sql", ex.get("query", "")),
            "db_id": "",
            "context": ""
        })
    
    count = save_to_jsonl(data, "nl2sql_data/train/squall.jsonl")
    print(f"  ✓ Squall: {count:,} examples")
    datasets_info.append(("Squall", count, "train"))
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

print("\n📊 Training Datasets:")
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

print("\n🚀 Next Steps:")
print("  1. Explore data: head nl2sql_data/all_train.jsonl")
print("  2. Train with LoRA on all_train.jsonl")
print("  3. Evaluate on spider_dev.jsonl")
print("  4. Compare results with Spider leaderboard!")

print("\n💡 Recommended Training Setup:")
print("  • Model: CodeLlama-7b, Mistral-7b, or DeepSeek-Coder")
print("  • Method: LoRA (r=16, alpha=32)")
print("  • Batch size: 4-8 depending on GPU")
print("  • Epochs: 3-5")
print("  • Evaluation metric: Exact Match on Spider dev")
