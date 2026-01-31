# Data Setup Guide

This document explains what data files are needed and how to set them up for the NL2SQL project.

## Quick Summary

**For most users - you only need the HuggingFace dataset:**
```bash
# No setup needed! Just run training/evaluation directly
# Training loads from HuggingFace automatically
# Baseline evaluation loads from HuggingFace automatically
```

**For data preparation or offline use:**
- Download raw datasets → Clean and deduplicate → Use local files

---

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  OPTION 1: Direct from HuggingFace (RECOMMENDED)               │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Training   │───→│ HuggingFace  │───→│   Load       │        │
│  │ Scripts   │    │ Repository   │    │  & Train     │        │
│  └────────────┘    └──────────────┘    └──────────────┘        │
│                                            (No setup needed)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  OPTION 2: Local Data Processing (For data preparation)        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ Download │──→│  Prepare │───→│ Push to │───→│  Local  │   │
│  │ Scripts  │   │  Scripts │   │    HF   │   │   Use   │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│       │              │                             │            │
│       ▼              ▼                             ▼            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Local Data Files (nl2sql_data/)            │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Files Needed (By Use Case)

### Use Case 1: Train and Evaluate Models (Most Common)

**Required Data Files:**
- **None!** All data loads from HuggingFace automatically

**How it works:**
- Training script (`train_unsloth_complete.py`) loads from: `AsadIsmail/nl2sql-deduplicated`
- Evaluation script (`baseline.py`) loads eval set from: `AsadIsmail/nl2sql-deduplicated`

**What you need:**
- Internet connection (first time to cache data)
- HuggingFace account (optional, for public datasets)

---

### Use Case 2: Data Preparation and Reproducibility

**Required Data Files:**

#### Step 1: Download Raw Datasets
**Script:** `src/nl2sql/data/download_all_datasets.py`

**What it downloads:**
- Spider training text (7,000 examples) from `xlangai/spider`
- SQaLe dataset (517,676 examples) from `trl-lab/SQaLe-text-to-SQL-dataset`
- Gretel synthetic (100,000 examples) from `gretelai/synthetic_text_to_sql`
- SQL-Context (78,577 examples) from `b-mc2/sql-create-context`
- Know-SQL (49,456 examples) from `knowrohit07/know_sql`
- Spider dev text (1,034 examples) from `xlangai/spider`

**Output files created:**
```
nl2sql_data/
├── train/
│   ├── spider_train.jsonl          (7,000 examples)
│   ├── sqale.jsonl                 (517,676 examples)
│   ├── gretel_train.jsonl          (100,000 examples)
│   ├── sql_context_train.jsonl     (78,577 examples)
│   └── know_sql.jsonl              (49,456 examples)
├── eval/
│   └── spider_dev.jsonl            (1,034 examples)
└── all_train.jsonl                 (752K combined)
```

**Dependencies:**
- Internet connection
- Python packages: `datasets`
- HuggingFace access

#### Step 2: Prepare and Clean Datasets
**Script:** `src/nl2sql/data/prepare_unsloth_data.py`

**Required input files:**
- All files from Step 1 (in `nl2sql_data/train/` and `nl2sql_data/eval/`)
- **IMPORTANT:** `nl2sql_data/database/spider_data/tables.json` (for schema enrichment)

**Output files created:**
```
nl2sql_data/unsloth/
├── spider_clean.jsonl             (6,956 deduplicated examples)
├── sqale_clean.jsonl              (502,837 deduplicated examples)
├── gretel_clean.jsonl             (99,013 deduplicated examples)
├── sql_context_clean.jsonl        (74,209 deduplicated examples)
├── spider_dev_clean.jsonl         (1,034 eval examples)
├── unsloth_config.json            (Training weights configuration)
└── preparation_stats.json         (Deduplication statistics)
```

**Dependencies:**
- Input: Raw datasets from Step 1
- Python packages: `sqlparse`, `tqdm`
- **IMPORTANT:** Spider database schemas (see "Spider Database Files" below)

#### Step 3: Push to HuggingFace (Optional)
**Script:** `src/nl2sql/data/push_to_hf.py`

**Required input files:**
- All files from `nl2sql_data/unsloth/` directory

**Output:**
- Uploads to HuggingFace repository: `AsadIsmail/nl2sql-deduplicated`

**Dependencies:**
- HuggingFace account with write access
- Python packages: `huggingface_hub`, `tqdm`

---

### Use Case 3: Evaluation with SQL Execution

**Required Data Files:**

#### For Text Generation Only (No SQL Execution)
- **None needed** - loads from HuggingFace automatically

#### For SQL Execution Validation
**Script:** `src/nl2sql/eval/baseline.py` (with `--execute-sql` flag)

**Required files:**
```
database/spider_data/
├── database/
│   ├── {db_id}/
│   │   └── {db_id}.sqlite        (SQLite database file)
│   └── ... (166 databases total)
└── tables.json                     (Schema definitions)
```

**Why these are needed:**
- `tables.json` - Schema information for context
- `.sqlite` files - Actual databases to execute generated SQL

**How to get Spider database files:**
```bash
# Option 1: Download from Spider GitHub repo
git clone https://github.com/taoyds/spider.git
# Copy database/ directory to nl2sql_data/database/spider_data/

# Option 2: Download from official Spider dataset
# Link: https://yale-lily.github.io/spider/
```

---

## Spider Database Files Details

The Spider dataset consists of two parts:

### Part 1: Text Data (Questions & SQL)
- **Source:** HuggingFace `xlangai/spider`
- **Contains:** Natural language questions and SQL queries
- **Used by:** All training and evaluation scripts
- **Downloaded by:** `download_all_datasets.py` or `load_dataset()`

### Part 2: Database Files (SQLite & Schemas)
- **Source:** Spider GitHub repository or official website
- **Contains:** Actual SQLite databases and schema definitions
- **Files needed:**
  - `database/spider_data/database/{db_id}/{db_id}.sqlite` (166 databases)
  - `database/spider_data/tables.json` (schema definitions)
- **Used by:**
  - `prepare_unsloth_data.py` (for schema enrichment)
  - `baseline.py` (for SQL execution validation)

**Do you need the database files?**
- For training only: **NO** (text data is sufficient)
- For SQL execution validation: **YES** (need `.sqlite` files)
- For schema enrichment during preparation: **YES** (need `tables.json`)

---

## Quick Start by Use Case

### Use Case A: I just want to train a model
```bash
# No data setup needed!
# Just run training directly
python src/nl2sql/train/train_unsloth_complete.py
```

### Use Case B: I want to evaluate a model
```bash
# No data setup needed if just generating SQL
python src/nl2sql/eval/baseline.py

# For SQL execution validation, need Spider databases:
# 1. Download Spider database files
# 2. Place in database/spider_data/
# 3. Run evaluation
python src/nl2sql/eval/baseline.py --execute-sql
```

### Use Case C: I want to reproduce the dataset preparation
```bash
# 1. Download raw datasets (text data only)
python src/nl2sql/data/download_all_datasets.py

# 2. Get Spider database files for schema enrichment
# Download from: https://github.com/taoyds/spider
# Extract to: database/spider_data/

# 3. Prepare and deduplicate
python src/nl2sql/data/prepare_unsloth_data.py

# 4. (Optional) Push to HuggingFace
python src/nl2sql/data/push_to_hf.py
```

### Use Case D: I want to create my own dataset
```bash
# Follow Use Case C steps 1-3
# Modify scripts as needed for your data
# Your cleaned data will be in nl2sql_data/unsloth/
```

---

## Directory Structure

```
nl2sql_data/
├── train/                           # Raw downloaded datasets
│   ├── spider_train.jsonl          # From Step 1
│   ├── sqale.jsonl
│   ├── gretel_train.jsonl
│   ├── sql_context_train.jsonl
│   └── know_sql.jsonl
├── eval/
│   └── spider_dev.jsonl            # From Step 1
├── all_train.jsonl                 # Combined raw data
├── unsloth/                         # Cleaned datasets
│   ├── spider_clean.jsonl          # From Step 2
│   ├── sqale_clean.jsonl
│   ├── gretel_clean.jsonl
│   ├── sql_context_clean.jsonl
│   ├── spider_dev_clean.jsonl
│   ├── unsloth_config.json         # Training config
│   └── preparation_stats.json      # Statistics
└── database/
    └── spider_data/                # From Spider repo (optional)
        ├── database/               # 166 SQLite databases
        │   ├── concert_singer/concert_singer.sqlite
        │   ├── department_management/department_management.sqlite
        │   └── ...
        └── tables.json             # Schema definitions
```

---

## File Dependencies

### By Script

#### download_all_datasets.py
- **Input:** None (downloads from HuggingFace)
- **Output:** `nl2sql_data/train/*.jsonl`, `nl2sql_data/eval/spider_dev.jsonl`
- **Dependencies:** Internet, `datasets` package

#### prepare_unsloth_data.py
- **Input:** `nl2sql_data/train/*.jsonl`, `nl2sql_data/eval/spider_dev.jsonl`
- **Additional input:** `database/spider_data/tables.json` (for schema enrichment)
- **Output:** `nl2sql_data/unsloth/*_clean.jsonl`, `unsloth_config.json`, `preparation_stats.json`
- **Dependencies:** Raw datasets from Step 1, `sqlparse`, `tqdm`

#### push_to_hf.py
- **Input:** All files in `nl2sql_data/unsloth/`
- **Output:** HuggingFace repository
- **Dependencies:** HuggingFace account, `huggingface_hub`, `tqdm`

#### train_unsloth_complete.py
- **Input:** None (loads from HuggingFace) OR `nl2sql_data/unsloth/*_clean.jsonl`
- **Output:** Trained model checkpoints
- **Dependencies:** `unsloth`, `trl`, `transformers`

#### baseline.py
- **Input:** None (loads from HuggingFace) OR `nl2sql_data/unsloth/spider_dev_clean.jsonl`
- **Additional for SQL execution:** `database/spider_data/database/{db_id}/{db_id}.sqlite`
- **Output:** Evaluation results
- **Dependencies:** `datasets`, `vllm` (for inference)

---

## Common Scenarios

### Scenario 1: New user, wants to train
```bash
# Minimal setup - just install and run
uv sync
source .venv/bin/activate
python src/nl2sql/train/train_unsloth_complete.py
# Data loads from HuggingFace automatically
```

### Scenario 2: Researcher, wants to understand the data
```bash
# Download and explore raw data
python src/nl2sql/data/download_all_datasets.py
head -20 nl2sql_data/all_train.jsonl | python -m json.tool

# Prepare and inspect statistics
python src/nl2sql/data/prepare_unsloth_data.py
cat nl2sql_data/unsloth/preparation_stats.json | python -m json.tool
```

### Scenario 3: Developer, wants to modify dataset
```bash
# Download raw data
python src/nl2sql/data/download_all_datasets.py

# Modify data processing logic in prepare_unsloth_data.py

# Re-run preparation
python src/nl2sql/data/prepare_unsloth_data.py

# Test with local files before pushing
python src/nl2sql/train/train_unsloth_complete.py --local-data
```

### Scenario 4: Offline environment
```bash
# On online machine:
python src/nl2sql/data/download_all_datasets.py
python src/nl2sql/data/prepare_unsloth_data.py
# Copy nl2sql_data/unsloth/ to offline machine

# On offline machine:
python src/nl2sql/train/train_unsloth_complete.py --data-dir nl2sql_data/unsloth/
```

---

## Troubleshooting

### Problem: "File not found: database/spider_data/tables.json"
**Solution:** You need to download the Spider database files separately. See "Spider Database Files" section above.

### Problem: "No such file or directory: nl2sql_data/train/"
**Solution:** Run `download_all_datasets.py` first to create the raw data files.

### Problem: "HuggingFace connection error"
**Solution:** Check your internet connection. The scripts will fall back to local files if available.

### Problem: "SQLite database not found"
**Solution:** Download Spider database files from GitHub and place in `database/spider_data/database/`

---

## Summary

| Use Case | Required Data | Setup Needed |
|----------|--------------|--------------|
| Training (online) | None | Just run script |
| Training (offline) | `nl2sql_data/unsloth/*_clean.jsonl` | Run Steps 1-2 first |
| Evaluation (text only) | None | Just run script |
| Evaluation (SQL execution) | `database/spider_data/database/*.sqlite` | Download Spider DB files |
| Data preparation | Raw datasets from HF | Run Step 1 |
| Data cleaning | Raw datasets + `tables.json` | Run Steps 1-2 |
| Push to HuggingFace | `nl2sql_data/unsloth/*` | Run Steps 1-2 first |

---

## Quick Reference

**Scripts that require NO data files:**
- `train_unsloth_complete.py` (loads from HF)
- `baseline.py` (loads from HF for text generation)

**Scripts that require raw data:**
- `prepare_unsloth_data.py` → needs `nl2sql_data/train/` and `nl2sql_data/eval/`

**Scripts that require cleaned data:**
- `train_unsloth_complete.py` (local mode)
- `push_to_hf.py`

**Scripts that require Spider databases:**
- `prepare_unsloth_data.py` → needs `tables.json`
- `baseline.py` (with `--execute-sql`) → needs `.sqlite` files
