# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NL2SQL is a Text-to-SQL research project comparing LLM-based approaches: zero-shot, few-shot, self-correction, DSPy optimization, TextGrad optimization, and LoRA fine-tuning. Uses 752K+ curated training examples from 5 datasets (Spider, SQaLe, Gretel, SQL-Context, Know-SQL).

**HuggingFace Dataset:** `AsadIsmail/nl2sql-deduplicated`

## Common Commands

```bash
# Setup - ALWAYS install packages and activate venv first
# Use uv to create virtual environment and install dependencies
uv sync
uv pip install -e .  # Install package in editable mode for proper imports

# ALWAYS activate the virtual environment before running any tests or commands
source .venv/bin/activate

# Start vLLM server (required for baseline evaluation)
vllm serve TheBloke/CodeLlama-7B-Instruct-AWQ \
    --host 0.0.0.0 --port 8000 --quantization awq \
    --gpu-memory-utilization 0.8 --max-model-len 8048 \
    --chat-template models/codellama_chat.jinja

# Run baseline evaluation
python src/nl2sql/eval/baseline.py --num-samples 100  # Quick test
python src/nl2sql/eval/baseline.py                    # Full evaluation (1,034 examples)

# Run fine-tuned model evaluation
python src/nl2sql/eval/sft.py --model-path models/your-model

# Training with LoRA
python src/nl2sql/train/train_unsloth_complete.py

# DSPy optimization
python src/nl2sql/optim/dspy_optim.py

# TextGrad optimization
python src/nl2sql/optim/textgrad_optim.py

# Download datasets (optional - training uses HuggingFace directly)
python src/nl2sql/data/download_all_datasets.py

# Dev tools
black src/ --line-length 100
ruff check src/
mypy src/
```

## Architecture

### Data Flow
1. **Download** - `src/nl2sql/data/download_all_datasets.py` fetches 5 datasets (752K examples)
2. **Training** - Unsloth + LoRA fine-tuning on CodeLlama-7b (or Mistral-7b, DeepSeek-Coder)
3. **Evaluation** - Test against Spider dev set (1,034 examples) in `database/spider_data/`

### Key Modules
- `src/nl2sql/eval/baseline.py` - Zero-shot, few-shot, self-correction evaluation
- `src/nl2sql/eval/sft.py` - Fine-tuned model evaluation
- `src/nl2sql/train/train_unsloth_complete.py` - LoRA training with Unsloth
- `src/nl2sql/optim/dspy_optim.py` - DSPy few-shot optimization
- `src/nl2sql/optim/textgrad_optim.py` - TextGrad prompt optimization
- `src/nl2sql/utils/util.py` - Shared utilities (SQL execution, metrics, result comparison)

### Key Utilities (src/nl2sql/utils/util.py)
- `load_schemas()` - Load database schemas from `database/spider_data/tables.json`
- `execute_sql(sql, db_path)` - Execute SQL on SQLite, returns (success, error, results)
- `compare_results(gen_results, gold_results)` - Row/column-order-independent comparison
- `extract_sql_from_text(text)` - Extract SQL from LLM output (handles markdown, delimiters)
- `calculate_metrics(results)` - Compute valid SQL %, match rates
- `generate_markdown_report()` - Generate evaluation reports

### Training Configuration
- **Model:** CodeLlama-7b, Mistral-7b, or DeepSeek-Coder (7B models)
- **Method:** LoRA (r=16, alpha=32) targeting `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Prompt format:** Alpaca-style with `### Instruction:` and `### Response:` delimiters

### File Locations
- Spider databases: `database/spider_data/database/{db_id}/{db_id}.sqlite`
- Results output: `results/baseline/` or `results/dspy_optimized/`
- Model outputs: `models/`

## Critical Conventions

### Environment Setup (CRITICAL)
- **Always use `uv`** for package installation and virtual environment creation
- **Always activate venv** before running tests, evaluation, or any commands: `source .venv/bin/activate`
- Standard workflow: `uv sync && source .venv/bin/activate`

### SQL Dialect Consistency
- **Standard SQL only** (SQLite/PostgreSQL/MySQL) - DuckDB excluded for PRAGMA statements
- JSONL format: `{"dataset": str, "question": str, "sql": str, "db_id": str, "context": str}`

### Code Style
- Type hints mandatory
- PEP 257 docstrings with Parameters/Returns sections
- Line length: 100 chars (black/ruff configured in pyproject.toml)
- No emojis in code or responses

### Dataset Field Handling
- Field names vary across datasets: `query` vs `sql`, `sql_prompt` vs `question`
- Always use `.get()` with fallbacks: `ex.get("query", ex.get("sql", ""))`

### SQL Extraction
- Generated text may include markdown/comments
- `extract_sql_from_text()` uses "SELECT" anchor + delimiter detection
- Schema context may need truncation for large schemas (SQaLe: truncate to 1000 chars)

### GPU Requirements
- 16GB+ VRAM for 7B model inference/training
- 4-bit quantization enabled by default
- Full 752K dataset needs gradient accumulation (set to 4)

## Gitignored Directories
- `nl2sql_data/` - Downloaded datasets (~900MB)
- `models/` - Trained models (except tokenizer configs)
- `results/` - Evaluation outputs
- `database/` - Spider SQLite databases
- `wandb/` - Training logs
