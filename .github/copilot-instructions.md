# NL2SQL Project - AI Coding Agent Instructions

## Project Overview
Text-to-SQL training pipeline focused on multi-dataset training (752K examples) with LoRA fine-tuning. Uses curated datasets with consistent SQL dialects for maximum generalization without confusion.

## Architecture & Data Flow

### Dataset Pipeline (Priority: SQL Consistency)
1. **Download** (`src/nl2sql/data/download_all_datasets.py`): Curated 5 datasets, 752K examples
   - Spider (7K benchmark), SQaLe (517K real schemas), Gretel (100K synthetic), SQL-Context (78K), Know-SQL (49K)
   - **Critical**: Removed DuckDB (different SQL dialect), Clinton (redundancy) - see `DATASET_SUMMARY.md`
   - Output: `nl2sql_data/train/*.jsonl` + `nl2sql_data/all_train.jsonl` (899MB combined)
   - Evaluation: `nl2sql_data/eval/spider_dev.jsonl` (1,034 examples)

2. **Augmentation** (`src/nl2sql/data/synthetic_augmentation.py`): 3x multiplier via:
   - Question paraphrasing (same SQL, different wording)
   - Schema shuffling (change names, keep logic)
   - Query augmentation (add LIMIT/ORDER BY)
   - All variations validated with `SQLValidator.is_valid_sql()`

### Training Architecture
- **Model**: CodeLlama-7b, Mistral-7b, or DeepSeek-Coder (7B models)
- **Method**: LoRA (r=16, alpha=32) targeting attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`)
- **Curriculum**: 3-stage progressive difficulty (easy→medium→hard) via `DatasetConfig.difficulty`
- **Prompt format**: 
  ```
  ### Task: Convert natural language question to SQL query
  Database: {db_id}
  Database Schema: {schema}
  ### Question: {question}
  ### SQL: {sql}
  ```

### Evaluation Pipeline
- **Baseline methods** (`src/nl2sql/eval/baseline.py`): zero-shot, few-shot (2 examples), self-correction (3 attempts)
- **Metrics**: Valid SQL % (execution without errors), inference time, attempts needed
- **Expected baseline**: 40-50% zero-shot, 55-65% few-shot, 65-75% self-correction

## Critical Conventions

### Dataset Management
- **Never mix SQL dialects**: Project enforces standard SQL (SQLite/PostgreSQL/MySQL) - DuckDB excluded for PRAGMA statements
- **Schema diversity > size**: SQaLe's 22,989 real schemas preferred over raw example count
- **JSONL format**: `{"dataset": str, "question": str, "sql": str, "db_id": str, "context": str}`
- **Gitignored**: `nl2sql_data/` directory (899MB) never committed

### Training Patterns
- **Weighted sampling**: Use `DatasetConfig.weight` to balance dataset contributions (e.g., Spider 0.5, WikiSQL 0.3)
- **Curriculum stages**: Easy (WikiSQL single-table) → Medium (Spider multi-table) → Hard (real-world complex)
- **Validation during training**: Execute generated SQL to verify correctness, not just syntax
- **GPU requirements**: 16GB+ VRAM for 7B model inference/training

### Code Style (per `python.instructions.md`)
- Type hints mandatory: `def generate_sql(prompt: str, max_new_tokens: int = 256) -> str`
- Docstrings: PEP 257 format with Parameters/Returns sections
- Line length: 100 chars (set in `pyproject.toml` for black/ruff)
- Error handling: Explicit try-except with logged errors (see `baseline.py` example loop)

## Key Commands & Workflows

### Setup & Download
```bash
uv sync && source .venv/bin/activate
python src/nl2sql/data/download_all_datasets.py  # Downloads 752K examples, ~20min
```

### Evaluation (Recommended First Step)
```bash
python src/nl2sql/eval/baseline.py --num-samples 100  # Quick test (30min)
python src/nl2sql/eval/baseline.py                    # Full eval (5hrs)
# Results: results/baseline/{zero_shot,few_shot,self_correction}_results.jsonl
```

### Training
```bash
python run_pipeline.py --full          # Download + augment + train
python run_pipeline.py --train-only    # Skip download if data exists
# Model output: models/nl2sql-curriculum-lora/stage{1,2,3}_final/
```

### Testing
```bash
python test_baseline.py  # Unit test for SQL extraction/execution
```

## Integration Points

### HuggingFace Datasets
- Load via `load_dataset(repo, split="train")` - check for deprecated loading scripts (WikiSQL issue)
- Handle missing fields: `ex.get("query", ex.get("sql", ""))` - field names vary across datasets
- Cache location: `~/.cache/huggingface/datasets/` (cleared during optimization)

### SQLite Execution
- Spider databases: `nl2sql_data/spider/database/{db_id}/{db_id}.sqlite`
- Connection pattern: `sqlite3.connect(db_path)` with exception handling for validation
- Used for: baseline execution validation, training-time correctness checks

### Transformers/PEFT
- Model loading: `AutoModelForCausalLM.from_pretrained()` with `torch_dtype=torch.float16, device_map="auto"`
- LoRA: `get_peft_model(model, LoraConfig(...))` - always print trainable params
- Generation: `model.generate()` with `temperature=0.1, do_sample=False` for deterministic SQL

## Project-Specific Gotchas

1. **SQL extraction fragility**: Generated text may include markdown/comments - `_extract_sql()` uses "SELECT" anchor + delimiter detection
2. **Schema context truncation**: SQaLe schemas can exceed token limits - truncate to 1000 chars (`schema[:1000]`)
3. **Dataset field inconsistency**: `query` vs `sql`, `sql_prompt` vs `question` - always use `.get()` with fallbacks
4. **Curriculum requires sorting**: `DatasetConfig.difficulty` must be sorted before training stages
5. **Memory management**: Full 752K dataset at 512 tokens = ~48GB RAM - use gradient accumulation (set to 4)

## Expected Performance Targets
- Baseline self-correction: 65-75% valid SQL on Spider dev
- After curriculum training: +5-10% improvement (70-85% target)
- Inference speed: 2-3s per query (zero-shot), 5-8s (self-correction with 3 attempts)
- Training time: ~8-12 hours on single A100 for full curriculum (3 stages, 2-3 epochs each)

## When Making Changes

### Adding New Datasets
1. Check SQL dialect compatibility (standard SQL only, no DuckDB-style PRAGMA)
2. Verify field mapping: question/sql/context fields
3. Update `DATASET_SUMMARY.md` with dataset details
4. Add to `.gitignore` if large files

### Modifying Training
1. Validate prompt format matches existing pattern (### Task/Question/SQL markers)
2. Test with small dataset first (`max_samples=100` in `DatasetConfig`)
3. Monitor GPU memory with `torch.cuda.memory_allocated()`
4. Save checkpoints per stage (`save_strategy="epoch"`)

### Evaluation Changes
1. Always test on Spider dev (standard benchmark) for comparability
2. Report valid SQL % first (not accuracy - Spider 2.0 would need gold results comparison)
3. Log inference times for performance tracking
4. Save results to `results/baseline/` directory structure


Dont add emojis to the response. It looks unprofessional.