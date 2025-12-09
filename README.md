# NL2SQL: Multi-Dataset Training Pipeline

Train Text-to-SQL models on 10+ free datasets (~400K examples) and evaluate against established baselines.

## Quick Start

### Option A: Use Pre-cleaned Dataset (Recommended)

```bash
# Load 683K deduplicated examples from HuggingFace
from datasets import load_dataset

dataset = load_dataset(
    "AsadIsmail/nl2sql-deduplicated", 
    data_files="*_clean.jsonl", 
    split="train"
)

print(f"Total examples: {len(dataset):,}")  # 683,015
print(dataset[0])  # View first example
```

**Features of pre-cleaned dataset:**
- 683K unique examples (input-only deduplication)
- Conflict resolution (same question → best SQL kept)
- SQL dialect validation (standard SQL only)
- Separated by source for weighted sampling
- See: https://huggingface.co/datasets/AsadIsmail/nl2sql-deduplicated

### Option B: Build Dataset from Scratch

```bash
# 1. Install dependencies
uv sync
source .venv/bin/activate

# 2. Download raw datasets
python src/nl2sql/data/download_all_datasets.py

# 3. Clean and deduplicate
python src/nl2sql/data/prepare_unsloth_data.py

# 4. (Optional) Push to your HuggingFace
python push_to_hf.py
```

### Run Baseline Evaluation

```bash

vllm serve TheBloke/CodeLlama-7B-Instruct-AWQ     --host 0.0.0.0     --port 8000     --quantization awq     --gpu-memory-utilization 0.4     --max-model-len 2048    --chat-template models/codellama_chat.jinja

vllm serve TheBloke/CodeLlama-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048


or if gpu mempry is not too low use codellama/CodeLlama-7b-hf

# Evaluate 3 baseline approaches on Spider dev set
python src/nl2sql/eval/baseline.py --num-samples 100

# Results saved to: results/baseline/
```

## Baseline Approaches

We evaluate 3 standard baselines:

1. **Zero-shot**: Single LLM call without examples
2. **Few-shot**: With 2 similar examples  
3. **Self-correction**: Generate → Execute → Fix errors (up to 3 attempts)

### Example Output
```
RESULTS SUMMARY
===========================================
ZERO_SHOT
  Valid SQL: 42/100 (42.0%)
  Avg time: 2.3s
  
FEW_SHOT
  Valid SQL: 58/100 (58.0%)
  Avg time: 3.1s
  
SELF_CORRECTION
  Valid SQL: 67/100 (67.0%)
  Avg time: 5.8s
  Avg attempts: 2.1
```

## What's Included

### 📊 10+ Training Datasets (~400K examples)
- Spider (~10K), WikiSQL (~80K), Gretel Synthetic (~100K)
- Clinton (~15K), SQL-Create-Context (~78K), Know-SQL
- Spider-Realistic, CoSQL, Squall, SEDE

### 🧪 Baseline Evaluation
Three standard baselines evaluated on Spider dev:
- **Zero-shot**: Single LLM call
- **Few-shot**: With example queries
- **Self-correction**: Iterative error fixing

### 📈 Expected Results
| Method | Valid SQL % | Notes |
|--------|-------------|-------|
| Zero-shot | ~40-50% | Weakest baseline |
| Few-shot | ~55-65% | Standard approach |
| Self-correction | ~65-75% | Best baseline |

## Usage

### Option 1: Quick Evaluation (Recommended to Start)

```bash
# Evaluate baselines on Spider dev set (100 samples)
python src/nl2sql/eval/baseline.py --num-samples 100

# Full evaluation (may take hours)
python src/nl2sql/eval/baseline.py
```

Results saved to `results/baseline/`:
- `zero_shot_results.jsonl`
- `few_shot_results.jsonl`
- `self_correction_results.jsonl`
- `evaluation_report.md`

### Option 2: Train Your Own Model

```bash
# 1. Download all training datasets
python src/nl2sql/data/download_all_datasets.py

# 2. (Optional) Generate synthetic variations
python src/nl2sql/data/synthetic_augmentation.py

# 3. Train with LoRA
python src/nl2sql/train/train_curriculum_lora.py
```

## Evaluation Details

### Spider Dataset
- **Dev set**: 1,034 examples for evaluation
- **Metric**: Valid SQL execution (query runs without errors)
- **Note**: We use Spider dev, not Spider 2.0 (requires registration)

### What Gets Measured
- **Valid SQL %**: Percentage of queries that execute without syntax/semantic errors
- **Inference Time**: Average time per query
- **Attempts**: For self-correction, how many tries needed

### Baseline Results (Expected)
Based on CodeLlama-7B:
- Zero-shot: 40-50% valid SQL
- Few-shot: 55-65% valid SQL  
- Self-correction: 65-75% valid SQL (best baseline)

## Advanced: Training Your Own Model

If you want to train on the full dataset:

```bash
# Train with multi-dataset + curriculum learning
python src/nl2sql/train/train_curriculum_lora.py

# Expected improvement: +5-10% over self-correction baseline
```

## Project Structure

```
nl2sql/
├── src/nl2sql/
│   ├── data/
│   │   ├── download_all_datasets.py       # Download 10+ raw datasets
│   │   ├── prepare_unsloth_data.py        # Clean, deduplicate, validate
│   │   └── synthetic_augmentation.py      # Generate synthetic data
│   ├── train/
│   │   └── train_curriculum_lora.py       # Curriculum training
│   └── eval/
│       └── baseline.py                    # Baseline evaluation
├── push_to_hf.py                          # Upload dataset to HuggingFace
├── run_pipeline.py                        # Full pipeline runner
└── pyproject.toml                         # Dependencies
```

## Configuration

Edit `train_curriculum_lora.py`:

```python
# Model
model_name = "codellama/CodeLlama-7b-hf"  # or Mistral, DeepSeek

# LoRA config
LoraConfig(
    r=16,              # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

# Training
epochs_per_stage = [2, 3, 3]
batch_size = 4
learning_rate = 2e-4
```

## Why This Approach?

### Problem with Standard Training
- Most work: Train on Spider only (~10K examples)
- Limited diversity, prone to overfitting
- Doesn't leverage free datasets

### Our Solution
1. **Multi-dataset**: Use ALL available free data (~400K)
2. **Augmentation**: Generate 3-5x more through valid transformations
3. **Curriculum**: Learn progressively (easy→hard)
4. **LoRA**: Efficient parameter-efficient fine-tuning

### Key Innovation
**Synthetic augmentation with validation** - Unlike other work that blindly augments, we:
- Validate all generated SQL (no syntax errors)
- Preserve semantic meaning (same query, different form)
- Increase diversity (robustness to paraphrasing, naming)

## Project Structure

```
nl2sql/
├── src/nl2sql/
│   ├── data/
│   │   ├── download_all_datasets.py       # Download Spider + 10 datasets
│   │   └── synthetic_augmentation.py      # Generate variations
│   ├── train/
│   │   └── train_curriculum_lora.py       # Training script
│   └── eval/
│       └── baseline.py                    # Baseline evaluation
├── results/
│   └── baseline/                          # Evaluation results
├── run_pipeline.py                        # Full pipeline runner
└── README.md
```

## Requirements

- Python >= 3.10
- GPU with 16GB+ VRAM (for inference on 7B models)
- ~5GB disk space for Spider + eval data
- ~20GB for full training datasets
- Dependencies in `pyproject.toml`

## FAQ

**Q: Do I need Spider 2.0?**  
A: No. We use Spider dev set (freely available). Spider 2.0 requires registration.

**Q: Can I run without GPU?**  
A: Evaluation works on CPU but is very slow. Training requires GPU.

**Q: What models are supported?**  
A: Any HuggingFace CausalLM model. Tested: CodeLlama-7B, Llama-2-7B, Mistral-7B

**Q: How long does evaluation take?**  
A: ~30 minutes for 100 samples, ~5 hours for full dev set (1034 examples)

## License

MIT
