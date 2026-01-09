# NL2SQL: Text-to-SQL Methods Comparison

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-nl2sql--deduplicated-yellow)](https://huggingface.co/datasets/AsadIsmail/nl2sql-deduplicated)

**Compare and evaluate different LLM-based approaches for Text-to-SQL generation**, from simple prompting to advanced optimization and fine-tuning.

**Methods covered:**
- Zero-shot prompting
- Few-shot in-context learning
- Self-correction with execution feedback
- DSPy few-shot optimization (YAML configurable)
- TextGrad prompt optimization
- LoRA fine-tuning (parameter-efficient)

**Bonus:** 750K+ curated training examples from 5 datasets (Spider, SQaLe, Gretel, SQL-Context, Know-SQL) for reproducible experiments.

## Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/Asad-Ismail/nl2sql.git
cd nl2sql

# Install dependencies
uv sync
source .venv/bin/activate
```

## Baseline Evaluation

Evaluate 3 baseline approaches on Spider dev set (1,034 examples):

```bash
#  Start vLLM server (CodeLlama-7B with AWQ quantization)
## you can reduce context size if not optmizing for fewshot and corresponding gpu utilization e.g context size 2048 with gpu usage of 0.4
vllm serve TheBloke/CodeLlama-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8048 \
    --chat-template models/codellama_chat.jinja
    

#  Run evaluation (in another terminal)
python src/nl2sql/eval/baseline.py --num-samples 100  # Quick test
python src/nl2sql/eval/baseline.py                    # Full evaluation
```

Results saved to `results/baseline/` with detailed reports.

## DSPy Optimization

Optimize prompts using DSPy optimizers with YAML configuration:

```bash
# Run with default config (BootstrapFewShotWithRandomSearch)
python src/nl2sql/optim/dspy_optim.py --config src/nl2sql/optim/configs/default.yaml

# Run with MIPRO optimizer (Bayesian optimization)
python src/nl2sql/optim/dspy_optim.py --config src/nl2sql/optim/configs/mipro.yaml

# Run with KNN few-shot (selects similar examples per query)
python src/nl2sql/optim/dspy_optim.py --config src/nl2sql/optim/configs/knn_fewshot.yaml

# Run with COPRO (coordinate ascent for instructions)
python src/nl2sql/optim/dspy_optim.py --config src/nl2sql/optim/configs/copro.yaml

# Override config via CLI
python src/nl2sql/optim/dspy_optim.py --config src/nl2sql/optim/configs/default.yaml \
    --optimizer MIPRO --train_size 1000 --output_dir results/mipro_run
```

**Available optimizers:**
| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `LabeledFewShot` | Simple k random examples | Quick baseline |
| `BootstrapFewShot` | Teacher-generated demos | Small datasets |
| `BootstrapFewShotWithRandomSearch` | Random search over demos | General use |
| `KNNFewShot` | k-Nearest Neighbors per query | Diverse SQL patterns |
| `COPRO` | Coordinate ascent for instructions | Instruction tuning |
| `MIPRO` | Bayesian optimization | Best quality |

Results saved to `results/dspy_optimized/` with model and reports.

## TextGrad Optimization

Optimize system prompts using gradient-based feedback:

```bash
# Run TextGrad optimization (requires NVIDIA API key)
export NVIDIA_API_KEY=your_key
python src/nl2sql/optim/textgrad_optim.py --epochs 3 --batch_size 3

# Results saved to: results/textgrad_v3/
```

## Evaluation Methods

**Methods compared:**
1. **Zero-shot**: Direct question-to-SQL conversion (no examples)
2. **Few-shot**: With 2 similar examples as context
3. **Self-correction**: Iterative refinement with execution feedback (up to 3 attempts)
4. **DSPy Optimization**: Automated few-shot example selection (configurable optimizers)
5. **TextGrad Optimization**: Gradient-based system prompt optimization
6. **Fine-tuned (LoRA)**: Model fine-tuned on training data

All methods include complexity tracking (JOIN, GROUP BY, subqueries, etc.) and detailed reports.

## Results Comparison

Expected performance on Spider dev set (1,034 examples) with CodeLlama-7B:

| Method | Valid SQL % | Avg Time (s) | Notes |
|--------|-------------|--------------|-------|
| Zero-shot | 40-50% | 2-3 | Baseline |
| Few-shot | 55-65% | 3-4 | +2 examples |
| Self-correction | 65-75% | 5-8 | Up to 3 attempts |
| DSPy Optimized | 70-80% | 3-5 | Optimized examples |
| TextGrad Optimized | 70-80% | 3-5 | Optimized prompt |
| Fine-tuned (LoRA) | 75-85% | 2-3 | Full training |

*Note: Results vary based on model, prompt template, and evaluation settings.*

## Training

**Recommended:** Use pre-cleaned HuggingFace dataset for training:

```bash
# Train with LoRA fine-tuning (uses HuggingFace dataset automatically)
python src/nl2sql/train/train_unsloth_complete.py

# Evaluate fine-tuned model
python src/nl2sql/eval/sft.py --model-path models/your-model
```

**Optional:** Download and prepare datasets locally:

```bash
# Download raw datasets (752K examples, ~900MB)
python src/nl2sql/data/download_all_datasets.py

# Clean and prepare for training
python src/nl2sql/data/prepare_unsloth_data.py
```

**Training features:**
- LoRA fine-tuning (r=16, alpha=32)
- Unsloth integration for efficient training
- Curriculum learning support
- WandB logging

## Training Data

**752K curated examples** from 5 datasets (see `data/DATASET_SUMMARY.md`):
- Spider (7K benchmark quality queries)
- SQaLe (517K with 22,989 real schemas)
- Gretel (100K high-quality synthetic)
- SQL-Create-Context (78K with schema context)
- Know-SQL (49K educational variety)

Standard SQL only (SQLite/PostgreSQL/MySQL) - no dialect-specific extensions.

## Project Structure

```
nl2sql/
├── src/nl2sql/
│   ├── data/          # Dataset download and preprocessing
│   ├── train/         # Training scripts (Unsloth/LoRA)
│   ├── eval/          # Evaluation (baseline, SFT)
│   ├── optim/         # DSPy and TextGrad optimization
│   │   ├── configs/   # YAML configuration files
│   │   ├── config.py  # Pydantic config schema
│   │   └── optimizers.py  # Optimizer registry
│   └── utils/         # Shared utilities (SQL execution, metrics)
├── models/            # Tokenizer configs and chat templates
├── data/              # Dataset documentation
└── results/           # Evaluation outputs
```

## Requirements

- Python 3.10+
- GPU with 16GB+ VRAM (for 7B model inference)
- ~5GB disk for Spider evaluation data
- ~20GB for full training datasets (if downloading locally)

## Citation

```bibtex
@software{nl2sql2025,
  author = {Ismail, Asad},
  title = {NL2SQL: Comprehensive Text-to-SQL Methods Comparison},
  year = {2025},
  url = {https://github.com/Asad-Ismail/nl2sql}
}
```