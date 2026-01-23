# NL2SQL: Text-to-SQL Methods Comparison

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-nl2sql--deduplicated-yellow)](https://huggingface.co/datasets/AsadIsmail/nl2sql-deduplicated)

**Compare and evaluate different LLM-based approaches for Text-to-SQL generation**, from simple prompting to advanced optimization and fine-tuning.

**Console Scripts:**
- `nl2sql-baseline` - Baseline evaluation (zero-shot, few-shot, self-correction)
- `nl2sql-dspy` - DSPy optimization with configurable optimizers
- `nl2sql-sft-eval` - Fine-tuned model evaluation

**Methods covered:**
- Zero-shot prompting
- Few-shot in-context learning
- Self-correction with execution feedback
- DSPy few-shot optimization (YAML configurable)
- TextGrad prompt optimization
- LoRA fine-tuning (parameter-efficient)

**Bonus:** 750K+ curated training examples from 5 datasets (Spider, SQaLe, Gretel, SQL-Context, Know-SQL) for reproducible experiments.

📖 **[Read the full analysis on my blog](https://asad-ismail.github.io/)** for detailed insights, methodology, and lessons learned.

## Results Summary

Performance on Spider dev set (1,034 examples):

### CodeLlama-7B Results

| Method | Valid SQL % | Execution Match % | Avg Time (s) | Notes |
|--------|-------------|-------------------|--------------|-------|
| **Baseline Methods** |
| Zero-Shot | 83.7% | 57.3% | 10.70s | Fast baseline |
| Few-Shot | 83.8% | 56.7% | 10.96s | Degraded performance ([see analysis](https://asad-ismail.github.io/)) |
| Self-Correction | 87.8% | 58.7% | 75.53s | +1.4% gain, but 7x slower |
| **Optimized Methods** |
| DSPy (MIPRO) | 84.0% | 59.0% | ~0s* | Matches Self-Correction instantly |
| **Fine-Tuned** |
| Unsloth LoRA | **90.1%** | **73.0%** | 2.50s | **Best 7B Result (+14% gain)** |

### Llama-3-70B Reference

| Method | Valid SQL % | Execution Match % | Avg Time (s) | Notes |
|--------|-------------|-------------------|--------------|-------|
| Zero-Shot | 98.8% | 78.3% | 4.34s | Large model ceiling |
| Self-Correction | 99.4% | 79.7% | 33.71s | State-of-the-art range |

*DSPy optimization time amortized across queries after initial training

**Key Findings:**
- **Fine-tuning wins**: LoRA fine-tuning delivers the best results for 7B models (+14% execution accuracy)
- **DSPy efficiency**: Matches self-correction performance without runtime overhead
- **Few-shot paradox**: Adding examples degraded performance (discussed in [blog analysis](https://asad-ismail.github.io/))
- **70B ceiling**: Large models approach 80% execution accuracy, setting the performance ceiling

## Installation

```bash
# Clone repository
git clone https://github.com/Asad-Ismail/nl2sql.git
cd nl2sql

# Install with dependencies
pip install -e .

# Set API keys for cloud providers (optional)
cp .env.example .env
# Edit .env and add your API keys
```

## Baseline Evaluation

Evaluate 3 baseline approaches (zero-shot, few-shot, self-correction) on Spider dev set:

```bash
# Local vLLM model (start server first)
vllm serve TheBloke/CodeLlama-7B-Instruct-AWQ --host 0.0.0.0 --port 8000
nl2sql-baseline --model codellama_7b --num-samples 100

# Cloud providers (no server needed, just set API key)
nl2sql-baseline --model claude_sonnet --num-samples 100
nl2sql-baseline --model llama_70b_nvidia --num-samples 100
nl2sql-baseline --model gpt4o --num-samples 100

# Full evaluation (all 1,034 Spider dev examples)
nl2sql-baseline --model codellama_7b
```

**Available models:** `codellama_7b`, `deepseek_coder_7b`, `mistral_7b`, `claude_sonnet`, `llama_70b_nvidia`, `gpt4o`, etc. (see `src/nl2sql/optim/configs/llm/providers.yaml`)

Results saved to `results/baseline_<model>/` with detailed reports.

## DSPy Optimization

Optimize prompts using DSPy optimizers with YAML configuration:

```bash
# Run with default config
nl2sql-dspy --config src/nl2sql/optim/configs/default.yaml

# Override config via CLI
nl2sql-dspy --config src/nl2sql/optim/configs/default.yaml \
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

## Training

**Recommended:** Use pre-cleaned HuggingFace dataset for training:

```bash
# Train with LoRA fine-tuning (uses HuggingFace dataset automatically)
python src/nl2sql/train/train_unsloth_complete.py

# Evaluate fine-tuned model
nl2sql-sft-eval --model models/your-model --num-samples 100
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

## LLM Provider System

Unified provider system supports multiple LLM backends with configurable rate limiting:

```python
from nl2sql.llm import get_llm

# Use local vLLM
llm = get_llm("codellama_7b")
response = llm.generate_text("Convert to SQL: show all users")

# Switch to Claude
llm = get_llm("claude_sonnet")

# Use with DSPy
from nl2sql.llm.dspy_adapter import configure_dspy_from_config
student_lm, teacher_lm = configure_dspy_from_config(
    student_model="codellama_7b",
    teacher_model="llama_70b_nvidia"
)
```

**Supported Providers:**
| Provider | Type | Models |
|----------|------|--------|
| vLLM (local) | OpenAI-compatible | CodeLlama, DeepSeek, Mistral |
| NVIDIA NIM | OpenAI-compatible | Llama 70B/405B, Kimi K2 |
| Anthropic | Native SDK | Claude Sonnet/Haiku/Opus |
| OpenRouter | OpenAI-compatible | Any model on OpenRouter |
| OpenAI | Native | GPT-4o, GPT-4o-mini |

Configure in `src/nl2sql/optim/configs/llm/providers.yaml`.

## Project Structure

```
nl2sql/
├── src/nl2sql/
│   ├── data/          # Dataset download and preprocessing
│   ├── train/         # Training scripts (Unsloth/LoRA)
│   ├── eval/          # Evaluation (baseline, SFT)
│   ├── llm/           # Unified LLM provider system
│   │   ├── config.py  # Provider config schemas
│   │   ├── factory.py # Provider factory with rate limiting
│   │   └── dspy_adapter.py  # DSPy integration
│   ├── optim/         # DSPy and TextGrad optimization
│   │   ├── configs/   # YAML configuration files
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

## Learn More

📖 **[Read the detailed analysis on my blog](https://asad-ismail.github.io/)** covering:
- Why few-shot learning degraded performance
- Self-correction vs DSPy optimization trade-offs
- Fine-tuning strategies and data preparation
- Comparative analysis across model sizes
- Recommendations for production deployments

## Citation

```bibtex
@software{nl2sql2025,
  author = {Ismail, Asad},
  title = {NL2SQL: Comprehensive Text-to-SQL Methods Comparison},
  year = {2025},
  url = {https://github.com/Asad-Ismail/nl2sql}
}
```
