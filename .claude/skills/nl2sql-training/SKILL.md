---
name: nl2sql-training
description: LoRA fine-tuning patterns for NL2SQL with Unsloth. Use when configuring training, setting up datasets, or optimizing model performance. Covers LoRA configuration, curriculum learning, dataset weighting, and GPU memory management.
---

# NL2SQL Training Patterns

## When to Use
- Configuring LoRA fine-tuning for text-to-SQL
- Setting up multi-dataset training with weighted sampling
- Optimizing GPU memory usage during training
- Implementing curriculum learning stages
- Creating training prompts with proper formatting

## Core Patterns

### LoRA Configuration

Standard LoRA setup for NL2SQL (7B models):

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="codellama/CodeLlama-7b-hf",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    lora_alpha=16,           # Scaling parameter (typically = r)
    target_modules=[         # Layers to fine-tune
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0,          # Unsloth optimization
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

**Key parameters:**
- `r=16, lora_alpha=16`: Standard LoRA configuration for efficient fine-tuning
- `target_modules`: Attention + FFN layers for comprehensive adaptation
- `load_in_4bit=True`: Reduces memory by ~50%
- `max_seq_length=2048`: Sufficient for most SQL queries + schema context

### Training Prompt Format (Alpaca-style)

Critical: Loss calculated ONLY on SQL response using `train_on_responses_only`:

```python
def create_prompt(example: Dict, tokenizer) -> str:
    """Create training prompt - loss on SQL only"""
    schema = example.get('context', '').strip()
    question = example.get('question', '').strip()
    sql = example.get('sql', '').strip()

    # Delimiter "### Response:\n" required for proper loss masking
    # EOS token teaches model when to stop generating
    prompt = f""" ### Instruction:
Generate a SQL query to answer the given question based on the provided database schema.

Database Schema:
{schema}

Question: {question}

 ### Response:
{sql}{tokenizer.eos_token}"""

    return {"text": prompt}
```

**Critical elements:**
- `### Instruction:` and ` ### Response:\n` delimiters (exact format required)
- EOS token after SQL (teaches stopping behavior)
- Schema provided as context, db_id excluded (generalization)
- `train_on_responses_only=True` in SFTTrainer masks instruction from loss

### Multi-Dataset Weighted Sampling

Use `DatasetConfig` for balancing datasets:

```python
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    path: str
    weight: float          # Sampling weight (higher = more examples)
    difficulty: int        # Curriculum stage (1=easy, 2=medium, 3=hard)
    max_samples: Optional[int] = None
    is_hf_dataset: bool = False

# Example configuration
dataset_configs = [
    DatasetConfig("WikiSQL", "data/wikisql.jsonl", weight=0.3, difficulty=1, max_samples=50000),
    DatasetConfig("Spider", "data/spider_train.jsonl", weight=0.5, difficulty=2),
    DatasetConfig("SQaLe", "data/sqale.jsonl", weight=0.2, difficulty=3),
]
```

**Interleaving datasets:**
```python
from datasets import interleave_datasets

# Normalize weights to probabilities
total_weight = sum(config.weight for config in dataset_configs)
probabilities = [config.weight / total_weight for config in dataset_configs]

# Interleave with weighted sampling
mixed_dataset = interleave_datasets(
    datasets_list,
    probabilities=probabilities,
    seed=42,
    stopping_strategy='all_exhausted'
)
```

### Training Configuration

Standard training setup for NL2SQL:

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="models/nl2sql-unsloth",
    num_train_epochs=1,
    per_device_train_batch_size=4,    # Base batch size
    gradient_accumulation_steps=4,     # Effective: 4 * 4 = 16
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to=["wandb"],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=2048,
    dataset_text_field="text",
    dataset_kwargs={
        "add_special_tokens": False,  # Prompt already has them
        "append_concat_token": False
    },
    train_on_responses_only=True,     # CRITICAL: Mask instruction from loss
)
```

**Key settings:**
- `gradient_accumulation_steps=4`: For large datasets (752K examples)
- `learning_rate=2e-4`: Standard for LoRA fine-tuning
- `train_on_responses_only=True`: Loss only on SQL, not instruction
- `save_total_limit=3`: Manage disk space with checkpoint rotation

## GPU Memory Management

### Memory Requirements by Model Size

| Model | VRAM (4-bit) | VRAM (8-bit) | Recommended GPU |
|-------|--------------|--------------|-----------------|
| 7B    | ~6-8 GB      | ~10-12 GB    | RTX 3060 (12GB) or better |
| 13B   | ~10-12 GB    | ~16-18 GB    | RTX 4090 (24GB) |
| 34B   | ~20-24 GB    | ~40+ GB      | A100 (40GB) or multi-GPU |

### Optimization Techniques

```python
# 1. Reduce max_seq_length if memory constrained
max_seq_length=1024  # Instead of 2048

# 2. Reduce batch size, increase accumulation
per_device_train_batch_size=2
gradient_accumulation_steps=8  # Effective batch still 16

# 3. Enable gradient checkpointing
use_gradient_checkpointing="unsloth"  # Reduces memory ~30%

# 4. Monitor memory during training
import torch
logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Curriculum Learning

Progressive difficulty stages:

```python
# Stage 1: Easy (single-table, simple queries)
easy_configs = [c for c in dataset_configs if c.difficulty == 1]

# Stage 2: Medium (multi-table, joins)
medium_configs = [c for c in dataset_configs if c.difficulty == 2]

# Stage 3: Hard (complex nested queries, real schemas)
hard_configs = [c for c in dataset_configs if c.difficulty == 3]

# Train each stage separately
for stage, configs in enumerate([easy_configs, medium_configs, hard_configs], 1):
    dataset = load_and_prepare_datasets(configs, tokenizer)
    trainer.train()
    model.save_pretrained(f"models/stage{stage}_final")
```

## Critical Gotchas

### Dataset Field Inconsistency
Different datasets use different field names:
```python
# Always use .get() with fallbacks
question = ex.get("question", ex.get("sql_prompt", ""))
sql = ex.get("sql", ex.get("query", ""))
```

### Schema Context Truncation
Large schemas (SQaLe) exceed token limits:
```python
schema = example.get('context', '')[:1000]  # Truncate to 1000 chars
```

### HuggingFace Dataset Loading
Handle deprecated loading scripts:
```python
# Old way (deprecated):
dataset = load_dataset("wikisql")

# New way:
dataset = load_dataset("json", data_files="data/wikisql.jsonl", split="train")
```

### Checkpoint Recovery
Resume from existing checkpoints:
```python
checkpoint_dir = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])

trainer.train(resume_from_checkpoint=checkpoint_dir)
```

## Integration

- **Related skill:** `prompt-patterns` - For inference prompt formatting
- **Related skill:** `data-processing` - For dataset preparation
- **Training script:** `src/nl2sql/train/train_unsloth_complete.py`

## Anti-Patterns

### What NOT to Do

**Don't forget EOS token:**
```python
# BAD: Model doesn't learn when to stop
prompt = f"### Response:\n{sql}"

# GOOD: Teaches stopping behavior
prompt = f"### Response:\n{sql}{tokenizer.eos_token}"
```

**Don't train on instruction tokens:**
```python
# BAD: Wastes capacity on learning the prompt format
SFTTrainer(..., train_on_responses_only=False)

# GOOD: Focus learning on SQL generation
SFTTrainer(..., train_on_responses_only=True)
```

**Don't mix curriculum stages:**
```python
# BAD: Hard examples overwhelm easy learning
dataset = load_all_datasets()  # All difficulties mixed

# GOOD: Progressive training
for stage_configs in [easy, medium, hard]:
    train(stage_configs)
```
