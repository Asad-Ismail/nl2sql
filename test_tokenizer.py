"""
Unsloth-based Training Pipeline for NL2SQL

Features:
- 2x faster training with Unsloth optimizations
- 50% less memory usage
- Multi-dataset weighted interleaving
- Weights & Biases logging
- Checkpoint recovery
"""
import os
import json
import torch
from datasets import load_dataset, interleave_datasets
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from unsloth.chat_templates import train_on_responses_only
import time
from tqdm import tqdm
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_unsloth.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    path: str
    weight: float
    difficulty: int
    max_samples: Optional[int] = None
    is_hf_dataset: bool = False
    data_file: Optional[str] = None


def create_prompt(example: Dict, tokenizer) -> str:
    """Create training prompt with Alpaca format for instruction fine-tuning.
    
    Loss will be calculated ONLY on the SQL response using train_on_responses_only.
    The model receives schema + question as input and generates SQL as output.
    db_id is excluded - the model should generalize from schema structure alone.
    
    CRITICAL: EOS token is appended after SQL to teach model when to stop generating.
    """
    schema = example.get('context', '').strip()
    question = example.get('question', '').strip()
    sql = example.get('sql', '').strip()
    
    # Simplified Alpaca format that works with train_on_responses_only
    # The delimiter must be exactly "### Response:\n" for proper loss masking
    # EOS token tells model when to stop generating
    prompt = f"""### Instruction:
Generate a SQL query to answer the given question based on the provided database schema.

Database Schema:
{schema}

Question: {question}

### Response:
{sql}{tokenizer.eos_token}"""
    
    return {"text": prompt}


def load_and_prepare_datasets(
    dataset_configs: List[DatasetConfig],
    tokenizer
) -> tuple:
    """Load datasets from HuggingFace with weighted interleaving"""
    datasets_list = []
    weights_list = []
    
    logger.info("\n" + "="*70)
    logger.info("Loading Datasets")
    logger.info("="*70)
    
    for config in dataset_configs:
        try:
            logger.info(f"\nLoading {config.name}...")
            
            if config.is_hf_dataset:
                dataset = load_dataset(
                    config.path,
                    data_files=config.data_file,
                    split='train'
                )
            elif os.path.exists(config.path):
                dataset = load_dataset('json', data_files=config.path, split='train')
            else:
                logger.warning(f"⚠ Skipping {config.name}: path not found")
                continue
            
            # Subsample if needed
            if config.max_samples and len(dataset) > config.max_samples:
                logger.info(f"  Subsampling {config.max_samples:,} from {len(dataset):,}")
                indices = np.random.choice(len(dataset), config.max_samples, replace=False)
                dataset = dataset.select(indices)
            
            logger.info(f"  ✓ Loaded {len(dataset):,} examples (weight={config.weight})")
            
            # Show sample
            if len(dataset) > 0:
                sample = dataset[0]
                logger.info(f"  Sample question: {sample['question'][:80]}...")
                logger.info(f"  Sample SQL: {sample['sql'][:80]}...")
            
            datasets_list.append(dataset)
            weights_list.append(config.weight)
            
        except Exception as e:
            logger.error(f"⚠ Error loading {config.name}: {e}")
            continue
    
    if not datasets_list:
        raise ValueError("No datasets loaded successfully!")
    
    # Normalize weights
    total_weight = sum(weights_list)
    probabilities = [w / total_weight for w in weights_list]
    
    logger.info("\n" + "="*70)
    logger.info("Dataset Mixing Strategy")
    logger.info("="*70)
    for config, prob in zip(dataset_configs[:len(probabilities)], probabilities):
        logger.info(f"  {config.name}: {prob:.1%} sampling probability")
    
    # Interleave datasets
    logger.info("\nInterleaving datasets...")
    mixed_dataset = interleave_datasets(
        datasets_list,
        probabilities=probabilities,
        seed=42,
        stopping_strategy='first_exhausted'  # Maintains exact weight ratios
    )
    
    logger.info(f"✓ Created mixed dataset with {len(mixed_dataset):,} examples\n")
    logger.info("ℹ️  Note: Using 'first_exhausted' strategy to maintain exact dataset weight ratios")
    
    # Format for training (add text field with EOS token)
    logger.info("Formatting dataset for Unsloth...")
    formatted_dataset = mixed_dataset.map(
        lambda x: create_prompt(x, tokenizer),
        batched=False,
        desc="Formatting"
    )
    
    logger.info(f"✓ Formatting complete\n")
    
    # Split into train/validation (98% train, 2% validation)
    logger.info("Splitting into train/validation sets...")
    split_dataset = formatted_dataset.train_test_split(test_size=0.02, seed=42)
    logger.info(f"✓ Train: {len(split_dataset['train']):,} examples")
    logger.info(f"✓ Validation: {len(split_dataset['test']):,} examples\n")
    
    return split_dataset


class ProgressCallback(TrainerCallback):
    """Custom callback for progress tracking"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info("\n🚀 Training started!\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        epoch_num = int(state.epoch) + 1 if state.epoch is not None else 1
        total_epochs = int(args.num_train_epochs) if args.num_train_epochs else 3
        logger.info(f"\n📊 Epoch {epoch_num}/{total_epochs} started")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            epoch_num = int(state.epoch) if state.epoch is not None else 1
            total_epochs = int(args.num_train_epochs) if args.num_train_epochs else 3
            logger.info(f"✓ Epoch {epoch_num}/{total_epochs} completed in {epoch_time/60:.1f} minutes")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            if state.max_steps > 0:
                progress = (state.global_step / state.max_steps) * 100
                
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    eta_seconds = (elapsed / state.global_step) * (state.max_steps - state.global_step)
                    eta_minutes = eta_seconds / 60
                    
                    log_msg = f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%)"
                    if 'loss' in logs:
                        log_msg += f" | Loss: {logs['loss']:.4f}"
                    if 'learning_rate' in logs:
                        log_msg += f" | LR: {logs['learning_rate']:.2e}"
                    log_msg += f" | ETA: {eta_minutes:.1f}min"
                    
                    logger.info(log_msg)


class UnslothTrainer:
    """Unsloth-optimized trainer for NL2SQL"""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        output_dir: str = "models/nl2sql-unsloth",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Initializing Unsloth Model: {model_name}")
        logger.info(f"{'='*70}\n")
        
        # Load model with Unsloth optimizations
        # Unsloth can load any HF model and optimize it
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,
        )
        
        logger.info(f"✓ Model and tokenizer loaded")
        logger.info(f"✓ Max sequence length: {max_seq_length}")
        logger.info(f"✓ 4-bit quantization: {load_in_4bit}")
        
        # Configure LoRA
        logger.info("\n⚙️  Applying LoRA configuration...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Optimized for speed
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=42,
        )
        
        logger.info("✓ LoRA applied with Unsloth optimizations\n")
    
    def train(
        self,
        dataset,
        num_epochs: int = 1,  # Default to 1 epoch (full pass over 1.6M examples)
        batch_size: int = 4,  # Increased from 3 - you have 7GB free VRAM
        learning_rate: float = 2e-4
    ):
        """Train the model"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Configuration")
        logger.info(f"{'='*70}")
        logger.info(f"Dataset size: {len(dataset):,} examples")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size} (gradient accumulation: 4)")
        logger.info(f"Effective batch size: {batch_size * 4}")
        logger.info(f"Learning rate: {learning_rate:.2e}")
        logger.info(f"Precision: Mixed (BF16 if supported, else FP16)")
        logger.info("")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check for existing checkpoints
        checkpoint_dir = None
        if os.path.exists(self.output_dir):
            checkpoint_dirs = [d for d in os.listdir(self.output_dir) 
                             if d.startswith('checkpoint-') and os.path.isdir(os.path.join(self.output_dir, d))]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
                checkpoint_dir = os.path.join(self.output_dir, checkpoint_dirs[-1])
                logger.info(f"📂 Found existing checkpoint: {checkpoint_dir}")
        
        # Create trainer with response-only training
        # CRITICAL: train_on_responses_only must be called BEFORE creating SFTTrainer
        # It modifies the dataset to mask input tokens from loss calculation
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],  # Monitor generalization during training
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=8,  # Use 8 cores for faster preprocessing (was 2)
            packing=False,  # Must be False when using train_on_responses_only (incompatible with loss masking)
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_ratio=0.03,  # Warmup for 3% of training (standard for full runs)
                num_train_epochs=num_epochs,  # Train on full dataset for num_epochs
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=100,  # Log every 100 steps (was 10, too frequent for long runs)
                eval_strategy="steps",  # Evaluate on validation set
                eval_steps=5000,  # Evaluate every 5K steps (same as save_steps)
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",  # Unsloth docs use linear (standard for transformers)
                seed=42,
                output_dir=self.output_dir,
                report_to=["wandb"],
                save_strategy="steps",  # Save at regular intervals during training
                save_steps=5000,  # Save checkpoint every 5K steps (~20 checkpoints per epoch)
                save_total_limit=2,  # Keep only last 2 checkpoints to save disk space
                logging_first_step=True,
                load_best_model_at_end=False,  # Don't load best for memory efficiency
            ),
            callbacks=[ProgressCallback()]
        )
        
        # CRITICAL: Mask input tokens from loss calculation
        # Debug: Check first example in dataset
        logger.info("\nDEBUG: Checking first training example...")
        first_example = dataset["train"][0]
        logger.info(f"First example keys: {first_example.keys()}")
        logger.info(f"First example text (first 500 chars):\n{first_example['text'][:500]}")
        logger.info(f"'### Instruction:' in text: {'### Instruction:' in first_example['text']}")
        logger.info(f"'### Response:' in text: {'### Response:' in first_example['text']}")
        
        logger.info("\nApplying response-only training mask...")
        trainer = train_on_responses_only(
            trainer,
            instruction_part="### Instruction:",
            response_part="### Response:",
        )
        logger.info("✓ Response-only masking applied\n")
        
        # Train
        trainer.train(resume_from_checkpoint=checkpoint_dir)
        
        # Save final model
        logger.info(f"\n💾 Saving final model...")
        self.model.save_pretrained(f"{self.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final")
        logger.info(f"✓ Model saved to {self.output_dir}/final\n")


def main():
    """Main training pipeline with Unsloth"""
    
    start_time = time.time()
    
    # Initialize W&B
    wandb.init(
        project="nl2sql-training",
        name=f"unsloth-codellama-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "framework": "unsloth",
            "model": "codellama/CodeLlama-7b-hf",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 16,
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "save_steps": 5000,
            "max_seq_length": 2048,
            "quantization": "4bit",
            "packing": True,
            "warmup_ratio": 0.03,
            "total_datasets": 5,
            "total_examples": "1.6M",
            "dataset_weights": {
                "spider": 0.4,
                "sqale": 0.3,
                "sql_context": 0.15,
                "gretel": 0.1,
                "know_sql": 0.05
            }
        }
    )
    
    logger.info("\n" + "="*70)
    logger.info("NL2SQL Unsloth Training Pipeline")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"W&B Run: {wandb.run.name}")
    logger.info("="*70 + "\n")
    
    # Define datasets (5 curated datasets, 752K total examples)
    dataset_configs = [
        DatasetConfig(
            name="Spider",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.4,
            difficulty=2,
            is_hf_dataset=True,
            data_file="spider_clean.jsonl"
        ),
        DatasetConfig(
            name="SQaLe",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.3,
            difficulty=2,
            is_hf_dataset=True,
            data_file="sqale_clean.jsonl"
        ),
        DatasetConfig(
            name="SQL-Context",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.15,
            difficulty=2,
            is_hf_dataset=True,
            data_file="sql_context_clean.jsonl"
        ),
        DatasetConfig(
            name="Gretel",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.1,
            difficulty=2,
            is_hf_dataset=True,
            data_file="gretel_clean.jsonl"
        ),
        DatasetConfig(
            name="Know-SQL",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.05,
            difficulty=2,
            is_hf_dataset=True,
            data_file="know_sql_clean.jsonl"
        ),
    ]
    
    # Initialize trainer first to get tokenizer
    trainer = UnslothTrainer(
        model_name="codellama/CodeLlama-7b-hf",
        output_dir="models/nl2sql-unsloth",
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Load and prepare datasets with tokenizer for EOS token
    dataset = load_and_prepare_datasets(dataset_configs, trainer.tokenizer)
    
    # Train
    logger.info("\n🚀 Starting Unsloth Training\n")
    trainer.train(
        dataset=dataset,
        num_epochs=1,  # 1 full epoch = ~100K steps with 1.6M examples
        batch_size=4,  # Using 4 for better GPU utilization
        learning_rate=2e-4
    )
    
    total_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("✅ Training Complete!")
    logger.info("="*70)
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Model saved to: {trainer.output_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70 + "\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()
