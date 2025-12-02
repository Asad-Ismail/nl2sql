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


def create_prompt(example: Dict) -> str:
    """Create training prompt"""
    context = example.get('context', '')
    db_id = example.get('db_id', '')
    
    prompt = f"""### Task: Convert natural language question to SQL query
Database: {db_id}
Database Schema: {context}
### Question: {example['question']}
### SQL: {example['sql']}"""
    
    return {"text": prompt}


def load_and_prepare_datasets(
    dataset_configs: List[DatasetConfig]
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
        stopping_strategy='all_exhausted'
    )
    
    logger.info(f"✓ Created mixed dataset with {len(mixed_dataset):,} examples\n")
    
    # Format for training (add text field)
    logger.info("Formatting dataset for Unsloth...")
    formatted_dataset = mixed_dataset.map(
        create_prompt,
        batched=False,
        desc="Formatting"
    )
    
    logger.info(f"✓ Formatting complete\n")
    
    return formatted_dataset


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
        model_name: str = "unsloth/llama-3-8b-bnb-4bit",
        output_dir: str = "models/nl2sql-unsloth",
        max_seq_length: int = 512,
        load_in_4bit: bool = True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Initializing Unsloth Model: {model_name}")
        logger.info(f"{'='*70}\n")
        
        # Load model with Unsloth optimizations
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
        num_epochs: int = 3,
        batch_size: int = 3,
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
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Don't pack sequences for SQL
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                output_dir=self.output_dir,
                report_to=["wandb"],
                save_strategy="epoch",
                save_total_limit=2,
                logging_first_step=True,
            ),
            callbacks=[ProgressCallback()]
        )
        
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
        name=f"unsloth-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "framework": "unsloth",
            "model": "unsloth/llama-3-8b-bnb-4bit",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "batch_size": 3,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 12,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "max_seq_length": 512,
            "quantization": "4bit",
            "dataset_weights": {
                "spider": 0.5,
                "sqale": 0.3,
                "sql_context": 0.03
            }
        }
    )
    
    logger.info("\n" + "="*70)
    logger.info("NL2SQL Unsloth Training Pipeline")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"W&B Run: {wandb.run.name}")
    logger.info("="*70 + "\n")
    
    # Define datasets
    dataset_configs = [
        DatasetConfig(
            name="Spider",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.5,
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
            data_file="sqale_clean.jsonl",
            max_samples=50000
        ),
        DatasetConfig(
            name="SQL-Context",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.03,
            difficulty=2,
            is_hf_dataset=True,
            data_file="sql_context_clean.jsonl",
            max_samples=10000
        ),
    ]
    
    # Load and prepare datasets
    dataset = load_and_prepare_datasets(dataset_configs)
    
    # Initialize trainer
    trainer = UnslothTrainer(
        model_name="unsloth/llama-3-8b-bnb-4bit",  # Or "unsloth/mistral-7b-bnb-4bit"
        output_dir="models/nl2sql-unsloth",
        max_seq_length=512,
        load_in_4bit=True
    )
    
    # Train
    logger.info("\n🚀 Starting Unsloth Training\n")
    trainer.train(
        dataset=dataset,
        num_epochs=3,
        batch_size=3,
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
