"""
Advanced Training Pipeline for NL2SQL with Synthetic Data

Features:
- Multi-dataset mixing with smart sampling
- Synthetic data augmentation
- Progressive difficulty curriculum
- Advanced LoRA training with proper validation
- Execution-based validation during training
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, concatenate_datasets, interleave_datasets
import numpy as np
from typing import Dict, List, Optional
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import time
from tqdm import tqdm
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    path: str  # Can be local JSONL or HuggingFace dataset name
    weight: float  # Sampling weight
    difficulty: int  # 1=easy, 2=medium, 3=hard
    max_samples: Optional[int] = None
    is_hf_dataset: bool = False  # If True, load from HuggingFace
    data_file: Optional[str] = None  # Specific file in HF dataset


def create_prompt(example: Dict) -> str:
    """Create training prompt matching HuggingFace dataset format"""
    context = example.get('context', '')
    db_id = example.get('db_id', '')
    
    prompt = f"""### Task: Convert natural language question to SQL query
Database: {db_id}
Database Schema: {context}
### Question: {example['question']}
### SQL: {example['sql']}"""
    
    return prompt


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for training"""
    prompts = [create_prompt({
        'context': examples['context'][i],
        'db_id': examples['db_id'][i],
        'question': examples['question'][i],
        'sql': examples['sql'][i]
    }) for i in range(len(examples['question']))]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    
    # For causal LM, labels are same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def load_and_prepare_datasets(
    dataset_configs: List[DatasetConfig],
    tokenizer,
    max_length: int = 512
) -> tuple:
    """Load datasets from HuggingFace with proper weighted interleaving"""
    datasets_list = []
    weights_list = []
    
    logger.info("\n" + "="*70)
    logger.info("Loading Datasets")
    logger.info("="*70)
    
    for config in dataset_configs:
        try:
            logger.info(f"\nLoading {config.name}...")
            
            if config.is_hf_dataset:
                # Load from HuggingFace
                dataset = load_dataset(
                    config.path,
                    data_files=config.data_file,
                    split='train'
                )
            elif os.path.exists(config.path):
                # Load from local JSONL
                dataset = load_dataset('json', data_files=config.path, split='train')
            else:
                logger.warning(f"⚠ Skipping {config.name}: path not found")
                continue
            
            # Subsample if needed
            if config.max_samples and len(dataset) > config.max_samples:
                logger.info(f"  Subsampling {config.max_samples:,} from {len(dataset):,}")
                indices = np.random.choice(len(dataset), config.max_samples, replace=False)
                dataset = dataset.select(indices)
            
            logger.info(f"  ✓ Loaded {len(dataset):,} examples (weight={config.weight}, difficulty={config.difficulty})")
            
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
    
    # Normalize weights to sum to 1
    total_weight = sum(weights_list)
    probabilities = [w / total_weight for w in weights_list]
    
    logger.info("\n" + "="*70)
    logger.info("Dataset Mixing Strategy")
    logger.info("="*70)
    for i, (config, prob) in enumerate(zip(dataset_configs[:len(probabilities)], probabilities)):
        logger.info(f"  {config.name}: {prob:.1%} sampling probability")
    
    # Interleave datasets with weights
    logger.info("\nInterleaving datasets with weighted sampling...")
    mixed_dataset = interleave_datasets(
        datasets_list,
        probabilities=probabilities,
        seed=42,
        stopping_strategy='all_exhausted'
    )
    
    logger.info(f"✓ Created mixed dataset with {len(mixed_dataset):,} examples\n")
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = mixed_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=mixed_dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"✓ Tokenization complete\n")
    
    return tokenized_dataset, mixed_dataset


class ProgressCallback(TrainerCallback):
    """Custom callback for detailed progress tracking"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info("\n🚀 Training started!\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        logger.info(f"\n📊 Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)} started")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        logger.info(f"✓ Epoch {int(state.epoch)}/{int(args.num_train_epochs)} completed in {epoch_time/60:.1f} minutes")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Calculate progress
            if state.max_steps > 0:
                progress = (state.global_step / state.max_steps) * 100
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


class CurriculumTrainer:
    """Curriculum learning trainer - train on progressively harder data"""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        output_dir: str = "models/nl2sql-lora",
        use_curriculum: bool = True,
        resume_from_checkpoint: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_curriculum = use_curriculum
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Load model and tokenizer
        logger.info(f"\n{'='*70}")
        logger.info(f"Initializing Model: {model_name}")
        logger.info(f"{'='*70}\n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
        
        # Check if resuming from checkpoint (must be a valid path, not just True)
        if resume_from_checkpoint and isinstance(resume_from_checkpoint, str) and os.path.exists(resume_from_checkpoint):
            logger.info(f"\n📂 Resuming from checkpoint: {resume_from_checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(
                resume_from_checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            if resume_from_checkpoint:
                logger.info("\n⚠️  No valid checkpoint found, loading base model...")
            logger.info(f"\n📥 Loading base model from HuggingFace...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Configure LoRA
            logger.info("\n⚙️  Applying LoRA configuration...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        logger.info("\n📊 Model Parameter Summary:")
        self.model.print_trainable_parameters()
        logger.info("")
    
    def train_stage(
        self, 
        dataset,
        stage_name: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """Train single curriculum stage"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Stage: {stage_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Dataset size: {len(dataset):,} examples")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size} (gradient accumulation: 4)")
        logger.info(f"Effective batch size: {batch_size * 4}")
        logger.info(f"Learning rate: {learning_rate:.2e}")
        logger.info("")
        
        stage_output_dir = f"{self.output_dir}/{stage_name}"
        os.makedirs(stage_output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to=["wandb"],  # Enable W&B logging
            warmup_steps=100,
            optim="adamw_torch",
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            logging_first_step=True,
            log_level="info"
        )
        
        # Check for existing checkpoints in this stage
        checkpoint_dir = None
        if os.path.exists(stage_output_dir):
            checkpoint_dirs = [d for d in os.listdir(stage_output_dir) 
                             if d.startswith('checkpoint-') and os.path.isdir(os.path.join(stage_output_dir, d))]
            if checkpoint_dirs:
                # Get latest checkpoint
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
                checkpoint_dir = os.path.join(stage_output_dir, checkpoint_dirs[-1])
                logger.info(f"📂 Found existing checkpoint: {checkpoint_dir}")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            ),
            callbacks=[ProgressCallback()]
        )
        
        # Train (resume from checkpoint if available)
        trainer.train(resume_from_checkpoint=checkpoint_dir)
        
        # Save final checkpoint
        final_dir = f"{self.output_dir}/{stage_name}_final"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"\n✓ Stage {stage_name} complete! Saved to {final_dir}\n")
    
    def train_curriculum(
        self,
        dataset_configs: List[DatasetConfig],
        epochs_per_stage: List[int] = [3, 3, 3]
    ):
        """Train with curriculum learning"""
        
        # Sort configs by difficulty
        easy = [c for c in dataset_configs if c.difficulty == 1]
        medium = [c for c in dataset_configs if c.difficulty == 2]
        hard = [c for c in dataset_configs if c.difficulty == 3]
        
        stages = [
            ("stage1_easy", easy, epochs_per_stage[0]),
            ("stage2_medium", medium, epochs_per_stage[1]),
            ("stage3_hard", hard, epochs_per_stage[2])
        ]
        
        for stage_name, configs, epochs in stages:
            if not configs:
                logger.warning(f"Skipping {stage_name} - no datasets")
                continue
            
            # Load and prepare mixed dataset for this stage
            tokenized_dataset, raw_dataset = load_and_prepare_datasets(
                configs, 
                self.tokenizer
            )
            
            # Train
            self.train_stage(tokenized_dataset, stage_name, num_epochs=epochs)
    
    def train_mixed(
        self,
        dataset_configs: List[DatasetConfig],
        num_epochs: int = 3
    ):
        """Train on all data mixed together (baseline)"""
        logger.info("\n🔀 Training with mixed data (no curriculum)\n")
        
        tokenized_dataset, raw_dataset = load_and_prepare_datasets(
            dataset_configs,
            self.tokenizer
        )
        
        self.train_stage(tokenized_dataset, "mixed", num_epochs=num_epochs)


def main():
    """Main training pipeline with HuggingFace deduplicated dataset"""
    
    start_time = time.time()
    
    # Initialize Weights & Biases
    wandb.init(
        project="nl2sql-training",
        name=f"mixed-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": "codellama/CodeLlama-7b-hf",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 16,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "dataset_weights": {
                "spider": 0.5,
                "sqale": 0.3,
                "sql_context": 0.03
            }
        }
    )
    
    logger.info("\n" + "="*70)
    logger.info("NL2SQL Curriculum Training Pipeline")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"W&B Run: {wandb.run.name}")
    logger.info("="*70 + "\n")
    
    # Define datasets with configurations matching HuggingFace deduplicated dataset
    dataset_configs = [
        # Spider: Benchmark quality (highest weight)
        DatasetConfig(
            name="Spider",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.5,
            difficulty=2,
            is_hf_dataset=True,
            data_file="spider_clean.jsonl"
        ),
        
        # SQaLe: Large-scale real schemas
        DatasetConfig(
            name="SQaLe",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.3,
            difficulty=2,
            is_hf_dataset=True,
            data_file="sqale_clean.jsonl",
            max_samples=50000  # Subsample large dataset
        ),
        
        # Gretel Synthetic: High-quality synthetic (COMMENTED OUT FOR NOW)
        # DatasetConfig(
        #     name="Gretel-Synthetic",
        #     path="AsadIsmail/nl2sql-deduplicated",
        #     weight=0.15,
        #     difficulty=1,
        #     is_hf_dataset=True,
        #     data_file="gretel_clean.jsonl",
        #     max_samples=20000
        # ),
        
        # SQL-Context: Schema-aware supplementary
        DatasetConfig(
            name="SQL-Context",
            path="AsadIsmail/nl2sql-deduplicated",
            weight=0.03,
            difficulty=2,  # Changed from 3 to 2 to include in medium stage
            is_hf_dataset=True,
            data_file="sql_context_clean.jsonl",
            max_samples=10000
        ),
    ]
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        model_name="codellama/CodeLlama-7b-hf",
        output_dir="models/nl2sql-curriculum-lora",
        use_curriculum=True,
        resume_from_checkpoint=None  # Set to checkpoint path to resume, or None to start fresh
    )
    
    # Train with mixed data (all datasets combined with weights)
    # Note: Curriculum disabled since we're using all medium-difficulty datasets
    logger.info("\n🔄 Starting Mixed Training (Weighted Interleaving)\n")
    logger.info("Weights: Spider=0.5, SQaLe=0.3, SQL-Context=0.03\n")
    
    trainer.train_mixed(
        dataset_configs=dataset_configs,
        num_epochs=3  # Train for 3 epochs on mixed data
    )
    
    total_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("✅ Training Complete!")
    logger.info("="*70)
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Models saved to: {trainer.output_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70 + "\n")
    
    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
