"""
Unsloth-based Training Pipeline for NL2SQL
Fixed using Daniel Hanchen's 'apply_chat_template' strategy.

Features:
- 2x faster training with Unsloth optimizations
- 50% less memory usage
- Multi-dataset weighted interleaving
- Weights & Biases logging
- Checkpoint recovery
- Response-only loss masking
"""
import os
import time
import torch
from datasets import load_dataset, interleave_datasets, Dataset
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
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

# --- Define the Exact Template from the GitHub Issue ---
# We define this string explicitly so we know EXACTLY what delimiters to use.
custom_alpaca_template = """ ### Instruction: Generate a SQL query to answer the given question based on the provided database schema
{}

 ### Response:
{}"""


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
        total_epochs = int(args.num_train_epochs) if args.num_train_epochs else 1
        logger.info(f"\n📊 Epoch {epoch_num}/{total_epochs} started")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            epoch_num = int(state.epoch) if state.epoch is not None else 1
            total_epochs = int(args.num_train_epochs) if args.num_train_epochs else 1
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


def format_columns(example):
    """
    Step 1: Convert NL2SQL columns to standard Instruction/Output format.
    """
    schema = example.get('context', '').strip()
    question = example.get('question', '').strip()
    sql = example.get('sql', '').strip()
    
    # We combine Schema + Question into the "Instruction" slot
    combined_instruction = f"Database Schema:\n{schema}\n\nQuestion: {question}"
    
    return {
        "instruction": combined_instruction,
        "output": sql
    }


def formatting_prompts_func(examples, tokenizer):
    """
    Step 2: Apply the template to create the final text field.
    This replaces 'apply_chat_template' for standard SFT usage.
    """
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    
    for instruction, output in zip(instructions, outputs):
        text = custom_alpaca_template.format(instruction, output) + EOS_TOKEN
        texts.append(text)
        
    return {"text": texts}


def load_and_prepare_datasets(dataset_configs: List[DatasetConfig], tokenizer) -> tuple:
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
            
            # Standardize columns immediately
            dataset = dataset.map(format_columns)
            
            logger.info(f"  ✓ Loaded {len(dataset):,} examples (weight={config.weight})")
            
            # Show sample
            if len(dataset) > 0:
                sample = dataset[0]
                logger.info(f"  Sample instruction: {sample['instruction'][:80]}...")
                logger.info(f"  Sample output: {sample['output'][:80]}...")
            
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
    
    logger.info(f"✓ Created mixed dataset with {len(mixed_dataset):,} examples")
    logger.info(" Using 'all_exhausted' strategy to train on entire dataset\n")
    
    # Apply the formatting function
    logger.info("Applying template formatting...")
    formatted_dataset = mixed_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        desc="Formatting"
    )
    
    # Filter empty/bad rows
    original_len = len(formatted_dataset)
    formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"]) > 10)
    filtered_count = original_len - len(formatted_dataset)
    if filtered_count > 0:
        logger.info(f"  Filtered {filtered_count} empty/invalid rows")
    
    logger.info(f"✓ Formatting complete\n")
    
    # Debug: Show sample prompt
    logger.info(f"[DEBUG] Sample Prompt:\n{formatted_dataset[0]['text'][:500]}...\n")
    
    # Split into train/validation (98% train, 2% validation)
    logger.info("Splitting into train/validation sets...")
    split_dataset = formatted_dataset.train_test_split(test_size=0.02, seed=42)
    logger.info(f"✓ Train: {len(split_dataset['train']):,} examples")
    logger.info(f"✓ Validation: {len(split_dataset['test']):,} examples\n")
    
    return split_dataset


class UnslothTrainer:
    """Unsloth-optimized trainer for NL2SQL with response-only loss masking"""
    
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
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,
        )
        
        logger.info(f"✓ Model and tokenizer loaded")
        logger.info(f"✓ Max sequence length: {max_seq_length}")
        logger.info(f"✓ 4-bit quantization: {load_in_4bit}")
        
        # Register Chat Template for Alpaca format
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="alpaca",
        )
        logger.info("✓ Alpaca chat template registered")
        
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
    
    def verify_response_masking(self, dataset):
        """Verify that the response masking will work correctly"""
        logger.info("🔍 Verifying tokenizer matching for response masking...")
        
        sample_text = dataset["train"][0]["text"]
        
        # Print raw text to check format
        logger.info(f"RAW TEXT SNIPPET:\n{repr(sample_text[:300])}")
        
        # Setup the matcher sequence
        response_template = "### Response:"
        response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        
        # Get tokenized sample
        tokenized_sample = self.tokenizer(sample_text, add_special_tokens=False)["input_ids"]
        
        # Simulate the search
        def find_start_of_training(full_sequence, search_sequence):
            seq_len = len(search_sequence)
            for i in range(len(full_sequence) - seq_len + 1):
                if full_sequence[i:i + seq_len] == search_sequence:
                    return i + seq_len
            return -1
        
        start_index = find_start_of_training(tokenized_sample, response_token_ids)
        
        if start_index != -1:
            training_tokens = tokenized_sample[start_index:]
            decoded_training = self.tokenizer.decode(training_tokens[:50])  # First 50 tokens
            
            logger.info("\n✅ SUCCESS: The trainer will mask correctly.")
            logger.info(f"   Split point found at token index: {start_index}")
            logger.info(f"   Model will learn from: '{decoded_training[:100]}...'")
            logger.info("   (Should be your SQL query starting with newline or SELECT)\n")
            return True
        else:
            logger.error("\n❌ FAILURE: Could not find response delimiter in tokenized text!")
            logger.error("   The train_on_responses_only may not work correctly.\n")
            return False
    
    def train(
        self,
        dataset,
        num_epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 100,
        save_steps: int = 5000,
        eval_steps: int = 5000,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        save_total_limit: int = 3
    ):
        """Train the model with response-only loss masking"""
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        total_steps = (len(dataset["train"]) * num_epochs) // effective_batch_size
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Configuration")
        logger.info(f"{'='*70}")
        logger.info(f"Train dataset size: {len(dataset['train']):,} examples")
        logger.info(f"Validation dataset size: {len(dataset['test']):,} examples")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Estimated total steps: {total_steps:,}")
        logger.info(f"Learning rate: {learning_rate:.2e}")
        logger.info(f"Warmup ratio: {warmup_ratio}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Precision: Mixed (BF16 if supported, else FP16)")
        logger.info(f"Logging steps: {logging_steps}")
        logger.info(f"Save steps: {save_steps}")
        logger.info(f"Eval steps: {eval_steps}")
        logger.info(f"Save total limit: {save_total_limit}")
        logger.info(f"Loss masking: Response-only (train_on_responses_only)")
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
                logger.info(f"   Training will resume from this checkpoint\n")
        
        # Verify response masking will work
        self.verify_response_masking(dataset)
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=8,
            packing=False,  # Must be False for train_on_responses_only
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=warmup_ratio,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=logging_steps,
                eval_strategy="steps",
                eval_steps=eval_steps,
                optim="adamw_8bit",
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                seed=42,
                output_dir=self.output_dir,
                report_to=["wandb"],
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                logging_first_step=True,
                load_best_model_at_end=False,  # Memory efficiency
            ),
            callbacks=[ProgressCallback()]
        )
        
        # Apply response-only training - CRITICAL: Must be after SFTTrainer creation
        logger.info("⚙️  Applying train_on_responses_only...")
        trainer = train_on_responses_only(
            trainer,
            instruction_part="### Instruction:",
            response_part="### Response:"
        )
        logger.info("✓ Response-only loss masking enabled\n")
        
        # Train
        logger.info("🚀 Starting training with Response-Only Loss Masking...\n")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
        
        # Save final model
        logger.info(f"\n Saving final model...")
        self.model.save_pretrained(f"{self.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final")
        logger.info(f"✓ Model saved to {self.output_dir}/final\n")
        
        return trainer


def main():
    """Main training pipeline with Unsloth"""
    
    start_time = time.time()
    
    # Initialize W&B with full config
    wandb.init(
        project="nl2sql-training",
        name=f"unsloth-response-only-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "framework": "unsloth",
            "model": "codellama/CodeLlama-7b-hf",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "save_steps": 5000,
            "eval_steps": 5000,
            "max_seq_length": 2048,
            "quantization": "4bit",
            "packing": False,  # Required for response-only
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "loss_masking": "response_only",
            "stopping_strategy": "all_exhausted",
            "total_datasets": 5,
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
    logger.info("NL2SQL Unsloth Training Pipeline (Response-Only Loss)")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"W&B Run: {wandb.run.name}")
    logger.info("="*70 + "\n")
    
    # Define datasets
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
    
    # Log dataset stats to W&B
    wandb.log({
        "train_examples": len(dataset["train"]),
        "val_examples": len(dataset["test"]),
    })
    
    # Train
    logger.info("\n🚀 Starting Unsloth Training\n")
    trainer.train(
        dataset=dataset,
        num_epochs=1,
        batch_size=8,
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
        logging_steps=100,
        save_steps=5000,
        eval_steps=5000,
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_total_limit=2
    )
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    logger.info("\n" + "="*70)
    logger.info("✅ Training Complete!")
    logger.info("="*70)
    logger.info(f"Total training time: {hours}h {minutes}m ({total_time/3600:.2f} hours)")
    logger.info(f"Model saved to: {trainer.output_dir}/final")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70 + "\n")
    
    # Log final metrics to W&B
    wandb.log({
        "total_training_time_hours": total_time / 3600,
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()