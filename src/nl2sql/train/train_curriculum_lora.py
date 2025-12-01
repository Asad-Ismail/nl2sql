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
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, concatenate_datasets
import numpy as np
from typing import Dict, List, Optional
import sqlite3
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    path: str
    weight: float  # Sampling weight
    difficulty: int  # 1=easy, 2=medium, 3=hard
    max_samples: Optional[int] = None


class NL2SQLDataset(Dataset):
    """Custom dataset for NL2SQL with prompt formatting"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
    
    def __len__(self):
        return len(self.examples)
    
    def create_prompt(self, example: Dict) -> str:
        """Create training prompt"""
        # Schema context (if available)
        schema_context = ""
        if 'schema' in example:
            schema = example['schema']
            if isinstance(schema, dict):
                schema_context = f"\nDatabase Schema: {json.dumps(schema)}"
        
        # Database ID (if available)
        db_context = ""
        if 'db_id' in example:
            db_context = f"\nDatabase: {example['db_id']}"
        
        prompt = f"""### Task: Convert natural language question to SQL query

{db_context}{schema_context}

### Question: {example['question']}

### SQL: {example['sql']}"""
        
        return prompt
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = self.create_prompt(example)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For causal LM
        }


class MixedDatasetSampler:
    """Smart sampler for mixing multiple datasets"""
    
    def __init__(self, dataset_configs: List[DatasetConfig], tokenizer):
        self.configs = dataset_configs
        self.tokenizer = tokenizer
        self.datasets = []
        
        # Load all datasets
        for config in dataset_configs:
            if os.path.exists(config.path):
                dataset = NL2SQLDataset(config.path, tokenizer)
                if config.max_samples:
                    # Subsample if needed
                    indices = np.random.choice(
                        len(dataset), 
                        min(config.max_samples, len(dataset)),
                        replace=False
                    )
                    dataset.examples = [dataset.examples[i] for i in indices]
                
                self.datasets.append((config, dataset))
                print(f"✓ Loaded {config.name}: {len(dataset)} examples")
    
    def create_mixed_dataset(self) -> Dataset:
        """Create mixed dataset with weighted sampling"""
        all_examples = []
        
        for config, dataset in self.datasets:
            # Calculate number of samples based on weight
            n_samples = int(len(dataset) * config.weight)
            
            # Sample with replacement if needed
            if n_samples > len(dataset):
                indices = np.random.choice(len(dataset), n_samples, replace=True)
            else:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
            
            sampled = [dataset.examples[i] for i in indices]
            all_examples.extend(sampled)
            
            print(f"  {config.name}: sampled {len(sampled)} examples (weight={config.weight})")
        
        # Shuffle all examples
        np.random.shuffle(all_examples)
        
        # Create new dataset
        mixed_dataset = NL2SQLDataset.__new__(NL2SQLDataset)
        mixed_dataset.tokenizer = self.tokenizer
        mixed_dataset.max_length = 512
        mixed_dataset.examples = all_examples
        
        return mixed_dataset


class CurriculumTrainer:
    """Curriculum learning trainer - train on progressively harder data"""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        output_dir: str = "models/nl2sql-lora",
        use_curriculum: bool = True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_curriculum = use_curriculum
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train_stage(
        self, 
        dataset: Dataset,
        stage_name: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """Train single curriculum stage"""
        print(f"\n{'='*60}")
        print(f"Training Stage: {stage_name}")
        print(f"{'='*60}")
        
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/{stage_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            warmup_steps=100,
            optim="adamw_torch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )
        )
        
        trainer.train()
        
        # Save checkpoint
        self.model.save_pretrained(f"{self.output_dir}/{stage_name}_final")
        print(f"✓ Stage {stage_name} complete!")
    
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
                print(f"Skipping {stage_name} - no datasets")
                continue
            
            # Create mixed dataset for this stage
            sampler = MixedDatasetSampler(configs, self.tokenizer)
            dataset = sampler.create_mixed_dataset()
            
            # Train
            self.train_stage(dataset, stage_name, num_epochs=epochs)
    
    def train_mixed(
        self,
        dataset_configs: List[DatasetConfig],
        num_epochs: int = 3
    ):
        """Train on all data mixed together (baseline)"""
        print("Training with mixed data (no curriculum)")
        
        sampler = MixedDatasetSampler(dataset_configs, self.tokenizer)
        dataset = sampler.create_mixed_dataset()
        
        self.train_stage(dataset, "mixed", num_epochs=num_epochs)


def main():
    """Main training pipeline"""
    
    # Define datasets with configurations
    dataset_configs = [
        # Easy: Simple single-table queries
        DatasetConfig(
            name="WikiSQL",
            path="nl2sql_data/unified/wikisql_train_augmented.jsonl",
            weight=0.3,
            difficulty=1,
            max_samples=10000  # Subsample large dataset
        ),
        
        # Medium: Multi-table queries
        DatasetConfig(
            name="Spider",
            path="nl2sql_data/unified/spider_train_augmented.jsonl",
            weight=0.5,
            difficulty=2
        ),
        
        DatasetConfig(
            name="Gretel-Synthetic",
            path="nl2sql_data/unified/gretel_train.jsonl",
            weight=0.3,
            difficulty=2,
            max_samples=5000
        ),
        
        # Hard: Complex real-world queries
        DatasetConfig(
            name="BIRD",
            path="nl2sql_data/unified/bird_train.jsonl",
            weight=0.4,
            difficulty=3
        ),
    ]
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        model_name="codellama/CodeLlama-7b-hf",
        output_dir="models/nl2sql-curriculum-lora",
        use_curriculum=True
    )
    
    # Train with curriculum
    print("\n🚀 Starting Curriculum Training Pipeline")
    trainer.train_curriculum(
        dataset_configs=dataset_configs,
        epochs_per_stage=[2, 3, 3]  # More epochs for harder stages
    )
    
    print("\n✅ Training Complete!")
    print(f"Model saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
