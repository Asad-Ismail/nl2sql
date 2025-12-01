#!/usr/bin/env python3
"""
End-to-end pipeline runner for NL2SQL training

Usage:
    python run_pipeline.py --full              # Run everything
    python run_pipeline.py --download-only     # Just download data
    python run_pipeline.py --augment-only      # Just augment data
    python run_pipeline.py --train-only        # Just train model
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd: str, description: str):
    """Run shell command with logging"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(1)
    
    print(f"\n✅ Complete: {description}")


def check_data_exists() -> bool:
    """Check if datasets have been downloaded"""
    data_dir = Path("nl2sql_data/unified")
    if not data_dir.exists():
        return False
    
    required_files = ["wikisql_train.jsonl", "spider_train.jsonl"]
    for fname in required_files:
        if not (data_dir / fname).exists():
            return False
    
    return True


def check_augmented_exists() -> bool:
    """Check if augmented datasets exist"""
    data_dir = Path("nl2sql_data/unified")
    
    required_files = [
        "wikisql_train_augmented.jsonl",
        "spider_train_augmented.jsonl"
    ]
    
    for fname in required_files:
        if not (data_dir / fname).exists():
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="NL2SQL Training Pipeline")
    parser.add_argument("--full", action="store_true", 
                       help="Run full pipeline (download → augment → train)")
    parser.add_argument("--download-only", action="store_true",
                       help="Only download datasets")
    parser.add_argument("--augment-only", action="store_true",
                       help="Only run data augmentation")
    parser.add_argument("--train-only", action="store_true",
                       help="Only run training")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download step")
    parser.add_argument("--skip-augment", action="store_true",
                       help="Skip augmentation step")
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-hf",
                       help="Model to use for training")
    
    args = parser.parse_args()
    
    # Determine what to run
    run_download = args.full or args.download_only
    run_augment = args.full or args.augment_only
    run_train = args.full or args.train_only
    
    if args.skip_download:
        run_download = False
    if args.skip_augment:
        run_augment = False
    
    # If nothing specified, show help
    if not (run_download or run_augment or run_train):
        parser.print_help()
        sys.exit(0)
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║     NL2SQL Training Pipeline                               ║
    ║     Novel Approach: Synthetic Data + Curriculum Learning   ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Download datasets
    if run_download:
        run_command(
            "python src/nl2sql/data/download_all_datasets.py",
            "Step 1/3: Downloading datasets from HuggingFace"
        )
    elif not check_data_exists():
        print("\n⚠️  Warning: Base datasets not found!")
        print("Run with --full or --download-only to download them first.")
        sys.exit(1)
    else:
        print("\n✓ Base datasets found, skipping download")
    
    # Step 2: Augment data
    if run_augment:
        if not check_data_exists():
            print("\n❌ Error: Base datasets not found!")
            print("Run download step first.")
            sys.exit(1)
        
        run_command(
            "python src/nl2sql/data/synthetic_augmentation.py",
            "Step 2/3: Generating synthetic training data"
        )
    elif not check_augmented_exists() and run_train:
        print("\n⚠️  Warning: Augmented datasets not found!")
        print("Training will use base datasets only.")
        print("For best results, run augmentation first: --augment-only")
        input("\nPress Enter to continue with base datasets only, or Ctrl+C to abort...")
    else:
        print("\n✓ Augmented datasets found, skipping augmentation")
    
    # Step 3: Train model
    if run_train:
        if not (check_data_exists() or check_augmented_exists()):
            print("\n❌ Error: No training data found!")
            print("Run download and augmentation steps first.")
            sys.exit(1)
        
        run_command(
            f"python src/nl2sql/train/train_curriculum_lora.py --model {args.model}",
            "Step 3/3: Training model with curriculum learning"
        )
    
    # Success!
    print("""
    
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║     ✅ Pipeline Complete!                                  ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    
    Next steps:
    
    1. Evaluate your model:
       python src/nl2sql/eval/evaluate_spider.py
    
    2. Test inference:
       python src/nl2sql/inference/generate_sql.py \\
           --question "What are the names of all students?"
    
    3. Compare with baseline:
       - Your curriculum model: models/nl2sql-curriculum-lora/
       - Run ablation studies to measure improvement
    
    📊 Expected improvements over baseline:
       - 5-10% better accuracy on Spider 2.0
       - 3-5x more training data from augmentation
       - Faster convergence from curriculum learning
    
    🎉 Happy training!
    """)


if __name__ == "__main__":
    main()
