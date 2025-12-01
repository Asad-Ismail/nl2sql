"""
Prepare curated datasets for Unsloth training with deduplication and SQL dialect validation.

This script:
1. Loads all training datasets separately
2. Validates SQL dialect consistency (standard SQL only)
3. Globally deduplicates based on (question + SQL) pair with quality priority
4. Saves each dataset separately for weighted sampling during training
5. Creates combined datasets with metadata for tracking
6. Prepares eval set from Spider dev

Output structure allows different weighting per dataset in Unsloth training.
"""

import os
import json
import hashlib
import sqlparse
from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm


class SQLDialectValidator:
    """Validate SQL queries are standard SQL (not DuckDB/other dialects)"""
    
    # DuckDB-specific keywords to reject
    DUCKDB_KEYWORDS = [
        'PRAGMA', 'ATTACH', 'DETACH', 'INSTALL', 'LOAD',
        'DESCRIBE', 'SUMMARIZE', 'UNNEST', 'STRUCT_PACK'
    ]
    
    # PostgreSQL-specific that we want to avoid
    POSTGRES_SPECIFIC = [
        'RETURNING', 'ILIKE', 'SIMILAR TO', '::',
        'LATERAL', 'GENERATE_SERIES'
    ]
    
    @staticmethod
    def is_standard_sql(sql: str) -> Tuple[bool, str]:
        """
        Check if SQL is standard SQL (SQLite/MySQL/PostgreSQL compatible)
        
        Returns:
            (is_valid, reason)
        """
        sql_upper = sql.upper()
        
        # Check for DuckDB-specific keywords
        for keyword in SQLDialectValidator.DUCKDB_KEYWORDS:
            if keyword in sql_upper:
                return False, f"Contains DuckDB keyword: {keyword}"
        
        # Check for PostgreSQL-specific (optional, can be relaxed)
        for keyword in SQLDialectValidator.POSTGRES_SPECIFIC:
            if keyword in sql_upper:
                return False, f"Contains PostgreSQL-specific: {keyword}"
        
        # Basic SQL validation
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, "Failed to parse SQL"
            
            stmt = parsed[0]
            stmt_type = stmt.get_type()
            
            # Accept SELECT, INSERT, UPDATE, DELETE, CREATE
            valid_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
            if stmt_type not in valid_types:
                return False, f"Invalid statement type: {stmt_type}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison (remove whitespace, lowercase, etc.)"""
        # Parse and format
        try:
            parsed = sqlparse.parse(sql)[0]
            formatted = sqlparse.format(
                str(parsed),
                reindent=False,
                keyword_case='upper',
                strip_whitespace=True
            )
            return formatted.strip()
        except:
            # Fallback: basic normalization
            sql = re.sub(r'\s+', ' ', sql)
            return sql.strip().upper()


class DatasetDeduplicator:
    """
    Deduplicate based on QUESTION ONLY (input-based deduplication).
    
    Critical: Prevents conflicting labels (same question -> different SQL).
    Uses priority system to resolve conflicts (Spider > SQaLe > Gretel > SQL-Context > Know-SQL).
    """
    
    # Dataset priority for keeping best version (higher = better quality)
    DATASET_PRIORITY = {
        'spider': 5,      # Benchmark quality - highest priority
        'sqale': 4,       # Real schemas - high priority
        'gretel': 3,      # Synthetic quality - medium
        'sql_context': 2, # Schema-aware - lower
        'know_sql': 1     # Educational - lowest
    }
    
    def __init__(self):
        self.seen_pairs: Dict[str, Dict] = {}  # question_hash -> best example
        self.duplicates_removed = 0
        self.conflicts_resolved = 0  # Same question, different SQL
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for comparison"""
        # Remove extra whitespace, lowercase, remove punctuation
        q = re.sub(r'\s+', ' ', question.lower().strip())
        q = re.sub(r'[^\w\s]', '', q)
        return q
    
    def _hash_question(self, question: str) -> str:
        """
        Create hash of question ONLY (not SQL).
        This prevents conflicting labels for same input.
        """
        normalized_q = self._normalize_question(question)
        return hashlib.md5(normalized_q.encode()).hexdigest()
    
    def _get_priority(self, example: Dict) -> int:
        """Get priority score for example based on source dataset"""
        dataset = example.get('source_dataset', 'unknown')
        return self.DATASET_PRIORITY.get(dataset, 0)
    
    def is_duplicate(self, example: Dict) -> Tuple[bool, str]:
        """
        Check if QUESTION (input) already exists.
        Uses INPUT-ONLY deduplication to prevent conflicting labels.
        
        Critical: If same question has different SQL in different datasets,
        we MUST pick ONE based on priority to avoid gradient confusion.
        
        Returns:
            (is_duplicate, reason)
        """
        question = example.get('question', '')
        sql = example.get('sql', '')
        
        if not question or not sql:
            return True, "Empty question or SQL"
        
        # Hash ONLY the question (not SQL) - this catches conflicts
        q_hash = self._hash_question(question)
        
        # Check if we've seen this QUESTION before (regardless of SQL)
        if q_hash in self.seen_pairs:
            existing = self.seen_pairs[q_hash]
            existing_sql = existing.get('sql', '')
            
            # Normalize both SQLs to check if they're actually different
            try:
                norm_new_sql = SQLDialectValidator.normalize_sql(sql)
                norm_existing_sql = SQLDialectValidator.normalize_sql(existing_sql)
                is_conflict = (norm_new_sql != norm_existing_sql)
            except:
                is_conflict = (sql != existing_sql)
            
            # Compare priorities
            new_priority = self._get_priority(example)
            existing_priority = self._get_priority(existing)
            
            if new_priority > existing_priority:
                # Replace with higher quality version
                self.seen_pairs[q_hash] = example
                conflict_msg = " [CONFLICT RESOLVED]" if is_conflict else ""
                return True, f"Duplicate{conflict_msg} (replaced {existing.get('source_dataset', 'unknown')} with higher priority)"
            elif new_priority == existing_priority:
                # Tie: keep shorter SQL as tiebreaker
                if len(sql) < len(existing_sql):
                    self.seen_pairs[q_hash] = example
                    return True, f"Duplicate (tie - kept shorter SQL from {example.get('source_dataset', 'unknown')})"
                else:
                    return True, f"Duplicate (tie - kept existing from {existing.get('source_dataset', 'unknown')})"
            else:
                # Keep existing (higher priority)
                conflict_msg = " [CONFLICT]" if is_conflict else ""
                return True, f"Duplicate{conflict_msg} (kept higher priority from {existing.get('source_dataset', 'unknown')})"
        
        # Not a duplicate - record it
        self.seen_pairs[q_hash] = example
        
        return False, "Unique"
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics"""
        return {
            'unique_questions': len(self.seen_pairs),
            'duplicates_removed': self.duplicates_removed,
            'conflicts_resolved': self.conflicts_resolved
        }
    
    def get_final_examples(self) -> List[Dict]:
        """Get final deduplicated examples (best version of each pair)"""
        return list(self.seen_pairs.values())


class UnslothDatasetPreparer:
    """Prepare datasets for Unsloth training with global deduplication"""
    
    def __init__(self, data_dir: str = "nl2sql_data", output_dir: str = "nl2sql_data/unsloth"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = SQLDialectValidator()
        self.global_deduplicator = DatasetDeduplicator()  # Global across all datasets
        
        self.stats = {
            'total_loaded': 0,
            'invalid_sql': 0,
            'duplicates': 0,
            'conflicts_resolved': 0,
            'valid_unique': 0,
            'per_dataset': {},
            'deduplication_strategy': 'Input-only (question) with conflict resolution via priority'
        }
    
    def load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file"""
        examples = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return examples
    
    def save_jsonl(self, data: List[Dict], filepath: Path) -> int:
        """Save to JSONL file"""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return len(data)
    
    def process_dataset(
        self, 
        dataset_name: str, 
        input_file: Path,
        validate_dialect: bool = True
    ) -> Tuple[int, int, int]:
        """
        Process single dataset with validation (deduplication is global)
        
        Returns:
            (loaded_count, invalid_count, processed_count)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        # Load data
        examples = self.load_jsonl(input_file)
        print(f"  Loaded: {len(examples):,} examples")
        
        self.stats['total_loaded'] += len(examples)
        self.stats['per_dataset'][dataset_name] = {
            'loaded': len(examples),
            'invalid': 0,
            'duplicate': 0,
            'kept': 0
        }
        
        invalid_count = 0
        processed_count = 0
        
        for ex in tqdm(examples, desc=f"  Processing {dataset_name}", unit="examples"):
            # Ensure required fields
            if 'question' not in ex or 'sql' not in ex:
                invalid_count += 1
                continue
            
            # Validate SQL dialect
            if validate_dialect:
                is_valid, reason = self.validator.is_standard_sql(ex['sql'])
                if not is_valid:
                    invalid_count += 1
                    self.stats['invalid_sql'] += 1
                    continue
            
            # Add dataset tag for tracking (before deduplication)
            ex['source_dataset'] = dataset_name
            
            # Check for global duplicates (may update global deduplicator)
            is_dup, reason = self.global_deduplicator.is_duplicate(ex)
            if is_dup:
                self.stats['duplicates'] += 1
                self.stats['per_dataset'][dataset_name]['duplicate'] += 1
                # Track conflicts specifically
                if '[CONFLICT' in reason:
                    self.global_deduplicator.conflicts_resolved += 1
            
            processed_count += 1
        
        self.stats['per_dataset'][dataset_name]['invalid'] = invalid_count
        
        print(f"  Processed: {processed_count:,} valid examples")
        print(f"  Invalid SQL: {invalid_count:,}")
        print(f"  Duplicates found: {self.stats['per_dataset'][dataset_name]['duplicate']:,}")
        
        return len(examples), invalid_count, processed_count
    
    def prepare_training_datasets(self) -> Dict[str, List[Dict]]:
        """
        Process all training datasets with global deduplication.
        Processes in priority order (high to low) so best examples are kept.
        
        Returns:
            Dict mapping dataset_name -> cleaned_examples
        """
        print("\n" + "="*70)
        print("PREPARING TRAINING DATASETS FOR UNSLOTH")
        print("Strategy: Input-only deduplication (prevents conflicting labels)")
        print("Priority: Spider > SQaLe > Gretel > SQL-Context > Know-SQL")
        print("Conflict resolution: Same question -> Keep highest priority SQL")
        print("="*70)
        
        # Define datasets with priorities (process in priority order)
        datasets = [
            ('spider', {
                'file': self.data_dir / 'train' / 'spider_train.jsonl',
                'priority': 5,
                'description': 'Complex multi-table queries (benchmark quality)'
            }),
            ('sqale', {
                'file': self.data_dir / 'train' / 'sqale.jsonl',
                'priority': 4,
                'description': '22,989 real schemas'
            }),
            ('gretel', {
                'file': self.data_dir / 'train' / 'gretel_train.jsonl',
                'priority': 3,
                'description': 'High-quality synthetic'
            }),
            ('sql_context', {
                'file': self.data_dir / 'train' / 'sql_context_train.jsonl',
                'priority': 2,
                'description': 'Schema-aware queries'
            }),
            ('know_sql', {
                'file': self.data_dir / 'train' / 'know_sql.jsonl',
                'priority': 1,
                'description': 'Educational variety'
            })
        ]
        
        # Sort by priority (highest first)
        datasets.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        for name, config in datasets:
            if not config['file'].exists():
                print(f"\n⚠️  Skipping {name}: file not found")
                continue
            
            self.process_dataset(name, config['file'])
        
        # After processing all datasets, get final deduplicated examples
        print("\n" + "="*70)
        print("EXTRACTING DEDUPLICATED EXAMPLES")
        print("="*70)
        
        all_examples = self.global_deduplicator.get_final_examples()
        self.stats['valid_unique'] = len(all_examples)
        self.stats['conflicts_resolved'] = self.global_deduplicator.conflicts_resolved
        
        print(f"\n  Total unique questions: {len(all_examples):,}")
        print(f"  Conflicts resolved: {self.stats['conflicts_resolved']:,} (same question, different SQL)")
        
        # Group by source dataset for separate files
        dataset_groups = defaultdict(list)
        for ex in tqdm(all_examples, desc="  Grouping by source", unit="examples"):
            source = ex.get('source_dataset', 'unknown')
            dataset_groups[source].append(ex)
        
        # Save each dataset separately
        print(f"\n  Saving by source dataset:")
        for name in [d[0] for d in datasets]:
            if name in dataset_groups:
                examples = dataset_groups[name]
                output_file = self.output_dir / f"{name}_clean.jsonl"
                count = self.save_jsonl(examples, output_file)
                self.stats['per_dataset'][name]['kept'] = count
                print(f"    {name:.<20} {count:>7,} examples -> {output_file.name}")
        
        return dataset_groups
    
    def prepare_eval_dataset(self) -> List[Dict]:
        """Prepare evaluation dataset (Spider dev) - no deduplication with training"""
        print("\n" + "="*70)
        print("PREPARING EVALUATION DATASET (No deduplication)")
        print("="*70)
        
        eval_file = self.data_dir / 'eval' / 'spider_dev.jsonl'
        
        if not eval_file.exists():
            print(f"⚠️  Eval file not found: {eval_file}")
            return []
        
        # Load and validate (no deduplication for eval)
        examples = self.load_jsonl(eval_file)
        print(f"  Loaded: {len(examples):,} examples")
        
        cleaned = []
        invalid_count = 0
        
        for ex in tqdm(examples, desc="  Validating eval", unit="examples"):
            if 'question' not in ex or 'sql' not in ex:
                invalid_count += 1
                continue
            
            # Validate SQL dialect
            is_valid, reason = self.validator.is_standard_sql(ex['sql'])
            if not is_valid:
                invalid_count += 1
                continue
            
            ex['source_dataset'] = 'spider_dev'
            cleaned.append(ex)
        
        print(f"  Valid: {len(cleaned):,} examples")
        print(f"  Invalid: {invalid_count:,} examples")
        
        if cleaned:
            output_file = self.output_dir / 'spider_dev_clean.jsonl'
            count = self.save_jsonl(cleaned, output_file)
            print(f"  ✓ Saved to: {output_file} ({count:,} examples)")
        
        return cleaned
    
    def create_weighted_config(self, datasets: Dict[str, List[Dict]]) -> Dict:
        """
        Create configuration file for Unsloth training with dataset weights
        
        Returns:
            Config dict with recommended weights
        """
        config = {
            'datasets': {},
            'recommended_weights': {},
            'training_config': {
                'total_examples': sum(len(d) for d in datasets.values()),
                'num_datasets': len(datasets),
                'weighted_sampling': True
            }
        }
        
        # Recommended weights based on quality and size
        weight_recommendations = {
            'spider': {
                'weight': 0.5,  # High weight - benchmark quality
                'reason': 'Benchmark quality, complex queries'
            },
            'sqale': {
                'weight': 0.3,  # High weight - real schemas
                'reason': 'Real-world schema diversity'
            },
            'gretel': {
                'weight': 0.15,  # Medium weight - synthetic
                'reason': 'High-quality synthetic data'
            },
            'sql_context': {
                'weight': 0.03,  # Lower weight - supplementary
                'reason': 'Schema-aware supplementary data'
            },
            'know_sql': {
                'weight': 0.02,  # Lowest weight - educational
                'reason': 'Educational variety, lower complexity'
            }
        }
        
        for name, examples in datasets.items():
            config['datasets'][name] = {
                'file': f'nl2sql_data/unsloth/{name}_clean.jsonl',
                'count': len(examples),
                'weight': weight_recommendations.get(name, {}).get('weight', 0.1),
                'reason': weight_recommendations.get(name, {}).get('reason', 'General purpose')
            }
        
        return config
    
    def save_statistics(self):
        """Save processing statistics"""
        stats_file = self.output_dir / 'preparation_stats.json'
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n✓ Statistics saved to: {stats_file}")
    
    def print_summary(self, config: Dict):
        """Print final summary"""
        print("\n" + "="*70)
        print("PREPARATION COMPLETE")
        print("="*70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Strategy: {self.stats['deduplication_strategy']}")
        print(f"  Total loaded: {self.stats['total_loaded']:,}")
        print(f"  Invalid SQL: {self.stats['invalid_sql']:,}")
        print(f"  Duplicates: {self.stats['duplicates']:,}")
        print(f"  Conflicts resolved: {self.stats['conflicts_resolved']:,} (same Q, different SQL)")
        print(f"  Final unique: {self.stats['valid_unique']:,}")
        print(f"  Reduction: {100 * (1 - self.stats['valid_unique'] / self.stats['total_loaded']):.1f}%")
        
        print(f"\n📁 Per-Dataset Contribution (after global deduplication):")
        for name, stats in self.stats['per_dataset'].items():
            if name == 'spider_dev':
                continue
            kept = stats.get('kept', 0)
            loaded = stats['loaded']
            print(f"  {name:.<20} {kept:>7,} kept from {loaded:,} loaded "
                  f"(invalid: {stats['invalid']}, duplicates: {stats['duplicate']})")
        
        print(f"\n⚖️  Recommended Weights for Unsloth Training:")
        for name, info in config['datasets'].items():
            print(f"  {name:.<20} weight={info['weight']:.2f}  "
                  f"({info['reason']})")
        
        print(f"\n📂 Output Files:")
        print(f"  Directory: {self.output_dir}/")
        print(f"  Training datasets: *_clean.jsonl (separated for weighting)")
        print(f"  Eval dataset: spider_dev_clean.jsonl")
        print(f"  Config: unsloth_config.json")
        print(f"  Stats: preparation_stats.json")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Review weights in unsloth_config.json")
        print(f"  2. Adjust weights based on your training goals")
        print(f"  3. Use separate files for weighted sampling in Unsloth")
        print(f"  4. Load datasets with: load_dataset('json', data_files='path')")


def main():
    """Main execution"""
    preparer = UnslothDatasetPreparer()
    
    # Process training datasets (keeps them separate)
    train_datasets = preparer.prepare_training_datasets()
    
    # Process eval dataset
    eval_dataset = preparer.prepare_eval_dataset()
    
    # Create weighted configuration
    config = preparer.create_weighted_config(train_datasets)
    
    # Save config
    config_file = preparer.output_dir / 'unsloth_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Config saved to: {config_file}")
    
    # Save statistics
    preparer.save_statistics()
    
    # Print summary
    preparer.print_summary(config)
    
    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
