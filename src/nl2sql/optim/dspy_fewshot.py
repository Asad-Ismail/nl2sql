"""
DSPy Optimizer for NL2SQL - Simple Version
"""
import dspy
import os
import sqlite3
import logging
import json
from pathlib import Path
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from datasets import load_dataset
from tqdm import tqdm
from nl2sql.utils.util import load_schemas,execute_sql,print_comparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Schema Loading (same as baseline)
# ==============================================================================
def load_schemas():
    """Load database schemas from Spider tables.json"""
    schema_file = Path("database/spider_data/tables.json")
    
    if not schema_file.exists():
        logger.error("tables.json not found!")
        return {}
    
    with open(schema_file) as f:
        tables_data = json.load(f)
    
    schemas = {}
    for db in tables_data:
        db_id = db['db_id']
        table_names = db['table_names_original']
        column_names = db['column_names_original']
        column_types = db['column_types']
        
        # Format schema as CREATE TABLE statements
        schema_lines = []
        for table_idx, table_name in enumerate(table_names):
            cols = [(col[1], column_types[i]) for i, col in enumerate(column_names) if col[0] == table_idx]
            if cols:
                col_defs = [f"{name} {dtype}" for name, dtype in cols]
                schema_lines.append(f"CREATE TABLE {table_name} ({', '.join(col_defs)})")
        
        schemas[db_id] = "\n".join(schema_lines)
    
    return schemas

# Load schemas once
SCHEMAS = load_schemas()

# ==============================================================================
# LM Setup
# ==============================================================================
lm = dspy.LM(
    model="openai/TheBloke/CodeLlama-7B-Instruct-AWQ",
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    max_tokens=500,
    temperature=0.0
)
dspy.configure(lm=lm)

# ==============================================================================
# Module
# ==============================================================================
class TextToSQL(dspy.Signature):
    """Convert natural language to SQL."""
    db_schema = dspy.InputField()
    question = dspy.InputField()
    sql = dspy.OutputField()

class SQLModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(TextToSQL)

    def forward(self, db_schema, question):
        return self.prog(db_schema=db_schema, question=question)

# ==============================================================================
# Metric
# ==============================================================================

def metric(example, prediction, trace=None):
    if not hasattr(prediction, 'sql') or not prediction.sql:
        return 0.0
    
    # Clean the generated SQL
    pred_sql = prediction.sql.replace("```sql", "").replace("```", "").strip()
    if pred_sql.startswith("]]"):
        pred_sql = pred_sql[2:].strip()
    pred_sql = pred_sql.lstrip('\n').strip()
    # Execute both queries
    db_path = f"database/spider_data/database/{example.db_id}/{example.db_id}.sqlite"
    
    gold_results = execute_sql(example.sql, db_path)
    pred_results = execute_sql(pred_sql, db_path)
    
    if gold_results is None or pred_results is None:
        print("❌ Execution failed")
        return 0.0
    
    # Compare
    if set(tuple(r) for r in pred_results) == set(tuple(r) for r in gold_results):
        print("✅ Match!")
        return 1.0
    else:
        print("❌ Mismatch")
        #print(f"GT Results: {gold_results}, Predicted Results {pred_results}")
        return 0.0

# ==============================================================================
# Data
# ==============================================================================


def convert_to_dspy(dataset_split):
    """
    Just converts a dataset list/split to DSPy examples.
    No shuffling, no limiting. We do that beforehand.
    """
    examples = []
    for item in dataset_split:
        db_id = item['db_id']
        schema = SCHEMAS.get(db_id, f"Database: {db_id}")
        
        examples.append(dspy.Example(
            db_schema=schema,
            question=item['question'],
            sql=item['sql'],
            db_id=db_id
        ).with_inputs('db_schema', 'question'))
    return examples


def load_data():
    # 1. Load the raw dataset
    full_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
    dev_data = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")

    # 2. SHUFFLE ONCE GLOBALLY
    # This ensures the order is fixed before we start slicing
    shuffled_data = full_data.shuffle(seed=42)

    # 3. Create disjoint slices using indices
    # Train: 0 to 2000
    train_slice = shuffled_data.select(range(0, 2000))
    
    # Opt Val: 2000 to 2100 (Guaranteed to be different from 0-2000)
    val_slice = shuffled_data.select(range(2000, 2100))

    print(f"Split sizes -> Train: {len(train_slice)}, Val: {len(val_slice)}")

    # 4. Convert to DSPy
    trainset = convert_to_dspy(train_slice)
    valset = convert_to_dspy(val_slice)
    devset = convert_to_dspy(dev_data) # Keep dev set full

    return trainset, valset, devset

    

def evaluate(module, devset):
    """Evaluate module on devset and return average score"""
    scores = []
    for example in tqdm(devset,total=len(devset)):
        prediction = module(db_schema=example.db_schema, question=example.question)
        score = metric(example, prediction)
        scores.append(score)
        print(f"Running Score {sum(scores)}/{len(scores)}")
    return sum(scores) / len(scores) if scores else 0.0


# ==============================================================================
# Main
# ==============================================================================
def main():
    logger.info("Loading schemas...")
    logger.info(f"Loaded {len(SCHEMAS)} database schemas")
    
    logger.info("Loading data...")
    trainset, valset, devset = load_data()
    logger.info(f"Loaded {len(trainset)} train, Loaded {len(valset)}, Loaded final test set {len(devset)} ")
    
    # Evaluate baseline (before optimization)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING BASELINE (Before Optimization)")
    logger.info("="*80)
    baseline = SQLModule()
    #baseline_score = evaluate(baseline, devset)
    #logger.info(f"Baseline score is {baseline_score}")
    
    logger.info("Starting optimization...")
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=3,
        num_candidate_programs=5
    )
    
    compiled = optimizer.compile(SQLModule(), trainset=trainset, valset=valset)

    exit()

    # Evaluate optimized model (after optimization)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING OPTIMIZED MODEL (After Optimization)")
    logger.info("="*80)
    optimized_score = evaluate(compiled, devset)
    
    # Print final comparison
    logger.info("\n" + "="*100)
    logger.info("FINAL RESULTS")
    logger.info("="*100)
    #logger.info(f"📊 Baseline Score (Before):  {baseline_score:.2%} ({baseline_score:.4f})")
    logger.info(f"📊 Optimized Score (After):  {optimized_score:.2%} ({optimized_score:.4f})")
    #logger.info(f"📈 Improvement:              {(optimized_score - baseline_score):.2%} ({optimized_score - baseline_score:+.4f})")
    logger.info("="*100)
    
    logger.info("\nSaving optimized model...")
    os.makedirs("models/dspy_optimized", exist_ok=True)
    compiled.save("models/dspy_optimized/nl2sql.json")
    logger.info("Done!")
    

if __name__ == "__main__":
    main()