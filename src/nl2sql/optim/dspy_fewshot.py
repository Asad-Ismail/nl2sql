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
    max_tokens=256,
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
def execute_sql(sql: str, db_path: str):
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except:
        return None

def metric(example, prediction, trace=None):
    if not hasattr(prediction, 'sql') or not prediction.sql:
        return 0.0
    
    # Clean the generated SQL
    pred_sql = prediction.sql.replace("```sql", "").replace("```", "").strip()
    if pred_sql.startswith("]]"):
        pred_sql = pred_sql[2:].strip()
    pred_sql = pred_sql.lstrip('\n').strip()

    print("Question")
    print(f"*"*100)
    print(f"\n❔ {example.question}")
    print(f"*"*100)
    print("GT")
    print(f"{example.sql}")
    print(f"*"*100)
    print("Prediction")
    print(f"*"*100)
    print(f"🤖 {pred_sql}")
    print(f"*"*100)
    
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
        return 0.0

# ==============================================================================
# Data
# ==============================================================================
def load_data():
    train = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_clean.jsonl", split="train")
    dev = load_dataset("AsadIsmail/nl2sql-deduplicated", data_files="spider_dev_clean.jsonl", split="train")
    
    def convert(dataset, limit):
        examples = []
        for i, item in enumerate(dataset.shuffle(seed=42)):
            if i >= limit:
                break
            
            db_id = item['db_id']
            # Get REAL schema from tables.json instead of item['context']
            schema = SCHEMAS.get(db_id, f"Database: {db_id}")
            
            examples.append(dspy.Example(
                db_schema=schema,  # Use real CREATE TABLE schema
                question=item['question'],
                sql=item['sql'],
                db_id=db_id
            ).with_inputs('db_schema', 'question'))
        return examples
    print(f"Using train data of size {len(train)//10} and test on complete dev set {len(dev)}")
    return convert(train, len(train)//10), convert(dev, len(dev))


def evaluate(module, devset):
    """Evaluate module on devset and return average score"""
    scores = []
    for example in devset:
        prediction = module(db_schema=example.db_schema, question=example.question)
        score = metric(example, prediction)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ==============================================================================
# Main
# ==============================================================================
def main():
    logger.info("Loading schemas...")
    logger.info(f"Loaded {len(SCHEMAS)} database schemas")
    
    logger.info("Loading data...")
    trainset, devset = load_data()
    logger.info(f"Loaded {len(trainset)} train, {len(devset)} dev")
    
    # Evaluate baseline (before optimization)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING BASELINE (Before Optimization)")
    logger.info("="*80)
    baseline = SQLModule()
    baseline_score = evaluate(baseline, devset)
    logger.info(f"Baseline score is {baseline_score}")
    
    logger.info("Starting optimization...")
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=1,
        num_candidate_programs=3
    )
    
    compiled = optimizer.compile(SQLModule(), trainset=trainset, valset=devset)

    # Evaluate optimized model (after optimization)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING OPTIMIZED MODEL (After Optimization)")
    logger.info("="*80)
    optimized_score = evaluate(compiled, devset)
    
    # Print final comparison
    logger.info("\n" + "="*100)
    logger.info("FINAL RESULTS")
    logger.info("="*100)
    logger.info(f"📊 Baseline Score (Before):  {baseline_score:.2%} ({baseline_score:.4f})")
    logger.info(f"📊 Optimized Score (After):  {optimized_score:.2%} ({optimized_score:.4f})")
    logger.info(f"📈 Improvement:              {(optimized_score - baseline_score):.2%} ({optimized_score - baseline_score:+.4f})")
    logger.info("="*100)
    
    logger.info("\nSaving optimized model...")
    os.makedirs("models/dspy_optimized", exist_ok=True)
    compiled.save("models/dspy_optimized/nl2sql.json")
    logger.info("Done!")
    

if __name__ == "__main__":
    main()