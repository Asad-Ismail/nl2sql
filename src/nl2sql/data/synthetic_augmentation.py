"""
Synthetic Data Generation for NL2SQL Training

Novel approach: Generate realistic variations of existing queries to expand training data
- Paraphrase questions (same SQL, different wording)
- Schema shuffling (change table/column names, keep SQL logic)
- Query augmentation (add/remove conditions while maintaining validity)
- Backtranslation (SQL -> NL -> SQL)
"""

import json
import random
import re
from typing import Dict, List, Tuple
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML


class SQLValidator:
    """Validate generated SQL queries"""
    
    @staticmethod
    def is_valid_sql(sql: str) -> bool:
        """Basic SQL syntax validation"""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False
            # Check has SELECT statement
            stmt = parsed[0]
            return stmt.get_type() == 'SELECT'
        except:
            return False
    
    @staticmethod
    def extract_tables(sql: str) -> List[str]:
        """Extract table names from SQL"""
        parsed = sqlparse.parse(sql)[0]
        tables = []
        from_seen = False
        
        for token in parsed.tokens:
            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.append(str(identifier))
                elif isinstance(token, Identifier):
                    tables.append(str(token))
                from_seen = False
            elif token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
        
        return tables


class QuestionParaphraser:
    """Generate paraphrases of natural language questions"""
    
    # Simple rule-based paraphrasing templates
    PATTERNS = [
        # What X -> Show me X
        (r"What (is|are) the (.+)\?", r"Show me the \2."),
        (r"What (.+)\?", r"List the \1."),
        
        # How many -> Count
        (r"How many (.+)\?", r"Count the \1."),
        (r"How many (.+) are there\?", r"What is the total number of \1?"),
        
        # Find -> Get/Retrieve
        (r"Find (.+)\.", r"Get \1."),
        (r"Find (.+)\.", r"Retrieve \1."),
        
        # List -> Show
        (r"List (.+)\.", r"Show \1."),
        (r"List all (.+)\.", r"Display all \1."),
        
        # Give me -> Show me
        (r"Give me (.+)\.", r"Show me \1."),
        
        # Which -> What
        (r"Which (.+)\?", r"What \1?"),
    ]
    
    @staticmethod
    def paraphrase(question: str, n: int = 3) -> List[str]:
        """Generate n paraphrases of the question"""
        paraphrases = [question]  # Include original
        
        for pattern, replacement in QuestionParaphraser.PATTERNS:
            new_q = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
            if new_q != question and new_q not in paraphrases:
                paraphrases.append(new_q)
            
            if len(paraphrases) >= n + 1:
                break
        
        return paraphrases[:n + 1]


class SchemaShuffler:
    """Shuffle schema names to create variations"""
    
    # Generic name replacements
    TABLE_REPLACEMENTS = {
        'student': ['pupil', 'learner', 'scholar'],
        'teacher': ['instructor', 'professor', 'educator'],
        'course': ['class', 'subject', 'lesson'],
        'department': ['division', 'unit', 'section'],
        'employee': ['worker', 'staff', 'personnel'],
        'customer': ['client', 'buyer', 'patron'],
        'order': ['purchase', 'transaction', 'sale'],
        'product': ['item', 'goods', 'merchandise'],
    }
    
    @staticmethod
    def shuffle_schema(sql: str, schema: Dict, seed: int = None) -> Tuple[str, Dict]:
        """Create schema variation with shuffled names"""
        if seed:
            random.seed(seed)
        
        # Create mapping of old -> new names
        mapping = {}
        new_schema = {}
        
        for table_name in schema.get('table_names', []):
            table_lower = table_name.lower()
            replacements = SchemaShuffler.TABLE_REPLACEMENTS.get(table_lower, [table_lower])
            new_name = random.choice(replacements) if replacements else table_name
            mapping[table_name] = new_name
            new_schema[new_name] = schema.get(table_name, {})
        
        # Apply mapping to SQL
        new_sql = sql
        for old, new in mapping.items():
            # Use word boundaries to avoid partial replacements
            new_sql = re.sub(r'\b' + re.escape(old) + r'\b', new, new_sql, flags=re.IGNORECASE)
        
        return new_sql, new_schema


class QueryAugmenter:
    """Augment SQL queries with valid variations"""
    
    @staticmethod
    def add_limit(sql: str) -> str:
        """Add LIMIT clause if not present"""
        if 'LIMIT' not in sql.upper():
            limits = [5, 10, 20, 50, 100]
            return f"{sql.rstrip(';')} LIMIT {random.choice(limits)}"
        return sql
    
    @staticmethod
    def add_order_by(sql: str, columns: List[str]) -> str:
        """Add ORDER BY if not present"""
        if 'ORDER BY' not in sql.upper() and columns:
            col = random.choice(columns)
            direction = random.choice(['ASC', 'DESC'])
            return f"{sql.rstrip(';')} ORDER BY {col} {direction}"
        return sql
    
    @staticmethod
    def simplify_query(sql: str) -> str:
        """Remove optional clauses to create simpler version"""
        # Remove ORDER BY
        sql = re.sub(r'\s+ORDER BY\s+[^;]+', '', sql, flags=re.IGNORECASE)
        # Remove LIMIT
        sql = re.sub(r'\s+LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
        return sql


class SyntheticDataGenerator:
    """Main class to generate synthetic training data"""
    
    def __init__(self):
        self.validator = SQLValidator()
        self.paraphraser = QuestionParaphraser()
        self.shuffler = SchemaShuffler()
        self.augmenter = QueryAugmenter()
    
    def generate_variations(self, example: Dict, n_variations: int = 5) -> List[Dict]:
        """
        Generate n variations of a single example
        
        Args:
            example: {'question': str, 'sql': str, 'db_id': str, 'schema': dict}
            n_variations: number of variations to generate
            
        Returns:
            List of augmented examples
        """
        variations = []
        
        # Original example
        variations.append(example)
        
        # 1. Question paraphrasing (same SQL)
        paraphrases = self.paraphraser.paraphrase(example['question'], n=2)
        for para in paraphrases[1:]:  # Skip original
            variations.append({
                **example,
                'question': para,
                'augmentation': 'paraphrase'
            })
        
        # 2. Schema shuffling
        try:
            new_sql, new_schema = self.shuffler.shuffle_schema(
                example['sql'], 
                example.get('schema', {}),
                seed=hash(example['question'])
            )
            if self.validator.is_valid_sql(new_sql):
                # Update question with new schema names (simplified)
                new_question = example['question']
                variations.append({
                    'question': new_question,
                    'sql': new_sql,
                    'db_id': example['db_id'] + '_shuffled',
                    'schema': new_schema,
                    'augmentation': 'schema_shuffle'
                })
        except:
            pass
        
        # 3. Query augmentation - add clauses
        if 'schema' in example:
            columns = list(example.get('schema', {}).get('columns', []))
            
            # Add LIMIT
            sql_with_limit = self.augmenter.add_limit(example['sql'])
            if self.validator.is_valid_sql(sql_with_limit):
                variations.append({
                    **example,
                    'sql': sql_with_limit,
                    'question': example['question'] + ' (top results)',
                    'augmentation': 'add_limit'
                })
            
            # Add ORDER BY
            if columns:
                sql_with_order = self.augmenter.add_order_by(example['sql'], columns)
                if self.validator.is_valid_sql(sql_with_order):
                    variations.append({
                        **example,
                        'sql': sql_with_order,
                        'question': example['question'] + ' (sorted)',
                        'augmentation': 'add_order'
                    })
        
        # 4. Simplification
        simplified_sql = self.augmenter.simplify_query(example['sql'])
        if simplified_sql != example['sql'] and self.validator.is_valid_sql(simplified_sql):
            variations.append({
                **example,
                'sql': simplified_sql,
                'augmentation': 'simplify'
            })
        
        return variations[:n_variations + 1]
    
    def augment_dataset(self, input_file: str, output_file: str, 
                       multiplier: int = 3, validate: bool = True):
        """
        Augment entire dataset
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            multiplier: How many variations per example (3x = 300% increase)
            validate: Whether to validate SQL before adding
        """
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            
            total_original = 0
            total_generated = 0
            
            for line in f_in:
                example = json.loads(line)
                total_original += 1
                
                # Generate variations
                variations = self.generate_variations(example, n_variations=multiplier)
                
                # Write all variations
                for var in variations:
                    # Validate SQL if requested
                    if validate and not self.validator.is_valid_sql(var['sql']):
                        continue
                    
                    f_out.write(json.dumps(var) + '\n')
                    total_generated += 1
                
                # Progress
                if total_original % 100 == 0:
                    print(f"Processed {total_original} examples, generated {total_generated} total")
        
        print(f"\n✓ Complete!")
        print(f"  Original examples: {total_original}")
        print(f"  Generated examples: {total_generated}")
        print(f"  Expansion rate: {total_generated/total_original:.1f}x")


def main():
    """Example usage"""
    import os
    
    generator = SyntheticDataGenerator()
    
    # Example: Augment WikiSQL
    data_dir = "nl2sql_data/unified"
    
    if os.path.exists(f"{data_dir}/wikisql_train.jsonl"):
        print("Augmenting WikiSQL dataset...")
        generator.augment_dataset(
            input_file=f"{data_dir}/wikisql_train.jsonl",
            output_file=f"{data_dir}/wikisql_train_augmented.jsonl",
            multiplier=3
        )
    
    # Example: Augment Spider
    if os.path.exists(f"{data_dir}/spider_train.jsonl"):
        print("\nAugmenting Spider dataset...")
        generator.augment_dataset(
            input_file=f"{data_dir}/spider_train.jsonl",
            output_file=f"{data_dir}/spider_train_augmented.jsonl",
            multiplier=3
        )


if __name__ == "__main__":
    main()
