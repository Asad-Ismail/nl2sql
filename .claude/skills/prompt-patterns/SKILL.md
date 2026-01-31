---
name: prompt-patterns
description: Prompt engineering patterns for NL2SQL. Use when creating prompts for SQL generation, few-shot learning, or self-correction. Covers zero-shot, few-shot, self-correction, and optimization patterns (DSPy, TextGrad).
---

# Prompt Engineering Patterns for NL2SQL

## When to Use
- Creating prompts for LLM-based SQL generation
- Designing few-shot examples for text-to-SQL
- Implementing self-correction workflows
- Optimizing prompts with DSPy or TextGrad
- Evaluating SQL quality with LLMs

## Core Patterns

### Zero-Shot Prompt Format

Standard zero-shot prompt for NL2SQL:

```python
def create_zero_shot_prompt(question: str, schema: str) -> str:
    """Create zero-shot prompt for SQL generation"""
    return f"""### Task: Convert the following natural language question to a SQL query. Give only SQL Query as Output

### Database Schema:
{schema}

### Question: {question}

### SQL Query:
"""
```

**Key elements:**
- Clear task specification with "Give only SQL Query as Output"
- Schema provided before question
- Explicit markers: `### Task:`, `### Database Schema:`, `### Question:`, `### SQL Query:`
- No examples, single-shot generation

### Few-Shot Prompt Format

Provide 2-3 similar examples for better performance:

```python
def create_few_shot_prompt(question: str, schema: str, examples: List[Dict]) -> str:
    """Create few-shot prompt with similar examples"""
    prompt = "### Task: Convert natural language questions to SQL queries. Give only SQL Query as Output\n\n"
    prompt += "### Examples:\n\n"

    # Add 2-3 similar examples
    for i, ex in enumerate(examples[:2], 1):
        prompt += f"**Example {i}:**\n"
        prompt += f"Schema: {ex.get('context', 'N/A')}\n"
        prompt += f"Question: {ex['question']}\n"
        prompt += f"SQL: {ex.get('query', ex.get('sql', ''))}\n\n"

    # Add target question
    prompt += f"### Now convert this question:\n"
    prompt += f"Schema:\n{schema}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"SQL Query:\n"

    return prompt
```

**Example selection strategy:**
- Select examples from same database if possible
- Match SQL patterns (JOIN, aggregation, subquery)
- Prefer similar complexity level
- Use 2-3 examples (more adds noise)

### Self-Correction Workflow

Generate → Validate → Fix → Compare (3 attempts max):

```python
def self_correction(question: str, schema: str, db_path: str,
                    generate_func, execute_func, validate_func) -> Dict:
    """Self-correction with up to 3 attempts"""

    sql = None
    is_valid = False
    num_attempts = 0
    max_attempts = 3
    error = None

    for attempt in range(1, max_attempts + 1):
        num_attempts = attempt

        # Attempt 1: Generate from scratch
        if attempt == 1:
            prompt = create_zero_shot_prompt(question, schema)
            sql = generate_func(prompt, max_new_tokens=250)

        # Attempt 2+: Fix based on error message
        else:
            fix_prompt = f"""### Task: Fix the SQL query based on the error message

### Database Schema:
{schema}

### Question: {question}

### Previous SQL (has error):
{sql}

### Error Message:
{error}

### Fixed SQL:
"""
            sql = generate_func(fix_prompt, max_new_tokens=250)

        # Execute SQL
        is_valid, error, results = execute_func(sql, db_path)

        if is_valid:
            # LLM validation for semantic correctness
            validation = validate_func(question, sql, schema)
            # Continue if validation fails, stop if passes
            if "Yes" in validation or "correct" in validation.lower():
                break

    return {
        "sql": sql,
        "is_valid": is_valid,
        "error": error,
        "results": results if is_valid else None,
        "num_attempts": num_attempts
    }
```

**Expected improvement:**
- Zero-shot: 40-50% valid SQL
- Few-shot: 55-65% valid SQL
- Self-correction (3 attempts): 65-75% valid SQL

### LLM Validation Prompt

Ask LLM if SQL correctly answers the question:

```python
def create_validation_prompt(question: str, sql: str, schema: str) -> str:
    """Create prompt for LLM-based SQL validation"""
    return f"""-- Database Schema
{schema}

-- Question: {question}
-- SQL Query: {sql}

Does this SQL query correctly answer the question? Answer with 'Yes' or 'No' followed by a brief explanation.
Answer:"""
```

**Usage:**
```python
from openai import OpenAI

response = client.chat.completions.create(
    model="codellama/CodeLlama-7b-hf",
    messages=[{"role": "user", "content": validation_prompt}],
    max_tokens=100,
    temperature=0.1
)

validation = response.choices[0].message.content
# Returns: "Yes, the query correctly selects..." or "No, the query is missing..."
```

## Optimization Patterns

### DSPy Prompt Optimization

DSPy optimizes few-shot examples automatically:

```python
import dspy

class NL2SQLSignature(dspy.Signature):
    """Input/output signature for NL2SQL"""
    schema = dspy.InputField(desc="Database schema")
    question = dspy.InputField(desc="Natural language question")
    sql = dspy.OutputField(desc="SQL query")

class NL2SQLModule(dspy.Module):
    """DSPy module for text-to-SQL"""

    def forward(self, schema: str, question: str):
        # Generate few-shot examples with DSPy optimizer
        predict = dspy.Predict(NL2SQLSignature)
        result = predict(schema=schema, question=question)
        return dspy.Output(sql=result.sql)

# Optimize with teleprompter
optimizer = dspy.BootstrapFewShot(
    max_labeled_demos=3,
    max_rounds=1
)

optimized_module = optimizer.compile(
    NL2SQLModule(),
    trainset=train_examples  # Labeled examples for optimization
)
```

**File**: `src/nl2sql/optim/dspy_optim.py`

### TextGrad Prompt Optimization

TextGrad uses gradient-based prompt optimization:

```python
import textgrad as tg

# Define prompt as variable
prompt = tg.Variable(
    "### Task: Convert the following natural language question to a SQL query.\n\n### Database Schema:\n{schema}\n\n### Question: {question}\n\n### SQL Query:",
    requires_grad=True,
    role_description="prompt for NL2SQL"
)

# Define loss function
def eval_prompt(prompt, examples):
    """Evaluate prompt on examples"""
    correct = 0
    for ex in examples:
        formatted = prompt.format(schema=ex['schema'], question=ex['question'])
        generated = llm_generate(formatted)
        if generated.strip() == ex['sql'].strip():
            correct += 1
    return correct / len(examples)

# Gradient descent on prompts
optimizer = tg.TextualGradientDescent()
for epoch in range(10):
    loss = eval_prompt(prompt, val_examples)
    prompt = tg.backward(loss, prompt)
    prompt = optimizer.update(prompt)

print(f"Optimized prompt: {prompt.value}")
```

**File**: `src/nl2sql/optim/textgrad_optim.py`

## Critical Gotchas

### Stop Sequences
Prevent model from generating beyond SQL:
```python
response = client.chat.completions.create(
    ...,
    stop=["\n\n", "###"]  # Stop at double newline or next section
)
```

### Temperature Settings
Lower temperature for deterministic SQL:
```python
# For generation: Use low temperature for consistency
temperature=0.0  # or 0.1

# For optimization: Can use slightly higher for exploration
temperature=0.3
```

### Schema Truncation
Large schemas exceed context windows:
```python
# Truncate to prevent token limit overflow
if len(schema) > 1000:
    schema = schema[:1000] + "\n-- (truncated for space)"
```

### Field Name Variability
Different datasets use different field names:
```python
# Always use .get() with fallbacks
sql = ex.get("sql", ex.get("query", ""))
question = ex.get("question", ex.get("sql_prompt", ""))
```

## Integration

- **Related skill:** `sql-evaluation` - For SQL extraction and validation
- **Related skill:** `nl2sql-training` - For training prompt format (Alpaca)
- **Evaluation script:** `src/nl2sql/eval/baseline.py` - All baseline methods
- **Optimization scripts:** `src/nl2sql/optim/dspy_optim.py`, `src/nl2sql/optim/textgrad_optim.py`

## Anti-Patterns

### What NOT to Do

**Don't forget stop sequences:**
```python
# BAD: Model keeps generating after SQL
response = client.chat.completions.create(
    model="codellama/CodeLlama-7b-hf",
    messages=[...],
    max_tokens=250
)

# GOOD: Stop at delimiters
response = client.chat.completions.create(
    model="codellama/CodeLlama-7b-hf",
    messages=[...],
    max_tokens=250,
    stop=["\n\n", "###"]
)
```

**Don't use too many few-shot examples:**
```python
# BAD: More examples ≠ better performance
examples = all_training_data[:50]  # Confuses the model

# GOOD: 2-3 similar examples
examples = find_similar(question, training_data, k=2)
```

**Don't mix instruction formats:**
```python
# BAD: Inconsistent delimiters confuse the model
prompt1 = "### Task:\n..."
prompt2 = "Convert to SQL:\n..."
prompt3 = "Instruction:\n..."

# GOOD: Consistent format throughout
prompt_template = "### Task: Convert natural language to SQL\n\n### Schema:\n{schema}\n\n### Question: {question}\n\n### SQL:"
```
