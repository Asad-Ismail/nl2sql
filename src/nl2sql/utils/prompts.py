"""Shared system prompts for NL2SQL tasks."""

BASELINE_SYSTEM_PROMPT = "### Task: Convert the following natural language question to a SQL query. Give only SQL Query as Output"

BASELINE_USER_PROMPT_TEMPLATE = """### Database Schema:
{schema}

### Question: {question}

### SQL Query:"""

ZERO_SHOT_PROMPT = """### Task: Convert the following natural language question to a SQL query. Give only SQL Query as Output

### Database Schema:
{schema}

### Question: {question}

### SQL Query:"""
