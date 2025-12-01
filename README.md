# nl2sql
NL to SQL fine-tuning

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. **Install UV:**

   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Or via Homebrew (macOS):
   ```bash
   brew install uv
   ```

2. **Install project dependencies:**
   ```bash
   uv pip install -e .
   ```

   For development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

### Using pip

Alternatively, you can use pip:
```bash
pip install -e .
```

## Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependency list
