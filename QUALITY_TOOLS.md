# Code Quality Tools

This project uses several code quality tools to maintain consistent code style and catch potential issues.

## Installed Tools

### Black
**Purpose**: Automatic code formatting
**Configuration**: `pyproject.toml` - `[tool.black]`
**Line length**: 100 characters
**Target**: Python 3.13

Black enforces a consistent code style across the entire codebase, eliminating style debates and making code reviews focus on logic rather than formatting.

### isort
**Purpose**: Import statement sorting and organization
**Configuration**: `pyproject.toml` - `[tool.isort]`
**Profile**: black-compatible

Automatically sorts and organizes import statements in a consistent manner, compatible with Black's formatting style.

### Flake8
**Purpose**: Style guide enforcement (PEP 8) and error detection
**Configuration**: `.flake8`
**Max line length**: 100 characters

Checks code for PEP 8 compliance and common programming errors. Some rules are relaxed to work well with Black and test files.

### MyPy
**Purpose**: Static type checking
**Configuration**: `pyproject.toml` - `[tool.mypy]`
**Python version**: 3.13

Performs static type analysis to catch type-related bugs before runtime (optional typing).

### Pylint
**Purpose**: Advanced code analysis and linting
**Configuration**: `pyproject.toml` - `[tool.pylint]`
**Max line length**: 100 characters

Provides detailed code analysis including complexity metrics, code smells, and potential bugs.

## Usage

### Quick Start - Format Your Code

To automatically format your code with Black and isort:

```bash
./format-fix.sh
```

### Check Formatting Without Changes

To see what would be formatted without making changes:

```bash
./format.sh
```

### Run Linting Checks

To run flake8 and pylint:

```bash
./lint.sh
```

### Run All Quality Checks

To run all quality checks (formatting, linting, type checking):

```bash
./check.sh
```

This runs in strict mode and will exit on the first error.

## Development Workflow

### Before Committing

1. Format your code:
   ```bash
   ./format-fix.sh
   ```

2. Run quality checks:
   ```bash
   ./check.sh
   ```

3. Fix any reported issues

4. Commit your changes

### Individual Tool Commands

If you need to run tools individually:

```bash
# Format with black
uv run black backend/

# Sort imports
uv run isort backend/

# Run flake8
uv run flake8 backend/ --max-line-length=100

# Run pylint
uv run pylint backend/

# Run mypy
uv run mypy backend/
```

## Installing Dependencies

All quality tools are installed as dev dependencies. To install them:

```bash
uv sync
```

## Configuration Details

### Black Configuration
- Line length: 100 characters
- Target Python: 3.13
- Excludes: `.venv`, `build`, `dist`, `chroma_db`

### isort Configuration
- Profile: black (compatible with Black)
- Line length: 100 characters
- Respects .gitignore
- Knows about `backend` as first-party

### Flake8 Configuration
- Max line length: 100
- Ignored rules:
  - E203: Whitespace before ':' (Black compatibility)
  - W503: Line break before binary operator (Black compatibility)
  - E402: Module level import not at top (common in test files with sys.path manipulation)
  - F401: Unused imports (common in __init__.py and test fixtures)
  - F841: Unused variables (common in test assertions)
  - F541: f-string without placeholders (used for consistent string formatting)

### MyPy Configuration
- Python version: 3.13
- Warns on returning Any
- Ignores missing imports (for external libraries without type stubs)
- Does not require type hints (gradual typing)

### Pylint Configuration
- Max line length: 100
- Disabled rules:
  - C0111: Missing docstrings (too verbose for rapid development)
  - C0103: Invalid naming (flexible naming for tests and local variables)
  - R0913: Too many arguments (common in initialization and configuration)

## Troubleshooting

### "Module not found" errors when running tools

Make sure you've installed dev dependencies:
```bash
uv sync
```

### Virtual environment warnings

If you see warnings about `VIRTUAL_ENV` mismatch, you can safely ignore them or run:
```bash
uv run --active [tool] [args]
```

### Formatting conflicts between Black and Flake8

The configuration is designed to avoid conflicts. If you encounter any:
1. Run `./format-fix.sh` first
2. Then run `./check.sh`

Black takes precedence over Flake8 for style choices.

## Continuous Integration

These tools can be integrated into CI/CD pipelines:

```bash
# In CI pipeline
uv sync
./check.sh
```

This will ensure all code meets quality standards before merging.
