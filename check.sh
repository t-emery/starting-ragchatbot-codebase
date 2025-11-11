#!/bin/bash
# Run all quality checks

echo "=========================================="
echo "Running Code Quality Checks"
echo "=========================================="
echo ""

# Track failures
FAILED=0

echo "1. Checking formatting with isort..."
if uv run isort backend/ --check-only; then
    echo "✓ isort passed"
else
    echo "✗ isort failed"
    FAILED=1
fi
echo ""

echo "2. Checking formatting with black..."
if uv run black backend/ --check; then
    echo "✓ black passed"
else
    echo "✗ black failed"
    FAILED=1
fi
echo ""

echo "3. Running flake8..."
if uv run flake8 backend/; then
    echo "✓ flake8 passed"
else
    echo "✗ flake8 failed"
    FAILED=1
fi
echo ""

echo "4. Running type checks with mypy (optional)..."
if uv run mypy backend/ --config-file=pyproject.toml; then
    echo "✓ mypy passed"
else
    echo "⚠ mypy found type issues (can be fixed incrementally)"
    # Don't set FAILED=1 for mypy, as type checking is optional
fi
echo ""

echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All critical quality checks passed!"
    echo "=========================================="
    exit 0
else
    echo "Some quality checks failed!"
    echo "Run ./format-fix.sh to auto-fix formatting"
    echo "=========================================="
    exit 1
fi
