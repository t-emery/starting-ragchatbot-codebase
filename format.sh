#!/bin/bash
# Format code using black and isort

echo "Running isort..."
uv run isort backend/ --check-only --diff

echo ""
echo "Running black..."
uv run black backend/ --check --diff

echo ""
echo "To apply formatting, run:"
echo "  uv run isort backend/"
echo "  uv run black backend/"
