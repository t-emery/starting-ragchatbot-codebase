#!/bin/bash
# Run linting checks

echo "Running flake8..."
uv run flake8 backend/

echo ""
echo "Running pylint..."
uv run pylint backend/ --rcfile=pyproject.toml

echo ""
echo "Linting complete!"
