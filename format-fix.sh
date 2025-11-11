#!/bin/bash
# Apply formatting using black and isort

echo "Running isort..."
uv run isort backend/

echo ""
echo "Running black..."
uv run black backend/

echo ""
echo "Formatting complete!"
