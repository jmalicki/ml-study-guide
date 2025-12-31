#!/bin/bash
# Pre-commit validation script for ML Study Guide
# Runs link validation and SVG checks before git commits

set -e

echo "Running pre-commit validation checks..."
echo ""

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run link validation
echo "1. Validating internal links..."
if ! make validate; then
    echo ""
    echo "ERROR: Link validation failed!"
    echo "Please fix broken links before committing."
    exit 1
fi

echo ""

# Run SVG check
echo "2. Checking for inline SVG..."
if ! make check-svg; then
    echo ""
    echo "ERROR: Inline SVG detected!"
    echo "GitHub doesn't support inline SVG in markdown. Please convert to image references."
    exit 1
fi

echo ""

# Validate SVG files
echo "3. Validating SVG files..."
if ! make validate-svg; then
    echo ""
    echo "ERROR: Invalid SVG files detected!"
    echo "Please fix SVG syntax errors before committing."
    exit 1
fi

echo ""
echo "All pre-commit checks passed!"
exit 0
