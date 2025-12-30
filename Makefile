.PHONY: help lint validate check-svg check install-hooks clean

help:
	@echo "ML Study Guide - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  make lint           - Run markdown linting"
	@echo "  make validate       - Run link validation"
	@echo "  make check-svg      - Check for inline SVG (not supported on GitHub)"
	@echo "  make check          - Run all checks (lint + validate + svg)"
	@echo "  make install-hooks  - Install pre-commit hooks"
	@echo "  make clean          - Clean temporary files"
	@echo ""

lint:
	@echo "Running markdown linting..."
	@command -v markdownlint >/dev/null 2>&1 || { echo "Installing markdownlint-cli..."; npm install -g markdownlint-cli; }
	markdownlint '**/*.md' --config .markdownlint.json

validate:
	@echo "Running link validation..."
	python3 scripts/validate_links.py

check-svg:
	@echo "Checking for inline SVG..."
	python3 scripts/check_inline_svg.py

check: lint validate check-svg
	@echo ""
	@echo "All checks passed!"

install-hooks:
	@echo "Installing pre-commit hooks..."
	@command -v pre-commit >/dev/null 2>&1 || { echo "Installing pre-commit..."; pip install pre-commit; }
	pre-commit install
	@echo "Pre-commit hooks installed successfully!"

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type f -name '*~' -delete
	find . -type f -name '*.bak' -delete
	@echo "Cleanup complete!"
