.PHONY: help lint validate check-svg validate-svg validate-activation-svg check-svg-contrast check-latex lint-latex check-markdown-syntax check install-hooks install-claude-hooks clean

help:
	@echo "ML Study Guide - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  make lint                     - Run markdown linting"
	@echo "  make validate                 - Run link validation"
	@echo "  make check-svg                - Check for inline SVG (not supported on GitHub)"
	@echo "  make validate-svg             - Validate SVG files with SVGO"
	@echo "  make validate-activation-svg  - Validate activation function SVG against PyTorch"
	@echo "  make check-svg-contrast       - Check SVG text contrast accessibility"
	@echo "  make check-latex              - Validate LaTeX syntax in markdown files (custom rules)"
	@echo "  make lint-latex               - Lint LaTeX with ChkTeX (external linter)"
	@echo "  make check-markdown-syntax    - Check for suspicious markdown rendering patterns"
	@echo "  make check                    - Run all checks (lint + validate + svg + latex + contrast + markdown)"
	@echo "  make install-hooks            - Install standard git pre-commit hooks"
	@echo "  make install-claude-hooks     - Install Claude Code hooks"
	@echo "  make clean                    - Clean temporary files"
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

validate-svg:
	@echo "Validating SVG files with SVGO..."
	@command -v svgo >/dev/null 2>&1 || { echo "Installing svgo..."; npm install -g svgo; }
	@for f in assets/diagrams/*.svg; do \
		if ! svgo --input "$$f" --output /dev/null 2>&1; then \
			echo "ERROR: Invalid SVG: $$f"; \
			exit 1; \
		fi; \
	done
	@echo "All SVG files are valid!"

validate-activation-svg:
	@echo "Validating activation function SVG against PyTorch implementations..."
	python3 scripts/validate_activation_svg.py

check-svg-contrast:
	@echo "Checking SVG text contrast accessibility..."
	python3 scripts/check_svg_contrast.py

check-latex:
	@echo "Validating LaTeX syntax..."
	python3 scripts/check_latex.py

lint-latex:
	@echo "Linting LaTeX with ChkTeX..."
	@command -v chktex >/dev/null 2>&1 || { echo "ChkTeX not found. Install with: sudo apt install chktex (or brew install chktex)"; exit 1; }
	python3 scripts/lint_latex.py

check-markdown-syntax:
	@echo "Checking markdown syntax..."
	python3 scripts/check_markdown_syntax.py

check: lint validate check-svg validate-svg check-svg-contrast check-latex lint-latex check-markdown-syntax
	@echo ""
	@echo "All checks passed!"

install-hooks:
	@echo "Installing pre-commit hooks..."
	@command -v pre-commit >/dev/null 2>&1 || { echo "Installing pre-commit..."; pip install pre-commit; }
	pre-commit install
	@echo "Pre-commit hooks installed successfully!"
	@echo ""
	@echo "To also enable Claude Code hooks, run: make install-claude-hooks"

install-claude-hooks:
	@echo "Installing Claude Code hooks..."
	@echo ""
	@echo "Claude Code hooks must be configured in .claude/settings.json"
	@echo ""
	@if [ -f .claude/settings.json ]; then \
		echo "WARNING: .claude/settings.json already exists!"; \
		echo "Please manually merge the hook configuration from:"; \
		echo "  claude-hooks/settings.json.template"; \
		echo ""; \
		echo "Or add this to your .claude/settings.json:"; \
		cat claude-hooks/settings.json.template | sed 's|PROJECT_ROOT|$(CURDIR)|g'; \
	else \
		echo "Creating .claude/settings.json..."; \
		mkdir -p .claude; \
		cat claude-hooks/settings.json.template | sed 's|PROJECT_ROOT|$(CURDIR)|g' > .claude/settings.json; \
		echo "Claude Code hooks installed successfully!"; \
	fi
	@echo ""
	@echo "Note: .claude/settings.local.json takes precedence if it exists."
	@echo "See claude-hooks/README.md for more information."

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type f -name '*~' -delete
	find . -type f -name '*.bak' -delete
	@echo "Cleanup complete!"
