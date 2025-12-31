# Git Hooks Setup for ML Study Guide

This document describes the pre-commit hook system for the ML Study Guide repository.

## Overview

Two hook systems are available:

1. **Claude Code Hooks** - Integration with Claude Code for AI-assisted development
2. **Standard Git Hooks** - Traditional pre-commit framework for all users

Both systems run the same validation checks before commits:

- `make validate` - Link validation
- `make check-svg` - Inline SVG check

## Quick Start

### For Claude Code Users

Run this command to install Claude Code hooks:

```bash
make install-claude-hooks
```

This will create `.claude/settings.json` with the hook configuration.

### For All Users (Standard Git Hooks)

Run this command to install standard git pre-commit hooks:

```bash
make install-hooks
```

This will install the `pre-commit` framework and configure git hooks.

## Files Created

### Claude Code Hooks

```text
claude-hooks/
├── README.md                    # Detailed documentation
├── precommit.sh                 # Pre-commit validation script
└── settings.json.template       # Template for .claude/settings.json
```

### Configuration Files

- `.pre-commit-config.yaml` - Updated with inline SVG check
- `.claude/settings.json` - Created by `make install-claude-hooks`
- `Makefile` - Updated with new `install-claude-hooks` target

## How It Works

### Claude Code Hook Flow

1. Claude Code intercepts `git commit` commands
2. `PreToolUse` hook checks if the command is a git commit
3. Runs `claude-hooks/precommit.sh` before allowing the commit
4. Script executes:
   - Link validation (`make validate`)
   - SVG check (`make check-svg`)
5. If checks fail, commit is blocked with error message
6. If checks pass, commit proceeds normally

### Standard Git Hook Flow

1. User runs `git commit`
2. Git pre-commit hook framework activates
3. Runs configured hooks in `.pre-commit-config.yaml`:
   - Standard hooks (trailing whitespace, EOF, YAML, etc.)
   - Markdown linting
   - Link validation
   - Inline SVG check
4. If any check fails, commit is blocked
5. If all pass, commit proceeds

## Validation Checks

### Link Validation

Script: `scripts/validate_links.py`

Checks:

- All internal markdown links point to existing files
- All anchor links point to valid headers
- No broken cross-references

### SVG Check

Script: `scripts/check_inline_svg.py`

Checks:

- No inline `<svg>` elements in markdown files
- GitHub doesn't render inline SVG properly
- Requires using image references instead

## Configuration

### Claude Code Hook Configuration

The hook is configured in `.claude/settings.json`:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.command' 2>/dev/null | grep -q '^git commit'; then echo 'Running pre-commit validation checks...'; /absolute/path/to/claude-hooks/precommit.sh; fi",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

### Standard Git Hook Configuration

The hooks are configured in `.pre-commit-config.yaml`:

```yaml
repos:

  - repo: local

    hooks:

      - id: validate-links

        name: Validate internal links
        entry: python scripts/validate_links.py
        language: python
        pass_filenames: false
        files: '\.(md)$'

      - id: check-inline-svg

        name: Check for inline SVG
        entry: python scripts/check_inline_svg.py
        language: python
        pass_filenames: false
        files: '\.(md)$'
```

## Manual Testing

Run validation checks manually:

```bash
# Run all checks
make check

# Run individual checks
make validate      # Link validation only
make check-svg     # SVG check only
make lint          # Markdown linting only
```

## Bypassing Hooks (Emergency Only)

If you need to commit urgently and fix issues later:

```bash
git commit --no-verify -m "Emergency commit message"
```

**Warning:** Only use `--no-verify` in emergencies. It bypasses all validation.

## Troubleshooting

### Claude Code Hook Not Running

1. Check `.claude/settings.json` exists and contains hook configuration
2. Verify `precommit.sh` is executable:


   ```bash
   chmod +x claude-hooks/precommit.sh
```


3. Ensure `jq` is installed:


   ```bash
   sudo apt-get install jq  # Debian/Ubuntu
   brew install jq          # macOS
```

### Standard Git Hook Not Running

1. Ensure hooks are installed:


   ```bash
   make install-hooks
```


2. Check `.git/hooks/pre-commit` exists
3. Verify pre-commit framework is installed:


   ```bash
   pip install pre-commit
```

### Validation Failures

If commits are blocked due to validation errors:

1. Run checks manually to see details:


   ```bash
   make check
```


2. Fix reported issues
3. Retry the commit

## Integration with CI/CD

The same validation commands can be used in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow

- name: Run validation

  run: |
    make validate
    make check-svg
```

## References

- [Claude Code Hooks Guide](https://code.claude.com/docs/en/hooks-guide)
- [GitButler Claude Code Hooks Documentation](https://docs.gitbutler.com/features/ai-integration/claude-code-hooks)
- [Demystifying Claude Code Hooks](https://www.brethorsting.com/blog/2025/08/demystifying-claude-code-hooks/)
- [Automate Your AI Workflows with Claude Code Hooks](https://blog.gitbutler.com/automate-your-ai-workflows-with-claude-code-hooks)
- [pre-commit Framework](https://pre-commit.com/)

## Security Considerations

From the [Claude Code documentation](https://code.claude.com/docs/en/hooks-guide):

> You must consider the security implication of hooks as you add them, because hooks run automatically during the agent loop with your current environment's credentials. For example, malicious hooks code can exfiltrate your data. Always review your hooks implementation before registering them.

**Always review hook scripts before installation.**

## Next Steps

1. Install hooks using `make install-claude-hooks` or `make install-hooks`
2. Test by making a commit
3. Verify that validation runs before the commit completes
4. Review `claude-hooks/README.md` for more details
