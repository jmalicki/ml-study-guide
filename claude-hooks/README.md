# Claude Code Hooks

This directory contains hook scripts for Claude Code integration with git workflows.

## Overview

The hooks in this directory enable Claude Code to automatically run validation checks before git commits, ensuring that all markdown files pass link validation and SVG checks.

## Files

- `precommit.sh` - Pre-commit validation script that runs `make validate` and `make check-svg`

## How It Works

### Claude Code Hook Integration

Claude Code hooks are configured in `.claude/settings.json` or `.claude/settings.local.json`. The hook configuration intercepts `git commit` commands and runs validation checks before allowing the commit to proceed.

#### Configuration Example

Add this to `.claude/settings.json`:

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
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.command' 2>/dev/null | grep -q '^git commit'; then echo 'Running pre-commit validation checks...'; /home/jmalicki/src/ml-study-guide/claude-hooks/precommit.sh; fi",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

### How the Hook Works

1. The `PreToolUse` hook runs before any Bash command
2. It checks if the command is a `git commit` command
3. If yes, it runs `claude-hooks/precommit.sh`
4. The script runs:
   - `make validate` - Validates all internal links in markdown files
   - `make check-svg` - Checks for inline SVG (not supported on GitHub)
5. If either check fails (exit code != 0), the commit is blocked

### Alternative: Standard Git Hooks

The project also supports standard git pre-commit hooks via the `pre-commit` framework.

Install git hooks using:

```bash
make install-hooks
```

This will:

1. Install the `pre-commit` Python package (if not already installed)
2. Install git hooks based on `.pre-commit-config.yaml`

The standard git hooks include:

- Trailing whitespace check
- End of file fixer
- YAML validation
- Large file detection
- Merge conflict detection
- Markdown linting
- Link validation
- Inline SVG check

## Validation Checks

### Link Validation

The `scripts/validate_links.py` script checks:

- All internal markdown links are valid
- Referenced files exist
- Anchors point to valid headers

### SVG Check

The `scripts/check_inline_svg.py` script checks:

- No inline SVG elements in markdown files
- GitHub doesn't render inline SVG, so we require image references instead

## Exit Codes

- `0` - All checks passed, commit proceeds
- `1` - Validation failed, commit blocked
- `2` - Critical error (for Claude Code hooks only)

## Security Note

As documented in the [Claude Code hooks guide](https://code.claude.com/docs/en/hooks-guide):

> You must consider the security implication of hooks as you add them, because hooks run automatically during the agent loop with your current environment's credentials. For example, malicious hooks code can exfiltrate your data. Always review your hooks implementation before registering them.

All scripts in this directory should be reviewed before use.

## Troubleshooting

### Hook Not Running

1. Check that `.claude/settings.json` contains the hook configuration
2. Verify `precommit.sh` is executable: `chmod +x claude-hooks/precommit.sh`
3. Check that `jq` is installed: `which jq`

### Validation Failing

Run checks manually:

```bash
make validate    # Check links
make check-svg   # Check for inline SVG
make check       # Run all checks
```

### Bypassing Hooks (Not Recommended)

For emergency commits only:

```bash
git commit --no-verify -m "Emergency commit message"
```

## References

- [Claude Code Hooks Guide](https://code.claude.com/docs/en/hooks-guide)
- [GitButler Claude Code Hooks Documentation](https://docs.gitbutler.com/features/ai-integration/claude-code-hooks)
- [pre-commit Framework](https://pre-commit.com/)
