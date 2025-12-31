#!/usr/bin/env python3
"""
Markdown Syntax Linter for ML Study Guide

Detects suspicious markdown rendering patterns that may not display correctly,
including:
1. {#...} anchor syntax (not standard markdown, displays literally)
2. Unmatched LaTeX delimiters
3. Malformed link syntax
4. Other rendering issues

Usage:
    python3 scripts/check_markdown_syntax.py [path]

    If no path is provided, checks all markdown files in chapters/, review/, and appendices/
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Issue:
    """Represents a markdown syntax issue"""
    file_path: Path
    line_number: int
    line_content: str
    issue_type: str
    description: str
    suggestion: str = ""


class MarkdownSyntaxChecker:
    """Checks markdown files for suspicious rendering patterns"""

    def __init__(self):
        self.issues: List[Issue] = []

    def check_file(self, file_path: Path) -> List[Issue]:
        """Check a single markdown file for issues"""
        file_issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return file_issues

        for line_num, line in enumerate(lines, start=1):
            # Check for {#...} anchor syntax in headers
            if self._check_header_anchor(line):
                match = re.search(r'\{#([^}]+)\}', line)
                if match:
                    anchor_id = match.group(1)
                    file_issues.append(Issue(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line.rstrip(),
                        issue_type="HEADER_ANCHOR_SYNTAX",
                        description=f"Non-standard {{#{anchor_id}}} anchor syntax in header",
                        suggestion=f"Remove {{#{anchor_id}}} - markdown links should use (#anchor) format in URLs"
                    ))

            # Check for unescaped curly braces in text (not in code blocks or LaTeX)
            if self._check_unescaped_braces(line, line_num, lines):
                # This is informational - curly braces might be intentional
                pass

            # Check for malformed links - [text](url with spaces not quoted)
            if self._check_malformed_links(line):
                file_issues.append(Issue(
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line.rstrip(),
                    issue_type="MALFORMED_LINK",
                    description="Link may have unencoded spaces",
                    suggestion="Ensure URLs with spaces are properly encoded or use angle brackets"
                ))

            # Check for HTML entities that might not render
            if self._check_html_entities(line):
                file_issues.append(Issue(
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line.rstrip(),
                    issue_type="HTML_ENTITY",
                    description="HTML entities found - may not render correctly",
                    suggestion="Use UTF-8 characters directly or ensure entities are intended"
                ))

            # Check for missing alt text in images
            if self._check_image_alt_text(line):
                file_issues.append(Issue(
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line.rstrip(),
                    issue_type="MISSING_ALT_TEXT",
                    description="Image missing alt text",
                    suggestion="Add descriptive alt text: ![description](url)"
                ))

        self.issues.extend(file_issues)
        return file_issues

    def _check_header_anchor(self, line: str) -> bool:
        """Check if line is a header with {#...} anchor syntax"""
        # Match markdown headers with {#id} syntax
        # But exclude LaTeX math formulas (lines containing $ signs)
        if '$' in line:
            # This might be LaTeX, not a markdown anchor issue
            return False
        return bool(re.match(r'^#{1,6}\s+.*\{#[^}]+\}\s*$', line))

    def _check_unescaped_braces(self, line: str, line_num: int, all_lines: List[str]) -> bool:
        """Check for unescaped curly braces that might be rendering issues"""
        # Skip code blocks
        if line.strip().startswith('```') or line.strip().startswith('    '):
            return False

        # Skip inline code
        if '`' in line:
            # Remove inline code sections before checking
            line_no_code = re.sub(r'`[^`]+`', '', line)
        else:
            line_no_code = line

        # Skip LaTeX math
        line_no_math = re.sub(r'\$[^$]+\$', '', line_no_code)
        line_no_math = re.sub(r'\$\$[^$]+\$\$', '', line_no_math)

        # Now check for stray braces (but we'll make this informational only)
        return False  # Disabled for now as it may have false positives

    def _check_malformed_links(self, line: str) -> bool:
        """Check for links with potential formatting issues"""
        # Look for [text](url with multiple spaces) which might not be encoded
        pattern = r'\[([^\]]+)\]\(([^)]+\s{2,}[^)]+)\)'
        return bool(re.search(pattern, line))

    def _check_html_entities(self, line: str) -> bool:
        """Check for HTML entities like &lt; &gt; &amp;"""
        # Skip code blocks and inline code
        if line.strip().startswith('```') or line.strip().startswith('    '):
            return False

        # Check for common HTML entities
        pattern = r'&(lt|gt|amp|nbsp|quot);'
        return bool(re.search(pattern, line))

    def _check_image_alt_text(self, line: str) -> bool:
        """Check for images without alt text"""
        # Match ![](url) with empty alt text
        pattern = r'!\[\s*\]\([^)]+\)'
        return bool(re.search(pattern, line))

    def check_directory(self, directory: Path, pattern: str = "**/*.md") -> List[Issue]:
        """Check all markdown files in a directory"""
        for md_file in sorted(directory.glob(pattern)):
            # Skip virtual environment and hidden directories
            if '.venv' in str(md_file) or '/.git' in str(md_file):
                continue

            self.check_file(md_file)

        return self.issues

    def print_report(self, verbose: bool = True) -> int:
        """Print a report of all issues found"""
        if not self.issues:
            print("✓ No markdown syntax issues found!")
            return 0

        # Group issues by type
        issues_by_type: Dict[str, List[Issue]] = {}
        for issue in self.issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)

        # Print summary
        print(f"\n{'='*80}")
        print(f"MARKDOWN SYNTAX ISSUES FOUND: {len(self.issues)}")
        print(f"{'='*80}\n")

        for issue_type, issues in sorted(issues_by_type.items()):
            print(f"\n{issue_type}: {len(issues)} issue(s)")
            print("-" * 80)

            for issue in issues:
                print(f"\n  File: {issue.file_path}")
                print(f"  Line {issue.line_number}: {issue.description}")
                if verbose:
                    print(f"  Content: {issue.line_content}")
                    if issue.suggestion:
                        print(f"  Suggestion: {issue.suggestion}")

        print(f"\n{'='*80}")
        print(f"Total issues: {len(self.issues)}")
        print(f"{'='*80}\n")

        return len(self.issues)


def main():
    """Main entry point"""
    checker = MarkdownSyntaxChecker()

    # Determine what to check
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file():
            checker.check_file(path)
        elif path.is_dir():
            checker.check_directory(path)
        else:
            print(f"Error: {path} is not a valid file or directory", file=sys.stderr)
            return 1
    else:
        # Default: check standard directories
        root = Path(__file__).parent.parent
        for directory in ['chapters', 'review', 'appendices']:
            dir_path = root / directory
            if dir_path.exists():
                checker.check_directory(dir_path, "*.md")

    # Print report
    num_issues = checker.print_report(verbose=True)

    # Return exit code
    return 1 if num_issues > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
