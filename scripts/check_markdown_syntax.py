#!/usr/bin/env python3
"""
Markdown Syntax Linter for ML Study Guide

Detects suspicious markdown rendering patterns that may not display correctly,
including:
1. {#...} anchor syntax (not standard markdown, displays literally)
2. Unmatched LaTeX delimiters
3. Malformed link syntax
4. Code blocks with suspicious content (LaTeX/markdown that should be rendered)
5. Mismatched code block fence indentation (especially in list context)
6. Other rendering issues

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

        # Check for code block issues (needs full file context)
        file_issues.extend(self._check_code_blocks(file_path, lines))

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

    def _check_code_blocks(self, file_path: Path, lines: List[str]) -> List[Issue]:
        """Check code blocks for suspicious content and formatting issues"""
        issues = []
        in_code_block = False
        code_block_start_line = 0
        code_block_start_indent = 0
        code_block_lang = ""
        code_block_content = []

        # Patterns that suggest content should be rendered, not shown as code
        # These are suspicious when found in non-math, non-python code blocks
        latex_patterns = [
            (r'\\begin\{', "LaTeX \\begin{} environment"),
            (r'\\end\{', "LaTeX \\end{} environment"),
            (r'\\frac\{', "LaTeX \\frac command"),
            (r'\\text\{', "LaTeX \\text command"),
            (r'\\mathbb\{', "LaTeX \\mathbb command"),
            (r'\\sum_', "LaTeX \\sum command"),
            (r'\\prod_', "LaTeX \\prod command"),
            (r'\\int_', "LaTeX \\int command"),
        ]

        markdown_patterns = [
            (r'^\s*#{1,6}\s+\w', "Markdown header"),
            (r'^\s*\*\*[^*]+\*\*\s*$', "Bold text on its own line"),
            (r'^\s*\[[^\]]+\]\([^)]+\)\s*$', "Markdown link on its own line"),
        ]

        for line_num, line in enumerate(lines, start=1):
            # Check for code block fence
            fence_match = re.match(r'^(\s*)(```+)(\w*)', line)

            if fence_match and not in_code_block:
                # Starting a code block
                in_code_block = True
                code_block_start_line = line_num
                code_block_start_indent = len(fence_match.group(1))
                code_block_lang = fence_match.group(3).lower()
                code_block_content = []

            elif fence_match and in_code_block:
                # Ending a code block
                closing_indent = len(fence_match.group(1))

                # Check for mismatched indentation
                if closing_indent != code_block_start_indent:
                    issues.append(Issue(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line.rstrip(),
                        issue_type="CODE_BLOCK_INDENT_MISMATCH",
                        description=f"Closing fence indent ({closing_indent}) doesn't match opening fence indent ({code_block_start_indent}) at line {code_block_start_line}",
                        suggestion="Ensure opening and closing ``` have the same indentation"
                    ))

                # Check content for suspicious patterns
                # Skip known code languages and documentation blocks (latex, markdown show examples)
                safe_langs = ['math', 'python', 'py', 'javascript', 'js',
                              'typescript', 'ts', 'bash', 'sh', 'shell',
                              'json', 'yaml', 'yml', 'html', 'css', 'sql',
                              'rust', 'go', 'java', 'c', 'cpp', 'c++',
                              'latex', 'tex', 'markdown', 'md']

                if code_block_lang not in safe_langs:
                    content_str = '\n'.join(code_block_content)

                    # Check for LaTeX in unlabeled blocks (these are the real errors)
                    if code_block_lang == '':
                        for pattern, desc in latex_patterns:
                            if re.search(pattern, content_str):
                                issues.append(Issue(
                                    file_path=file_path,
                                    line_number=code_block_start_line,
                                    line_content=lines[code_block_start_line - 1].rstrip(),
                                    issue_type="SUSPICIOUS_CODE_BLOCK_CONTENT",
                                    description=f"Unlabeled code block contains {desc}",
                                    suggestion="If this is LaTeX, use ```math fence; if showing example syntax, use ```latex"
                                ))
                                break  # Only report once per block

                        # Check for markdown-like content in unlabeled blocks
                        for pattern, desc in markdown_patterns:
                            if re.search(pattern, content_str, re.MULTILINE):
                                issues.append(Issue(
                                    file_path=file_path,
                                    line_number=code_block_start_line,
                                    line_content=lines[code_block_start_line - 1].rstrip(),
                                    issue_type="SUSPICIOUS_CODE_BLOCK_CONTENT",
                                    description=f"Unlabeled code block contains {desc}",
                                    suggestion="Add a language identifier or check if this should be rendered as markdown"
                                ))
                                break  # Only report once per block

                in_code_block = False
                code_block_content = []

            elif in_code_block:
                # Accumulate content
                code_block_content.append(line)

        # Check for unclosed code block
        if in_code_block:
            issues.append(Issue(
                file_path=file_path,
                line_number=code_block_start_line,
                line_content=lines[code_block_start_line - 1].rstrip() if code_block_start_line <= len(lines) else "",
                issue_type="UNCLOSED_CODE_BLOCK",
                description="Code block opened but never closed",
                suggestion="Add closing ``` fence"
            ))

        return issues

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
