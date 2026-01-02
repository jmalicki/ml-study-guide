#!/usr/bin/env python3
"""
Extract LaTeX from markdown files and lint using ChkTeX.

This script:
1. Finds all markdown files in chapters/ and review/ directories
2. Extracts LaTeX from ```math code blocks and inline $...$ expressions
3. Wraps the LaTeX in a minimal document for ChkTeX validation
4. Runs ChkTeX and reports any issues with original file/line references

Requirements:
    - ChkTeX must be installed (apt install chktex or brew install chktex)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class LatexFragment:
    """A LaTeX fragment extracted from a markdown file."""
    file_path: Path
    line_num: int
    math_type: str  # 'block' or 'inline'
    content: str
    original_line: str  # Original markdown line (for context)


@dataclass
class LintError:
    """A linting error from ChkTeX."""
    file_path: Path
    line_num: int
    error_code: str
    message: str
    latex_snippet: str
    severity: str  # 'error' or 'warning'


@dataclass
class MarkdownFormatError:
    """A markdown formatting error (not LaTeX-specific)."""
    file_path: Path
    line_num: int
    error_code: str
    message: str
    context: str


def check_chktex_installed() -> bool:
    """Check if ChkTeX is installed and available."""
    return shutil.which('chktex') is not None


def check_double_blank_before_math(content: str, file_path: Path) -> List[MarkdownFormatError]:
    """
    Check for double blank lines before math blocks.

    This breaks markdown list formatting - a list item followed by two blank lines
    before a math block causes the math block to not be properly associated with
    the list item.

    Pattern detected:
    - Any non-blank line
    - Two or more consecutive blank lines
    - A line containing ```math or starting with $$
    """
    errors = []
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        # Look for ```math or $$ lines
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('```math') or (stripped.startswith('$$') and not stripped.endswith('$$')):
            # Count blank lines before this
            blank_count = 0
            j = i - 1
            while j >= 0 and lines[j].strip() == '':
                blank_count += 1
                j -= 1

            if blank_count >= 2:
                # Find the last non-blank line for context
                context_line = lines[j] if j >= 0 else ""
                errors.append(MarkdownFormatError(
                    file_path=file_path,
                    line_num=i + 1,  # 1-indexed
                    error_code="MD001",
                    message=f"Double blank line before math block (found {blank_count} blank lines). This breaks list formatting.",
                    context=context_line[:80] if context_line else "(start of file)"
                ))
        i += 1

    return errors


def extract_latex_from_markdown(content: str, file_path: Path) -> List[LatexFragment]:
    """
    Extract LaTeX math expressions from markdown content.

    Returns list of LatexFragment objects with source file info.
    """
    fragments = []
    lines = content.split('\n')

    # Extract ```math code blocks
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('```math'):
            start_line = i + 1
            math_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                math_lines.append(lines[i])
                i += 1
            if math_lines:
                latex_code = '\n'.join(math_lines)
                fragments.append(LatexFragment(
                    file_path=file_path,
                    line_num=start_line + 1,  # 1-indexed
                    math_type='block',
                    content=latex_code,
                    original_line=latex_code[:80]
                ))
        i += 1

    # Extract inline $...$ math (but not $$ or escaped \$)
    for line_num, line in enumerate(lines, 1):
        pos = 0
        while pos < len(line):
            # Look for $ that's not preceded by \ and not followed by another $
            match = re.search(r'(?<!\\)\$(?!\$)', line[pos:])
            if not match:
                break

            start = pos + match.start()
            # Find closing $
            end_match = re.search(r'(?<!\\)\$(?!\$)', line[start + 1:])
            if not end_match:
                break

            end = start + 1 + end_match.start()
            latex_code = line[start + 1:end]
            if latex_code.strip():
                fragments.append(LatexFragment(
                    file_path=file_path,
                    line_num=line_num,
                    math_type='inline',
                    content=latex_code,
                    original_line=line.strip()[:80]
                ))

            pos = end + 1

    return fragments


def create_latex_document(fragments: List[LatexFragment]) -> Tuple[str, dict]:
    """
    Create a minimal LaTeX document containing all fragments.

    Returns (document_content, line_mapping) where line_mapping maps
    document line numbers back to original fragment info.
    """
    # LaTeX preamble with common packages used in ML study guide
    preamble = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{mathtools}
\usepackage{bm}

% Common commands that may be used
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\text{Var}}
\newcommand{\Cov}{\text{Cov}}

\begin{document}
"""

    document_lines = preamble.split('\n')
    line_mapping = {}  # doc_line_num -> LatexFragment

    current_doc_line = len(document_lines) + 1  # Next line number after preamble

    for fragment in fragments:
        # Add a comment with source info
        source_comment = f"% Source: {fragment.file_path}:{fragment.line_num} ({fragment.math_type})"
        document_lines.append(source_comment)
        current_doc_line += 1

        if fragment.math_type == 'block':
            # Block math: wrap in equation* or align* environment
            if '&' in fragment.content or '\\\\' in fragment.content:
                # Looks like an align environment
                document_lines.append(r'\begin{align*}')
                current_doc_line += 1

                for line in fragment.content.split('\n'):
                    document_lines.append(line)
                    line_mapping[current_doc_line] = fragment
                    current_doc_line += 1

                document_lines.append(r'\end{align*}')
                current_doc_line += 1
            else:
                # Simple equation
                document_lines.append(r'\begin{equation*}')
                current_doc_line += 1

                for line in fragment.content.split('\n'):
                    document_lines.append(line)
                    line_mapping[current_doc_line] = fragment
                    current_doc_line += 1

                document_lines.append(r'\end{equation*}')
                current_doc_line += 1
        else:
            # Inline math: wrap in $...$
            document_lines.append(f'${fragment.content}$')
            line_mapping[current_doc_line] = fragment
            current_doc_line += 1

        # Add blank line for separation
        document_lines.append('')
        current_doc_line += 1

    document_lines.append(r'\end{document}')

    return '\n'.join(document_lines), line_mapping


def run_chktex(latex_document: str, chktex_config: Optional[Path] = None) -> List[Tuple[int, str, str]]:
    """
    Run ChkTeX on the LaTeX document.

    Returns list of (line_num, error_code, message) tuples.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_document)
        temp_file = f.name

    try:
        cmd = ['chktex', '-q', '-v0']

        # Add config file if specified
        if chktex_config and chktex_config.exists():
            cmd.extend(['-l', str(chktex_config)])

        cmd.append(temp_file)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        errors = []
        # Parse ChkTeX output
        # Format: filename:line:col:type:code:message
        for line in result.stdout.split('\n'):
            if not line.strip():
                continue

            # ChkTeX output format varies, try to parse it
            # Common format: file:line:col: Warning N: message
            # or: file:line:col: Error N: message
            match = re.match(r'[^:]+:(\d+):\d+:\s*(Warning|Error)\s+(\d+):\s*(.+)', line)
            if match:
                line_num = int(match.group(1))
                severity = match.group(2)
                error_code = match.group(3)
                message = match.group(4)
                errors.append((line_num, f"{severity[0]}{error_code}", message))

        return errors

    finally:
        os.unlink(temp_file)


def lint_markdown_files(
    root_dir: Path,
    chktex_config: Optional[Path] = None,
    verbose: bool = False
) -> Tuple[List[LintError], List[MarkdownFormatError]]:
    """
    Lint LaTeX in all markdown files.

    Returns tuple of (LintError list, MarkdownFormatError list).
    """
    all_fragments = []
    all_md_files = []  # Track all markdown files for format checks

    # Collect fragments from chapters/
    chapters_dir = root_dir / "chapters"
    if chapters_dir.exists():
        for md_file in sorted(chapters_dir.glob("**/*.md")):
            try:
                content = md_file.read_text(encoding='utf-8')
                all_md_files.append((md_file, content))
                fragments = extract_latex_from_markdown(content, md_file)
                all_fragments.extend(fragments)
                if verbose:
                    print(f"  Found {len(fragments)} LaTeX fragments in {md_file.name}")
            except Exception as e:
                print(f"Error reading {md_file}: {e}", file=sys.stderr)

    # Collect fragments from review/
    review_dir = root_dir / "review"
    if review_dir.exists():
        for md_file in sorted(review_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding='utf-8')
                all_md_files.append((md_file, content))
                fragments = extract_latex_from_markdown(content, md_file)
                all_fragments.extend(fragments)
                if verbose:
                    print(f"  Found {len(fragments)} LaTeX fragments in {md_file.name}")
            except Exception as e:
                print(f"Error reading {md_file}: {e}", file=sys.stderr)

    # Collect from appendices/
    appendices_dir = root_dir / "appendices"
    if appendices_dir.exists():
        for md_file in sorted(appendices_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding='utf-8')
                all_md_files.append((md_file, content))
                fragments = extract_latex_from_markdown(content, md_file)
                all_fragments.extend(fragments)
                if verbose:
                    print(f"  Found {len(fragments)} LaTeX fragments in {md_file.name}")
            except Exception as e:
                print(f"Error reading {md_file}: {e}", file=sys.stderr)

    # Collect from root markdown files (excluding test files)
    for md_file in sorted(root_dir.glob("*.md")):
        if "test" not in md_file.name.lower():
            try:
                content = md_file.read_text(encoding='utf-8')
                all_md_files.append((md_file, content))
                fragments = extract_latex_from_markdown(content, md_file)
                all_fragments.extend(fragments)
                if verbose and fragments:
                    print(f"  Found {len(fragments)} LaTeX fragments in {md_file.name}")
            except Exception as e:
                print(f"Error reading {md_file}: {e}", file=sys.stderr)

    # Check for markdown format errors
    md_format_errors = []
    for md_file, content in all_md_files:
        md_format_errors.extend(check_double_blank_before_math(content, md_file))

    if verbose and md_format_errors:
        print(f"\nFound {len(md_format_errors)} markdown format issues")

    if not all_fragments:
        return [], md_format_errors

    if verbose:
        print(f"\nTotal: {len(all_fragments)} LaTeX fragments")
        print("Creating LaTeX document for validation...")

    # Create combined LaTeX document
    latex_document, line_mapping = create_latex_document(all_fragments)

    if verbose:
        print("Running ChkTeX...")

    # Run ChkTeX
    raw_errors = run_chktex(latex_document, chktex_config)

    # Map errors back to original files
    lint_errors = []
    for doc_line, error_code, message in raw_errors:
        if doc_line in line_mapping:
            fragment = line_mapping[doc_line]
            lint_errors.append(LintError(
                file_path=fragment.file_path,
                line_num=fragment.line_num,
                error_code=error_code,
                message=message,
                latex_snippet=fragment.content[:80],
                severity='warning' if error_code.startswith('W') else 'error'
            ))

    return lint_errors, md_format_errors


def main():
    parser = argparse.ArgumentParser(
        description='Lint LaTeX in markdown files using ChkTeX'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Path to ChkTeX config file (.chktexrc)'
    )
    parser.add_argument(
        '--warnings-as-errors',
        action='store_true',
        help='Treat warnings as errors'
    )
    parser.add_argument(
        'root_dir',
        type=Path,
        nargs='?',
        default=None,
        help='Root directory of the project (default: current directory or parent if in scripts/)'
    )

    args = parser.parse_args()

    # Determine root directory
    if args.root_dir:
        root_dir = args.root_dir
    else:
        current_dir = Path.cwd()
        if current_dir.name == "scripts":
            root_dir = current_dir.parent
        else:
            root_dir = current_dir

    print("=" * 70)
    print("ML Study Guide - LaTeX Linting with ChkTeX")
    print("=" * 70)
    print()

    # Check ChkTeX is installed
    if not check_chktex_installed():
        print("ERROR: ChkTeX is not installed.")
        print()
        print("Install it with:")
        print("  Ubuntu/Debian: sudo apt install chktex")
        print("  macOS:         brew install chktex")
        print("  Fedora:        sudo dnf install texlive-chktex")
        print()
        sys.exit(1)

    print("Extracting LaTeX from markdown files...")
    if args.verbose:
        print()

    # Find config file
    config_path = args.config
    if not config_path:
        default_config = root_dir / ".chktexrc"
        if default_config.exists():
            config_path = default_config

    # Run linting
    latex_errors, md_format_errors = lint_markdown_files(
        root_dir,
        chktex_config=config_path,
        verbose=args.verbose
    )

    print()
    print("=" * 70)
    print("Linting Results")
    print("=" * 70)
    print()

    has_issues = False

    # Print markdown format errors first
    if md_format_errors:
        has_issues = True
        print("Markdown Format Errors:")
        print("-" * 40)
        md_errors_by_file = {}
        for error in md_format_errors:
            rel_path = error.file_path.relative_to(root_dir)
            if rel_path not in md_errors_by_file:
                md_errors_by_file[rel_path] = []
            md_errors_by_file[rel_path].append(error)

        for file_path in sorted(md_errors_by_file.keys()):
            print(f"{file_path}:")
            for error in sorted(md_errors_by_file[file_path], key=lambda e: e.line_num):
                print(f"  Line {error.line_num}: [{error.error_code}] {error.message}")
                if args.verbose:
                    print(f"    Context: {error.context}")
            print()

    if not latex_errors and not md_format_errors:
        print("No issues found!")
        sys.exit(0)

    # Separate LaTeX errors and warnings
    actual_errors = [e for e in latex_errors if e.severity == 'error']
    warnings = [e for e in latex_errors if e.severity == 'warning']

    if latex_errors:
        has_issues = True
        print("LaTeX Errors:")
        print("-" * 40)
        # Group by file
        errors_by_file = {}
        for error in latex_errors:
            rel_path = error.file_path.relative_to(root_dir)
            if rel_path not in errors_by_file:
                errors_by_file[rel_path] = []
            errors_by_file[rel_path].append(error)

        # Print results
        for file_path in sorted(errors_by_file.keys()):
            print(f"{file_path}:")
            for error in sorted(errors_by_file[file_path], key=lambda e: e.line_num):
                severity_marker = "ERROR" if error.severity == 'error' else "warning"
                print(f"  Line {error.line_num}: [{error.error_code}] {severity_marker}: {error.message}")
                if args.verbose:
                    print(f"    LaTeX: {error.latex_snippet}")
            print()

    # Summary
    print(f"Summary: {len(md_format_errors)} markdown format error(s), {len(actual_errors)} LaTeX error(s), {len(warnings)} LaTeX warning(s)")
    print()

    # Exit code - markdown format errors are always errors
    if md_format_errors or actual_errors or (args.warnings_as_errors and warnings):
        print("Linting FAILED")
        sys.exit(1)
    else:
        print("Linting passed (warnings only)")
        sys.exit(0)


if __name__ == "__main__":
    main()
