#!/usr/bin/env python3
r"""
Validate LaTeX syntax in markdown files.

This script:
1. Finds all markdown files in the repository
2. Extracts LaTeX math from ```math code blocks and inline $...$ expressions
3. Validates LaTeX syntax including:
   - Balanced braces { }
   - Balanced \left and \right delimiters
   - Balanced \begin and \end environments
4. Reports any syntax errors found
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LatexError:
    """Represents a LaTeX syntax error."""
    file_path: Path
    line_num: int
    error_type: str
    message: str
    latex_snippet: str


class LatexValidator:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.errors: List[LatexError] = []

    def extract_math_blocks(self, content: str, file_path: Path) -> List[Tuple[int, str, str]]:
        """
        Extract LaTeX math from markdown content.
        Returns list of (line_number, math_type, latex_code) tuples.
        math_type is either 'block' or 'inline'.
        """
        math_expressions = []
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
                    math_expressions.append((start_line, 'block', latex_code))
            i += 1

        # Extract inline $...$ math (but not $$, and not escaped \$)
        for line_num, line in enumerate(lines, 1):
            # Find all inline math expressions
            # Match $...$ but not $$...$$, and skip \$
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
                if latex_code.strip():  # Only include non-empty expressions
                    math_expressions.append((line_num, 'inline', latex_code))

                pos = end + 1

        return math_expressions

    def check_balanced_braces(self, latex: str) -> Optional[str]:
        """
        Check if braces are balanced in LaTeX code.
        Returns error message if unbalanced, None if balanced.

        Checks all types of braces: {}, (), []
        However, mixed usage is allowed in mathematical contexts like intervals [0, 1)
        Only reports mismatches when they occur within the same nesting level.
        """
        stack = []
        brace_pairs = {'{': '}', '(': ')', '[': ']'}
        closing_braces = {'}': '{', ')': '(', ']': '['}

        i = 0
        while i < len(latex):
            char = latex[i]

            # Skip escaped characters
            if char == '\\' and i + 1 < len(latex):
                i += 2
                continue

            # Opening braces
            if char in brace_pairs:
                stack.append((char, i))
            # Closing braces
            elif char in closing_braces:
                if not stack:
                    # Allow unmatched closing parens/brackets in math notation
                    # Only flag closing curly braces
                    if char == '}':
                        return f"Extra closing brace '}}' at position {i} with no matching opening brace"
                else:
                    expected_opening = closing_braces[char]
                    actual_opening, opening_pos = stack[-1]

                    # For curly braces, enforce strict matching
                    if char == '}':
                        if actual_opening != '{':
                            return f"Mismatched braces: '{actual_opening}' at position {opening_pos} closed with '}}' at position {i}"
                        stack.pop()
                    # For parentheses and brackets opened with curly brace, that's an error
                    elif actual_opening == '{':
                        return f"Mismatched braces: '{{' at position {opening_pos} closed with '{char}' at position {i}"
                    # For matching () or [], pop the stack
                    elif actual_opening == expected_opening:
                        stack.pop()
                    # For mismatched () and [], that's okay in math (e.g., intervals)
                    # Just pop to keep stack clean
                    else:
                        stack.pop()

            i += 1

        if stack:
            # Only report unclosed curly braces as errors
            unclosed_curlies = [(char, pos) for char, pos in stack if char == '{']
            if unclosed_curlies:
                positions = ', '.join(f"'{{' at position {pos}" for char, pos in unclosed_curlies)
                return f"Unclosed opening brace(s): {positions}"

        return None

    def check_left_right_pairs(self, latex: str) -> Optional[str]:
        r"""
        Check if \left and \right delimiters are balanced.
        Returns error message if unbalanced, None if balanced.
        """
        # Find all \left and \right commands
        left_positions = [m.start() for m in re.finditer(r'\\left\b', latex)]
        right_positions = [m.start() for m in re.finditer(r'\\right\b', latex)]

        if len(left_positions) != len(right_positions):
            return f"Unbalanced \\left and \\right: {len(left_positions)} \\left but {len(right_positions)} \\right"

        # Check they're properly nested (simple check - each \right comes after a \left)
        for i, (left_pos, right_pos) in enumerate(zip(left_positions, right_positions)):
            if right_pos < left_pos:
                return f"\\right appears before \\left at position {right_pos}"

        return None

    def check_begin_end_pairs(self, latex: str) -> Optional[str]:
        r"""
        Check if \begin{...} and \end{...} environments are balanced.
        Returns error message if unbalanced, None if balanced.
        """
        # Find all \begin{env} and \end{env} pairs
        begin_pattern = r'\\begin\{(\w+)\}'
        end_pattern = r'\\end\{(\w+)\}'

        begin_matches = list(re.finditer(begin_pattern, latex))
        end_matches = list(re.finditer(end_pattern, latex))

        if len(begin_matches) != len(end_matches):
            return f"Unbalanced environments: {len(begin_matches)} \\begin but {len(end_matches)} \\end"

        # Check that environments match
        env_stack = []

        # Combine and sort by position
        events = []
        for m in begin_matches:
            events.append((m.start(), 'begin', m.group(1)))
        for m in end_matches:
            events.append((m.start(), 'end', m.group(1)))

        events.sort(key=lambda x: x[0])

        for pos, event_type, env_name in events:
            if event_type == 'begin':
                env_stack.append(env_name)
            else:  # end
                if not env_stack:
                    return f"\\end{{{env_name}}} without matching \\begin"
                expected_env = env_stack.pop()
                if expected_env != env_name:
                    return f"Environment mismatch: \\begin{{{expected_env}}} closed with \\end{{{env_name}}}"

        if env_stack:
            return f"Unclosed environment(s): {', '.join(env_stack)}"

        return None

    def validate_latex(self, latex: str, file_path: Path, line_num: int, math_type: str) -> None:
        """Validate a single LaTeX expression and record any errors."""
        # Check balanced braces
        error = self.check_balanced_braces(latex)
        if error:
            self.errors.append(LatexError(
                file_path=file_path,
                line_num=line_num,
                error_type="Unbalanced braces",
                message=error,
                latex_snippet=latex[:100]  # First 100 chars
            ))

        # Check \left and \right pairs (only for expressions that use them)
        if '\\left' in latex or '\\right' in latex:
            error = self.check_left_right_pairs(latex)
            if error:
                self.errors.append(LatexError(
                    file_path=file_path,
                    line_num=line_num,
                    error_type="Unbalanced \\left/\\right",
                    message=error,
                    latex_snippet=latex[:100]
                ))

        # Check \begin and \end pairs (only for expressions that use them)
        if '\\begin' in latex or '\\end' in latex:
            error = self.check_begin_end_pairs(latex)
            if error:
                self.errors.append(LatexError(
                    file_path=file_path,
                    line_num=line_num,
                    error_type="Unbalanced environments",
                    message=error,
                    latex_snippet=latex[:100]
                ))

    def validate_file(self, file_path: Path) -> None:
        """Validate all LaTeX expressions in a markdown file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return

        math_expressions = self.extract_math_blocks(content, file_path)

        for line_num, math_type, latex_code in math_expressions:
            self.validate_latex(latex_code, file_path, line_num, math_type)

    def validate_all_markdown(self) -> None:
        """Validate LaTeX in all markdown files."""
        # Check chapters directory
        chapters_dir = self.root_dir / "chapters"
        if chapters_dir.exists():
            for md_file in sorted(chapters_dir.glob("*.md")):
                self.validate_file(md_file)

        # Check review directory
        review_dir = self.root_dir / "review"
        if review_dir.exists():
            for md_file in sorted(review_dir.glob("*.md")):
                self.validate_file(md_file)

        # Check root level markdown files (but skip test files in production)
        for md_file in sorted(self.root_dir.glob("*.md")):
            # Skip test files unless explicitly testing
            if "test" not in md_file.name.lower():
                self.validate_file(md_file)

    def run(self) -> int:
        """Run all validations and return exit code."""
        print("=" * 70)
        print("ML Study Guide - LaTeX Validation")
        print("=" * 70)
        print()

        print("Checking LaTeX syntax in markdown files...")
        self.validate_all_markdown()

        print()
        print("=" * 70)
        print("Validation Results")
        print("=" * 70)
        print()

        if self.errors:
            print(f"ERRORS ({len(self.errors)}):")
            print()

            # Group errors by file
            errors_by_file = {}
            for error in self.errors:
                rel_path = error.file_path.relative_to(self.root_dir)
                if rel_path not in errors_by_file:
                    errors_by_file[rel_path] = []
                errors_by_file[rel_path].append(error)

            for file_path in sorted(errors_by_file.keys()):
                print(f"  {file_path}:")
                for error in errors_by_file[file_path]:
                    print(f"    Line {error.line_num}: {error.error_type}")
                    print(f"      {error.message}")
                    if len(error.latex_snippet) < 100:
                        print(f"      LaTeX: {error.latex_snippet}")
                    else:
                        print(f"      LaTeX: {error.latex_snippet}...")
                    print()

            print("Validation FAILED")
            return 1
        else:
            print("All LaTeX syntax checks PASSED")
            return 0


def main():
    """Main entry point."""
    # Find the root directory
    current_dir = Path.cwd()
    if current_dir.name == "scripts":
        root_dir = current_dir.parent
    else:
        root_dir = current_dir

    validator = LatexValidator(root_dir)
    exit_code = validator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
