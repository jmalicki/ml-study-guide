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
   - Proper bracing for multi-character superscripts and subscripts (^{...} and _{...})
   - Common LaTeX command errors (\frac, \sqrt)
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

    def check_superscript_subscript_braces(self, latex: str) -> Optional[str]:
        r"""
        Check that superscripts (^) and subscripts (_) use proper bracing.
        Multi-character superscripts/subscripts need braces: ^{abc} not ^abc.
        Returns error message if improper usage found, None if valid.
        """
        issues = []

        # Find all superscripts and subscripts
        i = 0
        while i < len(latex):
            # Skip escaped characters
            if latex[i] == '\\' and i + 1 < len(latex):
                i += 2
                continue

            if latex[i] in ('^', '_'):
                operator = latex[i]
                i += 1

                # Skip whitespace after operator
                while i < len(latex) and latex[i].isspace():
                    i += 1

                if i >= len(latex):
                    continue

                # Check what follows
                if latex[i] == '{':
                    # Properly braced - find matching closing brace
                    brace_count = 1
                    i += 1
                    while i < len(latex) and brace_count > 0:
                        if latex[i] == '\\' and i + 1 < len(latex):
                            i += 2
                            continue
                        if latex[i] == '{':
                            brace_count += 1
                        elif latex[i] == '}':
                            brace_count -= 1
                        i += 1
                elif latex[i] == '\\':
                    # Backslash command - check if it's a single command or needs braces
                    # Single commands like \alpha, \beta are OK
                    # Commands like \mathbb{R} followed by more content may need braces
                    start = i
                    i += 1
                    # Get the command name
                    while i < len(latex) and latex[i].isalpha():
                        i += 1
                    command = latex[start:i]

                    # If command has arguments like \mathbb{R}, process them
                    if i < len(latex) and latex[i] == '{':
                        # Skip the argument
                        brace_count = 1
                        i += 1
                        while i < len(latex) and brace_count > 0:
                            if latex[i] == '\\' and i + 1 < len(latex):
                                i += 2
                                continue
                            if latex[i] == '{':
                                brace_count += 1
                            elif latex[i] == '}':
                                brace_count -= 1
                            i += 1

                    # After command (and its arguments), check if there's more content
                    if i < len(latex):
                        # Skip whitespace
                        j = i
                        while j < len(latex) and latex[j].isspace():
                            j += 1
                        # If next char is alphanumeric or another command, might be an issue
                        if j < len(latex) and (latex[j].isalnum() or latex[j] == '\\'):
                            # Check for common patterns that should be braced
                            # Look ahead to see if this looks like multiple elements
                            lookahead = latex[i:i+10]
                            # If we see letters/numbers after a command without braces, flag it
                            if re.match(r'[A-Za-z0-9]', lookahead):
                                issues.append(f"Superscript/subscript at position {start-1} may need braces: '{latex[start-1:i+5]}'")
                else:
                    # Single character - might be OK, but check for certain patterns
                    start = i
                    # Count how many non-space, non-operator chars follow
                    char_count = 0
                    has_uppercase = False
                    while i < len(latex) and latex[i] not in ('^', '_', ' ', '\t', '\n', '{', '}', '(', ')', '[', ']', '+', '-', '*', '/', '=', '<', '>', ',', '.', '|', '\\'):
                        if latex[i].isupper():
                            has_uppercase = True
                        char_count += 1
                        i += 1

                    # If more than 1 char without braces, that's an issue
                    if char_count > 1:
                        issues.append(f"Multi-character {operator} at position {start-1} needs braces: '{latex[start-1:i]}'")
                    # Single uppercase letter as superscript might indicate a set/space (like R^V for R to the V dimension)
                    # However, some patterns like ^T (transpose) are acceptable
                    elif char_count == 1 and has_uppercase:
                        # Common acceptable patterns: ^T (transpose), ^H (Hermitian), ^* (not uppercase but similar)
                        # Skip these common patterns
                        if latex[start] in ('T', 'H'):
                            # ^T and ^H are commonly used for transpose/Hermitian without braces
                            pass
                        # For other single uppercase letters, check if at end or followed by delimiter
                        elif i >= len(latex) or latex[i] in (' ', '\t', '\n', '$', ',', '.', ')', '}', ']'):
                            issues.append(f"Single uppercase {operator}{latex[start]} at position {start-1} should be braced for clarity: '{latex[start-1:i]}'")

            else:
                i += 1

        if issues:
            return "; ".join(issues)
        return None

    def check_common_latex_errors(self, latex: str) -> Optional[str]:
        r"""
        Check for common LaTeX syntax errors.
        Returns error message if errors found, None if valid.
        """
        errors = []

        # Check for \frac with missing arguments
        # \frac should be followed by two brace groups: \frac{...}{...}
        # We need to properly count braces to find the end of the first argument
        i = 0
        while i < len(latex):
            if latex[i:i+5] == r'\frac':
                frac_start = i
                i += 5

                # Skip whitespace
                while i < len(latex) and latex[i].isspace():
                    i += 1

                # First argument must be in braces
                if i >= len(latex) or latex[i] != '{':
                    errors.append(f"\\frac at position {frac_start} missing first argument in braces")
                    continue

                # Count braces to find end of first argument
                brace_count = 1
                i += 1
                while i < len(latex) and brace_count > 0:
                    if latex[i] == '\\' and i + 1 < len(latex):
                        i += 2
                        continue
                    if latex[i] == '{':
                        brace_count += 1
                    elif latex[i] == '}':
                        brace_count -= 1
                    i += 1

                if brace_count != 0:
                    # Unclosed first argument
                    continue

                # Skip whitespace after first argument
                while i < len(latex) and latex[i].isspace():
                    i += 1

                # Second argument must be in braces
                if i >= len(latex) or latex[i] != '{':
                    errors.append(f"\\frac at position {frac_start} missing second argument in braces")
            else:
                i += 1

        # Check for \sqrt with improper usage
        # \sqrt can have optional [n] for nth root: \sqrt[n]{...} or \sqrt{...}
        i = 0
        while i < len(latex):
            if latex[i:i+5] == r'\sqrt':
                sqrt_start = i
                i += 5

                # Skip whitespace
                while i < len(latex) and latex[i].isspace():
                    i += 1

                # Check for optional [n] argument
                if i < len(latex) and latex[i] == '[':
                    # Find closing bracket
                    i += 1
                    while i < len(latex) and latex[i] != ']':
                        if latex[i] == '\\' and i + 1 < len(latex):
                            i += 2
                            continue
                        i += 1
                    if i < len(latex):
                        i += 1  # Skip closing ]

                    # Skip whitespace after optional argument
                    while i < len(latex) and latex[i].isspace():
                        i += 1

                # Argument should be in braces or be a single token
                if i < len(latex):
                    if latex[i] not in ('{', '\\') and not latex[i].isalnum():
                        errors.append(f"\\sqrt at position {sqrt_start} needs an argument")
            else:
                i += 1

        if errors:
            return "; ".join(errors)
        return None

    def check_latex_outside_math_mode(self, content: str, file_path: Path) -> None:
        r"""
        Check for LaTeX syntax that appears outside of math mode delimiters.
        This includes:
        - LaTeX commands like \alpha, \beta, \times, \frac, etc.
        - Subscripts and superscripts outside math mode (e.g., x_i, 2^n)
        - Mathematical symbols like ∈, ⊙, Σ, ≈, etc. combined with LaTeX-like syntax
        """
        lines = content.split('\n')

        # Track code blocks and math blocks to avoid false positives
        in_code_block = False
        in_math_block = False
        code_block_pattern = re.compile(r'^```')

        # Common LaTeX commands that should be in math mode
        latex_commands = [
            r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\epsilon', r'\\zeta',
            r'\\eta', r'\\theta', r'\\iota', r'\\kappa', r'\\lambda', r'\\mu',
            r'\\nu', r'\\xi', r'\\pi', r'\\rho', r'\\sigma', r'\\tau',
            r'\\upsilon', r'\\phi', r'\\chi', r'\\psi', r'\\omega',
            r'\\Alpha', r'\\Beta', r'\\Gamma', r'\\Delta', r'\\Epsilon', r'\\Zeta',
            r'\\Eta', r'\\Theta', r'\\Iota', r'\\Kappa', r'\\Lambda', r'\\Mu',
            r'\\Nu', r'\\Xi', r'\\Pi', r'\\Rho', r'\\Sigma', r'\\Tau',
            r'\\Upsilon', r'\\Phi', r'\\Chi', r'\\Psi', r'\\Omega',
            r'\\times', r'\\div', r'\\pm', r'\\mp', r'\\cdot',
            r'\\frac', r'\\sqrt', r'\\sum', r'\\prod', r'\\int',
            r'\\partial', r'\\nabla', r'\\infty', r'\\forall', r'\\exists',
            r'\\in', r'\\notin', r'\\subset', r'\\subseteq', r'\\supset', r'\\supseteq',
            r'\\cup', r'\\cap', r'\\wedge', r'\\vee', r'\\neg',
            r'\\mathbb', r'\\mathcal', r'\\mathbf', r'\\mathrm', r'\\text',
            r'\\left', r'\\right', r'\\begin', r'\\end',
            r'\\leq', r'\\geq', r'\\neq', r'\\approx', r'\\equiv',
        ]

        # Compile pattern for LaTeX commands
        latex_cmd_pattern = re.compile('(' + '|'.join(latex_commands) + r')\b')

        for line_num, line in enumerate(lines, 1):
            # Track code blocks and math blocks
            stripped_line = line.strip()
            if code_block_pattern.match(stripped_line):
                # Check if it's opening a math block
                if stripped_line.startswith('```math'):
                    in_math_block = True
                    continue
                # Check if closing any block
                elif stripped_line == '```':
                    if in_math_block:
                        in_math_block = False
                    elif in_code_block:
                        in_code_block = False
                    continue
                # Opening a non-math code block
                else:
                    in_code_block = True
                    continue

            # Skip lines inside code blocks or math blocks
            if in_code_block or in_math_block:
                continue

            # Remove inline math ($...$) and display math ($$...$$) from the line
            # to check what remains
            line_without_math = line

            # Remove $$...$$ first
            line_without_math = re.sub(r'\$\$[^$]*\$\$', '', line_without_math)

            # Remove $...$ (but not escaped \$)
            line_without_math = re.sub(r'(?<!\\)\$[^$]+\$', '', line_without_math)

            # Also remove ```math blocks (though we already skip code blocks)
            # This is for inline references

            # Check for LaTeX commands outside math mode
            matches = latex_cmd_pattern.finditer(line_without_math)
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 20)
                end = min(len(line_without_math), match.end() + 20)
                context = line_without_math[start:end].strip()

                self.errors.append(LatexError(
                    file_path=file_path,
                    line_num=line_num,
                    error_type="LaTeX command outside math mode",
                    message=f"LaTeX command '{match.group(0)}' found outside math mode delimiters ($...$)",
                    latex_snippet=context
                ))

            # Check for subscripts/superscripts that look like math
            # Pattern: word_letter or word^letter (not in URLs, not in code)
            # Avoid false positives in variable_names, URLs, etc.

            # Pattern for potential math subscripts: single_letter or word_digit
            subscript_pattern = re.compile(r'(?<![a-zA-Z_])([a-zA-Z])_([a-zA-Z0-9]+)(?![a-zA-Z0-9_])')
            superscript_pattern = re.compile(r'(?<![a-zA-Z_])([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)(?![a-zA-Z0-9_])')

            # Check subscripts
            for match in subscript_pattern.finditer(line_without_math):
                # Additional heuristics to reduce false positives
                full_match = match.group(0)

                # Skip if it looks like part of a longer identifier
                # Skip URLs
                if 'http://' in line or 'https://' in line:
                    continue

                # Check if this is in a context that looks like math
                # (e.g., followed by math symbols or in a bullet point describing math)
                context_start = max(0, match.start() - 10)
                context_end = min(len(line_without_math), match.end() + 10)
                context = line_without_math[context_start:context_end]

                # If the subscript looks like math notation (short subscript)
                if len(match.group(2)) <= 3:  # Short subscripts are more likely math
                    self.errors.append(LatexError(
                        file_path=file_path,
                        line_num=line_num,
                        error_type="Subscript outside math mode",
                        message=f"Subscript '{full_match}' should be in math mode: ${full_match}$",
                        latex_snippet=context.strip()
                    ))

            # Check superscripts
            for match in superscript_pattern.finditer(line_without_math):
                full_match = match.group(0)

                # Skip URLs
                if 'http://' in line or 'https://' in line:
                    continue

                context_start = max(0, match.start() - 10)
                context_end = min(len(line_without_math), match.end() + 10)
                context = line_without_math[context_start:context_end]

                # Superscripts are more likely to be math
                self.errors.append(LatexError(
                    file_path=file_path,
                    line_num=line_num,
                    error_type="Superscript outside math mode",
                    message=f"Superscript '{full_match}' should be in math mode: ${full_match}$",
                    latex_snippet=context.strip()
                ))

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

        # Check superscript/subscript bracing
        error = self.check_superscript_subscript_braces(latex)
        if error:
            self.errors.append(LatexError(
                file_path=file_path,
                line_num=line_num,
                error_type="Superscript/subscript bracing",
                message=error,
                latex_snippet=latex[:100]
            ))

        # Check common LaTeX errors
        error = self.check_common_latex_errors(latex)
        if error:
            self.errors.append(LatexError(
                file_path=file_path,
                line_num=line_num,
                error_type="Common LaTeX errors",
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

        # Check for LaTeX syntax outside math mode
        self.check_latex_outside_math_mode(content, file_path)

        # Extract and validate math blocks
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
