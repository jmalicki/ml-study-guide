#!/usr/bin/env python3
"""Check SVG files for typography and formula best practices.

Checks:
1. Non-standard fonts should have @font-face with CDN fallback
2. Text elements should have adequate vertical spacing (font-size + 4px minimum)
3. Mathematical formulas should use serif fonts (STIX Two Math preferred)
4. Fractions should use proper layout (numerator/bar/denominator) not "/"
5. Delimiters (parens, brackets) around fractions should scale to fraction height
6. Sum/product symbols should scale to match fraction height

Usage:
    python3 check_svg_typography.py [--strict] [files...]

Options:
    --strict    Treat warnings as errors (exit code 1 if any warnings)
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass


# Standard web-safe fonts that don't need CDN
STANDARD_FONTS = {
    'system-ui', '-apple-system', 'sans-serif', 'serif', 'monospace',
    'arial', 'helvetica', 'times new roman', 'times', 'georgia',
    'courier', 'courier new', 'verdana', 'tahoma', 'trebuchet ms',
    'impact', 'comic sans ms', 'lucida console', 'lucida sans unicode',
    'palatino linotype', 'book antiqua', 'palatino',
}

# Math fonts that should be loaded from CDN
MATH_FONTS = {
    'stix two math', 'stix two text', 'latin modern math',
    'cambria math', 'dejavu serif', 'tex gyre termes',
}

# Characters that indicate mathematical content
MATH_INDICATORS = set('∑∏∫√∂∇∆αβγδεζηθικλμνξπρστυφχψω∈∉⊂⊃∩∪∀∃¬∧∨⊕⊗≤≥≠≈≡±×÷')

# Subscript/superscript unicode ranges
SUBSCRIPT_CHARS = set('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ')
SUPERSCRIPT_CHARS = set('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ')


@dataclass
class TextElement:
    """Represents a text element in SVG."""
    x: float
    y: float
    font_family: str
    font_size: float
    content: str
    element: ET.Element


@dataclass
class FormulaContext:
    """Context for analyzing a formula region."""
    elements: List[TextElement]
    has_fraction: bool = False
    fraction_height: float = 0
    delimiter_sizes: List[float] = None
    sum_product_sizes: List[float] = None


def parse_font_family(font_family: str) -> List[str]:
    """Parse font-family string into list of font names."""
    fonts = []
    for font in font_family.split(','):
        font = font.strip().strip('"').strip("'").lower()
        if font:
            fonts.append(font)
    return fonts


def extract_font_face_fonts(css: str) -> Set[str]:
    """Extract font names defined in @font-face rules."""
    fonts = set()
    # Match @font-face { font-family: 'Font Name'; ... }
    pattern = r"@font-face\s*\{[^}]*font-family:\s*['\"]?([^'\";\}]+)['\"]?"
    for match in re.finditer(pattern, css, re.IGNORECASE | re.DOTALL):
        fonts.add(match.group(1).lower().strip())
    return fonts


def has_cdn_url(css: str) -> bool:
    """Check if CSS contains CDN URLs for fonts."""
    cdn_patterns = [
        r'cdn\.jsdelivr\.net',
        r'fonts\.googleapis\.com',
        r'cdnjs\.cloudflare\.com',
        r'unpkg\.com',
    ]
    return any(re.search(p, css) for p in cdn_patterns)


def extract_css(content: str) -> str:
    """Extract CSS from SVG style tags."""
    css = ""
    for match in re.finditer(r'<style[^>]*>(.*?)</style>', content, re.DOTALL):
        css += match.group(1) + "\n"
    return css


def get_text_elements(root: ET.Element) -> List[TextElement]:
    """Extract all text elements from SVG."""
    elements = []

    for text in list(root.iter('{http://www.w3.org/2000/svg}text')) + list(root.iter('text')):
        try:
            x = float(text.get('x', 0))
            y = float(text.get('y', 0))
        except ValueError:
            continue

        font_family = text.get('font-family', 'sans-serif')

        # Check style attribute for font-family
        style = text.get('style', '')
        style_font = re.search(r'font-family:\s*([^;]+)', style)
        if style_font:
            font_family = style_font.group(1)

        try:
            font_size = float(text.get('font-size', '12').rstrip('px'))
        except ValueError:
            font_size = 12

        # Get text content including tspan children
        content = ''.join(text.itertext())

        elements.append(TextElement(
            x=x, y=y, font_family=font_family,
            font_size=font_size, content=content, element=text
        ))

    return elements


def is_math_content(text: str) -> bool:
    """Check if text contains mathematical symbols."""
    if not text:
        return False
    # Check for math indicators
    if any(c in MATH_INDICATORS for c in text):
        return True
    # Check for subscripts/superscripts
    if any(c in SUBSCRIPT_CHARS or c in SUPERSCRIPT_CHARS for c in text):
        return True
    # Check for common math patterns
    math_patterns = [
        r'[a-z]_[a-z]',  # subscript notation
        r'\b(softmax|sigmoid|tanh|relu|exp|log|sin|cos|tan)\b',
        r'[=≈≡<>≤≥±×÷]',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in math_patterns)


def check_font_cdn(content: str) -> List[str]:
    """Check that non-standard fonts have CDN fallback."""
    issues = []
    css = extract_css(content)

    # Get fonts defined in @font-face
    defined_fonts = extract_font_face_fonts(css)
    has_cdn = has_cdn_url(css)

    # Find all font-family declarations in the SVG
    used_fonts = set()

    # From attributes
    for match in re.finditer(r'font-family\s*=\s*["\']([^"\']+)["\']', content):
        for font in parse_font_family(match.group(1)):
            used_fonts.add(font)

    # From CSS
    for match in re.finditer(r'font-family:\s*([^;}\n]+)', css):
        for font in parse_font_family(match.group(1)):
            used_fonts.add(font)

    # Check each non-standard font
    for font in used_fonts:
        if font not in STANDARD_FONTS:
            if font not in defined_fonts:
                issues.append(
                    f"  WARNING: Non-standard font '{font}' used without @font-face definition"
                )
            elif not has_cdn:
                issues.append(
                    f"  WARNING: Font '{font}' has @font-face but no CDN URL - may not load for all users"
                )

    return issues


def is_formula_symbol(text: str) -> bool:
    """Check if text is a large formula symbol (sum, product, integral, etc.)."""
    # These symbols are often intentionally larger and positioned differently
    formula_symbols = set('∑∏∫∮∬∭')
    return any(c in formula_symbols for c in text)


def check_vertical_spacing(elements: List[TextElement]) -> List[str]:
    """Check that text elements have adequate vertical spacing."""
    issues = []

    # Group elements by approximate x position (within 50px = same column)
    columns: Dict[int, List[TextElement]] = {}
    for elem in elements:
        col_key = int(elem.x // 50)
        if col_key not in columns:
            columns[col_key] = []
        columns[col_key].append(elem)

    for col_key, col_elements in columns.items():
        # Sort by y position
        sorted_elems = sorted(col_elements, key=lambda e: e.y)

        for i in range(len(sorted_elems) - 1):
            curr = sorted_elems[i]
            next_elem = sorted_elems[i + 1]

            # Skip if either element is a large formula symbol (intentionally sized differently)
            if is_formula_symbol(curr.content) or is_formula_symbol(next_elem.content):
                continue

            # Skip if elements are at very different x positions (not really in same column)
            if abs(curr.x - next_elem.x) > 30:
                continue

            # Calculate expected minimum spacing
            # Minimum = max font size + 4px breathing room
            min_spacing = max(curr.font_size, next_elem.font_size) + 4
            actual_spacing = next_elem.y - curr.y

            # Allow some tolerance for intentional tight spacing (like fractions)
            if actual_spacing < min_spacing * 0.7 and actual_spacing > 0:
                # Check if this might be intentional (fraction numerator/denominator)
                if abs(next_elem.x - curr.x) < 20:  # Same horizontal position
                    # Likely a fraction - skip
                    continue

                # Skip subscript-like elements (small text positioned slightly below)
                if next_elem.font_size < curr.font_size * 0.8:
                    continue

                issues.append(
                    f"  WARNING: Tight vertical spacing at y={curr.y:.0f}->{next_elem.y:.0f} "
                    f"({actual_spacing:.0f}px for font-size {max(curr.font_size, next_elem.font_size):.0f}px, "
                    f"recommend {min_spacing:.0f}px minimum)"
                )

    return issues


def check_math_fonts(elements: List[TextElement]) -> List[str]:
    """Check that mathematical content uses appropriate serif fonts."""
    issues = []

    for elem in elements:
        if is_math_content(elem.content):
            fonts = parse_font_family(elem.font_family)

            # Check if any math font is in the font stack
            has_math_font = any(f in MATH_FONTS for f in fonts)
            has_serif = 'serif' in fonts or any('serif' in f for f in fonts)

            if not has_math_font and not has_serif:
                issues.append(
                    f"  WARNING: Math content at ({elem.x:.0f}, {elem.y:.0f}) uses sans-serif font. "
                    f"Consider using STIX Two Math or serif fallback for: '{elem.content[:50]}...'"
                )

    return issues


def check_fraction_syntax(content: str) -> List[str]:
    """Check for inline division that should be proper fractions."""
    issues = []

    # Look for patterns like "x / y" or "a/b" in text elements that appear to be formulas
    # This is a heuristic - we look for "/" surrounded by math-like content

    # Extract text content from SVG
    text_pattern = r'>([^<]*(?:/[^<]*)+)<'

    for match in re.finditer(text_pattern, content):
        text = match.group(1)

        # Skip if it's clearly not a formula (e.g., file paths, dates)
        if re.search(r'\d{1,2}/\d{1,2}', text):  # Date-like
            continue
        if '/' in text and any(c in text for c in '\\:'):  # Path-like
            continue

        # Skip if "/" is inside parentheses with words (like "(index/label)")
        if re.search(r'\([^)]*\w+/\w+[^)]*\)', text):
            continue

        # Skip if "/" is between quoted strings or descriptive text
        if re.search(r'["\']\w*/\w*["\']', text):
            continue

        # Check for math-like division - specifically variable/variable patterns
        if is_math_content(text) and '/' in text:
            # Check if it's a math-like pattern: single letters, √, or short math expressions
            # Pattern: letter(s) or symbol / letter(s) or symbol, possibly with subscripts
            if re.search(r'[a-zA-Z_√][₀-₉ᵢⱼₖ]*\s*/\s*[a-zA-Z_√][₀-₉ᵢⱼₖ]*', text):
                issues.append(
                    f"  WARNING: Consider using proper fraction layout instead of '/': '{text[:60]}'"
                )

    return issues


def find_fraction_regions(root: ET.Element) -> List[Tuple[float, float, float, float]]:
    """Find regions that appear to be fractions (line elements as fraction bars)."""
    fractions = []

    # Find horizontal lines that could be fraction bars
    for line in list(root.iter('{http://www.w3.org/2000/svg}line')) + list(root.iter('line')):
        try:
            x1 = float(line.get('x1', 0))
            y1 = float(line.get('y1', 0))
            x2 = float(line.get('x2', 0))
            y2 = float(line.get('y2', 0))
        except ValueError:
            continue

        # Check if it's horizontal (y values close)
        if abs(y2 - y1) < 2 and abs(x2 - x1) > 20:
            # This looks like a fraction bar
            fractions.append((min(x1, x2), y1, max(x1, x2), y1))

    return fractions


def check_delimiter_scaling(elements: List[TextElement], fractions: List[Tuple[float, float, float, float]]) -> List[str]:
    """Check that delimiters near fractions are scaled appropriately.

    Delimiters include:
    - Parentheses, brackets, braces: () [] {}
    - Sum and product: ∑ (\\sum) and ∏ (\\prod)
    - Integrals: ∫ ∮ ∬ ∭
    """
    issues = []

    # All delimiter characters that should scale with fractions
    SCALING_DELIMITERS = set('()[]{}∑∏∫∮∬∭')

    for fx1, fy, fx2, _ in fractions:
        # Find text elements near this fraction
        nearby_elements = [
            e for e in elements
            if abs(e.y - fy) < 30 and (abs(e.x - fx1) < 30 or abs(e.x - fx2) < 30)
        ]

        # Look for delimiters vs regular content
        delimiters = []
        other_sizes = []

        for elem in nearby_elements:
            if any(c in elem.content for c in SCALING_DELIMITERS):
                delimiters.append(elem)
            else:
                other_sizes.append(elem.font_size)

        if not delimiters or not other_sizes:
            continue

        # Check if delimiters are larger than regular text
        avg_text_size = sum(other_sizes) / len(other_sizes)

        for delim in delimiters:
            # Delimiters should be at least 1.5x the regular text size for proper scaling
            if delim.font_size < avg_text_size * 1.3:
                symbol = ''.join(c for c in delim.content if c in SCALING_DELIMITERS)

                # Give specific advice based on symbol type
                if any(c in symbol for c in '∑∏'):
                    symbol_name = "Sum/product (∑/∏)"
                elif any(c in symbol for c in '∫∮∬∭'):
                    symbol_name = "Integral (∫)"
                else:
                    symbol_name = f"Delimiter '{symbol}'"

                issues.append(
                    f"  WARNING: {symbol_name} at ({delim.x:.0f}, {delim.y:.0f}) "
                    f"may need scaling (size {delim.font_size:.0f}px vs fraction context {avg_text_size:.0f}px). "
                    f"Consider font-size {avg_text_size * 1.8:.0f}px+ for LaTeX-like appearance."
                )

    return issues


def check_svg_file(svg_path: Path) -> List[str]:
    """Check a single SVG file for typography issues."""
    issues = []
    content = svg_path.read_text()

    # Check font CDN usage
    font_issues = check_font_cdn(content)
    issues.extend(font_issues)

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        elements = get_text_elements(root)

        if elements:
            # Check vertical spacing
            spacing_issues = check_vertical_spacing(elements)
            issues.extend(spacing_issues)

            # Check math fonts
            math_issues = check_math_fonts(elements)
            issues.extend(math_issues)

            # Check fraction syntax
            fraction_issues = check_fraction_syntax(content)
            issues.extend(fraction_issues)

            # Find fractions and check delimiter scaling
            fractions = find_fraction_regions(root)
            if fractions:
                delimiter_issues = check_delimiter_scaling(elements, fractions)
                issues.extend(delimiter_issues)

    except ET.ParseError as e:
        issues.append(f"  ERROR: Failed to parse SVG: {e}")

    return issues


def main():
    """Check SVG files for typography issues."""
    parser = argparse.ArgumentParser(
        description='Check SVG files for typography and formula best practices.'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors (exit code 1 if any warnings)'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='SVG files to check (default: all files in assets/diagrams/)'
    )

    args = parser.parse_args()

    if args.files:
        svg_files = [Path(f) for f in args.files]
        for svg_file in svg_files:
            if not svg_file.exists():
                print(f"ERROR: File not found: {svg_file}")
                return 1
    else:
        diagrams_dir = Path(__file__).parent.parent / 'assets' / 'diagrams'
        if not diagrams_dir.exists():
            print(f"ERROR: Directory not found: {diagrams_dir}")
            return 1
        svg_files = sorted(diagrams_dir.glob('*.svg'))

    if not svg_files:
        print("No SVG files found")
        return 0

    all_issues = []
    files_with_issues = []

    for svg_file in svg_files:
        issues = check_svg_file(svg_file)
        if issues:
            files_with_issues.append(svg_file.name)
            all_issues.append(f"\n{svg_file.name}:")
            all_issues.extend(issues)

    if all_issues:
        print("SVG Typography Issues Found:")
        print('\n'.join(all_issues))
        print(f"\n{len(files_with_issues)} file(s) with typography issues")

        error_count = sum(1 for issue in all_issues if 'ERROR' in issue)
        warning_count = sum(1 for issue in all_issues if 'WARNING' in issue)
        info_count = sum(1 for issue in all_issues if 'INFO' in issue)

        print(f"Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}")

        if args.strict:
            if error_count > 0 or warning_count > 0:
                print("\n--strict mode: treating warnings as errors")
                return 1
            return 0
        else:
            if error_count > 0:
                return 1
            return 0
    else:
        print("All SVG files passed typography checks!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
