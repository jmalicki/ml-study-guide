#!/usr/bin/env python3
"""Check SVG files for text contrast accessibility issues."""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# WCAG 2.1 contrast ratios
# Normal text: 4.5:1 (AA), 7:1 (AAA)
# Large text (18pt+): 3:1 (AA), 4.5:1 (AAA)
# Graphics: 3:1 (AA)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_luminance(rgb: Tuple[int, int, int]) -> float:
    """Calculate relative luminance according to WCAG formula."""
    def adjust(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [adjust(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color1: str, color2: str) -> float:
    """Calculate contrast ratio between two colors."""
    lum1 = rgb_to_luminance(hex_to_rgb(color1))
    lum2 = rgb_to_luminance(hex_to_rgb(color2))
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def is_light_color(color: str) -> bool:
    """Determine if a color is light (luminance > 0.5)."""
    return rgb_to_luminance(hex_to_rgb(color)) > 0.5


def normalize_color(color: str) -> str:
    """Normalize color to hex format."""
    color_map = {
        'white': '#ffffff',
        'black': '#000000',
        'red': '#ff0000',
        'green': '#008000',
        'blue': '#0000ff',
        'yellow': '#ffff00',
        'cyan': '#00ffff',
        'magenta': '#ff00ff',
    }
    color = color.lower().strip()
    if color in color_map:
        return color_map[color]
    if color.startswith('#'):
        return color
    return color


def extract_fill_from_style(style: str) -> Optional[str]:
    """Extract fill color from style attribute."""
    match = re.search(r'fill:\s*([#\w]+)', style)
    if match:
        return normalize_color(match.group(1))
    return None


def has_important_in_style(style: str) -> bool:
    """Check if style has !important flag."""
    return '!important' in style


def get_element_fill(elem: ET.Element) -> Optional[str]:
    """Get fill color from element, checking both fill attr and style."""
    # Check inline style first
    style = elem.get('style', '')
    if style:
        fill = extract_fill_from_style(style)
        if fill:
            return fill

    # Check fill attribute
    fill = elem.get('fill', '')
    if fill and fill != 'none':
        return normalize_color(fill)

    return None


def find_background_at_position(root: ET.Element, x: float, y: float) -> Optional[Tuple[str, str]]:
    """Find colored background shape at given position.

    Returns (fill_color, element_type) or None.
    """
    # SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Find all rectangles, circles, and paths with fill
    for shape_type in ['rect', 'circle', 'ellipse', 'path', 'polygon']:
        for shape in root.iter('{http://www.w3.org/2000/svg}' + shape_type):
            fill = get_element_fill(shape)
            if not fill or fill == 'none':
                continue

            # Check if position is within shape bounds (simplified)
            if shape_type == 'rect':
                sx = float(shape.get('x', 0))
                sy = float(shape.get('y', 0))
                width = float(shape.get('width', 0))
                height = float(shape.get('height', 0))

                if sx <= x <= sx + width and sy <= y <= sy + height:
                    return (fill, shape_type)

        # Also check without namespace
        for shape in root.iter(shape_type):
            fill = get_element_fill(shape)
            if not fill or fill == 'none':
                continue

            if shape_type == 'rect':
                sx = float(shape.get('x', 0))
                sy = float(shape.get('y', 0))
                width = float(shape.get('width', 0))
                height = float(shape.get('height', 0))

                if sx <= x <= sx + width and sy <= y <= sy + height:
                    return (fill, shape_type)

    return None


def has_dark_mode_css(content: str) -> bool:
    """Check if SVG has dark mode CSS media query."""
    return 'prefers-color-scheme: dark' in content


def check_svg_file(svg_path: Path) -> List[str]:
    """Check a single SVG file for contrast issues.

    Returns list of error/warning messages.
    """
    issues = []
    content = svg_path.read_text()

    # Check for dark mode CSS
    has_dark_mode = has_dark_mode_css(content)

    try:
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Find all text elements
        text_elements = []
        for text in root.iter('{http://www.w3.org/2000/svg}text'):
            text_elements.append(text)
        for text in root.iter('text'):
            text_elements.append(text)

        for text_elem in text_elements:
            # Get text position
            x = float(text_elem.get('x', 0))
            y = float(text_elem.get('y', 0))

            # Get text fill color
            text_fill = get_element_fill(text_elem)
            if not text_fill:
                continue

            # Check if text has inline style with !important
            style = text_elem.get('style', '')
            has_important = has_important_in_style(style)

            # Find background at text position
            bg_info = find_background_at_position(root, x, y)

            if bg_info:
                bg_color, shape_type = bg_info

                # Text is on a colored background
                try:
                    ratio = contrast_ratio(text_fill, bg_color)

                    # Check contrast ratio (use 3:1 for graphics/UI elements)
                    if ratio < 3.0:
                        issues.append(
                            f"  ERROR: Low contrast at ({x:.0f}, {y:.0f}): "
                            f"text {text_fill} on {bg_color} background "
                            f"(ratio: {ratio:.2f}:1, need 3:1)"
                        )

                    # Warn if no !important (might be overridden by dark mode CSS)
                    if not has_important and has_dark_mode:
                        issues.append(
                            f"  WARNING: Text at ({x:.0f}, {y:.0f}) on colored background "
                            f"should use 'style=\"fill: {text_fill} !important\"' "
                            f"to prevent dark mode override"
                        )
                except (ValueError, ZeroDivisionError):
                    pass  # Skip invalid colors
            else:
                # Text is NOT on a colored background
                # Should rely on CSS for dark/light mode
                if not has_dark_mode:
                    issues.append(
                        f"  WARNING: Text at ({x:.0f}, {y:.0f}) without colored background "
                        f"but SVG has no dark mode CSS"
                    )

    except ET.ParseError as e:
        issues.append(f"  ERROR: Failed to parse SVG: {e}")

    return issues


def main():
    """Check all SVG files in assets/diagrams/."""
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
        print("SVG Contrast Issues Found:")
        print('\n'.join(all_issues))
        print(f"\n{len(files_with_issues)} file(s) with contrast issues")

        # Count errors vs warnings
        error_count = sum(1 for issue in all_issues if 'ERROR' in issue)
        warning_count = sum(1 for issue in all_issues if 'WARNING' in issue)

        print(f"Errors: {error_count}, Warnings: {warning_count}")

        if error_count > 0:
            return 1
        else:
            print("\nNo critical errors - warnings only")
            return 0
    else:
        print("All SVG files passed contrast checks!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
