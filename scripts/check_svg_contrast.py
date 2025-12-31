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


def find_background_at_position(root: ET.Element, x: float, y: float) -> Optional[Tuple[str, str, ET.Element]]:
    """Find colored background shape at given position.

    Returns (fill_color, element_type, element) or None.
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
                    return (fill, shape_type, shape)

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
                    return (fill, shape_type, shape)

    return None


def has_dark_mode_css(content: str) -> bool:
    """Check if SVG has dark mode CSS media query."""
    return 'prefers-color-scheme: dark' in content


def extract_css_from_svg(content: str) -> str:
    """Extract CSS content from SVG style tags."""
    css = ""
    # Find all <style> tags
    style_matches = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
    for match in style_matches:
        css += match + "\n"
    return css


def parse_dark_mode_css(css: str) -> Dict[str, Dict[str, str]]:
    """Parse dark mode CSS and extract selector properties.

    Returns dict of {selector: {property: value}}
    """
    dark_mode_rules = {}

    # Find @media (prefers-color-scheme: dark) blocks
    # Match from @media to the closing brace, handling nested braces
    media_pattern = r'@media\s*\([^)]*prefers-color-scheme:\s*dark[^)]*\)\s*\{((?:[^{}]|\{[^}]*\})*)\}'
    media_matches = re.findall(media_pattern, css, re.DOTALL)

    for media_content in media_matches:
        # Parse CSS rules within the media query
        # Match selector { property: value; ... }
        rule_pattern = r'([^{}/]+)\{([^}]+)\}'
        rules = re.findall(rule_pattern, media_content)

        for selector, properties in rules:
            selector = selector.strip()
            # Skip comments
            if selector.startswith('/*'):
                continue
            if selector not in dark_mode_rules:
                dark_mode_rules[selector] = {}

            # Parse properties
            prop_pattern = r'([a-z-]+)\s*:\s*([^;!]+)(?:\s*!important)?'
            props = re.findall(prop_pattern, properties)
            for prop_name, prop_value in props:
                dark_mode_rules[selector][prop_name.strip()] = prop_value.strip()

    return dark_mode_rules


def parse_regular_css(css: str) -> Dict[str, Dict[str, str]]:
    """Parse regular (non-dark-mode) CSS rules.

    Returns dict of {selector: {property: value}}
    """
    regular_rules = {}

    # Remove @media blocks first
    css_without_media = re.sub(r'@media[^{]*\{(?:[^{}]|\{[^}]*\})*\}', '', css, flags=re.DOTALL)

    # Parse remaining CSS rules
    rule_pattern = r'([^{}/]+)\{([^}]+)\}'
    rules = re.findall(rule_pattern, css_without_media)

    for selector, properties in rules:
        selector = selector.strip()
        # Skip comments
        if selector.startswith('/*'):
            continue
        if selector not in regular_rules:
            regular_rules[selector] = {}

        # Parse properties
        prop_pattern = r'([a-z-]+)\s*:\s*([^;!]+)(?:\s*!important)?'
        props = re.findall(prop_pattern, properties)
        for prop_name, prop_value in props:
            regular_rules[selector][prop_name.strip()] = prop_value.strip()

    return regular_rules


def is_light_color_value(color_value: str) -> bool:
    """Check if a color value is light (white, light gray, etc)."""
    light_colors = ['white', '#fff', '#ffffff', '#f5f5f5', '#fafafa', '#f0f0f0',
                    '#eeeeee', '#e0e0e0', '#e5e5e5', '#f9f9f9']
    color_normalized = color_value.lower().strip()

    # Check exact matches
    if color_normalized in light_colors:
        return True

    # Check if it's a light hex color (high luminance)
    if color_normalized.startswith('#'):
        try:
            return is_light_color(color_normalized)
        except:
            pass

    return False


def is_dark_color_value(color_value: str) -> bool:
    """Check if a color value is dark."""
    dark_colors = ['black', '#000', '#000000', '#111', '#1a1a1a', '#222', '#333']
    color_normalized = color_value.lower().strip()

    # Check exact matches
    if color_normalized in dark_colors:
        return True

    # Check if it's a dark hex color (low luminance)
    if color_normalized.startswith('#'):
        try:
            return not is_light_color(color_normalized)
        except:
            pass

    return False


def element_has_dark_mode_rule(elem: ET.Element, dark_mode_rules: Dict[str, Dict[str, str]]) -> bool:
    """Check if a specific element has a dark mode CSS rule that changes its fill to dark."""
    # Check if element has a class attribute
    class_attr = elem.get('class', '')
    if class_attr:
        # Check if any dark mode rule matches this class
        for selector, props in dark_mode_rules.items():
            if 'fill' in props:
                # Check if selector matches this class
                if f'.{class_attr}' in selector or class_attr in selector:
                    if is_dark_color_value(props['fill']):
                        return True

    # Check if element has an id attribute
    id_attr = elem.get('id', '')
    if id_attr:
        # Check if any dark mode rule matches this id
        for selector, props in dark_mode_rules.items():
            if 'fill' in props:
                if f'#{id_attr}' in selector or id_attr in selector:
                    if is_dark_color_value(props['fill']):
                        return True

    # Check if element's fill color is specifically targeted
    # e.g., rect[fill="#4A90A4"]
    elem_fill = get_element_fill(elem)
    if elem_fill:
        elem_tag = elem.tag.replace('{http://www.w3.org/2000/svg}', '')
        for selector, props in dark_mode_rules.items():
            if 'fill' in props:
                # Check for attribute selectors like rect[fill="#4A90A4"]
                if elem_tag in selector and elem_fill.lower() in selector.lower():
                    if is_dark_color_value(props['fill']):
                        return True

    return False


def check_dark_mode_consistency(content: str) -> List[str]:
    """Check for dark mode CSS consistency issues.

    Specifically checks if text changes to white in dark mode but backgrounds stay light.
    """
    issues = []

    if not has_dark_mode_css(content):
        return issues

    # Extract and parse CSS
    css = extract_css_from_svg(content)
    dark_mode_rules = parse_dark_mode_css(css)
    regular_rules = parse_regular_css(css)

    # Check if text color changes to white/light in dark mode
    text_selectors_going_white = []
    for selector, props in dark_mode_rules.items():
        if 'fill' in props:
            fill_value = props['fill']
            if is_light_color_value(fill_value):
                text_selectors_going_white.append(selector)

    if not text_selectors_going_white:
        return issues

    # Now check if there are background elements that change to dark
    # Only consider selectors that are likely backgrounds (not text)
    background_selectors_going_dark = []
    for selector, props in dark_mode_rules.items():
        if 'fill' in props:
            fill_value = props['fill']
            # Skip text selectors - we only care about backgrounds
            if 'text' in selector.lower():
                continue
            # Check for background-like selectors
            if is_dark_color_value(fill_value):
                background_selectors_going_dark.append(selector)

    # Check for hardcoded light backgrounds in the SVG
    # Look for rect/path/etc with fill="white" or fill="#fff" etc. in XML
    # This matches white, #fff, #ffffff, #f5f5f5, #fafafa, #eee, #eeeeee, etc.
    hardcoded_light_bg_pattern = r'<(?:rect|path|circle|ellipse|polygon)[^>]*fill\s*=\s*["\']?\s*(?:white|#(?:fff(?:fff)?|[ef][0-9a-f][ef][0-9a-f][ef][0-9a-f]|[ef]{3}(?:[ef]{3})?))\s*["\']?[^>]*>'
    has_hardcoded_light_bg = bool(re.search(hardcoded_light_bg_pattern, content, re.IGNORECASE))

    # Also check for light backgrounds in regular CSS
    regular_light_backgrounds = []
    for selector, props in regular_rules.items():
        if 'fill' in props:
            fill_value = props['fill']
            if is_light_color_value(fill_value):
                # Check if this selector changes to dark in dark mode
                if selector not in background_selectors_going_dark:
                    regular_light_backgrounds.append(selector)

    # Report error if we have text going white but no backgrounds going dark
    # OR if we have hardcoded light backgrounds
    if text_selectors_going_white:
        if not background_selectors_going_dark and (has_hardcoded_light_bg or regular_light_backgrounds):
            issues.append(
                f"  ERROR: Dark mode makes text white but backgrounds stay light - text will be invisible"
            )
            issues.append(
                f"    Text selectors going white in dark mode: {', '.join(text_selectors_going_white)}"
            )
            if has_hardcoded_light_bg:
                issues.append(
                    f"    Found hardcoded light backgrounds in SVG elements (not using CSS)"
                )
            if regular_light_backgrounds:
                issues.append(
                    f"    Light background selectors not changing in dark mode: {', '.join(regular_light_backgrounds)}"
                )
        elif not background_selectors_going_dark and not has_hardcoded_light_bg and not regular_light_backgrounds:
            # Text goes white but we can't find backgrounds - might be transparent
            # This could be OK if the background is truly transparent, but warn anyway
            issues.append(
                f"  WARNING: Dark mode makes text white - ensure backgrounds also change to dark"
            )
            issues.append(
                f"    Text selectors going white: {', '.join(text_selectors_going_white)}"
            )

    return issues


def check_svg_file(svg_path: Path) -> List[str]:
    """Check a single SVG file for contrast issues.

    Returns list of error/warning messages.
    """
    issues = []
    content = svg_path.read_text()

    # Check for dark mode consistency issues
    dark_mode_issues = check_dark_mode_consistency(content)
    issues.extend(dark_mode_issues)

    # Check for dark mode CSS
    has_dark_mode = has_dark_mode_css(content)

    # Parse CSS to check for dark mode rules
    css = extract_css_from_svg(content)
    dark_mode_rules = parse_dark_mode_css(css) if has_dark_mode else {}

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
                bg_color, shape_type, bg_elem = bg_info

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

                    # NEW CHECK: Detect white/light text on colored backgrounds that don't adapt in dark mode
                    if has_dark_mode and is_light_color_value(text_fill):
                        # Check if this text will turn white in dark mode
                        text_will_be_white_in_dark_mode = False

                        # Check if dark mode CSS makes text white
                        for selector, props in dark_mode_rules.items():
                            if 'fill' in props and is_light_color_value(props['fill']):
                                # Check if selector applies to this text element
                                # Common selectors: text, text[fill="#fff"], text[fill="#ffffff"], etc.
                                if selector == 'text' or 'text[fill' in selector:
                                    text_will_be_white_in_dark_mode = True
                                    break

                        # If text will be white in dark mode, check if background also adapts
                        if text_will_be_white_in_dark_mode:
                            # Check if THIS SPECIFIC background element has a dark mode rule
                            bg_has_dark_mode_rule = element_has_dark_mode_rule(bg_elem, dark_mode_rules)

                            # If background doesn't change to dark, this is a problem
                            if not bg_has_dark_mode_rule:
                                issues.append(
                                    f"  ERROR: Dark mode contrast issue at ({x:.0f}, {y:.0f}): "
                                    f"text will be white in dark mode but {bg_color} background "
                                    f"doesn't change to dark - text will be invisible on light page backgrounds"
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
