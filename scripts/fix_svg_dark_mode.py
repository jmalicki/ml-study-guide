#!/usr/bin/env python3
"""Fix SVG files to properly support dark mode backgrounds.

Replaces hardcoded light background fills with CSS classes that have dark mode overrides.
"""
import re
import sys
from pathlib import Path

# Map of light colors to their CSS class names and dark mode equivalents
COLOR_MAP = {
    # White variants
    'white': ('bg-white', 'white', '#2d2d2d'),
    '#fff': ('bg-white', '#fff', '#2d2d2d'),
    '#ffffff': ('bg-white', '#ffffff', '#2d2d2d'),
    # Light blue
    '#e3f2fd': ('bg-blue-light', '#E3F2FD', '#1a237e'),
    '#e1f5fe': ('bg-cyan-light', '#E1F5FE', '#006064'),
    # Light green
    '#e8f5e9': ('bg-green-light', '#E8F5E9', '#1b5e20'),
    # Light red/pink
    '#ffebee': ('bg-red-light', '#FFEBEE', '#b71c1c'),
    # Light yellow/amber/orange
    '#fff8e1': ('bg-amber-light', '#FFF8E1', '#ff6f00'),
    '#fff3e0': ('bg-orange-light', '#FFF3E0', '#e65100'),
    '#fff9e6': ('bg-cream', '#fff9e6', '#5d4037'),
    '#ffcdd2': ('bg-pink-light', '#FFCDD2', '#c62828'),
    # Light indigo
    '#e8eaf6': ('bg-indigo-light', '#E8EAF6', '#1a237e'),
    # Named colors and other backgrounds
    '#87ceeb': ('bg-skyblue', '#87ceeb', '#01579b'),
    '#90ee90': ('bg-lightgreen', '#90ee90', '#1b5e20'),
    '#ffe4b5': ('bg-moccasin', '#ffe4b5', '#e65100'),
    '#dc143c': ('bg-crimson', '#dc143c', '#b71c1c'),
    '#7b1fa2': ('bg-purple-dark', '#7b1fa2', '#4a148c'),
    # Light purple
    '#f3e5f5': ('bg-purple-light', '#F3E5F5', '#4a148c'),
    # Light gray
    '#f5f5f5': ('bg-gray-light', '#f5f5f5', '#424242'),
    '#fafafa': ('bg-gray-light', '#fafafa', '#424242'),
}


def normalize_color(color: str) -> str:
    """Normalize color to lowercase."""
    return color.lower().strip()


def get_class_for_color(color: str) -> tuple:
    """Get CSS class name and dark mode color for a fill color."""
    normalized = normalize_color(color)
    return COLOR_MAP.get(normalized, (None, None, None))


def fix_svg_file(svg_path: Path) -> bool:
    """Fix a single SVG file. Returns True if changes were made."""
    content = svg_path.read_text()
    original_content = content

    classes_needed = {}  # class_name -> (light_color, dark_color)

    # Find all shape elements with hardcoded light fills
    # Pattern matches: <rect ... fill="white" ...> etc.
    def replace_fill(match):
        full_match = match.group(0)
        tag = match.group(1)
        fill_color = match.group(2)

        class_name, light_color, dark_color = get_class_for_color(fill_color)
        if class_name is None:
            return full_match

        classes_needed[class_name] = (light_color, dark_color)

        # Remove fill attribute and add class
        # First, check if there's already a class attribute
        if 'class="' in full_match:
            # Add to existing class
            result = re.sub(r'class="([^"]*)"', f'class="\\1 {class_name}"', full_match)
            # Remove the fill attribute
            result = re.sub(r'\s*fill\s*=\s*"[^"]*"', '', result)
            return result
        else:
            # Replace fill with class
            result = re.sub(r'fill\s*=\s*"[^"]*"', f'class="{class_name}"', full_match)
            return result

    # Match shape elements with fill attribute (any hex color or "white")
    content = re.sub(
        r'(<(?:rect|circle|ellipse|path|polygon)[^>]*)\bfill\s*=\s*"((?:white|#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}))"([^>]*>)',
        replace_fill,
        content,
        flags=re.IGNORECASE
    )

    if not classes_needed:
        return False

    # Now add CSS for the classes we need
    # Find the dark mode section
    dark_mode_match = re.search(
        r'(@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)\s*\{)(.*?)(\n\s*\}\s*\n\s*</style>)',
        content,
        re.DOTALL
    )

    if dark_mode_match:
        # Add classes to existing dark mode section
        start = dark_mode_match.group(1)
        existing = dark_mode_match.group(2)
        end = dark_mode_match.group(3)

        # Build new rules
        new_dark_rules = ""
        for class_name, (light_color, dark_color) in sorted(classes_needed.items()):
            if f'.{class_name}' not in existing:
                new_dark_rules += f"\n      .{class_name} {{ fill: {dark_color}; }}"

        # Replace dark mode section
        new_dark_mode = f"{start}{existing}{new_dark_rules}{end}"
        content = content[:dark_mode_match.start()] + new_dark_mode + content[dark_mode_match.end():]

        # Also add light mode rules before @media
        light_css = ""
        for class_name, (light_color, dark_color) in sorted(classes_needed.items()):
            if f'.{class_name}' not in content[:dark_mode_match.start()]:
                light_css += f"    .{class_name} {{ fill: {light_color}; }}\n"

        if light_css:
            # Find position before @media
            media_pos = content.find('@media')
            if media_pos > 0:
                # Find the preceding newline
                newline_pos = content.rfind('\n', 0, media_pos)
                if newline_pos > 0:
                    content = content[:newline_pos+1] + "\n    /* Background colors */\n" + light_css + content[newline_pos+1:]

    if content != original_content:
        svg_path.write_text(content)
        return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_svg_dark_mode.py <svg_files...>")
        return 1

    fixed_count = 0
    for arg in sys.argv[1:]:
        svg_path = Path(arg)
        if not svg_path.exists():
            print(f"File not found: {svg_path}")
            continue

        if fix_svg_file(svg_path):
            print(f"Fixed: {svg_path.name}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} file(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
