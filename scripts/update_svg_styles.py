#!/usr/bin/env python3
"""
Update SVG styles in the optimization book to match the main LLM book style.

Changes:
1. Remove white/gray background rectangles
2. Add dark mode support via CSS
3. Change font from Arial to system-ui
4. Replace harsh primary colors with softer palette
"""

import re
from pathlib import Path

# Color mappings: old → new
COLOR_MAP = {
    '#f44336': '#E57373',  # red
    '#4CAF50': '#81C784',  # green
    '#2196F3': '#4A90A4',  # blue
    '#FF9800': '#FFB74D',  # orange
    '#fafafa': 'none',     # background → transparent
}

# The dark mode style block to add
DARK_MODE_STYLE = '''  <style>
    /* Light mode (default) */
    text { fill: #000000; }
    .label-text { fill: #666666; }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
      text { fill: #ffffff; }
      .label-text { fill: #cccccc; }

      text[fill="#000"], text[fill="#000000"], text[fill="#333"], text[fill="#666"] {
        fill: #ffffff;
      }
      text[fill="#fff"], text[fill="#ffffff"], text[fill="white"] {
        fill: #ffffff;
      }
      text[fill="#E57373"] { fill: #ef5350; }
      text[fill="#81C784"] { fill: #66bb6a; }
      text[fill="#FFB74D"] { fill: #ffa726; }
      text[fill="#4A90A4"] { fill: #4fc3f7; }
    }
  </style>
'''


def update_svg(filepath: Path) -> bool:
    """Update a single SVG file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    # Skip if already has dark mode support
    if 'prefers-color-scheme: dark' in content:
        print(f"  Skipping {filepath.name} (already has dark mode)")
        return False

    # 1. Replace colors (case-insensitive for hex)
    for old_color, new_color in COLOR_MAP.items():
        # Handle both fill="color" and stroke="color" and style attributes
        content = re.sub(
            rf'(fill|stroke)="{old_color}"',
            rf'\1="{new_color}"',
            content,
            flags=re.IGNORECASE
        )
        # Also handle style= attributes
        content = re.sub(
            rf'stop-color:{old_color}',
            f'stop-color:{new_color}',
            content,
            flags=re.IGNORECASE
        )

    # 2. Change font family
    content = re.sub(
        r'font-family="Arial,\s*sans-serif"',
        'font-family="system-ui, -apple-system, sans-serif"',
        content
    )

    # 3. Remove background rect if it's the first/only full-size rect with fill
    # Match pattern like: <rect width="500" height="300" fill="#fafafa"/> or fill="none"
    content = re.sub(
        r'<!--\s*Background\s*-->\s*\n\s*<rect[^>]*fill="none"[^>]*/>\s*\n',
        '',
        content
    )
    # Also remove the comment if rect was removed
    content = re.sub(
        r'<!--\s*Background\s*-->\s*\n\s*(?=\n|<(?!rect))',
        '',
        content
    )

    # 4. Add dark mode style block after opening <svg> tag
    # Find the closing > of the <svg> tag
    svg_match = re.search(r'(<svg[^>]*>)', content)
    if svg_match:
        svg_end = svg_match.end()
        # Check if there's already a <style> or <defs> right after
        after_svg = content[svg_end:svg_end+100]
        if '<style>' not in after_svg:
            # Insert style block after <svg> tag
            content = content[:svg_end] + '\n' + DARK_MODE_STYLE + content[svg_end:]

    if content != original:
        filepath.write_text(content)
        print(f"  Updated {filepath.name}")
        return True
    else:
        print(f"  No changes needed for {filepath.name}")
        return False


def main():
    svg_dir = Path(__file__).parent.parent / 'books' / 'optimization-theory' / 'images'

    if not svg_dir.exists():
        print(f"Directory not found: {svg_dir}")
        return

    svg_files = sorted(svg_dir.glob('*.svg'))
    print(f"Found {len(svg_files)} SVG files in {svg_dir}")

    updated = 0
    for svg_file in svg_files:
        if update_svg(svg_file):
            updated += 1

    print(f"\nUpdated {updated} of {len(svg_files)} files")


if __name__ == '__main__':
    main()
