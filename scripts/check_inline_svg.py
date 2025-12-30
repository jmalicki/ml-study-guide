#!/usr/bin/env python3
"""Check for inline SVG in markdown files (not supported on GitHub)."""
import re
import sys
from pathlib import Path

def main():
    chapters_dir = Path(__file__).parent.parent / 'chapters'
    files_with_svg = []

    for md_file in chapters_dir.glob('*.md'):
        content = md_file.read_text()
        if re.search(r'<svg[\s>]', content, re.IGNORECASE):
            files_with_svg.append(md_file.name)

    if files_with_svg:
        print("ERROR: Found inline SVG (not supported on GitHub):")
        for f in files_with_svg:
            print(f"  - chapters/{f}")
        print("\nConvert to external files in assets/diagrams/ and use:")
        print('  ![Description](../assets/diagrams/filename.svg)')
        sys.exit(1)

    print("No inline SVG found - GitHub compatible!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
