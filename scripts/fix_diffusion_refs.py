#!/usr/bin/env python3
"""
Fix references to diffusion chapters that still point to old numbers.
"""

import re
from pathlib import Path

def main():
    chapters_dir = Path("/home/jmalicki/src/ml-study-guide/chapters")

    # Mappings for diffusion chapter references
    replacements = [
        ('31-diffusion-fundamentals.md', '35-diffusion-fundamentals.md'),
        ('32-diffusion-implementation.md', '36-diffusion-implementation.md'),
        ('33-diffusion-advanced.md', '37-diffusion-advanced.md'),
    ]

    files_to_check = list(chapters_dir.glob("*.md"))

    for filepath in files_to_check:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        for old_ref, new_ref in replacements:
            if old_ref in content:
                content = content.replace(old_ref, new_ref)
                print(f"  {filepath.name}: Replaced {old_ref} -> {new_ref}")

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')

    print("\nFixed all diffusion chapter references!")

if __name__ == "__main__":
    main()
