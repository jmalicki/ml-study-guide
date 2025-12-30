#!/usr/bin/env python3
"""
Validate internal links in the ML Study Guide.

This script:
1. Parses all markdown files in the repository
2. Extracts internal links (links to other .md files)
3. Verifies that all linked files exist
4. Checks that chapter numbers are sequential
5. Validates that README.md references match actual files
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


class LinkValidator:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.chapters_dir = root_dir / "chapters"
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def extract_internal_links(self, content: str, source_file: Path) -> List[Tuple[str, str]]:
        """Extract all internal markdown links from content."""
        # Pattern matches [text](path.md) or [text](path.md#anchor)
        pattern = r'\[([^\]]+)\]\(([^)]+\.md(?:#[^)]*)?)\)'
        links = []

        for match in re.finditer(pattern, content):
            link_text = match.group(1)
            link_path = match.group(2)
            links.append((link_text, link_path))

        return links

    def resolve_link_path(self, link: str, source_file: Path) -> Path:
        """Resolve a link path relative to the source file."""
        # Remove anchor if present
        link_without_anchor = link.split('#')[0]

        # Handle relative paths
        if link_without_anchor.startswith('chapters/'):
            return self.root_dir / link_without_anchor
        else:
            # Assume it's relative to the source file's directory
            return (source_file.parent / link_without_anchor).resolve()

    def validate_file_links(self, file_path: Path) -> None:
        """Validate all internal links in a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return

        links = self.extract_internal_links(content, file_path)

        for link_text, link_path in links:
            # Skip external URLs (http, https, ftp, etc.)
            if link_path.startswith(('http://', 'https://', 'ftp://', 'mailto:')):
                continue

            target_path = self.resolve_link_path(link_path, file_path)

            if not target_path.exists():
                self.errors.append(
                    f"Broken link in {file_path.relative_to(self.root_dir)}: "
                    f"'{link_text}' -> '{link_path}' (resolved to {target_path.relative_to(self.root_dir)})"
                )

    def validate_chapter_sequence(self) -> None:
        """Check that chapter numbers are sequential."""
        if not self.chapters_dir.exists():
            self.errors.append(f"Chapters directory not found: {self.chapters_dir}")
            return

        chapter_files = sorted(self.chapters_dir.glob("[0-9][0-9]-*.md"))
        expected_numbers = set(range(1, len(chapter_files) + 1))
        actual_numbers = set()

        for chapter_file in chapter_files:
            match = re.match(r'(\d+)-', chapter_file.name)
            if match:
                chapter_num = int(match.group(1))
                actual_numbers.add(chapter_num)

        missing_numbers = expected_numbers - actual_numbers
        if missing_numbers:
            self.warnings.append(
                f"Chapter sequence has gaps: missing chapters {sorted(missing_numbers)}"
            )

        extra_numbers = actual_numbers - expected_numbers
        if extra_numbers:
            self.warnings.append(
                f"Chapter sequence has unexpected numbers: {sorted(extra_numbers)}"
            )

    def validate_readme_references(self) -> None:
        """Validate that all chapter references in README.md exist."""
        readme_path = self.root_dir / "README.md"
        if not readme_path.exists():
            self.errors.append("README.md not found")
            return

        self.validate_file_links(readme_path)

    def validate_all_chapters(self) -> None:
        """Validate links in all chapter files."""
        if not self.chapters_dir.exists():
            return

        for chapter_file in self.chapters_dir.glob("*.md"):
            self.validate_file_links(chapter_file)

    def run(self) -> int:
        """Run all validations and return exit code."""
        print("=" * 70)
        print("ML Study Guide - Link Validation")
        print("=" * 70)
        print()

        print("Validating README.md references...")
        self.validate_readme_references()

        print("Validating chapter sequence...")
        self.validate_chapter_sequence()

        print("Validating links in chapter files...")
        self.validate_all_chapters()

        print()
        print("=" * 70)
        print("Validation Results")
        print("=" * 70)
        print()

        if self.warnings:
            print(f"WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        if self.errors:
            print(f"ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
            print()
            print("Validation FAILED")
            return 1
        else:
            print("All validations PASSED")
            if self.warnings:
                print(f"(with {len(self.warnings)} warning(s))")
            return 0


def main():
    """Main entry point."""
    # Find the root directory (where the script is run from or one level up from scripts/)
    current_dir = Path.cwd()
    if current_dir.name == "scripts":
        root_dir = current_dir.parent
    else:
        root_dir = current_dir

    validator = LinkValidator(root_dir)
    exit_code = validator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
