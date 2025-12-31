#!/usr/bin/env python3
"""
Complete reorganization to move diffusion chapters to the end.

Current state (based on filenames):
- 23: long-context (should be 24)
- 24: multimodality (should be 25)
- 25: in-context-learning (should be 26)
- 26: reasoning (should be 27)
- 27: model-architectures (should be 28)
- 28: merging-distillation (should be 29)
- 29: hardware-quantization-optimization (should be 30)
- 30: distributed-training (should be 31)
- 31: diffusion-fundamentals (should be 35)
- 32: diffusion-implementation (should be 36)
- 33: diffusion-advanced (should be 37)
- 34: evaluation-benchmarks (should be 32)

But we need to check the content inside files too since chapter titles have old numbers.
"""

import re
import shutil
from pathlib import Path

def main():
    chapters_dir = Path("/home/jmalicki/src/ml-study-guide/chapters")

    # Define mapping from current filename number to desired number
    # Based on moving diffusion to end
    renames = [
        # Move diffusion out of the way first
        (31, 35, "diffusion-fundamentals"),
        (32, 36, "diffusion-implementation"),
        (33, 37, "diffusion-advanced"),
        # Then shift everything else
        (23, 24, "long-context"),
        (24, 25, "multimodality"),
        (25, 26, "in-context-learning"),
        (26, 27, "reasoning"),
        (27, 28, "model-architectures"),
        (28, 29, "merging-distillation"),
        (29, 30, "hardware-quantization-optimization"),
        (30, 31, "distributed-training"),
        (34, 32, "evaluation-benchmarks"),
    ]

    print("Step 1: Renaming chapter files")
    print("=" * 70)

    # First pass: rename to temp names
    temp_files = []
    for old_num, new_num, slug in renames:
        old_file = chapters_dir / f"{old_num:02d}-{slug}.md"
        if old_file.exists():
            temp_file = chapters_dir / f"TEMP{new_num:03d}-{slug}.md"
            print(f"  {old_file.name} -> {temp_file.name}")
            shutil.move(str(old_file), str(temp_file))
            temp_files.append((temp_file, new_num, slug))
        else:
            print(f"  WARNING: {old_file.name} not found")

    # Second pass: rename from temp to final
    print("\nFinalizing file renames:")
    renamed_files = []
    for temp_file, new_num, slug in temp_files:
        final_file = chapters_dir / f"{new_num:02d}-{slug}.md"
        print(f"  {temp_file.name} -> {final_file.name}")
        shutil.move(str(temp_file), str(final_file))
        renamed_files.append((final_file, new_num, slug))

    print("\n\nStep 2: Updating chapter numbers inside files")
    print("=" * 70)

    # Now update the chapter numbers inside the files
    # We need to update:
    # 1. The main title (# Chapter XX: Title)
    # 2. Any cross-references to other chapters

    # Create mapping of old chapter numbers (in content) to new numbers
    content_number_map = {
        24: 35,  # diffusion-fundamentals
        25: 36,  # diffusion-implementation
        26: 37,  # diffusion-advanced
        27: 24,  # long-context
        28: 25,  # multimodality
        29: 26,  # in-context-learning
        30: 27,  # reasoning
        31: 28,  # model-architectures
        32: 29,  # merging-distillation
        33: 30,  # hardware-quantization-optimization
        16: 31,  # distributed-training (content says 16, filename was 30)
        34: 32,  # evaluation-benchmarks
    }

    # Update all chapter files
    for chapter_file in chapters_dir.glob("[0-9][0-9]-*.md"):
        update_chapter_file(chapter_file, content_number_map)

    print("\n\nReorganization of files complete!")
    print("Next steps:")
    print("  1. Update TABLE_OF_CONTENTS.md")
    print("  2. Update diagram references in chapters")
    print("  3. Run make validate")

def update_chapter_file(filepath, number_map):
    """Update chapter numbers in a single file."""
    content = filepath.read_text(encoding='utf-8')
    original_content = content

    # Update the main chapter title
    def replace_title(match):
        old_num = int(match.group(1))
        if old_num in number_map:
            new_num = number_map[old_num]
            print(f"  {filepath.name}: Updating title Ch{old_num} -> Ch{new_num}")
            return f"# Chapter {new_num}:"
        return match.group(0)

    content = re.sub(r'^# Chapter (\d+):', replace_title, content, flags=re.MULTILINE)

    # Update references to chapter files like "24-diffusion-fundamentals.md" or "chapter 24"
    # We need to be careful to only update when appropriate

    # Update markdown links to chapter files [text](NN-slug.md)
    def replace_chapter_link(match):
        old_num = int(match.group(2))
        link_text = match.group(1)
        slug = match.group(3)
        if old_num in number_map:
            new_num = number_map[old_num]
            return f'[{link_text}]({new_num:02d}-{slug}.md)'
        return match.group(0)

    content = re.sub(r'\[([^\]]+)\]\((\d+)-([a-z-]+)\.md\)', replace_chapter_link, content)

    # Update diagram references like ../assets/diagrams/chNN-
    def replace_diagram_ref(match):
        old_num = int(match.group(1))
        rest = match.group(2)
        if old_num in number_map:
            new_num = number_map[old_num]
            return f'../assets/diagrams/ch{new_num:02d}-{rest}'
        return match.group(0)

    content = re.sub(r'\.\./assets/diagrams/ch(\d+)-([^\s\)]+)', replace_diagram_ref, content)

    if content != original_content:
        filepath.write_text(content, encoding='utf-8')
        print(f"  Updated: {filepath.name}")

if __name__ == "__main__":
    main()
