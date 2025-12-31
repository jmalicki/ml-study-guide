#!/usr/bin/env python3
"""
Reorganize chapters to move diffusion models to the end.

Moves chapters 24-26 (diffusion) to 35-37
Renumbers chapters 27-34 to 24-31
"""

import shutil
from pathlib import Path

def main():
    chapters_dir = Path("/home/jmalicki/src/ml-study-guide/chapters")
    assets_dir = Path("/home/jmalicki/src/ml-study-guide/assets/diagrams")

    # Define the renaming plan
    # Format: (old_number, new_number, slug)
    renames = [
        # First move diffusion chapters out of the way
        (24, 35, "diffusion-fundamentals"),
        (25, 36, "diffusion-implementation"),
        (26, 37, "diffusion-advanced"),
        # Then renumber the chapters that move up
        (27, 24, "long-context"),
        (28, 25, "multimodality"),
        (29, 26, "in-context-learning"),
        (30, 27, "reasoning"),
        (31, 28, "model-architectures"),
        (32, 29, "merging-distillation"),
        (33, 30, "hardware-quantization-optimization"),
        # Note: 34-distributed-training and 35-evaluation-benchmarks have file naming issues
        # We'll handle these specially
    ]

    # Check current state
    print("Current chapter files:")
    for f in sorted(chapters_dir.glob("[0-9][0-9]-*.md")):
        print(f"  {f.name}")

    print("\nExecuting renames:")

    # Do renames in order to avoid conflicts
    # Use temp names first for safety
    temp_renames = []
    for old_num, new_num, slug in renames:
        old_file = chapters_dir / f"{old_num:02d}-{slug}.md"
        if old_file.exists():
            temp_file = chapters_dir / f"TEMP{new_num:02d}-{slug}.md"
            print(f"  {old_file.name} -> {temp_file.name}")
            shutil.move(str(old_file), str(temp_file))
            temp_renames.append((temp_file, new_num, slug))
        else:
            print(f"  WARNING: {old_file.name} not found, skipping")

    # Now rename from TEMP to final names
    print("\nFinalizing renames:")
    for temp_file, new_num, slug in temp_renames:
        final_file = chapters_dir / f"{new_num:02d}-{slug}.md"
        print(f"  {temp_file.name} -> {final_file.name}")
        shutil.move(str(temp_file), str(final_file))

    print("\nFinal chapter files:")
    for f in sorted(chapters_dir.glob("[0-9][0-9]-*.md")):
        print(f"  {f.name}")

    # Now handle diagram files
    print("\n\nRenaming diagram files:")
    diagram_renames = [
        (24, 35), (25, 36), (26, 37),  # diffusion diagrams
        (27, 24), (28, 25), (29, 26), (30, 27), (31, 28), (32, 29), (33, 30)
    ]

    for old_num, new_num in diagram_renames:
        old_pattern = f"ch{old_num:02d}-"
        new_pattern = f"ch{new_num:02d}-"
        temp_pattern = f"TEMPCH{new_num:02d}-"

        # Find all matching diagram files
        matching_files = list(assets_dir.glob(f"ch{old_num:02d}-*.svg"))
        if matching_files:
            print(f"\n  Moving ch{old_num:02d}-* to ch{new_num:02d}-*:")
            for old_file in matching_files:
                base_name = old_file.name.replace(old_pattern, "")
                temp_file = assets_dir / f"{temp_pattern}{base_name}"
                print(f"    {old_file.name} -> {temp_file.name}")
                shutil.move(str(old_file), str(temp_file))

    # Finalize diagram renames
    print("\n  Finalizing diagram renames:")
    for temp_file in sorted(assets_dir.glob("TEMPCH*")):
        final_name = temp_file.name.replace("TEMPCH", "ch")
        final_file = assets_dir / final_name
        print(f"    {temp_file.name} -> {final_file.name}")
        shutil.move(str(temp_file), str(final_file))

    print("\nReorganization complete!")

if __name__ == "__main__":
    main()
