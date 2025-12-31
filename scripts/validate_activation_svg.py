#!/usr/bin/env python3
"""Validate SVG activation function curves against PyTorch implementations."""

import re
import sys
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch is required for this validation script.")
    print("Install it with: pip install torch")
    sys.exit(1)


def parse_polyline_points(svg_content, class_name):
    """Extract points from a polyline with given class."""
    pattern = rf'<polyline class="{class_name}" points="([^"]+)"'
    match = re.search(pattern, svg_content)
    if not match:
        return None
    points_str = match.group(1)
    points = []
    for point in points_str.split():
        x, y = point.split(',')
        points.append((float(x), float(y)))
    return points


def svg_to_data_coords(px, py):
    """Convert SVG pixel coordinates to data coordinates.

    The SVG uses the following coordinate system (based on grid lines):
    - X-axis: SVG x=80 -> data x=-6, SVG x=680 -> data x=4
             Scale: 600 pixels / 10 units = 60 px/unit
    - Y-axis: SVG y=352 -> data y=0, SVG y=80 -> data y=4
             Scale: 272 pixels / 4 units = 68 px/unit

    Args:
        px: SVG x coordinate (pixels)
        py: SVG y coordinate (pixels)

    Returns:
        Tuple of (data_x, data_y)
    """
    # x: linear mapping from SVG pixels to data
    # x=80 -> -6, x=680 -> 4, scale = 60 px/unit
    data_x = (px - 80) / 60 - 6

    # y: inverted mapping (SVG y increases downward)
    # y=352 -> 0, y=80 -> 4, scale = 68 px/unit
    data_y = (352 - py) / 68

    return data_x, data_y


def get_pytorch_value(func_name, x):
    """Get the correct activation value using PyTorch.

    Args:
        func_name: Name of activation function ('relu', 'gelu', 'silu', 'geglu', 'sigmoid', 'tanh')
        x: Input value

    Returns:
        Activation function output for input x
    """
    x_tensor = torch.tensor([x], dtype=torch.float32)
    if func_name == 'relu':
        return F.relu(x_tensor).item()
    elif func_name == 'gelu':
        return F.gelu(x_tensor).item()
    elif func_name == 'silu':
        return F.silu(x_tensor).item()
    elif func_name == 'geglu':
        # GEGLU: x * GELU(x)
        return (x_tensor * F.gelu(x_tensor)).item()
    elif func_name == 'sigmoid':
        return torch.sigmoid(x_tensor).item()
    elif func_name == 'tanh':
        return torch.tanh(x_tensor).item()
    else:
        raise ValueError(f"Unknown function: {func_name}")


def validate_curve(svg_content, class_name, func_name, epsilon=0.05):
    """Validate a single activation curve against PyTorch implementation.

    Args:
        svg_content: Full SVG file content as string
        class_name: CSS class name of the polyline to validate
        func_name: PyTorch function name to compare against
        epsilon: Maximum allowed absolute difference

    Returns:
        List of error tuples: (data_x, svg_y, expected_y, diff)
        Empty list if all points are within epsilon
        Single-element list with error message if polyline not found
    """
    points = parse_polyline_points(svg_content, class_name)
    if points is None:
        return [(f"Could not find polyline with class '{class_name}'", None, None, None)]

    errors = []
    for px, py in points:
        data_x, svg_y = svg_to_data_coords(px, py)
        expected_y = get_pytorch_value(func_name, data_x)
        diff = abs(svg_y - expected_y)
        if diff > epsilon:
            errors.append((data_x, svg_y, expected_y, diff))
    return errors


def main():
    """Main validation function."""
    # Path to the SVG file
    svg_path = Path(__file__).parent.parent / 'assets/diagrams/ch10-activation-functions-comparison.svg'

    if not svg_path.exists():
        print(f"ERROR: SVG file not found: {svg_path}")
        sys.exit(1)

    # Read SVG content
    with open(svg_path, 'r') as f:
        svg_content = f.read()

    print("Validating activation function SVG...")
    print()

    # Define functions to validate: (class_name, func_name, display_name)
    functions = [
        ('relu', 'relu', 'ReLU'),
        ('gelu', 'gelu', 'GELU'),
        ('silu', 'silu', 'SiLU'),
        ('geglu', 'geglu', 'GEGLU'),
        ('sigmoid', 'sigmoid', 'Sigmoid'),
        ('tanh', 'tanh', 'Tanh'),
    ]

    epsilon = 0.05
    all_passed = True

    for class_name, func_name, display_name in functions:
        errors = validate_curve(svg_content, class_name, func_name, epsilon)

        # Check if we got an error message (first element is string)
        if errors and isinstance(errors[0][0], str):
            print(f"{display_name}: ERROR - {errors[0][0]}")
            all_passed = False
            continue

        # Get total number of points
        points = parse_polyline_points(svg_content, class_name)
        num_points = len(points) if points else 0

        if errors:
            print(f"{display_name}: ERROR - {len(errors)}/{num_points} points exceed epsilon={epsilon}:")
            # Show first 5 errors
            for i, (x, svg_y, expected_y, diff) in enumerate(errors[:5]):
                print(f"  x={x:.2f}: SVG={svg_y:.3f}, Expected={expected_y:.3f}, diff={diff:.3f}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            all_passed = False
        else:
            print(f"{display_name}: OK (all {num_points} points within epsilon={epsilon})")

    print()
    if all_passed:
        print("SUCCESS: All activation functions validated correctly!")
        sys.exit(0)
    else:
        print("FAILED: One or more functions have errors")
        sys.exit(1)


if __name__ == '__main__':
    main()
