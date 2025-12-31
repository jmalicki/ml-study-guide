#!/usr/bin/env python3
"""Test the activation function SVG validation script."""

import sys
from pathlib import Path

# Add parent directory to path to import the validation script
sys.path.insert(0, str(Path(__file__).parent))

try:
    import validate_activation_svg as val
except ImportError as e:
    print(f"ERROR: Could not import validation script: {e}")
    sys.exit(1)


def test_coordinate_transformation():
    """Test SVG to data coordinate transformation."""
    # Test center point: x=440 should map to data x=0
    data_x, data_y = val.svg_to_data_coords(440, 352)
    assert abs(data_x - 0.0) < 0.01, f"Expected x=0, got {data_x}"
    assert abs(data_y - 0.0) < 0.01, f"Expected y=0, got {data_y}"
    print("✓ Center point (0,0) transforms correctly")

    # Test x-axis extremes
    data_x, _ = val.svg_to_data_coords(80, 352)
    assert abs(data_x - (-6.0)) < 0.01, f"Expected x=-6, got {data_x}"
    print("✓ Left extreme x=-6 transforms correctly")

    data_x, _ = val.svg_to_data_coords(680, 352)
    assert abs(data_x - 4.0) < 0.01, f"Expected x=4, got {data_x}"
    print("✓ Right extreme x=4 transforms correctly")

    # Test y-axis extremes
    _, data_y = val.svg_to_data_coords(440, 80)
    assert abs(data_y - 4.5) < 0.01, f"Expected y=4.5, got {data_y}"
    print("✓ Top extreme y=4.5 transforms correctly")

    _, data_y = val.svg_to_data_coords(440, 420)
    assert abs(data_y - (-0.5)) < 0.01, f"Expected y=-0.5, got {data_y}"
    print("✓ Bottom extreme y=-0.5 transforms correctly")


def test_pytorch_functions():
    """Test PyTorch function implementations."""
    # Test ReLU
    assert val.get_pytorch_value('relu', -1.0) == 0.0
    assert val.get_pytorch_value('relu', 0.0) == 0.0
    assert val.get_pytorch_value('relu', 1.0) == 1.0
    print("✓ ReLU function works correctly")

    # Test Sigmoid
    sig_0 = val.get_pytorch_value('sigmoid', 0.0)
    assert abs(sig_0 - 0.5) < 0.01
    print("✓ Sigmoid function works correctly")

    # Test Tanh
    tanh_0 = val.get_pytorch_value('tanh', 0.0)
    assert abs(tanh_0 - 0.0) < 0.01
    print("✓ Tanh function works correctly")


def test_polyline_parsing():
    """Test polyline parsing from SVG content."""
    test_svg = '<polyline class="test" points="10,20 30,40 50,60"/>'
    points = val.parse_polyline_points(test_svg, "test")
    assert points is not None
    assert len(points) == 3
    assert points[0] == (10.0, 20.0)
    assert points[1] == (30.0, 40.0)
    assert points[2] == (50.0, 60.0)
    print("✓ Polyline parsing works correctly")

    # Test missing class
    points = val.parse_polyline_points(test_svg, "missing")
    assert points is None
    print("✓ Missing class returns None correctly")


def main():
    """Run all tests."""
    print("Testing SVG activation validation script...")
    print()

    try:
        test_coordinate_transformation()
        print()
        test_pytorch_functions()
        print()
        test_polyline_parsing()
        print()
        print("SUCCESS: All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
