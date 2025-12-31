# Activation Function SVG Validation

## Summary

Created a validation script that checks SVG activation function graphs against actual PyTorch implementations. This ensures the visual diagrams accurately represent the mathematical functions.

## Files Created

### 1. `/home/jmalicki/src/ml-study-guide/scripts/validate_activation_svg.py`
Main validation script that:
- Parses the SVG file to extract polyline points for each activation function
- Converts SVG pixel coordinates to data coordinates
- Compares each point against actual PyTorch implementations (F.relu, F.gelu, F.silu, torch.sigmoid, torch.tanh)
- Reports errors if any point deviates by more than epsilon (default: 0.05)

**Key functions:**
- `parse_polyline_points()` - Extracts points from SVG polylines
- `svg_to_data_coords()` - Converts SVG coordinates to data coordinates
- `get_pytorch_value()` - Computes activation function values using PyTorch
- `validate_curve()` - Validates a single activation curve
- `main()` - Orchestrates validation and reports results

### 2. `/home/jmalicki/src/ml-study-guide/scripts/test_validate_activation_svg.py`
Unit tests for the validation script that verify:
- Coordinate transformation accuracy
- PyTorch function implementations
- Polyline parsing logic

### 3. `/home/jmalicki/src/ml-study-guide/scripts/README_validation.md`
Documentation explaining how the validation works, including:
- Requirements and usage
- Coordinate transformation details
- Output format examples
- How to adjust tolerance

### 4. Updated `/home/jmalicki/src/ml-study-guide/Makefile`
Added new target `validate-activation-svg` that runs the validation script:
```bash
make validate-activation-svg
```

## Coordinate System

The script handles the transformation from SVG coordinates to data coordinates:

**SVG Plot Area:**
- X: [80, 680] pixels
- Y: [80, 420] pixels (top to bottom)

**Data Range:**
- X: [-6, 4]
- Y: [-0.5, 4] (bottom to top)

**Transformation:**
```python
data_x = -6 + (px - 80) / (680 - 80) * (4 - (-6))
data_y = 4 - (py - 80) / (420 - 80) * (4 - (-0.5))
```

## Validated Functions

1. **ReLU** - Rectified Linear Unit: `max(0, x)`
2. **GELU** - Gaussian Error Linear Unit (PyTorch implementation)
3. **SiLU** (Swish) - Sigmoid Linear Unit: `x * sigmoid(x)`
4. **Sigmoid** - `1 / (1 + exp(-x))`
5. **Tanh** - Hyperbolic tangent

## Usage Examples

### Successful validation:
```bash
$ make validate-activation-svg
Validating activation function SVG against PyTorch implementations...
Validating activation function SVG...

ReLU: OK (all 11 points within epsilon=0.05)
GELU: OK (all 101 points within epsilon=0.05)
SiLU: OK (all 101 points within epsilon=0.05)
Sigmoid: OK (all 101 points within epsilon=0.05)
Tanh: OK (all 101 points within epsilon=0.05)

SUCCESS: All activation functions validated correctly!
```

### When errors are found:
```bash
$ make validate-activation-svg
Validating activation function SVG against PyTorch implementations...
Validating activation function SVG...

ReLU: OK (all 11 points within epsilon=0.05)
GELU: ERROR - 5/101 points exceed epsilon=0.05:
  x=-1.50: SVG=0.120, Expected=-0.110, diff=0.230
  x=-1.40: SVG=0.100, Expected=-0.095, diff=0.195
  x=-1.30: SVG=0.085, Expected=-0.082, diff=0.167
  x=-1.20: SVG=0.072, Expected=-0.070, diff=0.142
  x=-1.10: SVG=0.061, Expected=-0.060, diff=0.121

FAILED: One or more functions have errors
make: *** [Makefile:46: validate-activation-svg] Error 1
```

## Requirements

The script requires PyTorch to be installed:

```bash
pip install torch
```

If PyTorch is not installed, the script will display a helpful error message:
```
ERROR: PyTorch is required for this validation script.
Install it with: pip install torch
```

## Integration with CI/CD

The validation can be added to the main `check` target in the Makefile if desired:

```makefile
check: lint validate check-svg validate-svg validate-activation-svg check-svg-contrast check-latex
```

However, it's kept as a separate target by default since:
1. It requires PyTorch as a dependency
2. It's specific to one particular SVG file
3. It's more of a content validation than a structural check

## Testing

Run the unit tests:
```bash
python3 scripts/test_validate_activation_svg.py
```

Expected output:
```
Testing SVG activation validation script...

✓ Center point (0,0) transforms correctly
✓ Left extreme x=-6 transforms correctly
✓ Right extreme x=4 transforms correctly
✓ Top extreme y=4.5 transforms correctly
✓ Bottom extreme y=-0.5 transforms correctly

✓ ReLU function works correctly
✓ Sigmoid function works correctly
✓ Tanh function works correctly

✓ Polyline parsing works correctly
✓ Missing class returns None correctly

SUCCESS: All tests passed!
```

## Error Tolerance

The default epsilon (error tolerance) is set to `0.05`. This means:
- For data values, points can be off by up to 0.05 units
- This is reasonable given SVG rendering precision and pixel-to-data conversion

You can adjust this in the script if needed:
```python
epsilon = 0.05  # in main() function
```

## Future Enhancements

Possible improvements:
1. Support for additional activation functions (LeakyReLU, ELU, Softplus, etc.)
2. Configurable epsilon via command-line argument
3. JSON output format for automated parsing
4. Validation of other SVG diagrams in the study guide
5. Visual diff output showing where errors occur
