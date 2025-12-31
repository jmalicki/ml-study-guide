# Example Validation Output

## Successful Validation Run

When all activation functions match the PyTorch implementations:

```
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

Exit code: 0

## Validation with Errors

Example output when some points don't match:

```
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
SiLU: OK (all 101 points within epsilon=0.05)
Sigmoid: OK (all 101 points within epsilon=0.05)
Tanh: OK (all 101 points within epsilon=0.05)

FAILED: One or more functions have errors
make: *** [Makefile:46: validate-activation-svg] Error 1
```

Exit code: 1

## Missing PyTorch

When PyTorch is not installed:

```
$ python3 scripts/validate_activation_svg.py
ERROR: PyTorch is required for this validation script.
Install it with: pip install torch
```

Exit code: 1

## Missing SVG File

When the SVG file is not found:

```
$ python3 scripts/validate_activation_svg.py
ERROR: SVG file not found: /home/jmalicki/src/ml-study-guide/assets/diagrams/ch10-activation-functions-comparison.svg
```

Exit code: 1

## Unit Tests

Running the test suite:

```
$ python3 scripts/test_validate_activation_svg.py
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

Exit code: 0

## Validation Details

### What's Being Checked

For each activation function, the script:

1. Extracts the polyline points from the SVG
2. For each point (px, py) in SVG coordinates:
   - Converts to data coordinates (x, y)
   - Computes the expected value using PyTorch: `expected = F.activation(x)`
   - Compares: `|y - expected| <= epsilon`
   - Reports if difference exceeds epsilon

### Example Point Validation

**ReLU at x=2:**
- SVG point: (560, 216)
- Converts to data: (2.0, 2.0)
- PyTorch ReLU(2.0) = 2.0
- Difference: |2.0 - 2.0| = 0.0
- Result: ✓ PASS (within epsilon=0.05)

**GELU at x=-0.85 (minimum point):**
- SVG point: approximately (363, 371)
- Converts to data: (-0.85, -0.17)
- PyTorch GELU(-0.85) ≈ -0.169
- Difference: |-0.17 - (-0.169)| ≈ 0.001
- Result: ✓ PASS (within epsilon=0.05)

### Coordinate Transformation Example

Given SVG point (440, 352):

```python
# X coordinate (440 px)
data_x = -6 + (440 - 80) / (680 - 80) * (4 - (-6))
       = -6 + (360 / 600) * 10
       = -6 + 6
       = 0.0

# Y coordinate (352 px) - note Y is inverted
data_y = 4.5 - (352 - 80) / (420 - 80) * (4.5 - (-0.5))
       = 4.5 - (272 / 340) * 5
       = 4.5 - 4.0
       = 0.5
```

Wait, let me recalculate:
```python
data_y = 4.5 - (352 - 80) / (420 - 80) * (4.5 - (-0.5))
       = 4.5 - (272 / 340) * 5.0
       = 4.5 - 4.0
       = 0.5
```

Actually, for the zero line (y=0 in data), the SVG y should be 352:
```python
0 = 4.5 - (y_px - 80) / (420 - 80) * 5.0
4.5 = (y_px - 80) / 340 * 5.0
y_px = 80 + (4.5 / 5.0) * 340
     = 80 + 0.9 * 340
     = 80 + 306
     = 386
```

Hmm, wait. Let me verify: the y=0 line in the SVG is at y=352 according to the comment.

Actually checking the SVG:
```xml
<!-- Horizontal line at y=0 (dashed) -->
<line x1="80" y1="352" x2="680" y2="352" .../>
```

So y=0 is at SVG y=352. Let's verify:
```python
data_y = 4.5 - (352 - 80) / (420 - 80) * (4.5 - (-0.5))
       = 4.5 - (272 / 340) * 5.0
       = 4.5 - 4.0
       = 0.5
```

That's wrong. Let me recalculate the range. Looking at the SVG:
- y=4 is at SVG y=80 (top)
- y=-0.5 is at SVG y=420 (bottom)

So the range is 4.5 (from -0.5 to 4).

For y=0:
```python
0 = y_max - (y_px - plot_y_min) / (plot_y_max - plot_y_min) * (y_max - y_min)
0 = 4.5 - (y_px - 80) / (420 - 80) * (4.5 - (-0.5))
```

Wait, I see the issue. The comment says:
```xml
<!-- Data range: x from -6 to 4, y from -0.5 to 4.5 -->
```

But the y-axis labels show:
- y=4 at SVG y=80
- y=3 at SVG y=148
- y=2 at SVG y=216
- y=1 at SVG y=284
- y=0 at SVG y=352
- y=-0.5 at SVG y=420

So y_max should be 4, not 4.5! Let me correct the example.
