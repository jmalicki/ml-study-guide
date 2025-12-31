# SVG Activation Function Validation

## Overview

The `validate_activation_svg.py` script validates that the activation function curves in the SVG diagram match the actual PyTorch implementations.

## Requirements

- Python 3.7+
- PyTorch (`pip install torch`)

## Usage

### Direct execution

```bash
python3 scripts/validate_activation_svg.py
```

### Via Makefile

```bash
make validate-activation-svg
```

## What it validates

The script validates the following activation functions in `/assets/diagrams/ch10-activation-functions-comparison.svg`:

1. **ReLU** - Rectified Linear Unit: `max(0, x)`
2. **GELU** - Gaussian Error Linear Unit
3. **SiLU** (Swish) - Sigmoid Linear Unit: `x * sigmoid(x)`
4. **Sigmoid** - `1 / (1 + exp(-x))`
5. **Tanh** - Hyperbolic tangent

## How it works

1. Parses the SVG file to extract polyline points for each activation function
2. Converts SVG pixel coordinates to data coordinates using the plot area mapping:
   - SVG plot area: x=[80, 680], y=[80, 420]
   - Data range: x=[-6, 4], y=[-0.5, 4]
3. Compares each point against the actual PyTorch implementation
4. Reports errors if any point deviates by more than epsilon (default: 0.05)

## Output format

Success:

```text
Validating activation function SVG...

ReLU: OK (all 11 points within epsilon=0.05)
GELU: OK (all 101 points within epsilon=0.05)
SiLU: OK (all 101 points within epsilon=0.05)
Sigmoid: OK (all 101 points within epsilon=0.05)
Tanh: OK (all 101 points within epsilon=0.05)

SUCCESS: All activation functions validated correctly!
```

Failure:

```text
Validating activation function SVG...

ReLU: OK (all 11 points within epsilon=0.05)
GELU: ERROR - 5/101 points exceed epsilon=0.05:
  x=-1.50: SVG=0.120, Expected=-0.110, diff=0.230
  x=-1.40: SVG=0.100, Expected=-0.095, diff=0.195
  ...

FAILED: One or more functions have errors
```

## Adjusting tolerance

To change the error threshold, modify the `epsilon` parameter in the `main()` function:

```python
epsilon = 0.05  # Increase for looser validation, decrease for stricter
```

## Coordinate transformation

The script handles the SVG coordinate system where:

- X-axis increases left to right (same as data)
- Y-axis increases top to bottom (inverted from data)

The transformation formulas are:

```python
data_x = x_min + (px - plot_x_min) / (plot_x_max - plot_x_min) * (x_max - x_min)
data_y = y_max - (py - plot_y_min) / (plot_y_max - plot_y_min) * (y_max - y_min)
```
