I want to make a study guide for ML interviews focusing on LLMs.

It should describe the algorithms, use LaTeX math notation where appropriate, and include sample python code using pytorch.

We should start with tokenization, then basic attention mechanisms.

We should have various position indications, including RoPE.

Normal, multi-head, etc. attention.

We should have a section on flash attention as well.

We should then move on to RLHF and DPO, etc.

We should also include diffusion models.

The goal will be for all of this to be runnable and to produce trainable runnable models, where we build up piece by piece, chapter by chapter.

For the first step, just make a markdown outline for review.

Each chapter should be its own file, and the outline should link to the (for now nonexistant) files.

## SVG Illustrations

Diagrams are stored as standalone SVG files in `assets/diagrams/` and referenced from markdown chapters via relative paths (e.g., `![Diagram](../assets/diagrams/ch03-attention-heatmap.svg)`).

### Why External SVGs (Not Inline)

GitHub's markdown renderer does not support inline SVG. All diagrams must be external files.

### Dark Mode Support

All SVGs must support both light and dark modes using CSS media queries:

```xml
<style>
  /* Light mode (default) */
  text { fill: #000000; }

  /* Dark mode */
  @media (prefers-color-scheme: dark) {
    text { fill: #ffffff; }
    /* Invert dark colors to light, keep colored backgrounds readable */
  }
</style>
```

Key patterns:
- Default text fill is black (`#000000`)
- Dark mode overrides text to white (`#ffffff`)
- Colored backgrounds may need darkening in dark mode for white text visibility
- Use CSS classes or attribute selectors to target specific elements

### Typography Requirements

- **Standard fonts**: Use system fonts (`system-ui, -apple-system, sans-serif`) for labels
- **Math formulas**: Use serif fonts, preferably STIX Two Math with CDN fallback
- **Subscripts**: Use proper SVG positioning or Unicode subscript characters, not ASCII `x_y` notation
- **Fractions**: Use proper stacked layout (numerator/bar/denominator), not slash notation
- **Vertical spacing**: Text elements need adequate spacing (font-size + 4px minimum)

### Accessibility

- Text must meet WCAG 2.1 contrast ratios (4.5:1 for normal text, 3:1 for large text)
- Content must not extend beyond the viewBox boundaries

### Validation

Run `make check` to validate all SVGs. Individual checks:

- `make check-svg` - Check for inline SVG in markdown (not allowed)
- `make validate-svg` - Validate SVG syntax with SVGO
- `make check-svg-contrast` - Check text contrast accessibility
- `make check-svg-typography` - Check fonts, spacing, formula formatting
- `make validate-activation-svg` - Validate activation function curves against PyTorch

The typography checker supports inline disable comments:
```xml
<!-- lint-disable subscript -->   - disable ASCII subscript check
<!-- lint-disable math-font -->   - disable math font check
<!-- lint-disable all -->         - disable all checks
```

### Naming Convention

Files are named `ch{NN}-{description}.svg` where NN is the chapter number (e.g., `ch03-attention-heatmap.svg`).
