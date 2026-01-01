#!/usr/bin/env python3
"""Check SVG files for overlapping text and box elements.

Detects cases where:
1. Text overlaps with rectangles it shouldn't be inside
2. Text elements overlap with each other
3. Rectangles overlap in unexpected ways
"""
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x: float
    y: float
    width: float
    height: float
    element_type: str
    content: str = ""
    line_number: int = 0

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def overlaps(self, other: 'BoundingBox', margin: float = 0) -> bool:
        """Check if this box overlaps with another."""
        return (self.x - margin < other.x2 and
                self.x2 + margin > other.x and
                self.y - margin < other.y2 and
                self.y2 + margin > other.y)

    def contains(self, other: 'BoundingBox', margin: float = 5) -> bool:
        """Check if this box fully contains another (with margin)."""
        return (self.x - margin <= other.x and
                self.x2 + margin >= other.x2 and
                self.y - margin <= other.y and
                self.y2 + margin >= other.y2)

    def overlap_area(self, other: 'BoundingBox') -> float:
        """Calculate the overlapping area."""
        dx = min(self.x2, other.x2) - max(self.x, other.x)
        dy = min(self.y2, other.y2) - max(self.y, other.y)
        if dx > 0 and dy > 0:
            return dx * dy
        return 0


@dataclass
class Transform:
    """Simple 2D transform (translation only for now)."""
    tx: float = 0
    ty: float = 0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return x + self.tx, y + self.ty

    def compose(self, other: 'Transform') -> 'Transform':
        """Compose with another transform."""
        return Transform(self.tx + other.tx, self.ty + other.ty)


def parse_transform(transform_str: Optional[str]) -> Transform:
    """Parse SVG transform attribute (supports translate only)."""
    if not transform_str:
        return Transform()

    match = re.search(r'translate\s*\(\s*([+-]?\d*\.?\d+)\s*[,\s]\s*([+-]?\d*\.?\d+)\s*\)', transform_str)
    if match:
        return Transform(float(match.group(1)), float(match.group(2)))

    match = re.search(r'translate\s*\(\s*([+-]?\d*\.?\d+)\s*\)', transform_str)
    if match:
        return Transform(float(match.group(1)), 0)

    return Transform()


def estimate_text_width(text: str, font_size: float) -> float:
    """Estimate text width based on character count and font size."""
    avg_char_width = font_size * 0.55
    return len(text) * avg_char_width


def get_text_anchor_offset(anchor: str, width: float) -> float:
    """Get x offset based on text-anchor attribute."""
    if anchor == "middle":
        return -width / 2
    elif anchor == "end":
        return -width
    return 0


def get_float_attr(elem: ET.Element, attr: str, default: float = 0) -> float:
    """Get float attribute value."""
    val = elem.get(attr)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_line_number(elem: ET.Element, svg_content: str) -> int:
    """Try to find the line number of an element."""
    if elem.text:
        search_text = elem.text[:30] if len(elem.text) > 30 else elem.text
        for i, line in enumerate(svg_content.split('\n'), 1):
            if search_text in line:
                return i

    if elem.tag.endswith('rect'):
        x = elem.get('x', '')
        y = elem.get('y', '')
        for i, line in enumerate(svg_content.split('\n'), 1):
            if f'x="{x}"' in line and f'y="{y}"' in line:
                return i

    return 0


def extract_bounding_boxes(svg_path: Path) -> tuple[list[BoundingBox], list[BoundingBox]]:
    """Extract bounding boxes for text and rect elements."""
    svg_content = svg_path.read_text()

    svg_content_no_ns = re.sub(r'\sxmlns="[^"]+"', '', svg_content)
    root = ET.fromstring(svg_content_no_ns)

    texts: list[BoundingBox] = []
    rects: list[BoundingBox] = []

    def process_element(elem: ET.Element, parent_transform: Transform):
        elem_transform = parse_transform(elem.get('transform'))
        current_transform = parent_transform.compose(elem_transform)

        tag = elem.tag.split('}')[-1]

        if tag == 'text':
            x = get_float_attr(elem, 'x')
            y = get_float_attr(elem, 'y')
            font_size = get_float_attr(elem, 'font-size', 12)
            anchor = elem.get('text-anchor', 'start')

            text_content = elem.text or ''
            for child in elem:
                if child.text:
                    text_content += child.text
                if child.tail:
                    text_content += child.tail

            if text_content.strip():
                width = estimate_text_width(text_content, font_size)
                height = font_size * 1.2

                x_offset = get_text_anchor_offset(anchor, width)
                abs_x, abs_y = current_transform.apply(x + x_offset, y - font_size)

                texts.append(BoundingBox(
                    x=abs_x,
                    y=abs_y,
                    width=width,
                    height=height,
                    element_type='text',
                    content=text_content.strip()[:50],
                    line_number=get_line_number(elem, svg_content)
                ))

        elif tag == 'rect':
            x = get_float_attr(elem, 'x')
            y = get_float_attr(elem, 'y')
            width = get_float_attr(elem, 'width')
            height = get_float_attr(elem, 'height')

            if width > 0 and height > 0:
                abs_x, abs_y = current_transform.apply(x, y)

                rects.append(BoundingBox(
                    x=abs_x,
                    y=abs_y,
                    width=width,
                    height=height,
                    element_type='rect',
                    line_number=get_line_number(elem, svg_content)
                ))

        for child in elem:
            process_element(child, current_transform)

    process_element(root, Transform())
    return texts, rects


def check_overlaps(svg_path: Path) -> list[str]:
    """Check for problematic overlaps in an SVG file."""
    errors = []
    texts, rects = extract_bounding_boxes(svg_path)

    for i, t1 in enumerate(texts):
        for t2 in texts[i+1:]:
            if t1.overlaps(t2):
                overlap = t1.overlap_area(t2)
                min_area = min(t1.width * t1.height, t2.width * t2.height)
                if overlap > min_area * 0.2:
                    errors.append(
                        f"  Line {t1.line_number}/{t2.line_number}: Text overlap - "
                        f"'{t1.content}' overlaps with '{t2.content}'"
                    )

    for text in texts:
        containing_rects = [r for r in rects if r.contains(text)]

        if not containing_rects:
            for rect in rects:
                if text.overlaps(rect) and not rect.contains(text, margin=0):
                    overlap = text.overlap_area(rect)
                    text_area = text.width * text.height
                    if 0.1 < overlap / text_area < 0.9:
                        errors.append(
                            f"  Line {text.line_number}: Text '{text.content}' "
                            f"partially overlaps rect at line {rect.line_number}"
                        )

    for i, r1 in enumerate(rects):
        for r2 in rects[i+1:]:
            if r1.overlaps(r2):
                if r1.contains(r2, margin=0) or r2.contains(r1, margin=0):
                    continue

                overlap = r1.overlap_area(r2)
                min_area = min(r1.width * r1.height, r2.width * r2.height)
                if overlap > min_area * 0.1:
                    errors.append(
                        f"  Line {r1.line_number}/{r2.line_number}: "
                        f"Rectangles overlap unexpectedly"
                    )

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: check_svg_overlaps.py <svg_files...>")
        return 1

    total_errors = 0

    for arg in sys.argv[1:]:
        svg_path = Path(arg)
        if not svg_path.exists():
            print(f"File not found: {svg_path}")
            continue

        errors = check_overlaps(svg_path)
        if errors:
            print(f"\n{svg_path.name}:")
            for error in errors:
                print(error)
            total_errors += len(errors)

    if total_errors == 0:
        print("All SVG files passed overlap checks!")
        return 0
    else:
        print(f"\nTotal: {total_errors} overlap issue(s) found")
        return 1


if __name__ == '__main__':
    sys.exit(main())
