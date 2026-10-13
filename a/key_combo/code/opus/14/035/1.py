import random
from typing import Iterable

import numpy as np


_BG = 0
_PALETTE = (1, 2, 3, 4, 6)
_SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)

_GEN_LAYER_COUNTS = (3, 4, 5)
_GEN_OUTER_EXTRA_MAX = 6
_GEN_MARGIN_EXTRA_MIN = 2
_GEN_MARGIN_EXTRA_MAX = 6
_GEN_ATTEMPTS = 500
_USED_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], tuple[int, ...]]] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hollow_rectangle_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_rectangle(grid: np.ndarray, color: int) -> dict:
    mask = grid == int(color)
    r0, c0, r1, c1 = _bbox(mask)
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 3, "rectangle bbox must be at least 3x3"
    expected = _hollow_rectangle_mask(h, w)
    assert np.array_equal(patch, expected), "object must be a 1-pixel-wide hollow rectangle"
    return {"color": int(color), "bbox": (r0, c0, r1, c1)}


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 + 1 < ir0 and oc0 + 1 < ic0 and or1 - 1 > ir1 and oc1 - 1 > ic1


def _extract_nested_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = [int(c) for c in np.unique(g) if int(c) != _BG]
    assert 3 <= len(colors) <= 5, "expected 3 to 5 non-background rectangles"

    rectangles = [_extract_rectangle(g, color) for color in colors]
    depths = []
    for rect in rectangles:
        bbox = rect["bbox"]
        depth = 1 + sum(
            _strictly_contains(other["bbox"], bbox)
            for other in rectangles
            if other is not rect
        )
        depths.append(depth)
        rect["depth"] = int(depth)

    expected_depths = list(range(1, len(rectangles) + 1))
    assert sorted(depths) == expected_depths, "rectangles must form one strict nesting chain"

    rectangles.sort(key=lambda rect: rect["depth"])
    for outer, inner in zip(rectangles, rectangles[1:]):
        assert _strictly_contains(outer["bbox"], inner["bbox"])
    return rectangles


def _build_output(rectangles: Iterable[dict]) -> np.ndarray:
    ordered = list(rectangles)
    out = np.zeros((3, len(ordered) * 4 - 1), dtype=int)
    for idx, rect in enumerate(ordered):
        cell = np.zeros((3, 3), dtype=int)
        for r, c in _SPIRAL[: rect["depth"]]:
            cell[r, c] = int(rect["color"])
        c0 = idx * 4
        out[:, c0 : c0 + 3] = cell
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert nested hollow rectangles into a left-to-right row of depth-sized spiral gauges.

    Input structure:
    - The grid has black background and 3 to 5 single-color hollow rectangles.
    - Each rectangle is a closed 1-pixel-wide border, and the rectangles are strictly nested.
    - Different rectangles use different visible colors, with at least one background cell between neighboring borders.

    Rule:
    - Detect each hollow rectangle and recover its bounding box.
    - Compute nesting depth by strict bbox containment: outermost depth 1, next depth 2, and so on.
    - Output one 3x3 gauge per rectangle, ordered from shallowest depth to deepest, with 1 black separator column between gauges.
    - In each gauge, fill exactly `depth` cells with that rectangle's color along the clockwise spiral order
      top-left, top-middle, top-right, right-middle, bottom-right, bottom-middle, bottom-left, left-middle, center.

    Key insight:
    - The gauge fill amount is not a geometric measurement of one rectangle; it is the rectangle's topological position
      inside the full nesting chain, so containment must be reconstructed before the spiral gauges can be drawn.
    """
    rectangles = _extract_nested_rectangles(np.asarray(grid, dtype=int))
    return _build_output(rectangles)


def _draw_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _choose_shrinks(size: int, min_next_size: int) -> tuple[int, int] | None:
    choices = []
    for a in range(2, 5):
        for b in range(2, 5):
            if size - a - b >= min_next_size:
                choices.append((a, b))
    if not choices:
        return None
    return random.choice(choices)


def _build_nested_bboxes(layer_count: int) -> list[tuple[int, int, int, int]] | None:
    min_outer = 3 + 4 * (layer_count - 1)
    max_outer = min(24, min_outer + _GEN_OUTER_EXTRA_MAX)
    if min_outer > max_outer:
        return None

    outer_h = random.randint(min_outer, max_outer)
    outer_w = random.randint(min_outer, max_outer)
    boxes = [(0, 0, outer_h - 1, outer_w - 1)]

    for level in range(1, layer_count):
        r0, c0, r1, c1 = boxes[-1]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        remaining_inner = layer_count - level - 1
        min_next_size = 3 + 4 * remaining_inner

        vertical = _choose_shrinks(h, min_next_size)
        horizontal = _choose_shrinks(w, min_next_size)
        if vertical is None or horizontal is None:
            return None

        top, bottom = vertical
        left, right = horizontal
        boxes.append((r0 + top, c0 + left, r1 - bottom, c1 - right))

    return boxes


def _offset_bboxes(
    boxes: list[tuple[int, int, int, int]]
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    outer = boxes[0]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1

    extra_h = random.randint(_GEN_MARGIN_EXTRA_MIN, min(_GEN_MARGIN_EXTRA_MAX, 30 - outer_h))
    extra_w = random.randint(_GEN_MARGIN_EXTRA_MIN, min(_GEN_MARGIN_EXTRA_MAX, 30 - outer_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    shifted = []
    for r0, c0, r1, c1 in boxes:
        shifted.append((r0 + top, c0 + left, r1 + top, c1 + left))
    return shifted, (outer_h + extra_h, outer_w + extra_w)


def _expected_output(colors_outer_to_inner: list[int]) -> np.ndarray:
    rectangles = [
        {"color": int(color), "depth": depth}
        for depth, color in enumerate(colors_outer_to_inner, start=1)
    ]
    return _build_output(rectangles)


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, grid.shape)), tuple(map(int, grid.flatten()))


def _validate_case(grid: np.ndarray, expected: np.ndarray) -> None:
    rectangles = _extract_nested_rectangles(grid)
    assert len(rectangles) in _GEN_LAYER_COUNTS
    for outer, inner in zip(rectangles, rectangles[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert ir0 - or0 >= 2 and ic0 - oc0 >= 2
        assert or1 - ir1 >= 2 and oc1 - ic1 >= 2

    inner_r0, inner_c0, inner_r1, inner_c1 = rectangles[-1]["bbox"]
    assert np.all(grid[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1] == _BG)

    solved = main(grid)
    assert np.array_equal(solved, expected), "oracle mismatch"
    assert solved.shape == (3, len(rectangles) * 4 - 1)

    for idx, rect in enumerate(rectangles):
        cell = solved[:, idx * 4 : idx * 4 + 3]
        assert int(np.count_nonzero(cell == rect["color"])) == rect["depth"]
        assert np.count_nonzero(cell) == rect["depth"]
        if idx < len(rectangles) - 1:
            assert np.all(solved[:, idx * 4 + 3] == _BG)


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.choice(_GEN_LAYER_COUNTS)
    colors = random.sample(_PALETTE, layer_count)
    boxes = _build_nested_bboxes(layer_count)
    if boxes is None:
        return None
    boxes, shape = _offset_bboxes(boxes)

    grid = np.zeros(shape, dtype=int)
    for color, bbox in zip(colors, boxes):
        _draw_rectangle(grid, bbox, color)

    expected = _expected_output(colors)
    if _output_signature(expected) in _USED_OUTPUT_SIGNATURES:
        return None

    try:
        _validate_case(grid, expected)
    except Exception:
        return None

    _USED_OUTPUT_SIGNATURES.add(_output_signature(expected))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle sample")
