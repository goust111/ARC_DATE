"""ARC task 6: reconstruct hidden rectangles from two colored projections."""

from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
FRAME_COLORS = (1, 3, 4, 5, 6, 8, 9)
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _single_run(edge: np.ndarray, color: int, offset: int) -> tuple[int, int]:
    """Return the inclusive grid-coordinate span of one contiguous edge run."""
    indices = np.flatnonzero(edge == color)
    assert indices.size >= 3, "each projection must describe a rectangle with an interior"
    assert np.all(np.diff(indices) == 1), "each color must form exactly one contiguous run"
    return int(indices[0]) + offset, int(indices[-1]) + offset


def _flood_enclosed(out: np.ndarray, seed: tuple[int, int], color: int) -> int:
    """Four-connected flood fill of the black region enclosed by a rebuilt frame."""
    height, width = out.shape
    sr, sc = seed
    assert out[sr, sc] == BLACK, "a reconstructed rectangle must have a black interior"

    stack = [(sr, sc)]
    out[sr, sc] = color
    area = 0
    while stack:
        r, c = stack.pop()
        area += 1
        for dr, dc in FOUR_NEIGHBORS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and out[rr, cc] == BLACK:
                out[rr, cc] = color
                stack.append((rr, cc))
    return area


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild hidden colored rectangles from their top/left projections and parity-fill their interiors.

    Input structure:
    - The background is black, and every cell away from the top row and left column is black.
    - Each foreground color occurs as exactly one contiguous run on the top row and one on the left column.
    - Same-color runs encode one non-edge-touching rectangle; encoded rectangles do not overlap.

    Rule:
    - For each color, use its left-column run as the inclusive row span and its top-row run as the
      inclusive column span, then draw the one-pixel-thick rectangular frame at their intersection.
    - Flood-fill the enclosed interior red when its area is even and orange when its area is odd.
    - Remove the two projection axes, leaving them black in the output.

    Key insight:
    - Color pairs two one-dimensional shadows into an invisible rectangle; only after combining both
      axes can the enclosed area parity, and therefore its fill color, be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 6 and width >= 6, "grid is too small for an interior rectangle"
    assert int(g[0, 0]) == BLACK, "the projection axes meet at a black corner"
    assert np.all(g[1:, 1:] == BLACK), "the input interior must be entirely black"

    top_colors = set(map(int, np.unique(g[0, 1:]))) - {BLACK}
    left_colors = set(map(int, np.unique(g[1:, 0]))) - {BLACK}
    assert top_colors and top_colors == left_colors, "both axes must contain the same color set"

    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in top_colors:
        r0, r1 = _single_run(g[1:, 0], color, offset=1)
        c0, c1 = _single_run(g[0, 1:], color, offset=1)
        assert 0 < r0 < r1 < height - 1 and 0 < c0 < c1 < width - 1, (
            "hidden rectangles must not touch the grid boundary"
        )
        rectangles.append((color, r0, c0, r1, c1))

    for i, (_, ar0, ac0, ar1, ac1) in enumerate(rectangles):
        for _, br0, bc0, br1, bc1 in rectangles[i + 1 :]:
            row_overlap = max(ar0, br0) <= min(ar1, br1)
            col_overlap = max(ac0, bc0) <= min(ac1, bc1)
            assert not (row_overlap and col_overlap), "hidden rectangles must not overlap"

    # Reconstruct every boundary before filling.  The subsequent flood is therefore
    # genuinely defined by the invisible frames inferred from both projection axes.
    out = np.full(g.shape, BLACK, dtype=int)
    for color, r0, c0, r1, c1 in rectangles:
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    for _color, r0, c0, r1, c1 in rectangles:
        expected_area = (r1 - r0 - 1) * (c1 - c0 - 1)
        fill = RED if expected_area % 2 == 0 else ORANGE
        filled_area = _flood_enclosed(out, (r0 + 1, c0 + 1), fill)
        assert filled_area == expected_area, "the rebuilt frame must enclose exactly its stated interior"

    return out


def _packed_intervals(
    order: list[int], lengths: list[int]
) -> tuple[dict[int, tuple[int, int]], int]:
    """Pack disjoint runs with visible black gaps and return spans plus axis size."""
    lead = random.randint(1, 2)
    trail = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(len(order) - 1)]
    cursor = 1 + lead
    spans: dict[int, tuple[int, int]] = {}
    for position, object_index in enumerate(order):
        length = lengths[object_index]
        spans[object_index] = (cursor, cursor + length - 1)
        cursor += length
        if position < len(gaps):
            cursor += gaps[position]
    axis_size = cursor + trail + 1
    return spans, axis_size


def _sample_dimensions(count: int) -> list[tuple[int, int]]:
    """Cover both parity branches and refute height-only or width-only fill rules."""
    odd = (random.choice((3, 5)), random.choice((3, 5)))
    odd_even = (random.choice((3, 5)), 4)
    even_odd = (4, random.choice((3, 5)))
    dimensions = [odd, odd_even, even_odd]
    if count == 4:
        dimensions.append((4, 4))
    random.shuffle(dimensions)
    return dimensions


def _explicit_expected(
    shape: tuple[int, int], objects: list[tuple[int, int, int, int, int]]
) -> np.ndarray:
    """Independent, direct renderer used as the generator's semantic oracle."""
    expected = np.zeros(shape, dtype=int)
    for color, r0, c0, r1, c1 in objects:
        expected[r0, c0 : c1 + 1] = color
        expected[r1, c0 : c1 + 1] = color
        expected[r0 : r1 + 1, c0] = color
        expected[r0 : r1 + 1, c1] = color
        area = (r1 - r0 - 1) * (c1 - c0 - 1)
        expected[r0 + 1 : r1, c0 + 1 : c1] = RED if area % 2 == 0 else ORANGE
    return expected


def _semantically_valid(
    inp: np.ndarray,
    objects: list[tuple[int, int, int, int, int]],
    row_order: list[int],
    col_order: list[int],
) -> bool:
    """Check puzzle rules and anti-shortcut properties, not merely solver closure."""
    if inp.ndim != 2 or max(inp.shape) > 30 or np.any(inp[1:, 1:] != BLACK):
        return False
    if any(a == b for a, b in zip(row_order, col_order)):
        return False  # no color keeps the same rank on both axes

    areas = [(r1 - r0 - 1) * (c1 - c0 - 1) for _, r0, c0, r1, c1 in objects]
    heights = [r1 - r0 + 1 for _, r0, _c0, r1, _c1 in objects]
    widths = [c1 - c0 + 1 for _, _r0, c0, _r1, c1 in objects]
    if not (any(a % 2 for a in areas) and any(a % 2 == 0 for a in areas)):
        return False
    if not any(h % 2 == 1 and w % 2 == 0 for h, w in zip(heights, widths)):
        return False
    if not any(h % 2 == 0 and w % 2 == 1 for h, w in zip(heights, widths)):
        return False

    # Projection spans must exactly match their associated rectangles, and both
    # axes must be needed to distinguish all objects.
    for color, r0, c0, r1, c1 in objects:
        if not np.array_equal(np.flatnonzero(inp[:, 0] == color), np.arange(r0, r1 + 1)):
            return False
        if not np.array_equal(np.flatnonzero(inp[0, :] == color), np.arange(c0, c1 + 1)):
            return False
    if len({(r0, r1) for _, r0, _c0, r1, _c1 in objects}) != len(objects):
        return False
    if len({(c0, c1) for _, _r0, c0, _r1, c1 in objects}) != len(objects):
        return False

    expected = _explicit_expected(inp.shape, objects)
    try:
        actual = main(inp)
    except (AssertionError, IndexError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, inp):
        return False
    if np.any(actual[0, :] != BLACK) or np.any(actual[:, 0] != BLACK):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked dual-projection reconstruction puzzle."""
    for _ in range(500):
        count = random.choice((3, 4))
        dimensions = _sample_dimensions(count)
        colors = random.sample(FRAME_COLORS, count)

        row_order = list(range(count))
        random.shuffle(row_order)
        # A nonzero cyclic shift is a derangement of the row order, ensuring that
        # matching by positional rank instead of visible color always fails.
        shift = random.randint(1, count - 1)
        col_order = row_order[shift:] + row_order[:shift]

        row_spans, height = _packed_intervals(row_order, [h for h, _ in dimensions])
        col_spans, width = _packed_intervals(col_order, [w for _, w in dimensions])
        if height > 30 or width > 30:
            continue

        inp = np.zeros((height, width), dtype=int)
        objects: list[tuple[int, int, int, int, int]] = []
        for index, color in enumerate(colors):
            r0, r1 = row_spans[index]
            c0, c1 = col_spans[index]
            inp[r0 : r1 + 1, 0] = color
            inp[0, c0 : c1 + 1] = color
            objects.append((color, r0, c0, r1, c1))

        if _semantically_valid(inp, objects, row_order, col_order):
            return inp
    raise RuntimeError("failed to generate a valid shadow-reconstruction puzzle")
