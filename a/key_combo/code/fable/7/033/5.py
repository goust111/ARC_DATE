"""ARC task 5: reconstruct mutually occluding frames from exterior corner traces."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OVERLAP_COLOR = 4
MARKER_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _bbox_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    """Return an inclusive (r0, c0, r1, c1) bbox, following arc_utils."""
    coords = np.argwhere(grid == color)
    assert coords.shape == (4, 2), "each marker color must occur exactly four times"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    expected = {
        (int(r0), int(c0)),
        (int(r0), int(c1)),
        (int(r1), int(c0)),
        (int(r1), int(c1)),
    }
    assert {tuple(map(int, p)) for p in coords} == expected, (
        "each color must occupy the four corners of its marker bbox"
    )
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "reconstructed frame needs a strict interior"
    return int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)


def _strict_interior_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct two hidden frames from exterior corner traces, erase mutual occlusions, and fill their common interior.

    Input structure:
    A black grid contains exactly two non-yellow marker colors, each appearing at the
    four corners of a bounding box.  Each marker is diagonally one cell outside the
    corresponding corner of an unpainted rectangular frame, and the two strict frame
    interiors overlap.

    Rule:
    For each color, shrink its four-marker bounding box inward by one cell and draw
    that color as a one-pixel rectangular frame, clearing all original markers.  Erase
    every frame pixel lying strictly inside the other frame, then color every cell in
    the intersection of the two strict interiors yellow.

    Key insight:
    The four visible pixels are exterior diagonal traces rather than frame corners;
    only after the one-cell inward reconstruction are occluded borders and the common
    interior determined in the correct coordinate system.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids are at most 30 by 30"
    values = set(map(int, np.unique(g)))
    assert BACKGROUND in values, "expected black background"
    colors = sorted(values - {BACKGROUND})
    assert len(colors) == 2 and OVERLAP_COLOR not in colors, (
        "expected exactly two distinct non-yellow marker colors"
    )

    boxes = [_bbox_for_color(g, color) for color in colors]
    interiors = [_strict_interior_mask(g.shape, box) for box in boxes]
    common = interiors[0] & interiors[1]
    assert np.any(common), "the two strict interiors must overlap"

    out = np.zeros(g.shape, dtype=g.dtype)
    for i, color in enumerate(colors):
        visible_border = _frame_mask(g.shape, boxes[i]) & ~interiors[1 - i]
        out[visible_border] = color
    out[common] = OVERLAP_COLOR
    return out


def _crossing_offsets(size_a: int, size_b: int) -> list[int]:
    """Offsets where B starts inside A, extends beyond it, and interiors overlap."""
    return [
        d
        for d in range(1, size_a - 2)
        if d + size_b > size_a and size_a - d - 2 >= 1
    ]


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 30)
    width = random.randint(14, 30)
    ah, aw = random.randint(6, 11), random.randint(6, 11)
    bh, bw = random.randint(6, 11), random.randint(6, 11)
    row_offsets = _crossing_offsets(ah, bh)
    col_offsets = _crossing_offsets(aw, bw)
    if not row_offsets or not col_offsets:
        return None

    dr, dc = random.choice(row_offsets), random.choice(col_offsets)
    # Independent reflections cover all four diagonal crossing orientations.
    rel_br = dr if random.choice((False, True)) else ah - bh - dr
    rel_bc = dc if random.choice((False, True)) else aw - bw - dc

    rel_boxes = [(0, 0, ah - 1, aw - 1), (rel_br, rel_bc, rel_br + bh - 1, rel_bc + bw - 1)]
    min_r = min(b[0] for b in rel_boxes) - 1
    min_c = min(b[1] for b in rel_boxes) - 1
    max_r = max(b[2] for b in rel_boxes) + 1
    max_c = max(b[3] for b in rel_boxes) + 1
    span_h, span_w = max_r - min_r + 1, max_c - min_c + 1
    if span_h > height or span_w > width:
        return None
    shift_r = random.randint(-min_r, height - 1 - max_r)
    shift_c = random.randint(-min_c, width - 1 - max_c)
    boxes = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in rel_boxes
    ]

    marker_sets = []
    for r0, c0, r1, c1 in boxes:
        marker_sets.append({(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)})
    if marker_sets[0] & marker_sets[1]:
        return None

    interiors = [_strict_interior_mask((height, width), b) for b in boxes]
    frames = [_frame_mask((height, width), b) for b in boxes]
    common = interiors[0] & interiors[1]
    erased = [int(np.count_nonzero(frames[i] & interiors[1 - i])) for i in range(2)]
    # Semantic anti-degeneration: both occlusions and the fill visibly matter,
    # while the overlap remains a minority of the combined rectangle interiors.
    union_interior = interiors[0] | interiors[1]
    if not np.any(common) or min(erased) < 2:
        return None
    if np.count_nonzero(common) * 3 > np.count_nonzero(union_interior):
        return None

    colors = random.sample(MARKER_COLORS, 2)
    grid = np.zeros((height, width), dtype=int)
    for color, points in zip(colors, marker_sets):
        for point in points:
            grid[point] = color

    # Verify that parsing the visible clues recovers exactly the intended boxes.
    recovered = {_bbox_for_color(grid, color) for color in colors}
    if recovered != set(boxes):
        return None
    out = main(grid)
    if not np.any(out == OVERLAP_COLOR):
        return None
    # A residual point must have only its stated role: after erasure it may not
    # coincide with the other frame or the yellow overlap in the final image.
    if any(int(out[point]) != BACKGROUND for points in marker_sets for point in points):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate pairs of crossing hidden rectangles."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid exterior-corner-trace puzzle")
