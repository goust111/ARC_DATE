from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3

PAIR_CASES: list[tuple[str, bool]] = [
    ("top", True),
    ("bottom", False),
    ("left", True),
    ("right", False),
    ("top", False),
    ("bottom", True),
    ("left", False),
    ("right", True),
]
CASE_INDEX = 0
GENERATION_RETRIES = 200


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _opposite(side: str) -> str:
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _side_of_point(r: int, c: int, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    matches = []
    if r == r0:
        matches.append("top")
    if r == r1:
        matches.append("bottom")
    if c == c0:
        matches.append("left")
    if c == c1:
        matches.append("right")
    assert len(matches) == 1, "each marker must lie on exactly one bbox side"
    return matches[0]


def _analyze_grid(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], str, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, RED}, "input must contain only black and red"

    coords = np.argwhere(g == RED)
    assert coords.shape == (3, 2), "expected exactly three red marker pixels"
    assert int(np.count_nonzero(g)) == 3, "only the three markers may be non-background"

    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "bbox must be at least 4x4"
    assert np.all(
        (coords[:, 0] == r0) | (coords[:, 0] == r1) | (coords[:, 1] == c0) | (coords[:, 1] == c1)
    ), "all markers must lie on the bbox boundary"

    row_values = [int(r) for r in coords[:, 0]]
    col_values = [int(c) for c in coords[:, 1]]
    unique_rows = sorted(set(row_values))
    unique_cols = sorted(set(col_values))

    horizontal = len(unique_rows) == 2 and len(unique_cols) == 3
    vertical = len(unique_rows) == 3 and len(unique_cols) == 2
    assert horizontal ^ vertical, "marker pattern must encode exactly one split axis"

    if horizontal:
        pair_row = next(row for row in unique_rows if row_values.count(row) == 2)
        single_row = next(row for row in unique_rows if row_values.count(row) == 1)
        assert {pair_row, single_row} == {r0, r1}, "horizontal markers must occupy the top and bottom edges"
        pair_side = "top" if pair_row == r0 else "bottom"
        single_side = _opposite(pair_side)
    else:
        pair_col = next(col for col in unique_cols if col_values.count(col) == 2)
        single_col = next(col for col in unique_cols if col_values.count(col) == 1)
        assert {pair_col, single_col} == {c0, c1}, "vertical markers must occupy the left and right edges"
        pair_side = "left" if pair_col == c0 else "right"
        single_side = _opposite(pair_side)
    return g, coords, bbox, pair_side, single_side


def _border_color(r: int, c: int, bbox: tuple[int, int, int, int], pair_side: str) -> int:
    r0, c0, r1, c1 = bbox
    if pair_side in ("top", "bottom"):
        cut_low = (r0 + r1) // 2
        cut_high = (r0 + r1 + 1) // 2
        if pair_side == "top":
            return GREEN if r <= cut_low else BLUE
        return GREEN if r >= cut_high else BLUE

    cut_low = (c0 + c1) // 2
    cut_high = (c0 + c1 + 1) // 2
    if pair_side == "left":
        return GREEN if c <= cut_low else BLUE
    return GREEN if c >= cut_high else BLUE


def _render_output(
    shape: tuple[int, int],
    marker_coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    pair_side: str,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    r0, c0, r1, c1 = bbox

    for c in range(c0, c1 + 1):
        out[r0, c] = _border_color(r0, c, bbox, pair_side)
        out[r1, c] = _border_color(r1, c, bbox, pair_side)
    for r in range(r0, r1 + 1):
        out[r, c0] = _border_color(r, c0, bbox, pair_side)
        out[r, c1] = _border_color(r, c1, bbox, pair_side)

    out[marker_coords[:, 0], marker_coords[:, 1]] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the rectangle from three red markers and split its border colors by the 2-vs-1 side pattern.

    Input structure:
    - The grid is black except for exactly three red marker pixels.
    - The markers lie on the boundary of their shared bounding box, with two on one side and one on the opposite side.
    - Each marker is edge-only rather than corner-ambiguous, so the 2-vs-1 side pattern is uniquely readable.

    Rule:
    - Compute the bounding box of the three red markers.
    - Identify which side has two markers and which opposite side has the remaining single marker.
    - Draw the full rectangular border of that box, splitting it across the box midpoint: the half containing the double-marker side is green and the opposite half is blue, with the midpoint row or column assigned to the double-marker half.
    - Restore the three original red markers on top of the border.

    Key insight:
    - The same three markers encode both the rectangle geometry and the direction of the green-vs-blue half-border split.
    """
    g, coords, bbox, pair_side, _single_side = _analyze_grid(grid)
    return _render_output(g.shape, coords, bbox, pair_side)


def _pick_size(min_size: int, max_size: int, want_odd: bool) -> int:
    choices = [size for size in range(min_size, max_size + 1) if (size % 2 == 1) == want_odd]
    return random.choice(choices)


def _next_case() -> tuple[str, bool]:
    global CASE_INDEX
    case = PAIR_CASES[CASE_INDEX % len(PAIR_CASES)]
    CASE_INDEX += 1
    return case


def _build_input(pair_side: str, split_axis_odd: bool) -> np.ndarray:
    if pair_side in ("top", "bottom"):
        box_h = _pick_size(5, 10, split_axis_odd)
        box_w = random.randint(5, 10)
    else:
        box_h = random.randint(5, 10)
        box_w = _pick_size(5, 10, split_axis_odd)

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    height = top_margin + box_h + bottom_margin
    width = left_margin + box_w + right_margin
    assert max(height, width) <= 30

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    grid = np.zeros((height, width), dtype=int)

    if pair_side in ("top", "bottom"):
        interior_cols = list(range(c0 + 1, c1))
        pair_mid_col = random.choice(interior_cols)
        if random.choice([False, True]):
            pair_cols = [c0, pair_mid_col]
            single_col = c1
        else:
            pair_cols = [pair_mid_col, c1]
            single_col = c0
        pair_row = r0 if pair_side == "top" else r1
        single_row = r1 if pair_side == "top" else r0
        markers = [(pair_row, pair_cols[0]), (pair_row, pair_cols[1]), (single_row, single_col)]
    else:
        interior_rows = list(range(r0 + 1, r1))
        pair_mid_row = random.choice(interior_rows)
        if random.choice([False, True]):
            pair_rows = [r0, pair_mid_row]
            single_row = r1
        else:
            pair_rows = [pair_mid_row, r1]
            single_row = r0
        pair_col = c0 if pair_side == "left" else c1
        single_col = c1 if pair_side == "left" else c0
        markers = [(pair_rows[0], pair_col), (pair_rows[1], pair_col), (single_row, single_col)]

    for r, c in markers:
        grid[r, c] = RED
    return grid


def _semantic_checks(grid: np.ndarray, expected_pair_side: str) -> None:
    g, coords, bbox, pair_side, _single_side = _analyze_grid(grid)
    assert pair_side == expected_pair_side

    out = main(g)
    expected = _render_output(g.shape, coords, bbox, pair_side)
    assert np.array_equal(out, expected)

    r0, c0, r1, c1 = bbox
    border_mask = np.zeros_like(out, dtype=bool)
    border_mask[r0, c0 : c1 + 1] = True
    border_mask[r1, c0 : c1 + 1] = True
    border_mask[r0 : r1 + 1, c0] = True
    border_mask[r0 : r1 + 1, c1] = True

    border_colors = set(map(int, np.unique(out[border_mask]))) - {RED}
    assert border_colors == {BLUE, GREEN}, "both border colors must appear"

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not border_mask[r, c] or int(out[r, c]) == RED:
                continue
            assert int(out[r, c]) == _border_color(r, c, bbox, pair_side)

    if pair_side in ("top", "bottom") and (r1 - r0 + 1) % 2 == 1:
        mid = (r0 + r1) // 2
        assert int(out[mid, c0]) == GREEN and int(out[mid, c1]) == GREEN
    if pair_side in ("left", "right") and (c1 - c0 + 1) % 2 == 1:
        mid = (c0 + c1) // 2
        assert int(out[r0, mid]) == GREEN and int(out[r1, mid]) == GREEN


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        pair_side, split_axis_odd = _next_case()
        grid = _build_input(pair_side, split_axis_odd)
        _semantic_checks(grid, pair_side)
        return grid
    raise RuntimeError("failed to generate a valid input")
