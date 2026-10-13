from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
BLACK = 0

BBox = tuple[int, int, int, int]


def bbox(mask: np.ndarray) -> BBox | None:
    """ARC bbox helper copied in the same (r0, c0, r1, c1) convention as arc_utils."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _analyze_grid(grid: np.ndarray) -> tuple[BBox, list[int], list[int], dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    box = bbox(g == BLUE)
    assert box is not None, "expected blue grid lines"
    r0, c0, _r1, _c1 = box
    patch = crop_bbox(g, box)
    blue = patch == BLUE

    horizontal_local = [
        r for r in range(patch.shape[0]) if np.any(blue[r, :-1] & blue[r, 1:])
    ]
    vertical_local = [
        c for c in range(patch.shape[1]) if np.any(blue[:-1, c] & blue[1:, c])
    ]

    horizontal_rows = [r0 + r for r in horizontal_local]
    vertical_cols = [c0 + c for c in vertical_local]
    horizontal_gaps = {r0 + r: int(np.count_nonzero(patch[r] == BLACK)) for r in horizontal_local}
    vertical_gaps = {c0 + c: int(np.count_nonzero(patch[:, c] == BLACK)) for c in vertical_local}
    return box, horizontal_rows, vertical_cols, horizontal_gaps, vertical_gaps


def _stroke_color(gap_count: int) -> int:
    return GREEN if int(gap_count) == 1 else YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect valid blue grid lines, fill their gaps red, and outline both sides using a color keyed by gap count.

    Input structure:
    - A black background contains blue orthogonal grid lines inside one foreground bounding box.
    - Valid horizontal lines are rows with at least one horizontally adjacent blue pair inside that bbox.
    - Valid vertical lines are columns with at least one vertically adjacent blue pair inside that bbox.

    Rule:
    - Within the blue bbox, fill every black gap on each valid line with red while keeping original blue pixels unchanged.
    - Count each line's original black gap total G inside the bbox.
    - For each valid horizontal line, draw a 1-pixel outline on the row above and below it across the bbox width; use green when G = 1 and yellow when G >= 2.
    - Then do the same for each valid vertical line on the columns left and right of it across the bbox height, skipping any cell that already holds blue, red, or an earlier outline color.

    Key insight:
    - The same missing black cells play two roles at once: they identify where red repairs happen and, through their per-line count, determine the later outline color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    box, horizontal_rows, vertical_cols, horizontal_gaps, vertical_gaps = _analyze_grid(g)
    r0, c0, r1, c1 = box

    out = g.copy()

    for r in horizontal_rows:
        for c in range(c0, c1 + 1):
            if g[r, c] == BLACK:
                out[r, c] = RED

    for c in vertical_cols:
        for r in range(r0, r1 + 1):
            if g[r, c] == BLACK:
                out[r, c] = RED

    for r in horizontal_rows:
        color = _stroke_color(horizontal_gaps[r])
        for rr in (r - 1, r + 1):
            if not (r0 <= rr <= r1):
                continue
            for c in range(c0, c1 + 1):
                if out[rr, c] == BLACK:
                    out[rr, c] = color

    for c in vertical_cols:
        color = _stroke_color(vertical_gaps[c])
        for cc in (c - 1, c + 1):
            if not (c0 <= cc <= c1):
                continue
            for r in range(r0, r1 + 1):
                if out[r, cc] == BLACK:
                    out[r, cc] = color

    return out


AXIS_RETRIES = 120
GEN_RETRIES = 300


def _sample_axis_positions() -> list[int] | None:
    for _ in range(AXIS_RETRIES):
        count = random.choice([3, 4])
        gaps = [random.randint(3, 5) for _ in range(count - 1)]
        positions = [0]
        for gap in gaps:
            positions.append(positions[-1] + gap)
        span = positions[-1] + 1
        if 10 <= span <= 14:
            return positions
    return None


def _sample_gap_counts(n: int) -> list[int]:
    counts = [1, random.randint(2, 3)]
    while len(counts) < n:
        counts.append(random.randint(1, 3))
    random.shuffle(counts)
    assert 1 in counts and any(v >= 2 for v in counts)
    return counts


def _build_instance() -> tuple[np.ndarray, dict] | None:
    row_positions = _sample_axis_positions()
    col_positions = _sample_axis_positions()
    if row_positions is None or col_positions is None:
        return None

    box_h = row_positions[-1] + 1
    box_w = col_positions[-1] + 1
    top = random.randint(1, 5)
    left = random.randint(1, 5)
    bottom = random.randint(1, 5)
    right = random.randint(1, 5)
    grid_h = top + box_h + bottom
    grid_w = left + box_w + right
    if max(grid_h, grid_w) > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    r0 = top
    c0 = left

    global_rows = [r0 + r for r in row_positions]
    global_cols = [c0 + c for c in col_positions]

    grid[r0 : r0 + box_h, global_cols] = BLUE
    grid[global_rows, c0 : c0 + box_w] = BLUE

    horizontal_gap_counts = _sample_gap_counts(len(global_rows))
    vertical_gap_counts = _sample_gap_counts(len(global_cols))

    horizontal_candidates = [c0 + c for c in range(box_w) if c not in col_positions]
    vertical_candidates = [r0 + r for r in range(box_h) if r not in row_positions]
    if len(horizontal_candidates) < 3 or len(vertical_candidates) < 3:
        return None

    horizontal_gaps: dict[int, list[int]] = {}
    vertical_gaps: dict[int, list[int]] = {}

    for r, gap_count in zip(global_rows, horizontal_gap_counts):
        gap_cols = sorted(random.sample(horizontal_candidates, gap_count))
        grid[r, gap_cols] = BLACK
        horizontal_gaps[int(r)] = [int(c) for c in gap_cols]

    for c, gap_count in zip(global_cols, vertical_gap_counts):
        gap_rows = sorted(random.sample(vertical_candidates, gap_count))
        grid[gap_rows, c] = BLACK
        vertical_gaps[int(c)] = [int(r) for r in gap_rows]

    meta = {
        "box": (r0, c0, r0 + box_h - 1, c0 + box_w - 1),
        "horizontal_rows": global_rows,
        "vertical_cols": global_cols,
        "horizontal_gap_counts": {r: len(cols) for r, cols in horizontal_gaps.items()},
        "vertical_gap_counts": {c: len(rows) for c, rows in vertical_gaps.items()},
    }
    return grid, meta


def _count_only_rows(grid: np.ndarray, box: BBox) -> list[int]:
    r0, c0, r1, c1 = box
    rows = []
    for r in range(r0, r1 + 1):
        if np.count_nonzero(grid[r, c0 : c1 + 1] == BLUE) >= 2:
            rows.append(r)
    return rows


def _count_only_cols(grid: np.ndarray, box: BBox) -> list[int]:
    r0, c0, r1, c1 = box
    cols = []
    for c in range(c0, c1 + 1):
        if np.count_nonzero(grid[r0 : r1 + 1, c] == BLUE) >= 2:
            cols.append(c)
    return cols


def _validate_semantics(inp: np.ndarray, meta: dict) -> bool:
    box, horizontal_rows, vertical_cols, horizontal_gaps, vertical_gaps = _analyze_grid(inp)
    if box != meta["box"]:
        return False
    if horizontal_rows != meta["horizontal_rows"]:
        return False
    if vertical_cols != meta["vertical_cols"]:
        return False
    if horizontal_gaps != meta["horizontal_gap_counts"]:
        return False
    if vertical_gaps != meta["vertical_gap_counts"]:
        return False

    if sum(v == 1 for v in horizontal_gaps.values()) < 1:
        return False
    if sum(v >= 2 for v in horizontal_gaps.values()) < 1:
        return False
    if sum(v == 1 for v in vertical_gaps.values()) < 1:
        return False
    if sum(v >= 2 for v in vertical_gaps.values()) < 1:
        return False

    count_rows = _count_only_rows(inp, box)
    count_cols = _count_only_cols(inp, box)
    if count_rows == horizontal_rows or count_cols == vertical_cols:
        return False

    has_row_counterexample = any(r not in horizontal_rows for r in count_rows)
    has_col_counterexample = any(c not in vertical_cols for c in count_cols)
    if not has_row_counterexample or not has_col_counterexample:
        return False

    out = main(inp)
    if np.array_equal(out, inp):
        return False

    colors = set(map(int, np.unique(out)))
    if not {RED, GREEN, YELLOW}.issubset(colors):
        return False

    for r, gap_count in horizontal_gaps.items():
        expected = _stroke_color(gap_count)
        touched = False
        for rr in (r - 1, r + 1):
            if not (box[0] <= rr <= box[2]):
                continue
            if np.any(out[rr, box[1] : box[3] + 1] == expected):
                touched = True
        if not touched:
            return False

    for c, gap_count in vertical_gaps.items():
        expected = _stroke_color(gap_count)
        touched = False
        for cc in (c - 1, c + 1):
            if not (box[1] <= cc <= box[3]):
                continue
            if np.any(out[box[0] : box[2] + 1, cc] == expected):
                touched = True
        if not touched:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_instance()
        if built is None:
            continue
        inp, meta = built
        if _validate_semantics(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
