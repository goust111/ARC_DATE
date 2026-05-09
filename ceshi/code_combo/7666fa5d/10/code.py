# Combo variant of 7666fa5d (index 10)
# combo_spec: D2/1+D4/2
# Name: 水平线段+扩展填充至边缘

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import components_mask, get_bg as _bg_color

"""
Design analysis for combo D2/1 + D4/2

1. Concrete rule
   - D2/1 changes the source objects from anti-diagonal segments to horizontal segments.
   - Filling therefore switches from "scan each main diagonal" to "scan each column".
   - On one fixed column c, gather every horizontal segment whose span covers c and sort them by row.
   - D4/2 extends the fill beyond the outermost boundaries, so once a column has at least one active
     segment, every background cell in that column becomes fill color except the segment cells
     themselves.

2. Why the fused rule is observable
   - Inputs contain 4-5 horizontal segments on distinct rows with staggered spans, so the active row
     set varies from column to column.
   - Outputs therefore show vertical stripes whose notch pattern changes by column: some columns have
     one line notch, some have several. That makes the per-column boundary logic visible.
   - Generator rejects samples that would look like the D2-only inner-fill version by requiring many
     top-edge and bottom-edge fill cells that only appear under D4/2.

3. Solver assumption audit
   - The useful foreground is still one dominant line color. Generator keeps exactly one line color.
   - Each connected component of that color is assumed to be a straight horizontal segment with no
     gaps. Generator enforces that.
   - The old anti-diagonal extraction and pairwise interior-only fill are both invalid here; they are
     replaced with horizontal-segment extraction and column-wise full background fill.

4. Visual cleanliness
   - Only three colors appear: background 8, one line color, and red fill 2.
   - Rows are spaced apart and spans overlap but are not identical, avoiding ambiguous clutter.
"""


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

NUM_ROWS_CHOICES = (4, 5)
ROW_GAP_MIN = 2
ROW_GAP_MAX = 4
SEG_LEN_MIN = 4
SEG_LEN_MAX = 10
SEG_OVERLAP_MIN = 2
SEG_RETRY = 80
GEN_RETRY = 800

MIN_FILL_CELLS = 70
MIN_OUTER_ONLY_CELLS = 25
MIN_TOP_EDGE_FILL = 5
MIN_BOTTOM_EDGE_FILL = 5
MIN_SINGLE_ACTIVE_COLS = 2
MIN_MULTI_ACTIVE_COLS = 4
MIN_DISTINCT_ACTIVE_COUNTS = 3

Segment = tuple[int, int, int]  # row, col_lo, col_hi


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _extract_horizontal_segments(mask: np.ndarray) -> list[Segment]:
    segs: list[Segment] = []
    for pts in _cc4(mask):
        rows = {r for r, _c in pts}
        if len(rows) != 1:
            continue
        row = int(next(iter(rows)))
        cols = sorted(int(c) for _r, c in pts)
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        segs.append((row, int(cols[0]), int(cols[-1])))
    segs.sort(key=lambda seg: (seg[0], seg[1], seg[2]))
    return segs


def _apply_combo_fill(grid: np.ndarray, segs: list[Segment], *, bg: int, fill: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape

    for c in range(w):
        if not any(c0 <= c <= c1 for _row, c0, c1 in segs):
            continue
        for r in range(h):
            if out[r, c] == bg:
                out[r, c] = fill
    return out


def _apply_inner_only_fill(grid: np.ndarray, segs: list[Segment], *, bg: int, fill: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    w = out.shape[1]

    for c in range(w):
        active_rows = sorted(row for row, c0, c1 in segs if c0 <= c <= c1)
        if len(active_rows) < 2:
            continue
        for upper, lower in zip(active_rows, active_rows[1:]):
            for r in range(upper + 1, lower):
                if out[r, c] == bg:
                    out[r, c] = fill
    return out


def _column_active_counts(segs: list[Segment]) -> list[int]:
    counts = [0] * W
    for c in range(W):
        counts[c] = sum(1 for _row, c0, c1 in segs if c0 <= c <= c1)
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover horizontal line segments, then fill by columns.

    Input structure:
    - Background is the dominant color.
    - One other color forms several straight horizontal segments on distinct rows.

    Rule:
    - For each column, find every horizontal segment that covers that column.
    - If a column has at least one active segment, paint every background cell in that column red.
    - Segment cells themselves stay unchanged, so the output becomes a vertical stripe pattern with
      horizontal notches where the source segments sit.

    Key insight:
    - Horizontal segments are the boundaries, but D4/2 means the paint does not stop at the outer
      pair. It reaches the top and bottom grid edges whenever a column is activated at all.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain a foreground line color"

    line = max(non_bg, key=lambda color: int(np.sum(g == color)))
    segs = _extract_horizontal_segments(g == line)
    assert segs, "expected horizontal line segments"
    return _apply_combo_fill(g, segs, bg=bg, fill=FILL)


def _choose_rows() -> list[int] | None:
    num_rows = random.choice(NUM_ROWS_CHOICES)
    gaps = [random.randint(ROW_GAP_MIN, ROW_GAP_MAX) for _ in range(num_rows - 1)]
    total_gap = sum(gaps)
    top_min = 2
    top_max = H - 3 - total_gap
    if top_min > top_max:
        return None

    row = random.randint(top_min, top_max)
    rows = [row]
    for gap in gaps:
        row += gap
        rows.append(row)
    if rows[-1] > H - 3:
        return None
    return rows


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _random_interval(prev: tuple[int, int] | None) -> tuple[int, int] | None:
    starts = list(range(1, W - SEG_LEN_MIN))
    random.shuffle(starts)
    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        valid_starts = [s for s in starts if s + length - 1 <= W - 2]
        if not valid_starts:
            continue
        start = random.choice(valid_starts)
        end = start + length - 1
        if prev is not None:
            if _interval_overlap(start, end, prev[0], prev[1]) < SEG_OVERLAP_MIN:
                continue
        return start, end
    return None


def _build_segments() -> list[Segment] | None:
    rows = _choose_rows()
    if rows is None:
        return None

    segs: list[Segment] = []
    prev: tuple[int, int] | None = None
    for row in rows:
        interval = _random_interval(prev)
        if interval is None:
            return None
        segs.append((int(row), int(interval[0]), int(interval[1])))
        prev = interval
    return segs


def _segments_to_grid(segs: list[Segment], line: int) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for row, c0, c1 in segs:
        grid[row, c0 : c1 + 1] = int(line)
    return grid


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    segs = _build_segments()
    if segs is None:
        return None

    counts = _column_active_counts(segs)
    positive_counts = [count for count in counts if count > 0]
    if len(set(positive_counts)) < MIN_DISTINCT_ACTIVE_COUNTS:
        return None
    if sum(1 for count in counts if count == 1) < MIN_SINGLE_ACTIVE_COLS:
        return None
    if sum(1 for count in counts if count >= 2) < MIN_MULTI_ACTIVE_COLS:
        return None

    grid = _segments_to_grid(segs, line)
    combo = _apply_combo_fill(grid, segs, bg=BG, fill=FILL)
    inner = _apply_inner_only_fill(grid, segs, bg=BG, fill=FILL)

    combo_mask = (grid == BG) & (combo == FILL)
    inner_mask = (grid == BG) & (inner == FILL)
    outer_only = combo_mask & (~inner_mask)

    if int(combo_mask.sum()) < MIN_FILL_CELLS:
        return None
    if int(outer_only.sum()) < MIN_OUTER_ONLY_CELLS:
        return None
    if int(outer_only[0].sum()) < MIN_TOP_EDGE_FILL:
        return None
    if int(outer_only[-1].sum()) < MIN_BOTTOM_EDGE_FILL:
        return None

    if not np.array_equal(main(grid), combo):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 7666fa5d combo D2/1+D4/2")
