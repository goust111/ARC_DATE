# Combo variant of 136b0064 (index 7)
# combo_spec: D10/0+D12/1
# Name: 环绕画布上的对角折线

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant D10/0 + D12/1
# a) Concrete behavior and trace:
#    - Input geometry stays the original one from the spec: two left tile columns of 3x3 glyphs,
#      divider at column 7, one gray start on the right, and output width 7 after removing the divider.
#    - The fusion changes the motion rule in two coupled ways:
#      1. plus(area 4) no longer means a short vertical stroke; it becomes a 3-cell descending diagonal.
#         The tile column is the visible cue: left tile column (c0=0) means down-right, right tile
#         column (c0=4) means down-left.
#      2. Any step that changes column now wraps modulo the 7-column canvas. This applies both to
#         horizontal segments and to each diagonal sub-step.
#    - Small example on width 7 from current endpoint (0, 6):
#         original left-column area 4: (1,6),(2,6) and stop at col 6
#         fused    left-column area 4: (1,6),(2,0),(3,1)
#      The same glyph both changes shape semantics and demonstrates wrap. A later area 6 from col 1
#      gives (4,1),(4,0),(4,6),(4,5), so both diagonal and horizontal wrap can coexist in one path.
# b) Complexity calibration:
#    - The rule space stays small: four glyph families, fixed color-shape binding, and only one
#      extra geometric cue (tile column for area 4 direction).
#    - To make the fusion visually recoverable, the generator enforces:
#      at least one plus glyph, at least one wrap event, and at least one wrap caused by a diagonal
#      plus segment specifically. It also keeps ordinary horizontal segments so the wrap rule is seen
#      across more than one segment family.
# c) Broken original assumptions:
#    - `if area == 4` used to assume "column never changes" and "row cost is 2"; both are false now.
#      The solver must update row and column step-by-step, and the generator must budget +3 rows for area 4.
#    - The old `assert 0 <= end_c < div_c` for horizontal segments is invalid because a segment may
#      leave the canvas and re-enter. Endpoint-only checking is insufficient; wrapped motion must be
#      simulated cell-by-cell.
#    - A single `_AREA_DELTA[4]` no longer works because plus direction depends on which tile column
#      the glyph came from. The generator therefore derives the diagonal sign from the instruction index.
# d) Placement legality:
#    - Input glyph placement remains legal because the original fixed 3x3 slots already keep components
#      disjoint. The gray start still shifts into columns 0..6 after divider removal.
#    - Output legality is enforced by simulating the exact wrapped trace used by the solver. Rows never
#      wrap; only columns do. The row budget check guarantees every generated segment stays within 0..H-1.
# e) Visual observability:
#    - Every sample contains at least one plus glyph and at least one wrapped plus trace, so the new
#      diagonal-wrap behavior is not optional noise.
#    - The input itself exposes the left-vs-right diagonal cue because identical plus glyphs appear in
#      fixed tile columns, and the output shows corresponding down-right or down-left wrapped diagonals.
# f) Visual neatness:
#    - The path remains monotone downward in rows, so later segments do not climb back into earlier rows.
#      With width 7 and maximum lateral run 4, wrapped fragments stay sparse rather than turning into a scribble.
# g) Color inference:
#    - Colors are unchanged from the base task: plus->6, diamond->1, cross->3, U->2.
#      The solver should infer new geometry, not a second color codebook.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes plus direction is determined by whether the glyph bbox starts in the left or
#      right tile column; `generate_input()` always uses the canonical tile slots that make this readable.
#    - `main()` assumes wrap is column-only; `generate_input()` only samples traces that satisfy the row budget.
#    - `main()` assumes one full-height divider and one gray start cell; the generator constructs exactly that.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _diag_dc_from_glyph_col(glyph_c0: int, div_c: int) -> int:
    """Left tile column => down-right diagonal; right tile column => down-left diagonal."""
    return 1 if int(glyph_c0) < int(div_c // 2) else -1


def _diag_dc_from_index(idx: int, n_rows: int) -> int:
    """Instruction indices 0..N-1 are in the left tile column; N..2N-1 are in the right one."""
    return 1 if int(idx) < int(n_rows) else -1


def _trace_segment_cells(
    r_cur: int,
    c_cur: int,
    area: int,
    *,
    width: int,
    diag_dc: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Return the painted cells for one glyph plus the number of lateral wrap events it caused."""
    rr = int(r_cur + 1)
    cc = int(c_cur)
    cells = [(int(rr), int(cc))]
    wraps = 0

    if int(area) == 4:
        assert diag_dc in (-1, 1)
        for _ in range(2):
            rr = int(rr + 1)
            nxt = int(cc + int(diag_dc))
            if not (0 <= nxt < int(width)):
                wraps += 1
            cc = int(nxt % int(width))
            cells.append((int(rr), int(cc)))
        return cells, int(wraps)

    area_to_len = {5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}
    seg_len = int(area_to_len[int(area)])
    dir_h = int(area_to_dir[int(area)])
    for _ in range(seg_len - 1):
        nxt = int(cc + dir_h)
        if not (0 <= nxt < int(width)):
            wraps += 1
        cc = int(nxt % int(width))
        cells.append((int(rr), int(cc)))
    return cells, int(wraps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode tiled 3x3 glyphs into a wrapped polyline on the left canvas.

    Input structure:
    - Left of the full-height divider: monochrome 3x3 glyphs in two fixed tile columns on background.
    - Right of the divider: one gray start pixel whose shifted column seeds the drawing.

    Rule:
    - Remove the divider, shift the gray pixel into the 7-column left canvas, and read glyphs in column-major tile order.
    - For each glyph, first step down one cell, then draw a segment in that glyph's color.
      area 5 means right 3, area 6 means left 4, area 7 means left 2, and area 4 means a descending diagonal of 3 cells.
    - The area-4 diagonal goes down-right for glyphs from the left tile column and down-left for glyphs from the right tile column.
      Any step that leaves the canvas horizontally wraps to the opposite side before continuing.

    Key insight: Glyph area still identifies the segment family, but now the plus glyph also uses its tile-column position to choose the diagonal slope, and every lateral move must be tracked modulo the canvas width.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None
    width = int(div_c)

    out = np.zeros((h, width), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (width + 1))
    if not (0 <= c_cur < width):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, :width]
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        bbox = comp["bbox"]
        if bbox_h(bbox) == 3 and bbox_w(bbox) == 3:
            r0, c0, _r1, _c1 = map(int, bbox)
            glyphs.append({"color": int(color), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        area = int(glyph["area"])
        color = int(glyph["color"])
        diag_dc = _diag_dc_from_glyph_col(int(glyph["c0"]), width) if area == 4 else None
        cells, _wraps = _trace_segment_cells(int(r_cur), int(c_cur), int(area), width=width, diag_dc=diag_dc)

        for rr, cc in cells:
            assert 0 <= int(rr) < int(h), f"segment exits bottom: rr={rr}, H={h}"
            assert 0 <= int(cc) < int(width), f"segment exits canvas: cc={cc}, width={width}"
            out[int(rr), int(cc)] = color

        r_cur, c_cur = map(int, cells[-1])

    return out


# --------------- generator ---------------

N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREAS = (4, 5, 6, 7)

MIN_DIRECTION_CHANGES = 2
MIN_HORIZONTAL_SEGMENTS = 2
MIN_DIAG_SEGMENTS = 1
MIN_WRAP_EVENTS = 1
MIN_DIAG_WRAP_EVENTS = 1
N_ROW_WEIGHTS = (1, 3, 3, 2)
REQUIRE_BOTH_DIAG_DIRECTIONS_MIN_ROWS = 3
MAX_ATTEMPTS = 1200
START_COL_WEIGHTS = (3, 2, 1, 1, 1, 2, 3)


def _direction_changes(areas: list[int]) -> int:
    kinds = ["diag" if int(area) == 4 else "h" for area in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _trace_path_metrics(start_col: int, areas: list[int], n_rows: int) -> dict[str, object]:
    """Simulate the fused wrapped path and collect metrics used for generator validation."""
    h = 4 * int(n_rows) - 1
    r = int(GRAY_START_ROW)
    c = int(start_col)
    cells: list[tuple[int, int, int]] = [(int(r), int(c), GRAY)]
    wrap_events = 0
    diag_wrap_events = 0
    horizontal_wrap_events = 0
    diag_segments = 0
    horizontal_segments = 0

    for idx, area in enumerate(areas):
        diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if int(area) == 4 else None
        seg_cells, seg_wraps = _trace_segment_cells(int(r), int(c), int(area), width=LEFT_W, diag_dc=diag_dc)

        for rr, cc in seg_cells:
            if not (0 <= int(rr) < int(h) and 0 <= int(cc) < LEFT_W):
                raise AssertionError(f"generated segment exits bounds: {(rr, cc)} for H={h}")
            cells.append((int(rr), int(cc), int(_AREA_TO_COLOR[int(area)])))

        wrap_events += int(seg_wraps)
        if int(area) == 4:
            diag_segments += 1
            diag_wrap_events += int(seg_wraps)
        else:
            horizontal_segments += 1
            horizontal_wrap_events += int(seg_wraps)

        r, c = map(int, seg_cells[-1])

    return {
        "cells": cells,
        "end": (int(r), int(c)),
        "wrap_events": int(wrap_events),
        "diag_wrap_events": int(diag_wrap_events),
        "horizontal_wrap_events": int(horizontal_wrap_events),
        "diag_segments": int(diag_segments),
        "horizontal_segments": int(horizontal_segments),
    }


def _choose_preferred_diag_indices(n_rows: int) -> set[int]:
    """Prefer one plus per tile column whenever the row budget can support both directions."""
    first_half = list(range(0, int(n_rows)))
    second_half = list(range(int(n_rows), 2 * int(n_rows)))

    if int(n_rows) >= REQUIRE_BOTH_DIAG_DIRECTIONS_MIN_ROWS:
        return {random.choice(first_half), random.choice(second_half)}
    return {random.choice(first_half if random.random() < 0.5 else second_half)}


def _has_both_diag_directions(areas: list[int], n_rows: int) -> bool:
    left_diag = any(int(area) == 4 for area in areas[: int(n_rows)])
    right_diag = any(int(area) == 4 for area in areas[int(n_rows) :])
    return bool(left_diag and right_diag)


def _sample_path(n_rows: int, start_col: int) -> list[int] | None:
    """Sample 2*N glyph instructions under the fused diagonal-plus and wrap rules."""
    h = 4 * int(n_rows) - 1
    total = 2 * int(n_rows)
    areas: list[int] = []
    preferred_diag_indices = _choose_preferred_diag_indices(int(n_rows))
    r = int(GRAY_START_ROW)
    c = int(start_col)
    diag_count = 0
    wrap_seen = 0
    diag_wrap_seen = 0
    horizontal_seen = 0

    for idx in range(total):
        remaining_after = total - idx - 1
        opts: list[int] = []
        weights: list[int] = []

        for area in _AREAS:
            next_r = int(r + (3 if int(area) == 4 else 1))
            if next_r + remaining_after > h - 1:
                continue

            diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if int(area) == 4 else None
            _cells, seg_wraps = _trace_segment_cells(int(r), int(c), int(area), width=LEFT_W, diag_dc=diag_dc)

            opts.append(int(area))
            weight = 1

            if int(area) == 4:
                weight += 2
                if idx in preferred_diag_indices:
                    weight += 4
                if seg_wraps > 0 and diag_wrap_seen == 0:
                    weight += 9
                elif seg_wraps > 0:
                    weight += 3
                if diag_count == 0:
                    weight += 2
            else:
                weight += 2
                if seg_wraps > 0 and wrap_seen == 0:
                    weight += 3
                if seg_wraps == 0 and horizontal_seen == 0:
                    weight += 2

            if c in (0, 1, LEFT_W - 2, LEFT_W - 1):
                if int(area) == 4 and seg_wraps > 0:
                    weight += 3
                if int(area) != 4 and seg_wraps > 0:
                    weight += 1

            weights.append(int(weight))

        if not opts:
            return None

        area = random.choices(opts, weights=weights, k=1)[0]
        diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if int(area) == 4 else None
        _cells, seg_wraps = _trace_segment_cells(int(r), int(c), int(area), width=LEFT_W, diag_dc=diag_dc)
        areas.append(int(area))

        if int(area) == 4:
            diag_count += 1
            diag_wrap_seen += int(seg_wraps)
        else:
            horizontal_seen += 1
        wrap_seen += int(seg_wraps)
        r += 3 if int(area) == 4 else 1
        c = int(_cells[-1][1])

    metrics = _trace_path_metrics(int(start_col), areas, int(n_rows))
    if _direction_changes(areas) < MIN_DIRECTION_CHANGES:
        return None
    if int(metrics["diag_segments"]) < MIN_DIAG_SEGMENTS:
        return None
    if int(metrics["horizontal_segments"]) < MIN_HORIZONTAL_SEGMENTS:
        return None
    if int(metrics["wrap_events"]) < MIN_WRAP_EVENTS:
        return None
    if int(metrics["diag_wrap_events"]) < MIN_DIAG_WRAP_EVENTS:
        return None
    if int(n_rows) >= REQUIRE_BOTH_DIAG_DIRECTIONS_MIN_ROWS and not _has_both_diag_directions(areas, int(n_rows)):
        return None
    return areas


def _render_grid(n_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    h = 4 * int(n_rows) - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // int(n_rows)]
        r0 = TILE_ROW_STRIDE * (i % int(n_rows))
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choices(range(N_MIN, N_MAX + 1), weights=N_ROW_WEIGHTS, k=1)[0]
    start_col = random.choices(range(LEFT_W), weights=START_COL_WEIGHTS, k=1)[0]
    areas = _sample_path(int(n_rows), int(start_col))
    if areas is None:
        return None
    metrics = _trace_path_metrics(int(start_col), areas, int(n_rows))
    assert int(metrics["diag_segments"]) >= MIN_DIAG_SEGMENTS
    assert int(metrics["horizontal_segments"]) >= MIN_HORIZONTAL_SEGMENTS
    assert int(metrics["wrap_events"]) >= MIN_WRAP_EVENTS
    assert int(metrics["diag_wrap_events"]) >= MIN_DIAG_WRAP_EVENTS
    return _render_grid(int(n_rows), int(start_col), areas)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
