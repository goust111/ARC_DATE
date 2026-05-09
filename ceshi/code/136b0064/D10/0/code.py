# Variant of 136b0064
# Dimension: D10 — 折线的边界约束 (meta)
# Change: 折线必须始终保持在左画布范围内（列 0..6，行 0..H-1），超出则生成失败... → 折线可以环绕（toroidal wrap）

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D10 variant
# a) Concrete behavior:
#    - The input layout is unchanged: two left tile columns of 3x3 glyphs, a full-height divider,
#      and one gray start pixel on the right. The output is still the 7-column left canvas.
#    - The only rule change is the handling of horizontal motion. Example on width 7:
#      original area-5 from col 6 would try to paint [6, 7, 8] and fail; now it paints
#      [6, 0, 1] on the same row. Likewise area-6 from col 1 paints [1, 0, 6, 5].
#    - This affects the segment-drawing stage only. Glyph parsing, column-major ordering,
#      color/shape binding, divider removal, and vertical motion all stay the same.
# b) Complexity calibration:
#    - Keep the same four areas and fixed color binding, so the task is still "decode shapes into
#      moves" rather than introducing a second independent codebook.
#    - Make wrap events mandatory in generation and also require at least one non-wrapping
#      horizontal segment, so the new behavior is visible by contrast instead of looking like a
#      lucky edge-touching sample.
# c) Original assumptions that break:
#    - `assert 0 <= end_c < div_c` depended on "horizontal moves never leave the 7-column canvas".
#      That is false now, so horizontal segments must be stepped cell-by-cell with modulo width.
#    - The final-column delta can no longer be checked with a single endpoint inequality because
#      the path may cross the boundary mid-segment even when the wrapped endpoint is in range.
#    - Vertical checks still hold because the variant only wraps in columns, not in rows.
# d) Placement legality:
#    - Glyph slots remain fixed 3x3 boxes at columns 0 and 4 with row stride 4, so components
#      stay disjoint and the solver can still recover them by bbox.
#    - The gray start remains on the right side and shifts into columns 0..6 after divider removal.
#    - The generator simulates the same wrapped walk used by the solver, so every sampled input is
#      oracle-valid before being returned.
# e) Visual observability:
#    - Every generated example has at least one wrap event, which creates a same-color segment
#      split across the left and right edges of one row. That is the visible cue humans can trace.
#    - Samples still contain repeated glyph types, so the solver must infer both the glyph grammar
#      and the new edge behavior from only a few examples.
# f) Visual neatness:
#    - The path remains a single monotone-down polyline. Because each instruction advances at least
#      one row and the longest horizontal span is four cells on a width-7 canvas, wrap can split a
#      row across the edges without causing self-overlap or dense clutter.
# g) Color inference:
#    - No new color logic is introduced. Colors remain tied to glyph shapes exactly as in the base
#      task, so the only new inference burden is the wrapped boundary behavior.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes the left canvas width is the divider column count and uses that width for
#      modulo wrapping; `generate_input()` always keeps the canonical divider at column 7.
#    - `main()` assumes only horizontal segments wrap; the generator preserves the original row
#      budget so every vertical extension still stays within 0..H-1.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode column-major 3x3 glyphs into a colored path drawn on the left canvas.

    Input structure:
    - Left of the full-height divider: fixed 3x3 glyph tiles on background, arranged in two columns.
    - Right of the divider: one gray start pixel whose shifted column seeds the drawing.

    Rule:
    - Remove the divider, shift the gray pixel into the 7-column drawing canvas, and read the glyphs column-major.
    - Area 4 still means a vertical segment of length 2; areas 5, 6, and 7 still mean horizontal segments of lengths 3, 4, and 2 with right/left/left motion.
    - Horizontal motion now wraps around the canvas width, so stepping past one side re-enters from the opposite side on the same row.

    Key insight: The glyph area still encodes the segment instruction, but the path now lives on a cylindrical left canvas, so horizontal steps must be followed modulo the canvas width.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, : int(div_c)]
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": int(col), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}
    width = int(div_c)

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len.get(area, 2))

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(area_to_dir.get(area, 1))
            for _ in range(seg_len - 1):
                c_cur = int((int(c_cur) + dir_h) % width)
                out[r_cur, c_cur] = color

    return out


# --------------- generator (wrapped-path variant) ---------------

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
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)

MIN_DIRECTION_CHANGES = 2
MIN_HORIZONTAL_SEGMENTS = 2
MIN_WRAP_EVENTS = 1
MAX_ATTEMPTS = 300
START_COL_WEIGHTS = (3, 2, 1, 1, 1, 2, 3)


def _segment_result(c0: int, area: int, width: int = LEFT_W) -> tuple[int, int, int]:
    """Return (final_col, wrap_count, wrap_dir_sign) for one decoded segment."""
    if area == 4:
        return int(c0), 0, 0

    cur = int(c0)
    wraps = 0
    dir_h = int(_AREA_TO_DIR[area])
    for _ in range(_AREA_TO_LEN[area] - 1):
        nxt = cur + dir_h
        if not (0 <= nxt < width):
            wraps += 1
        cur = int(nxt % width)
    return int(cur), int(wraps), int(dir_h if wraps else 0)


def _trace_wrapped_path(start_col: int, areas: list[int], H: int, *, width: int = LEFT_W) -> dict[str, object]:
    """Simulate the wrapped polyline and collect metrics used by the generator and checks."""
    r = int(GRAY_START_ROW)
    c = int(start_col)
    cells: list[tuple[int, int, int]] = [(int(r), int(c), GRAY)]
    wrap_events = 0
    wrap_dirs: list[int] = []
    horizontal_segments = 0

    for area in areas:
        seg_len = int(_AREA_TO_LEN[area])
        color = int(_AREA_TO_COLOR[area])

        r += 1
        if r >= H:
            raise AssertionError(f"segment exits bottom: r={r}, H={H}")
        cells.append((int(r), int(c), color))

        if area == 4:
            for _ in range(seg_len - 1):
                r += 1
                if r >= H:
                    raise AssertionError(f"vertical segment exits bottom: r={r}, H={H}")
                cells.append((int(r), int(c), color))
            continue

        horizontal_segments += 1
        dir_h = int(_AREA_TO_DIR[area])
        for _ in range(seg_len - 1):
            nxt = c + dir_h
            if not (0 <= nxt < width):
                wrap_events += 1
                wrap_dirs.append(dir_h)
            c = int(nxt % width)
            cells.append((int(r), int(c), color))

    return {
        "cells": cells,
        "end": (int(r), int(c)),
        "wrap_events": int(wrap_events),
        "wrap_dirs": wrap_dirs,
        "horizontal_segments": int(horizontal_segments),
    }


def _direction_changes(areas: list[int]) -> int:
    types = [area == 4 for area in areas]
    return sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Randomly sample 2*N instructions under wrapped horizontal motion."""
    H = 4 * N - 1
    total = 2 * N
    areas: list[int] = []
    r = int(GRAY_START_ROW)
    c = int(start_col)
    wrap_seen = 0
    nonwrap_horizontal_seen = 0

    for step_idx in range(total):
        remaining_after = total - step_idx - 1
        opts: list[int] = []
        weights: list[int] = []

        for area in _AREAS:
            next_r = r + (2 if area == 4 else 1)
            if next_r + remaining_after > H - 1:
                continue

            next_c, wraps, _wrap_dir = _segment_result(c, area)
            opts.append(area)

            weight = 1
            if area != 4:
                weight += 1
                if wraps and wrap_seen == 0:
                    weight += 5
                elif wraps:
                    weight += 2
                if wraps == 0 and nonwrap_horizontal_seen == 0:
                    weight += 2
                if c in (0, LEFT_W - 1):
                    weight += 1
            else:
                if step_idx not in (0, total - 1):
                    weight += 1

            weights.append(weight)

        if not opts:
            return None

        area = random.choices(opts, weights=weights, k=1)[0]
        areas.append(area)

        if area != 4:
            c, wraps, _wrap_dir = _segment_result(c, area)
            wrap_seen += wraps
            if wraps == 0:
                nonwrap_horizontal_seen += 1
        r += 2 if area == 4 else 1

    metrics = _trace_wrapped_path(start_col, areas, H)
    horizontal_segments = int(metrics["horizontal_segments"])
    wrap_events = int(metrics["wrap_events"])
    if _direction_changes(areas) < MIN_DIRECTION_CHANGES:
        return None
    if horizontal_segments < MIN_HORIZONTAL_SEGMENTS:
        return None
    if wrap_events < MIN_WRAP_EVENTS:
        return None
    if wrap_events >= horizontal_segments:
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.choices(range(LEFT_W), weights=START_COL_WEIGHTS, k=1)[0]
    areas = _sample_path(N, start_col)
    if areas is None:
        return None
    H = 4 * N - 1
    metrics = _trace_wrapped_path(start_col, areas, H)
    assert int(metrics["wrap_events"]) >= MIN_WRAP_EVENTS
    return _render_grid(N, start_col, areas)


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
