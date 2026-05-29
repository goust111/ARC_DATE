# Variant of e376de54
# Dimension: D14 — 轨迹：原始线段位置的处理 (trajectory)
# Change: 无轨迹——线段就地修改，延伸或裁剪后原始多余像素消失... → 保留原始线段轮廓

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg

# Design analysis for the D14 keep_source variant
# -----------------------------------------------
# variant_spec:
#   a_concretization_and_trace:
#     input:
#       - Same 16x16 canvas and same object family as the source task:
#         3 or 5 parallel line segments, all on distinct offsets and all sharing one anchor
#         (same start or same end along the dominant direction).
#       - Input still contains only the original colored segments.
#     output:
#       - Every original segment remains visible as a pale "source trail".
#       - The transformed segment is drawn in the original segment color on top of that trail.
#       - Overlap cells show the original strong color; only source-only leftovers become pale.
#     trace_example:
#       - Source task, horizontal, ref_len=5, shared start:
#         red length 7 at row 3 becomes length 5; the rightmost 2 red cells disappear.
#       - This variant:
#         those 7 original red cells are first recolored to gray, then the leftmost 5 cells
#         are recolored back to red. Result: a red 5-cell segment plus a 2-cell gray tail.
#       - For a short segment length 3, the 3 source cells become gray first and then red again,
#         while the newly extended 2 cells appear only in red.
#   b_complexity_calibration:
#     - The geometric rule is intentionally unchanged: same dominant direction detection,
#       same median-offset reference, same length/shift normalization.
#     - Only the rendering stage changes. This keeps the task recognizably close to e376de54
#       while making the new trajectory dimension plainly visible.
#     - To avoid an invisible trail, the generator reserves gray (5) for the source outline and
#       never uses 5 as an original segment color.
#   c_implicit_assumption_audit:
#     - Segment discovery by monochrome runs is still valid because input structure is unchanged.
#     - Reference selection by median perpendicular offset is still valid because offsets remain
#       unique and odd in count.
#     - The original clearing loop `out[r, c] = bg` depended on the old assumption that the
#       source pixels should vanish. That is no longer true; it must become `out[r, c] = trail`.
#     - The final redraw loop remains correct because same-offset segments never intersect other
#       segments, so drawing transformed colors after the trail cleanly produces the overlay.
#     - New assumption introduced by this variant: the trail color should be distinct from the
#       background and preferably from segment colors. The generator guarantees this by reserving 5.
#   d_placement_legality:
#     - Generator keeps the original output-first search so every transformed target segment stays
#       in bounds before the input is derived.
#     - Distinct perpendicular offsets still prevent inter-segment overlap in both input and output.
#     - Because one non-reference segment is longer and one is shorter than the reference, each
#       generated sample contains both a visible gray leftover and a visible colored extension.
#   e_visual_observability:
#     - The human-visible rule becomes: "the old line is kept faintly, and the corrected line is
#       drawn over it." This is inferable from 2-3 examples because the pale leftovers always sit
#       exactly where trimmed parts used to be, while extensions appear only in the vivid color.
#     - No invisible metadata is used for sorting or recoloring; only spatial median and line shape
#       control the transformation.
#   f_visual_cleanliness:
#     - Only two layers ever occupy one offset: source trail and transformed segment of the same
#       object. Different segments remain separated by the original spacing constraints, so the
#       output stays legible instead of forming mixed-color clutter.
#   g_color_inference:
#     - There is a single dedicated trail color across generated samples, reducing color ambiguity.
#       The original segment colors remain unchanged.
#   main_assumptions_checked_against_generator:
#     - `_pick_main_dir` assumes the segment direction is uniquely dominant; generator still builds
#       only straight parallel runs in one shared direction.
#     - Segment tracing assumes each multi-cell object is a contiguous monochrome run; generator
#       paints only such runs and does not add isolated distractors.
#     - Median reference selection assumes the number of segments is odd and offsets are unique;
#       generator still uses counts 3 or 5 with irregular distinct gaps.
#     - Recolor-then-redraw assumes target cells stay in bounds; generator retains the same
#       feasibility search used by the original task before accepting a sample.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_PREFERRED_TRAIL_COLORS = [5, 8, 4, 6, 1, 2, 3, 9, 0]
_GEN_TRAIL_COLOR = 5


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[int, int]:
    counts = []
    h, w = g.shape
    for dr, dc in _DIR_CANDIDATES:
        cnt = 0
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == bg:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                    cnt += 1
        counts.append(((dr, dc), cnt))

    best_cnt = max(cnt for _, cnt in counts)
    assert best_cnt > 0, "main direction must be detectable"
    best_dirs = [d for d, cnt in counts if cnt == best_cnt]
    assert len(best_dirs) == 1, "main direction must be unique"
    return best_dirs[0]


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
    return p[0] * d[1] - p[1] * d[0]


def _choose_trail_color(bg: int, colors: list[int]) -> int:
    for col in _PREFERRED_TRAIL_COLORS:
        if col != bg and col not in colors:
            return col
    for col in range(10):
        if col != bg and col not in colors:
            return col
    raise AssertionError("need a spare color for the source trail")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize parallel colored line segments to a reference extent while keeping the original lines visible underneath.

    Input structure:
    - bg = most common color.
    - Several parallel single-color line segments share one direction but can start/end at different positions along it.
    - Each multi-cell segment sits on its own perpendicular offset; isolated single pixels, if present, are not part of the transformation.

    Rule:
    - Detect the shared dominant direction and identify the segment at the median perpendicular offset as the reference.
    - Shift every segment so its start matches the reference start, then trim or extend it to the reference length.
    - Before drawing the normalized segment, keep the original segment as a pale trail; then paint the normalized segment in its original color on top.

    Key insight: The geometry rule is unchanged from the base task, but the output is now a before/after overlay: old positions remain as faint traces and corrected positions overwrite them.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    trail_color = _choose_trail_color(bg, colors)

    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cells) < 2:
                    continue
                start = cells[0]
                end = cells[-1]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "start": start,
                        "end": end,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )

    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = trail_color

    for s in segments:
        col = s["color"]
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col
            rr += dr
            cc += dc

    return out


# --------------- generator (o5x, D14 keep_source) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 4, 6, 8, 9]  # reserve 5 for the kept source trail

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
    if d == (0, 1):
        r, c = s, t
    elif d == (1, 0):
        r, c = t, -s
    elif d == (1, 1):
        if (t + s) % 2 or (t - s) % 2:
            return None
        r, c = (t + s) // 2, (t - s) // 2
    else:
        if (t - s) % 2 or (-s - t) % 2:
            return None
        r, c = (t - s) // 2, (-s - t) // 2
    if 0 <= r < _H and 0 <= c < _W:
        return r, c
    return None


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d: tuple[int, int], n: int) -> list[list[int]]:
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [g] for p in patterns for g in allowed]
    return [
        p
        for p in patterns
        if sum(p) <= max_span and (n == 3 and p[0] != p[1] or n == 5 and len(set(p)) > 1)
    ]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: list[int] | tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, starts, lengths, [_GEN_TRAIL_COLOR] * n):
        return None
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
