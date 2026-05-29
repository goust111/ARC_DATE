# Variant of e376de54
# Dimension: D3 — 目标位置：端点对齐方式 (dst)
# Change: 所有线段的两端都对齐到参考线段的起点和终点位置，保持平行间距不变... → 仅对齐终点（尾端对齐）

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D3 variant
# ----------------------------------
# concrete_behavior:
#   objects:
#     - A 16x16 grid with 3 or 5 parallel multi-cell segments.
#     - All segments lie on distinct parallel offsets and share one main direction.
#     - Colors stay as in the seed task: 1-3 non-background colors, no color remapping.
#   variant_effect:
#     - The median-offset segment is still the reference.
#     - The output now aligns only each segment's terminal endpoint to the reference terminal endpoint.
#     - To make this visually observable, each segment keeps its own original length.
#       Otherwise, "same end + same reference length" would also force the same start and collapse
#       back to the seed behavior.
#   traced_example:
#     - Suppose the reference runs from t=8..12 (length 5), and another segment in the input runs
#       from t=8..10 (length 3) because all inputs share the same start.
#     - Seed logic: shift to the reference start and redraw length 5 -> t=8..12.
#     - Variant logic: keep length 3 and align only the end to t=12 -> t=10..12.
#     - Difference: the tail becomes flush while the head stays staggered.
# complexity_calibration:
#   - Direction set, odd segment counts, offset gaps, and color pool stay unchanged to preserve
#     the original puzzle's complexity.
#   - Generator still forces at least one non-reference segment shorter than the reference and one
#     longer than the reference, so the ragged-head / flush-tail output is reliably visible.
# implicit_assumptions_review:
#   - Direction detection via repeated same-color neighbors still works because objects remain
#     straight same-color runs.
#   - Median-by-perpendicular-offset still works because offsets remain unique and the count is odd.
#   - The seed code's use of ref_t0 plus a shared target_len depended on the old "both ends align"
#     property; that assumption is invalid here and is replaced by ref_t1 plus each segment's own len.
# placement_legality:
#   - Output construction chooses one shared end coordinate t1 and places each segment backward from
#     that tail according to its own length.
#   - Input construction keeps a shared start coordinate t0 and places each segment forward from it.
#   - Lengths are chosen under both forward capacity from t0 and backward capacity from t1, so both
#     input and output stay in bounds.
#   - Distinct perpendicular offsets prevent overlap even when colors repeat.
# observability_and_cleanliness:
#   - Inputs are intentionally start-aligned only. If end-aligned inputs were allowed, the new
#     tail-alignment rule could produce identity examples and hide the D3 change.
#   - No extra markers are introduced; the visual cue is purely geometric: uneven heads, neat tails.
# color_change_inference:
#   - None. Colors are preserved exactly, avoiding hidden color-based logic.
# new_solver_assumptions_checked_against_generator:
#   - Every detected segment has length >= 2.
#   - Shifting by end-delta keeps every drawn cell in bounds.
#   - The reference segment itself remains unchanged because its own end already equals ref_t1.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel colored line segments so they all finish at the same endpoint as a reference line.

    Input structure:
    - bg = most common color.
    - Several parallel horizontal, vertical, or diagonal segments appear on distinct offsets.
    - Each segment is monochrome and at least 2 cells long; colors may repeat across offsets.

    Rule:
    - Detect the dominant direction shared by the segments.
    - Pick the segment at the median perpendicular offset as the reference.
    - Move every segment only along the shared direction until its terminal cell matches the reference terminal cell.
    - Keep each segment's original length, so only the tail becomes aligned while the heads remain uneven.

    Key insight: The reference contributes a shared end position, not a shared extent; preserving individual lengths is what makes the tail-only alignment visible.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue  # not a segment start
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
    if delta_t == 2:  # diagonal direction
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t1 = ref["t_end"]

    out = g.copy()

    # Clear only the original segment pixels (leave isolated pixels unchanged).
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        col = s["color"]
        length = s["len"]
        # Shift the segment so its end aligns to ref_t1, while preserving its own length.
        dt = ref_t1 - s["t_end"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        rr, cc = r0, c0
        for _ in range(length):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col
            rr += dr
            cc += dc

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]  # must be odd for unique median
_REF_LEN_RANGE = (3, 8)  # inclusive bounds for reference segment length
_MAX_SEG_LEN = 12  # upper bound for any individual segment

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]  # allowed gaps for orthogonal directions
_PERP_GAP_DIAG = [4, 6]  # allowed gaps for diagonal directions (spec: >= 4)

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3  # upper bound on distinct non-bg colors per puzzle
_SAME_LEN_AVOID_PROB = 0.75  # probability of avoiding ref_len for remaining segments

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)  # range(lo, hi) for feasible t0 search
_GEN_RETRIES = 300  # max attempts before raising


def _step(d):
    """Dot product of direction with itself (1 for ortho, 2 for diag)."""
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d, t, s):
    """Convert (t, s) coordinates back to (row, col) for direction d. Returns None if out of bounds."""
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


def _segment_cells(d, t0, s, length):
    """Return list of (r, c) for a segment starting at t0 with given perpendicular offset s, or None if any cell is OOB."""
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d, t0, s):
    """Max segment length extending forward from t0 at perpendicular offset s."""
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d, t1, s):
    """Max segment length extending backward from t1 at perpendicular offset s."""
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d):
    """Valid range (lo, hi) inclusive for perpendicular offset s given direction d."""
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d, n):
    """Enumerate all valid perpendicular gap sequences of length n-1, ensuring irregular spacing."""
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


def _choose_offsets(d, n):
    """Pick n distinct perpendicular offsets with irregular gaps."""
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, ref_len):
    """Find all (t0, start_caps, end_caps) where a start-aligned input and end-aligned output both fit."""
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        joint_caps = [min(a, b, _MAX_SEG_LEN) for a, b in zip(start_caps, end_caps)]
        if any(joint_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, joint_caps))
    return placements


def _choose_colors(s_vals):
    """Assign 1-3 distinct non-bg colors across segments, ensuring all chosen colors appear."""
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
    """Choose segment lengths preserved from input to output. Returns None if no longer-than-ref segment is possible."""
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, caps[long_i])
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, caps[i] + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint(grid, d, s_vals, starts, lengths, colors):
    """Render segments onto grid. Returns False if any segment cell is OOB."""
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt():
    """Build one candidate (input, output) pair; returns input grid or None on failure."""
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta

    # Build expected output: all segments preserve length but end at the shared reference tail.
    out_starts = [t1 - (length - 1) * delta for length in lengths]
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, out_starts, lengths, colors):
        return None

    # Build input: all segments share the reference start, so their ends are uneven.
    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, [t0] * n, lengths, colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
