# Variant of e376de54
# Dimension: D10 — 参考线段是否保持不变 (meta)
# Change: 参考线段（第二长线段）本身在输出中保持不变，其他线段调整到它的范围... → 参考线段也被移除（仅保留对齐后的其他线段）

"""
Design analysis for the D10 variant
===================================

variant_spec:
  preserved_structure:
    grid: 16x16, background is the most common color
    objects: 3 or 5 parallel single-color line segments, all on distinct perpendicular offsets
    direction: one shared main direction from horizontal / vertical / the two diagonals
    reference_selection: still the segment at the median perpendicular offset
    anchor_pattern: input segments still share one endpoint along the main direction
  changed_behavior:
    old_output: every segment, including the median reference, is shifted/resized onto the reference extent
    new_output: every non-reference segment is shifted/resized onto the reference extent, but the median
      reference itself is erased and never redrawn
  visual_effect:
    input shows an odd stack of parallel segments with a visible middle line
    output shows the same alignment target, but the middle lane is empty; the missing line is the clue

behavior_trace:
  example:
    direction: horizontal
    input_offsets: [2, 5, 8]
    reference_offset: 5
    reference_extent: columns 4..9
    old_output: rows 2, 5, 8 all become length 6 on columns 4..9
    new_output: rows 2 and 8 become length 6 on columns 4..9, row 5 is background
  difference_point:
    original logic differs only in the redraw stage; reference detection and target extent inference stay identical

complexity_calibration:
  why_not_simpler:
    - Removing the median line from the output is already a visible meta change because one whole lane disappears.
    - Keeping 3 or 5 segments preserves bilateral evidence around the hidden reference and avoids ambiguity.
    - Reusing the original direction set and spacing rules keeps the task recognizable instead of becoming a different puzzle.
  controlled_freedom:
    - Segment count remains in {3, 5}.
    - Reference length remains 3..8.
    - Colors remain 1..3 distinct non-background colors.
    - No color-id-based ordering or transformation is introduced.

implicit_assumption_audit:
  original_lines_that_depended_on_old_d10:
    - The redraw loop used to iterate over every segment and repaint all of them. That implicitly assumed the
      reference survives in the output.
    - Clearing all original segment cells is still safe only because the reference is supposed to disappear too.
  new_solver_assumptions:
    - The input must still contain the reference segment, otherwise the hidden standard cannot be inferred.
    - Segment count must stay odd with unique perpendicular offsets so the median reference is unique.
    - Non-reference segments must fit when aligned to the reference extent; the generator enforces this through
      output-first placement checks before accepting a sample.
    - The output is allowed to contain fewer visible segments than the input; this is intentional, not treated
      as a missing-object error.

placement_legality:
  generator_strategy:
    - Choose direction, odd number of offsets, colors, and a reference extent exactly as before.
    - Construct the intended output first by painting only non-reference segments on the shared target extent.
    - Construct the input by restoring the reference segment and varying the other lengths while keeping a common
      start or end anchor.
    - Accept the sample only if main(input) reproduces the planned output.
  safety:
    - Distinct perpendicular offsets prevent overlap between parallel segments.
    - Bounds checks in _segment_cells and capacity checks ensure all input/output segments remain inside the grid.

visual_observability:
  human_readout:
    - Across examples, the aligned output still points to one common invisible extent.
    - The missing median lane is stable across directions and colors, so the variant is inferable from 2-3 pairs.
    - Because the removed object is defined by position, not by color id, the rule remains visually grounded.

cleanliness_and_color_inference:
  - No extra markers are added, so the scene stays sparse and easy to parse.
  - Colors keep their identity; only geometry changes.
  - Repeated colors across different offsets are still allowed because the crucial cue is the empty median lane.
"""

import random

import numpy as np

from arc_utils import get_bg

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
    Align all non-reference parallel line segments to a shared hidden standard and remove that standard itself.

    Input structure:
    - bg is the most common color.
    - Several multi-cell colored segments lie on distinct parallel lines with one common direction.
    - The middle segment by perpendicular offset is still present in the input and serves as the implicit standard.

    Rule:
    - Detect the dominant direction and extract every multi-cell segment on that direction.
    - Use the segment at the median perpendicular offset as the reference extent.
    - Clear all original segment pixels, then redraw only the non-reference segments so they share the reference's start and length.
    - Leave isolated single pixels unchanged and leave the reference lane empty in the output.

    Key insight: The median segment still defines the target interval, but in this variant it vanishes after every
    other segment has been snapped onto its invisible span.
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
    assert len(s_vals) >= 3, "hidden-reference variant needs visible neighbors around the median"
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

    # Clear only the original segment pixels so the hidden reference can disappear cleanly.
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        if s["s"] == s_ref:
            continue
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
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
    """Find all (t0, anchor_mode, capacity_list) where segments can be placed with room for length variation."""
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


def _choose_colors(s_vals):
    """Assign 1-3 distinct non-bg colors across segments, ensuring all chosen colors appear."""
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
    """Choose input segment lengths: at least one shorter and one longer than ref_len. Returns None if infeasible."""
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable or ref_len <= 2:
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


def _paint(grid, d, s_vals, starts, lengths, colors, keep_mask=None):
    """Render selected segments onto grid. Returns False if any segment cell is OOB."""
    if keep_mask is None:
        keep_mask = [True] * len(s_vals)
    for keep, s, t0, length, color in zip(keep_mask, s_vals, starts, lengths, colors):
        if not keep:
            continue
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
    ref_idx = n // 2
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

    out = np.full((_H, _W), _BG, dtype=int)
    keep_mask = [i != ref_idx for i in range(n)]
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors, keep_mask=keep_mask):
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

    try:
        solved = main(inp)
        if np.array_equal(solved, out):
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
