# Variant of e376de54
# Dimension: D14 — 轨迹：原始线段位置的处理 (trajectory)
# Change: 无轨迹——线段就地修改，延伸或裁剪后原始多余像素消失... → 被裁剪像素用特殊颜色标记

"""
design_analysis:
  variant_goal:
    keep the original reference-selection and alignment rule unchanged, but make
    shortening visible: any original segment cells that are discarded because the
    segment is longer than the reference remain in place as a red warning trail.
    New cells created by extension do not get a trail.
  a_concretization_and_trace:
    input still contains only parallel monochrome segments on distinct offsets.
    output still places every segment on the reference interval [ref_t0, ref_t1].
    The only new visual element is red cells at the old positions of trimmed tails.
    Example with horizontal shared-start input:
      ref len = 4, long segment = color 5 at columns 3..8.
      Original task output: color-5 segment at 3..6, columns 7..8 become bg.
      Variant output: color-5 segment at 3..6, columns 7..8 stay visible as red.
    For shared-end inputs the same idea applies on the opposite side: the trimmed
    prefix turns red while the kept suffix becomes the aligned segment.
  b_complexity_calibration:
    this is intentionally a single-axis change. We keep:
      same 16x16 canvas, same direction set, same odd segment counts, same median
      reference rule, same alignment target, same length ranges.
    To keep the new behavior observable, generator guarantees at least one segment
    longer than the reference in every sample, so every valid output contains trail.
  c_implicit_assumptions_review:
    original main() cleared every segment cell to bg before redrawing; that relied
    on discarded cells being allowed to vanish. In this variant that assumption is
    false, so clearing must be followed by restoring trimmed-only cells in red.
    The color scan also relied on all non-bg colors meaning "segment colors". That
    remains true for input, so generator must forbid the special trail color in the
    input palette; otherwise humans could not distinguish segment color from trail.
  d_placement_legality:
    all segments remain on distinct parallel offsets, so final segments never
    overlap each other and red trail cells only lie on their own source line.
    Because trail = original_cells - final_cells, trail cells are always in-bounds
    and never overwrite kept cells of the same segment.
  e_visual_observability:
    the warning color is fixed red and absent from inputs. Across 2-4 examples a
    human can see that outputs match the old alignment rule plus red remnants only
    where a longer segment had excess length.
  f_visual_cleanliness:
    the original offset-gap constraints are kept, so red trails remain separated
    from neighboring lines instead of collapsing into merged masses.
  g_color_inference:
    there is no hidden color-dependent mapping. Red always means "trimmed away";
    segment colors themselves remain unchanged.
  new_solver_assumptions:
    input must not already use the trail color as a segment color, and generator
    enforces that by removing red from the palette. The rest of the solver's
    geometric assumptions are unchanged from the source task.
"""

import random

import numpy as np

from arc_utils import Color, get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_TRAIL_COLOR = int(Color.RED)


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
    Snap all parallel colored segments to the reference extent while preserving trimmed-off cells as a red trail.

    Input structure:
    - bg is the most frequent color, and the input contains several multi-cell line segments on distinct parallel offsets.
    - Every segment is monochrome, follows the same detectable direction, and the red trail color is reserved for output-only markings.

    Rule:
    - Detect the shared direction and choose the segment at the median perpendicular offset as the reference.
    - Move each segment so its start matches the reference start, then redraw it with exactly the reference length.
    - If a segment was longer than that target length, keep the discarded original cells visible at their old coordinates using red; isolated single pixels still stay untouched.

    Key insight:
    - The geometry is solved exactly as in the original task; the new information is not a different alignment target but an extra red record of which original cells were removed by shortening.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    assert _TRAIL_COLOR != bg, "trail color must differ from background"

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert _TRAIL_COLOR not in colors, "input must reserve red for output trail"
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
            out[r, c] = bg

    for s in segments:
        col = s["color"]
        start = s["start"]
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = start[0] + k * dr
        c0 = start[1] + k * dc

        final_cells = []
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            final_cells.append((rr, cc))
            rr += dr
            cc += dc
        final_set = set(final_cells)

        for cell in s["cells"]:
            if cell not in final_set:
                out[cell] = _TRAIL_COLOR

        for r, c in final_cells:
            out[r, c] = col

    return out


# --------------- generator (codex-d14) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 3, 5, 6, 8, 9]  # exclude red because red is the trail marker

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
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * _step(d)
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(s_vals):
    """Assign 1-3 distinct non-bg, non-red colors across segments, ensuring all chosen colors appear."""
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


def _paint(grid, d, s_vals, starts, lengths, colors):
    """Render segments onto grid. Returns False if any segment cell is OOB."""
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _build_output(d, s_vals, starts, lengths, colors, ref_t0, ref_len):
    """Render the variant output: aligned segments plus red cells where longer inputs were trimmed."""
    out = np.full((_H, _W), _BG, dtype=int)
    for s, start, length, color in zip(s_vals, starts, lengths, colors):
        original = _segment_cells(d, start, s, length)
        final = _segment_cells(d, ref_t0, s, ref_len)
        if original is None or final is None:
            return None
        final_set = set(final)
        for r, c in original:
            if (r, c) not in final_set:
                out[r, c] = _TRAIL_COLOR
        for r, c in final:
            out[r, c] = color
    return out


def _gen_attempt():
    """Build one candidate input; returns input grid or None on failure."""
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
    if _TRAIL_COLOR in set(map(int, np.unique(inp))):
        return None

    out = _build_output(d, s_vals, starts, lengths, colors, t0, ref_len)
    if out is None:
        return None
    if int(np.count_nonzero(out == _TRAIL_COLOR)) == 0:
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
