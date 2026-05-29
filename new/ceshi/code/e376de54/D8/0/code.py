# Variant of e376de54
# Dimension: D8 — 网格形状：输出尺寸 (grid_shape)
# Change: 输出网格与输入尺寸完全相同（same）... → 紧凑裁剪到线段区域

import random

import numpy as np

from arc_utils import get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D8 tight-crop variant
#
# a) Concrete behavior and flow impact
#    The executable seed does not actually use "second-longest"; it finds the
#    segment whose perpendicular offset is the median and uses that line as the
#    reference. This D8 variant keeps that geometric rule unchanged. The only
#    visible difference is the last stage: after all segments are shifted so
#    their starts align with the reference and all are redrawn at the reference
#    length, the solver returns only the minimal rectangle covering those
#    transformed segments.
#
#    Small trace:
#    - Original: three horizontal segments end up occupying rows 3, 6, 10 and
#      cols 4..8, but the output remains a 16x16 canvas with background around
#      them.
#    - Variant: the same transformed segments are first produced on the full
#      canvas, then cropped to rows 3..10 and cols 4..8, yielding an 8x5 grid.
#    - Difference: the alignment logic is identical; only the final framing is
#      changed.
#
# b) Complexity calibration
#    The crop should be easy to observe but not trivial. If the aligned segment
#    union touched the border, the output could accidentally keep a full row or
#    column span and the D8 change would be less visible. The generator
#    therefore requires at least a 1-cell margin around the transformed segment
#    bbox in the 16x16 source canvas. This keeps the task visually distinct
#    from the original same-size output while preserving the original object
#    family and reference-selection logic.
#
# c) Implicit assumptions audit
#    Relevant original assumptions:
#    - Direction detection assumes the input still contains several multi-cell
#      same-direction segments. This remains true.
#    - Segment extraction assumes each logical line has a unique perpendicular
#      offset. This remains true and prevents overlaps after shifting.
#    - `out = g.copy()` was harmless because the original output kept the full
#      canvas. Under tight crop, pixels outside the transformed segment bbox are
#      intentionally discarded. Therefore cropping must be based on the drawn
#      transformed segment cells, not on all old foreground cells.
#    - Clearing the old segment pixels before redrawing remains necessary;
#      otherwise stale tails from longer input segments would survive inside the
#      cropped area.
#
# d) Placement legality
#    The generator still works output-first. It first chooses the aligned
#    output-state segments, but now only keeps placements whose transformed bbox
#    stays strictly inside the canvas. Input segments are then derived by
#    varying lengths while sharing either the start or the end anchor, exactly
#    as before. Segment cells are always checked for in-bounds placement.
#
# e) Visual observability
#    Across 2-4 examples, a human can see two coupled effects:
#    - all parallel segments are snapped to the same extent as before;
#    - the output frame hugs the transformed segment cluster instead of showing
#      the original 16x16 board.
#    No color-based ordering or hidden metadata is introduced.
#
# f) Visual cleanliness
#    Only parallel line segments are present. Distinct perpendicular offsets and
#    enforced gaps keep them separate, so the crop yields a clean stripe-like
#    composition rather than a cluttered overlap.
#
# g) Color inference
#    Colors are unchanged by the transform. The variant does not add any new
#    color mapping problem; color remains a stable identity for each segment.
# ---------------------------------------------------------------------------

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


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel colored line segments to a shared reference extent, then return only their tight bounding box.

    Input structure:
    - `bg` is the most frequent color.
    - The foreground contains several multi-cell line segments in one dominant direction; segments lie on distinct parallel tracks and may have different lengths.

    Rule:
    - Detect the common direction, extract each multi-cell segment, and choose the segment at the median perpendicular offset as the reference.
    - Shift every segment along that direction so its start matches the reference start, then redraw each one with exactly the reference length.
    - After the redraw, crop away all surrounding background so the output is the minimal rectangle containing the transformed segments.

    Key insight:
    The geometric alignment rule is unchanged from the base task; the extra step is that the answer is framed by the transformed segment cluster rather than by the original full canvas.
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

    transformed_cells = []
    for s in segments:
        col = s["color"]
        start = s["start"]
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = start[0] + k * dr
        c0 = start[1] + k * dc
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col
            transformed_cells.append((rr, cc))
            rr += dr
            cc += dc

    assert transformed_cells, "transformed output must contain segment pixels"
    r0, c0, r1, c1 = _bbox_from_cells(transformed_cells)
    return out[r0 : r1 + 1, c0 : c1 + 1]


# --------------- generator (output-first, tight-crop aware) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

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

# --- Tight-crop constraints ---
_OUTER_MARGIN = 1

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


def _step(d):
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d, t, s):
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
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d, t0, s):
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d, t1, s):
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d):
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d, n):
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
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _all_cells_for_layout(d, s_vals, starts, lengths):
    cells = []
    for s, t0, length in zip(s_vals, starts, lengths):
        seg_cells = _segment_cells(d, t0, s, length)
        if seg_cells is None:
            return None
        cells.extend(seg_cells)
    return cells


def _layout_bbox(d, s_vals, starts, lengths):
    cells = _all_cells_for_layout(d, s_vals, starts, lengths)
    if cells is None:
        return None
    return _bbox_from_cells(cells)


def _has_outer_margin(d, s_vals, t0, ref_len):
    bbox = _layout_bbox(d, s_vals, [t0] * len(s_vals), [ref_len] * len(s_vals))
    if bbox is None:
        return False
    r0, c0, r1, c1 = bbox
    return (
        r0 >= _OUTER_MARGIN
        and c0 >= _OUTER_MARGIN
        and r1 <= _H - 1 - _OUTER_MARGIN
        and c1 <= _W - 1 - _OUTER_MARGIN
    )


def _feasible_placements(d, s_vals, ref_len):
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        if not _has_outer_margin(d, s_vals, t0, ref_len):
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
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
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
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _crop_to_cells(grid, cells):
    r0, c0, r1, c1 = _bbox_from_cells(cells)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _gen_attempt():
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

    out_full = np.full((_H, _W), _BG, dtype=int)
    aligned_cells = _all_cells_for_layout(d, s_vals, [t0] * n, [ref_len] * n)
    if aligned_cells is None:
        return None
    if not _paint(out_full, d, s_vals, [t0] * n, [ref_len] * n, colors):
        return None
    out = _crop_to_cells(out_full, aligned_cells)
    if out.shape == (_H, _W):
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
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
