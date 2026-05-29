# Variant of e376de54
# Dimension: D4 — 变换：延伸或裁剪线段 (transform)
# Change: 短于参考的线段延伸像素到参考长度，长于参考的线段裁剪多余像素... → 仅裁剪不延伸（长线段裁剪，短线段保持不变）

from pathlib import Path
import sys
import random

import numpy as np

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        _solvers_dir = str(_parent / "solvers")
        if _solvers_dir not in sys.path:
            sys.path.insert(0, _solvers_dir)
        break

from arc_utils import get_bg


# Design analysis for the D4 variant
# a) Concrete behavior and trace:
#    The input still contains 3 or 5 parallel solid segments with one shared direction and
#    distinct perpendicular offsets. The reference is still the segment at the median offset.
#    Original behavior on an end-anchored example:
#      ref length = 5, another segment length = 3, another = 7.
#      After start-alignment, the 3-cell segment is extended to 5 and the 7-cell segment is cut to 5.
#    Variant behavior:
#      After the same start-alignment, the 7-cell segment is cut to 5, but the 3-cell segment
#      stays 3. The visual difference is that all starts coincide, while only long segments
#      reach the reference end.
# b) Complexity calibration:
#    Keep the original odd segment count, irregular offset gaps, 1-3 colors, and four possible
#    directions so the task does not collapse into a trivial single-case pattern. The generator
#    forces at least one shorter and one longer non-reference segment, so the one-sided rule is
#    visible in every valid sample.
# c) Implicit assumptions in the original solver:
#    - Direction detection assumes multi-cell adjacency dominates one unique direction: unchanged.
#    - Segment extraction assumes each same-color run on a given offset is contiguous: unchanged.
#    - Median-by-offset reference selection assumes an odd number of distinct offsets: unchanged.
#    - The original redraw loop assumed every segment should be drawn with target_len. That is the
#      line that becomes invalid under the variant; we must replace it with min(original_len, target_len).
# d) Placement legality:
#    Output segments all start at the same t0 but can have mixed lengths. Because perpendicular
#    offsets remain distinct, aligned starts do not create overlaps. For input generation, the
#    shared-start or shared-end anchor is preserved, lengths stay within bounds, and feasibility
#    is checked before painting.
# e) Visual observability:
#    Each sample contains both a cropped segment and a preserved-short segment. This guarantees
#    that the variant output differs from the original task output on the same input, and that
#    the difference is visible without using hidden metadata such as color ids.
# f) Visual cleanliness:
#    The original spacing constraints are preserved, including larger diagonal gaps, so parallel
#    lines remain separated after alignment. No extra markers or overlapping structures are added.
# g) Color inference:
#    Colors remain identity-preserving. No color-dependent ordering or recoloring is introduced,
#    so the human-visible rule is purely geometric.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]


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
    return p[0] * d[1] - p[1] * d[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-anchor parallel colored line segments to the reference start, trimming only the ones that exceed the reference length.

    Input structure:
    - The background is the most frequent color.
    - Several horizontal, vertical, or diagonal solid segments lie on distinct parallel lines with one shared direction.
    - Those segments can be shorter than, equal to, or longer than the middle segment selected by perpendicular position.

    Rule:
    - Detect the dominant direction and extract every multi-cell segment along it.
    - Use the segment at the median perpendicular offset as the reference, then shift every segment so its start matches the reference start.
    - If a shifted segment is longer than the reference, crop it down to the reference length; otherwise keep its original length.
    - Keep the canvas size unchanged, and leave isolated single pixels alone.

    Key insight: Alignment is still dictated by the median-offset reference line, but length normalization is now one-sided: remove excess cells from long segments and never pad short ones.
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
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        draw_len = min(s["len"], target_len)
        rr, cc = r0, c0
        for _ in range(draw_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


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


def _feasible_placements(d, s_vals, ref_len):
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


def _paint(grid, d, s_vals, starts, lengths, colors):
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


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

    out_lengths = [min(length, ref_len) for length in lengths]
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, out_lengths, colors):
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

    if not any(length > ref_len for length in lengths):
        return None
    if not any(length < ref_len for length in lengths):
        return None

    return inp


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
