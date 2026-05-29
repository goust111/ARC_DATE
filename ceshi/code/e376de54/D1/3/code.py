# Variant of e376de54
# Dimension: D1 — 匹配条件：以第二长线段为参考基准 vs 其他选取规则 (match_cond)
# Change: 以第二长线段的端点范围为参考，所有其他线段的端点对齐到该参考线段的起止位置... → 以出现频率最高的长度为参考

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D1 variant
# ----------------------------------
# concretization_and_trace:
#   visual_form:
#   - The grid still contains several separated monochrome line segments on distinct
#     parallel tracks, all sharing one main direction from {(0,1), (1,0), (1,1), (1,-1)}.
#   - Every segment still shares one anchored endpoint orientation with the others:
#     either all starts coincide in t-space or all ends coincide in t-space.
#   - The variant changes only how the target length is chosen. Instead of reading one
#     special track and copying its extent, the solver must count lengths across all
#     tracks and use the unique modal length.
#   behavior_trace:
#   - Example with 5 horizontal segments and common left endpoint:
#       input lengths by offset order = [6, 4, 8, 4, 7]
#       original code would use the middle-offset segment length 8
#       variant must use the repeated length 4 because it appears twice
#       output lengths become [4, 4, 4, 4, 4] while keeping the common left endpoint fixed
#   - The crucial difference is that the reference is now a length class shared by
#     multiple segments, not a single geometrically privileged segment.
# complexity_calibration:
#   - To make "frequency" visible from only a few examples, the generator uses 5
#     segments only. With 3 segments, the repeated length is too weak a cue and it is
#     impossible to force a reliable separation from a "second-longest" heuristic.
#   - The generator enforces exactly one modal length with frequency 2. The other three
#     lengths are all distinct, with one shorter and two longer than the mode. This
#     keeps the counting signal obvious and prevents ambiguity.
# implicit_assumption_review:
#   - _pick_main_dir still assumes one direction dominates adjacency counts; the spacing
#     constraints remain unchanged so this stays valid.
#   - Segment tracing by color still assumes there is at most one segment per color per
#     parallel track; the generator preserves that structure.
#   - The original solver assumed "median perpendicular offset selects the reference".
#     That assumption is removed. The new solver instead assumes:
#       1. there is a unique most frequent segment length
#       2. all traced segments share either one common start or one common end
#     The generator explicitly constructs both properties.
# placement_legality:
#   - All output segments are placed first at the modal length, fully in bounds.
#   - Input segments are then derived by keeping the shared start or shared end fixed
#     and varying only the lengths. Distinct perpendicular offsets prevent overlap.
# visual_observability_and_cleanliness:
#   - Exactly two tracks already have the target length in the input, so the human
#     solver can see a repeated extent and infer that the odd-length tracks are being
#     snapped to that repeated extent.
#   - The repeated pair never includes the middle-offset segment, so this variant does
#     not silently collapse back to the original median-track behavior.
# colors:
#   - Colors remain passive labels only. No ordering, ranking, or transformation uses
#     color ids, so the rule stays visually grounded in geometry.

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


def _extract_segments(g: np.ndarray, d: tuple[int, int], bg: int) -> list[dict]:
    h, w = g.shape
    dr, dc = d
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
                        "end": end,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize parallel colored segments to the length that appears most often.

    Input structure:
    - bg is the most common color.
    - Several monochrome multi-cell segments lie on distinct parallel tracks and all run in one shared direction.
    - Their lengths differ, but they already share one anchored endpoint orientation: either every segment starts together or every segment ends together.

    Rule:
    - Detect the common direction and trace every multi-cell segment on that axis.
    - Count the segment lengths and pick the unique modal length, meaning the length used by the most segments.
    - Keep the shared anchored endpoint fixed for each segment, then trim or extend every segment so its length becomes that modal length.
    - Leave the canvas size unchanged and do not disturb isolated single cells.

    Key insight: The target extent comes from the repeated length pattern across the whole set, not from a geometrically central track.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    length_counts = Counter(s["len"] for s in segments)
    best_freq = max(length_counts.values())
    assert best_freq > 1, "at least one repeated length is required"
    mode_lens = [length for length, freq in length_counts.items() if freq == best_freq]
    assert len(mode_lens) == 1, "modal length must be unique"
    target_len = mode_lens[0]

    common_starts = {s["t_start"] for s in segments}
    common_ends = {s["t_end"] for s in segments}
    use_start_anchor = len(common_starts) == 1
    use_end_anchor = len(common_ends) == 1
    assert use_start_anchor or use_end_anchor, "segments must share one anchored endpoint"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        col = s["color"]
        if use_start_anchor:
            r0, c0 = s["start"]
        else:
            r0 = s["end"][0] - (target_len - 1) * dr
            c0 = s["end"][1] - (target_len - 1) * dc

        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [5]
_MODE_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3

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
    return [p for p in patterns if sum(p) <= max_span and len(set(p)) > 1]


def _choose_offsets(d, n):
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, mode_len):
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, mode_len) is None for s in s_vals):
            continue
        t1 = t0 + (mode_len - 1) * _step(d)

        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if sum(cap > mode_len for cap in start_caps) >= 2:
            placements.append((t0, "start", start_caps))

        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if sum(cap > mode_len for cap in end_caps) >= 2:
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(s_vals):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(mode_len, caps, n):
    assert n == 5, "generator uses 5 segments to make the frequency cue visible"
    mid = n // 2
    mode_pairs = list(combinations([i for i in range(n) if i != mid], 2))
    random.shuffle(mode_pairs)

    for mode_pair in mode_pairs:
        remaining = [i for i in range(n) if i not in mode_pair]
        longable = [i for i in remaining if caps[i] > mode_len]
        long_pairs = list(combinations(longable, 2))
        random.shuffle(long_pairs)

        for long_pair in long_pairs:
            short_i = [i for i in remaining if i not in long_pair][0]
            long_i, long_j = long_pair

            long_opts_i = list(range(mode_len + 1, min(_MAX_SEG_LEN, caps[long_i]) + 1))
            long_opts_j = list(range(mode_len + 1, min(_MAX_SEG_LEN, caps[long_j]) + 1))
            long_pairs_vals = [(a, b) for a in long_opts_i for b in long_opts_j if a != b]
            if not long_pairs_vals:
                continue

            short_len = random.randint(2, mode_len - 1)
            long_len_i, long_len_j = random.choice(long_pairs_vals)

            lengths = [None] * n
            for idx in mode_pair:
                lengths[idx] = mode_len
            lengths[short_i] = short_len
            lengths[long_i] = long_len_i
            lengths[long_j] = long_len_j

            counts = Counter(lengths)
            if counts[mode_len] != 2:
                continue
            if max(freq for length, freq in counts.items() if length != mode_len) > 1:
                continue
            if lengths[mid] == mode_len:
                continue
            if sorted(lengths)[-2] == mode_len:
                continue
            return lengths

    return None


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
    mode_len = random.randint(*_MODE_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, mode_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(mode_len, caps, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [mode_len] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (mode_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
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
