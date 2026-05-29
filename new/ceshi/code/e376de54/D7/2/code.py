# Variant of e376de54
# Dimension: D7 — 线段颜色的数量与关系 (meta)
# Change: 每条线段有唯一颜色，不同线段颜色互不相同... → 线段颜色在变换后按排名重新映射

"""
Design analysis for the D7 variant
==================================

spec_variant:
  unchanged_geometry:
    grid: 16x16, background 7
    segments:
      count: 3 or 5, all parallel, all multi-cell, odd count for unique median offset
      direction: one of horizontal / vertical / two diagonals
      offsets: distinct, irregularly spaced, diagonal gaps >= 4
      alignment_rule: all output segments still align to the median-offset reference segment
    reference:
      chosen_by: median perpendicular offset, exactly as in the original task
      target_length: reference segment's input length
  d7_change:
    input_colors:
      kept_unique: true
      reason: solver still needs to separate segments by color without adding new object markers
    output_colors:
      source: fixed canonical rank palette [1, 2, 3, 5, 6]
      mapping: longest input segment gets the first rank color, second-longest gets the second, etc.
      ranking_signal: based only on visible input lengths, never on color ids
    tie_policy:
      allowed: false
      reason: ties would make rank-color transfer ambiguous from 2-4 ARC examples

a_concretization_and_trace:
  picture:
    input: several distinct-color line segments with visibly different lengths, one shared direction,
      and one common anchored end as in the original task
    output: the same set of parallel positions snapped to the reference extent, but colors no longer
      match the input identities; instead they encode descending length rank from the input
  traced_example:
    original_logic:
      - lengths = [7, 5, 3] and colors = [8, 2, 9]
      - after alignment, output keeps [8, 2, 9] on those three offsets
    variant_logic:
      - the same lengths imply ranks longest/middle/shortest
      - after alignment, those offsets receive canonical rank colors [1, 2, 3]
    difference:
      - geometry is unchanged
      - only the color-writing stage changes, but it now depends on a second inference layer:
        recover input lengths, rank them, then map ranks to canonical colors

b_complexity_calibration:
  not_too_simple:
    - color change is global and affects every segment, not a single special case
    - output colors are not copied from input, so the task is visibly different from the original
  not_too_free:
    - only 3 or 5 segments are used
    - output rank palette is fixed and short
    - all lengths are unique, removing branching ambiguity

c_implicit_assumption_audit_main:
  old_dependency_1:
    code: segment extraction groups cells by input color
    status_in_variant: still valid because generator keeps input segment colors unique
  old_dependency_2:
    code: painting uses s["color"] directly
    status_in_variant: invalid now; must be replaced by a rank-derived output color
  old_dependency_3:
    code: no need to compare lengths across segments except for reference length
    status_in_variant: invalid now; solver must inspect every input length to build the rank map
  old_dependency_4:
    code: equal lengths are harmless
    status_in_variant: invalid; equal lengths would make rank colors ambiguous, so generator and
      solver both require pairwise distinct lengths

d_placement_legality:
  generator_guards:
    - output is still built first on the aligned reference extent
    - input is derived by preserving one shared endpoint and varying only lengths
    - every segment remains in bounds because capacities are computed in (t, s) coordinates
    - offsets remain unique, so aligned output segments never overlap

e_visual_observability:
  - humans can read the ordering cue directly from the visible input lengths
  - the canonical output colors stay consistent across examples, so the mapping is learnable
  - generator rejects cases where recoloring is too weak, preventing collapse back to the original task

f_visual_cleanliness:
  - no extra markers are introduced
  - each offset contains exactly one segment before and after transformation
  - irregular spacing is preserved, so lines remain visually separable

g_color_change_inferability:
  - the mapping never uses hidden metadata like numeric color ids
  - output colors come from a small fixed palette and correspond to visible rank positions
  - lengths are unique in every sample, so "longest/second/third..." is always well-defined

new_main_assumptions_checked_against_generator:
  - odd number of segments -> enforced by _SEGMENT_COUNTS
  - unique perpendicular offsets -> enforced by _choose_offsets
  - unique input lengths -> enforced by _choose_lengths
  - enough canonical rank colors for the segment count -> max count is 5, palette length is 5
  - input colors unique so color-based tracing remains sound -> enforced by _choose_input_colors
"""

import random
from itertools import combinations

import numpy as np

from arc_utils import get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_RANK_OUTPUT_COLORS = [1, 2, 3, 5, 6]


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


def _rank_color_by_length(segments: list[dict]) -> dict[int, int]:
    lengths = [s["len"] for s in segments]
    assert len(lengths) <= len(_RANK_OUTPUT_COLORS), "rank palette too short"
    assert len(lengths) == len(set(lengths)), "segment lengths must be pairwise distinct"
    ranked_ids = sorted(range(len(segments)), key=lambda i: -segments[i]["len"])
    out = {}
    for rank, seg_idx in enumerate(ranked_ids):
        out[seg_idx] = _RANK_OUTPUT_COLORS[rank]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel line segments to the reference extent, then recolor them by input-length rank.

    Input structure:
    - bg is the most common color.
    - Several multi-cell line segments share one direction and have distinct input colors, but their lengths differ.
    - The median-offset segment is still the geometric reference that determines the common output span.

    Rule:
    - Detect the dominant direction and trace every colored segment along it.
    - Use the segment at the median perpendicular offset as the reference, then shift all segments so their starts match it and redraw them at the reference length.
    - Do not preserve original segment colors in the output: assign a fixed canonical color sequence by descending input length rank (longest gets the first rank color, next-longest the second, etc.).

    Key insight: the geometry still comes from the median-offset reference segment, but the output colors come from a separate ranking of the INPUT lengths, so solving requires combining alignment and color remapping.
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

    rank_colors = _rank_color_by_length(segments)
    out = g.copy()

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for seg_idx, s in enumerate(segments):
        start = s["start"]
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = start[0] + k * dr
        c0 = start[1] + k * dc
        rr, cc = r0, c0
        col = rank_colors[seg_idx]
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col
            rr += dr
            cc += dc

    return out


# --------------- generator (output-first ranked recolor variant) ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_INPUT_COLOR_POOL = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = {3: (3, 8), 5: (4, 7)}
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


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


def _assign_distinct_lengths(ref_len, caps, role_map):
    order = sorted(role_map, key=lambda i: caps[i])
    options = {}
    for idx, role in role_map.items():
        if role == "short":
            hi = min(ref_len - 1, min(_MAX_SEG_LEN, caps[idx]))
            options[idx] = list(range(2, hi + 1))
        else:
            hi = min(_MAX_SEG_LEN, caps[idx])
            options[idx] = list(range(ref_len + 1, hi + 1))
        if not options[idx]:
            return None
        random.shuffle(options[idx])

    picked = {}
    used = {ref_len}

    def backtrack(pos):
        if pos == len(order):
            return True
        idx = order[pos]
        for value in options[idx]:
            if value in used:
                continue
            used.add(value)
            picked[idx] = value
            if backtrack(pos + 1):
                return True
            used.remove(value)
            del picked[idx]
        return False

    if not backtrack(0):
        return None
    return picked


def _choose_lengths(ref_len, caps, n):
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    role_maps = []

    if n == 3:
        for long_i in nonrefs:
            short_i = [i for i in nonrefs if i != long_i][0]
            role_maps.append({short_i: "short", long_i: "long"})
    else:
        for long_pair in combinations(nonrefs, 2):
            role_map = {}
            for idx in nonrefs:
                role_map[idx] = "long" if idx in long_pair else "short"
            role_maps.append(role_map)

    random.shuffle(role_maps)
    for role_map in role_maps:
        picked = _assign_distinct_lengths(ref_len, caps, role_map)
        if picked is None:
            continue
        for idx, value in picked.items():
            lengths[idx] = value
        if len(set(lengths)) == n:
            return lengths
    return None


def _rank_colors_for_lengths(lengths):
    assert len(lengths) == len(set(lengths)), "length ranks must be unique"
    n = len(lengths)
    assert n <= len(_RANK_OUTPUT_COLORS)
    order = sorted(range(n), key=lambda i: -lengths[i])
    colors = [None] * n
    for rank, idx in enumerate(order):
        colors[idx] = _RANK_OUTPUT_COLORS[rank]
    return tuple(colors)


def _choose_input_colors(output_colors):
    n = len(output_colors)
    target_mismatches = max(2, n // 2)
    out_set = set(output_colors)
    for _ in range(100):
        colors = random.sample(_INPUT_COLOR_POOL, n)
        random.shuffle(colors)
        mismatch_count = sum(c != oc for c, oc in zip(colors, output_colors))
        has_outside_color = any(c not in out_set for c in colors)
        if mismatch_count >= target_mismatches and has_outside_color:
            return tuple(colors)
    raise RuntimeError("failed to choose sufficiently distinct input colors")


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
    ref_len = random.randint(*_REF_LEN_RANGE[n])
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    output_colors = _rank_colors_for_lengths(lengths)
    input_colors = _choose_input_colors(output_colors)

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, output_colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, input_colors):
        return None

    if sum(ic != oc for ic, oc in zip(input_colors, output_colors)) < max(2, n // 2):
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
