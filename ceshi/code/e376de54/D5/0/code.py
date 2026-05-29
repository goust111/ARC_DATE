# Variant of e376de54
# Dimension: D5 — 长度比较的排序规则 (meta)
# Change: 按线段像素数从长到短排序，选取第二长（rank=2）作为参考... → 第三长线段为参考
"""
Design analysis for the D5 variant
==================================

variant_spec:
  invariant_structure:
    - 16x16 grid, background is the most frequent color.
    - Several parallel multi-cell segments share one of four directions:
      horizontal, vertical, diagonal \\, diagonal /.
    - Segments stay on distinct parallel lines (distinct perpendicular offsets),
      use 1-3 non-background colors, and may still share one anchored endpoint
      along the main direction in the input.
    - The output still aligns every segment to one reference segment's extent:
      same start coordinate along the main direction and same final length.
  core_change:
    - The reference is no longer tied to positional order.
    - Sort all detected segment lengths from longest to shortest and use the
      third entry (rank=3) as the target length/reference segment.
    - Because "third longest" must be visually and logically unambiguous, the
      generator enforces pairwise-distinct segment lengths.

  concrete_behavior_trace:
    - Example with five horizontal segments of lengths [10, 8, 6, 4, 2] and a
      shared start column.
    - Old rank-2 behavior would choose length 8 and extend/crop every segment
      to length 8.
    - New behavior chooses length 6 instead, so the 10/8 segments shrink to 6
      while the 4/2 segments extend to 6.
    - The only visible cue that determines the target is counting the ordered
      segment lengths; color ids and vertical placement do not matter.

  complexity_calibration:
    - Keep segment counts at 3 or 5 so the scene is still sparse and readable.
    - Preserve the original gap rules so adjacent parallel lines do not merge
      visually or create ambiguous dominant directions.
    - Lengths are distinct and the reference offset is randomized independently
      of spatial order; this prevents accidental fallback to "middle line" or
      "same color" heuristics.

  implicit_assumption_audit:
    - _pick_main_dir still assumes one direction yields the unique maximum count
      of same-color adjacencies; generator verification keeps this true.
    - Segment extraction assumes each color-line pair forms one contiguous run
      along the main direction; generator paints exactly one run per offset.
    - Ranking by length introduces a new assumption: the third-longest length
      must be unique. The solver asserts unique lengths and the generator
      constructs them that way.
    - Redrawing assumes shifting each segment to the reference start remains
      in bounds. The output-first generator chooses a legal output placement and
      derives input from it, so the solver's redraw is guaranteed to fit.

  placement_legality:
    - Output is built first with all segments already aligned at the chosen
      reference start/end, guaranteeing a valid target state.
    - Input is then derived by keeping either the shared start or shared end
      fixed and assigning distinct lengths around the reference length.
    - For rank=3, generation requires at least two segments that can be longer
      than the reference; for 5 segments it also requires two distinct shorter
      lengths.

  visual_observability:
    - The rank change is visible because the generator forbids the second- and
      third-longest lengths from matching.
    - The reference line is not coupled to median offset, median color, or any
      hidden metadata; humans must inspect the length ordering itself.
    - With 3 segments, the third-longest is the shortest; with 5 segments, it
      is the middle-by-length segment. Both cases remain inferable from a small
      number of examples because the output always snaps to one clearly counted
      input length.
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


def _extract_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
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
    Align parallel colored segments so they all inherit the same span as the third-longest segment.

    Input structure:
    - The background is the most frequent color.
    - Several parallel multi-cell segments appear in one shared direction, each living on a distinct parallel line.
    - Segment lengths differ, and isolated single pixels are irrelevant noise that should remain untouched.

    Rule:
    - Detect the dominant segment direction and extract every contiguous segment of length at least 2.
    - Order those segments by pixel count from longest to shortest, then use the third item as the reference segment.
    - Move every segment along the common direction so its start matches the reference start, and redraw it with the reference length.
    - Keep the grid size unchanged and leave any non-segment singleton pixels alone.

    Key insight: spatial order does not choose the target anymore; the decisive cue is the third place in the length ranking.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc
    segments = _extract_segments(g, bg, d)

    assert len(segments) >= 3, "at least three multi-cell segments are required"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    lengths_desc = sorted((s["len"] for s in segments), reverse=True)
    assert len(lengths_desc) == len(set(lengths_desc)), "segment lengths must be unique"
    target_len = lengths_desc[2]
    ref_cands = [s for s in segments if s["len"] == target_len]
    assert len(ref_cands) == 1, "third-longest reference segment must be unique"
    ref = ref_cands[0]

    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
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
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE_BY_COUNT = {
    3: (2, 8),
    5: (4, 8),
}
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3

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


def _feasible_placements(d, s_vals, ref_len):
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta

        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if sum(cap > ref_len for cap in start_caps) >= 2:
            placements.append((t0, "start", start_caps))

        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if sum(cap > ref_len for cap in end_caps) >= 2:
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps):
    n = len(caps)
    ref_idx = random.randrange(n)
    candidate_long_idxs = [i for i in range(n) if i != ref_idx and caps[i] > ref_len]
    if len(candidate_long_idxs) < 2:
        return None

    long_idxs = random.sample(candidate_long_idxs, 2)
    remaining = [i for i in range(n) if i not in long_idxs and i != ref_idx]

    long_options = [
        list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[i]) + 1))
        for i in long_idxs
    ]
    long_pairs = [
        (a, b)
        for a in long_options[0]
        for b in long_options[1]
        if a != b
    ]
    if not long_pairs:
        return None

    if len(remaining) > 0 and ref_len < 2 + len(remaining):
        return None

    lengths = [None] * n
    lengths[ref_idx] = ref_len
    long_vals = list(random.choice(long_pairs))
    random.shuffle(long_vals)
    for idx, value in zip(long_idxs, long_vals):
        lengths[idx] = value

    if remaining:
        short_vals = random.sample(list(range(2, ref_len)), len(remaining))
        random.shuffle(short_vals)
        for idx, value in zip(remaining, short_vals):
            lengths[idx] = value

    assert all(v is not None for v in lengths)
    assert len(set(lengths)) == n
    assert sorted(lengths, reverse=True)[2] == ref_len
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
    ref_len = random.randint(*_REF_LEN_RANGE_BY_COUNT[n])
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    lengths = _choose_lengths(ref_len, caps)
    if lengths is None:
        return None
    colors = _choose_colors(n)

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
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
