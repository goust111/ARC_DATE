# Variant of e376de54
# Dimension: D1 — 匹配条件：以第二长线段为参考基准 vs 其他选取规则 (match_cond)
# Change: 以第二长线段的端点范围为参考，所有其他线段的端点对齐到该参考线段的起止位置... → 以中位数长度线段为参考基准

"""
Design analysis for the D1 variant
==================================

spec_like_summary:
  grid: 16x16, background 7
  objects:
    - 5 parallel line segments, each at a distinct perpendicular offset
    - shared direction is one of horizontal / vertical / diagonal / anti-diagonal
    - every segment is a solid 1-cell-thick run of length >= 2
    - all segments share one anchored endpoint in the input: either same start or same end
    - segment lengths are all distinct, with exactly two shorter and two longer than the reference length
  transformation:
    - choose the reference by segment length order, not by spatial order
    - sort lengths ascending and take the low median:
      odd count -> middle value
      even count -> shorter of the two middle values
    - use that segment's start/end span as the target span
    - slide every segment along the common direction so its start matches the reference start
    - redraw every segment at the reference length

a) concrete behavior trace:
  - Visible scene: five colored parallel segments stacked at different offsets. They clearly have
    different lengths but still share one endpoint, so a human can compare lengths directly.
  - Example: five horizontal segments have lengths [2, 4, 6, 8, 10] and all end at column 12.
    If the 6-cell segment sits on the top line and the 8-cell segment sits in the middle line,
    old "second-longest" logic would copy the 8-cell span, and old provided code would copy the
    middle-offset line's span. The new variant must instead copy the 6-cell segment's span.
    Therefore both the chosen length and often the chosen offset line change.

b) complexity calibration:
  - Keeping 5 segments is deliberate. With 3 segments, median length equals second-longest, so
    the new rule would visually collapse back to the old one. Fixing count=5 makes the D1 change
    observable in every generated sample without adding unrelated object types.
  - Lengths are all distinct so the reference is unique; otherwise a tie at the median length
    would make the endpoint target ambiguous.

c) implicit-assumption audit for main():
  - Direction detection still assumes one direction is uniquely dominant; generator keeps all
    multi-cell objects parallel and separated so this remains true.
  - Segment extraction by color/start-cell scanning assumes each segment is a straight contiguous
    run. Generator still constructs exactly that.
  - The original reference-selection block assumed uniqueness came from perpendicular position.
    That assumption is invalid here and is replaced with uniqueness of the median length instead.
  - Redrawing still assumes shifting by whole direction steps keeps cells on-grid. Generator
    preserves the original output-first placement strategy, so the chosen reference span is known
    to fit for every segment.

d) placement legality:
  - Perpendicular offsets are unique and spaced apart, so segments never overlap.
  - Output is built first with a legal shared span; input is then derived by shortening or
    lengthening from a common anchor, which preserves non-overlap and in-bounds placement.
  - Two non-reference segments are forced longer and two shorter, so the length median is
    well-supported by visible evidence.

e) visual observability:
  - The reference line is not allowed to sit at the median perpendicular offset. This prevents the
    variant from accidentally looking like the provided median-offset code.
  - No invisible property such as color id is used for sorting or tie-breaking; only visible run
    length determines the reference.

f) visual cleanliness:
  - The scene contains only separated solid segments on a blank background.
  - Gaps between offsets remain irregular but generous, which keeps each line easy to count.

g) color-change inference:
  - There is no color remapping in this variant. Colors only label segments, so all reasoning
    pressure stays on geometric length order.
"""

import itertools
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
    return p[0] * d[1] - p[1] * d[0]


def _low_median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize a family of parallel line segments to the span of the median-length segment.

    Input structure:
    - Background is the most common color.
    - The foreground contains several straight, parallel, multi-cell segments in one shared direction.
    - Segments occupy distinct parallel lines, may reuse colors, and can begin or end at different positions.

    Rule:
    - Detect the common direction and extract every contiguous segment of length at least 2.
    - Order the segment lengths and choose the low median as the reference length
      (middle length for odd counts, shorter middle for even counts).
    - Use the unique segment with that length as the reference span, then slide every segment so its start
      aligns with the reference start and redraw it at the reference length.
    - Keep the canvas size unchanged and leave isolated single pixels untouched.

    Key insight:
    The deciding feature is the visible order statistic of segment lengths, not spatial position:
    the reference is the median-length segment, and every other segment is cropped or extended to its extent.
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
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    target_len = _low_median([s["len"] for s in segments])
    ref_cands = [s for s in segments if s["len"] == target_len]
    assert len(ref_cands) == 1, "median-length reference segment must be unique"
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
_SEGMENT_COUNTS = [5]  # fixed to keep median-length distinct from second-longest
_REF_LEN_RANGE = (4, 7)
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
    return [p for p in patterns if sum(p) <= max_span and len(set(p)) > 1]


def _choose_offsets(d, n):
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _has_two_distinct_long_choices(caps, ref_len, ref_idx):
    nonrefs = [i for i in range(len(caps)) if i != ref_idx]
    options = {
        i: list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[i]) + 1))
        for i in nonrefs
    }
    for i, j in itertools.combinations(nonrefs, 2):
        for vi in options[i]:
            for vj in options[j]:
                if vi != vj:
                    return True
    return False


def _feasible_placements(d, s_vals, ref_len):
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        placements.append((t0, "start", [_max_len_from(d, t0, s) for s in s_vals]))
        placements.append((t0, "end", [_max_len_to(d, t1, s) for s in s_vals]))
    return placements


def _choose_colors(s_vals):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, ref_idx):
    nonrefs = [i for i in range(len(caps)) if i != ref_idx]
    short_values = list(range(2, ref_len))
    if len(short_values) < 2:
        return None

    short_pairs = list(itertools.combinations(nonrefs, 2))
    random.shuffle(short_pairs)
    for short_idx in short_pairs:
        long_idx = [i for i in nonrefs if i not in short_idx]
        long_a = list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[long_idx[0]]) + 1))
        long_b = list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[long_idx[1]]) + 1))
        long_pairs = [(a, b) for a in long_a for b in long_b if a != b]
        if not long_pairs:
            continue

        lengths = [None] * len(caps)
        lengths[ref_idx] = ref_len

        chosen_shorts = random.sample(short_values, 2)
        random.shuffle(chosen_shorts)
        lengths[short_idx[0]] = chosen_shorts[0]
        lengths[short_idx[1]] = chosen_shorts[1]

        long_pair = random.choice(long_pairs)
        lengths[long_idx[0]] = long_pair[0]
        lengths[long_idx[1]] = long_pair[1]
        assert len(set(lengths)) == len(lengths)
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
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    middle_offset_idx = n // 2
    usable = []
    for t0, anchor, caps in placements:
        for ref_idx in range(n):
            if ref_idx == middle_offset_idx:
                continue
            if _has_two_distinct_long_choices(caps, ref_len, ref_idx):
                usable.append((t0, anchor, caps, ref_idx))
    if not usable:
        return None

    t0, anchor, caps, ref_idx = random.choice(usable)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, ref_idx)
    if lengths is None:
        return None

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
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
