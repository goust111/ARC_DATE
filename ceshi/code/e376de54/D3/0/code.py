# Variant of e376de54
# Dimension: D3 — 目标位置：端点对齐方式 (dst)
# Change: 所有线段的两端都对齐到参考线段的起点和终点位置，保持平行间距不变... → 仅对齐起点（头端对齐）

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D3 variant
# ----------------------------------
# a) Concretization and behavior trace
#    - The objects are still several parallel monochrome segments on a 16x16 orange background.
#    - They still share one dominant direction and have distinct perpendicular offsets with visible gaps.
#    - The median-by-offset segment is still the reference. The changed behavior is only the destination:
#      every segment moves so that its start lands on the reference start, but its own length is preserved.
#    - Concrete trace on a horizontal case:
#      original rule: input lengths [3,5,8] with common end at column 10 -> output starts all at the
#      reference start and lengths become [5,5,5];
#      new rule: same input -> output starts all at the reference start but lengths stay [3,5,8].
#      So the changed step is the redraw phase, not direction detection or reference selection.
# b) Complexity calibration
#    - Keep the original odd segment count {3,5}, same four directions, same gap discipline, same
#      color palette and possible color reuse. This preserves enough visual structure to detect the
#      reference line while avoiding new ambiguity from extra object types.
#    - To make the D3 change visible, generated inputs are biased to the "common end" branch only.
#      Then the output flips from shared ends to shared starts, and the preserved varying lengths are
#      easy to see. We also force at least one shorter and one longer non-reference segment.
# c) Hidden assumption audit against the original main()
#    - Segment extraction, dominant direction detection, unique median offset, and clearing original
#      pixels do not depend on equalizing lengths, so they remain valid.
#    - The old code depended on ref_t1 and target_len and redrew every segment with that one length.
#      That assumption is invalid here because non-reference lengths must survive. The redraw must use
#      each segment's own len(cells) instead.
#    - The old generator allowed both shared-start and shared-end inputs. Shared-start inputs would
#      often become nearly identity examples under this variant, so the generator is restricted to
#      shared-end inputs for observability.
# d) Placement legality
#    - Output segments all start at the same t0 but may have different lengths.
#    - Input segments all end at the reference end t1 and use the same per-segment lengths as output.
#    - For each offset we therefore need both forward capacity from t0 and backward capacity from t1;
#      the usable cap is their minimum. This guarantees both input and output stay in bounds.
# e) Visual observability
#    - Every generated puzzle contains at least one segment shorter than the reference and one longer.
#      Therefore the new output has visibly staggered ends, while the original task would have flattened
#      them to one end. This makes the variant distinguishable from the base task in a few examples.
# f) Visual cleanliness
#    - Parallel offsets remain distinct and irregular. Because all segments stay on separate parallel
#      lines, shifting them to a shared start still cannot create overlaps or crossings.
# g) Color inference
#    - No color-based ordering or recoloring is introduced. Colors only label segments, exactly as in
#      the original task, so the human-visible rule remains geometric.
# h) Quantifiable non-degeneracy target used in testing
#    - For generated samples, the number of distinct output lengths should exceed 1, and the new output
#      should differ from the original solver's output on the same input. Those are measured after coding.
# i) New solver assumptions to guarantee via generation
#    - The solver still assumes one uniquely dominant direction, an odd number of multi-cell segments,
#      unique perpendicular offsets, and enough spacing for diagonal disambiguation. The generator keeps
#      all of those constraints unchanged.

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
    Shift parallel colored line segments so their starting endpoints line up with a reference segment.

    Input structure:
    - bg is the most frequent color.
    - Several multi-cell line segments run in one shared direction (horizontal, vertical, or diagonal).
    - The segments lie on distinct parallel tracks and may have different lengths, while isolated pixels are irrelevant.

    Rule:
    - Detect the dominant direction and extract every segment of length at least two.
    - Use the segment at the median perpendicular offset as the reference line.
    - Move each segment along the common direction until its start matches the reference start, but keep its original length.
    - Clear only the original segment cells first; then redraw the shifted segments on the same grid.

    Key insight: The reference supplies only the shared start position. Unlike the base task, segment lengths are not normalized to the reference.
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
    ref_t0 = ref["t_start"]
    assert ref["t_end"] >= ref_t0, "traced segment coordinates must increase"

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
        rr, cc = r0, c0
        for _ in range(s["len"]):
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
    """Return list of (r, c) for a segment starting at t0 with given offset s, or None if OOB."""
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
    """
    Find all (t0, cap_list) where:
    - the output can place every segment starting at t0, and
    - the input can place every segment ending at the reference end t1,
    - with room for at least one non-reference segment to be longer than ref_len.
    """
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        caps = []
        for s in s_vals:
            start_cap = _max_len_from(d, t0, s)
            end_cap = _max_len_to(d, t1, s)
            caps.append(min(start_cap, end_cap))
        if any(caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, caps))
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
    """Choose per-segment lengths, preserving the reference length and forcing shorter/longer contrasts."""
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


def _gen_attempt():
    """Build one candidate input whose output uses shared starts and preserved lengths."""
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

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, lengths, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
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
