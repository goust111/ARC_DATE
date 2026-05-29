# Combo variant of e376de54 (index 5)
# combo_spec: D3/0+D13/0
# Name: 头端对齐+阶梯排列

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    - The scene still contains 3 or 5 parallel monochrome segments on a 16x16 background, all lying on
#      distinct parallel tracks of one dominant direction.
#    - The reference segment is still chosen by the median perpendicular offset, exactly as in the base task.
#    - Fusion change 1 (D13): the input no longer shares one common start or end. Instead, when segments are
#      ordered by perpendicular offset, their starts form a strict staircase along the travel direction.
#    - Fusion change 2 (D3): in the output, every segment is shifted so only its start matches the reference
#      start; each segment keeps its own original length, so output ends remain staggered.
#    - Concrete horizontal example with three rows ordered top->bottom:
#      input starts = [2, 5, 8], lengths = [7, 4, 6], reference = middle row starting at 5.
#      base task would redraw all three rows from start 5 with length 4, flattening both ends.
#      combo task redraws them from start 5 with lengths [7, 4, 6], so the left edges align while the
#      right edges stay uneven. The changed parts are therefore both the input arrangement and the redraw rule.
# b) Complexity calibration
#    - Keep the original direction set, odd segment counts, gap discipline, and color reuse policy. That
#      preserves the original object vocabulary and avoids introducing extra cues unrelated to the combo rule.
#    - The staircase is generated with the reference in the middle and non-reference starts on both sides of it.
#      This guarantees visible motion in both directions and avoids weak near-identity examples.
#    - Lengths are forced to include at least one segment shorter than the reference and one longer than it.
#      This ensures the output shows aligned heads plus uneven tails, which is the distinctive visual effect.
# c) Hidden assumption audit against the original main()
#    - Direction detection via same-color adjacency remains valid because all segments are still maximal straight
#      runs on well-separated parallel tracks.
#    - Segment extraction by scanning from cells with no predecessor along the main direction remains valid even
#      when starts are staggered, because each segment is still contiguous and single-track.
#    - The base solver assumed every output should use ref_t0 and ref_t1, hence a single target length. That
#      assumption is invalid here; the redraw must use each segment's own length after aligning only the start.
#    - The solver now assumes only that every segment start can be shifted onto the reference start on the same
#      t-lattice. The generator guarantees this by constructing all starts as integer direction steps from ref_t0.
# d) Placement legality
#    - Output legality requires that each segment of its own length fits when redrawn from the common reference
#      start on its parallel track.
#    - Input legality requires that the same segment length also fits from that segment's staircase start.
#    - The generator therefore computes, per track, the minimum of forward capacity from the shared output start
#      and forward capacity from the staircase input start. Chosen lengths never exceed that cap.
# e) Visual observability
#    - In every sample, input starts are all distinct and monotone by offset, so a human can see the staircase.
#    - In every sample, output starts collapse to one aligned head while at least two output lengths differ, so
#      the preserved-tail rule is visible in a few examples and cannot be mistaken for the base task.
# f) Visual cleanliness
#    - Parallel offsets remain distinct with the original orthogonal/diagonal spacing constraints, so shifting
#      segments cannot create overlaps or crossings.
#    - Because the transformation only moves segments along their own direction, each color stays on one clean
#      parallel track before and after the move.
# g) Change inferability
#    - No recoloring or color-dependent ordering is introduced. Colors serve only as persistent identities for
#      segments, keeping the rule purely geometric and visually deducible from position and length.
# h) Quantifiable non-degeneracy target used in testing
#    - Generated samples should satisfy:
#      1. input starts are not all equal and are strictly monotone by perpendicular offset,
#      2. output lengths contain more than one distinct value,
#      3. this combo output differs from the base solver output on the same input.
# i) New solver assumptions to guarantee via generation
#    - There is one uniquely dominant direction, an odd number of multi-cell segments, unique perpendicular
#      offsets with sufficient diagonal gaps, a unique median reference, and all staircase offsets lie on the
#      same integer step lattice as the reference start. The generator enforces each of these.

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


# Shift staircase-arranged parallel segments so their leading endpoints match a median reference.
# Input structure:
# - The grid contains 3 or 5 multi-cell parallel segments on distinct tracks of one dominant direction.
# - Segment starts are staggered into a staircase, lengths vary, and the median-by-offset segment is the reference.
# Rule:
# - Detect the shared direction, extract maximal segments, and find the reference by median perpendicular offset.
# - Shift every segment so only its start matches the reference start, then redraw it with its original length.
# Key insight:
# - The transformation collapses the staircase only at the head end; tail positions still encode each segment's input length.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift parallel colored line segments so their leading endpoints coincide with a median reference segment.

    Input structure:
    - The background is the most common color.
    - Several multi-cell segments share one horizontal, vertical, or diagonal direction and sit on distinct parallel tracks.
    - Their starts are visibly staggered into a staircase across tracks, and their lengths may differ; isolated single pixels are irrelevant.

    Rule:
    - Detect the dominant shared direction and extract each maximal segment of length at least two.
    - Select the segment at the median perpendicular offset as the reference.
    - Translate every segment along the common direction until its start matches the reference start.
    - Keep each segment's own original length when redrawing, so only the heads align while the tails remain uneven.

    Key insight: The reference contributes a shared start coordinate only. Unlike the base task, the output does not normalize segment lengths or terminal positions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
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

# --- Staircase controls ---
_START_GAP_OPTIONS = [1, 2, 3]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


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


def _stair_patterns(n):
    """Relative input start offsets, in segment steps, with the reference fixed at 0."""
    patterns = []
    if n == 3:
        for a in _START_GAP_OPTIONS:
            for b in _START_GAP_OPTIONS:
                if a == b:
                    continue
                patterns.append((-a, 0, b))
    else:
        for a in _START_GAP_OPTIONS:
            for b in _START_GAP_OPTIONS:
                for c in _START_GAP_OPTIONS:
                    for d in _START_GAP_OPTIONS:
                        gaps = [a, b, c, d]
                        if len(set(gaps)) == 1:
                            continue
                        patterns.append((-(a + b), -b, 0, c, c + d))
    return patterns


def _feasible_placements(d, s_vals, ref_len, rel_steps):
    """
    Find all placements where:
    - output segments can all start at reference t0,
    - input segments can all start at their staircase positions,
    - the reference has length ref_len,
    - at least one non-reference track can exceed ref_len.
    """
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        caps = []
        starts = []
        ok = True
        for s, rel in zip(s_vals, rel_steps):
            in_t0 = t0 + rel * delta
            out_cap = _max_len_from(d, t0, s)
            in_cap = _max_len_from(d, in_t0, s)
            cap = min(out_cap, in_cap, _MAX_SEG_LEN)
            if cap < 2:
                ok = False
                break
            caps.append(cap)
            starts.append(in_t0)
        if not ok or caps[ref_idx] < ref_len:
            continue
        if any(caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, tuple(starts), tuple(caps)))
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
    lengths[long_i] = random.randint(ref_len + 1, caps[long_i])
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, caps[i] + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return tuple(lengths)


def _paint(grid, d, s_vals, starts, lengths, colors):
    """Render segments onto grid. Returns False if any segment cell is OOB."""
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _is_strict_staircase(starts):
    return all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))


def _gen_attempt():
    """Build one candidate input whose starts are staggered and whose output aligns only the starts."""
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    rel_steps = random.choice(_stair_patterns(n))
    placements = _feasible_placements(d, s_vals, ref_len, rel_steps)
    if not placements:
        return None

    t0, starts, caps = random.choice(placements)
    assert _is_strict_staircase(starts)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, lengths, colors):
        return None

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
