# Combo variant of e376de54 (index 21)
# combo_spec: D6/0+D13/0
# Name: 等间距化+阶梯排列

import random
from collections import Counter

import numpy as np

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


# Design analysis for the D6/0 + D13/0 combo
# a) Concrete behavior and trace
#    The objects remain 3 or 5 parallel colored segments on a 16x16 canvas. Compared with the base task, two visible
#    changes happen at once in the input: the perpendicular offsets are irregular, and the segment starts form a
#    staircase along the drawing direction instead of sharing one aligned start.
#    Small horizontal example:
#      input rows      = [2, 5, 8]
#      input starts    = [1, 4, 6]
#      input lengths   = [7, 4, 2]
#      reference line  = the median-by-row segment at row 5, so ref_start=4 and ref_len=4
#      base task out   = rows stay [2, 5, 8], all starts become 4, all lengths become 4
#      combo task out  = rows become [2, 5, 8] if already equal, or for an irregular input like [2, 4, 8] they
#                        become [2, 5, 8]; in every case all starts become 4 and all lengths become 4
#    So the combo differs from the base in two places: the solver must ignore staggered input starts when taking the
#    reference extent, and the output redraw must change both t positions and s positions.
# b) Complexity calibration
#    The added freedom is intentionally narrow. There is still one shared direction, 3 or 5 segments, and one median
#    reference line. Input spacing is irregular but output spacing is one arithmetic progression; input starts are a
#    monotone staircase chosen from a small template family. That makes the fusion visually obvious without exploding
#    the search space or introducing hidden case splits.
# c) Review of original assumptions that break here
#    - The base generator assumed all segments shared a start or end anchor. D13 breaks that, so the combo generator
#      must build explicit per-segment starts instead of deriving them from one anchor mode.
#    - The base redraw reused each segment's original perpendicular offset. D6 breaks that, so output must be drawn
#      from explicit (t, s) coordinates via _cell_from_ts.
#    - Any logic that infers the reference from start alignment is invalid here. The only stable selector is still the
#      median perpendicular offset, which remains visible even when starts are staggered.
#    - Diagonal equal-spacing still needs parity-safe gaps, and the staircase offsets must move in whole segment steps,
#      so start offsets are stored in segment-step units and converted with delta_t.
# d) Placement legality
#    Generation is output-first. We first choose an equal-spaced output band, then perturb only interior s gaps to get
#    the input band, and finally choose a staircase pattern for starts around the reference start. Feasibility checks
#    ensure every input segment has length >= 2, at least one non-reference segment can be longer than the reference,
#    and both input and output stay in bounds with no overlaps.
# e) Visual observability
#    A human can see both changes directly: input lines are not equally spaced, and their left/top starts walk as a
#    staircase across the ordered band. In the output, that staircase collapses to one aligned start while the band
#    itself becomes evenly spaced. No color id is used for ordering or any hidden arithmetic.
# f) Visual cleanliness
#    Minimum perpendicular gaps stay >= 2 for orthogonal directions and >= 4 for diagonals, so parallel lines do not
#    touch. Staircase offsets are moderate and monotone, which keeps the input readable instead of producing a tangled
#    fan of almost-overlapping segments.
# g) Color reasoning
#    Colors are unchanged labels. They help the viewer track which segment moved, but the transformation never sorts,
#    groups, or remaps by color.
# h) Variant validation metric used during testing
#    A sample is treated as non-degenerate when two conditions hold together:
#    - input adjacent perpendicular gaps are not all equal
#    - input segment starts, after sorting by perpendicular offset, are strictly monotone and not all identical
#    This guarantees the sample really contains both D6-style spacing irregularity and D13-style staircase starts.
# i) New solver assumptions enforced by the generator
#    - segment count is odd and offsets are unique
#    - the outer offset span is divisible by N-1 so equal spacing is integral
#    - diagonal equal gaps are even, preserving lattice parity
#    - the reference segment is the middle one by offset and starts exactly at the target output start
#    - start offsets across the ordered band are strictly monotone, so the staircase is visually unambiguous


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


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


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
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


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


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
    Straighten a staggered band of parallel colored segments into equal spacing and one shared reference extent.

    Input structure:
    - The background is the most frequent color.
    - There are 3 or 5 multi-cell segments, all parallel, on distinct perpendicular offsets.
    - Their starts are staircase-shifted along the common direction, and their perpendicular gaps are uneven.

    Rule:
    - Detect the common direction and sort the segments by perpendicular offset.
    - Use the median-by-offset segment as the reference: its start and length define the common output extent.
    - Keep the outermost two offsets as the band span, redistribute the ordered offsets to equal spacing, and redraw
      every segment at the reference start with the reference length.
    - Clear only the original multi-cell segment pixels; any isolated pixels would remain untouched.

    Key insight: The reference line is still chosen by offset order, not by length or by the staircase pattern. The
    output is produced by normalizing two independent geometric dimensions at once: all starts collapse to the
    reference start, and all perpendicular offsets collapse to an evenly spaced progression.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    segments = sorted(segments, key=lambda s: s["s"])
    s_vals = [s["s"] for s in segments]
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    span = s_vals[-1] - s_vals[0]
    assert span > 0, "need at least two distinct offsets"
    assert span % (len(s_vals) - 1) == 0, "outer span must support equal spacing"
    gap = span // (len(s_vals) - 1)
    if delta_t == 2:
        assert gap % 2 == 0, "diagonal equal gap must preserve parity"

    ref = segments[len(segments) // 2]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1
    target_s_vals = [s_vals[0] + i * gap for i in range(len(segments))]

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg

    for seg, target_s in zip(segments, target_s_vals):
        t = ref_t0
        for _ in range(target_len):
            cell = _cell_from_ts(d, t, target_s)
            assert cell is not None, "normalized segment must stay in bounds"
            rr, cc = cell
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = seg["color"]
            t += delta_t

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

# --- Equal output spacing choices ---
_ORTHO_EQUAL_GAPS = {3: [3, 4, 5, 6], 5: [3]}
_DIAG_EQUAL_GAPS = {3: [6], 5: [6]}

# --- Input minimum gaps ---
_MIN_GAP_ORTHO = 2
_MIN_GAP_DIAG = 4

# --- Staircase templates in segment-step units, with the reference fixed at 0 ---
_STAIR_PATTERNS = {
    3: [(-3, 0, 2), (-2, 0, 3), (-3, 0, 1)],
    5: [(-5, -2, 0, 2, 4), (-4, -1, 0, 2, 5), (-4, -2, 0, 1, 4), (-3, -1, 0, 2, 4)],
}

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _equal_gap_choices(d: tuple[int, int], n: int) -> list[int]:
    if d in [(1, 1), (1, -1)]:
        return _DIAG_EQUAL_GAPS[n]
    return _ORTHO_EQUAL_GAPS[n]


def _min_gap(d: tuple[int, int]) -> int:
    return _MIN_GAP_DIAG if d in [(1, 1), (1, -1)] else _MIN_GAP_ORTHO


def _gap_unit(d: tuple[int, int]) -> int:
    return 2 if d in [(1, 1), (1, -1)] else 1


def _choose_output_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gap = random.choice(_equal_gap_choices(d, n))
    span = (n - 1) * gap
    s0 = random.randint(lo, hi - span)
    return tuple(s0 + i * gap for i in range(n))


def _choose_irregular_gaps(d: tuple[int, int], n: int, equal_gap: int) -> list[int]:
    min_gap = _min_gap(d)
    unit = _gap_unit(d)
    base = [equal_gap] * (n - 1)
    for _ in range(200):
        gaps = base.copy()
        moves = 1 if n == 3 else random.randint(1, 3)
        for _ in range(moves):
            donors = [i for i, g in enumerate(gaps) if g - unit >= min_gap]
            if not donors:
                break
            donor = random.choice(donors)
            receiver = random.choice([i for i in range(len(gaps)) if i != donor])
            gaps[donor] -= unit
            gaps[receiver] += unit
        if len(set(gaps)) > 1 and min(gaps) >= min_gap and sum(gaps) == (n - 1) * equal_gap:
            return gaps
    raise RuntimeError("failed to build irregular gaps")


def _choose_input_offsets(d: tuple[int, int], s_out: tuple[int, ...]) -> tuple[int, ...]:
    gaps = _choose_irregular_gaps(d, len(s_out), s_out[1] - s_out[0])
    s_vals = [s_out[0]]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    assert s_vals[-1] == s_out[-1]
    assert tuple(s_vals) != tuple(s_out)
    return tuple(s_vals)


def _stair_patterns(n: int) -> list[tuple[int, ...]]:
    patterns = []
    for base in _STAIR_PATTERNS[n]:
        patterns.append(base)
        patterns.append(tuple(-x for x in base))
    return patterns


def _feasible_layouts(
    d: tuple[int, int], input_s_vals: tuple[int, ...], output_s_vals: tuple[int, ...], ref_len: int
) -> list[tuple[int, tuple[int, ...], list[int]]]:
    delta = _step(d)
    n = len(input_s_vals)
    ref_idx = n // 2
    layouts = []
    for steps in _stair_patterns(n):
        starts_are_monotone = all(steps[i] < steps[i + 1] for i in range(n - 1)) or all(
            steps[i] > steps[i + 1] for i in range(n - 1)
        )
        assert starts_are_monotone and steps[ref_idx] == 0
        for t0 in range(*_T_SCAN_RANGE):
            if any(_segment_cells(d, t0, s, ref_len) is None for s in output_s_vals):
                continue
            starts = [t0 + step * delta for step in steps]
            caps = []
            ok = True
            for s, start in zip(input_s_vals, starts):
                cap = _max_len_from(d, start, s)
                if cap < 2:
                    ok = False
                    break
                caps.append(cap)
            if not ok or caps[ref_idx] < ref_len:
                continue
            if not any(caps[i] > ref_len for i in range(n) if i != ref_idx):
                continue
            layouts.append((t0, steps, caps))
    return layouts


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(2, min(ref_len - 1, caps[short_i]))

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)

    s_out = _choose_output_offsets(d, n)
    s_in = _choose_input_offsets(d, s_out)
    layouts = _feasible_layouts(d, s_in, s_out, ref_len)
    if not layouts:
        return None

    t0, steps, caps = random.choice(layouts)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_out, [t0] * n, [ref_len] * n, colors):
        return None

    delta = _step(d)
    starts = [t0 + step * delta for step in steps]
    assert all(starts[i] < starts[i + 1] for i in range(n - 1)) or all(starts[i] > starts[i + 1] for i in range(n - 1))

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_in, starts, lengths, colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
