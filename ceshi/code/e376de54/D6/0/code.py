# Variant of e376de54
# Dimension: D6 — 线段之间的空间间距 (meta)
# Change: 平行线段之间保持固定间距（通常1-3像素），间距在变换后不变... → 间距统一化为等间距

import random
from collections import Counter

import numpy as np

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


# Design analysis for the D6 variant
# a) Concrete behavior and trace
#    The objects are still 3 or 5 same-direction colored segments on a 16x16 canvas.
#    Input now differs from the original task in one visible way: the perpendicular offsets are not equally spaced.
#    Output still aligns all segment lengths to the median-by-offset segment, but it also redistributes the ordered
#    segment band into equal spacing while keeping the outermost two lines at the same offsets.
#    Small horizontal example:
#      input rows = [2, 4, 8], lengths = [7, 4, 2], middle row 4 is the reference length 4
#      original rule would output rows [2, 4, 8] with all lengths 4
#      new rule outputs rows [2, 5, 8] with all lengths 4 because the outer span 8-2=6 is repartitioned into
#      two equal gaps of 3 while preserving the row order and the two outer rows.
#      The flow difference is: old solver changed t only; new solver changes t and ordered s positions.
# b) Complexity calibration
#    The new freedom is limited to one arithmetic-progression spacing per instance. Segment count stays 3 or 5, the
#    line direction set is unchanged, and colors still do not encode behavior. This keeps the task inferable while
#    making the D6 change obvious by eye.
# c) Review of original assumptions that break here
#    - Original drawing reused each segment's original start cell after shifting only along the main direction.
#      That assumes each segment keeps its old perpendicular offset. It is false in this variant.
#      Fix: redraw output from explicit (t, s) coordinates via _cell_from_ts instead of mutating the original start.
#    - Original generator chose one irregular set of offsets and used it for both input and output.
#      That directly contradicts the new rule. Fix: generate an equal-spaced output band first, then perturb interior
#      gaps in the input while preserving order and outer span.
#    - Original diagonal safety only needed gaps >= 4 in the input. Here the output gap must also preserve parity, so
#      diagonal equal gaps are forced to be even and the generator uses gap 6 to allow irregular-but-separated inputs.
# d) Placement legality
#    The generator keeps the outer span fixed between input and output and checks the union of input/output offsets
#    at the reference length before choosing anchored lengths. This guarantees no overlaps, no OOB cells, and that
#    both states are solvable by the same detected direction.
# e) Visual observability
#    With outer lines fixed and interior lines moved to equal slots, the spacing change is directly visible in 2-3
#    examples. The transformation is geometric, not tied to color ids.
# f) Visual cleanliness
#    Minimum perpendicular gaps remain >= 2 for orthogonal directions and >= 4 for diagonals, so parallel lines do
#    not touch or create ambiguous merged shapes.
# g) Color reasoning
#    Colors are unchanged. They only label segments so a human can track which line moved; no color-based sorting or
#    remapping is used.
# h) Variant validation metric used during testing
#    A sample is considered non-degenerate when input adjacent gaps are not all equal and the new output differs from
#    the original spacing-preserving solver on that same input.
# i) New solver assumptions enforced by the generator
#    - segment count is odd and offsets are unique
#    - the outer span is divisible by N-1 so equal spacing is integral
#    - diagonal outer spans imply an even equal gap, hence valid lattice cells


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
    Normalize a band of parallel colored segments so they share one reference length and become evenly spaced.

    Input structure:
    - The background is the most frequent color.
    - There are 3 or 5 multi-cell segments, all parallel and all lying on distinct perpendicular offsets.
    - Their perpendicular gaps are intentionally uneven in the input, while their lengths and aligned endpoint may vary.

    Rule:
    - Detect the common direction and sort the segments by perpendicular offset.
    - Use the median-by-offset segment as the length reference: every output segment adopts that segment's start and end
      coordinates along the main direction.
    - Keep the ordered outer span of the segment band, but redistribute the offsets into equal spacing from the first
      line to the last line.
    - Redraw each segment at its new offset with the reference length; isolated single pixels stay unchanged.

    Key insight: The central line still defines the shared extent, but the output is no longer obtained by sliding only
    along the segment direction; the whole stack is re-laid onto an evenly spaced arithmetic progression of offsets.
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
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for seg, target_s in zip(segments, target_s_vals):
        t = ref_t0
        for _ in range(target_len):
            cell = _cell_from_ts(d, t, target_s)
            assert cell is not None, "equalized segment must stay in bounds"
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

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    length = 0
    while _cell_from_ts(d, t0 + length * _step(d), s) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    length = 0
    while _cell_from_ts(d, t1 - length * _step(d), s) is not None:
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
            receiver_choices = [i for i in range(len(gaps)) if i != donor]
            receiver = random.choice(receiver_choices)
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


def _feasible_placements(
    d: tuple[int, int], input_s_vals: tuple[int, ...], output_s_vals: tuple[int, ...], ref_len: int
) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    ref_idx = len(input_s_vals) // 2
    placements = []
    all_s_vals = sorted(set(input_s_vals) | set(output_s_vals))
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in all_s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in input_s_vals]
        if any(start_caps[i] > ref_len for i in range(len(input_s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in input_s_vals]
        if any(end_caps[i] > ref_len for i in range(len(input_s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


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
    lengths[short_i] = random.randint(2, ref_len - 1)

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
    placements = _feasible_placements(d, s_in, s_out, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_out, [t0] * n, [ref_len] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

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
