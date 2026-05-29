# Combo variant of e376de54 (index 29)
# combo_spec: D3/2+D6/0+D13/2
# Name: 中心对齐+等间距+居中排列

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    The objects stay as 3 or 5 parallel straight segments on a 16x16 canvas, but the input is no
#    longer "left-anchored" or "right-anchored". Instead, every input segment has odd length and all
#    of them share one common center along the main direction, so a horizontal instance looks like a
#    diamond band rather than a staircase:
#      input rows [2, 5, 9], lengths [3, 7, 5], all centered at column 8
#      original task would align starts/ends to the median row and keep rows fixed
#      this combo keeps the common center at column 8, changes every length to the median row's
#      length 7, and also redistributes rows to equal spacing across the same outer span, e.g.
#      [2, 5, 9] -> [2, 6, 10].
#    The visible result is an equal-length, equal-gap, center-built array. The input already hints at
#    "center" as the organizing principle, and the output completes that theme by making both the
#    lengths and the spacing uniform.
# b) Complexity calibration
#    Odd lengths are the key simplification: they make each segment's center a real cell, so the
#    center rule is readable without adding extra markers. Segment count stays odd, direction still
#    comes from the original 4-way set, and spacing is normalized by one arithmetic progression per
#    puzzle. That is enough structure to show the fusion clearly without introducing multiple object
#    types or color-dependent logic.
# c) Hidden assumptions audited against the original solver
#    - Original code assumed a shared start or end and only shifted segments along the main direction.
#      That breaks here because the input no longer has a common endpoint. The solver must rebuild
#      each output segment from its target center and target perpendicular offset.
#    - Original code allowed arbitrary lengths >= 2. That breaks center reconstruction because even
#      lengths have half-cell centers. The combo therefore restricts all generated segment lengths,
#      including the reference, to odd values.
#    - Original spacing logic kept each segment on its original perpendicular line. That is false in
#      the D6 part of the combo: ordered offsets are re-laid onto equal spacing, so output drawing
#      must be done from explicit (t, s) coordinates instead of from the old starts.
#    - Diagonal equal-spacing has a parity requirement. If the output gap were odd, some target cells
#      would leave the lattice; the generator therefore uses even diagonal equal gaps and the solver
#      asserts that divisibility condition.
# d) Placement legality
#    The generator is output-first. It chooses an equal-spaced output band, perturbs only the
#    interior perpendicular gaps to make the input irregular, then searches for a common center that
#    keeps both input and output in bounds. A sample is accepted only if every segment remains
#    disjoint and main(input) reproduces the constructed output exactly.
# e) Visual observability
#    The input always has unequal adjacent gaps and at least one segment shorter and one longer than
#    the reference. That makes both halves of the combo visible: equal spacing changes the band
#    geometry, and center-based length normalization changes the diamond profile into a uniform bar
#    array. Colors are preserved and do not encode the rule.
# f) Visual cleanliness
#    Minimum perpendicular gaps remain >= 2 for orthogonal directions and >= 4 for diagonals, so
#    parallel segments do not touch. Because every segment shares one center instead of carrying extra
#    markers, the scene stays sparse and readable.
# g) Change inferability
#    No color remapping is used. The only cues are geometric: same center in the input, then same
#    center plus equal spacing plus equal length in the output. With 2-4 examples, a human can infer
#    that the median-by-offset line supplies the target length while the whole band is regularized.
# h) Degeneracy checks used after implementation
#    A generated sample is treated as healthy only if input gaps are not already equal and if the
#    combo solver's output differs from the original e376de54 endpoint-aligning behavior on that same
#    input. Those checks verify that both the center rule and the spacing rule are materially active.
# i) New solver assumptions that the generator must satisfy
#    - all multi-cell segments are odd-length
#    - the ordered outer span supports integral equal spacing
#    - diagonal equal gaps are even
#    - the median-by-offset segment is unique
#    - all input segments truly share one center coordinate along the main direction

import random
import sys

import numpy as np

_SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS not in sys.path:
    sys.path.append(_SOLVER_UTILS)

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


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int, h: int, w: int) -> tuple[int, int] | None:
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
    if 0 <= r < h and 0 <= c < w:
        return r, c
    return None


def _segment_cells(
    d: tuple[int, int], t0: int, s: int, length: int, h: int, w: int
) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s, h, w)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _centered_segment_cells(
    d: tuple[int, int], center_t: int, s: int, length: int, h: int, w: int
) -> list[tuple[int, int]] | None:
    assert length % 2 == 1, "centered variant requires odd lengths"
    half = length // 2
    t0 = center_t - half * _step(d)
    return _segment_cells(d, t0, s, length, h, w)


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
                assert len(cells) % 2 == 1, "each segment must have odd length"
                start = cells[0]
                end = cells[-1]
                center = cells[len(cells) // 2]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "t_center": _dot(center, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a centered band of parallel segments into an equal-length, equal-spaced centered array.

    Input structure:
    - The background is the most frequent color.
    - There are 3 or 5 multi-cell segments in one dominant direction, each on a distinct parallel offset.
    - Every segment has odd length, and the whole input band is center-built: all segment centers lie on
      the same coordinate along the main direction, while the perpendicular gaps are intentionally uneven.

    Rule:
    - Detect the shared direction and extract the odd-length segments.
    - Choose the median-by-offset segment as the reference; its center gives the target center and its
      length gives the target odd length.
    - Keep the ordered outer span of the band, but redistribute the offsets into equal spacing from the
      first line to the last line.
    - Redraw every segment at the reference center, with the reference length, on the equalized offsets.
      The grid size stays unchanged and isolated single pixels are untouched.

    Key insight: The input already advertises "center" as the alignment cue, so the solver must think in
    terms of center coordinates instead of shared endpoints. The output is produced by combining two
    regularizations at once: symmetric length normalization around the reference center and arithmetic
    equalization of the ordered parallel offsets.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
    segments = _extract_segments(g, bg, d)
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
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4, "diagonal input gaps must separate segments"

    ref = segments[len(segments) // 2]
    assert ref["t_end"] >= ref["t_start"], "traced segment coordinates must increase"
    target_len = ref["len"]
    assert target_len % 2 == 1, "reference length must be odd"
    target_center = ref["t_center"]
    for s in segments:
        assert s["t_center"] == target_center, "all input segments must share one center"

    target_s_vals = [s_vals[0] + i * gap for i in range(len(segments))]

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for seg, target_s in zip(segments, target_s_vals):
        cells = _centered_segment_cells(d, target_center, target_s, target_len, h, w)
        assert cells is not None, "repositioned segment must stay in bounds"
        for r, c in cells:
            out[r, c] = seg["color"]

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Segment counts and centered lengths ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_OPTIONS = [5, 7]
_MAX_SEG_LEN = 11
_ODD_LENGTHS = [3, 5, 7, 9, 11]

# --- Equal output spacing choices ---
_ORTHO_EQUAL_GAPS = {3: [3, 4, 5, 6], 5: [3]}
_DIAG_EQUAL_GAPS = {3: [6], 5: [6]}

# --- Input minimum gaps ---
_MIN_GAP_ORTHO = 2
_MIN_GAP_DIAG = 4

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_KEEP_REF_LEN_PROB = 0.25

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


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
    for _ in range(300):
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


def _max_centered_len(d: tuple[int, int], center_t: int, s: int) -> int:
    delta = _step(d)
    half = 0
    while True:
        next_half = half + 1
        left_ok = _cell_from_ts(d, center_t - next_half * delta, s, _H, _W) is not None
        right_ok = _cell_from_ts(d, center_t + next_half * delta, s, _H, _W) is not None
        if not (left_ok and right_ok):
            break
        half = next_half
    return 2 * half + 1


def _feasible_centers(
    d: tuple[int, int], input_s_vals: tuple[int, ...], output_s_vals: tuple[int, ...], ref_len: int
) -> list[tuple[int, list[int]]]:
    centers = []
    all_s_vals = sorted(set(input_s_vals) | set(output_s_vals))
    ref_idx = len(input_s_vals) // 2
    for center_t in range(*_T_SCAN_RANGE):
        if any(_centered_segment_cells(d, center_t, s, ref_len, _H, _W) is None for s in all_s_vals):
            continue
        caps = [_max_centered_len(d, center_t, s) for s in input_s_vals]
        if not any(any(length > ref_len for length in _ODD_LENGTHS if length <= min(_MAX_SEG_LEN, caps[i])) for i in range(len(caps)) if i != ref_idx):
            continue
        centers.append((center_t, caps))
    return centers


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
    shorter_options = [length for length in _ODD_LENGTHS if length < ref_len]
    assert shorter_options, "reference length must admit a shorter odd length"
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [
        i
        for i in nonrefs
        if any(length > ref_len for length in _ODD_LENGTHS if length <= min(_MAX_SEG_LEN, caps[i]))
    ]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.choice(
        [length for length in _ODD_LENGTHS if ref_len < length <= min(_MAX_SEG_LEN, caps[long_i])]
    )
    lengths[short_i] = random.choice(shorter_options)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = [length for length in _ODD_LENGTHS if length <= min(_MAX_SEG_LEN, caps[i])]
        if not options:
            return None
        if len(options) > 1 and ref_len in options and random.random() > _KEEP_REF_LEN_PROB:
            varied = [length for length in options if length != ref_len]
            if varied:
                options = varied
        lengths[i] = random.choice(options)
    return lengths


def _paint_centered(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    center_t: int,
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    used = set()
    objects = []
    for s, length, color in zip(s_vals, lengths, colors):
        cells = _centered_segment_cells(d, center_t, s, length, _H, _W)
        if cells is None:
            return False
        for cell in cells:
            if cell in used:
                return False
            used.add(cell)
        objects.append((cells, color))
    for cells, color in objects:
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.choice(_REF_LEN_OPTIONS)

    s_out = _choose_output_offsets(d, n)
    s_in = _choose_input_offsets(d, s_out)
    center_data = _feasible_centers(d, s_in, s_out, ref_len)
    if not center_data:
        return None

    center_t, caps = random.choice(center_data)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None
    colors = _choose_colors(n)

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint_centered(out, d, s_out, center_t, [ref_len] * n, colors):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_centered(inp, d, s_in, center_t, lengths, colors):
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
