# Variant of e376de54
# Dimension: D3 — 目标位置：端点对齐方式 (dst)
# Change: 所有线段的两端都对齐到参考线段的起点和终点位置，保持平行间距不变... → 中心对齐（居中延伸）

# Design analysis for the D3 variant
# - Concrete object model:
#   Each colored line keeps the original "parallel segment" body, but gains one same-color nub
#   placed exactly one perpendicular step away from the segment's center cell. This extra center
#   marker is the minimal adaptation needed to make "center alignment" visible; without it, forcing
#   all segments to the reference length would collapse center-alignment and endpoint-alignment to
#   the same final straight segment footprint.
# - Behavior trace:
#   Original logic on a horizontal example: bodies at rows 2/5/8 with common start or end are all
#   redrawn to share the reference row-wise extent. In the variant, each body also has a center nub.
#   The solver finds the median-offset reference body, reads its center t-coordinate, shifts every
#   other body so its own center matches that reference center, then extends/crops symmetrically
#   around that center and redraws the nub at the aligned center. The visible difference is that the
#   nubs line up by center, not by whichever endpoint happened to be anchored in the input.
# - Complexity calibration:
#   The nub is always same-color and exactly one cell away, so there is still only one object type to
#   infer. Lengths are restricted to odd values, giving each segment an exact center cell and avoiding
#   ambiguous half-cell centers.
# - Hidden assumptions audited against the original solver:
#   The old code assumed straight same-color runs were the entire object and could safely ignore
#   isolated same-color pixels. That is no longer valid: the nub must be detected, cleared, and
#   redrawn with the body. The old "align starts, then draw target_len cells" step also depends on a
#   destination specified by endpoints; here that step is replaced by "align centers, then draw
#   target_len cells symmetrically around the reference center".
# - Placement legality:
#   The generator keeps the original odd segment count, unique perpendicular offsets, and dominant
#   direction. It also enforces gap sizes large enough that center nubs do not overlap neighboring
#   bodies. Every generated sample is checked by running main(input) and matching the constructed
#   output.
# - Visual observability and cleanliness:
#   Inputs are built with at least one non-reference segment shifted to the negative side of the
#   reference center and one shifted to the positive side, plus one shorter and one longer body than
#   the reference. This makes the center re-alignment obvious in 2-4 ARC examples. The nub is kept on
#   one consistent perpendicular side per puzzle to avoid clutter.
# - Color reasoning:
#   Colors are preserved exactly; no ordering or mapping depends on color ids. Color only labels which
#   body/nub pair belongs together.

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


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int, h: int, w: int) -> list[tuple[int, int]] | None:
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
    assert length % 2 == 1, "variant uses odd lengths so each segment has a unique center cell"
    half = length // 2
    t0 = center_t - half * _step(d)
    return _segment_cells(d, t0, s, length, h, w)


def _nub_cell(
    d: tuple[int, int], center_t: int, s: int, side: int, h: int, w: int
) -> tuple[int, int] | None:
    delta = _step(d)
    return _cell_from_ts(d, center_t, s + side * delta, h, w)


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
                assert len(cells) % 2 == 1, "each segment body must have an odd length"
                center = cells[len(cells) // 2]
                nub_cands = []
                for side, (qr, qc) in ((1, (dc, -dr)), (-1, (-dc, dr))):
                    nr, nc = center[0] + qr, center[1] + qc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        nub_cands.append((side, (nr, nc)))
                assert len(nub_cands) == 1, "each segment must carry one same-color center nub"
                start = cells[0]
                end = cells[-1]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "nub": nub_cands[0][1],
                        "nub_side": nub_cands[0][0],
                        "start": start,
                        "end": end,
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
    Center-align parallel colored line segments to a reference line, then resize them symmetrically.

    Input structure:
    - bg is the most common color.
    - Several parallel same-color segment bodies appear in one dominant direction; each body has a
      one-cell nub attached perpendicular to its center cell, using the same color.
    - The segment bodies live on distinct parallel offsets, and isolated pixels unrelated to those
      bodies are left alone.

    Rule:
    - Detect the dominant shared direction and extract each odd-length segment body together with its
      center nub.
    - Choose the reference segment as the one at the median perpendicular offset.
    - Move every segment so its center matches the reference center, then extend or crop it equally on
      both sides until its body length matches the reference length; redraw the nub at that aligned
      center on the same perpendicular side as before.
    - The grid size stays unchanged, and only the detected segment objects are rewritten.

    Key insight: The destination is no longer defined by the reference endpoints. The reference gives a
    target center and a target odd length, so non-reference segments must be reconstructed around that
    center rather than snapped from one end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
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
    assert ref["t_end"] >= ref["t_start"], "traced segment coordinates must increase"
    assert (ref["t_end"] - ref["t_start"]) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref["t_end"] - ref["t_start"]) // delta_t + 1
    assert target_len % 2 == 1, "reference length must be odd in the centered variant"
    target_center = ref["t_center"]

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg
        nr, nc = s["nub"]
        out[nr, nc] = bg

    for s in segments:
        cells = _centered_segment_cells(d, target_center, s["s"], target_len, h, w)
        nub = _nub_cell(d, target_center, s["s"], s["nub_side"], h, w)
        assert cells is not None and nub is not None, "repositioned centered segment must stay in bounds"
        for r, c in cells:
            out[r, c] = s["color"]
        out[nub[0], nub[1]] = s["color"]

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]          # still odd for a unique median reference line
_REF_LEN_OPTIONS = [3, 5, 7]      # odd only so the center is a real cell
_MAX_SEG_LEN = 11                 # odd upper bound for input bodies
_ODD_LENGTHS = [3, 5, 7, 9, 11]

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_NONZERO_SHIFT_PREFERRED = 0.8

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400
_SHIFT_SCAN = range(-8, 9)


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d: tuple[int, int], n: int) -> list[list[int]]:
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


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _object_cells(
    d: tuple[int, int], center_t: int, s: int, length: int, nub_side: int, h: int, w: int
) -> list[tuple[int, int]] | None:
    cells = _centered_segment_cells(d, center_t, s, length, h, w)
    nub = _nub_cell(d, center_t, s, nub_side, h, w)
    if cells is None or nub is None:
        return None
    return cells + [nub]


def _max_centered_len(d: tuple[int, int], center_t: int, s: int, h: int, w: int) -> int:
    delta = _step(d)
    half = 0
    while True:
        next_half = half + 1
        left_ok = _cell_from_ts(d, center_t - next_half * delta, s, h, w) is not None
        right_ok = _cell_from_ts(d, center_t + next_half * delta, s, h, w) is not None
        if not (left_ok and right_ok):
            break
        half = next_half
    return 2 * half + 1


def _feasible_ref_centers(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int, nub_side: int) -> list[int]:
    centers = []
    for center_t in range(*_T_SCAN_RANGE):
        if all(_object_cells(d, center_t, s, ref_len, nub_side, _H, _W) is not None for s in s_vals):
            centers.append(center_t)
    return centers


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _candidate_states(
    d: tuple[int, int], ref_center: int, s: int, ref_len: int, nub_side: int
) -> list[tuple[int, int]]:
    delta = _step(d)
    states = []
    for shift in _SHIFT_SCAN:
        center_t = ref_center + shift * delta
        max_len = min(_MAX_SEG_LEN, _max_centered_len(d, center_t, s, _H, _W))
        for length in _ODD_LENGTHS:
            if length > max_len:
                continue
            if _object_cells(d, center_t, s, length, nub_side, _H, _W) is not None:
                states.append((shift, length))
    return states


def _pick_with_constraints(
    state_lists: list[list[tuple[int, int]]], ref_idx: int, ref_len: int
) -> list[tuple[int, int]] | None:
    n = len(state_lists)
    assignments: list[tuple[int, int] | None] = [None] * n
    assignments[ref_idx] = (0, ref_len)
    nonrefs = [i for i in range(n) if i != ref_idx]

    pattern_defs = [
        ("short_left", lambda st: st[0] < 0 and st[1] < ref_len, "long_right", lambda st: st[0] > 0 and st[1] > ref_len),
        ("long_left", lambda st: st[0] < 0 and st[1] > ref_len, "short_right", lambda st: st[0] > 0 and st[1] < ref_len),
    ]
    random.shuffle(pattern_defs)

    for _, pred_a, _, pred_b in pattern_defs:
        a_lines = [i for i in nonrefs if any(pred_a(st) for st in state_lists[i])]
        b_lines = [i for i in nonrefs if any(pred_b(st) for st in state_lists[i])]
        random.shuffle(a_lines)
        random.shuffle(b_lines)
        for i in a_lines:
            for j in b_lines:
                if i == j:
                    continue
                trial = assignments[:]
                trial[i] = random.choice([st for st in state_lists[i] if pred_a(st)])
                trial[j] = random.choice([st for st in state_lists[j] if pred_b(st)])
                for k in nonrefs:
                    if trial[k] is not None:
                        continue
                    options = state_lists[k]
                    preferred = [st for st in options if st[0] != 0]
                    varied = [st for st in options if st != (0, ref_len)]
                    pool = preferred if preferred and random.random() < _NONZERO_SHIFT_PREFERRED else varied or options
                    trial[k] = random.choice(pool)
                return trial  # type: ignore[return-value]
    return None


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    centers: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
    nub_side: int,
) -> bool:
    used = set()
    objects = []
    for s, center_t, length, color in zip(s_vals, centers, lengths, colors):
        coords = _object_cells(d, center_t, s, length, nub_side, _H, _W)
        if coords is None:
            return False
        for cell in coords:
            if cell in used:
                return False
            used.add(cell)
        objects.append((coords, color))
    for coords, color in objects:
        for r, c in coords:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.choice(_REF_LEN_OPTIONS)
    s_vals = _choose_offsets(d, n)
    nub_side = random.choice([-1, 1])
    ref_centers = _feasible_ref_centers(d, s_vals, ref_len, nub_side)
    if not ref_centers:
        return None

    ref_idx = n // 2
    ref_center = random.choice(ref_centers)
    state_lists = [_candidate_states(d, ref_center, s, ref_len, nub_side) for s in s_vals]
    if any(not states for states in state_lists):
        return None
    states = _pick_with_constraints(state_lists, ref_idx, ref_len)
    if states is None:
        return None

    colors = _choose_colors(s_vals)
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [ref_center] * n, [ref_len] * n, colors, nub_side):
        return None

    centers = [ref_center + shift * _step(d) for shift, _ in states]
    lengths = [length for _, length in states]
    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, centers, lengths, colors, nub_side):
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
