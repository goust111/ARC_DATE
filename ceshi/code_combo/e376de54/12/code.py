# Combo variant of e376de54 (index 12)
# combo_spec: D2/3+D11/0
# Name: 反对角线+虚线模式

import random

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    from solvers.arc_utils import get_bg

# Design analysis for the combo variant
# a) Pixel concretization and behavior trace:
#    The original task may contain solid segments in several orientations. This combo removes both freedoms that
#    the old solver relied on: every meaningful object is now a dashed anti-diagonal segment on direction
#    d=(1,-1), so visible cells advance by (+2,-2) while the hidden geometric line still advances by (+1,-1).
#    Example: if a segment occupies the continuous anti-diagonal positions
#    [(10,2),(9,3),(8,4),(7,5),(6,6)], the dashed rendering keeps only [(10,2),(8,4),(6,6)] and leaves
#    [(9,3),(7,5)] as background. In the original logic, tracing by adjacent same-color cells would find three
#    isolated pixels and infer length 1 each. In the fused logic, those three visible cells are one segment with
#    geometric span 5. After transformation, every segment is shifted so its first visible endpoint matches the
#    median-offset reference segment, then redrawn with the same anti-diagonal dashed cadence and the same span.
# b) Complexity calibration:
#    The two changes reinforce each other without adding unrelated branches: only one non-axis direction exists,
#    and the dashed cadence is always period-2. To keep the behavior visually obvious, generated spans are odd
#    (5/7/9 for the reference, odd 3..11 overall), so both visible endpoints remain present. Segment counts stay
#    at 3 or 5, preserving a unique median offset exactly as in the original task.
# c) Implicit-assumption audit against the original main():
#    - Adjacent-neighbor direction detection is invalid because dashed anti-diagonals have no same-color
#      immediate neighbors.
#    - Walking cell-by-cell along d=(1,-1) would split one dashed segment into isolated singletons.
#    - `len(cells)` no longer equals the intended length; the correct measure is endpoint span in continuous
#      anti-diagonal positions.
#    - Redrawing every step would incorrectly fill the holes and revert the combo to the solid-line task.
#    - The original multi-direction branch space can be removed because this combo fixes the orientation.
#    The new solver therefore uses one fixed direction, traces links across a single background gap, and computes
#    span from `t_start`/`t_end` instead of painted-cell count.
# d) Placement legality:
#    The generator is still output-first. It first places the target anti-diagonal dashed segments at distinct
#    perpendicular offsets with gaps >= 4, ensuring no overlap between parallel lines. The input is derived by
#    keeping either all starts or all ends anchored and varying only the odd spans. Every candidate is checked
#    for in-bounds dashed cells before painting, and solver self-verification rejects malformed attempts.
# e) Visual observability:
#    Across examples, a human can see two stable cues at once: all objects slant from lower-left to upper-right,
#    and each object has regular one-cell holes. The output keeps those holes, so the rule cannot be mistaken for
#    "fill into solid lines"; only endpoint alignment and span normalization change.
# f) Visual cleanliness:
#    Parallel offsets are irregular but well separated, and each segment gets its own color. That prevents visual
#    merging across nearby anti-diagonals and also avoids accidental alternative same-color chains.
# g) Change inferability:
#    No color remapping is introduced. Colors only identify which dashed anti-diagonal segment moves where, so
#    the inferable novelty is purely geometric: fixed anti-diagonal direction plus span-based dashed alignment.

_DIR = (1, -1)
_H, _W = 16, 16
_BG = 7
_COLORS = [1, 2, 3, 5, 6, 8, 9]
_SEGMENT_COUNTS = [3, 5]
_REF_SPANS = [5, 7, 9]
_MIN_SPAN = 3
_MAX_SEG_SPAN = 11
_PERP_GAP_DIAG = [4, 6]
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400
_SAME_SPAN_AVOID_PROB = 0.75


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(t: int, s: int) -> tuple[int, int] | None:
    if (t - s) % 2 or (-s - t) % 2:
        return None
    r = (t - s) // 2
    c = (-s - t) // 2
    if 0 <= r < _H and 0 <= c < _W:
        return r, c
    return None


def _has_dash_link(g: np.ndarray, bg: int, r: int, c: int, d: tuple[int, int], color: int) -> bool:
    dr, dc = d
    mr, mc = r + dr, c + dc
    nr, nc = r + 2 * dr, c + 2 * dc
    if not (_in_bounds(g.shape, mr, mc) and _in_bounds(g.shape, nr, nc)):
        return False
    return int(g[mr, mc]) == bg and int(g[nr, nc]) == color


def _trace_dashed_segment(
    g: np.ndarray, bg: int, r: int, c: int, d: tuple[int, int]
) -> list[tuple[int, int]]:
    color = int(g[r, c])
    cells = [(r, c)]
    rr, cc = r, c
    while _has_dash_link(g, bg, rr, cc, d, color):
        rr += 2 * d[0]
        cc += 2 * d[1]
        cells.append((rr, cc))
    return cells


def _extract_segments(g: np.ndarray, bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    d = _DIR
    dr, dc = d
    delta_t = _step(d)
    h, w = g.shape
    segments = []
    used = set()

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg:
                continue
            if _has_dash_link(g, bg, r, c, (-dr, -dc), color):
                continue

            cells = _trace_dashed_segment(g, bg, r, c, d)
            if len(cells) < 2:
                continue

            start = cells[0]
            end = cells[-1]
            t_start = _dot(start, d)
            t_end = _dot(end, d)
            assert (t_end - t_start) % delta_t == 0
            span = (t_end - t_start) // delta_t + 1
            assert span >= _MIN_SPAN and span % 2 == 1
            assert len(cells) == (span + 1) // 2
            segments.append(
                {
                    "color": color,
                    "cells": cells,
                    "start": start,
                    "end": end,
                    "t_start": t_start,
                    "t_end": t_end,
                    "s": _perp_dot(start, d),
                    "span": span,
                }
            )
            used.update(cells)

    return segments, used


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align dashed anti-diagonal segments so they all inherit the reference segment's full span.

    Input structure:
    - bg is the most common color.
    - Several parallel segments slant from lower-left to upper-right only.
    - Each segment is dashed: one colored anti-diagonal cell, one background hole on the same underlying line,
      then the next colored cell.

    Rule:
    - Treat d=(1,-1) as the shared line direction and recover each segment by linking same-color cells across a
      single background gap.
    - Choose the segment with the median perpendicular offset as the reference.
    - Shift every recovered segment so its first visible endpoint aligns with the reference start, then redraw it
      with the reference's full endpoint-to-endpoint span while preserving the dashed cadence.
    - The grid size stays fixed, and non-segment stray pixels remain unchanged.

    Key insight: For dashed anti-diagonals, length means the full continuous anti-diagonal span between the two
    visible endpoints, not the number of painted cells that happen to survive the dash pattern.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    d = _DIR
    dr, dc = d
    delta_t = _step(d)

    segments, _ = _extract_segments(g, bg)
    assert segments, "at least one dashed anti-diagonal segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    for i in range(len(s_vals) - 1):
        assert s_vals[i + 1] - s_vals[i] >= 4, "anti-diagonal segment gaps must stay separated"
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0 = ref["t_start"]
    target_span = ref["span"]

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    occ_target = (target_span + 1) // 2
    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must align with the anti-diagonal lattice"
        k = dt // delta_t
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
        for _ in range(occ_target):
            assert 0 <= rr < h and 0 <= cc < w, "shifted dashed segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += 2 * dr
            cc += 2 * dc

    return out


def _dash_cells(t0: int, s: int, span: int) -> list[tuple[int, int]] | None:
    assert span >= _MIN_SPAN and span % 2 == 1
    delta_t = _step(_DIR)
    cells = []
    for offset in range(0, span, 2):
        cell = _cell_from_ts(t0 + offset * delta_t, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_span_from(t0: int, s: int) -> int:
    delta_t = _step(_DIR)
    occ = 0
    while _cell_from_ts(t0 + 2 * occ * delta_t, s) is not None:
        occ += 1
    return 2 * occ - 1 if occ else 0


def _max_span_to(t1: int, s: int) -> int:
    delta_t = _step(_DIR)
    occ = 0
    while _cell_from_ts(t1 - 2 * occ * delta_t, s) is not None:
        occ += 1
    return 2 * occ - 1 if occ else 0


def _s_limits() -> tuple[int, int]:
    return -30, 0


def _gap_patterns(n: int) -> list[list[int]]:
    lo, hi = _s_limits()
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [g] for p in patterns for g in _PERP_GAP_DIAG]
    return [
        p
        for p in patterns
        if sum(p) <= max_span and (n == 3 and p[0] != p[1] or n == 5 and len(set(p)) > 1)
    ]


def _choose_offsets(n: int) -> tuple[int, ...]:
    lo, hi = _s_limits()
    gaps = random.choice(_gap_patterns(n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(s_vals: tuple[int, ...], ref_span: int) -> list[tuple[int, str, list[int]]]:
    ref_idx = len(s_vals) // 2
    delta_t = _step(_DIR)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_dash_cells(t0, s, ref_span) is None for s in s_vals):
            continue
        t1 = t0 + (ref_span - 1) * delta_t
        start_caps = [_max_span_from(t0, s) for s in s_vals]
        if any(start_caps[i] >= ref_span + 2 for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_span_to(t1, s) for s in s_vals]
        if any(end_caps[i] >= ref_span + 2 for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    return tuple(random.sample(_COLORS, n))


def _odd_span_options(lo: int, hi: int) -> list[int]:
    lo = max(lo, _MIN_SPAN)
    if lo % 2 == 0:
        lo += 1
    if hi < lo:
        return []
    return list(range(lo, hi + 1, 2))


def _choose_spans(ref_span: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    spans = [ref_span] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] >= ref_span + 2]
    if len(nonrefs) < 2 or not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])

    long_opts = _odd_span_options(ref_span + 2, min(_MAX_SEG_SPAN, caps[long_i]))
    short_opts = _odd_span_options(_MIN_SPAN, ref_span - 2)
    if not long_opts or not short_opts:
        return None
    spans[long_i] = random.choice(long_opts)
    spans[short_i] = random.choice(short_opts)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = _odd_span_options(_MIN_SPAN, min(_MAX_SEG_SPAN, caps[i]))
        if not options:
            return None
        if len(options) > 1 and ref_span in options and random.random() < _SAME_SPAN_AVOID_PROB:
            options = [span for span in options if span != ref_span]
        spans[i] = random.choice(options)
    return spans


def _paint(
    grid: np.ndarray,
    s_vals: tuple[int, ...],
    starts: list[int],
    spans: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, span, color in zip(s_vals, starts, spans, colors):
        cells = _dash_cells(t0, s, span)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_SEGMENT_COUNTS)
    ref_span = random.choice(_REF_SPANS)
    s_vals = _choose_offsets(n)
    placements = _feasible_placements(s_vals, ref_span)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    spans = _choose_spans(ref_span, caps, n)
    if spans is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, s_vals, [t0] * n, [ref_span] * n, colors):
        return None

    delta_t = _step(_DIR)
    t1 = t0 + (ref_span - 1) * delta_t
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (span - 1) * delta_t for span in spans]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, s_vals, starts, spans, colors):
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
