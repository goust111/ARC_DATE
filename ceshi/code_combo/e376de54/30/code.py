# Combo variant of e376de54 (index 30)
# combo_spec: D2/3+D11/0+D13/0
# Name: 反对角线+虚线+阶梯排列

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the combo variant
# a) Pixel concretization and behavior trace:
#    The original task may contain solid horizontal / vertical / diagonal segments that share one endpoint.
#    This combo fixes the family to anti-diagonal segments only, changes every segment into a dashed run,
#    and removes the shared-start/shared-end anchor in the input. Concretely, each object lies on a `/`
#    anti-diagonal and is painted as `color, bg, color, bg, ...` along that line. Example in a 9x9 mental
#    sketch, using anti-diagonal direction from lower-left to upper-right:
#      input spans on three parallel lines:
#        line A start at t=-4, span 3  -> painted endpoints two steps apart
#        line B start at t=0,  span 7  -> the median-offset reference
#        line C start at t=4,  span 5
#      Their visible lower-left endpoints form a staircase because A, B, C start at different anti-diagonal
#      coordinates. The output keeps the same three lines and colors, but every segment is redrawn to start
#      at B's lower-left endpoint coordinate on its own line and to occupy B's full endpoint-to-endpoint span.
#    Relative to the original logic, three things change at once: the orientation is inverted to `/`, the
#    segment span must be inferred through gaps rather than adjacency, and the alignment target is no longer
#    recoverable from a shared endpoint pattern in the input.
# b) Complexity calibration:
#    The combo is visually strong but still learnable. Direction has only one family (`/` anti-diagonal),
#    dash cadence is fixed at exactly one background hole between visible cells, and staircase offsets come
#    from a small catalog of monotone patterns centered on the reference line. That keeps the task from
#    exploding into too many subcases while making the difference from the source task obvious by eye.
# c) Implicit-assumption audit against the original main():
#    - Adjacent same-color counting no longer detects direction, because dashed segments never have immediate
#      same-color neighbors.
#    - Walking cell-by-cell along the line would split each dashed segment into isolated pixels.
#    - Allowing all four directions is now wrong: this combo fixes the geometry to the anti-diagonal family.
#    - Assuming all segments share `t_start` or `t_end` is invalid: staircase inputs intentionally break that.
#    - Using `len(cells)` as the line length undercounts the true dashed span, which must include holes.
#    The replacement solver therefore parses only `/`-oriented dashed chains, measures full geometric span
#    from first visible cell to last visible cell, and aligns every segment to the reference's start without
#    relying on any shared input anchor.
# d) Placement legality:
#    The generator remains output-first. It first places the output as aligned dashed anti-diagonal segments
#    with a common reference start and span. It then derives the input by assigning each line a distinct
#    staircase start shift and a valid odd span. Parallel anti-diagonals with distinct `s=r+c` values never
#    intersect, and every candidate is accepted only if all visible cells stay inside the 16x16 board.
# e) Visual observability:
#    Humans can see all three fused changes directly. The objects are clearly `/` slanted, every segment has
#    regular one-cell holes, and the lower-left endpoints are visibly stepped rather than aligned. Across a
#    few examples, the output regularizes those stepped starts into one shared start while preserving both
#    color identity and the dashed cadence, making the rule inferable from 2-4 pairs.
# f) Visual cleanliness:
#    Distinct perpendicular offsets keep the anti-diagonals separated. The dash pattern reduces clutter, and
#    using distinct colors per segment avoids accidental same-color pseudo-links in the wrong orientation.
#    Gap sizes remain in a narrow range so the board stays sparse and legible.
# g) Change inferability:
#    No color remapping is introduced. Colors only label which input segment becomes which output segment.
#    The inferred action is purely geometric: find the median-offset dashed anti-diagonal, then snap every
#    other dashed anti-diagonal to its start and span.
# Post-edit assumption audit:
#    - `_extract_segments` assumes every meaningful colored cell belongs to a `/` dashed chain of length >= 2
#      visible cells; the generator guarantees this by constructing only such chains.
#    - Reference selection assumes an odd number of unique anti-diagonal offsets; the generator uses 3 or 5
#      distinct offsets with diagonal gaps >= 4.
#    - Redraw assumes the reference span is odd and at least 5 so a shorter dashed span also exists; the
#      generator restricts reference spans to {5, 7, 9}.
#    - Staircase evidence is not required by the solver, but the generator guarantees all non-reference
#      starts are distinct and monotone in offset so the fused behavior is always present in samples.

_DIR = (-1, 1)  # anti-diagonal, lower-left to upper-right
_STEP = 2

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Segment colors ---
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Segment counts and spans ---
_SEGMENT_COUNTS = [3, 5]  # odd for unique median offset
_REF_SPANS_BY_COUNT = {
    3: [5, 7, 9],
    5: [5],
}
_MIN_SPAN = 3
_MAX_SEG_SPAN = 11

# --- Offset constraints ---
_PERP_GAP_DIAG = [4, 6]
_STAIR_PATTERNS = {
    3: [(-2, 0, 2), (-3, 0, 3), (-3, 0, 2), (-2, 0, 3)],
    5: [
        (-2, -1, 0, 1, 2),
    ],
}

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _cell_from_ts(t: int, s: int) -> tuple[int, int] | None:
    # For d=(-1,1): t = c-r, s = r+c => r=(s-t)/2, c=(s+t)/2
    if (s - t) % 2 or (s + t) % 2:
        return None
    r = (s - t) // 2
    c = (s + t) // 2
    if 0 <= r < _H and 0 <= c < _W:
        return r, c
    return None


def _dash_cells(t0: int, s: int, span: int) -> list[tuple[int, int]] | None:
    assert span >= _MIN_SPAN and span % 2 == 1
    cells = []
    for offset in range(0, span, 2):
        cell = _cell_from_ts(t0 + offset * _STEP, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_span_from(t0: int, s: int) -> int:
    occ = 0
    while _cell_from_ts(t0 + 2 * occ * _STEP, s) is not None:
        occ += 1
    return 2 * occ - 1 if occ else 0


def _has_dash_link(
    g: np.ndarray, bg: int, r: int, c: int, d: tuple[int, int], color: int
) -> bool:
    dr, dc = d
    mr, mc = r + dr, c + dc
    nr, nc = r + 2 * dr, c + 2 * dc
    if not (_in_bounds(g.shape, mr, mc) and _in_bounds(g.shape, nr, nc)):
        return False
    return int(g[mr, mc]) == bg and int(g[nr, nc]) == color


def _trace_dashed_segment(
    g: np.ndarray, bg: int, r: int, c: int
) -> list[tuple[int, int]]:
    color = int(g[r, c])
    cells = [(r, c)]
    rr, cc = r, c
    while _has_dash_link(g, bg, rr, cc, _DIR, color):
        rr += 2 * _DIR[0]
        cc += 2 * _DIR[1]
        cells.append((rr, cc))
    return cells


def _extract_segments(
    g: np.ndarray, bg: int
) -> tuple[list[dict], set[tuple[int, int]]]:
    segments = []
    used = set()
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg:
                continue
            if _has_dash_link(g, bg, r, c, (-_DIR[0], -_DIR[1]), color):
                continue

            cells = _trace_dashed_segment(g, bg, r, c)
            if len(cells) < 2:
                continue

            start = cells[0]
            end = cells[-1]
            t_start = _dot(start, _DIR)
            t_end = _dot(end, _DIR)
            span = (t_end - t_start) // _STEP + 1
            segments.append(
                {
                    "color": color,
                    "cells": cells,
                    "start": start,
                    "t_start": t_start,
                    "t_end": t_end,
                    "s": _perp_dot(start, _DIR),
                    "occ_len": len(cells),
                    "span": span,
                }
            )
            used.update(cells)

    return segments, used


def _segment_summary(g: np.ndarray) -> list[dict]:
    bg = get_bg(g)
    segments, used = _extract_segments(g, bg)
    non_bg = {
        (r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) != bg
    }
    assert used == non_bg, "all colored cells must belong to anti-diagonal dashed segments"
    s_vals = [seg["s"] for seg in segments]
    assert len(segments) in _SEGMENT_COUNTS, "segment count must be 3 or 5"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_sorted = sorted(s_vals)
    for i in range(len(s_sorted) - 1):
        assert s_sorted[i + 1] - s_sorted[i] >= 4, "anti-diagonal offsets must be well separated"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize stepped anti-diagonal dashed segments to the median-offset segment's start and span.

    Input structure:
    - The background is the most frequent color.
    - Several single-color `/` anti-diagonal segments appear as dashed runs: one colored cell, one gap, then
      the next colored cell. Their parallel offsets are distinct, and their visible lower-left starts are
      staggered into a staircase rather than sharing one common start or end.

    Rule:
    - Parse every colored object as a dashed anti-diagonal segment and order the segments by perpendicular
      offset (`r+c`).
    - Use the segment at the median offset as the reference.
    - Erase the original dashed segments, then redraw every segment on its own offset line so that its start
      matches the reference start and its total geometric span matches the reference span.
    - Keep the one-gap dash cadence unchanged and leave the board size fixed.

    Key insight: The objects are compared by their full endpoint-to-endpoint anti-diagonal span, not by the
    number of painted cells. The staircase only changes where segments begin; the output snaps all starts to
    the median-offset segment while preserving the dashed anti-diagonal form.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    segments = _segment_summary(g)

    s_vals = sorted(seg["s"] for seg in segments)
    s_ref = s_vals[len(s_vals) // 2]
    ref = [seg for seg in segments if seg["s"] == s_ref][0]
    ref_t0 = ref["t_start"]
    target_span = ref["span"]
    assert target_span >= 3 and target_span % 2 == 1, "reference dashed span must be odd"

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg

    for seg in segments:
        cells = _dash_cells(ref_t0, seg["s"], target_span)
        assert cells is not None, "redrawn segment must stay in bounds"
        for r, c in cells:
            assert 0 <= r < h and 0 <= c < w
            out[r, c] = seg["color"]

    return out


def _s_limits() -> tuple[int, int]:
    return 0, 30


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


def _feasible_layouts(
    s_vals: tuple[int, ...], ref_span: int, stair_steps: tuple[int, ...]
) -> list[tuple[int, list[int]]]:
    ref_idx = len(s_vals) // 2
    assert stair_steps[ref_idx] == 0
    layouts = []
    for ref_t0 in range(*_T_SCAN_RANGE):
        if any(_dash_cells(ref_t0, s, ref_span) is None for s in s_vals):
            continue
        caps = []
        ok = True
        for s, shift in zip(s_vals, stair_steps):
            start_t = ref_t0 + shift * _STEP
            cap = _max_span_from(start_t, s)
            if cap < _MIN_SPAN:
                ok = False
                break
            caps.append(cap)
        if not ok:
            continue
        if any(caps[i] >= ref_span + 2 for i in range(len(s_vals)) if i != ref_idx):
            layouts.append((ref_t0, caps))
    return layouts


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
    if not longable:
        return None

    short_opts = _odd_span_options(_MIN_SPAN, ref_span - 2)
    if not short_opts:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    spans[long_i] = random.choice(_odd_span_options(ref_span + 2, min(_MAX_SEG_SPAN, caps[long_i])))
    spans[short_i] = random.choice(short_opts)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = _odd_span_options(_MIN_SPAN, min(_MAX_SEG_SPAN, caps[i]))
        if not options:
            return None
        if len(options) > 1 and ref_span in options:
            options = [x for x in options if x != ref_span] or options
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
    ref_span = random.choice(_REF_SPANS_BY_COUNT[n])
    s_vals = _choose_offsets(n)
    stair_steps = random.choice(_STAIR_PATTERNS[n])
    layouts = _feasible_layouts(s_vals, ref_span, stair_steps)
    if not layouts:
        return None

    ref_t0, caps = random.choice(layouts)
    spans = _choose_spans(ref_span, caps, n)
    if spans is None:
        return None

    colors = tuple(random.sample(_COLORS, n))

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, s_vals, [ref_t0] * n, [ref_span] * n, colors):
        return None

    starts = [ref_t0 + shift * _STEP for shift in stair_steps]
    if len(set(starts)) != n:
        return None
    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, s_vals, starts, spans, colors):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
