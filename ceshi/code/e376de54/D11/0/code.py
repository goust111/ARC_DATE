# Variant of e376de54
# Dimension: D11 — 线段的连续性与断裂 (meta)
# Change: 每条线段是连续的像素链（对角线每像素步进1行1列，水平/垂直每像素步进1格）... → 线段带间隔（虚线模式）

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D11 variant
# a) Pixel concretization and behavior trace:
#    Each object is still a single-color segment on one of the four original directions, but it is no longer
#    a fully connected run. Instead, a painted cell is followed by exactly one background cell, then the next
#    painted cell, and so on. To keep both endpoints visually visible, every generated segment uses an odd
#    total span: 3, 5, 7, 9, or 11 positions along the underlying continuous line. Example on a horizontal
#    row: original segment `22222` becomes `2 2 2` (span 5, only 3 colored cells). In the original solver,
#    the reference length is recovered from the number of connected cells. In the variant, the reference span
#    must be recovered from the first and last painted cells and interpreted as 5 rather than 3. The input
#    and output therefore keep the same "align all segments to the median-offset reference" rule, but both
#    states preserve the dash cadence instead of drawing solid lines.
# b) Complexity calibration:
#    The dash cadence is the only new concept. Other task structure remains stable: same 16x16 board, same
#    four directions, same odd number of parallel segments, same median perpendicular-offset reference, same
#    start-anchor or end-anchor family. Odd spans make endpoints visible, so the new length definition is
#    inferable from a few examples without adding extra subcases.
# c) Implicit-assumption audit against the original main():
#    - Original direction detection counts adjacent same-color cells; that fails because dashed segments have
#      no adjacent same-color neighbors.
#    - Original segment tracing walks by one step while the same-color mask continues; that would split every
#      dashed segment into isolated singletons.
#    - Original `len(cells)` equals geometric length; in the variant it undercounts by roughly half.
#    - Original redraw paints every step continuously; in the variant redraw must paint every other step only.
#    The new solver replaces all four assumptions with "same color reappears after one background gap".
# d) Placement legality:
#    Generator still uses output-first construction. It places all reference-aligned dashed segments first,
#    then derives the input by keeping either the start or the end fixed and varying only the total span.
#    Parallel offsets remain distinct and irregular, all occupied cells stay in-bounds, and the median offset
#    still identifies a unique reference. Distinct colors are used per segment as a targeted adaptation so
#    wrong directions cannot accidentally form same-color two-step chains after continuity is broken.
# e) Visual observability:
#    With odd spans, both visible endpoints move exactly as in the original task, while the interior now has
#    regular one-cell holes. A human can compare input/output and see that the dash pattern is preserved and
#    the alignment operates on the full endpoint-to-endpoint span rather than on the count of painted cells.
# f) Visual cleanliness:
#    Objects remain parallel, non-overlapping, and separated by the original gap constraints, so the board is
#    sparse and legible despite the extra holes.
# g) Color inferability:
#    No color mapping is introduced. Colors are purely identity markers for separate segments, which keeps the
#    new abstraction focused on dashed geometry instead of adding unrelated color logic.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


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


def _extract_segments(
    g: np.ndarray, bg: int, d: tuple[int, int]
) -> tuple[list[dict], set[tuple[int, int]]]:
    dr, dc = d
    delta_t = _step(d)
    segments = []
    used = set()
    h, w = g.shape

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
            segments.append(
                {
                    "color": color,
                    "cells": cells,
                    "start": start,
                    "end": end,
                    "t_start": t_start,
                    "t_end": t_end,
                    "s": _perp_dot(start, d),
                    "occ_len": len(cells),
                    "span": span,
                }
            )
            for cell in cells:
                used.add(cell)

    return segments, used


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[tuple[int, int], list[dict]]:
    non_bg_cells = {
        (r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) != bg
    }
    scored = []
    for d in _DIR_CANDIDATES:
        segments, used = _extract_segments(g, bg, d)
        score = (len(used), len(segments), sum(s["occ_len"] for s in segments))
        scored.append((score, d, segments))

    best_score = max(score for score, _, _ in scored)
    best = [(d, segments) for score, d, segments in scored if score == best_score]
    assert best_score[0] > 0, "main direction must be detectable"
    assert len(best) == 1, "main direction must be unique"
    d, segments = best[0]
    assert best_score[0] == len(non_bg_cells), "all colored cells must belong to dashed segments"
    return d, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel dashed line segments so they all inherit the reference segment's full span.

    Input structure:
    - bg is the most common color.
    - Several same-direction, parallel segments appear as dashed runs: one colored cell, one background gap,
      then the next colored cell, in a horizontal, vertical, or diagonal direction.

    Rule:
    - Detect the shared direction by finding which orientation explains all non-background cells as dashed
      segments.
    - Use the segment at the median perpendicular offset as the reference.
    - Shift every segment so its visible start matches the reference start, then redraw it with the same total
      endpoint-to-endpoint span as the reference while preserving the one-cell dash gap pattern.
    - The grid size stays fixed, and stray non-segment pixels are left untouched.

    Key insight: A dashed segment's length is not its number of painted cells; it is the full geometric span
    from first visible endpoint to last visible endpoint, including the background holes between them.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d, segments = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)

    assert segments, "at least one dashed multi-cell segment must exist"

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
    target_span = ref["span"]
    assert target_span >= 3 and target_span % 2 == 1, "reference dashed span must be odd and visible"

    out = g.copy()

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    occ_target = (target_span + 1) // 2
    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to the base step"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc

        rr, cc = r0, c0
        for _ in range(occ_target):
            assert 0 <= rr < h and 0 <= cc < w, "shifted dashed segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += 2 * dr
            cc += 2 * dc

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Segment counts and reference span ---
_SEGMENT_COUNTS = [3, 5]           # must be odd for unique median
_REF_SPANS = [5, 7, 9]             # odd geometric spans for clearly visible dashed endpoints
_MIN_SPAN = 3
_MAX_SEG_SPAN = 11

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Span selection ---
_SAME_SPAN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


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


def _dash_cells(
    d: tuple[int, int], t0: int, s: int, span: int
) -> list[tuple[int, int]] | None:
    assert span >= _MIN_SPAN and span % 2 == 1
    delta = _step(d)
    cells = []
    for offset in range(0, span, 2):
        cell = _cell_from_ts(d, t0 + offset * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_span_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    occ = 0
    while _cell_from_ts(d, t0 + 2 * occ * delta, s) is not None:
        occ += 1
    return 2 * occ - 1 if occ else 0


def _max_span_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    occ = 0
    while _cell_from_ts(d, t1 - 2 * occ * delta, s) is not None:
        occ += 1
    return 2 * occ - 1 if occ else 0


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


def _feasible_placements(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_span: int
) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_dash_cells(d, t0, s, ref_span) is None for s in s_vals):
            continue
        t1 = t0 + (ref_span - 1) * delta
        start_caps = [_max_span_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_span for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_span_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_span for i in range(len(s_vals)) if i != ref_idx):
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
    if not longable or len(nonrefs) < 2:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])

    spans[long_i] = random.choice(_odd_span_options(ref_span + 2, min(_MAX_SEG_SPAN, caps[long_i])))
    short_opts = _odd_span_options(_MIN_SPAN, ref_span - 2)
    if not short_opts:
        return None
    spans[short_i] = random.choice(short_opts)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = _odd_span_options(_MIN_SPAN, min(_MAX_SEG_SPAN, caps[i]))
        if not options:
            return None
        if len(options) > 1 and ref_span in options and random.random() < _SAME_SPAN_AVOID_PROB:
            options = [x for x in options if x != ref_span]
        spans[i] = random.choice(options)
    return spans


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    spans: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, span, color in zip(s_vals, starts, spans, colors):
        cells = _dash_cells(d, t0, s, span)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_span = random.choice(_REF_SPANS)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_span)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    spans = _choose_spans(ref_span, caps, n)
    if spans is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_span] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_span - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (span - 1) * delta for span in spans]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, spans, colors):
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
