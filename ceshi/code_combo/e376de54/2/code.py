# Combo variant of e376de54 (index 2)
# combo_spec: D1/1+D4/1
# Name: 最短基准+仅裁剪

from pathlib import Path
import sys
import random

import numpy as np

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        _solvers_dir = str(_parent / "solvers")
        if _solvers_dir not in sys.path:
            sys.path.insert(0, _solvers_dir)
        break

from arc_utils import get_bg


"""
Design analysis for the combo variant D1/1 + D4/1

a) Concrete behavior and trace
   - The visible objects stay in the same family as the source task: a 16x16 background with 3 or 5
     parallel monochrome segments on distinct offsets, all sharing one of four directions.
   - The fused rule changes two linked steps at once:
       1. reference selection changes from "median offset" to "unique shortest segment";
       2. normalization changes from "extend or crop" to "crop only".
   - Because the reference itself is the shortest, the second change becomes operationally simple:
     every other segment is longer, so every rewritten segment is clipped to the shortest span.
   - Small trace, horizontal case:
       input lengths by offset = [8, 5, 10], with a common right endpoint.
       original task: pick the middle offset line (length 5 only if it happens to be median by offset),
       align starts to that line, then extend/crop to its length.
       fused variant: find the unique shortest line by length, here 5, align every segment's start to
       that line's start, and keep only 5 cells. The 8-cell and 10-cell lines both contract to the same
       5-cell span; no line is ever padded.
   - The visually important difference from the original task is that the decisive cue is minimum length,
     not position among offsets. To make that visible, the generator keeps the shortest segment away from
     the median offset line.

b) Complexity calibration
   - I keep the original direction set, odd segment counts, irregular offset gaps, and 1-3 colors so the
     puzzle still looks like the same ARC family.
   - The free variables are limited to direction, segment count, anchor side, shortest-line index, target
     shortest length, and the longer lengths of the remaining segments. That is enough variation to avoid
     triviality while keeping the rule legible from a few examples.
   - Every generated sample has exactly one shortest segment and at least two longer ones, so the "crop to
     the shortest span" effect is always visible rather than occasionally collapsing into equality cases.

c) Hidden assumptions reviewed against main()
   - `_pick_main_dir` assumes same-color adjacency is strongest along one shared direction. The generator
     preserves that by drawing only proper segments and spacing parallel lines apart.
   - Segment tracing assumes each segment has a unique predecessor-free start along the main direction.
     That remains true because each object is a single contiguous run on its own offset.
   - The original reference-by-median assumption is invalid under the combo; the solver removes it and now
     requires exactly one minimum-length segment instead.
   - The redraw loop assumes the reference span is legal on every offset after start alignment. The
     generator builds that shortest span first and only lengthens non-reference segments away from the
     shared anchor, so replaying the shortest span is always in bounds.
   - The fused "crop only" rule would be ambiguous if some other segment were shorter than the reference.
     The generator forbids that by making the reference uniquely shortest.

d) Placement legality
   - Output-first construction chooses a shortest span that already fits on every chosen offset.
   - Inputs are derived by extending non-reference segments either forward from a common start or backward
     from a common end, so they stay collinear, non-overlapping, and in bounds.
   - Orthogonal gaps stay in 2..6 and diagonal gaps stay in {4, 6}, preventing neighboring lines from
     touching or visually merging.

e) Visual observability
   - A human can compare 2-4 examples and see one segment is uniquely shortest in each input and every
     output line contracts to that shortest extent.
   - The shortest segment is not color-coded and not fixed to the middle offset, so the rule must be read
     from geometry rather than invisible metadata.
   - The generator explicitly avoids the degenerate case where the shortest segment is also the original
     median-offset reference, because that would make the combo harder to distinguish from the source task.

f) Visual cleanliness
   - No auxiliary markers are added. The scene remains just a sparse set of separated line segments on a
     plain background.
   - Distinct offsets and irregular spacing keep each segment visually isolated even when colors repeat.

g) Change inferability
   - Colors are preserved exactly; there is no color remapping. The only changing attribute is geometric
     extent along the common direction, which is directly visible from the input/output pairs.
   - Since all non-reference segments are longer than the reference, the one-sided "crop only" rule is not
     hidden behind mixed short/long cases. Humans see a consistent contraction pattern.

Post-change solver assumptions guaranteed by generate_input()
   - At least one multi-cell segment exists.
   - All segment offsets are distinct, with diagonal gaps large enough for clean separation.
   - Exactly one segment has the minimum length.
   - The unique shortest segment is not the median-offset segment, so the combo does not collapse back to
     the original reference-selection behavior on generated samples.
   - Every non-reference segment is strictly longer than the shortest one, making the transformation purely
     cropping and ensuring every output segment ends up at the same shortest length.
"""

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-align parallel colored segments to the span of the unique shortest segment, trimming only longer lines.

    Input structure:
    - The background is the most common color.
    - Several horizontal, vertical, or diagonal monochrome segments lie on distinct parallel offsets and share one dominant direction.
    - Exactly one multi-cell segment is the shortest; the other segments are at least as long and may sit at different offsets or anchors.

    Rule:
    - Detect the shared direction and trace every multi-cell segment along it.
    - Choose the unique shortest segment as the reference, using its start and end coordinates as the target span.
    - Shift each segment so its start aligns with the reference start, then keep only the cells that fit inside the reference length.
    - Do not extend any segment, keep the canvas size unchanged, and leave isolated single pixels untouched.

    Key insight: The fusion hinges on two coupled cues: the shortest line defines the common span, and normalization is one-sided, so every output segment is a cropped copy of that shortest extent.
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

    min_len = min(s["len"] for s in segments)
    ref_cands = [s for s in segments if s["len"] == min_len]
    assert len(ref_cands) == 1, "shortest reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

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
        draw_len = min(s["len"], target_len)
        rr, cc = r0, c0
        for _ in range(draw_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (2, 6)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3

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
    return [
        p
        for p in patterns
        if sum(p) <= max_span and (n == 3 and p[0] != p[1] or n == 5 and len(set(p)) > 1)
    ]


def _choose_offsets(d, n):
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, ref_len, ref_idx):
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if all(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if all(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, ref_idx):
    lengths = []
    for i, cap in enumerate(caps):
        if i == ref_idx:
            lengths.append(ref_len)
            continue
        hi = min(_MAX_SEG_LEN, cap)
        if hi <= ref_len:
            return None
        lengths.append(random.randint(ref_len + 1, hi))
    assert sum(length == ref_len for length in lengths) == 1
    return lengths


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
    median_idx = n // 2
    ref_idx = random.choice([i for i in range(n) if i != median_idx])
    placements = _feasible_placements(d, s_vals, ref_len, ref_idx)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, ref_idx)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    starts = []
    for length in lengths:
        if anchor == "start":
            starts.append(t0)
        else:
            starts.append(t1 - (length - 1) * delta)

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
