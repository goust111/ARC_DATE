# Variant of e376de54
# Dimension: D1 — 匹配条件：以第二长线段为参考基准 vs 其他选取规则 (match_cond)
# Change: 以第二长线段的端点范围为参考，所有其他线段的端点对齐到该参考线段的起止位置... → 以特定颜色的线段为参考

import random

import numpy as np

from arc_utils import get_bg

# Design analysis
# a) Concrete behavior and trace:
#    The objects stay the same as the seed task: 3 or 5 straight parallel segments on a 16x16
#    grid, all sharing one direction and one anchored endpoint, with distinct perpendicular
#    offsets and varied lengths. The visible change is that one segment now always uses a fixed
#    reference color (`_REF_COLOR`), and that segment alone dictates the target span in the
#    output. Example with horizontal lines:
#    - old input lengths at offsets top/mid/bot: [6, 3, 8], all sharing the same left end
#    - old solver: pick the middle offset line, so every output line becomes length 3
#    - new solver: if the top line has the reference color, every output line becomes length 6
#    The geometry is unchanged; only the condition that chooses the controlling segment changes.
#
# b) Complexity calibration:
#    The reference color is fixed across all generated samples and appears on exactly one
#    multi-cell segment. This is strong enough to be human-visible across 2-4 examples, but it
#    does not add extra markers or secondary objects. Non-reference segments still reuse 1-2
#    other colors, so the task remains about recognizing one special color rather than memorizing
#    many one-off color-role pairs.
#
# c) Hidden assumptions review against the seed `main()`:
#    - The seed code assumed a reference chosen from perpendicular-order statistics, so it needed
#      an odd segment count and a unique median offset. Those assumptions are no longer part of
#      the reasoning rule, so the new solver removes the median-selection asserts.
#    - The tracing logic by color remains valid even when non-reference colors repeat, because
#      segments are still separated by offset gaps and are traced independently from their starts.
#    - Clearing then redrawing remains safe because segments occupy distinct parallel lines; the
#      generator keeps those offsets unique and non-overlapping.
#    - A new assumption is introduced: exactly one multi-cell segment must have the fixed
#      reference color. The generator enforces this directly.
#
# d) Placement legality:
#    The generator still uses output-first construction. It first places the desired output where
#    every segment spans the reference-color segment's extent. It then derives an input by keeping
#    a common start or common end while changing non-reference lengths. All cells are checked for
#    in-bounds placement in `(t, s)` coordinates before painting.
#
# e) Visual observability:
#    The special behavior is visible because the same color controls the output in every sample.
#    To prevent accidental regression to the seed task, the generator never puts the reference-
#    color segment on the median offset, and it forces the median-offset segment to have a
#    different length from the reference segment. Therefore the old and new rules produce
#    different outputs on every accepted sample.
#
# f) Visual cleanliness:
#    Offsets stay distinct with the same gap rules as the seed task, so segments do not touch or
#    overlap. No extra labels are added; the only extra cue is the consistent reference color.
#
# g) Color inference:
#    No invisible color-id ordering is used. The rule is purely visual: the uniquely recurring
#    reference color across training examples determines which line controls the alignment span.
#
# 9) New solver assumptions checked against the generator:
#    - `_pick_main_dir` assumes one direction is dominant: guaranteed because every multi-cell
#      object is a segment in the same direction.
#    - `_extract_segments` assumes each segment has a detectable start along that direction:
#      guaranteed by non-overlapping straight runs.
#    - Reference selection assumes exactly one multi-cell segment of `_REF_COLOR`: guaranteed by
#      `_choose_colors`.
#    - Redrawing assumes the reference span fits for all offsets: guaranteed by the output-first
#      feasibility search.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_REF_COLOR = 2
_NONREF_COLORS = [1, 3, 5, 6, 8, 9]

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300
_SAME_LEN_AVOID_PROB = 0.75


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
    Align every parallel segment to the span of a uniquely colored reference segment.

    Input structure:
    - The background is the most frequent color.
    - Several straight parallel segments appear in one dominant direction, each lying on its own offset line.
    - Exactly one multi-cell segment uses a fixed reference color; the others use non-background colors and may share colors.
    - The segments already share one endpoint directionally, but their lengths differ.

    Rule:
    - Detect the common direction and trace every multi-cell segment in that direction.
    - Find the single segment drawn in the fixed reference color.
    - Read that segment's directional start and end as the target span.
    - Clear the original segments and redraw all of them in place-offset order so they cover exactly that same span; isolated single pixels are left alone.

    Key insight:
    - The controlling segment is identified by color, not by geometric rank, so the special-colored line defines the final extent for every segment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
    dr, dc = d

    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4

    ref_cands = [s for s in segments if s["color"] == _REF_COLOR]
    assert len(ref_cands) == 1, "reference color segment must be unique"
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
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


def _cell_from_ts(d: tuple[int, int], t: int, s: int):
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


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int):
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
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
        if sum(p) <= max_span and ((n == 3 and p[0] != p[1]) or (n == 5 and len(set(p)) > 1))
    ]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _choose_ref_index(n: int) -> int:
    median_idx = n // 2
    return random.choice([i for i in range(n) if i != median_idx])


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int, ref_idx: int):
    nonrefs = [i for i in range(len(s_vals)) if i != ref_idx]
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * _step(d)
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in nonrefs):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in nonrefs):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n: int, ref_idx: int) -> tuple[int, ...]:
    n_extra = random.randint(1, min(2, n - 1))
    palette = random.sample(_NONREF_COLORS, n_extra)
    nonref_positions = [i for i in range(n) if i != ref_idx]
    colors = [None] * n
    colors[ref_idx] = _REF_COLOR
    for pos, col in zip(nonref_positions, palette):
        colors[pos] = col
    for pos in nonref_positions[len(palette):]:
        colors[pos] = random.choice(palette)
    return tuple(int(c) for c in colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int, ref_idx: int, median_idx: int):
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    if median_idx in longable and (random.random() < 0.5 or len(longable) == 1):
        long_i = median_idx
        short_i = random.choice([i for i in nonrefs if i != median_idx])
    else:
        short_i = median_idx
        long_choices = [i for i in longable if i != median_idx]
        if not long_choices:
            return None
        long_i = random.choice(long_choices)

    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)

    assert lengths[median_idx] != ref_len
    return lengths


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals, starts, lengths, colors) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _render_output_from_reference(d: tuple[int, int], s_vals, starts, lengths, colors, ref_idx: int):
    out = np.full((_H, _W), _BG, dtype=int)
    ref_t0 = starts[ref_idx]
    target_len = lengths[ref_idx]
    if not _paint(out, d, s_vals, [ref_t0] * len(s_vals), [target_len] * len(s_vals), colors):
        return None
    return out


def _gen_attempt():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    median_idx = n // 2
    ref_idx = _choose_ref_index(n)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len, ref_idx)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n, ref_idx)
    lengths = _choose_lengths(ref_len, caps, n, ref_idx, median_idx)
    if lengths is None:
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    out = _render_output_from_reference(d, s_vals, starts, lengths, colors, ref_idx)
    if out is None:
        return None

    legacy_out = _render_output_from_reference(d, s_vals, starts, lengths, colors, median_idx)
    if legacy_out is None or np.array_equal(out, legacy_out):
        return None

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
