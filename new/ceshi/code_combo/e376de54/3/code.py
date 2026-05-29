# Combo variant of e376de54 (index 3)
# combo_spec: D1/4+D5/2
# Name: 颜色决定参考+颜色排序

import random

import numpy as np

from arc_utils import get_bg

# Parent review patch:
# The first generated version picked the reference by hidden ARC color-id order, which fails the
# ARC-solvability audit. This repaired version keeps the "color decides the reference" idea but
# makes the color ranking explicit via a visible isolated legend row in the input/output.

# Design analysis
# a) Concrete behavior and trace:
#    The visible objects remain the seed task's parallel straight segments on a 16x16 grid.
#    What changes is how the controlling segment is chosen. In the seed, the reference came
#    from geometric order: the segment at the median perpendicular offset. In this combo, the
#    reference comes from color order: sort the segment colors by ARC color id and pick the
#    segment whose color is the middle value. Example with three horizontal segments whose left
#    ends already align:
#    - offsets top/mid/bot use colors [6, 1, 9] and lengths [8, 5, 3]
#    - seed rule: middle offset wins, so the color-1 segment sets target length 5
#    - fused rule: sorted colors are [1, 6, 9], so the color-6 segment wins and target length
#      becomes 8 even though it is not the middle offset line
#    The output action is otherwise unchanged: every segment is shifted so its start matches the
#    reference start, then it is trimmed/extended to the reference span.
#
# b) Complexity calibration:
#    Repeated colors would make "the middle color segment" ambiguous, so this variant uses one
#    distinct color per segment. The segment count stays odd (3 or 5), which makes the median
#    color unique. The palette stays small and fixed ([1,2,3,5,6,8,9]) so the task is still
#    visually clean and structurally close to the seed rather than turning into arbitrary noise.
#
# c) Hidden assumptions review against the seed `main()`:
#    - The seed's reference-selection block relied on median perpendicular offset; that is no
#      longer a valid cue and is replaced with median color-id selection.
#    - The seed generator allowed color reuse across segments. That breaks the new rule, so the
#      fused generator now guarantees all multi-cell segments have distinct colors.
#    - Clearing and redrawing still assumes distinct parallel support lines; the offset-spacing
#      constraints from the seed are kept unchanged to preserve that safety.
#    - The solver now assumes an odd number of multi-cell segments and an odd number of distinct
#      segment colors. The generator enforces both by construction.
#
# d) Placement legality:
#    Generation remains output-first. First choose offsets, a non-median-offset reference index,
#    and a distinct color set whose median numeric color is placed on that reference index. Then
#    build the output where every segment already occupies the reference span. Finally derive the
#    input by keeping a shared start or shared end while varying segment lengths. Every candidate
#    placement is checked in (t,s) coordinates for in-bounds legality before painting.
#
# e) Visual observability:
#    The intended difference from the seed is that geometry no longer identifies the reference.
#    To make this non-degenerate, the generator never places the median-color segment on the
#    median offset, and it forces the median-offset segment to have a different span from the
#    median-color segment. Therefore the fused output always differs from the original rule on
#    accepted samples.
#
# f) Visual cleanliness:
#    Segments still occupy distinct parallel lines with irregular gaps, exactly as in the seed,
#    so no two lines touch or overlap. The output only changes extents along those existing lines,
#    which keeps the puzzles easy to parse by eye.
#
# g) Change inferability:
#    This combo deliberately moves the ranking signal from geometry to color ids. Because that
#    would be ambiguous with duplicate colors, the generator uses unique visible colors only and
#    keeps the transformation limited to span alignment. The only question the solver must answer
#    is which existing colored segment is the numeric middle one.
#
# 9) New solver assumptions checked against the generator:
#    - `_pick_main_dir` needs one clearly dominant direction: guaranteed because every multi-cell
#      object is a segment in the same direction.
#    - `_extract_segments` assumes segments can be traced independently along that direction:
#      guaranteed by distinct offsets and the seed gap rules.
#    - `_pick_reference_segment` assumes an odd number of uniquely colored segments: guaranteed
#      by `_choose_colors`.
#    - Redrawing assumes the reference span fits on every offset line: guaranteed by the
#      output-first feasibility search and verified again by `_paint`.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]

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

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300
_SAME_LEN_AVOID_PROB = 0.75
_LEGEND_SPACING = 2


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
    for col in sorted(int(c) for c in np.unique(g) if int(c) != bg):
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


def _extract_legend_order(g: np.ndarray, bg: int, segment_colors: set[int]) -> list[int]:
    h, w = g.shape
    singles: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v not in segment_colors:
                continue
            isolated = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) == (r, c):
                        continue
                    if int(g[rr, cc]) != bg:
                        isolated = False
                        break
                if not isolated:
                    break
            if isolated:
                singles.append((r, c, v))

    assert len(singles) == len(segment_colors), "legend must expose one isolated swatch per segment color"
    rows = {r for r, _, _ in singles}
    cols = {c for _, c, _ in singles}
    if len(rows) == 1:
        ordered = sorted(singles, key=lambda item: item[1])
    elif len(cols) == 1:
        ordered = sorted(singles, key=lambda item: item[0])
    else:
        raise AssertionError("legend swatches must lie on one visible row or column")

    legend_colors = [v for _, _, v in ordered]
    assert sorted(legend_colors) == sorted(segment_colors), "legend colors must match segment colors"
    return legend_colors


def _pick_reference_segment(g: np.ndarray, bg: int, segments: list[dict]) -> dict:
    colors = {int(s["color"]) for s in segments}
    legend_colors = _extract_legend_order(g, bg, colors)
    assert len(legend_colors) % 2 == 1, "legend must have an odd visible color order"
    ref_color = legend_colors[len(legend_colors) // 2]
    ref_cands = [s for s in segments if int(s["color"]) == ref_color]
    assert len(ref_cands) == 1, "legend median color must identify exactly one segment"
    return ref_cands[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align every parallel segment to the span of the segment whose color is the median visible color.

    Input structure:
    - The background is the most common color.
    - Several straight multi-cell segments share one dominant direction and lie on distinct parallel offsets.
    - Each segment uses a distinct non-background color, the number of segments is odd, and isolated
      single-pixel legend swatches show the intended color order on one visible row or column.
    - The segments already share one directional endpoint, but their lengths and exact spans differ.

    Rule:
    - Detect the common segment direction and trace every multi-cell segment along it.
    - Read the isolated legend swatches in their visible order and select the segment whose color is the middle legend color.
    - Read that segment's directional start and end as the target span.
    - Clear the original segment pixels, then redraw every segment so it starts at the same position and has the same length as that median-color segment; isolated single pixels remain untouched.

    Key insight:
    - The controlling line is chosen by the visible legend order rather than geometric position or length, so the middle legend color determines the final shared span.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)

    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4

    ref = _pick_reference_segment(g, bg, segments)
    ref_t0, ref_t1 = int(ref["t_start"]), int(ref["t_end"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        dt = ref_t0 - int(s["t_start"])
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        rr = int(s["start"][0]) + k * dr
        cc = int(s["start"][1]) + k * dc
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = int(s["color"])
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
    chosen = sorted(random.sample(_COLORS, n))
    median_color = chosen[n // 2]
    other_colors = chosen[: n // 2] + chosen[n // 2 + 1 :]
    random.shuffle(other_colors)

    colors = [None] * n
    colors[ref_idx] = median_color
    other_positions = [i for i in range(n) if i != ref_idx]
    for pos, col in zip(other_positions, other_colors):
        colors[pos] = col
    return tuple(int(c) for c in colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int, ref_idx: int, legacy_idx: int):
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    if legacy_idx in longable and (random.random() < 0.5 or len(longable) == 1):
        long_i = legacy_idx
        short_i = random.choice([i for i in nonrefs if i != legacy_idx])
    else:
        short_i = legacy_idx
        long_choices = [i for i in longable if i != legacy_idx]
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

    assert lengths[legacy_idx] != ref_len, "legacy reference span must differ from fused reference span"
    return lengths


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals, starts, lengths, colors) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _legend_slots_are_clear(inp: np.ndarray, out: np.ndarray, row: int, cols: list[int]) -> bool:
    for c in cols:
        for rr in range(max(0, row - 1), min(_H, row + 2)):
            for cc in range(max(0, c - 1), min(_W, c + 2)):
                if int(inp[rr, cc]) != _BG or int(out[rr, cc]) != _BG:
                    return False
    return True


def _pick_legend_slots(inp: np.ndarray, out: np.ndarray, n: int) -> list[tuple[int, int]] | None:
    width = 1 + (n - 1) * _LEGEND_SPACING
    preferred_rows = [1, _H - 2, 0, _H - 1]
    row_order = preferred_rows + [r for r in range(_H) if r not in preferred_rows]
    for row in row_order:
        for start in range(0, _W - width + 1):
            cols = [start + i * _LEGEND_SPACING for i in range(n)]
            if _legend_slots_are_clear(inp, out, row, cols):
                return [(row, c) for c in cols]
    return None


def _paint_legend(grid: np.ndarray, slots: list[tuple[int, int]], legend_colors: list[int]) -> None:
    for (r, c), color in zip(slots, legend_colors):
        grid[r, c] = color


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
    legacy_idx = n // 2
    ref_idx = _choose_ref_index(n)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len, ref_idx)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n, ref_idx)
    lengths = _choose_lengths(ref_len, caps, n, ref_idx, legacy_idx)
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

    legacy_out = _render_output_from_reference(d, s_vals, starts, lengths, colors, legacy_idx)
    if legacy_out is None or np.array_equal(out, legacy_out):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    legend_colors = sorted(int(color) for color in colors)
    legend_slots = _pick_legend_slots(inp, out, n)
    if legend_slots is None:
        return None
    _paint_legend(inp, legend_slots, legend_colors)
    _paint_legend(out, legend_slots, legend_colors)

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
