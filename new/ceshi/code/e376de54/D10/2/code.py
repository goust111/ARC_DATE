# Variant of e376de54
# Dimension: D10 — 参考线段是否保持不变 (meta)
# Change: 参考线段（第二长线段）本身在输出中保持不变，其他线段调整到它的范围... → 所有线段包括参考线段都移动到新位置

import random
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg


# Design analysis for the D10 variant
# a) Concretization and behavior trace:
#    - The input still contains 3 or 5 same-direction colored segments on distinct parallel tracks.
#      The middle track by perpendicular order is the reference; its length remains the target length.
#    - Original behavior: keep the reference exactly where it is, and slide all other segments along the
#      main direction until they share the reference extent.
#    - Variant behavior: erase the whole bundle and redraw every segment, including the reference, inside a
#      direction-specific canonical corner strip. The relative ladder of tracks is preserved only through the
#      visible gap pattern between adjacent tracks.
#    - Example (horizontal): if input rows are [4, 9, 12] with gaps [5, 3], output rows become [0, 5, 8]
#      and every segment starts at column 0 with the reference length. The reference keeps its role as
#      length-provider but loses its old absolute coordinates.
# b) Complexity calibration:
#    - The canonical relocation preserves direction, segment count, order by track, and irregular gaps.
#      This keeps enough structure for humans to match input and output while making the D10 change visible:
#      the entire bundle jumps to a fixed border region instead of only trimming/extending in place.
# c) Implicit-assumption audit on the original solver:
#    - `ref_t0/ref_t1` used to be copied directly into the output. That relied on "reference stays put".
#      The variant breaks this assumption, so the new solver computes a fresh canonical placement from
#      the ordered gap pattern instead of from the reference coordinates.
#    - Clearing original segment pixels is still safe because all segments are fully redrawn elsewhere.
#    - Unique median by perpendicular offset is still required; generator keeps odd segment counts and
#      distinct track coordinates.
# d) Placement legality:
#    - The canonical placement function validates every target cell before drawing. The generator rejects
#      gap/reference-length combinations whose canonical bundle would exceed the 16x16 canvas.
#    - Input placement keeps the original one-end anchor constraint so the solver still receives the same
#      style of source configuration.
# e) Visual observability:
#    - The new rule is visible because the reference segment moves too. On generated samples, the output
#      bundle always touches the canonical border zone instead of staying around the reference line.
# f) Visual cleanliness:
#    - Only the line bundle is present. Track gaps remain irregular and non-overlapping, so the relocated
#      output is compact and readable instead of cluttered.
# g) Color predictability:
#    - Colors are unchanged. They are only identifiers for segments, not hidden ordering keys.

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


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
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
    d: tuple[int, int], t0: int, s: int, length: int, shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s, shape)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _segment_cells_from_start(
    start: tuple[int, int], d: tuple[int, int], length: int, shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = shape
    rr, cc = start
    cells = []
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        cells.append((rr, cc))
        rr += d[0]
        cc += d[1]
    return cells


def _canonical_starts(
    shape: tuple[int, int], d: tuple[int, int], gaps: list[int], ref_len: int
) -> list[tuple[int, int]] | None:
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    span = offsets[-1]
    starts: list[tuple[int, int]] = []
    ref_last = ref_len - 1

    if d == (0, 1):
        starts = [(off, 0) for off in offsets]
    elif d == (1, 0):
        starts = [(0, span - off) for off in offsets]
    elif d == (1, 1):
        if any(off % 2 for off in offsets):
            return None
        span2 = span // 2
        starts = [(off // 2, span2 - off // 2) for off in offsets]
    else:
        if any(off % 2 for off in offsets):
            return None
        span2 = span // 2
        starts = [(span2 - off // 2, span2 - off // 2 + ref_last) for off in offsets]

    for start in starts:
        if _segment_cells_from_start(start, d, ref_len, shape) is None:
            return None
    return starts


def _paint_from_starts(
    grid: np.ndarray,
    d: tuple[int, int],
    starts: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> bool:
    for start, length, color in zip(starts, lengths, colors):
        cells = _segment_cells_from_start(start, d, length, grid.shape)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize a bundle of parallel colored segments by relocating the whole bundle to a canonical border region.

    Input structure:
    - bg is the most common color.
    - There are several multi-cell segments in one shared direction, each on a distinct parallel track; isolated single cells may also exist and should be ignored.
    - The reference is the segment on the median track when tracks are ordered by perpendicular offset.

    Rule:
    - Detect the common direction and extract every multi-cell segment.
    - Use the median-track segment only to determine the target length.
    - Preserve the visible ordering of tracks and their irregular inter-track gaps, but redraw every segment, including the reference, in a fixed direction-specific canonical zone.
    - In that relocated zone, all segments start from the canonical aligned frontier and are trimmed or extended to the reference length.

    Key insight:
    The median segment is no longer a stationary anchor; it contributes only the target length, while the whole family of segments is globally repositioned.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)

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

    assert segments, "at least one multi-cell segment must exist"

    segments_sorted = sorted(segments, key=lambda s: s["s"])
    s_vals = [s["s"] for s in segments_sorted]
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    ref = segments_sorted[len(segments_sorted) // 2]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    gaps = [s_vals[i + 1] - s_vals[i] for i in range(len(s_vals) - 1)]
    starts = _canonical_starts(g.shape, d, gaps, target_len)
    assert starts is not None, "canonical placement must stay in bounds"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    ok = _paint_from_starts(
        out,
        d,
        starts,
        [target_len] * len(segments_sorted),
        [s["color"] for s in segments_sorted],
    )
    assert ok, "canonical drawing must stay in bounds"
    return out


# --------------- generator (o5x) ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    length = 0
    while _cell_from_ts(d, t0 + length * _step(d), s, (_H, _W)) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    length = 0
    while _cell_from_ts(d, t1 - length * _step(d), s, (_H, _W)) is not None:
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


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int):
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len, (_H, _W)) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
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


def _paint_input(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts_t: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts_t, lengths, colors):
        cells = _segment_cells(d, t0, s, length, grid.shape)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    gaps = [s_vals[i + 1] - s_vals[i] for i in range(n - 1)]
    out_starts = _canonical_starts((_H, _W), d, gaps, ref_len)
    if out_starts is None:
        return None

    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint_from_starts(out, d, out_starts, [ref_len] * n, list(colors)):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts_t = [t0] * n
    else:
        starts_t = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_input(inp, d, s_vals, starts_t, lengths, colors):
        return None

    ref_idx = n // 2
    inp_ref = _segment_cells(d, starts_t[ref_idx], s_vals[ref_idx], ref_len, (_H, _W))
    out_ref = _segment_cells_from_start(out_starts[ref_idx], d, ref_len, (_H, _W))
    if inp_ref is None or out_ref is None or inp_ref == out_ref:
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
