# Variant of e376de54
# Dimension: D14 — 轨迹：原始线段位置的处理 (trajectory)
# Change: 无轨迹——线段就地修改，延伸或裁剪后原始多余像素消失... → 延伸区域填充为不同于线段色的标记色

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D14 variant
# a) Concrete behavior / trace:
#    The original task normalizes several parallel segments to the reference span and
#    redraws every kept/added cell in the segment's own color, so extension is invisible.
#    Here the target span is still the reference segment's span, but only the pixels that
#    already existed inside that span stay in the segment color; newly created extension
#    pixels become a separate marker color 2. Example on a horizontal row:
#    input:  [A A A].......... and reference span length = 5 with shared start
#    old out:[A A A A A]
#    new out:[A A A 2 2]
#    For a longer segment sharing the end, e.g. [A A A A A A] vs target length 5,
#    both old and new variants keep the in-range part [A A A A A] and delete the excess.
#    So the only changed stage is output rendering for segments shorter than the target.
# b) Complexity calibration:
#    To keep the rule visible but not overcomplicated, the marker color is one global
#    constant (2), inspired by source task 7666fa5d's flood-fill color. It is not chosen
#    per segment or by hidden arithmetic. Inputs still contain 3 or 5 segments, and at
#    least one non-reference segment is strictly shorter so the marker appears.
# c) Implicit assumption review:
#    - Direction detection still relies on repeated same-color neighbors; valid because
#      the input remains unchanged from the original family and never contains marker cells.
#    - Segment extraction still treats each color component along the main direction as a
#      single segment; safe because generator keeps offsets distinct and avoids overlap.
#    - The old redraw loop assumed every output cell should be recolored with the segment
#      color. That is no longer valid; the new loop must test whether each target cell was
#      already present in the input interval.
#    - The old code was insensitive to which endpoint was shared. The new code needs that
#      information implicitly: a segment must share ref_t0 or ref_t1, otherwise "which side
#      is extension" would be ambiguous. The generator guarantees this anchor property.
# d) Placement legality:
#    The generator still picks a target reference span first, then places all input
#    segments on distinct parallel lines. Because lines never share a perpendicular offset,
#    no segment overlaps another. Capacity checks ensure longer variants stay in bounds
#    when anchored at the common start or common end.
# e) Visual observability:
#    The marker color never appears in the input and always appears exactly on the missing
#    side of shortened segments in the output. Across examples, humans can see that the
#    middle segment defines the final span and color 2 marks only the added cells.
# f) Visual cleanliness:
#    Only one layer is ever drawn on a cell: each line has its own offset, and the target
#    span lies on that same line. There are no mixed overlays or secondary symbols.
# g) Color-change inferability:
#    The palette excludes 2 from segment colors, so every red cell in the output is a
#    structural cue for "extension", not a line identity. Because the marker is global and
#    repeated on every shortened line, a small number of examples is enough to infer it.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_MARKER = 2


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


def _cell_from_ts(
    d: tuple[int, int], t: int, s: int, shape: tuple[int, int]
) -> tuple[int, int] | None:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel colored segments to the reference span, but paint only the newly added cells with a marker color.

    Input structure:
    - The background is the most frequent color.
    - Several multi-cell segments lie on distinct parallel lines with one shared direction; colors may repeat across lines.
    - All segments already share either the future start point or the future end point of the reference span.

    Rule:
    - Detect the common segment direction and identify the reference as the segment at the median perpendicular offset.
    - Use the reference segment's span as the target interval for every segment on its own parallel line.
    - Inside that target interval, keep cells that already existed in the input in the segment's own color, and paint only the missing extension cells with marker color 2; cells outside the target interval disappear.

    Key insight: The geometry is still governed by the median-offset reference line, but the output now preserves a visible trajectory: original pixels stay in their line color while only the added extension is recolored.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)

    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - d[0], c - d[1]
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += d[0]
                    cc += d[1]
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
                    }
                )

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
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        assert s["t_start"] == ref_t0 or s["t_end"] == ref_t1, "segment must share one anchored endpoint"
        for t in range(ref_t0, ref_t1 + 1, delta_t):
            cell = _cell_from_ts(d, t, s["s"], (h, w))
            assert cell is not None, "target segment cell must stay in bounds"
            r, c = cell
            if s["t_start"] <= t <= s["t_end"]:
                out[r, c] = s["color"]
            else:
                out[r, c] = _MARKER

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 3, 4, 5, 6, 8, 9]  # exclude bg=7 and marker=2

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (4, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s, (_H, _W))
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s, (_H, _W)) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s, (_H, _W)) is not None:
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


def _feasible_placements(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int
) -> list[tuple[int, str, list[int]]]:
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * _step(d)
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
    lengths[short_i] = random.randint(2, ref_len - 2)

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
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _build_output(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    t0: int,
    ref_len: int,
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> np.ndarray | None:
    out = np.full((_H, _W), _BG, dtype=int)
    delta = _step(d)
    target_ts = [t0 + k * delta for k in range(ref_len)]
    for s, start, length, color in zip(s_vals, starts, lengths, colors):
        t_end = start + (length - 1) * delta
        for t in target_ts:
            cell = _cell_from_ts(d, t, s, (_H, _W))
            if cell is None:
                return None
            r, c = cell
            out[r, c] = color if start <= t <= t_end else _MARKER
    return out


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_input(inp, d, s_vals, starts, lengths, colors):
        return None

    out = _build_output(d, s_vals, t0, ref_len, starts, lengths, colors)
    if out is None:
        return None
    if int(np.sum(out == _MARKER)) < 2:
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
