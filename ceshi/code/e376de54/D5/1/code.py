# Variant of e376de54
# Dimension: D5 — 长度比较的排序规则 (meta)
# Change: 按线段像素数从长到短排序，选取第二长（rank=2）作为参考... → 倒数第二短线段为参考

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D5 variant
# a) Concrete behavior and trace:
#    - The grid still contains several parallel monochrome segments on distinct offsets,
#      all sharing one endpoint directionally, so the output can align every segment to a
#      single reference extent.
#    - The changed behavior is only how that reference is chosen: sort by visible segment
#      length from short to long and take the second entry. Example with 5 horizontal
#      segments ending at the same column and lengths [2, 4, 6, 8, 9]:
#      old D5 would use length 8; this variant uses length 4. The output therefore snaps
#      all segments to the start/end span of the 4-cell segment, not the 8-cell one.
#    - This affects reference selection inside `main()` and the input-length construction
#      inside the generator; direction detection, tracing, and repainting stay unchanged.
# b) Complexity calibration:
#    - The original generator allowed 3 or 5 segments, but with 3 segments the second
#      shortest equals the second longest, making the D5 change invisible. The variant
#      therefore fixes the count at 5 so the new ranking rule is observable.
#    - Lengths are constructed as exactly one shorter, one reference-length, and three
#      longer segments, with all five lengths distinct. This keeps the ranking unique and
#      gives enough visual evidence across a few examples.
# c) Hidden assumptions from the old code:
#    - The old reference logic depended on perpendicular-offset median and odd segment
#      count. That assumption is removed; we now require a unique second-shortest segment.
#    - Repainting still assumes all segments lie on distinct parallel lines, so unique `s`
#      offsets remain enforced.
#    - The shift logic assumes the reference start is a valid common target. The generator
#      still builds puzzles from a valid output state first, so every shifted segment stays
#      in bounds after trimming/extending to the chosen reference length.
# d) Placement legality:
#    - The generator still places one segment per perpendicular offset, keeps every cell in
#      bounds, and uses output-first verification with `main(inp) == out`.
#    - For diagonal directions the offset gaps stay >= 4, preserving clear separation and
#      preventing accidental touching that would confuse direction detection.
# e) Visual observability:
#    - The reference segment is forced away from the median perpendicular offset, so the
#      puzzle cannot be re-explained by the old "middle line by position" rule.
#    - No color id is used for ranking; only visible segment length matters.
# f) Visual cleanliness:
#    - Segments remain single-color, non-overlapping, and regularly parallel. The grid
#      contains only these segments on a plain background, so the before/after change is
#      easy to inspect.
# g) Color inference:
#    - Colors are preserved exactly; there is no color remapping. They only label segments,
#      which keeps the cognitive load on the new length-order rule rather than on color.

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
    Align parallel colored line segments to the extent of the second-shortest segment.

    Input structure:
    - The background is the most frequent color.
    - Several multi-cell line segments share one direction and occupy distinct parallel offsets.
    - Segment colors may repeat across offsets; isolated single pixels are ignored by the transformation.

    Rule:
    - Detect the common direction and trace every multi-cell segment along it.
    - Rank those segments by visible length from shortest to longest and choose the unique second-shortest one as the reference.
    - Shift every segment along the common direction so its start matches the reference start, then trim or extend it to the reference length.
    - Keep the grid size unchanged and redraw only the traced segment cells.

    Key insight:
    - The decisive cue is short-end length order: the reference is not the middle line by position and not a long-end rank.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
    assert len(segments) >= 2, "at least two multi-cell segments must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    sorted_lens = sorted(s["len"] for s in segments)
    ref_len = sorted_lens[1]
    assert sum(s["len"] < ref_len for s in segments) == 1, "reference must be the unique second-shortest"
    ref_cands = [s for s in segments if s["len"] == ref_len]
    assert len(ref_cands) == 1, "second-shortest reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1
    assert target_len == ref_len

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
        rr, cc = r0, c0
        for _ in range(target_len):
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

_SEGMENT_COUNT = 5
_REF_LEN_RANGE = (3, 5)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 600


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
    return [p for p in patterns if sum(p) <= max_span and len(set(p)) > 1]


def _choose_offsets(d, n):
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, ref_len):
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        placements.append((t0, "start", start_caps))
        placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _ref_index_choices(n):
    center = n // 2
    return [i for i in range(n) if i != center]


def _choose_distinct_longs(ref_len, long_idxs, caps):
    order = long_idxs[:]
    random.shuffle(order)
    assign = {}
    used = set()

    def dfs(pos):
        if pos == len(order):
            return True
        i = order[pos]
        hi = min(_MAX_SEG_LEN, caps[i])
        options = list(range(ref_len + 1, hi + 1))
        random.shuffle(options)
        for value in options:
            if value in used:
                continue
            used.add(value)
            assign[i] = value
            if dfs(pos + 1):
                return True
            used.remove(value)
            del assign[i]
        return False

    return assign if dfs(0) else None


def _choose_lengths(ref_len, caps, ref_idx):
    assert len(caps) == _SEGMENT_COUNT
    nonrefs = [i for i in range(len(caps)) if i != ref_idx]
    random.shuffle(nonrefs)
    for short_idx in nonrefs:
        long_idxs = [i for i in nonrefs if i != short_idx]
        if any(min(_MAX_SEG_LEN, caps[i]) <= ref_len for i in long_idxs):
            continue
        long_assign = _choose_distinct_longs(ref_len, long_idxs, caps)
        if long_assign is None:
            continue
        lengths = [None] * len(caps)
        lengths[ref_idx] = ref_len
        lengths[short_idx] = random.randint(2, ref_len - 1)
        for i, value in long_assign.items():
            lengths[i] = value
        return tuple(lengths)
    return None


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
    n = _SEGMENT_COUNT
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    random.shuffle(placements)
    ref_indices = _ref_index_choices(n)
    colors = _choose_colors(n)
    delta = _step(d)

    for t0, anchor, caps in placements:
        ref_order = ref_indices[:]
        random.shuffle(ref_order)
        for ref_idx in ref_order:
            lengths = _choose_lengths(ref_len, caps, ref_idx)
            if lengths is None:
                continue

            out = np.full((_H, _W), _BG, dtype=int)
            if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
                continue

            t1 = t0 + (ref_len - 1) * delta
            if anchor == "start":
                starts = [t0] * n
            else:
                starts = [t1 - (length - 1) * delta for length in lengths]

            inp = np.full((_H, _W), _BG, dtype=int)
            if not _paint(inp, d, s_vals, starts, lengths, colors):
                continue

            try:
                if not np.array_equal(main(inp), out):
                    continue
            except Exception:
                continue

            segments = sorted(_extract_segments(inp, _BG, d), key=lambda s: s["s"])
            if sorted(s["len"] for s in segments) != sorted(lengths):
                continue
            if [s["len"] for s in segments][n // 2] == ref_len:
                continue
            return inp
    return None


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
