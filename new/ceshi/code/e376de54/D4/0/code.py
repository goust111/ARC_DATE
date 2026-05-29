# Variant of e376de54
# Dimension: D4 — 变换：延伸或裁剪线段 (transform)
# Change: 短于参考的线段延伸像素到参考长度，长于参考的线段裁剪多余像素... → 仅延伸不裁剪（短线段延伸，长线段保持不变）

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D4 variant
# a) Concrete behavior:
#    The scene still consists of 3 or 5 parallel single-color segments on distinct offsets, all in one
#    dominant direction. The median offset segment is the reference. Every output segment is shifted so
#    its start matches the reference start. If a segment is shorter than the reference, new pixels are
#    appended forward until it reaches the reference length. If a segment is already longer, the extra
#    tail remains visible in the output. Example: with horizontal segments and reference length 5,
#    input lengths [3,5,8] become output lengths [5,5,8] after all starts are aligned.
# b) Complexity calibration:
#    The generator still uses only 4 directions, odd segment counts {3,5}, and 1-3 colors. This keeps
#    the task readable while ensuring the new one-way rule is visible: each sample contains at least
#    one shorter segment that grows and at least one longer segment that stays longer.
# c) Old assumptions that break:
#    The original solver cleared every segment and redrew exactly the reference length, which silently
#    depended on "longer segments must be cropped". In this variant, the redraw length must become
#    max(original_length, reference_length); otherwise the preserved long tail would be destroyed.
# d) Placement legality:
#    For anchor="start", any chosen input length must fit from the common start. For anchor="end", a
#    long segment must fit both backward from the shared end in the input and forward from the aligned
#    start in the output. The generator therefore tracks both capacities and only chooses long lengths
#    that satisfy both.
# e) Visual observability:
#    The new rule is visible because each sample contains both branches of the transform: a short
#    segment extends, and a long segment remains longer than the reference in the output. No hidden
#    ordering by color id or any other non-visual attribute is used.
# f) Visual cleanliness:
#    Offsets remain unique and well separated, especially for diagonals, so shifted/extended segments
#    do not overlap. The output preserves the same sparse banded layout as the original task.
# g) Color reasoning:
#    Colors are unchanged. They identify segments only; no color remapping is introduced, avoiding
#    extra inference burden unrelated to D4.
# h) New solver assumptions and generator guarantees:
#    The solver still assumes one unique dominant direction, one segment per perpendicular offset, an
#    odd number of multi-cell segments, and that aligned outputs stay in bounds. The generator enforces
#    those conditions and validates each sample with main(input) == expected_output.

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


def _collect_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
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
    Align parallel colored segments to a shared reference start, but only lengthen segments that are too short.

    Input structure:
    - The background is the most frequent color.
    - Several multi-cell line segments lie on distinct parallel tracks in one shared direction; isolated single pixels may also exist.
    - Segment lengths vary, and the segments collectively share one endpoint pattern from the original task.

    Rule:
    - Detect the dominant direction and trace every multi-cell segment in that direction.
    - Use the segment at the median perpendicular offset as the reference, taking its start position and length.
    - Shift every segment so its start aligns with the reference start.
    - After shifting, extend only the segments shorter than the reference; segments already at least that long keep their full length.

    Key insight: The reference defines a minimum aligned extent rather than a fixed final extent, so long segments stay long while short ones are filled out.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc
    segments = _collect_segments(g, bg, d)

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
        draw_len = max(s["len"], target_len)
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
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

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


def _feasible_placements(d, s_vals, ref_len):
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps, end_caps))
        if any(min(start_caps[i], end_caps[i]) > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", start_caps, end_caps))
    return placements


def _choose_colors(s_vals):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _length_options(ref_len, input_cap, keep_cap):
    top_keep = min(_MAX_SEG_LEN, keep_cap)
    options = list(range(2, ref_len + 1))
    if top_keep > ref_len:
        options.extend(range(ref_len + 1, top_keep + 1))
    assert input_cap >= ref_len
    return options


def _choose_lengths(ref_len, input_caps, keep_caps, n):
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if keep_caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_choices = [i for i in nonrefs if i != long_i]
    if not short_choices:
        return None
    short_i = random.choice(short_choices)

    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, keep_caps[long_i]))
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = _length_options(ref_len, input_caps[i], keep_caps[i])
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options = [x for x in options if x != ref_len]
        lengths[i] = random.choice(options)
    return lengths


def _paint(grid, d, s_vals, starts, lengths, colors):
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _build_case():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, start_caps, end_caps = random.choice(placements)
    if anchor == "start":
        input_caps = start_caps
        keep_caps = start_caps
    else:
        input_caps = end_caps
        keep_caps = [min(a, b) for a, b in zip(start_caps, end_caps)]

    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, input_caps, keep_caps, n)
    if lengths is None:
        return None

    out_lengths = [max(length, ref_len) for length in lengths]
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, out_lengths, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    meta = {
        "direction": d,
        "anchor": anchor,
        "ref_len": ref_len,
        "input_lengths": tuple(lengths),
        "output_lengths": tuple(out_lengths),
        "num_shorter": sum(length < ref_len for length in lengths),
        "num_longer": sum(length > ref_len for length in lengths),
    }
    return inp, out, meta


def _gen_attempt():
    case = _build_case()
    if case is None:
        return None
    inp, out, _ = case
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
