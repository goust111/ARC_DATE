# Combo variant of e376de54 (index 14)
# combo_spec: D3/1+D4/0
# Name: 尾端对齐+仅延伸

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for combo D3/1 + D4/0
# ---------------------------------------
# a) Concrete behavior and traced example:
#    The canvas still contains 3 or 5 parallel monochrome segments on distinct offsets in one shared
#    direction. The median-offset segment remains the reference. The seed task aligned both ends and
#    forced every segment to exactly the reference length. This combo changes both of those decisions:
#    only the terminal endpoint is shared, and the transform is one-way. Example in t-coordinates:
#    if the reference occupies t=8..12 (length 5), then a short segment at t=8..10 becomes t=8..12
#    because it must end at 12 and extend backward to length 5, while a long segment at t=8..14
#    becomes t=6..12 because it keeps length 7 and only slides so its tail lands at 12. Compared with
#    the seed behavior, long tails are no longer cropped and the aligned side moves from "both ends"
#    to "tail only".
# b) Complexity calibration:
#    The original direction set, odd segment counts, reference-length range, offset gaps, and color
#    pool are preserved. That keeps the puzzle recognizable while ensuring the combo is still legible:
#    every generated case contains at least one shorter-than-reference segment and one longer-than-
#    reference segment, so both branches of the new rule are visible in a single example.
# c) Implicit-assumption review:
#    The seed solver depended on two assumptions that now fail: (1) using ref_t0 as the redraw anchor
#    assumed start alignment was part of the rule, and (2) drawing exactly target_len assumed longer
#    segments had to be cropped. In the combo, the correct redraw length is max(original_len, ref_len),
#    and the correct start position is derived from the shared reference tail ref_t1 and that redraw
#    length. Direction detection, median-offset reference selection, and unique offsets remain valid.
# d) Placement legality:
#    The generator uses an output-first parameterization with a shared reference span t0..t1, but it
#    intentionally constructs the input in a single start-aligned mode. Any chosen input length must
#    fit forward from t0. Any length longer than the reference must also fit backward from t1, because
#    the output keeps that long segment and only shifts it left/upstream to align the tail. Distinct
#    perpendicular offsets and the original gap rules keep segments non-overlapping even when colors
#    repeat.
# e) Visual observability:
#    Restricting inputs to the common-start pattern makes both fused changes human-visible. Short
#    segments visibly grow until their tails meet the reference tail, and long segments visibly slide
#    so their extra head remains. If end-aligned inputs were allowed, many cases would display only
#    the "extend" change and would weaken the evidence for tail-only alignment.
# f) Visual cleanliness:
#    Outputs form neat flush tails with staggered heads, which is easy to parse by eye. The generator
#    keeps the original sparse band layout, preserves color identities, and avoids extra markers or
#    overlapping redraws that would make it unclear which segment moved or extended.
# g) Change inferability:
#    No color remapping is introduced. Colors only preserve segment identity across input and output,
#    so the solver does not rely on color order or numeric color values. The geometric cue alone must
#    reveal the rule, which is appropriate for 2-4 ARC examples.
# h) New solver assumptions checked against the generator:
#    The solver now assumes every multi-cell segment can be re-drawn with start coordinate
#    new_t0 = ref_t1 - (max(len, ref_len) - 1) * delta_t and remain in bounds. The generator enforces
#    that by choosing long lengths only from the joint capacity that fits both the input and the kept-
#    long output. It also guarantees an odd number of unique offsets and validates cases with
#    main(input) == expected_output before accepting them.

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
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
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


def _pick_reference(segments: list[dict], delta_t: int) -> dict:
    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    return ref_cands[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel colored segments so they all end at the reference tail, extending only the ones that are too short.

    Input structure:
    - The most common color is the background.
    - Several horizontal, vertical, or diagonal multi-cell segments lie on distinct parallel offsets.
    - Segment colors are preserved identities; colors may repeat across offsets, and isolated single pixels are ignored.

    Rule:
    - Detect the dominant shared direction and trace every multi-cell segment along it.
    - Use the segment at the median perpendicular offset as the reference, taking its terminal endpoint and length.
    - For each segment, set its final length to the larger of its original length and the reference length.
    - Reposition that segment only along the main direction so its terminal cell matches the reference terminal cell, then redraw it at the computed length.

    Key insight: The reference supplies a shared tail and a minimum required length. Short segments are completed backward to the reference span, while long segments keep their extra head instead of being cropped.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc
    segments = _collect_segments(g, bg, d)

    assert segments, "at least one multi-cell segment must exist"

    ref = _pick_reference(segments, delta_t)
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        draw_len = max(s["len"], target_len)
        new_t0 = ref_t1 - (draw_len - 1) * delta_t
        dt = new_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
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
        joint_caps = [min(a, b, _MAX_SEG_LEN) for a, b in zip(start_caps, end_caps)]
        if any(joint_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, joint_caps))
    return placements


def _choose_colors(s_vals):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, caps[long_i])
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, caps[i] + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
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

    t0, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    out_lengths = [max(length, ref_len) for length in lengths]
    out_starts = [t1 - (length - 1) * delta for length in out_lengths]

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, out_starts, out_lengths, colors):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, [t0] * n, lengths, colors):
        return None

    meta = {
        "direction": d,
        "ref_len": ref_len,
        "input_lengths": tuple(lengths),
        "output_lengths": tuple(out_lengths),
        "num_shorter": sum(length < ref_len for length in lengths),
        "num_longer": sum(length > ref_len for length in lengths),
        "num_extended": sum(length < ref_len for length in lengths),
        "num_preserved_long": sum(length > ref_len for length in lengths),
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
