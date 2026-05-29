# Combo variant of e376de54 (index 20)
# combo_spec: D1/4+D10/0
# Name: 颜色基准+参考消失

import random

import numpy as np

from arc_utils import get_bg

# Design analysis
# a) Concrete behavior and trace:
#    The seed task shows 3 or 5 parallel segments on a 16x16 grid. Originally the segment at the
#    median perpendicular offset controlled the common output span, and that segment stayed visible.
#    In this combo, one fixed color (`_REF_COLOR`) marks the controller instead, and that same
#    colored segment disappears from the output. Example with horizontal lines at offsets [2, 5, 9]:
#    - input lengths/start-alignment: top color 2 length 6, middle color 5 length 3, bottom color 1 length 8
#    - seed rule: use the middle offset line, so all three outputs become length 3 and the middle line remains
#    - D1-only: use the color-2 line, so all three outputs become length 6 and the color-2 line remains
#    - combo rule: use the color-2 line for the hidden target span, make the other two lines length 6,
#      and leave the color-2 lane empty
#    The two changed stages are therefore coupled: color chooses the invisible reference, then the
#    redraw step omits exactly that controller.
#
# b) Complexity calibration:
#    The variant keeps the same sparse geometry and only adds one stable cue: a fixed reference
#    color recurring across samples. That color appears on exactly one multi-cell segment, while
#    other segments reuse only 1-2 additional colors from a small palette. This keeps the task
#    inferable from a few examples without introducing too many free categorical choices.
#
# c) Hidden assumptions review:
#    - The seed `main()` assumed the controller came from median perpendicular rank and therefore
#      relied on odd segment count and median uniqueness. The combo solver drops that selection
#      logic entirely and instead assumes exactly one multi-cell segment of `_REF_COLOR`.
#    - The redraw loop in the seed repainted every segment, including the controller. That
#      assumption is invalid here; the combo must skip the reference-color segment after clearing.
#    - Segment tracing by color still works even when non-reference colors repeat, because each
#      segment occupies a distinct parallel offset and starts are detectable along the main
#      direction.
#    - No line-ordering by color id is used. The only semantic color fact is that one fixed color
#      indicates the controller across training examples.
#
# d) Placement legality:
#    The generator stays output-first around the reference-color span. It first chooses a feasible
#    target extent that fits on every offset, then derives an input by varying lengths while
#    preserving a common start or common end. All cells are checked in `(t, s)` coordinates before
#    painting, so segments neither overlap nor leave the board.
#
# e) Visual observability:
#    A human can detect the rule from 2-4 examples because the same color repeatedly vanishes in
#    the output while the remaining segments snap to that vanished line's span. The generator also
#    forces the reference-color segment away from the median offset, so "remove the middle line"
#    is not a sufficient explanation.
#
# f) Visual cleanliness:
#    Parallel offsets remain distinct and irregularly spaced, with diagonal gaps kept large enough
#    to avoid corner-touch ambiguity. The output is cleaner than the input because one whole lane
#    is removed rather than overwritten by markers or extra objects.
#
# g) Change inferability:
#    The color change is structural rather than arbitrary: the fixed reference color is the only
#    color excluded from non-reference segments, so its disappearance is visible and repeatable.
#    Because all other geometry is preserved, the solver's two-step abstraction remains readable:
#    first identify the special-colored segment, then use its hidden span.
#
# 9) New solver assumptions checked against the generator:
#    - `_pick_main_dir` needs a unique dominant direction: guaranteed because every multi-cell
#      object is one of the intended segments in the same direction.
#    - `_extract_segments` needs clean segment starts and unique offsets: guaranteed by the gap
#      pattern construction and single-run painting.
#    - Reference detection needs exactly one `_REF_COLOR` multi-cell segment: guaranteed by
#      `_choose_colors`.
#    - Redrawing non-reference segments onto the reference span must stay in bounds: guaranteed by
#      `_feasible_placements`, which chooses a target span valid for every offset.
#    - Anti-degeneration checks need alternative median-based outputs to be renderable: `_gen_attempt`
#      rejects samples where those comparison outputs cannot be formed.

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
_GEN_RETRIES = 400
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
    Align every non-reference segment to the span of a special-colored controller, then erase that controller.

    Input structure:
    - The background is the most frequent color.
    - Several multi-cell line segments lie on distinct parallel offsets but share one dominant direction.
    - Exactly one multi-cell segment has a fixed reference color, while the remaining segments use other non-background colors.
    - The segments share one directional endpoint pattern, yet their lengths vary.

    Rule:
    - Detect the common direction and trace each multi-cell segment along it.
    - Identify the unique segment in the fixed reference color and read its start/end as the target span.
    - Clear the original segment pixels.
    - Redraw every non-reference segment so it occupies that target span, but do not redraw the reference-color segment; isolated single pixels stay untouched.

    Key insight:
    - The hidden standard is chosen by color rather than by position, so the solver must infer an invisible span from the one color that consistently disappears in the output.
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

    ref_cands = [s for s in segments if s["color"] == _REF_COLOR]
    assert len(ref_cands) == 1, "reference color segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with direction step"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        if s["color"] == _REF_COLOR:
            continue
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to direction step"
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
    delta = _step(d)
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
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
    if not longable or ref_len <= 2:
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


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts,
    lengths,
    colors,
    keep_mask=None,
) -> bool:
    if keep_mask is None:
        keep_mask = [True] * len(s_vals)
    for keep, s, t0, length, color in zip(keep_mask, s_vals, starts, lengths, colors):
        if not keep:
            continue
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _render_output_from_controller(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts,
    lengths,
    colors,
    controller_idx: int,
    hidden_idx: int | None,
):
    out = np.full((_H, _W), _BG, dtype=int)
    keep_mask = [i != hidden_idx for i in range(len(s_vals))]
    target_len = lengths[controller_idx]
    target_starts = [starts[controller_idx]] * len(s_vals)
    target_lengths = [target_len] * len(s_vals)
    if not _paint(out, d, s_vals, target_starts, target_lengths, colors, keep_mask=keep_mask):
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

    combo_out = _render_output_from_controller(d, s_vals, starts, lengths, colors, ref_idx, ref_idx)
    color_only_out = _render_output_from_controller(d, s_vals, starts, lengths, colors, ref_idx, None)
    median_hidden_out = _render_output_from_controller(d, s_vals, starts, lengths, colors, median_idx, median_idx)
    median_full_out = _render_output_from_controller(d, s_vals, starts, lengths, colors, median_idx, None)
    if any(x is None for x in [combo_out, color_only_out, median_hidden_out, median_full_out]):
        return None
    if np.array_equal(combo_out, color_only_out):
        return None
    if np.array_equal(combo_out, median_hidden_out):
        return None
    if np.array_equal(combo_out, median_full_out):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        solved = main(inp)
        if np.array_equal(solved, combo_out):
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
