# Combo variant of e376de54 (index 16)
# combo_spec: D10/1+D14/0
# Name: 参考变色+保留原始轮廓

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg

# Design analysis for the combo variant
# -------------------------------------
# variant_spec:
#   a_concretization_and_trace:
#     - Base geometry is preserved exactly: 3 or 5 parallel multi-cell segments share one direction and
#       one anchor, and the reference is still the unique median perpendicular offset.
#     - Fusion output has three simultaneously visible layers of information:
#       1) every original segment position is kept as a pale trail,
#       2) every normalized non-reference segment is redrawn in its own original color,
#       3) the normalized reference segment is redrawn in a fixed grey marker color.
#     - Concrete trace, horizontal example:
#       row 2: red source length 7, row 6: green reference length 5, row 10: blue source length 3.
#       Base task output would be red/green/blue lines of length 5 aligned to row 6's span.
#       In this combo:
#         * first paint the three source spans in pale teal,
#         * then redraw the target spans,
#         * row 2 becomes red length 5 plus a teal 2-cell leftover,
#         * row 10 becomes blue length 5 with no teal at the extension side,
#         * row 6 becomes grey length 5, marking the chosen reference explicitly.
#       Difference from D10 alone: source-only leftovers remain visible.
#       Difference from D14 alone: the reference overlay is grey instead of its old color.
#   b_complexity_calibration:
#     - The geometry rule stays single-track and learnable: all segments still normalize to one hidden
#       reference extent, so the only new burden is interpreting the rendering layers.
#     - Two fixed output-only colors are enough and do not explode the hypothesis space:
#       grey=5 for the selected reference, pale teal=8 for preserved source locations.
#     - Generator forces both a shorter and a longer non-reference segment, ensuring each sample contains
#       both a colored extension and at least one trail-only leftover. That keeps the fusion visible.
#   c_implicit_assumption_audit:
#     - Original clearing loop assumed source pixels should disappear. In the combo, that assumption is
#       false; source pixels must become trail color instead of background.
#     - Original redraw loop assumed every target segment keeps its input color. That also becomes false;
#       the reference must override to grey while non-references keep their colors.
#     - Layer order now matters. Trail must be painted before normalized segments, otherwise source-only
#       leftovers would be lost or the reference marker could be overwritten.
#     - A new color-reservation assumption is introduced: input segment colors must avoid both output-only
#       colors (grey marker and teal trail), otherwise the fusion could become ambiguous. Generator enforces
#       that reservation explicitly.
#   d_placement_legality:
#     - The generator still uses output-first feasibility search, so every normalized target segment is
#       guaranteed to stay in bounds before any input is derived.
#     - Distinct perpendicular offsets prevent different segments from crossing or overwriting each other.
#       Because each object only ever overlaps its own trail, the overlay stays well-defined.
#     - One longer and one shorter non-reference are enforced, so trail leftovers and colored extensions
#       are physically realizable without needing extra objects or markers.
#   e_visual_observability:
#     - A human sees one grey target line among several colored target lines, plus pale source remnants.
#       Across 2-4 examples it is recoverable that the grey line is the reference and the pale cells show
#       where lines used to be before alignment.
#     - No sorting depends on hidden color ids or non-visual metadata; reference choice still comes only
#       from the median spatial offset.
#   f_visual_cleanliness:
#     - Only two layers can occupy the same offset: that segment's old position and its new position.
#       Since different offsets remain separated by the original spacing constraints, the output does not
#       collapse into multicolor clutter.
#     - Using teal for trails instead of grey avoids conflating "reference marker" with "kept source".
#   g_color_inference:
#     - Color changes are structurally keyed, not arbitrary:
#       median-offset normalized segment -> grey,
#       any source-only cell from any segment -> teal,
#       normalized non-reference cells -> original color.
#       These are few, repeated, and tied to visible roles, so they are inferable from examples.
#   h_new_solver_assumptions_checked_against_generator:
#     - `_pick_main_dir` requires a unique dominant direction; generator still paints only one family of
#       parallel segments.
#     - Segment tracing requires contiguous monochrome runs of length >= 2; generator paints only such runs.
#     - Median reference selection requires odd count and unique offsets; generator still uses counts 3 or 5
#       with distinct irregular gaps.
#     - Fixed preferred trail color assumes a spare non-input color exists; generator reserves teal=8 and
#       grey=5 from the input palette so the preferred colors are always available.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_REF_MARKER = 5
_PREFERRED_TRAIL_COLORS = [8, 4, 6, 1, 2, 3, 9, 0]
_GEN_TRAIL_COLOR = 8


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


def _choose_trail_color(bg: int, colors: list[int]) -> int:
    forbidden = set(colors)
    forbidden.add(bg)
    forbidden.add(_REF_MARKER)
    for col in _PREFERRED_TRAIL_COLORS:
        if col not in forbidden:
            return col
    for col in range(10):
        if col not in forbidden:
            return col
    raise AssertionError("need a spare trail color")


def _extract_segments(g: np.ndarray, d: tuple[int, int], bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    dr, dc = d
    segments: list[dict[str, object]] = []
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
    Normalize parallel segments to one reference span while showing both the old positions and the chosen reference explicitly.

    Input structure:
    - The background is the most frequent color.
    - Foreground contains 3 or 5 multi-cell line segments, all parallel and placed on distinct perpendicular offsets.
    - Each segment is monochrome in the input and shares one anchor direction with the others; isolated single cells are ignored.

    Rule:
    - Detect the unique dominant direction and trace every multi-cell segment in that direction.
    - Select the reference as the segment at the median perpendicular offset, then use its start and length as the target extent.
    - First repaint every original segment cell into a pale trail color so the pre-transform layout remains visible.
    - Next redraw each normalized segment on top: non-reference segments keep their original colors, while the reference segment becomes grey.

    Key insight: The geometry still comes from the median-offset segment exactly as in the base task, but the output now overlays
    a before/after comparison and also reveals which segment supplied the shared target span.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    trail_color = _choose_trail_color(bg, colors)

    s_vals = sorted(int(s["s"]) for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if int(s["s"]) == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0 = int(ref["t_start"])
    ref_t1 = int(ref["t_end"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = trail_color

    for s in segments:
        col = _REF_MARKER if int(s["s"]) == s_ref else int(s["color"])
        dt = ref_t0 - int(s["t_start"])
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        start_r, start_c = s["start"]
        rr = int(start_r) + k * dr
        cc = int(start_c) + k * dc
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col
            rr += dr
            cc += dc

    return out


# --------------- generator (o5x, combo D10/1 + D14/0) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_INPUT_COLORS = [1, 2, 3, 4, 6, 9]  # reserve 5 for reference marker, 8 for source trail

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
# Cap the visible non-marker palette so the full output stays within five colors
# (bg + trail + reference + at most two source colors).
_MAX_DISTINCT_COLORS = 2
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


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


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
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


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_INPUT_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
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


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: list[int] | tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
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
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    ref_idx = n // 2
    assert colors[ref_idx] not in (_REF_MARKER, _GEN_TRAIL_COLOR)

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
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, starts, lengths, [_GEN_TRAIL_COLOR] * n):
        return None
    out_colors = list(colors)
    out_colors[ref_idx] = _REF_MARKER
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, out_colors):
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
