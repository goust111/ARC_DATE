# Combo variant of e376de54 (index 8)
# combo_spec: D4/3+D14/1
# Name: 裁剪留痕+裁剪标记

import random

import numpy as np

from arc_utils import Color, get_bg

# ---------------------------------------------------------------------
# Design analysis for the combo variant D4/3 + D14/1
#
# combo_intent:
#   The seed task already aligns every parallel segment to the reference
#   interval chosen by the median perpendicular offset. This combo keeps that
#   geometry exactly, but changes what happens to cells that get cropped away.
#   Instead of disappearing into background, every truly deleted cell remains
#   visible at its original coordinate in one fixed warning color: red.
#   D4/3 contributes "deleted information stays visible"; D14/1 resolves the
#   visual encoding by making that residue a special trail color.
#
# a_concretization_and_trace:
#   Example, horizontal shared-end input:
#     reference extent = cols 4..8, ref_len = 5
#     long blue input  = cols 2..8
#     short green input = cols 6..8
#   Seed output:
#     blue  -> cols 4..8
#     green -> cols 4..8
#     cols 2..3 vanish to bg
#   Combo output:
#     blue  -> cols 4..8
#     green -> cols 4..8
#     old blue cells at cols 2..3 stay red
#   Difference:
#     the kept aligned segment geometry is unchanged, but shortening now leaves
#     a visible warning scar. Extension still uses the segment's own color and
#     does not create any red trail.
#
# b_complexity_calibration:
#   Keep the original 16x16 canvas, direction set, odd segment counts, median
#   reference rule, and irregular offset spacing. Use exactly one reserved
#   warning color, red, and exclude it from input segment colors. Generator
#   still forces both a shorter and a longer non-reference segment, so examples
#   simultaneously demonstrate extension and cropping without adding new object
#   types or a large color vocabulary.
#
# c_implicit_assumption_audit:
#   - Direction detection still works because the input contains only plain
#     monochrome segments; red exists only in output.
#   - Segment extraction by monochrome runs remains valid because every input
#     segment is still a single-color collinear run on a unique parallel line.
#   - The seed solver cleared all segment cells to background before redraw.
#     That assumption breaks here because deleted cells must survive. We still
#     clear first, but then explicitly restore cropped-only cells in red and
#     paint final aligned cells last.
#   - Marking "orig minus final" as trail is only semantically correct when
#     segments share one anchor endpoint, because then shortened segments are
#     strict supersets of the kept interval while shorter ones are strict
#     subsets. The generator preserves that original anchor invariant.
#   - To avoid false trails on odd out-of-distribution inputs, main marks red
#     only when a segment is actually longer than the target length.
#
# d_placement_legality:
#   Output-first generation is preserved. First choose the final aligned
#   placement for all segments; then derive the input by anchoring either all
#   starts or all ends and varying lengths within per-line capacity. Distinct
#   perpendicular offsets prevent overlap across segments, and every original,
#   final, and cropped cell is guaranteed in-bounds by the same t/s coordinate
#   checks used by the seed generator.
#
# e_visual_observability:
#   A human sees the same alignment behavior as the seed task plus red cells
#   exactly where longer lines were cut back. Across a few examples, start-
#   anchored cases place red at suffixes and end-anchored cases place red at
#   prefixes, making it clear that red denotes trimmed-overhang evidence rather
#   than a new segment color or a shifted duplicate.
#
# f_visual_cleanliness:
#   Red trail cells lie on the same support line as their source segment and
#   never branch. Offset gaps remain wide enough that colored finals and red
#   remnants do not merge into ambiguous blobs, including diagonals where gaps
#   stay at least four in perpendicular coordinates.
#
# g_change_inferability:
#   The color rule is structural, not identity-based: red always means "this
#   cell existed in the input but was removed by cropping." Because red is
#   absent from the input palette and the kept segment retains its own original
#   color, the mapping is inferable from 2-4 examples without hidden numeric
#   color ordering.
#
# h_new_solver_assumptions_and_generator_guarantees:
#   - Input reserves red for output-only warning marks.
#   - There is at least one multi-cell segment, the dominant direction is
#     unique, and perpendicular offsets are unique and odd in count.
#   - At least one segment is longer than the reference, so the combo behavior
#     differs from the seed output on every generated sample.
#   - Shorter segments are also present, preserving the original extend/crop
#     contrast instead of collapsing into a pure-cropping puzzle family.
# ---------------------------------------------------------------------

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_TRAIL_COLOR = int(Color.RED)


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


def _cells_from_start(start: tuple[int, int], d: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = d
    rr, cc = start
    cells = []
    for _ in range(length):
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all parallel segments to the median reference span, while keeping every cropped-away cell visible in red.

    Input structure:
    - The background is the most common color.
    - Several multi-cell monochrome line segments share one direction and sit on distinct parallel offsets.
    - Red is reserved for output-only crop marks, while isolated single pixels may still appear and must stay untouched.

    Rule:
    - Detect the common segment direction and choose the segment at the median perpendicular offset as the reference.
    - Shift every detected segment so its start matches the reference start, then redraw exactly the reference length in the segment's own color.
    - If a segment was longer than that target length, any original cells that get cut off remain at their old coordinates in red instead of reverting to background.

    Key insight:
    - The alignment target is unchanged from the seed task; the only new signal is that red explicitly records the cells lost to cropping, while extensions remain ordinary segment-colored growth.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    assert _TRAIL_COLOR != bg, "trail color must differ from background"

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert _TRAIL_COLOR not in colors, "input must reserve red for crop marks"
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

    placements = []
    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        final_cells = _cells_from_start((r0, c0), d, target_len)
        for rr, cc in final_cells:
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
        placements.append(
            {
                "color": s["color"],
                "orig_cells": s["cells"],
                "final_cells": final_cells,
                "was_trimmed": s["len"] > target_len,
            }
        )

    for p in placements:
        if not p["was_trimmed"]:
            continue
        final_set = set(p["final_cells"])
        for r, c in p["orig_cells"]:
            if (r, c) not in final_set:
                out[r, c] = _TRAIL_COLOR

    for p in placements:
        for r, c in p["final_cells"]:
            out[r, c] = p["color"]

    return out


# --------------- generator (combo D4/3 + D14/1) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 3, 5, 6, 8, 9]  # exclude red because red is the trail marker

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75
_EXTRA_LONG_PROB = 0.45

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
        return int(r), int(c)
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
    ref_idx = len(s_vals) // 2
    placements = []
    delta = _step(d)
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
    palette = random.sample(_COLORS, n_colors)
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
        cap_hi = min(_MAX_SEG_LEN, caps[i])
        options = list(range(2, cap_hi + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
        if caps[i] > ref_len and random.random() < _EXTRA_LONG_PROB:
            lengths[i] = random.randint(ref_len + 1, cap_hi)
    return lengths


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...]) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _render_output(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
    ref_t0: int,
    ref_len: int,
    *,
    keep_trails: bool,
) -> np.ndarray | None:
    out = np.full((_H, _W), _BG, dtype=int)
    for s, start_t, length, color in zip(s_vals, starts, lengths, colors):
        orig_cells = _segment_cells(d, start_t, s, length)
        final_cells = _segment_cells(d, ref_t0, s, ref_len)
        if orig_cells is None or final_cells is None:
            return None
        if keep_trails and length > ref_len:
            final_set = set(final_cells)
            for r, c in orig_cells:
                if (r, c) not in final_set:
                    out[r, c] = _TRAIL_COLOR
        for r, c in final_cells:
            out[r, c] = color
    return out


def _build_case():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
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
    if _TRAIL_COLOR in set(map(int, np.unique(inp))):
        return None

    out = _render_output(d, s_vals, starts, lengths, colors, t0, ref_len, keep_trails=True)
    seed_out = _render_output(d, s_vals, starts, lengths, colors, t0, ref_len, keep_trails=False)
    if out is None or seed_out is None:
        return None

    trail_cells = int(np.count_nonzero(out == _TRAIL_COLOR))
    long_segments = sum(1 for length in lengths if length > ref_len)
    seed_gap = int(np.count_nonzero(out != seed_out))
    if trail_cells <= 0 or long_segments <= 0 or seed_gap <= 0:
        return None

    meta = {
        "direction": d,
        "anchor": anchor,
        "ref_len": int(ref_len),
        "lengths": tuple(int(x) for x in lengths),
        "trail_cells": trail_cells,
        "long_segments": int(long_segments),
        "seed_gap": seed_gap,
    }
    return inp, out, meta


def _gen_attempt() -> np.ndarray | None:
    case = _build_case()
    if case is None:
        return None
    inp, out, _meta = case

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
