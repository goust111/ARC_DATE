# Variant of e376de54
# Dimension: D4 — 变换：延伸或裁剪线段 (transform)
# Change: 短于参考的线段延伸像素到参考长度，长于参考的线段裁剪多余像素... → 裁剪的像素用背景色覆盖并留痕

import random

import numpy as np

from arc_utils import Color, get_bg

# ---------------------------------------------------------------------
# Design analysis for the D4 variant
#
# variant_spec:
#   core_change:
#     original_behavior:
#       Every detected segment is shifted so its start matches the reference
#       start, then re-drawn at the reference length. Any overhang from a
#       longer input segment simply disappears.
#     new_behavior:
#       The aligned final segment is still the same as in the original task,
#       but any input cell that belonged to a detected segment and does not
#       survive in that segment's final aligned footprint is re-colored with a
#       dedicated trail color. Short segments still extend in their own color.
#   concrete_example:
#     direction: horizontal
#     reference: length 5 at columns 4..8
#     longer_input: length 7 at columns 2..8 (end-anchored)
#     original_output: colored segment only at columns 4..8
#     new_output: gray trail at columns 2..3, colored segment at columns 4..8
#     difference:
#       The final kept geometry is unchanged; only the removed overhang is now
#       visible as explicit deletion evidence.
#   complexity_calibration:
#     Keep 3 or 5 segments, a single shared direction, unique perpendicular
#     offsets, and the same median-reference rule from the seed task.
#     Use exactly one dedicated trail color (gray=5) and remove gray from the
#     segment palette, so the new mechanism is visible without adding a second
#     inference problem about color identity.
#     Require at least one segment longer than the reference in every sample,
#     so deletion traces are always present and the variant cannot collapse
#     back to the seed behavior.
#   implicit_assumption_audit:
#     - Direction detection still relies only on the input, which contains
#       plain line segments and no trail cells yet, so the old detector stays
#       valid.
#     - Segment extraction by monochrome runs still works because each segment
#       remains a single-color collinear object in the input.
#     - The seed code used "clear all segment cells to bg" before redraw.
#       That was harmless when deleted cells were meant to vanish, but it is
#       wrong for this variant because it would erase the required evidence.
#       We therefore compute each segment's final footprint first, mark only
#       orig-minus-final cells as gray, then paint final colored cells last.
#     - Final paint must happen after trail paint, otherwise overlap cases
#       where a shifted segment reuses some old cells would be corrupted. The
#       generator guarantees orig-minus-final and final are disjoint for the
#       same segment, but painting final last is still the robust order.
#     - Gray must be reserved for trails only; otherwise input segments could
#       be mistaken for deletion evidence in the output. The generator enforces
#       this and main asserts it.
#   placement_legality:
#     Output-first generation is preserved: first choose the aligned final
#     placement shared by all segments, then derive input segments by anchoring
#     either their starts or their ends and varying lengths.
#     Unique perpendicular offsets prevent any overlap between segments or
#     between trails from different segments. The same bounds logic as the seed
#     task ensures every original cell, final cell, and trail cell stays in
#     bounds.
#   visual_observability:
#     Human-visible signal comes from gray scars appearing exactly on the side
#     where a longer segment was cut. Start-anchored cases produce suffix
#     trails; end-anchored cases produce prefix trails. Shorter segments still
#     show extension in their own color, so both "growth" and "cut with scar"
#     can be inferred from a few examples.
#   visual_cleanliness:
#     Trails lie on the same straight support line as the affected segment and
#     never branch or overlap with other segments because offsets are distinct.
#     This keeps the output legible despite the extra information channel.
#   color_inference:
#     The color change is structural, not id-based: gray always means "this
#     exact line cell used to exist in the input but was trimmed away."
#     Because only one reserved trail color is used, 2-4 examples are enough
#     to infer the mapping.
#   main_assumptions_and_generator_guarantees:
#     - There is at least one multi-cell segment and the dominant direction is
#       unique: ensured by construction.
#     - Segment line coordinates are unique and odd in count, so the median
#       reference is well-defined: ensured by the offset sampler.
#     - At least one segment is longer than the reference, so trail count is
#       positive in every generated puzzle: enforced by length selection.
# ---------------------------------------------------------------------

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_TRAIL = int(Color.GREY)


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
    Align parallel colored segments to the median reference extent, while preserving trimmed-off cells as gray scars.

    Input structure:
    - bg = most common color.
    - Several multi-cell line segments share one direction and sit on distinct parallel offsets.
    - Segment colors avoid the reserved gray trail color; isolated single pixels may exist and should remain untouched.

    Rule:
    - Detect the common direction and choose the segment at the median perpendicular offset as the reference.
    - Shift every segment so its start matches the reference start, then keep exactly the reference length.
    - Cells from the original segment that are not part of that segment's final aligned footprint are not erased silently: recolor them gray as deletion marks. Newly extended cells use the segment's own color.

    Key insight: The geometric target is unchanged from the seed task; the only new information channel is that every cropped-overhang cell remains visible as a gray residue.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert _TRAIL not in colors, "input must reserve gray for deletion trails"
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue  # not a segment start
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
            }
        )

    for p in placements:
        final_set = set(p["final_cells"])
        for cell in p["orig_cells"]:
            if cell not in final_set:
                r, c = cell
                out[r, c] = _TRAIL

    for p in placements:
        for r, c in p["final_cells"]:
            out[r, c] = p["color"]

    return out


# --------------- generator (variant D4) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 6, 8, 9]  # gray is reserved for deletion trails

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


def _step(d):
    """Dot product of direction with itself (1 for ortho, 2 for diag)."""
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d, t, s):
    """Convert (t, s) coordinates back to (row, col) for direction d. Returns None if out of bounds."""
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
    """Return list of (r, c) for a segment starting at t0 with given perpendicular offset s, or None if any cell is OOB."""
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d, t0, s):
    """Max segment length extending forward from t0 at perpendicular offset s."""
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d, t1, s):
    """Max segment length extending backward from t1 at perpendicular offset s."""
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d):
    """Valid range (lo, hi) inclusive for perpendicular offset s given direction d."""
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d, n):
    """Enumerate all valid perpendicular gap sequences of length n-1, ensuring irregular spacing."""
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
    """Pick n distinct perpendicular offsets with irregular gaps."""
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, ref_len):
    """Find all (t0, anchor_mode, capacity_list) where segments can be placed with room for length variation."""
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


def _choose_colors(s_vals):
    """Assign 1-3 distinct non-bg, non-trail colors across segments, ensuring all chosen colors appear."""
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
    """Choose lengths with at least one shorter and one longer than ref_len. Returns None if infeasible."""
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


def _paint(grid, d, s_vals, starts, lengths, colors):
    """Render segments onto grid. Returns False if any segment cell is OOB."""
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _expected_output(d, s_vals, ref_t0, ref_len, starts, lengths, colors):
    """Render the variant output: final aligned segments plus gray trails on trimmed cells."""
    out = np.full((_H, _W), _BG, dtype=int)
    delta = _step(d)

    for s, start_t, length, color in zip(s_vals, starts, lengths, colors):
        orig_cells = _segment_cells(d, start_t, s, length)
        final_cells = _segment_cells(d, ref_t0, s, ref_len)
        if orig_cells is None or final_cells is None:
            return None
        final_set = set(final_cells)
        for r, c in orig_cells:
            if (r, c) not in final_set:
                out[r, c] = _TRAIL
        for r, c in final_cells:
            out[r, c] = color

    return out


def _build_case():
    """Build one candidate (input, output, meta) triple; returns None on failure."""
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

    out = _expected_output(d, s_vals, t0, ref_len, starts, lengths, colors)
    if out is None:
        return None

    trail_cells = int(np.sum(out == _TRAIL))
    if trail_cells <= 0:
        return None

    meta = {
        "direction": d,
        "anchor": anchor,
        "ref_len": int(ref_len),
        "lengths": tuple(int(x) for x in lengths),
        "trail_cells": trail_cells,
    }
    return inp, out, meta


def _gen_attempt():
    """Build one validated input grid or None on failure."""
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


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
