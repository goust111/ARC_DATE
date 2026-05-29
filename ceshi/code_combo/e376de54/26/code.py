# Combo variant of e376de54 (index 26)
# combo_spec: D4/2+D10/1+D14/0
# Name: 延伸渐变+参考变色+保留原始轮廓

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg

# Design analysis for the fused combo variant
# ------------------------------------------
# fused_spec:
#   concretization_and_trace:
#     - Keep the original object family from the spec: a 16x16 grid with 3 or 5 parallel,
#       monochrome, multi-cell segments sharing one dominant direction and one anchor mode.
#     - Output is now a layered rendering of the same normalization:
#       1) every original segment cell is first recolored to a pale trail color;
#       2) each non-reference segment is shifted to the reference start and normalized to the
#          reference length, with the aligned carried-over prefix in the original color and any
#          newly appended tail cells in a fixed extension marker color;
#       3) the reference segment itself is finally painted in grey to expose which offset was chosen.
#     - Small trace, horizontal, target length 5:
#       input: red length 7, blue length 3, green length 5 on the median row.
#       base task output: all three rows become solid red/blue/green length 5.
#       fused output:
#         * the old 7/3/5 input positions are kept as pale trail cells;
#         * red row draws a 5-cell red target, leaving a 2-cell pale tail where trimming happened;
#         * blue row draws 3 blue cells then 2 yellow extension cells;
#         * green reference row is repainted grey, overriding its original green.
#       Difference vs base task: geometry is unchanged, but output now reveals old position,
#       extension amount, and reference identity at once.
#   complexity_calibration:
#     - Use exactly three reserved functional colors across all samples:
#       `_EXT_MARK=4`, `_REF_MARK=5`, `_TRAIL_MARK=8`.
#     - This is intentionally simpler than per-source-color recolor mappings; ARC users can infer
#       the semantics from repeated structural roles instead of hidden color arithmetic.
#     - Generator still forces at least one short non-reference and one long non-reference segment,
#       so every accepted sample shows both extension markers and source-only leftovers.
#   implicit_assumption_audit:
#     - Segment extraction by scanning monochrome runs still holds because inputs stay monochrome.
#     - The original "clear segment cells to bg" assumption is invalid now; those cells become trail.
#     - The original "redraw whole target with source color" assumption is also invalid:
#       non-reference short lines need a two-tone target, and the reference needs a grey target.
#     - The prefix test `idx < original_keep` remains valid after fusion because the base geometry
#       always normalizes by translating a segment so its start lands on `ref_t0`; the translated
#       source cells therefore occupy the first `len(segment)` cells of the target span.
#     - New solver assumption: input does not already use the three functional colors, otherwise
#       role decoding becomes ambiguous. Generator enforces that by sampling from `_INPUT_COLORS`.
#   placement_legality:
#     - Keep the original output-first search: first choose a fully in-bounds aligned reference span,
#       then derive input starts from a shared start or shared end anchor.
#     - Distinct perpendicular offsets still prevent overlap between tracks in both input and output.
#     - Reserving the functional colors only changes color legality, not occupancy legality.
#   visual_observability:
#     - Every sample is constructed to contain:
#       * a grey reference segment,
#       * at least one yellow extension tail,
#       * at least one pale leftover from trimming or shifting.
#     - These three cues are visible directly in pixels and do not depend on color ids as numbers.
#   cleanliness:
#     - Only one trail layer and one normalized layer exist per track, so the scene stays readable.
#     - Large perpendicular gaps from the base task remain, which avoids mixed-color clutter.
#   color_inference:
#     - Fixed functional colors create repeated evidence across 2-4 examples:
#       yellow always means "newly extended", grey always means "chosen reference",
#       light blue always means "old location kept as context".
#   main_assumptions_checked_against_generator:
#     - Unique dominant direction: preserved by generating only one parallel family.
#     - Unique median reference: preserved by odd segment count and distinct offsets.
#     - Visible fusion behavior: preserved by forcing one short and one long non-reference segment.
#     - Functional-color separation: preserved by excluding 4/5/8 from input colors.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_EXT_MARK = 4
_REF_MARK = 5
_TRAIL_MARK = 8


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


def _extract_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict[str, object]]:
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
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )
    assert segments, "at least one multi-cell segment must exist"
    return segments


def _pick_reference(segments: list[dict[str, object]], delta_t: int) -> tuple[int, int, int]:
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
    return s_ref, ref_t0, target_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize a family of parallel segments while preserving the old layout, marking extensions, and greying the chosen reference.

    Input structure:
    - The most frequent color is background.
    - Foreground consists of 3 or 5 monochrome multi-cell segments, all parallel and all lying on distinct perpendicular offsets.
    - The segments share a single dominant direction and one anchor mode; isolated single cells, if any, are ignored.

    Rule:
    - Detect the dominant direction, extract every multi-cell segment, and choose the reference as the segment at the median perpendicular offset.
    - Keep every original segment cell as a pale trail in the output.
    - Shift each segment so its start matches the reference start and redraw a segment of the reference length.
    - On non-reference tracks, the carried-over aligned prefix keeps the original color, any extra cells needed for extension become the extension marker color, and trimmed source-only cells remain visible only as trail.
    - On the reference track, the normalized segment is painted entirely grey to expose the selected reference line.

    Key insight: The geometry still follows the base task exactly, but the rendering now encodes three facts simultaneously: where each line used to be, which pixels were newly added by extension, and which line served as the hidden reference.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
    s_ref, ref_t0, target_len = _pick_reference(segments, delta_t)

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = _TRAIL_MARK

    for s in segments:
        dt = ref_t0 - int(s["t_start"])
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = int(s["start"][0]) + k * dr
        c0 = int(s["start"][1]) + k * dc
        rr, cc = r0, c0
        if int(s["s"]) == s_ref:
            for _ in range(target_len):
                assert 0 <= rr < h and 0 <= cc < w, "shifted reference must stay in bounds"
                out[rr, cc] = _REF_MARK
                rr += dr
                cc += dc
            continue

        original_keep = min(int(s["len"]), target_len)
        for idx in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = int(s["color"]) if idx < original_keep else _EXT_MARK
            rr += dr
            cc += dc

    return out


# --------------- generator (o5x, combo 26) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_INPUT_COLORS = [1, 2, 3, 6, 9]

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
# Keep one shared source color so the layered rendering stays within five
# visible colors (bg + source + trail + extension + reference).
_MAX_DISTINCT_COLORS = 1
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

    short_hi = ref_len - 1
    if ref_len >= 4 and random.random() < 0.8:
        short_hi = ref_len - 2
    short_hi = max(2, short_hi)
    lengths[short_i] = random.randint(2, short_hi)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)

    if not any(lengths[i] < ref_len for i in nonrefs):
        return None
    if not any(lengths[i] > ref_len for i in nonrefs):
        return None
    return lengths


def _starts_from_anchor(anchor: str, t0: int, ref_len: int, lengths: list[int], d: tuple[int, int]) -> list[int]:
    if anchor == "start":
        return [t0] * len(lengths)
    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    return [t1 - (length - 1) * delta for length in lengths]


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
            grid[r, c] = color
    return True


def _paint_combo_output(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    input_starts: list[int],
    t0: int,
    ref_len: int,
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    ref_idx = len(s_vals) // 2
    for s, start, length in zip(s_vals, input_starts, lengths):
        cells = _segment_cells(d, start, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = _TRAIL_MARK

    for idx, (s, orig_len, color) in enumerate(zip(s_vals, lengths, colors)):
        cells = _segment_cells(d, t0, s, ref_len)
        if cells is None:
            return False
        if idx == ref_idx:
            for r, c in cells:
                grid[r, c] = _REF_MARK
            continue
        keep = min(orig_len, ref_len)
        for cell_idx, (r, c) in enumerate(cells):
            grid[r, c] = color if cell_idx < keep else _EXT_MARK
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
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    starts = _starts_from_anchor(anchor, t0, ref_len, lengths, d)

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_input(inp, d, s_vals, starts, lengths, colors):
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint_combo_output(out, d, s_vals, starts, t0, ref_len, lengths, colors):
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
