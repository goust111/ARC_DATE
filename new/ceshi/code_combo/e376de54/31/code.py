# Combo variant of e376de54 (index 31)
# combo_spec: D3/0+D9/2
# Name: 头端对齐+两阶段处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The scene stays the same family as the supplied solver/spec: a 16x16 orange background
#      with 3 or 5 parallel monochrome segments on distinct offsets, all sharing one dominant
#      direction from {(0,1), (1,0), (1,1), (1,-1)}.
#    - The reference is still the segment at the median perpendicular offset. That part of the
#      visual logic is unchanged because the combo_spec only alters the endpoint-handling process.
#    - The fusion changes the transformation into two explicit passes:
#      1. Head-align every segment so its start lands on the reference start, while preserving
#         that segment's own current length.
#      2. From those aligned starts, redraw every segment with the reference length.
#    - Concrete horizontal trace:
#         input segments share end column 10 with lengths [3, 5, 8]
#         -> starts are [8, 6, 3], reference start is 6 and reference length is 5.
#         base/original direct redraw: output spans [6..10] for all three segments.
#         combo phase 1: spans become [6..8], [6..10], [6..13].
#         combo phase 2: all three are then reset to [6..10].
#      The final raster matches the direct redraw, but the combo semantics require the solver to
#      factor the move into a visible head-alignment stage followed by a length-normalization stage.
# b) Complexity calibration
#    - The object inventory, direction set, gap rules, color palette, and odd segment count stay
#      unchanged so the task still reads as "parallel-line alignment", not as a different puzzle.
#    - To keep the head-alignment stage human-meaningful, the generator only uses shared-end
#      inputs and forces at least one non-reference segment shorter than the reference and one
#      longer than the reference. This guarantees phase 1 is non-trivial and phase 2 changes it.
#    - Reference lengths remain in 3..8 and raw segment lengths in 2..12, preserving the original
#      balance between easy visibility and enough length variation.
# c) Hidden assumption audit against the supplied main()
#    - Dominant-direction detection, segment tracing, median-offset reference selection, and
#      clearing original pixels remain valid because neither D3/0 nor D9/2 changes object identity.
#    - The original redraw loop implicitly collapsed "shift to ref start" and "use ref length"
#      into one step. Under this combo that code is no longer an adequate explanation, even though
#      it lands on the same final cells. The new solver therefore materializes an aligned-start
#      placement list before performing the length reset.
#    - A new assumption appears: after phase 1, every preserved-length segment must still fit on
#      the canvas long enough for phase 2 to redraw from the aligned start. The generator enforces
#      this by requiring enough forward room from the chosen shared start and enough backward room
#      from the shared end used in the input.
# d) Placement legality
#    - Parallel offsets are still unique and irregular, so segments never overlap after either
#      phase. They merely slide along their own tracks.
#    - For each offset s, the generator computes a capacity that is valid both for the final
#      shared-start placement and for the shared-end input placement. That joint capacity prevents
#      out-of-bounds writes in both directions.
# e) Visual observability
#    - Across samples, humans can see that inputs share one endpoint but not the other, while
#      outputs share both. Because lengths vary on input, the "first make the near ends coincide,
#      then equalize the far ends" interpretation is the natural one.
#    - The generator rejects degenerate length sets: there is always at least one shorter and one
#      longer non-reference segment, so the hidden phase-1 picture would have staggered ends and
#      the final output visibly differs from a head-align-only rule.
# f) Visual cleanliness
#    - The puzzle remains visually sparse: only thin parallel segments on separate tracks, with no
#      crossings and no extra markers.
#    - Diagonal offsets keep gaps >= 4, which avoids accidental touching that could blur segment
#      extraction or make two tracks look connected.
# g) Color inference
#    - Colors continue to function only as segment labels. No recoloring, ranking by palette id,
#      or hidden color-based correspondence is introduced.
# h) Quantifiable non-degeneracy target used in testing
#    - For generated samples, the phase-1 head-aligned state must have more than one distinct
#      length, and it must differ from the final output. Equivalently, at least one segment is
#      shorter than the reference and at least one is longer.
#    - This combo is intentionally distinguished from the D3-only single variant, whose output
#      would stop after phase 1. The generator measures non-degeneracy against that failure mode.
# i) Main() assumption audit after the change
#    - `_extract_segments` assumes each meaningful object is a maximal same-color run along the
#      dominant direction, with isolated single pixels ignored as before.
#    - `_select_reference` assumes an odd number of multi-cell segments and unique perpendicular
#      offsets; diagonal gaps remain large enough for unambiguous median selection.
#    - `_head_align_segments` assumes every shift along the main direction is a whole number of
#      direction steps. This follows from the t-coordinate construction and is asserted.
#    - `generate_input()` guarantees all of the above and validates every sample with `main()`.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

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

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


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


def _segment_cells(
    d: tuple[int, int], t0: int, s: int, length: int
) -> list[tuple[int, int]] | None:
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


def _extract_segments(
    g: np.ndarray, bg: int, d: tuple[int, int]
) -> list[dict[str, object]]:
    h, w = g.shape
    dr, dc = d
    segments: list[dict[str, object]] = []

    for col in [int(c) for c in np.unique(g) if int(c) != bg]:
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
    return segments


def _select_reference(
    segments: list[dict[str, object]], delta_t: int
) -> tuple[dict[str, object], int]:
    s_vals = sorted(int(s["s"]) for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if int(s["s"]) == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    return ref_cands[0], s_ref


def _head_align_segments(
    segments: list[dict[str, object]], ref_t0: int, d: tuple[int, int], delta_t: int
) -> list[dict[str, object]]:
    dr, dc = d
    aligned = []
    for s in segments:
        dt = ref_t0 - int(s["t_start"])
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        start = s["start"]
        r0 = int(start[0]) + k * dr
        c0 = int(start[1]) + k * dc
        aligned.append(
            {
                "color": int(s["color"]),
                "start": (r0, c0),
                "len": int(s["len"]),
                "s": int(s["s"]),
            }
        )
    return aligned


def _paint_segments(
    out: np.ndarray,
    segments: list[dict[str, object]],
    d: tuple[int, int],
    lengths: list[int] | None = None,
) -> np.ndarray:
    h, w = out.shape
    dr, dc = d
    if lengths is None:
        lengths = [int(s["len"]) for s in segments]
    for s, length in zip(segments, lengths):
        rr, cc = s["start"]
        for _ in range(length):
            assert 0 <= rr < h and 0 <= cc < w, "redrawn segment must stay in bounds"
            out[rr, cc] = int(s["color"])
            rr += dr
            cc += dc
    return out


def _head_aligned_grid(
    shape: tuple[int, int],
    bg: int,
    d: tuple[int, int],
    aligned_segments: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    return _paint_segments(grid, aligned_segments, d)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild parallel colored segments by first lining up their leading endpoints, then standardizing their lengths to a reference segment.

    Input structure:
    - The background is the most frequent color.
    - Several multi-cell segments appear in one shared horizontal, vertical, or diagonal direction.
    - Those segments sit on distinct parallel offsets, may reuse colors, and may have different lengths; isolated single pixels do not participate.

    Rule:
    - Detect the unique dominant direction and extract every maximal same-color segment of length at least two.
    - Choose the segment at the median perpendicular offset as the reference, and read both its start position and its length.
    - Phase 1: shift each segment along the common direction until its start matches the reference start, while keeping that segment's original length.
    - Phase 2: redraw from those aligned starts again, but now give every segment the reference length. Only the original segment cells are cleared first.

    Key insight:
    - The reference controls two attributes in sequence rather than all at once: first it supplies a common head position, then it supplies the common span. The final picture has shared starts and shared ends, but the intended reasoning path is explicitly two-stage.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    delta_t = _step(d)

    segments = _extract_segments(g, bg, d)
    ref, _ = _select_reference(segments, delta_t)
    ref_t0 = int(ref["t_start"])
    ref_t1 = int(ref["t_end"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    aligned = _head_align_segments(segments, ref_t0, d, delta_t)
    _paint_segments(out, aligned, d, lengths=[target_len] * len(aligned))
    return out


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
        patterns = [p + [gap] for p in patterns for gap in allowed]
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


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _feasible_placements(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int
) -> list[tuple[int, list[int]]]:
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        caps = []
        for s in s_vals:
            caps.append(min(_max_len_from(d, t0, s), _max_len_to(d, t1, s)))
        if any(caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, caps))
    return placements


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
    colors: tuple[int, ...],
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

    t0, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    aligned = np.full((_H, _W), _BG, dtype=int)
    if not _paint(aligned, d, s_vals, [t0] * n, lengths, colors):
        return None
    if np.array_equal(aligned, out):
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
