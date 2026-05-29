# Combo variant of e376de54 (index 10)
# combo_spec: D6/2+D8/0
# Name: 零间距紧凑+紧凑裁剪

"""
Design analysis for the D6/2 + D8/0 combo
=========================================

a) Concrete behavior and trace
   - The source task keeps several sparse parallel segments on the 16x16 board,
     chooses the segment at the median perpendicular offset, and forces every
     segment to match that reference segment's extent.
   - D6/2 removes the gaps between transformed segments. D8/0 then removes all
     surrounding background. For horizontal and vertical lines, a literal
     gapless packing already becomes a solid rectangle after cropping.
   - The diagonal case is the important fusion point. If we only copied the D6
     single-variant geometry, packed diagonals would still crop to a slanted
     ribbon with holes. The combo description explicitly asks for a compact
     rectangular color block with no extra blank space, so the fused rule must
     express the packed band in its intrinsic coordinates: one axis = segment
     order, the other axis = position along the reference length.
   - Small trace:
       input: 5 sparse "\" segments with colors ordered by perpendicular
       position, lengths 2/4/5/7/3, all sharing one anchored end.
       source output: the 5 diagonals are aligned to the median segment's
       extent on the full 16x16 canvas.
       fused output: keep the same sorted segment order, keep the median
       length, erase spacing entirely, and return a 5 x ref_len rectangle whose
       rows are solid color stripes. The diagonal orientation is no longer
       represented in screen space; it has been compacted into the stripe/step
       coordinate frame required by the combo.

b) Complexity calibration
   - The task still depends on three nontrivial observations: detect a unique
     dominant direction, recover the median-by-offset reference, and read the
     reference length.
   - The generator keeps 3 or 5 segments, irregular sparse input gaps, and one
     shorter plus one longer non-reference segment, so the reference cannot be
     guessed from color or length extremes.
   - To keep the compact rectangle legible, generated segments use distinct
     colors. This avoids adjacent packed stripes visually merging.

c) Hidden assumptions audited from the source solver
   - Direction detection by same-color adjacency is still valid because the
     input remains a sparse set of parallel multi-cell segments.
   - Segment extraction by walking along the detected direction is still valid.
   - The original redraw logic assumed every output segment stayed on its
     original parallel track and only shifted along the main direction. That
     assumption is broken by the combo: the output no longer uses the original
     board coordinates at all, only the ordered segment list and the reference
     length.
   - The original "leave isolated pixels untouched" behavior is not preserved.
     Under tight crop, only the transformed segment family belongs in the
     answer. The generator therefore omits singleton noise entirely.
   - The unique-median assumption remains essential and is still enforced.

d) Placement legality
   - Input generation still uses the original sparse line model on the 16x16
     board. Every candidate segment is checked cell-by-cell for in-bounds
     placement.
   - Perpendicular offsets remain distinct and irregular. Diagonal inputs keep
     the original >= 4 spacing so the input lines never touch or overlap.
   - Length selection still guarantees one non-reference segment is shorter and
     another is longer than the reference, while respecting forward/backward
     capacity from the chosen anchor.

e) Visual observability
   - Across examples, the input always shows separated parallel segments, while
     the output always becomes a background-free striped rectangle.
   - The output dimensions directly reveal the rule: height = number of
     segments, width = median-segment length. This makes the D6/2 compaction
     and D8/0 crop simultaneously visible.
   - The ordering cue is human-visible because colors are preserved and each
     output row corresponds to one input segment in perpendicular order.

f) Visual cleanliness
   - The output contains only solid horizontal stripes, so there are no holes,
     overlaps, or leftover canvas margins.
   - The input remains uncluttered: only the segment family is present, with
     enough spacing to read separate objects at a glance.

g) Color-rule inferability
   - Colors are preserved exactly; there is no remapping.
   - Color does not decide the transformation. It only labels which output row
     came from which input segment.

h) Variant metrics used during validation
   - Every generated output has shape (segment_count, reference_length).
   - Every generated output has zero background cells, proving the result is a
     truly tight compact block rather than a cropped sparse layout.
   - Input perpendicular gaps are always > 1, while output stripe gaps are 0.
   - For diagonal inputs specifically, the combo output differs from the naive
     crop of the D6-only slanted ribbon because the combo has no holes.

i) New solver assumptions audited against the generator
   - `main()` assumes the ordered segment list is recoverable from the input
     via unique perpendicular coordinates. The generator guarantees this.
   - `main()` assumes the reference segment is the unique median in that order.
     The generator keeps an odd segment count and distinct offsets.
   - `main()` assumes all meaningful foreground objects are multi-cell segments
     in the detected direction. The generator only emits that object family.
"""

import random

import numpy as np

from arc_utils import get_bg

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
                    }
                )
    return segments


def _target_length(ref: dict, d: tuple[int, int]) -> int:
    dr, dc = d
    delta_t = dr * dr + dc * dc
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    return (ref_t1 - ref_t0) // delta_t + 1


def _ordered_segments_and_ref(segments: list[dict], d: tuple[int, int]) -> tuple[list[dict], dict]:
    dr, dc = d
    delta_t = dr * dr + dc * dc
    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ordered = sorted(segments, key=lambda s: s["s"])
    return ordered, ref_cands[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert sparse parallel segments into a zero-gap compact stripe block whose width matches the median-reference length.

    Input structure:
    - The background is the most common color.
    - The foreground contains 3 or 5 multi-cell segments in one dominant direction, each on a distinct parallel track.
    - Segment lengths vary, but they share one anchored end along the main direction and remain well separated in the input.

    Rule:
    - Detect the common direction, extract every multi-cell segment, and order them by perpendicular position.
    - Choose the unique median segment in that order as the reference and read its length.
    - Redraw the whole family as a tightly packed rectangle: one output stripe per ordered segment, no gaps between stripes,
      and exactly the reference length along the stripe direction. Return only that minimal rectangle.

    Key insight:
    The combo keeps the original median-reference logic but discards the original board coordinates after that point:
    once spacing becomes zero and the output is tightly cropped, the right representation is the packed stripe-by-step
    coordinate grid, which is a solid rectangular color block.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    ordered, ref = _ordered_segments_and_ref(segments, d)
    target_len = _target_length(ref, d)

    out = np.full((len(ordered), target_len), bg, dtype=int)
    for i, seg in enumerate(ordered):
        out[i, :] = int(seg["color"])
    return out


# --------------- generator (abstract-output-first, sparse-input realization) ---------------

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

# --- Sparse input gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


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
    allowed = _PERP_GAP_DIAG if d in ((1, 1), (1, -1)) else _PERP_GAP_ORTHO
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
        options = [x for x in range(2, min(_MAX_SEG_LEN, caps[i]) + 1) if x != ref_len]
        lengths[i] = random.choice(options or [ref_len])
    return lengths


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...]) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _expected_output(colors: tuple[int, ...], ref_len: int) -> np.ndarray:
    out = np.empty((len(colors), ref_len), dtype=int)
    for i, color in enumerate(colors):
        out[i, :] = int(color)
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
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    colors = tuple(random.sample(_COLORS, n))
    out = _expected_output(colors, ref_len)

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
