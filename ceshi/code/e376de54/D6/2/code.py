# Variant of e376de54
# Dimension: D6 — 线段之间的空间间距 (meta)
# Change: 平行线段之间保持固定间距（通常1-3像素），间距在变换后不变... → 间距缩小到相邻（间距为0）

"""
Design analysis for the D6 variant
==================================

a) Concrete behavior and trace
   - Input stays close to the source task: a 16x16 grid with 3 or 5 same-direction colored segments,
     odd count, unique median by perpendicular order, irregular large gaps between lines, and one shared
     anchored end along the main direction.
   - Output changes only the spacing rule. The median segment still supplies the target length, but the
     sorted segment order is compressed into a gapless band. Orthogonal cases become adjacent rows/columns.
     Diagonal cases cannot remain gapless if every line keeps the exact same t_start, so each neighboring
     diagonal stripe also shifts one cell along the main direction; the result is a solid slanted band.
   - Example for d=(1,1), three segments ordered by s:
       old output: same t_start, same length, still far apart at s=-6,0,6
       new output: middle line stays fixed; the upper/lower lines move to the two immediately adjacent
       diagonals, with starts at ref_start+(-1,0) and ref_start+(1,0). This creates three touching bands.

b) Complexity calibration
   - The rule is still single-step and visually local: "match the median length, then squeeze the gaps out."
   - The generator keeps only 3 or 5 segments and one shared direction, so the pattern is inferable.
   - Colors are forced distinct across segments in generated examples so the packed stripes stay legible and
     do not merge into a single monochrome slab.

c) Hidden assumptions from the source solver
   - Segment extraction by color and direction is still valid because the input remains sparse.
   - Choosing the median segment by sorted perpendicular coordinate is still valid.
   - The old code assumed output lines keep their original perpendicular offsets and can be aligned by a pure
     shift along d. That is no longer true; packed output requires changing perpendicular placement too.
   - The old diagonal gap assertion belongs to the input model, not the output model, so it remains safe.
   - Clearing original segment cells before redraw is still correct because output lines can occupy new cells.

d) Placement legality
   - Packed output uses a direction-specific "touching offset" p:
       (0,1) -> (1,0)
       (1,0) -> (0,-1)
       (1,1) -> (1,0)
       (1,-1) -> (0,-1)
     Repeated copies of p move to the next adjacent parallel stripe without overlap.
   - The generator first selects a reference segment start that keeps the entire packed band in bounds, then
     chooses sparse input offsets around that reference line, then checks forward/backward capacity for the
     anchored varying-length input.

e) Visual observability
   - Inputs are deliberately spread out using the source-task gap ranges.
   - Outputs are deliberately packed to minimal distance. For orthogonal directions the empty rows/columns
     disappear entirely; for diagonals the scattered diagonals become a filled slanted ribbon.

f) Visual cleanliness
   - No extra markers are added.
   - Distinct colors and odd segment counts keep the median stripe visually interpretable.
   - Packed stripes do not overlap because p is never parallel to d.

g) Color-rule inferability
   - Colors are preserved, not remapped.
   - Color is never used for ordering or for deciding the transformation.

h) Variant metric used during validation
   - Input max perpendicular gap must exceed the packed output gap.
   - Output perpendicular gaps are exactly 1 in all generated cases, which is the minimal pixel-adjacent
     spacing for the chosen packed representation.

i) New solver assumptions audited against the generator
   - `main()` now assumes sorted segments can be redrawn at consecutive packed starts around the median start.
     The generator guarantees those packed starts stay in bounds.
   - `main()` assumes the median segment remains unique and that the input still uses sparse non-overlapping
     lines. The generator enforces odd count, distinct perpendicular offsets, and large input gaps.
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


def _touching_offset(d: tuple[int, int]) -> tuple[int, int]:
    if d in ((0, 1), (1, 1)):
        return 1, 0
    return 0, -1


def _cells_from_start(start: tuple[int, int], d: tuple[int, int], length: int):
    dr, dc = d
    rr, cc = start
    cells = []
    for _ in range(length):
        if not (0 <= rr < _H and 0 <= cc < _W):
            return None
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress parallel colored line segments into a tightly packed band using the median segment as the length reference.

    Input structure:
    - bg is the dominant background color.
    - Several multi-cell segments of different colors lie on distinct parallel lines in one shared direction.
    - The lines are separated in the input; isolated single pixels may also exist and should be ignored.

    Rule:
    - Detect the common segment direction and extract every multi-cell segment along it.
    - Use the segment at the median perpendicular position as the reference and keep its start cell and length.
    - Clear all original segments, then redraw them in the same top-to-bottom / left-to-right stripe order but packed
      onto immediately adjacent parallel lines, each with the reference length.

    Key insight: The median segment still controls the target extent, but the defining change is perpendicular compaction:
    the output no longer preserves the original gaps and instead collapses the sorted segments into a dense color band.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

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

    ordered = sorted(segments, key=lambda s: s["s"])
    ref_idx = len(ordered) // 2
    pr, pc = _touching_offset(d)
    for i, s in enumerate(ordered):
        r0 = ref["start"][0] + (i - ref_idx) * pr
        c0 = ref["start"][1] + (i - ref_idx) * pc
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "packed segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator (o5x) ---------------

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


def _choose_sparse_offsets(d, n, ref_s):
    allowed = _PERP_GAP_DIAG if d in ((1, 1), (1, -1)) else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    patterns = []
    if n == 3:
        for gl in allowed:
            for gr in allowed:
                s_vals = (ref_s - gl, ref_s, ref_s + gr)
                if s_vals[0] < lo or s_vals[-1] > hi:
                    continue
                if gl == gr:
                    continue
                patterns.append(s_vals)
    else:
        for g0 in allowed:
            for g1 in allowed:
                for g2 in allowed:
                    for g3 in allowed:
                        s_vals = (
                            ref_s - g1 - g0,
                            ref_s - g1,
                            ref_s,
                            ref_s + g2,
                            ref_s + g2 + g3,
                        )
                        if s_vals[0] < lo or s_vals[-1] > hi:
                            continue
                        gaps = [s_vals[i + 1] - s_vals[i] for i in range(4)]
                        if len(set(gaps)) == 1:
                            continue
                        patterns.append(s_vals)
    return random.choice(patterns) if patterns else None


def _choose_lengths(ref_len, caps, n):
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
        if len(options) > 1 and ref_len in options:
            options.remove(ref_len)
        lengths[i] = random.choice(options or [ref_len])
    return lengths


def _paint_ts(grid, d, s_vals, starts, lengths, colors):
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _paint_starts(grid, d, starts, length, colors):
    for start, color in zip(starts, colors):
        cells = _cells_from_start(start, d, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _compact_starts(ref_start, d, n):
    pr, pc = _touching_offset(d)
    ref_idx = n // 2
    return [
        (ref_start[0] + (i - ref_idx) * pr, ref_start[1] + (i - ref_idx) * pc)
        for i in range(n)
    ]


def _reference_start_candidates(d, n, ref_len):
    cands = []
    for r in range(_H):
        for c in range(_W):
            ref_start = (r, c)
            if _cells_from_start(ref_start, d, ref_len) is None:
                continue
            starts = _compact_starts(ref_start, d, n)
            if all(_cells_from_start(start, d, ref_len) is not None for start in starts):
                cands.append(ref_start)
    return cands


def _gen_attempt():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    ref_starts = _reference_start_candidates(d, n, ref_len)
    if not ref_starts:
        return None

    ref_start = random.choice(ref_starts)
    ref_t0 = _dot(ref_start, d)
    ref_s = _perp_dot(ref_start, d)
    s_vals = _choose_sparse_offsets(d, n, ref_s)
    if s_vals is None:
        return None

    anchor = random.choice(["start", "end"])
    delta = _step(d)
    ref_t1 = ref_t0 + (ref_len - 1) * delta
    if anchor == "start":
        caps = [_max_len_from(d, ref_t0, s) for s in s_vals]
    else:
        caps = [_max_len_to(d, ref_t1, s) for s in s_vals]
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    colors = tuple(random.sample(_COLORS, n))

    out = np.full((_H, _W), _BG, dtype=int)
    packed_starts = _compact_starts(ref_start, d, n)
    if not _paint_starts(out, d, packed_starts, ref_len, colors):
        return None

    if anchor == "start":
        starts = [ref_t0] * n
    else:
        starts = [ref_t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_ts(inp, d, s_vals, starts, lengths, colors):
        return None

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
