# Combo variant of e376de54 (index 22)
# combo_spec: D2/0+D3/0
# Name: 水平线段+头端对齐

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    - The fused task keeps the original object family: several monochrome parallel segments on a
#      16x16 background, with an odd count so one segment is the median by perpendicular offset.
#    - D2/0 removes the direction choice: every meaningful segment is horizontal, so the
#      perpendicular coordinate is simply the row index.
#    - D3/0 removes the old "match both endpoints" redraw. The output only aligns the left endpoint
#      of every segment to the reference segment's left endpoint, while each segment keeps its own
#      length.
#    - Concrete trace: suppose rows [2, 6, 9] hold lengths [3, 5, 8], and all three input segments
#      end at column 10. Their starts are therefore [8, 6, 3]. The median row is 6, so the
#      reference start is column 6. The base task would redraw all three as length 5 at columns
#      6..10. This combo redraws them at starts all equal to 6 but keeps lengths [3, 5, 8], so the
#      new output spans [6..8], [6..10], [6..13]. The visible signature is "left edges flush, right
#      edges staggered".
# b) Complexity calibration
#    - Restricting to horizontal only could over-simplify the task if the input were already
#      left-aligned. To prevent near-identity examples, the generator always makes the input
#      right-aligned instead. Then the output visibly switches from shared right ends to shared left
#      ends.
#    - The original segment-count set {3, 5}, color palette size {1..3}, and irregular parallel
#      spacing are preserved. This keeps enough structure for humans to spot the median reference and
#      prevents the fused variant from collapsing into a trivial "shift all bars" exercise.
# c) Hidden assumption audit against the original solver
#    - Original direction detection assumed one dominant direction among four candidates. That still
#      holds, but now the solver must reject non-horizontal dominance because the combo spec fixes the
#      source geometry to horizontal segments only.
#    - Original reference selection used perpendicular-dot coordinates. Under horizontal-only inputs,
#      that reduces to unique rows; the solver therefore relies on one multi-cell segment per chosen
#      row and an odd number of distinct rows.
#    - Original redraw used the reference length for every segment. That assumption is invalid here:
#      the redraw must preserve each segment's extracted length.
#    - Clearing original cells remains valid because all objects are still disjoint segments and
#      isolated single pixels are still irrelevant noise.
# d) Placement legality
#    - Rows are chosen with gaps at least 2, so no two horizontal segments touch vertically or
#      diagonally. This makes the horizontal direction uniquely detectable even if colors repeat.
#    - The output places every segment with a shared left column c0. The input places the same
#      segments with a shared right column c1 = c0 + ref_len - 1. Each chosen length must fit both
#      to the right from c0 and to the left from c1, so the usable per-row capacity is
#      min(W - c0, c1 + 1).
#    - Because the generator keeps all segments on distinct rows, repainting cannot create overlaps or
#      crossings.
# e) Visual observability
#    - Every generated puzzle forces at least one non-reference segment shorter than the reference and
#      at least one longer. Therefore the output always has multiple right endpoints, which separates
#      this combo from the base task where all right endpoints would match.
#    - Using only horizontal geometry makes the "head-end" concept visually legible: humans can read
#      it directly as "left-align these bars".
# f) Visual cleanliness
#    - The rows are sparse and irregularly spaced, so the bars remain visually separable.
#    - No extra markers, recoloring, or background decorations are introduced; the scene stays clean
#      enough that the shift itself is the dominant cue.
# g) Change inferability
#    - Colors do not encode order or mapping. They merely tag segments across input/output, which is
#      enough for humans to see that each bar keeps its own color and length while only its left edge
#      snaps to the reference left edge.
# h) Quantified non-degeneracy target used after coding
#    - We will test that generated outputs have more than one distinct segment length and that they
#      differ from the original base behavior on the same input. If both checks failed often, the
#      generator would be degenerating back to the original task.
# i) New solver assumptions that the generator must guarantee
#    - The solver now assumes: the dominant direction is horizontal; the count of multi-cell segments
#      is odd; their rows are distinct and separated by at least one blank row; and shifting every
#      segment start to the reference start keeps the preserved length in bounds. The generator below
#      constructs exactly that regime.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]


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


def _extract_segments(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            c = 0
            while c < w:
                if not mask[r, c]:
                    c += 1
                    continue
                if c > 0 and mask[r, c - 1]:
                    c += 1
                    continue
                cc = c
                cells = []
                while cc < w and mask[r, cc]:
                    cells.append((r, cc))
                    cc += 1
                if len(cells) >= 2:
                    segments.append(
                        {
                            "color": col,
                            "cells": cells,
                            "row": r,
                            "start_col": c,
                            "len": len(cells),
                        }
                    )
                c = cc
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reposition several horizontal colored segments so their left endpoints line up with one reference bar.

    Input structure:
    - The background is the most common color.
    - All meaningful multi-cell objects are horizontal line segments, possibly reusing colors across rows.
    - The segments occupy distinct rows with gaps between rows, and isolated single pixels are ignored.

    Rule:
    - Detect the unique dominant direction and require it to be horizontal.
    - Extract every horizontal segment of length at least two, then choose the segment on the median row as the reference.
    - Clear the original segment cells, shift each segment left or right so its left endpoint matches the reference left endpoint, and keep that segment's original length unchanged.
    - Redraw the shifted bars on the same rows; only the shared left edge is normalized.

    Key insight: The median segment provides a common left anchor, not a common width. The output is therefore a set of left-aligned horizontal bars whose right ends remain uneven.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    assert d == (0, 1), "combo variant expects horizontal dominance"

    segments = _extract_segments(g, bg)
    assert segments, "at least one horizontal multi-cell segment must exist"

    rows = sorted(s["row"] for s in segments)
    assert len(rows) % 2 == 1, "segment count must have a single median"
    assert len(rows) == len(set(rows)), "segment rows must be unique"
    for i in range(len(rows) - 1):
        assert rows[i + 1] - rows[i] >= 2, "segment rows must be visually separated"

    ref_row = rows[len(rows) // 2]
    ref_cands = [s for s in segments if s["row"] == ref_row]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_c0 = ref["start_col"]

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        row = s["row"]
        c0 = ref_c0
        c1 = c0 + s["len"] - 1
        assert 0 <= row < h and 0 <= c0 <= c1 < w, "shifted segment must stay in bounds"
        out[row, c0 : c1 + 1] = s["color"]

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Colors and sizes ---
_COLORS = [1, 2, 3, 5, 6, 8, 9]
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Row spacing ---
_ROW_GAPS = [2, 3, 4, 5, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_GEN_RETRIES = 300


def _gap_patterns(n: int) -> list[tuple[int, ...]]:
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [gap] for p in patterns for gap in _ROW_GAPS]
    return [
        tuple(p)
        for p in patterns
        if sum(p) <= _H - 1 and ((n == 3 and p[0] != p[1]) or (n == 5 and len(set(p)) > 1))
    ]


def _choose_rows(n: int) -> tuple[int, ...]:
    gaps = random.choice(_gap_patterns(n))
    r0 = random.randint(0, _H - 1 - sum(gaps))
    rows = [r0]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return tuple(rows)


def _feasible_left_starts(ref_len: int) -> list[tuple[int, int]]:
    placements = []
    for c0 in range(_W - ref_len + 1):
        c1 = c0 + ref_len - 1
        cap = min(_W - c0, c1 + 1)
        if cap > ref_len:
            placements.append((c0, cap))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, cap: int, n: int) -> list[int] | None:
    limit = min(_MAX_SEG_LEN, cap)
    if limit <= ref_len:
        return None

    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]

    long_i = random.choice(nonrefs)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, limit)
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, limit + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint(grid: np.ndarray, rows: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...]) -> bool:
    for row, c0, length, color in zip(rows, starts, lengths, colors):
        c1 = c0 + length - 1
        if not (0 <= row < _H and 0 <= c0 <= c1 < _W):
            return False
        grid[row, c0 : c1 + 1] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    rows = _choose_rows(n)
    placements = _feasible_left_starts(ref_len)
    if not placements:
        return None

    ref_c0, cap = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, cap, n)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, rows, [ref_c0] * n, lengths, colors):
        return None

    ref_c1 = ref_c0 + ref_len - 1
    starts = [ref_c1 - length + 1 for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, rows, starts, lengths, colors):
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
