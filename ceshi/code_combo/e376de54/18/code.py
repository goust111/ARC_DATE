# Combo variant of e376de54 (index 18)
# combo_spec: D5/0+D12/1
# Name: 第三长基准+大量线段
"""
Design analysis for the combo variant
=====================================

combo_spec:
  invariant_structure:
    - The scene is still a 16x16 grid whose background is the most frequent color.
    - Every meaningful object is a contiguous multi-cell segment on one shared
      direction: horizontal, vertical, diagonal \\, or diagonal /.
    - Segments stay on distinct parallel lines and the output still snaps every
      segment to one common start position and one common final length.
  fused_change:
    - Replace the old reference heuristic with a pure length ranking:
      sort all segment lengths from longest to shortest and pick rank 3.
    - Increase the crowding from a sparse 3-5 lines to a dense 7-10 lines.
    - These two changes interact: with 7-10 segments, the solver cannot rely on
      "middle line" or any shallow comparison, because the target is hidden
      inside a longer ordering chain across many parallel objects.

  concrete_behavior_trace:
    - Example input: 8 horizontal segments with anchored left endpoints and
      pairwise-distinct lengths [12, 11, 9, 8, 7, 5, 4, 2].
    - A second-longest rule would align everything to length 11.
    - This fused rule aligns everything to length 9 instead, because 12 > 11 > 9.
    - With dense spacing, the visual task is now "find the third entry in a
      crowded ranking", not "spot the obvious long line" or "use positional order".

  complexity_calibration:
    - Dense counts force the generator to tighten perpendicular gaps; otherwise
      9-10 segments do not fit on a 16x16 canvas.
    - Reference lengths are raised with segment count so there are enough
      distinct shorter lengths below the reference and at least two distinct
      longer lengths above it.
    - Colors are made mostly unique so the crowded layout remains readable and
      the dominant direction stays unambiguous even when gaps are small.

  implicit_assumption_audit:
    - _pick_main_dir assumes one direction has the unique maximum count of
      same-color adjacencies. Dense layouts could break this if nearby parallel
      lines reused the same color, so generation avoids local same-color repeats.
    - Segment extraction assumes one contiguous run per color-line pair. The
      generator paints exactly one run on each chosen offset.
    - Third-longest selection assumes the ranking is strict. Generation enforces
      pairwise-distinct lengths, so rank 3 maps to exactly one segment.
    - Redrawing assumes shifting every segment to the reference start stays in
      bounds. The generator builds a legal aligned output first and derives the
      input by changing only lengths while preserving one anchored endpoint.

  placement_legality:
    - Output-first construction guarantees the target arrangement is in bounds.
    - Offsets are chosen from dense-but-separated gap patterns so segments never
      overlap or merge into thicker shapes.
    - Feasible placements are filtered by forward/backward capacity, and length
      assignment uses only values that fit each segment's anchor direction.

  visual_observability:
    - The key variable is fully visible: line length measured in pixels.
    - Samples always contain 7-10 segments, so the crowding difference from the
      sparse original is obvious to a human solver.
    - Because all lengths are distinct, third-longest and second-longest never
      coincide; the ranked-length rule leaves a visible signature in the output.

  cleanliness_and_inference:
    - Dense scenes can become messy if every line has the same color or if
      diagonal lines are packed too tightly. The generator counters that with
      mostly unique colors and modest reference-length ranges.
    - No hidden metadata is used: color ids, offset order, and line count do
      not determine the target directly. The intended inference remains
      "identify the third-longest segment, then copy its span to all others".
"""

import itertools
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


def _extract_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
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
    Align a crowded family of parallel colored segments to the span of the third-longest segment.

    Input structure:
    - The background is the most frequent color.
    - Seven to ten multi-cell segments lie on distinct parallel lines in one shared direction.
    - Segment lengths are all different, and isolated single pixels are irrelevant clutter that must stay untouched.

    Rule:
    - Detect the common direction and extract every contiguous segment of length at least 2.
    - Sort those segments by length from longest to shortest and use the third item as the reference.
    - Shift every segment along the shared direction until its start matches the reference start, then redraw it with the reference length.
    - Keep the canvas size unchanged and only rewrite the segment pixels themselves.

    Key insight: the target is determined by a deep length ranking inside a dense set of lines; neither spatial order nor color chooses the reference.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc
    segments = _extract_segments(g, bg, d)

    assert len(segments) >= 3, "at least three multi-cell segments are required"

    s_vals = sorted(s["s"] for s in segments)
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    lengths_desc = sorted((s["len"] for s in segments), reverse=True)
    assert len(lengths_desc) == len(set(lengths_desc)), "segment lengths must be unique"
    target_len = lengths_desc[2]

    ref_cands = [s for s in segments if s["len"] == target_len]
    assert len(ref_cands) == 1, "third-longest reference segment must be unique"
    ref = ref_cands[0]

    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

_SEGMENT_COUNTS = [7, 8, 9, 10]
_REF_LEN_RANGE_BY_COUNT = {
    7: (6, 9),
    8: (7, 9),
    9: (8, 10),
    10: (9, 10),
}
_MAX_SEG_LEN = 13

_PERP_GAP_ORTHO = [1, 2]
_PERP_GAP_DIAG = [2, 4]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 800


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


def _gap_patterns(d: tuple[int, int], n: int) -> list[tuple[int, ...]]:
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [gap] for p in patterns for gap in allowed]
    return [tuple(p) for p in patterns if sum(p) <= max_span and len(set(p)) > 1]


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
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta

        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if sum(cap > ref_len for cap in start_caps) >= 2:
            placements.append((t0, "start", start_caps))

        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if sum(cap > ref_len for cap in end_caps) >= 2:
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    if n <= len(_COLORS):
        colors = random.sample(_COLORS, n)
        return tuple(colors)

    palette = random.sample(_COLORS, len(_COLORS))
    return tuple(palette + [palette[0]])


def _choose_lengths(ref_len: int, caps: list[int]) -> list[int] | None:
    n = len(caps)
    short_pool = list(range(2, ref_len))
    if len(short_pool) < n - 3:
        return None

    ref_candidates = list(range(n))
    random.shuffle(ref_candidates)
    for ref_idx in ref_candidates:
        longable = [i for i in range(n) if i != ref_idx and min(_MAX_SEG_LEN, caps[i]) > ref_len]
        if len(longable) < 2:
            continue

        pair_order = list(itertools.combinations(longable, 2))
        random.shuffle(pair_order)
        for i, j in pair_order:
            opts_i = list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[i]) + 1))
            opts_j = list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[j]) + 1))
            long_pairs = [(a, b) for a in opts_i for b in opts_j if a != b]
            if not long_pairs:
                continue

            lengths = [None] * n
            lengths[ref_idx] = ref_len
            a, b = random.choice(long_pairs)
            lengths[i] = a
            lengths[j] = b

            short_vals = random.sample(short_pool, n - 3)
            random.shuffle(short_vals)
            short_idxs = [idx for idx in range(n) if idx not in (ref_idx, i, j)]
            for idx, value in zip(short_idxs, short_vals):
                lengths[idx] = value

            assert all(v is not None for v in lengths)
            assert len(set(lengths)) == n
            assert sorted(lengths, reverse=True)[2] == ref_len
            return [int(v) for v in lengths]
    return None


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
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE_BY_COUNT[n])
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    random.shuffle(placements)
    for t0, anchor, caps in placements[:24]:
        lengths = _choose_lengths(ref_len, caps)
        if lengths is None:
            continue
        colors = _choose_colors(n)

        out = np.full((_H, _W), _BG, dtype=int)
        if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
            continue

        delta = _step(d)
        t1 = t0 + (ref_len - 1) * delta
        if anchor == "start":
            starts = [t0] * n
        else:
            starts = [t1 - (length - 1) * delta for length in lengths]

        inp = np.full((_H, _W), _BG, dtype=int)
        if not _paint(inp, d, s_vals, starts, lengths, colors):
            continue

        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
