# Combo variant of e376de54 (index 13)
# combo_spec: D1/2+D12/1
# Name: 中位数基准+大量线段

"""
Design analysis for the combo variant
====================================

spec_like_summary:
  grid: 16x16, background 7
  objects:
    - 7 or 8 parallel line segments, all in one shared direction
    - direction is one of horizontal / vertical / diagonal / anti-diagonal
    - every segment is a solid run of length >= 2 on its own perpendicular offset
    - all segment lengths are distinct, so the low-median length identifies one unique reference
    - input keeps one anchored endpoint shared by all segments: either same start or same end
    - colors come from a small 2-3 color palette and may repeat across different offsets
  transformation:
    - sort segment lengths and choose the low median:
      odd count -> middle length
      even count -> shorter of the two middle lengths
    - use that segment's span as the target span
    - shift every segment along the common direction so its start matches the reference start
    - redraw every segment at the reference length

a) concrete behavior trace:
  - Example: seven horizontal segments all end at column 13, with distinct lengths
    [2, 4, 5, 7, 8, 10, 11] from top to bottom in some scrambled spatial order.
    The old summary rule "second-longest" would copy the 10-cell span.
    The provided baseline code would instead copy the segment at the median perpendicular offset.
    This combo must copy the 7-cell segment because 7 is the median length.
  - With eight segments, the contrast is even stronger: lengths might be
    [2, 3, 5, 6, 8, 9, 11, 12]. The reference is 6, not 11, and not a spatial middle line.
    After solving, all eight lines align to the 6-cell span.

b) complexity calibration:
  - The dense-scene change is real only when the count rises well above 5, so this generator uses
    7 or 8 segments. Under the inherited 16x16 canvas and minimum gap rules, 9-10 segments are not
    reliably placeable without breaking the original spacing discipline, so 7-8 is the maximal
    dense regime that still preserves clean ARC visuals.
  - Distinct lengths are mandatory here. Otherwise the low median could tie, making the reference
    ambiguous and weakening the intended "count and sort many segments" burden.

c) implicit-assumption audit for main():
  - Direction detection still assumes one direction is uniquely dominant. The generator keeps all
    multi-cell objects parallel and separated enough that no rival direction accumulates equal runs.
  - Segment extraction assumes each visible object is a single contiguous 1-cell-thick run. The
    generator only paints legal straight segments and ignores isolated pixels.
  - The baseline reference-by-offset assumption is invalid now. The new code replaces it with a
    reference-by-length assumption and therefore requires a unique median length.
  - Redrawing assumes that shifting every segment onto the reference start keeps all cells in
    bounds. The output-first generator guarantees this by constructing a legal shared target span
    before deriving the input.

d) placement legality:
  - Offsets are distinct and gap-controlled, so different segments never overlap.
  - The output state is placed first. Input lengths are then derived by shortening or lengthening
    from a shared anchor, which preserves the same parallel lines and keeps every segment on-grid.
  - Longer-than-reference assignments are capacity-checked per offset, so dense scenes do not push
    any segment past the border.

e) visual observability:
  - The rule depends only on visible segment length, never on color ids or hidden ordering.
  - The reference segment is forced away from the central offset line(s), so the fused rule does
    not collapse back into "pick the middle line in the stack".
  - Even-count cases appear as well, making the "low median" convention observable instead of
    accidentally matching a simpler odd-only median heuristic.

f) visual cleanliness:
  - The scene contains only one family of parallel segments on a blank background.
  - Colors are limited to 2-3 values and spacing remains generous for the canvas size, so the
    larger segment count looks dense but still countable by eye.

g) color-change inference:
  - There is no color remapping. Colors only tag segments, so all reasoning effort stays on length
    comparison and the final shared span.
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


def _low_median_index(n: int) -> int:
    return (n - 1) // 2


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
    Snap a dense bundle of parallel segments onto the span of the low-median length segment.

    Input structure:
    - The most frequent color is background.
    - Foreground cells form 7-8 straight parallel segments in one shared direction.
    - Segments live on distinct parallel lines, may reuse colors, and have visibly different lengths
      while sharing one anchored endpoint in the input.

    Rule:
    - Detect the common direction and extract every contiguous segment of length at least 2.
    - Sort segment lengths and choose the low median as the reference
      (middle value for odd counts, shorter middle value for even counts).
    - Take the unique segment with that length, align every segment start to its start, and redraw
      each segment at exactly the reference length.
    - Keep the canvas size unchanged and leave isolated single pixels untouched.

    Key insight:
    The target span comes from a visible order statistic over many lengths, not from spatial order:
    once the low-median length segment is identified, every parallel segment is cropped or extended
    to match that segment's extent.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    lengths = sorted(s["len"] for s in segments)
    target_len = lengths[_low_median_index(len(lengths))]
    ref_cands = [s for s in segments if s["len"] == target_len]
    assert len(ref_cands) == 1, "median-length reference segment must be unique"
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


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [7, 8]
_REF_LEN_RANGE = (5, 7)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MIN_DISTINCT_COLORS = 2
_MAX_DISTINCT_COLORS = 3

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_OFFSET_RETRIES = 500
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


def _pick_gap_sequence(d: tuple[int, int], n: int) -> list[int] | None:
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    for _ in range(_OFFSET_RETRIES):
        gaps = [random.choice(allowed) for _ in range(n - 1)]
        if sum(gaps) > max_span:
            continue
        if len(set(gaps)) == 1:
            continue
        return gaps
    return None


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...] | None:
    lo, hi = _s_limits(d)
    for _ in range(_OFFSET_RETRIES):
        gaps = _pick_gap_sequence(d, n)
        if gaps is None:
            return None
        s0 = random.randint(lo, hi - sum(gaps))
        s_vals = [s0]
        for gap in gaps:
            s_vals.append(s_vals[-1] + gap)
        return tuple(s_vals)
    return None


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        placements.append((t0, "start", [_max_len_from(d, t0, s) for s in s_vals]))
        placements.append((t0, "end", [_max_len_to(d, t1, s) for s in s_vals]))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(_MIN_DISTINCT_COLORS, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _assign_distinct_values(indices: tuple[int, ...], choices: dict[int, list[int]]) -> dict[int, int] | None:
    order = sorted(indices, key=lambda idx: len(choices[idx]))
    used = set()
    assigned: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        vals = choices[idx][:]
        random.shuffle(vals)
        for val in vals:
            if val in used:
                continue
            used.add(val)
            assigned[idx] = val
            if backtrack(pos + 1):
                return True
            del assigned[idx]
            used.remove(val)
        return False

    if backtrack(0):
        return assigned.copy()
    return None


def _choose_lengths(ref_len: int, caps: list[int], ref_idx: int) -> list[int] | None:
    n = len(caps)
    short_count = _low_median_index(n)
    short_values = list(range(2, ref_len))
    if len(short_values) < short_count:
        return None

    nonrefs = [i for i in range(n) if i != ref_idx]
    short_choices = list(itertools.combinations(nonrefs, short_count))
    random.shuffle(short_choices)

    long_options = {
        i: list(range(ref_len + 1, min(_MAX_SEG_LEN, caps[i]) + 1))
        for i in nonrefs
    }

    for short_idx in short_choices:
        long_idx = tuple(i for i in nonrefs if i not in short_idx)
        if any(not long_options[i] for i in long_idx):
            continue

        lengths = [None] * n
        lengths[ref_idx] = ref_len

        chosen_shorts = random.sample(short_values, short_count)
        random.shuffle(chosen_shorts)
        for i, val in zip(short_idx, chosen_shorts):
            lengths[i] = val

        assigned_longs = _assign_distinct_values(long_idx, {i: long_options[i] for i in long_idx})
        if assigned_longs is None:
            continue
        for i, val in assigned_longs.items():
            lengths[i] = val

        assert all(val is not None for val in lengths)
        assert len(set(lengths)) == n
        return [int(val) for val in lengths]
    return None


def _eligible_ref_indices(n: int) -> list[int]:
    if n % 2 == 1:
        blocked = {n // 2}
    else:
        blocked = {n // 2 - 1, n // 2}
    return [i for i in range(n) if i not in blocked]


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
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    if s_vals is None:
        return None

    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    ref_candidates = _eligible_ref_indices(n)
    random.shuffle(ref_candidates)
    random.shuffle(placements)

    for t0, anchor, caps in placements:
        for ref_idx in ref_candidates:
            lengths = _choose_lengths(ref_len, caps, ref_idx)
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
                solved = main(inp)
            except Exception:
                continue
            if np.array_equal(solved, out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
