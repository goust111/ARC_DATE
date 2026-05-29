# Combo variant of e376de54 (index 1)
# combo_spec: D1/0+D4/0
# Name: 最长基准+仅延伸

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "solvers"
        if candidate.exists():
            sys.path.append(str(candidate))
            break
    from arc_utils import get_bg

# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - The grid still shows 3 or 5 separated parallel single-color segments in one dominant direction,
#      with all segments sharing either their start or their end in the input.
#    - Original behavior: the reference was chosen by median perpendicular offset, then every segment
#      was shifted to that start and redrawn to exactly the median segment length, so shorter segments
#      extended and longer ones were cropped.
#    - Combo behavior: choose the unique longest segment instead, and keep the one-way transform rule
#      "only extend". Because the reference is already the longest, no segment can exceed it, so every
#      non-reference segment only grows to the longest extent after alignment.
#    - Example: three horizontal segments on rows 2, 6, 10 all start at column 3 with lengths
#      [7, 4, 2], and the row-2 segment is the unique longest while the median-offset row is row 6.
#      Old logic outputs three length-4 segments. This combo outputs three length-7 segments. The D4
#      half is still real: the shorter rows only extend; no row is ever shortened.
# b) Complexity calibration:
#    - Keep the original 16x16 canvas, four candidate directions, odd segment counts, irregular track
#      spacing, and 1-3 colors so the puzzle remains recognizably in the same family.
#    - To avoid the combo collapsing visually back into the original task, the generator forces the
#      longest segment to be unique and not on the median offset track.
#    - To make "only extend" observable rather than vacuous, every non-reference segment is strictly
#      shorter than the longest one, so the output visibly lengthens multiple runs and never removes
#      any tail pixels.
# c) Implicit assumptions review:
#    - Direction detection still depends on one unique dominant same-color adjacency direction; the
#      generator keeps all meaningful runs parallel and well separated so no tie appears.
#    - Segment extraction assumes each useful object is a contiguous run of length >= 2 on one track.
#      The generator paints only such runs and rejects overlap.
#    - Reference selection no longer uses the median offset at all; main() now assumes a unique
#      longest run exists, so the generator enforces exactly one maximum length.
#    - The transform loop must not silently rely on cropping. In the combo, draw_len uses the
#      one-sided rule max(original_len, target_len), and the unique-longest guarantee means this
#      simplifies to target_len for every generated case.
# d) Placement legality:
#    - Output-first construction paints the final aligned longest extent first, guaranteeing that the
#      shared target interval fits on every perpendicular offset.
#    - Input segments are then chosen as anchored subsets of that interval. Since all input lengths are
#      <= the reference length, both start-anchored and end-anchored variants remain in bounds.
#    - Offsets are distinct and diagonals use gaps >= 4, so neither input nor output segments touch or
#      overlap.
# e) Visual observability:
#    - A solver can see that the final common extent matches the visibly longest input segment rather
#      than the central track. Because that longest track is forced away from the median, samples do
#      not support the old rule.
#    - The output never deletes pixels relative to any aligned segment. All changes are forward fills
#      on shorter runs, which makes the "only extend" directionality human-visible.
# f) Visual cleanliness:
#    - The scene remains a sparse set of parallel bands with generous spacing. There are no auxiliary
#      markers, no mixed-color segments, and no local clutter near the aligned ends.
# g) Change inferability:
#    - Colors are preserved and carry no semantic ordering. The only changing attributes are segment
#      position along the shared direction and segment length, both directly visible from the grids.
# h) New solver assumptions and generator guarantees:
#    - main() assumes there is at least one multi-cell segment, exactly one longest segment, and one
#      segment per perpendicular offset. generate_input() guarantees all three.
#    - main() also assumes the aligned target interval stays in bounds for every offset. The
#      output-first generator guarantees that by first placing the full longest span on every track.

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
    Align several parallel colored segments to the span of the unique longest segment, extending only where needed.

    Input structure:
    - The background is the most frequent color.
    - Foreground objects are multiple multi-cell straight segments on distinct parallel tracks in one
      shared horizontal, vertical, or diagonal direction.
    - The segments have varied lengths and share one anchored endpoint pattern in the input; isolated
      single pixels may also exist and should be left alone.

    Rule:
    - Detect the common direction and trace every multi-cell segment in that direction.
    - Choose the unique longest segment as the reference, taking its start position and full span as
      the aligned target extent.
    - Shift every segment so its start matches the reference start.
    - Apply the one-way length rule: extend segments that are shorter than the reference, but do not
      crop any segment that is already at least that long.

    Key insight:
    - The puzzle combines two inferences at once: the reference is the longest visible run, and the
      transform is one-sided. In generated cases those reinforce each other, so every non-reference
      segment grows to the longest span and none are shortened.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = [s["s"] for s in segments]
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    if delta_t == 2:
        s_sorted = sorted(s_vals)
        for i in range(len(s_sorted) - 1):
            assert s_sorted[i + 1] - s_sorted[i] >= 4

    max_len = max(s["len"] for s in segments)
    ref_cands = [s for s in segments if s["len"] == max_len]
    assert len(ref_cands) == 1, "reference segment must be the unique longest"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

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
        draw_len = max(s["len"], target_len)
        for _ in range(draw_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (4, 8)
_MAX_DISTINCT_COLORS = 3

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400
_SAME_SHORT_LEN_AVOID_PROB = 0.75


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
        if sum(p) <= max_span and ((n == 3 and p[0] != p[1]) or (n == 5 and len(set(p)) > 1))
    ]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_t0s(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[int]:
    feasible = []
    for t0 in range(*_T_SCAN_RANGE):
        if all(_segment_cells(d, t0, s, ref_len) is not None for s in s_vals):
            feasible.append(t0)
    return feasible


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_ref_idx(n: int) -> int:
    median_idx = n // 2
    return random.choice([i for i in range(n) if i != median_idx])


def _choose_lengths(ref_len: int, n: int, ref_idx: int) -> list[int]:
    lengths = [None] * n
    lengths[ref_idx] = ref_len
    short_pool = list(range(2, ref_len))
    assert short_pool, "reference length must leave room for shorter segments"
    for i in range(n):
        if i == ref_idx:
            continue
        options = short_pool.copy()
        if len(options) > 1 and random.random() < _SAME_SHORT_LEN_AVOID_PROB:
            forbidden = max(2, ref_len - 1)
            if forbidden in options:
                options.remove(forbidden)
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
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _old_median_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc
    segments = _extract_segments(g, bg, d)
    s_vals = sorted(s["s"] for s in segments)
    s_ref = s_vals[len(s_vals) // 2]
    ref = [s for s in segments if s["s"] == s_ref][0]
    target_len = (ref["t_end"] - ref["t_start"]) // delta_t + 1
    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg
    for s in segments:
        dt = ref["t_start"] - s["t_start"]
        k = dt // delta_t
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
        for _ in range(target_len):
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc
    return out


def _build_case():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    t0_choices = _feasible_t0s(d, s_vals, ref_len)
    if not t0_choices:
        return None

    t0 = random.choice(t0_choices)
    anchor = random.choice(["start", "end"])
    ref_idx = _choose_ref_idx(n)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, n, ref_idx)

    out = np.full((_H, _W), _BG, dtype=int)
    out_lengths = [ref_len] * n
    if not _paint(out, d, s_vals, [t0] * n, out_lengths, colors):
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

    meta = {
        "direction": d,
        "anchor": anchor,
        "ref_idx": ref_idx,
        "median_idx": n // 2,
        "ref_len": ref_len,
        "input_lengths": tuple(lengths),
        "all_nonrefs_shorter": all(lengths[i] < ref_len for i in range(n) if i != ref_idx),
    }
    return inp, out, meta


def _gen_attempt():
    case = _build_case()
    if case is None:
        return None
    inp, out, meta = case
    try:
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        if np.array_equal(_old_median_output(inp), out):
            return None
        if meta["ref_idx"] == meta["median_idx"]:
            return None
        if not meta["all_nonrefs_shorter"]:
            return None
        return inp
    except Exception:
        return None


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
