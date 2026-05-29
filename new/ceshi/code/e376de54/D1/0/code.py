# Variant of e376de54
# Dimension: D1 — 匹配条件：以第二长线段为参考基准 vs 其他选取规则 (match_cond)
# Change: 以第二长线段的端点范围为参考，所有其他线段的端点对齐到该参考线段的起止位置... → 以最长线段为参考基准

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg

# Design analysis for this variant
# a) Concrete behavior and trace:
#    - The grid still contains 3 or 5 parallel monochrome segments on distinct lines with a shared
#      direction. They still share one anchored endpoint in the input.
#    - The only intended conceptual change is how the target extent is chosen:
#      original code used the segment at the median perpendicular offset; this variant uses the
#      unique longest segment.
#    - Example: three horizontal segments at rows 2, 6, 10 all start in column 3, with lengths
#      [7, 4, 2]. Old logic would treat the middle row as the reference and output three length-4
#      segments. New logic treats the top row as the unique longest reference and outputs three
#      length-7 segments. The visible change is the amount of extension/cropping, not the direction.
# b) Complexity calibration:
#    - Keep the original direction set, canvas size, irregular spacing, and shared anchor rule so the
#      puzzle still reads like the same family.
#    - Force the longest segment to be unique and to sit away from the median offset; otherwise many
#      samples would accidentally match the old rule and the D1 change would be visually weak.
# c) Implicit assumptions review:
#    - Direction detection still assumes one direction has the largest same-color adjacency count.
#      Generator keeps all non-background runs parallel and sufficiently separated.
#    - Segment extraction assumes each useful object is a contiguous run of length >= 2. Generator
#      only paints such runs and never overlaps them.
#    - Reference selection now depends on there being exactly one maximum length. Generator enforces a
#      unique longest segment and makes every other segment strictly shorter.
#    - Repainting clears only detected segment cells, so isolated pixels would survive; generator does
#      not add isolated pixels, but solver remains compatible with them.
# d) Placement legality:
#    - Output-first construction places the full target extent for every offset first, guaranteeing the
#      output fits in bounds. Input segments are then shortened subsets of those output cells, so they
#      also stay in bounds and never overlap.
# e) Visual observability:
#    - In every generated sample, the longest segment is not the median-offset segment, and the median
#      segment is strictly shorter. A human can see that the final common extent matches the visibly
#      longest input run rather than the centered line.
# f) Visual cleanliness:
#    - Offsets remain distinct with irregular gaps; diagonal gaps stay large enough to avoid touching.
#      No extra markers or color remapping are introduced.
# g) Color inference:
#    - Colors are unchanged by the transformation. They only distinguish segments, so there is no
#      hidden color-based rule.
# h) New solver assumptions and generator guarantees:
#    - main() assumes the longest segment is unique; generate_input() guarantees this.
#    - main() assumes shifting every segment to the reference extent stays in bounds; output-first
#      placement guarantees the target extent is valid for all offsets.

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
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
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
    Normalize several parallel colored segments so they all occupy the same extent as one reference run.

    Input structure:
    - The background is the most frequent color.
    - Foreground objects are multiple parallel straight segments, each lying on its own horizontal,
      vertical, or diagonal track, with varied lengths and one shared anchored endpoint direction.

    Rule:
    - Detect the common direction of the segments and extract each multi-cell run.
    - Choose the unique longest segment as the reference, using its start and end positions along the
      shared direction as the target extent.
    - Move every segment along that direction so its start matches the reference start, then redraw it
      with the reference length. Grid size stays unchanged and isolated pixels remain untouched.

    Key insight:
    - The target range is determined by the visibly longest segment, not by where a segment sits among
      the parallel tracks. Every segment is extended or trimmed to that longest span.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"
    assert len({s["s"] for s in segments}) == len(segments), "segment line coordinates must be unique"

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
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_DISTINCT_COLORS = 3
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


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


def _s_limits(d):
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d, n):
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


def _choose_offsets(d, n):
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_t0s(d, s_vals, ref_len):
    feasible = []
    for t0 in range(*_T_SCAN_RANGE):
        if all(_segment_cells(d, t0, s, ref_len) is not None for s in s_vals):
            feasible.append(t0)
    return feasible


def _choose_colors(n):
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_ref_idx(n):
    median_idx = n // 2
    options = [i for i in range(n) if i != median_idx]
    return random.choice(options)


def _choose_lengths(ref_len, n, ref_idx):
    lengths = [None] * n
    lengths[ref_idx] = ref_len
    for i in range(n):
        if i == ref_idx:
            continue
        lengths[i] = random.randint(2, ref_len - 1)
    return lengths


def _paint(grid, d, s_vals, starts, lengths, colors):
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


def _gen_attempt():
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
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
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

    try:
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        if np.array_equal(_old_median_output(inp), out):
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
