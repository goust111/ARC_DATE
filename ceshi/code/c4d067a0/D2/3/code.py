# Variant of c4d067a0
# Dimension: D2 — 源对象：图例的结构与位置 (src)
# Change: 左上角稀疏像素模式（非背景色像素以偶数间隔排列在奇数行/列上），每个像素代表一个待放大的色块... → 图例为边缘像素行/列

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg

# Design analysis for the D2 variant
# ----------------------------------
# spec_variant:
#   grid:
#     size: 23-27 square, one uniform background color
#   legend:
#     form: exactly one 1D edge legend, never a 2D sprite
#     orientation: either a top-edge row or a left-edge column
#     placement:
#       - horizontal samples place singleton pixels at (0, 1+2*i)
#       - vertical samples place singleton pixels at (1+2*i, 0)
#     length: 4-5 pixels on a step-2 lattice
#     colors: 2-4 distinct non-bg colors, with the first two legend pixels different
#   expanded_object:
#     horizontal:
#       - legend colors columns of an upper-right staircase
#       - filled lattice cells are (r, c) with 0 <= r <= c < n
#       - all cells in column c use legend[c]
#     vertical:
#       - legend colors rows of a lower-left staircase
#       - filled lattice cells are (r, c) with 0 <= c <= r < n
#       - all cells in row r use legend[r]
#   blocks:
#     block_size: 2-4
#     step: block_size + gap, with gap 1-3 and total step 3-5
#     input_count: exactly 2 kept blocks
#     kept_blocks:
#       - horizontal: lattice cells (0,0) and (0,1)
#       - vertical: lattice cells (0,0) and (1,0)
#
# a) Concrete behavior trace
#    Original task:
#      - a singleton at legend lattice cell (rr,cc) directly meant one block at (rr,cc).
#    New task:
#      - if the top edge legend is [red, blue, green, yellow], it no longer lists all 2D cells.
#      - instead it expands to columns:
#          col 0 -> [(0,0)] red
#          col 1 -> [(0,1),(1,1)] blue
#          col 2 -> [(0,2),(1,2),(2,2)] green
#          col 3 -> [(0,3),(1,3),(2,3),(3,3)] yellow
#      - the solver must first rebuild these hidden 2D lattice cells, then apply the same scale/offset
#        projection as in the original task.
#
# b) Complexity calibration
#    - The legend length is only 4-5, so the rule is visible from few examples.
#    - The expanded object contains 10 or 15 blocks: enough to be clearly 2D, but not visually crowded.
#    - Orientation varies between row and column so the "edge legend" change is observable across samples.
#
# c) Old assumptions that break
#    - Original code assumed singleton legend pixels already covered both row and column coordinates.
#      With a 1D line, normalizing legend singletons directly would collapse one axis and lose cells.
#    - Original voting was safe because every legend pixel represented one output block. Here one legend
#      pixel fans out into many blocks, so generator and solver both need an explicit expansion step.
#    - The original "same-row adjacent anchor pair" is only valid for the horizontal expansion. Vertical
#      samples need a same-column adjacent pair, so block-step inference must accept diffs from one axis.
#
# d) Placement legality
#    - The staircase fits in an n x n lattice window where n is the legend length.
#    - The generator chooses base_r/base_c so every enlarged block stays in-bounds.
#    - Legend pixels are on the outer edge; enlarged blocks start at least 3 cells away from both edges,
#      so there is no overlap and the edge marker remains visually isolated.
#
# e) Visual observability
#    - A human can see that the edge line is too small to specify all large blocks one-by-one.
#    - In horizontal samples, each legend pixel controls an entire staircase column; in vertical samples,
#      it controls an entire staircase row. This is visible without relying on color ids or hidden order.
#
# f) Visual cleanliness
#    - Only three layers exist: background, edge legend singletons, and square blocks.
#    - The block area is contiguous and staircase-shaped; no overlays or mixed object types occupy
#      the same local region.
#
# g) Color inference
#    - Colors are copied, not remapped.
#    - Repetition is structured: a repeated color always appears as a full row or full column of blocks
#      in the staircase, giving clear support for the expansion rule.


def _extract_objects(g: np.ndarray, bg: int):
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return singles, blocks


def _read_edge_legend(singles: list[tuple[int, int, int]], h: int, w: int):
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "grid must contain an edge legend in the top-left window"

    rows = sorted({r for r, _, _ in legend})
    cols = sorted({c for _, c, _ in legend})
    if len(rows) == 1 and len(cols) >= 2:
        orientation = "h"
        axis_vals = cols
        fixed = rows[0]
        assert fixed <= 1, "horizontal legend must lie on the top edge band"
        by_idx = {}
        diffs = [axis_vals[i + 1] - axis_vals[i] for i in range(len(axis_vals) - 1)]
        step_l = reduce(gcd, diffs)
        start = min(axis_vals)
        for _r, c, v in legend:
            idx = (c - start) // step_l
            assert idx not in by_idx, "legend indices must be unique"
            by_idx[idx] = int(v)
    elif len(cols) == 1 and len(rows) >= 2:
        orientation = "v"
        axis_vals = rows
        fixed = cols[0]
        assert fixed <= 1, "vertical legend must lie on the left edge band"
        by_idx = {}
        diffs = [axis_vals[i + 1] - axis_vals[i] for i in range(len(axis_vals) - 1)]
        step_l = reduce(gcd, diffs)
        start = min(axis_vals)
        for r, _c, v in legend:
            idx = (r - start) // step_l
            assert idx not in by_idx, "legend indices must be unique"
            by_idx[idx] = int(v)
    else:
        raise AssertionError("legend must be a single row or a single column")

    assert step_l > 0, "legend lattice step must be positive"
    keys = sorted(by_idx)
    assert keys == list(range(len(keys))), "legend indices must be contiguous from 0"
    seq = [by_idx[i] for i in keys]
    assert len(seq) >= 2, "legend must contain at least two pixels"
    return orientation, seq


def _expand_edge_legend(orientation: str, seq: list[int]):
    pts: list[tuple[int, int, int]] = []
    if orientation == "h":
        for cc, col in enumerate(seq):
            for rr in range(cc + 1):
                pts.append((int(rr), int(cc), int(col)))
    else:
        for rr, col in enumerate(seq):
            for cc in range(rr + 1):
                pts.append((int(rr), int(cc), int(col)))
    return pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a staircase of enlarged blocks from a 1D edge legend.

    Input structure:
    - bg is the dominant background color.
    - Near the top-left border, the legend is a single row or a single column of isolated pixels on a step-2 lattice.
    - Elsewhere, only two uniformly colored square blocks of the larger pattern are already present.

    Rule:
    - Read the edge legend as an ordered 1D color sequence.
    - Expand that sequence into a 2D staircase: a top-edge row colors staircase columns, while a left-edge column colors staircase rows.
    - Use the existing blocks to infer the enlarged lattice step and base offset, then add every missing block from the expanded staircase.

    Key insight:
    - The legend no longer enumerates the 2D layout directly; it provides one axis of a triangular pattern, and each legend pixel must be repeated across a whole staircase row or column before projection.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _extract_objects(g, bg)
    assert singles and blocks, "grid must contain legend pixels and enlarged blocks"

    orientation, seq = _read_edge_legend(singles, h, w)
    legend_pts = _expand_edge_legend(orientation, seq)

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))
    for col in by_color:
        by_color[col].sort()

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "existing blocks must reveal a grid step"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            base_r = br0 - rr * step
            base_c = bc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    assert bases, "existing blocks must determine the legend-to-grid offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    base_r, base_c = best_bases[0]

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: edge legend ===
LEGEND_STEP = 2
LEGEND_OFFSET = 1
LEGEND_LENGTHS = (4, 5)
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 3

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_MARGIN = 3

# === CFG: generation ===
MAX_RETRIES = 400

_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]


def _random_color_counts(total: int, k: int):
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _build_sequence(length: int, palette: list[int]):
    counts = dict(zip(palette, _random_color_counts(length, len(palette))))
    seq: list[int] = []

    def search():
        i = len(seq)
        if i == length:
            return all(v == 0 for v in counts.values())
        options = [col for col in palette if counts[col] > 0]
        if i == 1:
            options = [col for col in options if col != seq[0]]
        random.shuffle(options)
        for col in options:
            if i >= 2 and (seq[i - 1], col) == (seq[0], seq[1]):
                continue
            seq.append(col)
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            seq.pop()
        return False

    if not search():
        return None
    return seq


def _pattern_from_sequence(orientation: str, seq: list[int]):
    pts: dict[tuple[int, int], int] = {}
    if orientation == "h":
        for cc, col in enumerate(seq):
            for rr in range(cc + 1):
                pts[(rr, cc)] = int(col)
        keep = {(0, 0), (0, 1)}
    else:
        for rr, col in enumerate(seq):
            for cc in range(rr + 1):
                pts[(rr, cc)] = int(col)
        keep = {(0, 0), (1, 0)}
    return pts, keep


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    orientation = random.choice(["h", "v"])
    length = random.choice(LEGEND_LENGTHS)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (length + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, length))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    seq = _build_sequence(length, palette)
    if seq is None:
        return None

    pts, keep = _pattern_from_sequence(orientation, seq)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step
    c_hi = size - block_size - max_c * step
    if r_hi < BASE_MARGIN or c_hi < BASE_MARGIN:
        return None
    base_r = random.randint(BASE_MARGIN, r_hi)
    base_c = random.randint(BASE_MARGIN, c_hi)

    out = np.full((size, size), bg, dtype=int)
    if orientation == "h":
        for i, col in enumerate(seq):
            out[0, LEGEND_OFFSET + i * LEGEND_STEP] = col
    else:
        for i, col in enumerate(seq):
            out[LEGEND_OFFSET + i * LEGEND_STEP, 0] = col

    for (rr, cc), col in pts.items():
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        br = base_r + rr * step
        bc = base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 D2 input")
