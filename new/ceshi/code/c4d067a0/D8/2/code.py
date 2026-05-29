# Variant of c4d067a0
# Dimension: D8 — 多步处理顺序 (multi_step)
# Change: Step 1: 识别左上角图例精灵; Step 2: 定位网格中已有的同色色块; Step 3: 计算缩放因子和偏移; ... → 逆向过程：从放大结果推断图例

import random
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D8 inversion variant
# ============================================
# a) Concrete behavior and trace
#    - The large square blocks now carry the complete information. The top-left legend
#      lattice is empty or mostly empty in the input, while the main block matrix is
#      already complete.
#    - Output restores the miniature legend by shrinking each solid block to one sample
#      pixel placed on the fixed legend lattice at (1,1) with step 2.
#    - Example trace:
#      original: legend pixel at (rr,cc) -> infer base/scale from a few blocks -> draw
#      missing block at (base_r + rr*step, base_c + cc*step).
#      variant: full block already exists at (base_r + rr*step, base_c + cc*step) ->
#      infer its lattice coordinate (rr,cc) by subtracting the minimum block origin and
#      dividing by the common step -> draw the missing legend pixel at
#      (1 + rr*2, 1 + cc*2).
#      The changed stage is the information flow: the solver no longer uses legend
#      pixels to predict blocks; it uses blocks to reconstruct legend pixels.
# b) Complexity calibration
#    - Keep the original staircase family, block sizes, gaps, and 2-4 colors. This
#      preserves enough repeated structure for humans to detect the shrink relation.
#    - Input legend keeps at most 2 pixels, so the reverse behavior is visually obvious
#      and cannot be mistaken for the original "fill missing blocks" task.
# c) Hidden-assumption audit against the original solver
#    - The old `assert singles and blocks` is invalid because the legend may be empty.
#    - The old legend-step inference from singleton pixels cannot run when the legend is
#      erased.
#    - The old color-voting base inference is unnecessary here because the block matrix
#      is complete; the minimum block top-left is the lattice origin.
#    - The old unconditional block repaint would leave the input unchanged and fail to
#      create the intended output difference. The new solver must modify the legend area.
# d) Placement legality
#    - The generator still separates the legend area from the block area with
#      `base_c >= 8`, so shrinking back into the top-left cannot overlap any block.
#    - All blocks are same-sized solid squares, all aligned on one lattice, and every
#      block fits inside the canvas exactly as in the source task.
# e) Visual observability
#    - In each input, the large matrix is complete while the small legend is absent or
#      nearly absent. In each output, only the tiny sprite changes. Across 2-4 examples,
#      the natural human hypothesis is "the top-left pixels are the miniature version of
#      the block matrix".
# f) Visual cleanliness
#    - Only two regions are active: the sparse top-left legend window and the separated
#      block area. No overlays or competing marks are introduced.
# g) Color inferability
#    - Colors are copied directly from blocks to legend without any hidden remapping or
#      color-id ordering. The same color appears in both scales, making the rule visible.
# h) New solver assumptions and generator guarantees
#    - Assumption: all non-singleton components used by the task are solid NxN squares.
#      Guarantee: generator creates only those blocks plus optional singleton legend hints.
#    - Assumption: at least one positive lattice difference exists to recover the step.
#      Guarantee: staircase shapes always span multiple rows and multiple columns.
#    - Assumption: normalized block coords start at (0,0).
#      Guarantee: every staircase includes cell (0,0), so min top/min left are the base.


def _positive_gcd(values: list[int]) -> int:
    vals = [int(v) for v in values if int(v) > 0]
    assert vals, "need at least one positive difference"
    return reduce(gcd, vals)


def _extract_blocks(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    comps = components_mask(grid != bg, connectivity=4)
    blocks: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if len(pts) <= 1:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a small top-left legend by shrinking a completed matrix of colored blocks.

    Input structure:
    - bg = most common color.
    - Top-left corner: the legend lattice is empty or only partially populated with a few singleton hints.
    - Main area: all uniformly colored square blocks (NxN) of the magnified pattern are already present.

    Rule:
    - Treat each solid square block as one enlarged legend pixel.
    - Recover the block lattice step and base offset from the aligned block positions.
    - Convert every block back to a legend coordinate and place one pixel of the same color on the fixed top-left legend lattice.

    Key insight: The main block matrix is the full scaled-up realization of the sprite, so solving the task means sampling one color per block and projecting that pattern back into the miniature legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    blocks = _extract_blocks(g, bg)
    assert blocks, "grid must contain enlarged blocks"

    sizes = [s for _, _, s, _ in blocks]
    size_vals, size_cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(size_vals[size_cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = _positive_gcd(step_candidates)

    base_r = min(tops)
    base_c = min(lefts)

    legend_cells: dict[tuple[int, int], int] = {}
    for r0, c0, _size, col in blocks:
        dr = int(r0) - base_r
        dc = int(c0) - base_c
        assert dr % step == 0 and dc % step == 0, "blocks must align to one lattice"
        rr = dr // step
        cc = dc // step
        legend_cells[(int(rr), int(cc))] = int(col)

    assert legend_cells, "block lattice must define at least one legend pixel"
    max_rr = max(rr for rr, _ in legend_cells)
    max_cc = max(cc for _, cc in legend_cells)

    legend_r0, legend_c0 = LEGEND_ORIGIN
    for rr in range(max_rr + 1):
        for cc in range(max_cc + 1):
            tr = legend_r0 + rr * LEGEND_STEP
            tc = legend_c0 + cc * LEGEND_STEP
            assert 0 <= tr < h and 0 <= tc < w, "legend lattice must fit inside the grid"
            out[tr, tc] = bg

    for (rr, cc), col in legend_cells.items():
        tr = legend_r0 + rr * LEGEND_STEP
        tc = legend_c0 + cc * LEGEND_STEP
        out[tr, tc] = int(col)

    return out


# --------------- generator (D8 inversion) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6
LEGEND_KEEP_MAX = 2

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 300


def _enumerate_staircase_shapes() -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix: list[int], remaining: int, r: int) -> None:
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase() -> list[tuple[int, int]]:
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, width in enumerate(widths) for c in range(width)]


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, count in enumerate(counts) if count < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(
    pts: list[tuple[int, int]], palette: list[int]
) -> dict[tuple[int, int], int]:
    counts = _random_color_counts(len(pts), len(palette))
    pool: list[int] = []
    for col, count in zip(palette, counts):
        pool.extend([int(col)] * int(count))
    random.shuffle(pool)
    return {pos: pool[i] for i, pos in enumerate(pts)}


def _legend_hint_keep_count(num_pts: int) -> int:
    return random.randint(0, min(LEGEND_KEEP_MAX, num_pts))


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts = _pick_staircase()

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, palette)

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    keep_count = _legend_hint_keep_count(len(pts))
    keep = set(random.sample(pts, keep_count))
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = bg

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
    raise RuntimeError("failed to generate valid c4d067a0 D8 inversion input")
