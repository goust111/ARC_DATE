# Combo variant of c4d067a0 (index 24)
# combo_spec: D6/1+D8/2
# Name: 仅输出新增区域+逆向推断

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


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The magnified block matrix is now the authoritative source, as in the inversion
#      variant: input already contains every solid NxN block of the staircase pattern.
#    - The top-left legend is only partially visible. We keep one entire extreme row of
#      legend dots (top row or bottom row) as hints and erase the other legend pixels.
#    - Solver flow:
#      original: legend dots -> infer scale/offset from a couple of blocks -> add missing
#      enlarged blocks into the large matrix on the original canvas.
#      combo: complete enlarged block matrix -> infer lattice/base directly from blocks ->
#      shrink every block back to one legend cell -> subtract legend cells that are
#      already visible in the input -> return only the missing legend cells as a compact
#      patch in legend-coordinate space.
#    - Example trace:
#      full legend coords = {(0,0):red, (0,1):blue, (1,0):yellow, (1,1):red, (2,0):blue}.
#      input keeps row 0 hints {(0,0), (0,1)} and keeps all large blocks.
#      solver recovers the full set from blocks, removes the kept hints, and outputs the
#      compact patch spanning rows 1..2 and cols 0..1 with the remaining colors only.
# b) Complexity calibration
#    - Keep the original staircase family, 2-4 colors, uniform NxN blocks, and common
#      lattice step. This preserves repeated structure humans can see across examples.
#    - Keep 1-3 hint pixels by preserving exactly one extreme legend row. This makes the
#      subtraction step visible while avoiding ambiguity about what counts as "already
#      present" versus "newly inferred".
#    - Output is compacted to legend-cell resolution instead of preserving the input's
#      step-2 spacing. That makes the D6 change visually obvious rather than a nearly
#      unchanged crop with empty spacer columns.
# c) Hidden-assumption audit against the original solver
#    - `assert singles and blocks` no longer holds: the task can be solved from blocks
#      alone, so legend hints are optional for inference.
#    - Inferring legend lattice step from singleton pixels is invalid because most legend
#      dots are absent. The combo solver therefore relies on the fixed legend lattice
#      origin/step from task design and reconstructs legend coords from the block matrix.
#    - The old color-voting offset logic is unnecessary because the block matrix is
#      complete; the minimum block top-left is the lattice origin.
#    - Returning the full input-sized canvas would violate D6/1. The solver must emit a
#      fresh array whose size is the bbox of the missing legend cells only.
# d) Placement legality
#    - Generator keeps `base_c >= 8`, so the enlarged matrix never overlaps the legend
#      area and shrinking back into the top-left remains legal.
#    - All kept hint pixels lie on the fixed legend lattice and are a subset of the true
#      legend, so subtracting them never creates contradictions.
#    - Every generated instance leaves at least one missing legend cell, so the compact
#      output is always non-empty.
# e) Visual observability
#    - Inputs show a complete block matrix plus a partial miniature row in the top-left.
#      Outputs show a much smaller patch containing only the unseen miniature cells.
#    - Across examples, the same colors and staircase geometry appear at two scales, so a
#      human can infer "shrink blocks to legend" and then notice that already-shown
#      legend cells are omitted from the answer.
# f) Visual cleanliness
#    - Only two separated regions are active: the top-left hint row and the main block
#      area. The output contains a single compact patch with no overlays or clutter.
# g) Color inferability
#    - There is no hidden recoloring. Each output pixel copies the color of a matching
#      solid block, and repeated colors occur multiple times within the staircase.
# h) New solver assumptions and generator guarantees
#    - Assumption: all task-relevant non-singleton components are solid equal-sized
#      squares on one lattice. Guarantee: generator only creates such blocks.
#    - Assumption: the hint pixels, if present, live on the fixed legend lattice at
#      `LEGEND_ORIGIN + LEGEND_STEP * (rr, cc)`. Guarantee: generator uses that layout.
#    - Assumption: block positions span multiple rows and columns so the lattice step can
#      be recovered from positive coordinate differences. Guarantee: every staircase has
#      at least 3 rows and max width at least 2.


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
LEGEND_WINDOW = 12

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 300


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


def _extract_legend_hints(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    comps = components_mask(grid != bg, connectivity=4)
    lr0, lc0 = LEGEND_ORIGIN
    hints: dict[tuple[int, int], int] = {}
    for pts in comps:
        if len(pts) != 1:
            continue
        r, c = map(int, pts[0])
        if r >= LEGEND_WINDOW or c >= LEGEND_WINDOW:
            continue
        if r < lr0 or c < lc0:
            continue
        if (r - lr0) % LEGEND_STEP != 0 or (c - lc0) % LEGEND_STEP != 0:
            continue
        rr = (r - lr0) // LEGEND_STEP
        cc = (c - lc0) // LEGEND_STEP
        hints[(int(rr), int(cc))] = int(grid[r, c])
    return hints


def _infer_full_legend(blocks: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], int]:
    sizes = [size for _, _, size, _ in blocks]
    size_vals, size_cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(size_vals[size_cnts.argmax()])
    filtered = [b for b in blocks if b[2] == block_size]
    assert filtered, "must contain blocks of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in filtered})
    lefts = sorted({c0 for _, c0, _, _ in filtered})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = _positive_gcd(step_candidates)

    base_r = min(tops)
    base_c = min(lefts)
    legend_cells: dict[tuple[int, int], int] = {}
    for r0, c0, _size, col in filtered:
        dr = int(r0) - base_r
        dc = int(c0) - base_c
        assert dr % step == 0 and dc % step == 0, "blocks must align to one lattice"
        rr = dr // step
        cc = dc // step
        legend_cells[(int(rr), int(cc))] = int(col)
    assert legend_cells, "blocks must define at least one legend cell"
    return legend_cells


def _render_missing_patch(
    full_legend: dict[tuple[int, int], int], hints: dict[tuple[int, int], int], bg: int
) -> np.ndarray:
    missing = {
        pos: col for pos, col in full_legend.items() if hints.get(pos) != col
    }
    assert missing, "input must omit at least one legend cell"

    min_r = min(rr for rr, _ in missing)
    max_r = max(rr for rr, _ in missing)
    min_c = min(cc for _, cc in missing)
    max_c = max(cc for _, cc in missing)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for (rr, cc), col in missing.items():
        out[rr - min_r, cc - min_c] = int(col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing miniature legend cells from a completed matrix of large blocks.

    Input structure:
    - bg = most common color.
    - Top-left corner: a sparse legend lattice at fixed origin/step, but only one row of
      legend pixels is still visible; the rest of the legend is erased.
    - Main area: the full magnified staircase of uniformly colored square blocks (NxN) is
      already present and completely specifies the miniature pattern.

    Rule:
    - Treat each solid block as one enlarged legend cell and recover its legend
      coordinate from the shared block lattice.
    - Reconstruct the full miniature legend from the large block matrix.
    - Remove the legend cells that are already present in the input, then output only the
      remaining inferred legend cells as a compact subgrid in legend-coordinate space.

    Key insight: The puzzle is solved in reverse: shrink the complete block matrix back
    into the tiny sprite, then isolate just the legend cells that were absent from the
    input instead of redrawing the whole scene.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    blocks = _extract_blocks(g, bg)
    assert blocks, "grid must contain enlarged blocks"
    full_legend = _infer_full_legend(blocks)

    hints = _extract_legend_hints(g, bg)
    for pos, col in hints.items():
        assert full_legend.get(pos) == col, "legend hints must agree with the blocks"

    return _render_missing_patch(full_legend, hints, bg)


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


def _choose_kept_hints(pts: list[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = sorted({r for r, _ in pts})
    candidates = []
    for target_row in (rows[0], rows[-1]):
        row_cells = {pos for pos in pts if pos[0] == target_row}
        if 0 < len(row_cells) < len(pts):
            candidates.append(row_cells)
    assert candidates, "must be able to keep one non-empty extreme legend row"
    return set(random.choice(candidates))


def _expected_output(
    legend_colors: dict[tuple[int, int], int], kept_hints: set[tuple[int, int]], bg: int
) -> np.ndarray:
    hints = {pos: legend_colors[pos] for pos in kept_hints}
    return _render_missing_patch(legend_colors, hints, bg)


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
    kept_hints = _choose_kept_hints(pts)

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    inp = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        if (rr, cc) in kept_hints:
            inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * step
        bc = base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = col

    expected = _expected_output(legend_colors, kept_hints, bg)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
