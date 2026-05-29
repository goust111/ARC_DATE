# Combo variant of c4d067a0 (index 15)
# combo_spec: D11/2+D12/1
# Name: 背景色与图例颜色相同+高颜色多样性

import random
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, get_bg

# Design analysis
# ----------------
# spec_variant:
#   concreteization_and_trace:
#     - Keep the original "legend pixels -> scaled square blocks" task, but make the true legend start with one hidden pixel.
#     - The hidden pixel is always the lattice origin at absolute screen position (1,1); its color equals the background,
#       so it is invisible both in the legend and in the corresponding block area.
#     - To make that hidden pixel matter visibly, the true first legend row contains only that single hidden pixel.
#       Therefore the first visible legend row is actually the second logical row.
#     - Example: true widths [1,2,3,3]. The visible legend shows rows of widths [2,3,3] beginning one lattice step lower.
#       An old solver that normalizes by the first visible row would treat those rows as logical rows [0,1,2] and place every
#       visible block one block-step too high. The fused solver instead measures legend rows from the fixed hidden origin.
#   complexity_calibration:
#     - High color diversity is implemented structurally rather than by noisy repetition: every visible legend pixel gets its
#       own non-background color, while the hidden origin uses the background color.
#     - The legend uses 7-9 total logical pixels, so 6-8 visible colors repeat the "many distinct colors" effect without
#       exhausting placement room or making the board unreadable.
#     - Only one hidden pixel is introduced, and it always occupies the same semantic role (the hidden singleton top row), so
#       the task stays learnable from a few examples instead of becoming a combinatorial missing-cell puzzle.
#   implicit_assumption_audit:
#     - Original solver assumption: every legend pixel is a non-bg singleton component. Broken here because the origin pixel is bg.
#       New solver assumption: the legend origin is fixed at (1,1), and the hidden pixel is always located there. The generator
#       enforces that directly.
#     - Original solver assumption: repeated colors may need voting to recover the offset. Replaced here by a high-diversity
#       assumption: every visible legend color is unique, so each kept anchor block names exactly one legend position.
#     - Original solver assumption: using min visible legend row/col is safe for legend coordinates. Broken here because the
#       hidden origin row is missing. The new solver computes legend coordinates from the fixed lattice origin instead.
#     - Component scan still assumes only legend singletons exist in the top-left window and only uniform square blocks exist in
#       the main area. The generator keeps the block area separated from the legend columns and uses only solid NxN anchors.
#   placement_legality:
#     - The legend stays in the top-left on a step-2 lattice.
#     - The block area uses the original size/gap bounds, with base_c >= 8 so blocks never overlap the legend window.
#     - Shapes are chosen so all projected blocks fit inside the canvas; failed attempts are discarded.
#   observability_and_tidiness:
#     - The missing origin is visible as an empty lattice slot at the legend start, not as a color-id trick.
#     - Because every visible legend pixel has a different color, the two retained anchor blocks clearly identify their legend
#       positions, and the only nontrivial inference is the hidden top row.
#     - The generator rejects samples where the old "first visible row is row 0" baseline happens to produce the same output,
#       so the fusion effect remains active rather than silently collapsing back to the base task.

LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_WINDOW_R = 10
LEGEND_WINDOW_C = 8

GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

LEGEND_SHAPES = [
    (1, 3, 3),     # 7 logical cells, 6 visible
    (1, 2, 2, 3),  # 8 logical cells, 7 visible
    (1, 2, 3, 3),  # 9 logical cells, 8 visible
]

BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1
MAX_RETRIES = 400

_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE
    if STEP_MIN <= n + g <= STEP_MAX
]


def _gcd_all(vals: list[int]) -> int:
    vals = [abs(int(v)) for v in vals if int(v) != 0]
    return reduce(gcd, vals) if vals else 0


def _extract_singles_and_blocks(grid: np.ndarray, bg: int):
    comps = components_mask(grid != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(grid[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return singles, blocks


def _extract_visible_legend(grid: np.ndarray):
    bg = get_bg(grid)
    singles, blocks = _extract_singles_and_blocks(grid, bg)
    legend = [
        (r, c, col)
        for (r, c, col) in singles
        if r < LEGEND_WINDOW_R and c < LEGEND_WINDOW_C
    ]
    assert legend, "top-left legend must contain visible singleton pixels"

    rows = sorted({r for r, _, _ in legend})
    cols = sorted({c for _, c, _ in legend})
    step = _gcd_all(
        [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        + [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    )
    assert step == LEGEND_STEP, "legend lattice step must remain 2"

    lr0, lc0 = LEGEND_ORIGIN
    legend_pts = []
    for r, c, col in legend:
        assert (r - lr0) % step == 0 and (c - lc0) % step == 0, "legend must align to fixed origin"
        rr = (r - lr0) // step
        cc = (c - lc0) // step
        legend_pts.append((int(rr), int(cc), int(col)))

    visible_colors = [col for _, _, col in legend_pts]
    assert len(visible_colors) == len(set(visible_colors)), "visible legend colors must be unique"
    assert (lr0, lc0) not in {(r, c) for r, c, _ in legend}, "hidden origin must stay invisible"
    return bg, legend_pts, blocks


def _infer_block_layout(blocks: list[tuple[int, int, int, int]]):
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) >= 2, "at least two visible anchor blocks are required"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step = _gcd_all(
        [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
        + [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    )
    assert step > 0, "block lattice step must be recoverable from anchor blocks"
    return block_size, blocks, step


def _solve_with_origin(grid: np.ndarray, use_visible_origin: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg, legend_pts, blocks = _extract_visible_legend(g)
    block_size, blocks, step = _infer_block_layout(blocks)

    if use_visible_origin:
        min_rr = min(rr for rr, _, _ in legend_pts)
        min_cc = min(cc for _, cc, _ in legend_pts)
        norm_legend = [(rr - min_rr, cc - min_cc, col) for rr, cc, col in legend_pts]
    else:
        norm_legend = list(legend_pts)

    by_color = {col: (rr, cc) for rr, cc, col in norm_legend}
    base_votes = set()
    for br0, bc0, _size, col in blocks:
        assert col in by_color, "anchor block color must appear in legend"
        rr, cc = by_color[col]
        base_votes.add((int(br0 - rr * step), int(bc0 - cc * step)))
    assert len(base_votes) == 1, "anchors must imply a unique legend-to-grid offset"
    base_r, base_c = next(iter(base_votes))

    hidden_and_visible = [(0, 0, bg)] + norm_legend
    seen = set()
    for rr, cc, col in hidden_and_visible:
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, "projected block must fit"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a scaled block layout from a top-left legend whose first logical pixel is hidden by the background.

    Input structure:
    - The canvas background is also used as one legend color, so the true legend origin pixel at (1,1) is invisible.
    - All visible legend pixels are isolated singletons on the usual step-2 lattice, and each visible legend color is unique.
    - The main area contains exactly two visible square anchor blocks taken from that legend; the hidden background-colored block is absent visually.

    Rule:
    - Treat the legend as starting at the fixed lattice origin (1,1), even though that first pixel blends into the background.
    - Use the two visible blocks to recover the block scale and translation, then project every legend pixel to its corresponding NxN block position.
    - Paint the missing visible blocks and keep the hidden background-colored block implicit, since painting it leaves the background unchanged.

    Key insight:
    - The first visible legend row is not logical row 0: it is row 1 because row 0 contains only the invisible background-colored origin pixel.
    - High color diversity makes each anchor color identify one legend position directly, so the main challenge is correcting the origin, not resolving repeated colors.
    """
    return _solve_with_origin(grid, use_visible_origin=False)


def _shape_points(widths: tuple[int, ...]):
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    assert pts[0] == (0, 0) and widths[0] == 1
    visible = [p for p in pts if p != (0, 0)]
    adj_pairs = [p for p in visible if (p[0], p[1] + 1) in set(visible)]
    return pts, visible, adj_pairs


def _assign_colors(visible_pts: list[tuple[int, int]], bg: int):
    colors = {(0, 0): bg}
    palette = [c for c in range(NUM_COLORS) if c != bg]
    chosen = random.sample(palette, len(visible_pts))
    random.shuffle(chosen)
    for pt, col in zip(visible_pts, chosen):
        colors[pt] = col
    return colors


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap

    widths = random.choice(LEGEND_SHAPES)
    pts, visible_pts, adj_pairs = _shape_points(widths)
    anchor_pair = random.choice(adj_pairs)

    bg = random.randrange(NUM_COLORS)
    legend_colors = _assign_colors(visible_pts, bg)

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
        if (rr, cc) != (0, 0):
            out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br, bc = base_r + rr * step, base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in visible_pts:
        if (rr, cc) in keep:
            continue
        br, bc = base_r + rr * step, base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

    try:
        solved = main(inp)
        baseline = _solve_with_origin(inp, use_visible_origin=True)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(baseline, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
