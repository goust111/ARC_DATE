# Combo variant of c4d067a0 (index 13)
# combo_spec: D2/1+D9/0
# Name: 密集图例像素+密集排列结构

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original c4d067a0 used isolated legend pixels on a step-2 lattice, so the
#    solver could recover legend coordinates from singleton components plus a
#    gcd over row/col spacings. Here the legend is a dense top-left rectangle:
#    e.g. a 3x3 legend like
#        2 4 1
#        3 2 4
#        1 3 2
#    maps to spaced NxN blocks in the main area. The visible anchor blocks might
#    only keep the cells (1,0)->color 3 and (1,1)->color 2, so the solver must
#    read the compact legend directly, infer the horizontal block step from the
#    two kept blocks, locate the unique matching adjacent color pair in legend
#    coordinates, then paint every legend cell as a block.
# b) Complexity calibration:
#    D2/1 and D9/0 both densify the legend, so to make the change visually real
#    rather than cosmetic, the generator uses fully filled dense rectangles
#    instead of the original sparse staircase. This removes the step-2 checker
#    look and makes the top-left sprite a single connected object.
# c) Broken implicit assumptions from the original solver:
#    - legend cells are no longer singleton components
#    - legend spacing is no longer recoverable via gcd(row_diffs, col_diffs)=2
#    - the legend is now one mixed-color component, so component classification
#      must separate "dense multicolor rectangle" from "uniform square block"
# d) Placement legality:
#    The generator chooses base offsets so all projected blocks fit the canvas
#    and stay well to the right of the dense legend rectangle.
# e) Visual observability:
#    In samples, the legend appears as a compact multicolor rectangle while the
#    main area still contains large spaced blocks. The dense-vs-spaced contrast
#    is directly visible from 2-4 examples.
# f) Visual cleanliness:
#    Only one dense object exists in the top-left and only two anchor blocks are
#    visible in the main field; regions do not overlap, so the scene remains
#    easy to parse.
# g) New solver assumptions and generator support:
#    The solver assumes the legend contains no background holes and that the two
#    visible blocks correspond to one unique horizontal adjacent color pair in
#    the legend. The generator enforces both properties by construction.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Finish the large block pattern by reading a compact legend in the top-left corner.

    Input structure:
    - Background = the most frequent color.
    - Top-left: one dense multicolor rectangle whose cells touch edge-to-edge; each
      cell is a legend entry, not an isolated dot on a sparse lattice.
    - Main area: two visible monochrome square blocks of equal size, separated by a
      fixed horizontal step; the rest of the block pattern is missing.

    Rule:
    - Treat each cell of the dense legend rectangle as one block position in a coarse
      layout.
    - Use the two visible blocks to infer the block size, the spacing between block
      origins, and which adjacent legend cells they correspond to.
    - Project every legend cell into the main area with that same scale and offset,
      filling the missing square blocks with the legend cell colors.

    Key insight:
    The legend no longer advertises its coordinates through empty spacer cells, so the
    solver must read the compact rectangle directly and anchor it using the unique
    left-to-right color pair shown by the two kept blocks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain non-background objects"

    comp_boxes: list[tuple[int, int, int, int, np.ndarray]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        comp_boxes.append((int(r0), int(c0), int(r1), int(c1), pts))

    legend_r0, legend_c0, legend_r1, legend_c1, _legend_pts = min(
        comp_boxes, key=lambda item: (item[0], item[1])
    )
    legend = g[legend_r0 : legend_r1 + 1, legend_c0 : legend_c1 + 1]
    legend_h, legend_w = legend.shape
    assert legend_h >= 2 and legend_w >= 2, "legend must be a compact rectangle"
    assert np.all(legend != bg), "dense legend may not contain background holes"

    blocks: list[tuple[int, int, int, int]] = []
    for r0, c0, r1, c1, _pts in comp_boxes:
        if (r0, c0, r1, c1) == (legend_r0, legend_c0, legend_r1, legend_c1):
            continue
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col):
            blocks.append((int(r0), int(c0), int(hh), col))

    assert len(blocks) >= 2, "input must contain the visible anchor blocks"

    sizes = [size for _, _, size, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) >= 2, "need two blocks of the same size"

    pair_candidates: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a = blocks[i]
            b = blocks[j]
            if a[0] != b[0]:
                continue
            left, right = (a, b) if a[1] < b[1] else (b, a)
            step = right[1] - left[1]
            if step > block_size:
                pair_candidates.append((left, right))
    assert len(pair_candidates) == 1, "anchor blocks must form one horizontal pair"

    left_block, right_block = pair_candidates[0]
    step = right_block[1] - left_block[1]
    anchor_colors = (left_block[3], right_block[3])

    matches: list[tuple[int, int]] = []
    for rr in range(legend_h):
        for cc in range(legend_w - 1):
            if int(legend[rr, cc]) == anchor_colors[0] and int(legend[rr, cc + 1]) == anchor_colors[1]:
                matches.append((rr, cc))
    assert len(matches) == 1, "legend anchor pair must be unique"

    anchor_rr, anchor_cc = matches[0]
    base_r = left_block[0] - anchor_rr * step
    base_c = left_block[1] - anchor_cc * step

    for rr in range(legend_h):
        for cc in range(legend_w):
            col = int(legend[rr, cc])
            tr0 = base_r + rr * step
            tc0 = base_c + cc * step
            assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
                "projected block must fit in canvas"
            )
            out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: dense legend ===
LEGEND_ORIGIN = (1, 1)
LEGEND_DIMS = ((3, 2), (3, 3), (4, 2))
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_R_MIN = 1
BASE_C_MIN = 8

# === CFG: generation ===
MAX_RETRIES = 400
_BLOCK_GAP_PAIRS = [
    (block_size, gap)
    for block_size in BLOCK_SIZE_RANGE
    for gap in GAP_RANGE
    if STEP_MIN <= block_size + gap <= STEP_MAX
]


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, value in enumerate(counts) if value < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_dense_legend(rows: int, cols: int, anchor_pair: tuple[int, int], palette: list[int]):
    total = rows * cols
    anchor_left = anchor_pair
    anchor_right = (anchor_pair[0], anchor_pair[1] + 1)
    cells = [(r, c) for r in range(rows) for c in range(cols)]

    for _ in range(300):
        counts = dict(zip(palette, _random_color_counts(total, len(palette))))
        left_color, right_color = random.sample(palette, 2)
        counts[left_color] -= 1
        counts[right_color] -= 1
        if counts[left_color] < 0 or counts[right_color] < 0:
            continue

        colors = {anchor_left: left_color, anchor_right: right_color}
        bag: list[int] = []
        for color, count in counts.items():
            bag.extend([color] * count)
        random.shuffle(bag)

        rest = [cell for cell in cells if cell not in colors]
        for cell, color in zip(rest, bag):
            colors[cell] = color

        occurrences = 0
        for r in range(rows):
            for c in range(cols - 1):
                if colors[(r, c)] == left_color and colors[(r, c + 1)] == right_color:
                    occurrences += 1
        if occurrences != 1:
            continue
        return colors
    return None


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    legend_rows, legend_cols = random.choice(LEGEND_DIMS)
    anchor_pair = (
        random.randrange(legend_rows),
        random.randrange(legend_cols - 1),
    )

    bg = random.randrange(NUM_COLORS)
    total = legend_rows * legend_cols
    k_min = max(PALETTE_SIZE_MIN, (total + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, total))
    palette = random.sample([color for color in range(NUM_COLORS) if color != bg], k)
    legend_colors = _assign_dense_legend(legend_rows, legend_cols, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_r = legend_rows - 1
    max_c = legend_cols - 1
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for rr in range(legend_rows):
        for cc in range(legend_cols):
            color = legend_colors[(rr, cc)]
            out[lr0 + rr, lc0 + cc] = color
            br = base_r + rr * step
            bc = base_c + cc * step
            out[br : br + block_size, bc : bc + block_size] = color

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr in range(legend_rows):
        for cc in range(legend_cols):
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
    raise RuntimeError("failed to generate valid c4d067a0 dense-legend input")
