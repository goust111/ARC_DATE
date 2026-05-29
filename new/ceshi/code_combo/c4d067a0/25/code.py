# Combo variant of c4d067a0 (index 25)
# combo_spec: D7/2+D10/2
# Name: 颜色数量编码缩放+无锚点参考

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    - Original: the top-left legend is a sparse minimap, and two pre-placed
#      enlarged blocks elsewhere reveal both scale and offset.
#    - New combo: the input contains only the legend. If the legend uses
#      k distinct colors, every legend pixel expands to a k x k square. Those
#      squares are laid out with a fixed 1-cell gap and the whole enlarged
#      pattern is pushed against the bottom-right corner with a 1-cell outer
#      margin. Example: a legend with 3 colors yields 3 x 3 blocks, step 4,
#      and the rightmost/bottommost block edges sit one cell away from the
#      canvas border.
#    - Difference: scale is now decoded from color diversity, while offset is
#      recovered from canvas geometry rather than anchored block evidence.
#
# b) Complexity calibration:
#    - k is restricted to 2..4, matching the original palette range and keeping
#      the counting rule visually learnable from a few examples.
#    - The block gap is fixed at 1 cell so the only changing geometric degree
#      of freedom is the scale, not an extra unknown spacing parameter.
#    - The legend shape remains a connected staircase, which preserves the
#      original visual identity and avoids ambiguous isolated dots.
#
# c) Hidden-assumption audit against the original solver:
#    - Any logic that depended on non-singleton components, dominant block size,
#      block-position gcds, or color-voting for base offsets is invalid now.
#    - The new solver instead assumes:
#      1. all informative colored cells in the input are singleton legend dots;
#      2. the legend occupies the top-left small window on a regular lattice;
#      3. the number of distinct non-bg legend colors equals the enlargement
#         factor;
#      4. the enlarged copy is bottom-right aligned with a fixed outer margin.
#    - The generator explicitly constructs inputs to satisfy those assumptions.
#
# d) Placement legality:
#    - With grid sizes 23..27, staircase width <= 3, staircase height <= 4,
#      and k <= 4, the bottom-right aligned projection always fits.
#    - The smallest possible right-aligned base column is 8, which keeps the
#      enlarged region separated from the legend columns.
#
# e) Visual observability:
#    - Across examples, the only visible cue for block size is the count of
#      distinct legend colors. This is human-visible and repeated.
#    - The offset rule is also human-visible because the enlarged structure
#      consistently hugs the lower-right corner.
#
# f) Visual cleanliness:
#    - There are only two regions: sparse legend at top-left and enlarged block
#      pattern at bottom-right. They do not overlap, and no extra markers or
#      overlays are introduced.
#
# g) Inferability of the color rule:
#    - No hidden ordering by numeric color ids is used. Only the visible number
#      of distinct colors matters. The colors themselves are copied unchanged
#      from legend dots to enlarged squares.


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

# === CFG: enlarged projection ===
OUTER_MARGIN = 1
BLOCK_GAP = 1

# === CFG: generation ===
MAX_RETRIES = 300


def _enumerate_staircase_shapes():
    """Enumerate all valid staircase width tuples."""
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
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


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, w in enumerate(widths) for c in range(w)]


def _random_color_counts(total, k):
    """Distribute total across k positive bins with an upper bound."""
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, palette):
    counts = _random_color_counts(len(pts), len(palette))
    bag = []
    for col, cnt in zip(palette, counts):
        bag.extend([col] * cnt)
    random.shuffle(bag)
    order = pts[:]
    random.shuffle(order)
    return {pos: col for pos, col in zip(order, bag)}


def _extract_legend(g):
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert all(len(pts) == 1 for pts in comps), "this combo variant has no pre-placed enlarged blocks"
    singles = []
    for pts in comps:
        if len(pts) != 1:
            continue
        r, c = pts[0]
        singles.append((int(r), int(c), int(g[r, c])))

    assert singles, "input must contain singleton legend pixels"

    win_r = min(g.shape[0], 12)
    win_c = min(g.shape[1], 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "legend must appear in the top-left window"
    assert len(legend) == len(singles), "all foreground pixels should belong to the top-left legend"
    return bg, legend


def _legend_to_lattice(legend):
    rows = sorted({r for r, _, _ in legend})
    cols = sorted({c for _, c, _ in legend})
    assert len(rows) >= 2 and len(cols) >= 2, "legend must span multiple rows and cols"

    step_candidates = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    step_candidates += [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    step = reduce(gcd, step_candidates)
    assert step > 0, "legend lattice step must be positive"

    r0 = min(r for r, _, _ in legend)
    c0 = min(c for _, c, _ in legend)
    pts = []
    for r, c, col in legend:
        dr = r - r0
        dc = c - c0
        assert dr % step == 0 and dc % step == 0, "legend dots must sit on one lattice"
        pts.append((dr // step, dc // step, col))
    return pts


def _project_params(legend_pts, shape):
    h, w = shape
    scale = len({col for _, _, col in legend_pts})
    assert PALETTE_SIZE_MIN <= scale <= PALETTE_SIZE_MAX, "legend color count must encode scale"

    block_size = scale
    step = block_size + BLOCK_GAP
    max_rr = max(rr for rr, _, _ in legend_pts)
    max_cc = max(cc for _, cc, _ in legend_pts)
    proj_h = max_rr * step + block_size
    proj_w = max_cc * step + block_size

    base_r = h - OUTER_MARGIN - proj_h
    base_c = w - OUTER_MARGIN - proj_w
    assert base_r >= 0 and base_c >= 0, "projection must fit inside the canvas"
    return base_r, base_c, block_size, step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sparse legend by creating its enlarged copy without any anchor blocks.

    Input structure:
    - The background is the most frequent color.
    - The only meaningful foreground object is a top-left legend made of isolated
      single pixels on a regular lattice.
    - There are no pre-placed enlarged blocks elsewhere in the grid.

    Rule:
    - Count how many different non-background colors appear in the legend; that
      number is the enlargement factor and the side length of each solid square.
    - Keep a fixed 1-cell gap between neighboring enlarged squares.
    - Place the whole enlarged pattern so its bounding box sits 1 cell away from
      the bottom and right borders of the canvas, then copy every legend color to
      its corresponding enlarged square.

    Key insight:
    - The task no longer provides geometric anchors in the main field. Scale comes
      from color diversity in the legend, while translation comes from the canvas
      boundaries because the enlarged copy is always bottom-right aligned.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    _bg, legend = _extract_legend(g)
    legend_pts = _legend_to_lattice(legend)
    base_r, base_c, block_size, step = _project_params(legend_pts, g.shape)

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= g.shape[0] - block_size
        assert 0 <= tc0 <= g.shape[1] - block_size
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    pts = _pick_staircase()

    k_hi = min(PALETTE_SIZE_MAX, len(pts))
    k = random.randint(PALETTE_SIZE_MIN, k_hi)

    bg = random.randrange(NUM_COLORS)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, palette)

    inp = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    out = main(inp)

    # Ensure the fused behavior is really present:
    # 1. no enlarged blocks exist in the input;
    # 2. scale equals the number of distinct legend colors.
    comps = components_mask(inp != bg, connectivity=4)
    assert all(len(comp) == 1 for comp in comps), "input must be anchor-free"
    scale = len(set(legend_colors.values()))
    base_r, base_c, block_size, _step = _project_params(
        [(rr, cc, col) for (rr, cc), col in legend_colors.items()],
        inp.shape,
    )
    assert block_size == scale
    assert base_c >= 8, "projection must stay visually separate from the legend"

    # Cross-check with an explicit re-render so the generator stays output-first.
    expected = inp.copy()
    for (rr, cc), col in legend_colors.items():
        br = base_r + rr * (block_size + BLOCK_GAP)
        bc = base_c + cc * (block_size + BLOCK_GAP)
        expected[br : br + block_size, bc : bc + block_size] = col
    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
