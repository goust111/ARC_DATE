# Variant of c4d067a0
# Dimension: D1 — 匹配条件：坐标映射方式 (match_cond)
# Change: 图例像素位置通过统一缩放因子+偏移量映射到网格位置，缩放因子由已有色块的间距与图例像素间距的比值确定... → 锚点匹配缩放
#
# Design analysis
# ===============
# a) Concrete behavior and traced difference
#    - The top-left legend stays a sparse staircase of singleton pixels on a step-2 lattice.
#    - Exactly one legend pixel uses a unique anchor color. The main area contains exactly one
#      same-color square block. That block is the only prefilled enlarged block in the input.
#    - The anchor pixel at logical legend coordinate (ar, ac) maps to the top-left corner of the
#      anchor block. The anchor block side length N is the scaling factor. Every legend pixel
#      at (r, c) expands to an N x N block placed at:
#          (base_r + r * N, base_c + c * N)
#      where (base_r, base_c) is recovered from the anchor:
#          base_r = anchor_block_r0 - ar * N
#          base_c = anchor_block_c0 - ac * N
#    - Small example:
#      original logic: two visible blocks at legend cols 0 and 1 with top-lefts 10 and 14 imply
#      step=4, then colors vote for base.
#      variant logic: a unique cyan legend pixel at (1,2) and a cyan 3x3 block at (8,15) imply
#      N=3 and base=(5,9); then every legend dot expands with step 3 from that anchor.
#    - The visible difference is deliberate: spacing no longer comes from block-to-block gaps.
#      Blocks are packed on an N-scaled lattice whose pitch equals the anchor block size itself.
#
# b) Complexity calibration
#    - Using only the anchor block would be too easy if the enlarged output lost local structure,
#      so adjacent legend cells are forced to have different colors. This keeps tile boundaries
#      visible after gapless scaling.
#    - The anchor color is unique in the legend and absent from every other enlarged input block
#      because the input contains only the anchor block. This removes ambiguity without adding
#      hidden metadata.
#    - Palette size is 3-4, slightly higher than the original minimum, because gapless tiling
#      with a unique anchor color needs at least two additional colors to avoid flat monochrome
#      regions and preserve human-readable structure.
#
# c) Audit of original assumptions that break under this variant
#    - Original block-step inference via gcd(top-left differences) assumed at least two visible
#      blocks. That fails here because only the anchor block is shown.
#    - Original offset inference via color voting assumed multiple visible block colors and a
#      shared step already known. Here offset must come directly from the unique anchor match.
#    - Original dominant-block-size filtering was harmless because every visible block used the
#      same size. In this variant the anchor block itself is the scale carrier, so solver logic
#      should bind to the anchor match rather than to spacing statistics.
#
# d) Placement legality
#    - Legend remains in the top-left 12x12 window.
#    - Enlarged blocks start at col >= 8 so the main area stays visually separated.
#    - base_r/base_c are sampled so every projected N x N block fits the canvas.
#    - Input contains one isolated square block, so connected-component parsing is stable.
#
# e) Visual observability
#    - Across examples the anchor location changes inside the legend, and N varies from 2 to 4.
#      A human can see that the special legend color chooses the reference cell and that the
#      anchor block size determines the enlargement factor.
#    - No color-id ordering is used; only visible uniqueness and geometric alignment matter.
#
# f) Visual cleanliness
#    - The legend and the enlarged mosaic occupy separate regions.
#    - Orthogonally adjacent legend cells never share a color, so enlarged tiles do not merge
#      into indistinguishable same-color slabs.
#
# g) Color inference
#    - Colors are copied, not remapped.
#    - The only special color behavior is visible uniqueness of the anchor color, which is
#      supported structurally by the lone same-color block in the main area.
#
# New solver assumptions and generator guarantees
# -----------------------------------------------
# - Solver assumes there is exactly one color that appears once in the legend and also appears
#   as exactly one square block in the main area. Generator enforces this.
# - Solver assumes the legend occupies multiple rows/cols so the step-2 lattice can be inferred.
#   Generator uses 3-4 legend rows and 2-3 cols in staircase form.
# - Solver assumes the anchor block size is the intended scale for every legend pixel. Generator
#   constructs the full output using that rule and validates it with main().

import numpy as np
from functools import reduce
from math import gcd

from arc_utils import bbox_coords, components_mask, get_bg
import random


def _extract_singles_and_blocks(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    """Return singleton pixels and uniform square blocks as component summaries."""
    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a scaled block mosaic from a top-left legend using a unique anchor match.

    Input structure:
    - bg = most common color.
    - Top-left corner: a small staircase legend of singleton colored pixels on a regular lattice.
    - Main area: exactly one visible square block shares a unique color with one legend pixel; it is the anchor.

    Rule:
    - Identify the legend pixel whose color appears exactly once in the legend and also as the lone square block color.
    - Use that anchor block's side length as the scale factor, and align the anchor pixel to the anchor block.
    - Expand every legend pixel into a same-colored square block of that size at the correspondingly scaled position.

    Key insight: Unlike the original task, scale does not come from the spacing between visible blocks; it is encoded directly by the anchor block's size, and the anchor color fixes the offset.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _extract_singles_and_blocks(g, bg)
    assert singles and blocks, "grid must contain legend pixels and at least one block"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    diffs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    diffs += [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    assert diffs, "legend must span multiple rows or columns"
    stepL = reduce(gcd, diffs)
    assert stepL > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)
    legend_pts: list[tuple[int, int, int]] = []
    legend_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))
        legend_by_color.setdefault(int(v), []).append((int(rr), int(cc)))

    blocks_by_color: dict[int, list[tuple[int, int, int]]] = {}
    for r0, c0, size, col in blocks:
        blocks_by_color.setdefault(int(col), []).append((int(r0), int(c0), int(size)))

    anchor_matches: list[tuple[int, int, int, int, int, int]] = []
    for col, pts in legend_by_color.items():
        if len(pts) != 1:
            continue
        matches = blocks_by_color.get(int(col), [])
        if len(matches) != 1:
            continue
        (rr, cc) = pts[0]
        br0, bc0, block_size = matches[0]
        anchor_matches.append((int(col), int(rr), int(cc), int(br0), int(bc0), int(block_size)))

    assert len(anchor_matches) == 1, "grid must contain exactly one unique legend-to-block anchor match"
    _anchor_col, ar, ac, abr, abc, block_size = anchor_matches[0]
    assert block_size >= 2, "anchor block must be at least 2x2"

    base_r = abr - ar * block_size
    base_c = abc - ac * block_size

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * block_size
        tc0 = base_c + cc * block_size
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "scaled legend projection must fit in canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (o5x) ---------------

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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 3, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
BASE_C_MIN = 8
BASE_R_MIN = 1

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


def _neighbors4_in_pts(pts: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pos_set = set(pts)
    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {p: [] for p in pts}
    for r, c in pts:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pos_set:
                neigh[(r, c)].append(q)
    return neigh


def _assign_legend_colors(pts, anchor_pt, palette):
    """
    Color the staircase so:
    - anchor color is unique,
    - orthogonally adjacent cells never share a color,
    - every non-anchor palette color appears at least once,
    - no color exceeds MAX_COUNT_PER_COLOR uses.
    """
    neigh = _neighbors4_in_pts(pts)
    anchor_color = random.choice(palette)
    other_colors = [col for col in palette if col != anchor_color]
    colors = {anchor_pt: anchor_color}
    counts = {int(col): 0 for col in palette}
    counts[anchor_color] = 1
    cells = [p for p in pts if p != anchor_pt]
    unused = set(other_colors)

    def feasible(pos, col):
        if counts[col] >= MAX_COUNT_PER_COLOR:
            return False
        for q in neigh[pos]:
            if colors.get(q) == col:
                return False
        return True

    def search():
        if not cells:
            return len(unused) == 0
        best_i = None
        best_opts = None
        for i, pos in enumerate(cells):
            opts = []
            for col in other_colors:
                if not feasible(pos, col):
                    continue
                rem_unused = unused - {col}
                if len(rem_unused) > len(cells) - 1:
                    continue
                opts.append(col)
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] += 1
            removed = col in unused
            if removed:
                unused.remove(col)
            if search():
                return True
            if removed:
                unused.add(col)
            counts[col] -= 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _gen_case():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size = random.choice(list(BLOCK_SIZE_RANGE))
    pts = _pick_staircase()
    anchor_pt = random.choice(pts)

    bg = random.randrange(NUM_COLORS)
    k = random.randint(PALETTE_SIZE_MIN, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pt, palette)
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - (max_r + 1) * block_size
    c_hi = size - (max_c + 1) * block_size
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * block_size
        bc = base_c + cc * block_size
        out[br : br + block_size, bc : bc + block_size] = col

    inp = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    ar, ac = anchor_pt
    anchor_color = int(legend_colors[anchor_pt])
    abr = base_r + ar * block_size
    abc = base_c + ac * block_size
    inp[abr : abr + block_size, abc : abc + block_size] = anchor_color

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "block_size": int(block_size),
        "anchor_pt": (int(ar), int(ac)),
        "palette_size": int(k),
        "anchor_color": int(anchor_color),
    }
    return inp, out, meta


def generate_input():
    for _ in range(MAX_RETRIES):
        case = _gen_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate valid c4d067a0 D1 input")
