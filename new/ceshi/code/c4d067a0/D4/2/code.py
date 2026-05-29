# Variant of c4d067a0
# Dimension: D4 — 变换：色块的颜色与形状 (transform)
# Change: 图例像素的颜色直接保留，放大为 NxN 的同色实心方块，无颜色变换... → 根据图例位置调整色块大小

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D4 variant
# a) Concrete behavior:
#    - The top-left legend is still a sparse staircase of single pixels on a step-2 lattice.
#    - The main area still places one solid square per legend pixel on a coarser lattice.
#    - The new visible change is that block size depends on legend column:
#      col 0 -> 2x2, col 1 -> 3x3, col 2 -> 4x4.
#    - Example trace:
#      original: legend (r=1,c=2,color=red) -> fixed 3x3 red block.
#      variant:  legend (r=1,c=2,color=red) -> 4x4 red block because it is in legend column 2.
#      The scale/offset inference is unchanged, but the render size now comes from column index.
# b) Complexity calibration:
#    - Only three size classes are used, so the rule is easy to spot and repeatedly supported.
#    - The legend is forced to use 3 columns somewhere, guaranteeing all 2x2/3x3/4x4 blocks appear.
#    - Two input anchor blocks remain sufficient because adjacent columns have different sizes.
# c) Hidden assumptions from the original solver that must change:
#    - "dominant block size" is invalid because blocks are no longer uniform.
#    - Painting every projected block with one global size would overwrite correct larger/smaller shapes.
#    - Offset voting cannot use color alone; it must also match the expected size for the legend column.
# d) Placement legality:
#    - A constant top-left step is preserved, but the generator increases it to 5 or 6 so 4x4 blocks fit
#      with at least one background cell of separation from the next row/column of blocks.
#    - All projected blocks are checked to fit inside the canvas before acceptance.
# e) Visual observability:
#    - The size rule depends only on visible spatial position, never on color ids.
#    - Because the legend always reaches column 2, every solved sample displays all three sizes.
# f) Visual cleanliness:
#    - Legend and block field stay separated horizontally.
#    - Blocks never overlap because the generator enforces step > max block size.
# g) Color-rule audit:
#    - Colors are preserved exactly; there is no hidden color mapping to infer.
# h) New solver assumptions introduced here:
#    - Main-area objects are still solid monochrome squares.
#    - The size rule is fixed globally as size = 2 + legend column.
#    - Generator guarantees at least two horizontally adjacent legend pixels so the input still has 2 anchors.


BASE_BLOCK_SIZE = 2


def _size_for_col(cc: int) -> int:
    return BASE_BLOCK_SIZE + int(cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a field of colored squares from a legend whose columns determine the block sizes.

    Input structure:
    - The background is the dominant color.
    - In the top-left corner, single colored pixels form a staircase legend on a regular sparse lattice.
    - Elsewhere, a few solid colored squares already appear on a coarser lattice; their top-left alignment matches the legend, but their side lengths vary by legend column.

    Rule:
    - Each legend pixel keeps its color and projects to one square block in the main area.
    - The block lattice offset and spacing are inferred from the existing squares, just as in the original task.
    - The new twist is that block size is column-dependent: legend column 0 makes a 2x2 block, column 1 makes a 3x3 block, column 2 makes a 4x4 block.

    Key insight:
    - The existing blocks still reveal the legend-to-grid translation, but now a block's size also tells you which legend column it came from.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []  # (r0, c0, size, color)
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

    assert singles and blocks, "grid must contain legend pixels and main blocks"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = reduce(gcd, drs + dcs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

    step_candidates: list[int] = []
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates += [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "existing blocks must reveal the block lattice step"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        expected_size = _size_for_col(cc)
        for br0, bc0, size, bcol in blocks:
            if bcol != col or size != expected_size:
                continue
            base_r = br0 - rr * step
            base_c = bc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    assert bases, "existing blocks must determine the legend-to-grid offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    base_r, base_c = best_bases[0]

    for rr, cc, col in legend_pts:
        size = _size_for_col(cc)
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - size and 0 <= tc0 <= w - size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + size, tc0 : tc0 + size] = col
    return out


# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (3,)
LEGEND_EMPTIES = (0, 1, 2, 3)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
STEP_MIN, STEP_MAX = 5, 6
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
    """Pick a staircase shape; return filled points and horizontal adjacency anchors."""
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _random_color_counts(total, k):
    """Distribute `total` among `k` bins, each in [1, MAX_COUNT_PER_COLOR]."""
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[i]))
        counts[i] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    """Assign colors while keeping the chosen horizontal anchor pair unique."""
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    colors = {first: a, second: b}
    counts[a] -= 1
    counts[b] -= 1
    if counts[a] < 0 or counts[b] < 0:
        return None

    anchor_colors = (a, b)
    cells = [p for p in pts if p not in colors]

    def allowed(pos, col):
        r, c = pos
        left = (r, c - 1)
        if left in pos_set and left in colors:
            if (colors[left], col) == anchor_colors and (left, pos) != (first, second):
                return False
        right = (r, c + 1)
        if right in pos_set and right in colors:
            if (col, colors[right]) == anchor_colors and (pos, right) != (first, second):
                return False
        return True

    def search():
        if not cells:
            return all(v == 0 for v in counts.values())
        best_i, best_opts = None, None
        for i, pos in enumerate(cells):
            opts = [col for col in palette if counts[col] > 0 and allowed(pos, col)]
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    step = random.randint(STEP_MIN, STEP_MAX)
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_r_extent = max(rr * step + _size_for_col(cc) for rr, cc in pts)
    max_c_extent = max(cc * step + _size_for_col(cc) for _, cc in pts)
    r_hi = size - max_r_extent
    c_hi = size - max_c_extent
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        block_size = _size_for_col(cc)
        br, bc = base_r + rr * step, base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        block_size = _size_for_col(cc)
        br, bc = base_r + rr * step, base_c + cc * step
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
    raise RuntimeError("failed to generate valid c4d067a0 D4 input")
