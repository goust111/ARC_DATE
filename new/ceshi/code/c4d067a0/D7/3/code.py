# Variant of c4d067a0
# Dimension: D7 — 缩放因子的确定方式 (symbol_instructions)
# Change: 缩放因子通过已有色块的大小隐式推断（色块边长即为缩放因子），色块间距与图例像素间距成比例... → 多级缩放：不同颜色使用不同缩放因子

import sys
import random
from functools import reduce
from math import gcd

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D7 variant
# a) Concrete behavior:
#    The top-left legend stays a sparse staircase of singleton pixels on a step-2 lattice.
#    The main area still uses one shared placement lattice `(base_r + rr*step, base_c + cc*step)`,
#    but the filled square placed at each lattice site now depends on color:
#    e.g. every red legend pixel becomes a 2x2 block, every blue one a 4x4 block.
#    Original flow: infer one dominant `block_size`, then paint every legend point with that size.
#    Variant flow: infer `size_by_color[color]` from the visible example blocks, keep one global
#    placement step/offset, then paint each legend point with the size attached to its own color.
# b) Complexity calibration:
#    Colors are limited to 2-3 and sizes to {2,3,4}. Each chosen color gets a distinct size.
#    This is visibly different from the seed task, but still small enough that repeated evidence
#    exists for each rule. Each color appears at least twice in the legend.
# c) Old implicit assumptions that break:
#    - Picking the dominant block size and filtering all other blocks is invalid now.
#    - Using one `block_size` in the final paint loop would overwrite correct smaller/larger blocks.
#    - Treating step inference as size inference is invalid; step stays global while size is per-color.
# d) Placement legality:
#    Blocks share a common top-left lattice and `step = max(scale_by_color.values()) + gap` with gap>=1,
#    so even the largest blocks cannot overlap adjacent sites. Base ranges ensure all blocks fit and the
#    main area remains separated from the legend.
# e) Visual observability:
#    Training pairs will show that blocks of different colors have different side lengths while occupying
#    the same coarse lattice. Since each color repeats, the rule "size follows color" is visible in 2-3
#    examples; it is not a hidden color-id ordering trick.
# f) Visual cleanliness:
#    The legend remains isolated in the top-left. The main-area lattice is regular, and varying sizes only
#    change how much of each lattice cell is filled; no overlaps or extra marker layers are introduced.
# g) Color-rule inferability:
#    No recoloring occurs. Color only selects which square size to use, and generator guarantees at least
#    one visible example block for every color that appears in the legend.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a grid of color-coded square blocks using a legend in the top-left corner.

    Input structure:
    - The background is the most common color.
    - In the top-left corner, singleton pixels form a sparse legend on a regular lattice.
    - In the main area, some legend positions already have solid square blocks, but different
      colors may use different square sizes while sharing the same coarse placement lattice.

    Rule:
    - Every legend pixel projects to one location in the main area through one common offset and step.
    - The visible blocks reveal two things: the shared placement lattice, and the square size
      associated with each color.
    - Fill in every missing legend position by placing a solid square of the matching color and
      the matching per-color size at its projected location.

    Key insight: the legend gives the spatial pattern, while the existing blocks split the mapping
    into a global position rule plus a color-specific size rule.
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

    size_by_color: dict[int, int] = {}
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, size, col in blocks:
        if col in size_by_color:
            assert size_by_color[col] == size, "each color must use one consistent block size"
        size_by_color[col] = int(size)
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))
    for col in by_color:
        by_color[col].sort()

    legend_colors = {col for _, _, col in legend_pts}
    assert legend_colors.issubset(size_by_color), "every legend color needs a visible size example"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = []
    step_candidates += [tops[i + 1] - tops[i] for i in range(len(tops) - 1) if tops[i + 1] > tops[i]]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1) if lefts[i + 1] > lefts[i]]
    assert step_candidates, "visible blocks must expose the placement step"
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
        size = size_by_color[col]
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - size and 0 <= tc0 <= w - size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + size, tc0 : tc0 + size] = col
    return out


# --------------- generator ---------------

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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 3
MAX_COUNT_PER_COLOR = 5
MIN_COUNT_PER_COLOR = 2

# === CFG: block area ===
SCALE_OPTIONS = (2, 3, 4)
GAP_RANGE = range(1, 3)
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
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
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    pairs = [(r, c) for r, width in enumerate(widths) for c in range(width - 1)]
    return pts, pairs


def _random_color_counts(total: int, k: int):
    counts = [MIN_COUNT_PER_COLOR] * k
    remaining = total - MIN_COUNT_PER_COLOR * k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        if not open_idxs:
            return None
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    count_list = _random_color_counts(len(pts), len(palette))
    if count_list is None:
        return None
    counts = dict(zip(palette, count_list))
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


def _pick_visible_positions(pts, legend_colors, anchor_pair):
    keep = [anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)]
    kept = set(keep)
    anchor_colors = {legend_colors[keep[0]], legend_colors[keep[1]]}
    for col in sorted(set(legend_colors.values()) - anchor_colors):
        candidates = [p for p in pts if legend_colors[p] == col and p not in kept]
        candidates.sort(key=lambda p: (p[0] == anchor_pair[0], abs(p[0] - anchor_pair[0]), p[1]))
        assert candidates, "every non-anchor color must have another position"
        keep.append(candidates[0])
        kept.add(candidates[0])
    return keep


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    max_palette = min(PALETTE_SIZE_MAX, len(SCALE_OPTIONS), len(pts), len(pts) // MIN_COUNT_PER_COLOR)
    k = random.randint(PALETTE_SIZE_MIN, max_palette)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    scales = random.sample(SCALE_OPTIONS, k)
    scale_by_color = dict(zip(palette, scales))
    max_scale = max(scale_by_color.values())
    gap = random.choice(tuple(GAP_RANGE))
    step = max_scale + gap

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - max_scale - max_r * step
    c_hi = size - max_scale - max_c * step
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        block_size = scale_by_color[col]
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    keep = set(_pick_visible_positions(pts, legend_colors, anchor_pair))
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        col = legend_colors[(rr, cc)]
        block_size = scale_by_color[col]
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
    raise RuntimeError("failed to generate valid c4d067a0 D7 input")
