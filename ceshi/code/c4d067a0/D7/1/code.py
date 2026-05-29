# Variant of c4d067a0
# Dimension: D7 — 缩放因子的确定方式 (symbol_instructions)
# Change: 缩放因子通过已有色块的大小隐式推断（色块边长即为缩放因子），色块间距与图例像素间距成比例... → 缩放因子由网格边缘色条长度决定

import os
import sys
from functools import reduce
from math import gcd
import random

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D7 variant
# a) Concrete behavior:
#    - The top-left legend stays the same: isolated colored pixels on a step-2 lattice.
#    - The main area no longer shows full-size anchor blocks. Instead it shows two small
#      seed squares at the top-left corners of two future blocks. A border bar gives the
#      missing scale: its straight-line length equals the final block side length.
#    - Example trace: in the original task, if the visible anchors are 3x3 squares then
#      the solver can directly conclude "every legend pixel expands to 3x3". In this
#      variant the visible anchors are always 2x2 seeds, so that inference is invalid;
#      the solver must read a length-4 bar on the bottom or right edge and expand every
#      legend pixel to 4x4 instead.
# b) Complexity calibration:
#    - The scale only takes a few values (3 or 4), which is enough to make the bar matter
#      visually while keeping the symbol easy to decode across examples.
#    - The bar uses a dedicated non-legend color so it is visually distinct from both
#      legend dots and seed squares.
# c) Old implicit assumptions that must change:
#    - Original code assumes every square component in the main area already has the final
#      side length. That is false here: square components are 2x2 seeds only.
#    - Original code paints using the detected square size. That would under-fill the
#      answer in this variant, so the new solver must separate "seed size" from "final
#      block size".
#    - Original code can infer scale without extra symbols. Here that would be a harmful
#      shortcut, so generation guarantees the final scale is always larger than the seed.
# d) Placement legality:
#    - All final blocks fit inside the canvas while leaving the bottom row or rightmost
#      column free for the border bar.
#    - The two seeds sit inside their corresponding final blocks, at the exact future
#      top-left corners, so expanding them cannot misalign the output.
# e) Visual observability:
#    - Humans can see that the anchors are too small to explain the final output by
#      themselves; the border strip is the only length-bearing symbol, so the new rule is
#      legible from examples.
# f) Visual cleanliness:
#    - Legend, border bar, and main lattice occupy separate regions. The bar is on the
#      outer border; seeds/blocks stay one cell away from that border.
# g) Color inference:
#    - No invisible color ordering is used. Colors only support matching legend dots to
#      seeds, as in the original task.
# New solver assumptions and generator guarantees:
#    - There is exactly one border bar, always a 1xk or kx1 monochrome component touching
#      the grid edge, with k in {3, 4}.
#    - There are exactly two solid square seeds, both size 2, placed on the same row and
#      one lattice step apart.
#    - Their ordered color pair is unique among adjacent horizontal legend pairs, so base
#      offset voting has a unique winner.


def _solid_square_info(g: np.ndarray, pts: np.ndarray, bg: int):
    if len(pts) < 4:
        return None
    r0, c0, r1, c1 = bbox_coords(pts)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh != ww:
        return None
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    col = int(sub[0, 0])
    if col == bg:
        return None
    if np.all(sub == col) and sub.size == len(pts):
        return int(r0), int(c0), int(hh), col
    return None


def _find_scale_bar(g: np.ndarray, comps: list[np.ndarray], bg: int):
    h, w = g.shape
    bars = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if not touches_edge:
            continue
        if hh == 1 or ww == 1:
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            col = int(sub[0, 0])
            if col != bg and np.all(sub == col) and sub.size == len(pts):
                bars.append((max(hh, ww), int(r0), int(c0), int(hh), int(ww), col))
    assert bars, "grid must contain a border scale bar"
    max_len = max(item[0] for item in bars)
    bars = [item for item in bars if item[0] == max_len]
    assert len(bars) == 1, "scale bar must be unique"
    return bars[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged blocks from a legend, two seed squares, and a border scale bar.

    Input structure:
    - bg is the dominant background color.
    - The top-left corner contains isolated legend pixels on a small regular lattice.
    - The grid edge contains one straight monochrome bar whose length encodes the final block size.
    - The main area contains two small uniform seed squares marking the positions of two future blocks.

    Rule:
    - Read the enlargement size from the edge bar length.
    - Use the two seed squares and their colors to align the legend lattice with the main-area lattice and recover the translation/step.
    - For every legend pixel, paint a full square block of the decoded size at the corresponding main-area location.

    Key insight:
    - The seed squares only reveal where the projected lattice sits; they are intentionally too small to reveal the final block size.
    - The edge bar supplies that missing size symbol, so solving requires combining positional evidence with symbol decoding.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain non-background objects"

    bar_len, _bar_r0, _bar_c0, _bar_h, _bar_w, _bar_col = _find_scale_bar(g, comps, bg)
    scale = int(bar_len)

    singles: list[tuple[int, int, int]] = []
    seeds: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        sq = _solid_square_info(g, pts, bg)
        if sq is not None:
            seeds.append(sq)

    assert singles and seeds, "grid must contain legend pixels and seed squares"

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

    sizes = sorted({s for _, _, s, _ in seeds})
    assert len(sizes) == 1, "seed squares must share one size"
    seeds.sort(key=lambda item: (item[0], item[1]))
    assert len(seeds) == 2, "input must contain exactly two seed squares"
    (r0a, c0a, _seed_size, _cola), (r0b, c0b, _seed_size2, _colb) = seeds
    assert r0a == r0b and c0b > c0a, "seed squares must form one horizontal anchor pair"
    step = c0b - c0a
    assert step > scale, "lattice step must leave a positive gap between final blocks"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in seeds:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            base_r = br0 - rr * step
            base_c = bc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    assert bases, "seed squares must determine the legend-to-grid offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    base_r, base_c = best_bases[0]

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - scale and 0 <= tc0 <= w - scale, \
            "projected block must fit inside the canvas"
        out[tr0 : tr0 + scale, tc0 : tc0 + scale] = col
    return out


# --------------- generator (D7 variant) ---------------

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

# === CFG: block area ===
SCALE_RANGE = range(3, 5)   # true output block size, encoded by the edge bar
SEED_SIZE = 2               # visible square clue size in the input
GAP_RANGE = range(1, 4)
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
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _random_color_counts(total, k):
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


def _sample_bar_placement(size, scale):
    if random.choice([False, True]):
        return "bottom", random.randint(0, size - scale)
    return "right", random.randint(0, size - scale)


def _place_bar(grid, bar_color, scale, placement):
    size = grid.shape[0]
    side, start = placement
    if side == "bottom":
        grid[size - 1, start : start + scale] = bar_color
    else:
        grid[start : start + scale, size - 1] = bar_color


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    scale = random.choice(list(SCALE_RANGE))
    gap = random.choice(list(GAP_RANGE))
    step = scale + gap
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    usable_colors = [c for c in range(NUM_COLORS) if c != bg]
    palette = random.sample(usable_colors, k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    bar_choices = [c for c in usable_colors if c not in palette]
    if not bar_choices:
        return None
    bar_color = random.choice(bar_choices)

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - scale - max_r * step - 1
    c_hi = size - scale - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br, bc = base_r + rr * step, base_c + cc * step
        out[br : br + scale, bc : bc + scale] = col
    placement = _sample_bar_placement(size, scale)
    _place_bar(out, bar_color, scale, placement)

    inp = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    _place_bar(inp, bar_color, scale, placement)

    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in keep:
        col = legend_colors[(rr, cc)]
        br, bc = base_r + rr * step, base_c + cc * step
        inp[br : br + SEED_SIZE, bc : bc + SEED_SIZE] = col

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
    raise RuntimeError("failed to generate valid c4d067a0 D7 variant input")
