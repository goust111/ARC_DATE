# Variant of c4d067a0
# Dimension: D8 — 多步处理顺序 (multi_step)
# Change: Step 1: 识别左上角图例精灵; Step 2: 定位网格中已有的同色色块; Step 3: 计算缩放因子和偏移; ... → 先放大再删除干扰项

import sys
import random
from functools import reduce
from math import gcd

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D8 variant
# a) Concrete behavior and trace:
#    - The top-left legend is still the same sparse staircase sprite on a step-2 lattice.
#    - The main area now contains two kinds of NxN monochrome squares:
#      1. real anchor squares that lie exactly on the legend's scaled lattice;
#      2. distractor squares with legend colors but off that lattice.
#    - Output keeps only the true scaled pattern: missing true blocks are added and every
#      distractor square is erased back to background.
#    - Small trace:
#      Original: if legend has red at (0,0) and blue at (0,1), and input shows those two
#      blocks at (10,10) and (10,14), the solver directly infers step=4, base=(10,10).
#      Variant: the same input may also contain an extra blue block at (13,18). The old
#      "use every block as evidence" logic would let this off-lattice block vote for bad
#      bases. The new logic must first test candidate (step, base) pairs and count which
#      existing blocks fit the legend projection, then ignore the outlier before filling
#      all projected blocks and wiping the outlier.
# b) Complexity calibration:
#    - Keep 3-4 true anchor blocks and 1-3 distractors. This is enough for a visible
#      "majority pattern vs. off-grid noise" effect without making the scene chaotic.
#    - Step values stay in the original small range 3-5, so the solver can enumerate a
#      tiny candidate space instead of needing an unconstrained search.
# c) Implicit assumptions audited against the original main():
#    - Original "gcd of all block top-left differences" assumes every same-size block is
#      valid. That breaks immediately once distractors exist, so it is replaced.
#    - Original base voting also assumes every same-size block provides trustworthy votes.
#      In the variant, only the subset consistent with one candidate mapping is reliable.
#    - Original unconditional painting onto a copy of the input leaves distractors intact.
#      The variant must erase all same-size main-area blocks first, then repaint only the
#      projected true pattern.
# d) Placement legality:
#    - Generator prevents overlap among legend pixels, true blocks, and distractor blocks.
#    - Distractors are forced off the true lattice and away from every projected true slot.
#    - All projected true blocks fit completely inside the canvas before the sample is kept.
# e) Visual observability:
#    - Humans can see a regular block lattice suggested by several aligned anchors plus the
#      legend, while the distractors sit "off rhythm". Output both completes the lattice and
#      removes those out-of-rhythm blocks, so the D8 change is visible.
# f) Visual cleanliness:
#    - Distractors use the same square size as real blocks, but only 1-3 are added and they
#      stay in the same general region without overlapping the true lattice.
# g) Color inferability:
#    - Distractors reuse legend colors only; there is no hidden color remapping or color-id
#      ordering. Colors remain directly readable from the legend.
# 9) New solver assumptions and generator guarantees:
#    - Solver assumes the dominant square size belongs to all main-area blocks. Generator
#      enforces that both true blocks and distractors use that same size.
#    - Solver assumes there is exactly one candidate (step, base) with maximal support from
#      existing blocks and that at least 3 blocks support it. Generator validates this by
#      solving every sample before returning it.


# === Shared search constraints ===
SEARCH_STEP_MIN, SEARCH_STEP_MAX = 3, 5
MIN_TRUE_BLOCKS = 3


def _extract_objects(g: np.ndarray, bg: int):
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


def _extract_legend(singletons: list[tuple[int, int, int]], shape: tuple[int, int]):
    h, w = shape
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singletons if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = reduce(gcd, drs + dcs)
    assert step_l > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
    return legend_pts


def _dominant_block_size(blocks: list[tuple[int, int, int, int]]) -> int:
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _project_positions(
    legend_pts: list[tuple[int, int, int]],
    step: int,
    base_r: int,
    base_c: int,
    block_size: int,
    shape: tuple[int, int],
):
    h, w = shape
    projected: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return None
        projected[(int(tr0), int(tc0))] = int(col)
    return projected


def _find_mapping(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    block_size: int,
    shape: tuple[int, int],
):
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    candidates: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    step_lo = max(SEARCH_STEP_MIN, block_size + 1)
    for step in range(step_lo, SEARCH_STEP_MAX + 1):
        for rr, cc, col in legend_pts:
            for br0, bc0 in by_color.get(col, []):
                base_r = br0 - rr * step
                base_c = bc0 - cc * step
                key = (int(step), int(base_r), int(base_c))
                if key in candidates:
                    continue
                projected = _project_positions(legend_pts, step, base_r, base_c, block_size, shape)
                if projected is None:
                    continue
                support = 0
                support_colors: set[int] = set()
                support_rows: set[int] = set()
                conflicts = 0
                for r0, c0, _s, bcol in blocks:
                    exp = projected.get((int(r0), int(c0)))
                    if exp == int(bcol):
                        support += 1
                        support_colors.add(int(bcol))
                        support_rows.add(int((r0 - base_r) // step))
                    else:
                        conflicts += 1
                candidates[key] = (support, len(support_colors), len(support_rows), -conflicts)

    assert candidates, "existing blocks must propose at least one mapping"
    best_score = max(candidates.values())
    best_keys = [k for k, v in candidates.items() if v == best_score]
    assert len(best_keys) == 1, "legend-to-grid mapping must be unique"
    step, base_r, base_c = best_keys[0]
    support = best_score[0]
    assert support >= MIN_TRUE_BLOCKS, "variant requires at least three aligned true blocks"
    return int(step), int(base_r), int(base_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the scaled block pattern defined by the top-left legend while removing off-lattice distractor blocks.

    Input structure:
    - bg is the most common color.
    - Top-left: singleton legend pixels on a small regular lattice, describing the intended color layout.
    - Main area: several same-size square blocks; some belong to the legend's scaled layout, and some are distractors placed at inconsistent positions.

    Rule:
    - Use the legend as the only authoritative template for color and relative position.
    - Among the visible square blocks, find the unique scale/offset whose projected legend positions explain the largest consistent subset of blocks, ignoring off-lattice outliers.
    - Repaint the full projected layout and clear every distractor block that does not belong to that layout.

    Key insight:
    - The true blocks form a coherent scaled copy of the legend, while distractors cannot all agree on one scale-and-offset hypothesis; the best-supported hypothesis identifies which blocks to trust.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    singles, blocks = _extract_objects(g, bg)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legend_pts = _extract_legend(singles, (h, w))
    block_size = _dominant_block_size(blocks)
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    step, base_r, base_c = _find_mapping(legend_pts, blocks, block_size, (h, w))
    projected = _project_positions(legend_pts, step, base_r, base_c, block_size, (h, w))
    assert projected is not None

    out = g.copy()
    for r0, c0, s, _col in blocks:
        out[r0 : r0 + s, c0 : c0 + s] = bg

    for (tr0, tc0), col in projected.items():
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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: noisy variant ===
TRUE_KEEP_RANGE = (3, 4)
DISTRACTOR_RANGE = (1, 3)
MAX_SUBSET_TRIES = 40
MAX_DISTRACTOR_PLACEMENT_TRIES = 400

# === CFG: generation ===
MAX_RETRIES = 500


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
_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if SEARCH_STEP_MIN <= n + g <= SEARCH_STEP_MAX
]


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    return pts


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


def _assign_legend_colors(pts, palette):
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    colors = {}
    cells = list(pts)
    random.shuffle(cells)
    for pos in cells:
        opts = [col for col in palette if counts[col] > 0]
        if not opts:
            return None
        col = random.choice(opts)
        colors[pos] = col
        counts[col] -= 1
    return colors


def _render_output(size, bg, legend_colors, block_size, step, base_r, base_c):
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col
    return out


def _choose_true_subset(pts, legend_colors):
    max_keep = min(TRUE_KEEP_RANGE[1], len(pts) - 2)
    min_keep = min(TRUE_KEEP_RANGE[0], max_keep)
    if max_keep < TRUE_KEEP_RANGE[0]:
        return None
    for _ in range(MAX_SUBSET_TRIES):
        keep_n = random.randint(min_keep, max_keep)
        keep = set(random.sample(pts, keep_n))
        colors = {legend_colors[p] for p in keep}
        rows = {p[0] for p in keep}
        if len(colors) >= 2 and len(rows) >= 2:
            return keep
    return None


def _make_occupied(size, legend_colors, block_size, step, base_r, base_c):
    occ = np.zeros((size, size), dtype=bool)
    lr0, lc0 = LEGEND_ORIGIN
    for rr, cc in legend_colors:
        occ[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = True
        br = base_r + rr * step
        bc = base_c + cc * step
        occ[br : br + block_size, bc : bc + block_size] = True
    return occ


def _place_distractors(inp, occ, palette, count, block_size, base_r, base_c, step, max_r, max_c):
    size = inp.shape[0]
    r_lo = max(0, base_r - step)
    r_hi = min(size - block_size, base_r + max_r * step + step)
    c_lo = max(BASE_C_MIN, base_c - step)
    c_hi = min(size - block_size, base_c + max_c * step + step)
    placed = 0
    tries = 0
    while placed < count and tries < MAX_DISTRACTOR_PLACEMENT_TRIES:
        tries += 1
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        if (r0 - base_r) % step == 0 and (c0 - base_c) % step == 0:
            continue
        if occ[r0 : r0 + block_size, c0 : c0 + block_size].any():
            continue
        col = random.choice(palette)
        inp[r0 : r0 + block_size, c0 : c0 + block_size] = col
        occ[r0 : r0 + block_size, c0 : c0 + block_size] = True
        placed += 1
    return placed == count


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
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out = _render_output(size, bg, legend_colors, block_size, step, base_r, base_c)

    keep = _choose_true_subset(pts, legend_colors)
    if keep is None:
        return None

    inp = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        if (rr, cc) in keep:
            br = base_r + rr * step
            bc = base_c + cc * step
            inp[br : br + block_size, bc : bc + block_size] = col

    occ = _make_occupied(size, legend_colors, block_size, step, base_r, base_c)
    distractor_count = random.randint(DISTRACTOR_RANGE[0], DISTRACTOR_RANGE[1])
    if not _place_distractors(inp, occ, palette, distractor_count, block_size, base_r, base_c, step, max_r, max_c):
        return None

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
    raise RuntimeError("failed to generate valid c4d067a0 D8 variant input")
