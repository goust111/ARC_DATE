# Variant of c4d067a0
# Dimension: D1 — 匹配条件：坐标映射方式 (match_cond)
# Change: 图例像素位置通过统一缩放因子+偏移量映射到网格位置，缩放因子由已有色块的间距与图例像素间距的比值确定... → 镜像翻转后缩放映射

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis:
# - Concrete behavior: the top-left legend remains a sparse staircase of colored singleton
#   pixels on a step-2 lattice. The main area still contains same-colored square blocks on a
#   coarser regular lattice, but their lattice coordinates are no longer a direct copy of the
#   legend. Before scaling into block positions, the legend lattice is mirrored either left-right
#   or top-bottom around its occupied bounding box. Example on lattice coords:
#   legend {(0,0):A, (0,1):B, (1,0):C}, horizontal flip -> {(0,1):A, (0,0):B, (1,1):C}.
#   The original solver would map A,B,C to base+(0,0),(0,1),(1,0); the variant must instead map
#   them to base+(0,1),(0,0),(1,1). The changed step is therefore the coordinate projection only;
#   object extraction, scale inference, and block painting stay the same.
# - Complexity calibration: keeping exactly two visible blocks preserves the original task's sparse
#   evidence level, but the mirror choice adds one extra binary decision. To avoid accidental
#   simplification, the generator rejects legend shapes/colorings that are symmetric under the
#   chosen flip, because such samples would make the mirrored output visually identical to the
#   direct-copy output.
# - Hidden assumptions audited from the original main():
#   * `legend_pts` used to be projected as `(rr, cc) -> base + step*(rr, cc)`. This no longer
#     holds; the new solver must transform `(rr, cc)` by a mirror before using the same affine map.
#   * The original base-voting logic implicitly assumed only one projection mode existed. Here we
#     must vote over `(mode, base)` pairs and require a unique winner.
#   * Using `max_r`/`max_c` as mirror axes assumes the occupied legend bbox is tight. The staircase
#     generator guarantees that: every row has at least one cell, and at least one row reaches the
#     rightmost used column.
#   * Unconditional painting remains safe because all projected blocks are uniform and generator
#     placement prevents overlaps or out-of-bounds projections.
# - Placement legality: block size, gap, and base are still chosen so every projected block fits
#   within the canvas and the block area stays separated from the legend columns. Mirror transforms
#   preserve the lattice extents, so the original fit calculation remains valid once applied to the
#   transformed coordinates.
# - Visual observability: humans can compare the legend and the completed block layout across a few
#   examples and notice that either left-right or top-bottom order is reversed before magnification.
#   The generator varies both flip directions and rejects samples where mirrored and direct layouts
#   would coincide.
# - Visual cleanliness: no new markers or overlays are introduced. The grid still has two clearly
#   separated zones: tiny legend dots at top-left and large solid blocks in the main area.
# - Color inference: colors are copied, not remapped. The only new reasoning burden is choosing the
#   correct mirror orientation from visible spatial/color correspondences.


def _extract_primitives(g: np.ndarray):
    bg = get_bg(g)
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
    return int(bg), singles, blocks


def _extract_legend(singles: list[tuple[int, int, int]], h: int, w: int):
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
    legend_pts.sort()
    max_r = max(rr for rr, _, _ in legend_pts)
    max_c = max(cc for _, cc, _ in legend_pts)
    return legend_pts, stepL, max_r, max_c


def _dominant_block_size(blocks: list[tuple[int, int, int, int]]) -> int:
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _block_step(blocks: list[tuple[int, int, int, int]]) -> int:
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"
    return int(step)


def _transform_coord(rr: int, cc: int, max_r: int, max_c: int, mode: str) -> tuple[int, int]:
    if mode == "identity":
        return rr, cc
    if mode == "h":
        return rr, max_c - cc
    if mode == "v":
        return max_r - rr, cc
    raise ValueError(f"unknown mode: {mode}")


def _transform_legend(
    legend_pts: list[tuple[int, int, int]], max_r: int, max_c: int, mode: str
) -> list[tuple[int, int, int]]:
    out = []
    for rr, cc, col in legend_pts:
        trr, tcc = _transform_coord(rr, cc, max_r, max_c, mode)
        out.append((int(trr), int(tcc), int(col)))
    out.sort()
    return out


def _projection_fits(
    transformed: list[tuple[int, int, int]],
    base_r: int,
    base_c: int,
    step: int,
    block_size: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc, _ in transformed:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return False
    return True


def _build_expected_blocks(
    transformed: list[tuple[int, int, int]], base_r: int, base_c: int, step: int
) -> dict[tuple[int, int], int]:
    return {
        (int(base_r + rr * step), int(base_c + cc * step)): int(col)
        for rr, cc, col in transformed
    }


def _infer_mapping(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    block_size: int,
    step: int,
    h: int,
    w: int,
    *,
    modes: tuple[str, ...] = ("h", "v"),
):
    max_r = max(rr for rr, _, _ in legend_pts)
    max_c = max(cc for _, cc, _ in legend_pts)

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, s, col in blocks:
        if s != block_size:
            continue
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    candidates = []
    for mode in modes:
        transformed = _transform_legend(legend_pts, max_r, max_c, mode)
        base_votes: dict[tuple[int, int], int] = {}
        for rr, cc, col in transformed:
            for br0, bc0 in by_color.get(col, []):
                base = (int(br0 - rr * step), int(bc0 - cc * step))
                base_votes[base] = base_votes.get(base, 0) + 1

        for (base_r, base_c), votes in base_votes.items():
            if not _projection_fits(transformed, base_r, base_c, step, block_size, h, w):
                continue
            expected = _build_expected_blocks(transformed, base_r, base_c, step)
            ok = True
            for r0, c0, s, col in blocks:
                if s != block_size or expected.get((r0, c0)) != col:
                    ok = False
                    break
            if ok:
                candidates.append((int(votes), mode, int(base_r), int(base_c), transformed))

    assert candidates, "existing blocks must determine a mirrored legend mapping"
    max_votes = max(v for v, *_ in candidates)
    best = [cand for cand in candidates if cand[0] == max_votes]
    assert len(best) == 1, "mirrored legend mapping must be unique"
    _, mode, base_r, base_c, transformed = best[0]
    return mode, base_r, base_c, transformed


def _solve_core(grid: np.ndarray, *, modes: tuple[str, ...]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg, singles, blocks = _extract_primitives(g)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legend_pts, _stepL, _max_r, _max_c = _extract_legend(singles, h, w)

    block_size = _dominant_block_size(blocks)
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"
    step = _block_step(blocks)

    _mode, base_r, base_c, transformed = _infer_mapping(
        legend_pts, blocks, block_size, step, h, w, modes=modes
    )

    for rr, cc, col in transformed:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "legend block projection must fit in canvas"
        )
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a grid of colored blocks using a mirrored version of the top-left legend.

    Input structure:
    - bg is the dominant color.
    - In the top-left corner there is a sparse legend made of singleton colored pixels on a
      small regular lattice.
    - In the main area there are uniformly colored square blocks on a coarser regular lattice,
      but only a few of the legend positions are already instantiated as blocks.

    Rule:
    - The main block layout is obtained from the legend after first applying either a horizontal
      or a vertical mirror across the legend's occupied lattice bbox, and then applying one
      uniform scale-and-offset map.
    - The existing blocks reveal both the lattice step and which mirror orientation is active.
    - Paint every mirrored legend position as a same-colored square block, preserving the blocks
      that are already present.

    Key insight:
    - The visible blocks are not aligned with the legend in its original orientation. One must
      test the two mirror hypotheses, find the unique mirrored coordinate system that explains the
      anchors, and only then expand the full block pattern.
    """
    return _solve_core(grid, modes=("h", "v"))


# --------------- generator (o5x mirrored projection) ---------------

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
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 300
MIRROR_MODES = ("h", "v")


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


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return widths, pts, pairs


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


def _shape_visible_under_mode(widths: tuple[int, ...], mode: str) -> bool:
    if mode == "h":
        return len(set(widths)) > 1 or widths[0] != max(widths)
    if mode == "v":
        return widths != tuple(reversed(widths))
    raise ValueError(mode)


def _colored_layout_changes(legend_colors: dict[tuple[int, int], int], mode: str) -> bool:
    pts = [(rr, cc, col) for (rr, cc), col in legend_colors.items()]
    max_r = max(rr for rr, _, _ in pts)
    max_c = max(cc for _, cc, _ in pts)
    original = frozenset(pts)
    mirrored = frozenset(_transform_legend(pts, max_r, max_c, mode))
    return mirrored != original


def _render_output(
    size: int,
    bg: int,
    legend_colors: dict[tuple[int, int], int],
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
    mode: str,
):
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    pts = [(rr, cc, col) for (rr, cc), col in legend_colors.items()]
    max_r = max(rr for rr, _, _ in pts)
    max_c = max(cc for _, cc, _ in pts)
    transformed = _transform_legend(pts, max_r, max_c, mode)

    for rr, cc, col in pts:
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    for rr, cc, col in transformed:
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col
    return out, transformed


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    widths, pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)
    mode = random.choice(MIRROR_MODES)
    if not _shape_visible_under_mode(widths, mode):
        return None

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None or not _colored_layout_changes(legend_colors, mode):
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out, transformed = _render_output(
        size, bg, legend_colors, block_size, step, base_r, base_c, mode
    )
    plain_out, _ = _render_output(
        size, bg, legend_colors, block_size, step, base_r, base_c, "identity"
    )
    if np.array_equal(out, plain_out):
        return None

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc, col in transformed:
        src = None
        for key, value in legend_colors.items():
            trr, tcc = _transform_coord(key[0], key[1], max_r, max_c, mode)
            if (trr, tcc, value) == (rr, cc, col):
                src = key
                break
        assert src is not None
        if src in keep:
            continue
        br, bc = base_r + rr * step, base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    try:
        legacy = _solve_core(inp, modes=("identity",))
    except Exception:
        legacy = None
    if legacy is not None and np.array_equal(legacy, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 mirrored input")
