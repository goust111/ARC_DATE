# Variant of c4d067a0
# Dimension: D4 — 变换：色块的颜色与形状 (transform)
# Change: 图例像素的颜色直接保留，放大为 NxN 的同色实心方块，无颜色变换... → 分形自相似放大

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import get_bg


# Design analysis for this variant
# a) Concrete behavior:
#    The top-left legend is still a sparse staircase of single pixels on a step-2 lattice.
#    The main area still lives on a coarser lattice inferred from two anchor placements.
#    The D4 change happens only inside each projected macro square: instead of a solid NxN
#    block, each macro square contains a shrunken copy of the full legend shape, drawn in
#    one uniform color (the color of the driving legend pixel). This creates one level of
#    self-similarity that is clearly visible in the output.
#    Example: if the legend shape occupies lattice rows/cols [(0,0), (0,1), (1,0), (2,0)],
#    the old solver filled every target square completely. The new solver maps those lattice
#    coordinates to positions spanning 0..N-1 inside the target square and paints only those
#    positions, leaving the rest as background.
# b) Complexity calibration:
#    To make the fractal visible, macro squares are restricted to side 4 or 5. This is large
#    enough for a sparse embedded copy of a 3-4 by 2-3 legend to be seen, but still small
#    enough that the original scale/offset inference remains the dominant reasoning step.
# c) Hidden-assumption audit against the original solver:
#    - The original component scan assumed every main-area object was a solid 4-connected
#      square component. That fails here because each macro square becomes a sparse set of
#      disconnected pixels. The new solver therefore extracts the legend first, then searches
#      the main area for exact translated occurrences of the recolored mini-legend mask.
#    - The original code could safely overwrite target squares with a solid color. That would
#      destroy the new sparse structure, so the new painter first clears each target square
#      to background and then paints only the mini-legend pixels.
#    - The original legend detector relied on singleton components. Here anchor pixels in the
#      main area are also singleton components, so generation enforces a larger column gap and
#      the solver reads the legend only from a dedicated top-left window.
# d) Placement legality:
#    The generator guarantees: block_size >= legend height/width in lattice units, gap=1 so
#    macro squares do not overlap, base_c is large enough that anchor tiles never enter the
#    legend window, and every projected square stays fully inside the canvas.
# e) Visual observability:
#    Across examples, the completed main area is visibly not solid. Each square repeats the
#    same staircase silhouette, so a human can compare the top-left legend with any filled
#    square and see the self-similar rule directly.
# f) Visual cleanliness:
#    Only two layers exist: the small legend and the sparse macro placements. There is no
#    extra border or marker system, and gap=1 keeps neighboring macro squares distinct.
# g) Color reasoning:
#    There is no hidden recoloring by color id. A legend pixel simply chooses the single color
#    used for the embedded mini-legend inside its macro square.


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 29, 32
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
BLOCK_SIZE_RANGE = range(4, 6)
BLOCK_GAP = 1
BASE_C_MIN = 12
BASE_R_MIN = 1
LEGEND_WINDOW = 12

# === CFG: generation ===
MAX_RETRIES = 400


def _spread_positions(count: int, size: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]
    assert size >= count, "mini-legend must fit inside the macro square"
    return [(i * (size - 1)) // (count - 1) for i in range(count)]


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


def _extract_legend(g: np.ndarray, bg: int):
    h, w = g.shape
    win_r = min(h, LEGEND_WINDOW)
    win_c = min(w, LEGEND_WINDOW)
    legend = [
        (r, c, int(g[r, c]))
        for r in range(win_r)
        for c in range(win_c)
        if int(g[r, c]) != bg
    ]
    assert legend, "top-left legend must contain colored pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = reduce(gcd, drs + dcs)
    assert stepL > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)
    legend_pts = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))
    legend_pts.sort()
    return legend, legend_pts


def _tile_relative_points(legend_pts, block_size: int):
    rows = max(rr for rr, _, _ in legend_pts) + 1
    cols = max(cc for _, cc, _ in legend_pts) + 1
    row_pos = _spread_positions(rows, block_size)
    col_pos = _spread_positions(cols, block_size)
    rel = sorted({(row_pos[rr], col_pos[cc]) for rr, cc, _ in legend_pts})
    assert rel and rel[0] == (0, 0), "staircase legend must touch macro-square top-left"
    assert max(r for r, _ in rel) == block_size - 1, "mini-legend must touch macro-square bottom"
    assert max(c for _, c in rel) == block_size - 1, "mini-legend must touch macro-square right"
    return rel


def _find_fractal_tiles(g: np.ndarray, bg: int, legend_abs, legend_pts):
    h, w = g.shape
    by_color = {}
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val == bg or (r, c) in legend_abs:
                continue
            by_color.setdefault(val, set()).add((r, c))
    assert by_color, "grid must contain macro-tile anchor pixels"

    best = None
    for block_size in BLOCK_SIZE_RANGE:
        rel_pts = _tile_relative_points(legend_pts, block_size)
        found = []
        for col, coords in by_color.items():
            for r0, c0 in sorted(coords):
                if r0 + block_size > h or c0 + block_size > w:
                    continue
                needed = {(r0 + dr, c0 + dc) for dr, dc in rel_pts}
                if not needed.issubset(coords):
                    continue
                ok = True
                for rr in range(r0, r0 + block_size):
                    for cc in range(c0, c0 + block_size):
                        val = int(g[rr, cc])
                        if (rr, cc) in needed:
                            if val != col:
                                ok = False
                                break
                        elif val != bg:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    found.append((int(r0), int(c0), int(block_size), int(col)))
        if len(found) < 2:
            continue

        tops = sorted({r0 for r0, _, _, _ in found})
        lefts = sorted({c0 for _, c0, _, _ in found})
        step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1) if tops[i + 1] > tops[i]]
        step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1) if lefts[i + 1] > lefts[i]]
        if not step_candidates:
            continue
        step = reduce(gcd, step_candidates)
        if step <= 0:
            continue

        placements_by_color = {}
        for r0, c0, _s, col in found:
            placements_by_color.setdefault(col, []).append((r0, c0))

        bases = {}
        for rr, cc, col in legend_pts:
            if col not in placements_by_color:
                continue
            for br0, bc0 in placements_by_color[col]:
                base = (br0 - rr * step, bc0 - cc * step)
                bases[base] = bases.get(base, 0) + 1
        if not bases:
            continue
        max_votes = max(bases.values())
        best_bases = [base for base, votes in bases.items() if votes == max_votes]
        if len(best_bases) != 1:
            continue

        candidate = (len(found), max_votes, block_size, step, best_bases[0], found, rel_pts)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None, "failed to identify fractal macro tiles"
    _count, _votes, block_size, step, base, found, rel_pts = best
    return int(block_size), int(step), tuple(map(int, base)), found, rel_pts


def _paint_fractal_tile(out: np.ndarray, r0: int, c0: int, block_size: int, rel_pts, col: int, bg: int):
    out[r0 : r0 + block_size, c0 : c0 + block_size] = bg
    for dr, dc in rel_pts:
        out[r0 + dr, c0 + dc] = col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a coarse block layout by replacing each target square with a self-similar mini-legend.

    Input structure:
    - The background is the most common color.
    - The top-left corner contains a sparse legend of single pixels on a regular lattice.
    - Elsewhere, only a few anchor placements are already present; each one is a square bbox whose
      occupied cells form a recolored miniature copy of the full legend shape.

    Rule:
    - Use the existing anchor placements to infer the macro square size, the coarse lattice step,
      and the translation from legend coordinates into the main area.
    - For every legend pixel, project its lattice position into the main area as before.
    - Instead of filling that square solidly, clear the square to background and draw the shrunken
      legend mask inside it, using the legend pixel's own color on every non-background mini cell.

    Key insight: The main-area anchors are not arbitrary sparse noise; each is the same mini-legend
    silhouette recolored uniformly, so matching those silhouettes reveals the same scale/offset map
    as the original task while changing only the internal block transformation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    legend_abs, legend_pts = _extract_legend(g, bg)
    legend_abs_set = {(r, c) for r, c, _ in legend_abs}
    block_size, step, (base_r, base_c), _found, rel_pts = _find_fractal_tiles(g, bg, legend_abs_set, legend_pts)

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "legend macro-square projection must fit in canvas"
        )
        _paint_fractal_tile(out, tr0, tc0, block_size, rel_pts, col, bg)
    return out


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size = random.choice(list(BLOCK_SIZE_RANGE))
    step = block_size + BLOCK_GAP
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    rel_pts = _tile_relative_points([(rr, cc, 1) for rr, cc in pts], block_size)

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * step
        bc = base_c + cc * step
        _paint_fractal_tile(out, br, bc, block_size, rel_pts, col, bg)

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
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
    raise RuntimeError("failed to generate valid c4d067a0 D4 input")
