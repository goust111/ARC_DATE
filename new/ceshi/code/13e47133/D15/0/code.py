# Variant of 13e47133
# Dimension: D15 — 符号指令：调色板构建的特殊规则 (symbol_instructions)
# Change: 无显式符号编码；标记像素的距离隐式决定调色板顺序，距离 > 0 时背景色自动前置... → 网格边缘的图例行定义调色板

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the D15 variant
# -----------------------------------
# a) Concrete behavior and trace:
#    The original task stored palette order inside each region by placing a few
#    marker pixels at boundary distances 0/1/2. This variant removes those
#    in-region markers completely. Instead, row 0 is a legend row. The body
#    below row 0 is still partitioned by wall-colored vertical lines into 3-6
#    rectangular regions. For each region, the legend cells directly above that
#    region repeat the intended palette cyclically, e.g. legend segment
#    [3, 5, 1, 3, 5] encodes palette [3, 5, 1]. The output keeps the legend
#    unchanged and fills the region with palette[distance % len(palette)].
#    Small trace:
#    - Original: region has base 8 and markers 7 at d=0, 4 at d=1, so palette
#      becomes [7, 4]. Output alternates 7/4 inward.
#    - Variant: region interior contains only 8, while the top legend segment is
#      7 4 7 4 7. Solver recovers palette [7, 4] from the repeated legend and
#      produces the same style of concentric bands.
#    - Difference: palette source moved from implicit in-region distance coding
#      to explicit boundary legend coding.
# b) Complexity calibration:
#    To make the legend-to-region mapping obvious to humans, the body is limited
#    to full-height vertical strips. This is the one compatibility adjustment:
#    every region touches the top legend row, so the corresponding legend segment
#    is spatially aligned and visually attributable. Palette lengths stay in
#    1..3 so the repeated pattern is easy to see and decode from a single row.
# c) Old assumptions that no longer hold:
#    - Inferring bg/wall from the entire grid is no longer safe because the top
#      legend row injects several extra colors. The solver therefore infers them
#      from the body only.
#    - Scanning each component for non-base marker pixels is invalid because the
#      regions now contain no markers at all.
#    - Auto-prepending the base color when the first marker depth is > 0 is also
#      invalid; if bg should appear in the palette, it must appear explicitly in
#      the legend sequence.
# d) Placement legality:
#    The generator chooses region widths first, inserts wall columns between
#    strips, and paints row 0 using the same strip boundaries. This guarantees
#    no overlap between legend segments and no ambiguity about which legend cells
#    belong to which region. Each region width is at least 3 so its legend shows
#    a complete palette period.
# e) Visual observability:
#    A solver can no longer rely on the presence or depth of special pixels
#    inside a region; the only visible cue is the repeated color order on row 0.
#    Because the legend sits directly above each strip, 2-3 examples are enough
#    to infer that row 0 defines the palette and the strip below uses it.
# f) Visual cleanliness:
#    The body contains only base-colored cells and straight wall columns. The
#    only symbolic overlay is the top row, so there is no clutter from mixing
#    markers with the region texture itself.
# g) Color inference:
#    Palette colors are used structurally: the same ordered tuple appears on the
#    legend and reappears as concentric bands in the aligned strip below. The
#    generator limits palette length and keeps colors distinct inside a palette
#    so the minimal period is recoverable without relying on numeric color ids.

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns dist array with -1 outside component.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    is_boundary = True
                    break
                if not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _recover_palette(seq: list[int]) -> list[int]:
    """Return the shortest repeating prefix that generates seq."""
    if not seq:
        raise ValueError("empty legend segment")
    for k in range(1, len(seq) + 1):
        ok = True
        for i, v in enumerate(seq):
            if int(v) != int(seq[i % k]):
                ok = False
                break
        if ok:
            return [int(v) for v in seq[:k]]
    return [int(v) for v in seq]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned regions with concentric colored bands driven by a legend row.

    Input structure:
    - The top row is a legend strip, and the rows below it are partitioned into regions by wall-colored columns.
    - Every region interior contains only its base color; there are no marker pixels inside the regions.
    - The legend cells directly above a region repeat that region's palette order.

    Rule:
    - For each non-wall region in the body, read the aligned legend segment above it and recover the shortest repeating color cycle.
    - Use that recovered cycle as the region palette exactly as written; the base color is not inserted automatically.
    - Paint each region cell with palette[chebyshev_dist_from_boundary % palette_length], leaving the legend row and walls unchanged.

    Key insight: The symbolic instruction has moved from hidden marker depths to an explicit edge legend — decode the periodic color order first, then apply the same distance-based filling rule as before.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h < 2:
        return g.copy()

    legend = g[0]
    body = g[1:]
    bh, bw = body.shape

    # New assumption: bg/wall must be inferred from the body only, because the
    # legend row may contain several extra palette colors.
    bg = _mode(body)
    other = body[body != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = body != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((bh, bw), dtype=bool)
        cols = []
        for r, c in pts:
            comp_mask[r, c] = True
            cols.append(int(c))

        c0, c1 = min(cols), max(cols)
        legend_seq = [int(legend[c]) for c in range(c0, c1 + 1)]
        palette = _recover_palette(legend_seq)
        dist = _boundary_distance(comp_mask)
        m = len(palette)

        for r, c in pts:
            d = int(dist[r, c])
            out[r + 1, c] = int(palette[d % m])

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MIN_REGION_W = 3
MAX_PALETTE_LEN = 3

ALT_COLORS_MIN, ALT_COLORS_MAX = 2, 6
BG_IN_PALETTE_PROB = 0.5

GEN_MAX_ATTEMPTS = 300


def _choose_colors(rng: random.Random) -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    alt_n = rng.randint(ALT_COLORS_MIN, ALT_COLORS_MAX)
    alt_pool = rng.sample(colors, alt_n)
    return bg, wall, alt_pool


def _choose_region_count(W: int, rng: random.Random) -> int | None:
    feasible = []
    for n in range(N_REGIONS_MIN, N_REGIONS_MAX + 1):
        if W >= n * MIN_REGION_W + (n - 1):
            feasible.append(n)
    if not feasible:
        return None
    return rng.choice(feasible)


def _partition_widths(total: int, n: int, minimum: int, rng: random.Random) -> list[int]:
    widths = [minimum] * n
    extra = total - minimum * n
    for _ in range(extra):
        widths[rng.randrange(n)] += 1
    return widths


def _make_palette(width: int, bg: int, alt_pool: list[int], rng: random.Random) -> list[int]:
    allow_bg = rng.random() < BG_IN_PALETTE_PROB
    candidates = list(alt_pool)
    if allow_bg:
        candidates = [bg] + candidates

    max_len = min(MAX_PALETTE_LEN, width, len(candidates))
    k = rng.randint(1, max_len)

    palette = rng.sample(candidates, k)
    if len(set(palette)) != len(palette):
        raise ValueError("palette colors must be distinct")
    return [int(v) for v in palette]


def _build_grid(H: int, W: int, bg: int, wall: int, widths: list[int], palettes: list[list[int]]) -> np.ndarray:
    grid = np.full((H, W), int(bg), dtype=int)
    row = np.full(W, int(wall), dtype=int)

    c = 0
    for i, (width, palette) in enumerate(zip(widths, palettes)):
        for j in range(width):
            row[c + j] = int(palette[j % len(palette)])
        c += width
        if i != len(widths) - 1:
            grid[1:, c] = int(wall)
            c += 1

    grid[0] = row
    return grid


def _variant_metrics(grid: np.ndarray) -> tuple[int, int, int]:
    """Return (#bg-using regions, #nontrivial regions, #regions with non-leading bg)."""
    g = np.asarray(grid, dtype=int)
    legend = g[0]
    body = g[1:]
    bg = _mode(body)
    wall = _mode(body[body != bg])
    comps = _cc4(body != wall)

    explicit_bg = 0
    nontrivial = 0
    bg_nonleading = 0
    for pts in comps:
        cols = [int(c) for _r, c in pts]
        palette = _recover_palette([int(legend[c]) for c in range(min(cols), max(cols) + 1)])
        if bg in palette:
            explicit_bg += 1
            if palette.index(bg) > 0:
                bg_nonleading += 1
        if palette != [bg]:
            nontrivial += 1
    return explicit_bg, nontrivial, bg_nonleading


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    n_regions = _choose_region_count(W, rng)
    if n_regions is None:
        return None

    bg, wall, alt_pool = _choose_colors(rng)
    free_cols = W - (n_regions - 1)
    widths = _partition_widths(free_cols, n_regions, MIN_REGION_W, rng)
    palettes = [_make_palette(width, bg, alt_pool, rng) for width in widths]

    grid = _build_grid(H, W, bg, wall, widths, palettes)

    # Reject degenerate cases where the legend has no effect.
    explicit_bg, nontrivial, bg_nonleading = _variant_metrics(grid)
    if nontrivial == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if explicit_bg == 0:
        return None
    if bg_nonleading == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
