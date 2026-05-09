# Variant of 13e47133
# Dimension: D6 — 网格形状：输出的裁剪方式 (grid_shape)
# Change: 输出与输入网格尺寸完全相同... → 每个区域单独输出并拼接

import random

import numpy as np

from arc_utils import components_mask, paste

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# Design analysis for the D6 variant:
# - Concrete behavior:
#   Keep the original within-region logic exactly where it matters: markers are still
#   read by Chebyshev depth from the region boundary, and the region is still filled
#   with the resulting cyclic palette. The change happens after that fill. Instead of
#   putting the filled cells back into the original wall-partitioned canvas, each
#   non-wall region becomes its own cropped tile and the tiles are reassembled.
#   Small example: if the input has top-left region A, top-right region B, and a
#   bottom region C, the original solver would output A and B side by side above C.
#   This variant fills A/B/C the same way, then outputs tile(A), separator, tile(B),
#   separator, tile(C) as three stacked panels.
# - Complexity calibration:
#   A pure horizontal-only or vertical-only partition would make the packed output too
#   close to the old layout. The generator therefore uses 2-3 horizontal bands and
#   forces at least one band to be split vertically, creating 3-5 regions whose
#   reading-order stack is visibly different from the input arrangement.
# - Hidden-assumption audit:
#   The old code assumes "write filled values back into a copy of the input grid".
#   That is no longer valid because the output shape changes. The updated solver still
#   infers bg/wall colors the same way, still finds 4-connected non-wall components,
#   and still builds the same palettes, but now it assumes components can be ordered
#   by their top-left coordinate and packed into a new wall-colored canvas. The
#   generator guarantees that these packed dimensions stay within 30x30.
# - Placement legality:
#   Input regions are rectangular components separated by full horizontal wall rows
#   and optional vertical wall columns within each band. Every region has height/width
#   at least 3, so marker diagonals fit; marker placement is checked before stamping.
#   Output tiles are packed top-to-bottom with one full-width wall row between them.
# - Visual observability / cleanliness:
#   No color-id arithmetic is used. The visible cue is structural: side-by-side
#   regions in the input become separate stacked tiles in the output, while each tile
#   still shows the same concentric-frame rule as the source task. The wall color is
#   reused as the separator color so the split-and-reassemble operation reads cleanly.


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


def _component_key(pts: np.ndarray) -> tuple[int, int]:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min())


def _filled_region_tile(g: np.ndarray, pts: np.ndarray, wall: int) -> np.ndarray:
    h, w = g.shape
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        comp_mask[int(r), int(c)] = True

    vals = g[comp_mask]
    base = _mode(vals)
    dist = _boundary_distance(comp_mask)

    dist_to_color: dict[int, int] = {}
    for r, c in sorted((int(r), int(c)) for r, c in pts):
        v = int(g[r, c])
        if v == base:
            continue
        d = int(dist[r, c])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = v

    if dist_to_color:
        ds = sorted(dist_to_color.keys())
        palette = [int(base)] if ds[0] > 0 else []
        palette.extend(int(dist_to_color[d]) for d in ds)
    else:
        palette = [int(base)]

    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), wall, dtype=int)

    m = len(palette)
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        tile[rr - r0, cc - c0] = palette[int(dist[rr, cc]) % m]
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated regions with cyclic boundary-distance palettes, then separate them into a new packed layout.

    Input structure:
    - The input is divided by one wall color into several non-wall regions.
    - Each region has one dominant base color plus a few marker pixels placed at specific Chebyshev depths from that region's boundary.

    Rule:
    - For each region, sort marker colors by boundary distance and optionally prepend the base color when the first marker is not on the boundary.
    - Fill the region with palette[distance mod palette_length], exactly as in the source task.
    - After filling, crop each region to its own bounding box and stack those region tiles from left-to-right, top-to-bottom reading order, using full wall-colored separator rows between tiles.

    Key insight:
    - The hard part is still local palette recovery inside each region; the D6 change only alters the final canvas assembly, so the solver must split regions, solve them independently, and then rebuild a different-shaped output grid.
    """
    g = np.asarray(grid, dtype=int)

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    comps = sorted(_cc4(g != wall), key=_component_key)
    if not comps:
        return g.copy()

    tiles = [_filled_region_tile(g, pts, wall) for pts in comps]
    out_h = sum(tile.shape[0] for tile in tiles) + (len(tiles) - 1)
    out_w = max(tile.shape[1] for tile in tiles)
    out = np.full((out_h, out_w), wall, dtype=int)

    r = 0
    for tile in tiles:
        paste(out, tile, r, 0)
        r += tile.shape[0] + 1
    return out


# --------------- generator ---------------

# Keep the physical canvas comfortably below ARC's 30x30 cap so the repacked output
# also fits after separators are inserted.
H_MIN, H_MAX = 10, 18
W_MIN, W_MAX = 13, 22
OUT_MAX = 30

MIN_REGION_DIM = 3
N_BANDS_MIN, N_BANDS_MAX = 2, 3
N_REGIONS_MIN, N_REGIONS_MAX = 3, 5

MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
GEN_MAX_ATTEMPTS = 500

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _random_parts(total: int, n: int, min_part: int, rng: random.Random) -> list[int] | None:
    if total < n * min_part:
        return None
    parts = [min_part] * n
    extra = total - n * min_part
    for _ in range(extra):
        parts[rng.randrange(n)] += 1
    rng.shuffle(parts)
    return parts


def _column_options(width: int) -> list[int]:
    opts = []
    for k in (1, 2, 3):
        if width >= k * MIN_REGION_DIM + (k - 1):
            opts.append(k)
    return opts


def _build_partition(H: int, W: int, rng: random.Random):
    free_rows_min = N_BANDS_MIN * MIN_REGION_DIM + (N_BANDS_MIN - 1)
    if H < free_rows_min:
        return None

    for _ in range(200):
        n_bands = rng.randint(N_BANDS_MIN, N_BANDS_MAX)
        band_heights = _random_parts(H - (n_bands - 1), n_bands, MIN_REGION_DIM, rng)
        if band_heights is None:
            continue

        band_cols = []
        band_widths = []
        multi_band_count = 0
        for _bh in band_heights:
            opts = _column_options(W)
            if not opts:
                return None
            weights = [4 if k == 1 else 3 if k == 2 else 1 for k in opts]
            k = rng.choices(opts, weights=weights)[0]
            widths = _random_parts(W - (k - 1), k, MIN_REGION_DIM, rng)
            if widths is None:
                break
            band_cols.append(k)
            band_widths.append(widths)
            multi_band_count += int(k > 1)
        if len(band_cols) != n_bands:
            continue

        if multi_band_count == 0:
            idxs = [i for i in range(n_bands) if max(_column_options(W)) > 1]
            if not idxs:
                continue
            i = rng.choice(idxs)
            opts = [k for k in _column_options(W) if k > 1]
            k = rng.choice(opts)
            widths = _random_parts(W - (k - 1), k, MIN_REGION_DIM, rng)
            if widths is None:
                continue
            band_cols[i] = k
            band_widths[i] = widths

        n_regions = sum(band_cols)
        if not (N_REGIONS_MIN <= n_regions <= N_REGIONS_MAX):
            continue

        out_h = sum(bh * k for bh, k in zip(band_heights, band_cols)) + (n_regions - 1)
        out_w = max(sum(ws) + (len(ws) - 1) for ws in band_widths)
        if out_h > OUT_MAX or out_w > OUT_MAX:
            continue

        wall_mask = np.zeros((H, W), dtype=int)
        rects: list[tuple[int, int, int, int]] = []

        r = 0
        for band_idx, bh in enumerate(band_heights):
            r0, r1 = r, r + bh - 1
            widths = band_widths[band_idx]
            c = 0
            for j, ww in enumerate(widths):
                c0, c1 = c, c + ww - 1
                rects.append((r0, c0, r1, c1))
                c = c1 + 1
                if j != len(widths) - 1:
                    wall_mask[r0 : r1 + 1, c] = 1
                    c += 1
            r = r1 + 1
            if band_idx != len(band_heights) - 1:
                wall_mask[r, :] = 1
                r += 1

        return wall_mask, rects, band_cols, band_heights
    return None


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    start_d = rng.randint(0, min(START_DIST_MAX, max_d - n))
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    colors = rng.sample(marker_pool, n)
    return [
        (start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k))
        for k, col in enumerate(colors)
    ]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    markers_per_region = []
    for rect in rects:
        placed = []
        for d, col, rr, cc in _place_markers(rect, marker_pool, rng):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(col)
            placed.append((d, col, rr, cc))
        markers_per_region.append(placed)
    return used, markers_per_region


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    built = _build_partition(H, W, rng)
    if built is None:
        return None

    wall_mask, rects, band_cols, band_heights = built
    bg, wall, marker_pool = _pick_palette(rng)

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, markers_per_region = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not any(len(marks) > 0 for marks in markers_per_region):
        return None
    if not any(k > 1 for k in band_cols):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    # D6 should be visible: the canvas shape changes and at least one band was split,
    # so side-by-side input regions become separate stacked output tiles.
    if out.shape == grid.shape:
        return None
    if out.shape[0] > OUT_MAX or out.shape[1] > OUT_MAX:
        return None

    # Reject nearly trivial all-base outputs.
    if len(set(map(int, out.ravel()))) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
