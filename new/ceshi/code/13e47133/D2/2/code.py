# Variant of 13e47133
# Dimension: D2 — 轨迹：颜色交替模式 (trajectory)
# Change: 调色板按距离循环：palette[dist % len]，标记颜色与背景色交替... → 距离约束式间断轨迹

import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask


# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original_behavior: >
#       Within each non-wall region, markers sorted by boundary distance formed a cyclic
#       palette, and every layer used palette[distance % len(palette)].
#     new_behavior: >
#       Marker colors no longer repeat. A marker now licenses exactly one concentric layer:
#       the layer at that same Chebyshev distance, and only if that distance is even.
#       Odd-distance layers stay at the region base color, creating visible gaps.
#   a_concretization:
#     input_view: >
#       The input still shows wall-partitioned regions with a dominant base color and a few
#       isolated marker pixels. The generator now places markers only on even distances
#       such as d=0,2,4 along a corner diagonal.
#     output_view: >
#       Each marker expands to a full rectangular ring at its own depth. The ring between
#       two colored rings is left base-colored, so a d=0 and d=2 pair becomes:
#       outer ring = marker A, d=1 ring = base, d=2 ring = marker B.
#     trace_example: >
#       Example region with markers at d=0 color 7 and d=2 color 3.
#       Old rule with palette [7,3] would paint depths 0,2,4,... as 7 and depths 1,3,5,...
#       as 3. New rule paints only depth 0 as 7 and depth 2 as 3; depth 1 and every other
#       unclaimed layer remain base. The difference is both visual sparsity and the fact
#       that a marker claims a single layer instead of starting a periodic sequence.
#   b_complexity_calibration:
#     choices:
#       - Keep the wall layout and component extraction unchanged so the puzzle still reads
#         as "regions separated by walls".
#       - Restrict marker distances to even layers only; this avoids hidden odd-distance
#         markers that would never manifest in the output.
#       - Use only 0-3 markers per region and 3-9 distinct marker colors grid-wide, which
#         matches the original task's scale and keeps the color logic learnable.
#     anti_simplification: >
#       Samples are rejected unless at least one marker lies at depth >= 2, so the skip
#       pattern is visible as a colored ring separated by a base-colored gap.
#   c_implicit_assumption_review:
#     - bg = mode(grid) and wall = second mode still hold because the generator preserves a
#       large shared base color and sparse markers.
#     - components of non-wall cells are still the correct regions because D2 does not alter
#       partition topology.
#     - base = mode(region_values) remains valid because marker pixels stay isolated and few.
#     - dist_to_color still assumes at most one meaningful marker per distance per region.
#       The generator guarantees unique distances inside each region.
#     - The old write `out[r,c] = palette[d % m]` depended on periodic repetition. That
#       premise is removed; the new solver must test parity and exact distance membership.
#   d_placement_legality:
#     - Recursive full-row/full-column wall splits still produce rectangular free regions.
#     - A marker at bbox-corner offset d lies inside the rectangle whenever d <= max depth.
#     - Markers are placed on distinct cells because each chosen distance is unique.
#     - No marker overwrites a wall or another marker; the generator validates both.
#   e_visual_observability:
#     - Even-depth-only placement is visible from the input because markers sit every other
#       inward step on a corner diagonal.
#     - The output shows alternating colored/base rings, so humans can infer "skip one
#       layer" from 2-3 examples without needing color-id arithmetic.
#   f_visual_cleanliness:
#     - Regions do not overlap because walls partition the canvas.
#     - Each region uses only a few markers, avoiding dense noise.
#     - Colored rings remain axis-aligned and easy to parse.
#   g_color_inference:
#     - Marker colors are copied directly to their rings; there is no hidden color mapping.
#     - Distances are inferred from visible position, not from numeric color ids.
#   main_to_generator_contract:
#     - main assumes markers that matter are on even distances and that each region has at
#       most one marker per distance. generate_input enforces both.
#     - main treats odd-distance markers as inert; generate_input never emits them.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with sparse concentric rings keyed by marker depth.

    Input structure:
    - The grid is split into regions by a wall color.
    - Inside each region, one base color dominates and a few isolated marker pixels sit at chosen depths from the region boundary.

    Rule:
    - For each region, read marker colors ordered by their Chebyshev distance from the boundary, but do not build a repeating palette.
    - A marker colors only the full layer at its own distance, and only when that distance is even; odd layers and unmarked even layers stay at the region base color.

    Key insight: A marker is now a one-time claim on a specific concentric ring, so the output shows colored frames separated by untouched base-colored gaps.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d < 0 or d % 2 == 1:
                continue
            if d not in dist_to_color:
                dist_to_color[d] = v

        for r, c in pts:
            d = int(dist[r, c])
            if d >= 0 and d % 2 == 0 and d in dist_to_color:
                out[r, c] = int(dist_to_color[d])
            else:
                out[r, c] = int(base)

    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 4, 4]
START_EVEN_CHOICES = (0, 2)

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0:c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0:r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _even_distances(max_d: int, start_d: int) -> list[int]:
    return [d for d in range(start_d, max_d + 1, 2) if d % 2 == 0]


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    """
    Produce 0..MARKERS_PER_REGION_MAX markers along a diagonal from one bbox corner.
    Markers only occupy even boundary distances so every emitted marker is visible
    under the new sparse-ring rule.
    """
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    if max_d < 0:
        return []

    viable_starts = [d for d in START_EVEN_CHOICES if d <= max_d]
    if not viable_starts:
        return []
    start_d = rng.choice(viable_starts)

    dists = _even_distances(max_d, start_d)
    upper = min(MARKERS_PER_REGION_MAX, len(dists))
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    chosen_dists = dists[:n]
    colors = rng.sample(marker_pool, n)
    return [(d, col, br + dr * d, bc + dc * d) for d, col in zip(chosen_dists, colors)]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    placements = []
    for rect in rects:
        region_markers = _place_markers(rect, marker_pool, rng)
        for d, col, rr, cc in region_markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)
            placements.append((rect, d, col, rr, cc))
    return used, placements


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray, out: np.ndarray):
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = grid != wall

    colored_even_layers = 0
    deeper_even_layers = 0
    differing_cells_vs_input = int(np.count_nonzero(grid != out))
    differing_cells_vs_legacy = 0

    legacy = _legacy_main(grid)
    differing_cells_vs_legacy = int(np.count_nonzero(legacy != out))

    for pts in _cc4(free):
        comp_mask = np.zeros(grid.shape, dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        base = _mode(grid[comp_mask])
        dist = _boundary_distance(comp_mask)
        seen_even = set()
        for r, c in pts:
            if int(grid[r, c]) == int(base):
                continue
            d = int(dist[r, c])
            if d >= 0 and d % 2 == 0 and d not in seen_even:
                seen_even.add(d)
                colored_even_layers += 1
                if d >= 2:
                    deeper_even_layers += 1

    return {
        "colored_even_layers": colored_even_layers,
        "deeper_even_layers": deeper_even_layers,
        "differing_cells_vs_input": differing_cells_vs_input,
        "differing_cells_vs_legacy": differing_cells_vs_legacy,
    }


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            if d not in dist_to_color:
                dist_to_color[d] = v

        palette: list[int] = []
        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            out[r, c] = palette[d % m]

    return out


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    stamped = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if stamped is None:
        return None
    used, placements = stamped

    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    metrics = _variant_metrics(grid, out)
    if metrics["colored_even_layers"] < 3:
        return None
    if metrics["deeper_even_layers"] < 1:
        return None
    if metrics["differing_cells_vs_input"] <= 0:
        return None
    if metrics["differing_cells_vs_legacy"] <= 0:
        return None

    if not placements:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
