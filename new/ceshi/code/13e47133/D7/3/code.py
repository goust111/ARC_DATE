# Variant of 13e47133
# Dimension: D7 — 区域分割方式 (obstacle)
# Change: 单一颜色的轴对齐墙壁线段将网格分割为 3-6 个矩形或 L/T 形区域，墙壁像素不变... → 无墙壁分割，区域由背景色变化界定

import random

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _repair_markers(grid: np.ndarray) -> np.ndarray:
    """
    Replace isolated marker pixels by the unique majority orthogonal-neighbor color.

    In this variant, each region is mostly one background color and marker pixels are
    isolated exceptions. Restoring those single-pixel exceptions yields the true region
    partition without relying on any explicit wall color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    repaired = g.copy()

    for r in range(h):
        for c in range(w):
            counts: dict[int, int] = {}
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    counts[v] = counts.get(v, 0) + 1
            if not counts:
                continue

            max_count = max(counts.values())
            if max_count < 2:
                continue
            winners = [color for color, cnt in counts.items() if cnt == max_count]
            if len(winners) != 1:
                continue

            fill = winners[0]
            if fill != int(g[r, c]):
                repaired[r, c] = fill

    return repaired


def _region_components(repaired: np.ndarray) -> list[np.ndarray]:
    comps: list[np.ndarray] = []
    for color in np.unique(repaired):
        comps.extend(_cc4(repaired == int(color)))
    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


# Design analysis for D7 (region partition by background-color changes)
# spec_variant:
#   input_shape:
#     - The whole canvas is tiled by 3-6 axis-aligned rectangles with no wall gaps.
#     - Adjacent regions touch directly edge-to-edge; boundaries are visible only because
#       neighboring regions use different dominant background colors.
#     - Inside each region, 0-3 isolated marker pixels sit on one diagonal from a corner,
#       at consecutive Chebyshev depths exactly as in the source task.
#   behavior_trace:
#     - Original branch: detect one global wall color -> regions are non-wall connected
#       components -> region base is usually the global background color.
#     - Variant branch: there is no wall color to exclude. Instead, a marker pixel is the
#       only local violation of an otherwise monochrome region, so we first repair such
#       single-pixel violations from orthogonal-neighbor majority color. After that repair,
#       same-color connected components are the regions. The later palette and distance
#       logic stays unchanged.
#     - Example: if a blue rectangle touches a yellow rectangle directly and contains one
#       red corner marker, the original solver would incorrectly treat "second most common
#       color" as a wall. The variant solver instead restores that red pixel to blue using
#       its two or more blue orthogonal neighbors, then recovers the full blue rectangle.
#   complexity_calibration:
#     - Region base colors are unique per grid so color-change boundaries are explicit and
#       human-visible rather than hidden behind repeated colors.
#     - Marker pixels remain single-cell anomalies. This keeps region inference difficult
#       enough to differ from the wall version, but still regular enough to learn from a
#       few examples.
#     - The number of free choices stays small: 3-6 regions, 0-3 markers per region, and
#       marker depths remain consecutive.
#   hidden_assumption_audit:
#     - The old `bg = _mode(g)` and `wall = second_mode` assumption is invalid because the
#       grid now has several large background colors and no separator color at all.
#     - The old `free = g != wall` assumption would merge unrelated regions or erase one
#       legitimate region if its color happened to be second-most-common.
#     - The palette phase still assumes each marker belongs to exactly one region. The
#       generator guarantees this by placing only isolated markers within rectangles of
#       minimum side length 3.
#   placement_legality:
#     - The generator recursively splits rectangles without inserting wall rows/cols, so
#       the final rectangles tile the whole canvas with no overlap and no holes.
#     - Each marker is placed on a valid corner-to-interior diagonal of its own rectangle,
#       never outside bounds and never on another marker.
#   observability:
#     - The defining D7 cue is direct adjacency between different region background colors.
#       That cue is absent in the source task, where a wall color separates regions.
#     - Some regions start markers at depth 0 while others can start at depth 1, so the
#       "prepend base color when the boundary marker is missing" subrule remains visible.
#   visual_cleanliness:
#     - Regions are plain rectangles, not jagged mosaics, so human solvers can see where
#       one zone ends and another begins before any repainting happens.
#     - Markers are sparse single pixels; they do not obscure the region shape.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill color-delimited regions with concentric bands derived from marker pixels.

    Input structure:
    - The grid has no wall color; instead, several directly touching regions are distinguished by different dominant background colors.
    - Each region is mostly monochrome, with a few isolated marker pixels of other colors placed at increasing depths from one region corner.

    Rule:
    - First restore each isolated marker to the surrounding majority color so the true region map becomes visible, then treat each monochrome component as one region.
    - Within a region, sort marker colors by Chebyshev distance from the region boundary; if the first marker is not on the boundary, prepend the region's background color, then repaint the region with palette[distance mod palette_length].

    Key insight: The obstacle disappeared, so the hard part is recovering the hidden region mask from color changes; once isolated markers are folded back into their host region, the original distance-driven cyclic palette rule applies unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    repaired = _repair_markers(g)
    out = g.copy()

    for pts in _region_components(repaired):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = repaired[comp_mask]
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

        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            palette = [int(base)] if ds[0] > 0 else []
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            out[r, c] = palette[int(dist[r, c]) % m]

    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Region subdivision (no walls) ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _split_regions(H: int, W: int, rng: random.Random):
    """
    Recursively split the canvas into rectangles that tile the full area.

    Returns a list of inclusive bboxes (r0, c0, r1, c1). No wall row/column is removed,
    so adjacent rectangles touch directly and are later distinguished only by base color.
    """
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
            rr = rng.randint(r0 + MIN_REGION_DIM - 1, r1 - MIN_REGION_DIM)
            rects.append((r0, c0, rr, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM - 1, c1 - MIN_REGION_DIM)
            rects.append((r0, c0, r1, cc))
            rects.append((r0, cc + 1, r1, c1))

    return rects


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    """
    Produce 0..MARKERS_PER_REGION_MAX markers along a diagonal from one bbox corner.
    Each marker = (dist, color, r, c) with consecutive Chebyshev distances.
    """
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


def _paint_regions(grid: np.ndarray, rects: list[tuple[int, int, int, int]], base_colors: list[int]):
    for color, (r0, c0, r1, c1) in zip(base_colors, rects):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _stamp_markers(grid, rects, marker_pool, rng):
    """
    Place markers into grid in place.

    Returns:
      used_colors: set[int]
      starts: list[int] of marker start distances per non-empty region
    or (None, None) if placement would violate the rectangle layout.
    """
    used = set()
    starts: list[int] = []
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        if markers:
            starts.append(markers[0][0])
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            region = grid[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1]
            base = _mode(region)
            if grid[rr, cc] != base:
                return None, None
            grid[rr, cc] = col
            used.add(col)
    return used, starts


def _count_touching_base_edges(grid: np.ndarray) -> int:
    repaired = _repair_markers(grid)
    h, w = repaired.shape
    edges = 0
    for r in range(h):
        for c in range(w):
            if r + 1 < h and repaired[r, c] != repaired[r + 1, c]:
                edges += 1
            if c + 1 < w and repaired[r, c] != repaired[r, c + 1]:
                edges += 1
    return edges


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    rects = _split_regions(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    colors = list(range(10))
    rng.shuffle(colors)
    if len(colors) - len(rects) < DISTINCT_MARKER_COLORS_MIN:
        return None
    base_colors = colors[: len(rects)]
    marker_pool = colors[len(rects) :]
    if len(marker_pool) > DISTINCT_MARKER_COLORS_MAX:
        marker_pool = marker_pool[:DISTINCT_MARKER_COLORS_MAX]

    grid = np.zeros((H, W), dtype=int)
    _paint_regions(grid, rects, base_colors)

    used, starts = _stamp_markers(grid, rects, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if starts and 0 not in starts and all(_max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1) >= 2 for r0, c0, r1, c1 in rects):
        return None
    if _count_touching_base_edges(grid) == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
