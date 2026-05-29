# Variant of 13e47133
# Dimension: D3 — 源对象：标记像素的定义与放置 (src)
# Change: 每个区域内 0-3 个孤立彩色像素，位于区域角落对角线上，按切比雪夫距离递增放置... → 控制像素放置在网格边缘

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the D3 variant
# ----------------------------------
# concretization:
#   grid geometry stays close to the original task: full-row/full-column wall splits
#   partition the canvas into 3-6 rectangular free regions. The output rule also
#   stays the same: every region is repainted by repeating a palette over its
#   Chebyshev distance-from-boundary layers.
#
#   The changed object is the marker source. Instead of placing marker colors on an
#   inward diagonal inside a region, each region may place 0-3 control pixels on
#   exactly one outer grid edge touched by that region. The control pixels occupy
#   consecutive positions along that edge segment. Their offsets along the edge
#   encode the same distance indices that used to be read from interior diagonal
#   positions:
#     - top/bottom side: distance = column offset from the region's leftmost cell
#       on that side
#     - left/right side: distance = row offset from the region's topmost cell
#       on that side
#
#   Example trace on a top-edge rectangle with bbox cols 4..9:
#     - original: markers at (0,4), (1,5), (2,6) would encode d=0,1,2
#     - variant: controls at (0,4), (0,5), (0,6) encode the same d=0,1,2
#     - old solver would misread all three as boundary-depth 0, so only the first
#       color survives; the new solver must instead recover the ordered offsets
#       from the outer edge segment and then apply the original concentric fill.
#
# complexity calibration:
#   - only one hosting side per region keeps the mapping visible to a human
#   - the useful control degrees of freedom remain small and repeated across a
#     sample set: marker count in {0,1,2,3}, start offset in {0,1}
#   - regions stay rectangular so the "offset on the edge" visibly matches the
#     original "distance from a corner diagonal" idea rather than introducing a
#     second unrelated spatial puzzle
#
# implicit assumption audit against the original solver:
#   - original line "markers are cells whose color differs from the region base"
#     no longer implies "their region depth is dist[r,c]"; border controls all lie
#     on depth 0 cells, so using their cell depth would collapse the palette
#   - original "pick at most one marker color per distance" still applies, but the
#     distance must now be reconstructed from outer-edge offsets instead of the
#     control cell's own distance-to-boundary
#   - original generator assumed marker cells could be stamped anywhere inside a
#     region as long as they stayed within the rectangle; the variant generator must
#     additionally guarantee that controls lie on an outer border segment of the
#     same region and that singleton controls on corner-sharing regions are not
#     ambiguous about which side they belong to
#
# placement legality and solver assumptions:
#   - edge-touching regions may carry controls; fully internal regions are forced
#     into the zero-control branch and therefore stay base-colored in the output
#   - at most one side per region carries controls
#   - if a region touches multiple outer sides and has exactly one control, that
#     control is kept away from corners so the hosting side is unambiguous
#   - the solver therefore assumes that any non-base border controls for a region
#     either all lie on one border row/col, or only a single d=0 corner control is
#     present (where either matching side yields the same palette)
#
# observability / cleanliness:
#   - the only colored clues outside the final pattern are edge controls, so the D3
#     change is immediately visible in the input
#   - walls and region interiors remain uncluttered; the output still shows clean,
#     large concentric bands
#   - color ids are never used for ordering; only visible spatial offsets matter


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SIDE_PRIORITY = ("top", "bottom", "left", "right")


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


def _edge_controls_for_component(g: np.ndarray, comp_mask: np.ndarray, pts: np.ndarray, base: int):
    """
    Read border control pixels for one region.

    Controls are non-base cells that lie on the outer canvas border. Their palette
    index is reconstructed from the offset along the chosen outer side, not from
    the control cell's own boundary depth.
    """
    h, w = g.shape
    side_cells = {side: [] for side in SIDE_PRIORITY}
    side_controls = {side: [] for side in SIDE_PRIORITY}

    for r, c in pts:
        r = int(r)
        c = int(c)
        if r == 0:
            side_cells["top"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["top"].append((r, c))
        if r == h - 1:
            side_cells["bottom"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["bottom"].append((r, c))
        if c == 0:
            side_cells["left"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["left"].append((r, c))
        if c == w - 1:
            side_cells["right"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["right"].append((r, c))

    candidates = [(side, side_controls[side]) for side in SIDE_PRIORITY if side_controls[side]]
    if not candidates:
        return {}

    best_len = max(len(cells) for _side, cells in candidates)
    chosen_side = next(side for side, cells in candidates if len(cells) == best_len)
    controls = side_controls[chosen_side]

    if chosen_side in ("top", "bottom"):
        span_start = min(c for _r, c in side_cells[chosen_side])
        ordered = sorted((int(c - span_start), int(g[r, c])) for r, c in controls)
    else:
        span_start = min(r for r, _c in side_cells[chosen_side])
        ordered = sorted((int(r - span_start), int(g[r, c])) for r, c in controls)

    dist_to_color = {}
    for d, col in ordered:
        if d not in dist_to_color:
            dist_to_color[d] = int(col)
    return dist_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions using palettes encoded by edge control pixels.

    Input structure:
    - The grid is split into free regions by a wall color.
    - Region interiors are mostly a single base color.
    - Any non-base clues for a region live on one outer grid edge touched by that
      region, not on an inward diagonal inside the region.

    Rule:
    - For each region, read its border control pixels from the canvas edge and sort
      them by their offset along that edge segment to build the palette.
    - If the smallest encoded offset is greater than 0, prepend the base color.
    - Repaint each region cell with palette[chebyshev_distance_to_region_boundary mod len(palette)].

    Key insight: A control pixel's own cell depth is no longer meaningful because
    all controls sit on the global border; the relevant distance is the visible
    offset along the region's outer edge segment.
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
            comp_mask[int(r), int(c)] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        dist_to_color = _edge_controls_for_component(g, comp_mask, pts, base)

        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            palette = []
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            out[int(r), int(c)] = palette[int(dist[int(r), int(c)]) % m]

    return out


# --------------- generator ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Controls per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 4, 4]
START_OFFSET_MAX = 1

# ---- Distinct control colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 300


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


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _outer_sides(rect, H: int, W: int):
    r0, c0, r1, c1 = rect
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == H - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == W - 1:
        sides.append("right")
    return sides


def _side_len(rect, side: str):
    r0, c0, r1, c1 = rect
    if side in ("top", "bottom"):
        return c1 - c0 + 1
    return r1 - r0 + 1


def _control_coord(rect, side: str, offset: int, H: int, W: int):
    r0, c0, r1, c1 = rect
    if side == "top":
        return 0, c0 + offset
    if side == "bottom":
        return H - 1, c0 + offset
    if side == "left":
        return r0 + offset, 0
    return r0 + offset, W - 1


def _place_controls(rect, H: int, W: int, marker_pool: list[int], rng: random.Random):
    sides = _outer_sides(rect, H, W)
    if not sides:
        return [], {"count": 0, "start": None, "side": "none"}
    side = rng.choice(sides)
    side_len = _side_len(rect, side)
    upper = min(MARKERS_PER_REGION_MAX, side_len)
    if upper <= 0:
        return [], {"count": 0, "start": None, "side": side}

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return [], {"count": 0, "start": None, "side": side}

    multi_side = len(sides) > 1
    max_start = min(START_OFFSET_MAX, side_len - n)

    if n == 1 and multi_side:
        interior_starts = [d for d in range(max_start + 1) if 0 < d < side_len - 1]
        if interior_starts:
            start = rng.choice(interior_starts)
        else:
            start = rng.randint(0, max_start)
    else:
        start = rng.randint(0, max_start)

    colors = rng.sample(marker_pool, n)
    placed = []
    for k, col in enumerate(colors):
        d = start + k
        rr, cc = _control_coord(rect, side, d, H, W)
        placed.append((d, int(col), int(rr), int(cc), side))
    return placed, {"count": n, "start": start, "side": side}


def _stamp_controls(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    advanced_regions = 0
    side_counts = {side: 0 for side in SIDE_PRIORITY + ("none",)}
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    start_counts = {0: 0, 1: 0}

    H, W = grid.shape
    for rect in rects:
        controls, meta = _place_controls(rect, H, W, marker_pool, rng)
        n = int(meta["count"])
        counts[n] += 1
        if meta["start"] is not None:
            start_counts[int(meta["start"])] += 1
        side_counts[str(meta["side"])] += 1
        if n > 0 and (n > 1 or int(meta["start"]) > 0):
            advanced_regions += 1

        for _d, col, rr, cc, _side in controls:
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)

    return {
        "used_colors": used,
        "advanced_regions": advanced_regions,
        "count_hist": counts,
        "start_hist": start_counts,
        "side_hist": side_counts,
    }


def _validate_color_order(grid, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    meta = _stamp_controls(grid, rects, bg, marker_pool, rng)
    if meta is None:
        return None
    used = meta["used_colors"]
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if meta["advanced_regions"] == 0:
        return None
    if not _validate_color_order(grid, bg, wall):
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
