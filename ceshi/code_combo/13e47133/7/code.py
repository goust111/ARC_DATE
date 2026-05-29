# Combo variant of 13e47133 (index 7)
# combo_spec: D1/1+D10/2
# Name: 从中心向外扩展+基于标记距离

import random

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


# Design analysis for the combo D1/1 + D10/2 variant
# - Concrete behavior / trace:
#   Old task: markers near a region corner were sorted by boundary depth, then the
#   whole region was painted by boundary shell number. Example: if a rectangular
#   region had marker colors A/B/C at boundary depths 0/1/2, output rings were
#   palette[boundary_dist % 3], so the outer border was A and shells moved inward.
#   New combo: each marked region has a center marker at radius 0 plus optional
#   markers on consecutive radii 1, 2 along one center-to-corner diagonal. The
#   palette is read by distance from that center marker, then every cell uses its
#   Chebyshev radius from the center marker. With A at the center, B one diagonal
#   step out, C two steps out, the output is A on the center cell, B on the first
#   square ring, C on the next ring, then A/B/C repeat farther outward. The same
#   colors now radiate away from the center instead of collapsing from the walls.
# - Complexity calibration:
#   Using a single center marker only would often collapse a region to one color
#   and make the combo visually weak. Using a nearest-marker Voronoi rule would
#   drift toward the standalone D10 variant and lose the clean "one center, many
#   rings" geometry needed for D1. The chosen design keeps 0/2/3 markers per
#   region, requires consecutive radii from one anchor, and preserves a small,
#   repeated palette that is visible from a few examples.
# - Implicit assumptions from the original main():
#   The original `dist = _boundary_distance(comp_mask)` assumed one boundary-based
#   scalar field per region. That assumption is false here; distance is now tied
#   to a center marker. The original `dist_to_color[d]` also assumed markers were
#   indexed by boundary depth. In the combo, markers are indexed by Chebyshev
#   radius from the anchor marker, so the solver must first identify the center
#   anchor and only then read the palette.
# - Placement legality:
#   The generator keeps the wall partitioning rectangular via recursive splits.
#   For every marked region it places the anchor on the integer bbox center and
#   only extends the marker chain along a diagonal that has enough in-bounds room
#   for 1 or 2 outward steps. This guarantees no overlap, no wall contact, and no
#   marker outside its own region.
# - Visual observability:
#   Markers are moved away from boundaries and concentrated near region centers.
#   That makes the contrast with the original task obvious: the output rings are
#   centered where the markers are, not aligned to the enclosing box boundary.
#   The generator also rejects samples where this combo matches either the original
#   boundary solver or the standalone nearest-marker D10 variant.
# - Cleanliness:
#   Each region still contains only one base color plus a short center-outward
#   marker chain, so there is only one plausible anchor and one visible palette.
#   Walls remain a separate uniform color, keeping regions easy to parse.
# - New solver assumptions and generator support:
#   The solver assumes marked regions have exactly one marker nearest the bbox
#   center and that remaining markers lie on consecutive radii from it. The
#   generator enforces that construction directly, so palette order and ring radii
#   are always recoverable from visible geometry rather than hidden color ids.


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _bbox_center(pts: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [int(r) for r, _c in pts]
    cols = [int(c) for _r, c in pts]
    return ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Original boundary field kept only for generator-side regression checks."""
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


def _nearest_marker_index_and_distance(
    r: int, c: int, markers: list[tuple[int, int, int]]
) -> tuple[int, int]:
    best_idx = 0
    best_dist = None
    for idx, (mr, mc, _color) in enumerate(markers):
        d = max(abs(r - mr), abs(c - mc))
        if best_dist is None or d < best_dist or (d == best_dist and idx < best_idx):
            best_idx = idx
            best_dist = d
    return best_idx, int(best_dist)


def _anchor_and_palette(
    pts: list[tuple[int, int]], base: int, grid: np.ndarray
) -> tuple[tuple[int, int] | None, list[int]]:
    center = _bbox_center(pts)
    markers: list[tuple[int, int, int]] = []
    for r, c in pts:
        color = int(grid[r, c])
        if color != base:
            markers.append((int(r), int(c), color))

    if not markers:
        return None, [int(base)]

    markers.sort(key=lambda item: (item[0], item[1]))
    anchor_r, anchor_c, _anchor_color = min(
        markers, key=lambda item: (_cheb((item[0], item[1]), center), item[0], item[1])
    )

    radius_to_color: dict[int, int] = {}
    for r, c, color in sorted(
        markers, key=lambda item: (_cheb((item[0], item[1]), (anchor_r, anchor_c)), item[0], item[1])
    ):
        radius = _cheb((r, c), (anchor_r, anchor_c))
        if radius not in radius_to_color:
            radius_to_color[radius] = int(color)

    palette = [radius_to_color[radius] for radius in sorted(radius_to_color)]
    return (anchor_r, anchor_c), palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each wall-bounded region with concentric square bands that expand from a central marker.

    Input structure:
    - One wall color partitions the canvas into rectangular free regions.
    - Each free region is mostly one base color; some regions also contain a short
      diagonal chain of isolated marker pixels clustered around the region center.

    Rule:
    - In a marked region, find the marker closest to the region's bbox center and
      treat it as radius 0. Order the marker colors by their Chebyshev radius from
      that anchor marker.
    - Paint every cell in the region with palette[radius_from_anchor % palette_length],
      where radius is the cell's Chebyshev distance to the anchor marker.
    - Regions with no markers remain the base color, and walls stay unchanged.

    Key insight:
    The marker chain no longer encodes boundary shells. Instead it shows the first
    few radii of a center-origin square wave, so the solver must recover the center
    anchor first and then let the palette radiate outward from that point.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        vals = [int(g[r, c]) for r, c in pts]
        base = _mode(vals)
        anchor, palette = _anchor_and_palette(pts, base, g)

        if anchor is None:
            for r, c in pts:
                out[r, c] = base
            continue

        m = len(palette)
        for r, c in pts:
            d = _cheb((int(r), int(c)), anchor)
            out[r, c] = palette[d % m]

    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 12, 30
W_MIN, W_MAX = 15, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 5
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKER_CHOICES = [0, 2, 3]
MARKER_COUNT_WEIGHTS = [1, 4, 4]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 600


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
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _center_anchor(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _directional_capacity(
    rect: tuple[int, int, int, int], anchor: tuple[int, int], dr: int, dc: int
) -> int:
    r0, c0, r1, c1 = rect
    ar, ac = anchor
    row_steps = (r1 - ar) if dr > 0 else (ar - r0)
    col_steps = (c1 - ac) if dc > 0 else (ac - c0)
    return min(row_steps, col_steps)


def _place_markers(rect: tuple[int, int, int, int], marker_pool: list[int], rng: random.Random):
    """
    Produce a center-outward diagonal chain of 0/2/3 markers.
    Each marker = (radius, color, r, c).
    """
    anchor = _center_anchor(rect)
    feasible_dirs: dict[int, list[tuple[int, int]]] = {}
    feasible_choices = [0]

    for n in (2, 3):
        dirs = []
        for dr, dc in DIAG4:
            if _directional_capacity(rect, anchor, dr, dc) >= n - 1:
                dirs.append((dr, dc))
        if dirs:
            feasible_dirs[n] = dirs
            feasible_choices.append(n)

    weights = [MARKER_COUNT_WEIGHTS[MARKER_CHOICES.index(n)] for n in feasible_choices]
    n = rng.choices(feasible_choices, weights=weights)[0]
    if n == 0:
        return []

    dr, dc = rng.choice(feasible_dirs[n])
    colors = rng.sample(marker_pool, n)
    ar, ac = anchor
    return [(radius, colors[radius], ar + dr * radius, ac + dc * radius) for radius in range(n)]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    region_marker_counts = []
    marked_regions = 0
    radius2_regions = 0
    center_matches = 0

    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        region_marker_counts.append(len(markers))
        if markers:
            marked_regions += 1
            center_matches += int(markers[0][2:] == _center_anchor(rect))
            radius2_regions += int(len(markers) >= 3)

        r0, c0, r1, c1 = rect
        for _radius, color, rr, cc in markers:
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            used.add(color)

    stats = {
        "region_marker_counts": region_marker_counts,
        "marked_regions": marked_regions,
        "radius2_regions": radius2_regions,
        "center_matches": center_matches,
    }
    return used, stats


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _boundary_style_main(grid: np.ndarray) -> np.ndarray:
    """Original solver logic kept only to reject degenerate combo samples."""
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
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

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
            d = int(dist[r, c])
            out[r, c] = palette[d % m]

    return out


def _nearest_marker_style_main(grid: np.ndarray) -> np.ndarray:
    """Standalone D10-style solver kept only for generator-side degeneration checks."""
    g = np.asarray(grid, dtype=int)

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        vals = [int(g[r, c]) for r, c in pts]
        base = _mode(vals)
        markers: list[tuple[int, int, int]] = []
        for r, c in pts:
            v = int(g[r, c])
            if v != base:
                markers.append((int(r), int(c), v))
        markers.sort(key=lambda item: (item[0], item[1]))

        if not markers:
            for r, c in pts:
                out[r, c] = base
            continue

        palette = [color for _r, _c, color in markers]
        m = len(palette)
        for r, c in pts:
            idx, d = _nearest_marker_index_and_distance(int(r), int(c), markers)
            out[r, c] = palette[(idx + d) % m]

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
    used, stats = stamped

    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if stats["marked_regions"] < 2:
        return None
    if stats["radius2_regions"] < 1:
        return None
    if stats["center_matches"] != stats["marked_regions"]:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        boundary_out = _boundary_style_main(grid)
        nearest_out = _nearest_marker_style_main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, boundary_out):
        return None
    if np.array_equal(out, nearest_out):
        return None

    diff_boundary = np.count_nonzero(out != boundary_out)
    diff_nearest = np.count_nonzero(out != nearest_out)
    if diff_boundary < max(8, (H * W) // 18):
        return None
    if diff_nearest < max(8, (H * W) // 24):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
