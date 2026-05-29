# Combo variant of 13e47133 (index 21)
# combo_spec: D4/1+D7/2
# Name: 碰撞停止+不规则色块分隔

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    The seed task uses one straight wall color to carve the canvas into mostly
#    rectangular regions, then reads singleton marker colors by Chebyshev depth
#    and fills each whole region with a cyclic frame pattern. This combo changes
#    two stages together. First, the separating wall is no longer a thin straight
#    line system: attached wall blobs make the free regions visibly non-rectangular.
#    Second, some regions also contain compact multi-cell blocker patches. Example
#    on a small region:
#    - Original: if a cell has boundary depth 2, it always becomes palette[2].
#    - After D7/2 only: the same palette logic is still used, but the region's
#      depth-2 cells now follow an uneven silhouette because the wall boundary has
#      bites and bulges.
#    - After adding D4/1: a blocker near the upper side interrupts vertical growth.
#      A cell can still be painted from left/right/bottom if those straight paths
#      reach the true region boundary before hitting the blocker. If every outward
#      ray hits a blocker first, that cell stays base-colored.
#    - Difference: irregular walls change the outer frame geometry, while blockers
#      locally truncate the frames inside that already-irregular geometry.
# b) Complexity calibration
#    The visual signal must be strong but not noisy. Wall irregularity is limited
#    to attached blobs grown from a simple split skeleton, so regions stay easy to
#    parse as a few large components instead of random speckle. Blockers are one
#    patch per selected region at most, and markers stay singleton pixels with
#    consecutive depths. This keeps the inference space small enough for ARC while
#    still making the combo visibly different from the seed.
# c) Implicit assumptions audit against the original main()
#    - `wall = second-most-common color` can still work, but only if the generator
#      keeps wall area comfortably above blocker area. The new generator enforces
#      that.
#    - `free = g != wall` still defines regions, but those regions are no longer
#      rectangles, so bbox-depth is no longer a valid proxy for frame depth.
#    - Non-base colors inside a region cannot all be treated as markers anymore:
#      singleton components are markers, while multi-cell same-color components are
#      blockers and must remain unchanged in the output.
#    - Collision tests may not look to the bbox edge. In an irregular region,
#      reaching an outside-of-region gap is success because that is the real local
#      boundary; hitting a blocker first is failure. The solver therefore computes
#      directional reachability to the true boundary, not to the bbox perimeter.
# d) Placement legality
#    The generator starts from legal wall splits, then adds wall blobs only by
#    attaching to existing wall cells. Final free components are recomputed and
#    filtered by count and area. Markers are placed on a descending 8-neighbor
#    depth chain so each chosen depth definitely exists. Blocker shapes are tested
#    cell-by-cell to stay inside one region, off the wall, off markers, and within
#    bounds.
# e) Visual observability
#    A human can see three distinct object roles: dominant base background, one
#    separator color forming irregular islands/ridges, and a separate blocker color
#    forming compact patches inside some regions. The output shows that marker
#    singletons still define the palette order, but some bands stop early and some
#    outer rings are asymmetric because the region boundary itself is jagged.
# f) Visual cleanliness
#    Region count stays low, regions are large, and each active region has at most
#    one blocker patch. This avoids layering too many mechanisms in one place. The
#    irregularity comes from boundary shape, not from dense texture.
# g) Color inference
#    The rule never depends on numeric color ids. Marker colors are only ordered by
#    visible boundary depth, wall color is identified structurally as the big
#    separator mass, and blocker color is identified structurally as multi-cell
#    interior patches that survive unchanged.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _cc4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _mask_from_points(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask


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


def _classify_region_colors(g: np.ndarray, comp_mask: np.ndarray):
    base = _mode(g[comp_mask])
    marker_cells: list[tuple[int, int, int]] = []
    blocker_mask = np.zeros_like(comp_mask, dtype=bool)

    for color in np.unique(g[comp_mask]):
        color = int(color)
        if color == base:
            continue
        for pts in _cc4(comp_mask & (g == color)):
            if len(pts) == 1:
                r, c = map(int, pts[0])
                marker_cells.append((r, c, color))
            else:
                for r, c in pts:
                    blocker_mask[int(r), int(c)] = True
    return base, marker_cells, blocker_mask


def _build_palette(marker_cells: list[tuple[int, int, int]], base: int, dist: np.ndarray) -> list[int]:
    dist_to_color: dict[int, int] = {}
    for r, c, v in sorted(marker_cells, key=lambda x: (int(dist[x[0], x[1]]), x[0], x[1])):
        d = int(dist[r, c])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = int(v)

    if not dist_to_color:
        return [int(base)]

    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def _directional_depths(comp_mask: np.ndarray, blocker_mask: np.ndarray):
    rows, cols = np.where(comp_mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    local_comp = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
    local_block = blocker_mask[r0 : r1 + 1, c0 : c1 + 1]
    local_free = local_comp & (~local_block)
    bh, bw = local_comp.shape

    top = np.full((bh, bw), -1, dtype=int)
    bottom = np.full((bh, bw), -1, dtype=int)
    left = np.full((bh, bw), -1, dtype=int)
    right = np.full((bh, bw), -1, dtype=int)

    for c in range(bw):
        for r in range(bh):
            if not local_free[r, c]:
                continue
            if r == 0 or not local_comp[r - 1, c]:
                top[r, c] = 0
            elif local_block[r - 1, c]:
                top[r, c] = -1
            elif top[r - 1, c] >= 0:
                top[r, c] = top[r - 1, c] + 1

    for c in range(bw):
        for r in range(bh - 1, -1, -1):
            if not local_free[r, c]:
                continue
            if r == bh - 1 or not local_comp[r + 1, c]:
                bottom[r, c] = 0
            elif local_block[r + 1, c]:
                bottom[r, c] = -1
            elif bottom[r + 1, c] >= 0:
                bottom[r, c] = bottom[r + 1, c] + 1

    for r in range(bh):
        for c in range(bw):
            if not local_free[r, c]:
                continue
            if c == 0 or not local_comp[r, c - 1]:
                left[r, c] = 0
            elif local_block[r, c - 1]:
                left[r, c] = -1
            elif left[r, c - 1] >= 0:
                left[r, c] = left[r, c - 1] + 1

    for r in range(bh):
        for c in range(bw - 1, -1, -1):
            if not local_free[r, c]:
                continue
            if c == bw - 1 or not local_comp[r, c + 1]:
                right[r, c] = 0
            elif local_block[r, c + 1]:
                right[r, c] = -1
            elif right[r, c + 1] >= 0:
                right[r, c] = right[r, c + 1] + 1

    return (r0, c0), local_free, top, bottom, left, right


def _paint_region(
    out: np.ndarray,
    comp_mask: np.ndarray,
    base: int,
    palette: list[int],
    blocker_mask: np.ndarray,
    dist: np.ndarray,
    *,
    collision_stop: bool,
) -> None:
    rows, cols = np.where(comp_mask)
    if not collision_stop:
        m = len(palette)
        for r, c in zip(rows, cols):
            r = int(r)
            c = int(c)
            if blocker_mask[r, c]:
                continue
            out[r, c] = int(palette[int(dist[r, c]) % m])
        return

    (r0, c0), local_free, top, bottom, left, right = _directional_depths(comp_mask, blocker_mask)
    m = len(palette)
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        if blocker_mask[r, c]:
            continue

        lr, lc = r - r0, c - c0
        if not local_free[lr, lc]:
            continue

        depths = [v for v in (top[lr, lc], bottom[lr, lc], left[lr, lc], right[lr, lc]) if v >= 0]
        if depths:
            out[r, c] = int(palette[min(depths) % m])
        else:
            out[r, c] = int(base)


def _solve(grid: np.ndarray, *, collision_stop: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts in _cc4(free):
        comp_mask = _mask_from_points((h, w), pts)
        base, marker_cells, blocker_mask = _classify_region_colors(g, comp_mask)
        dist = _boundary_distance(comp_mask)
        palette = _build_palette(marker_cells, base, dist)
        _paint_region(out, comp_mask, base, palette, blocker_mask, dist, collision_stop=collision_stop)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill irregular wall-separated regions with palette bands that can be truncated by interior blockers.

    Input structure:
    - One dominant base color occupies the free space.
    - A second, more massive non-base color forms irregular connected wall blobs that split the canvas into 3-6 non-rectangular regions.
    - Inside each region, isolated singleton pixels of other colors act as palette markers, while any multi-cell same-color interior patch is a blocker rather than a marker.

    Rule:
    - For each non-wall region, compute every cell's Chebyshev distance to that region's true outer boundary and sort singleton marker colors by that depth.
    - If the shallowest marker is not on the boundary, prepend the base color to form the cyclic palette.
    - Repaint the region using the same concentric-band idea, but a cell may only inherit depth from a direction whose straight row/column ray reaches the real region boundary before hitting a blocker patch.
    - Use the smallest reachable directional depth to pick the palette color; if all four directions are blocked first, leave that cell base-colored. Wall blobs and blocker patches themselves stay unchanged.

    Key insight: The marker singletons still encode the color cycle, but now two spatial effects interact: irregular separator blobs distort the outer rings, and interior blocker patches cast directional shadows that stop some frame segments early.
    """
    return _solve(grid, collision_stop=True)


# --------------- generator ---------------

H_MIN, H_MAX = 12, 28
W_MIN, W_MAX = 13, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 4
MIN_REGION_DIM = 4
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MIN_REGION_AREA = 18

MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

MIN_BLOCKER_REGIONS = 1
MAX_BLOCKER_REGIONS = 3
BLOCKER_BOUNDARY_MAX = 3

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7

IRREGULAR_GAP_MIN = 4
MIN_IRREGULAR_REGIONS = 2

GEN_MAX_ATTEMPTS = 700


def _poly_orientations(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    cells = list(shape)
    for flip in (1, -1):
        for rot in range(4):
            pts = []
            for r, c in cells:
                rr, cc = r, c * flip
                for _ in range(rot):
                    rr, cc = cc, -rr
                pts.append((rr, cc))
            min_r = min(r for r, _ in pts)
            min_c = min(c for _, c in pts)
            norm = tuple(sorted((r - min_r, c - min_c) for r, c in pts))
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


_BASE_BLOCKER_SHAPES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]

BLOCKER_SHAPES = []
for _shape in _BASE_BLOCKER_SHAPES:
    BLOCKER_SHAPES.extend(_poly_orientations(_shape))


def _place_wall_skeleton(H: int, W: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=bool)
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
            wall_mask[rr, c0 : c1 + 1] = True
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = True
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask


def _grow_wall_blob(wall_mask: np.ndarray, rng: random.Random) -> bool:
    H, W = wall_mask.shape
    wall_cells = [tuple(map(int, x)) for x in np.argwhere(wall_mask)]
    if not wall_cells:
        return False

    for _ in range(30):
        ar, ac = rng.choice(wall_cells)
        frontier = []
        for dr, dc in DIR4:
            rr, cc = ar + dr, ac + dc
            if 0 <= rr < H and 0 <= cc < W and not wall_mask[rr, cc]:
                frontier.append((rr, cc))
        if not frontier:
            continue

        blob = {rng.choice(frontier)}
        target_size = rng.randint(1, 4)
        while len(blob) < target_size:
            cand = []
            for br, bc in list(blob):
                for dr, dc in DIR4:
                    rr, cc = br + dr, bc + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if wall_mask[rr, cc] or (rr, cc) in blob:
                        continue
                    if any(
                        0 <= rr2 < H
                        and 0 <= cc2 < W
                        and (wall_mask[rr2, cc2] or (rr2, cc2) in blob)
                        for dr2, dc2 in DIR4
                        for rr2, cc2 in [(rr + dr2, cc + dc2)]
                    ):
                        cand.append((rr, cc))
            if not cand:
                break
            blob.add(rng.choice(cand))

        for rr, cc in blob:
            wall_mask[rr, cc] = True
        return True

    return False


def _irregularize_walls(base_wall: np.ndarray, rng: random.Random) -> np.ndarray:
    wall_mask = base_wall.copy()
    n_blobs = rng.randint(max(6, int(wall_mask.sum()) // 5), max(8, int(wall_mask.sum()) // 2))
    for _ in range(n_blobs):
        _grow_wall_blob(wall_mask, rng)
    return wall_mask


def _pick_colors(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    blocker = rng.choice(colors)
    colors.remove(blocker)
    return bg, wall, blocker, colors


def _reconstruct_depth_path(comp_mask: np.ndarray, dist: np.ndarray, end: tuple[int, int], rng: random.Random):
    h, w = comp_mask.shape
    path = {int(dist[end]): end}
    r, c = end
    d = int(dist[r, c])
    while d > 0:
        prev = []
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc] and int(dist[rr, cc]) == d - 1:
                prev.append((rr, cc))
        if not prev:
            return None
        r, c = rng.choice(prev)
        d -= 1
        path[d] = (r, c)
    return path


def _place_markers_in_region(comp_mask: np.ndarray, marker_pool: list[int], rng: random.Random):
    dist = _boundary_distance(comp_mask)
    max_d = int(dist[comp_mask].max())
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return [], dist

    count_choices = list(range(1, len(MARKER_COUNT_WEIGHTS)))
    n = rng.choices(count_choices, weights=MARKER_COUNT_WEIGHTS[1:])[0]
    n = min(n, upper)

    start_max = min(START_DIST_MAX, max_d - (n - 1))
    if start_max < 0:
        return [], dist
    start_d = rng.randint(0, start_max)
    deepest = start_d + n - 1
    candidates = [tuple(map(int, x)) for x in np.argwhere((dist == deepest) & comp_mask)]
    rng.shuffle(candidates)
    for end in candidates:
        path = _reconstruct_depth_path(comp_mask, dist, end, rng)
        if path is None:
            continue
        colors = rng.sample(marker_pool, n)
        markers = []
        for k in range(n):
            d = start_d + k
            r, c = path[d]
            markers.append((d, colors[k], int(r), int(c)))
        return markers, dist
    return [], dist


def _stamp_markers(grid: np.ndarray, comps: list[np.ndarray], bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    markers_by_region: dict[int, list[tuple[int, int, int, int]]] = {}
    dist_by_region: dict[int, np.ndarray] = {}

    for idx, pts in enumerate(comps):
        comp_mask = _mask_from_points(grid.shape, pts)
        markers, dist = _place_markers_in_region(comp_mask, marker_pool, rng)
        dist_by_region[idx] = dist
        markers_by_region[idx] = markers
        for _d, col, rr, cc in markers:
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(int(col))
    return used, (markers_by_region, dist_by_region)


def _bbox_gap(comp_mask: np.ndarray) -> int:
    rows, cols = np.where(comp_mask)
    area = len(rows)
    bbox_area = (int(rows.max()) - int(rows.min()) + 1) * (int(cols.max()) - int(cols.min()) + 1)
    return int(bbox_area - area)


def _blocker_candidates(
    comp_mask: np.ndarray,
    dist: np.ndarray,
    occupied: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
):
    H, W = comp_mask.shape
    out = []
    for shape in BLOCKER_SHAPES:
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        for r0 in range(H - sh + 1):
            for c0 in range(W - sw + 1):
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if any(not comp_mask[r, c] for r, c in cells):
                    continue
                if any((r, c) in occupied for r, c in cells):
                    continue
                if min(int(dist[r, c]) for r, c in cells) < 1:
                    continue
                if min(int(dist[r, c]) for r, c in cells) > BLOCKER_BOUNDARY_MAX:
                    continue
                too_close = False
                for mr, mc in marker_cells:
                    if any(abs(r - mr) <= 1 and abs(c - mc) <= 1 for r, c in cells):
                        too_close = True
                        break
                if too_close:
                    continue
                out.append(cells)
    return out


def _stamp_blockers(
    grid: np.ndarray,
    comps: list[np.ndarray],
    blocker_color: int,
    marker_meta,
    rng: random.Random,
):
    markers_by_region, dist_by_region = marker_meta
    eligible = []
    for idx, pts in enumerate(comps):
        comp_mask = _mask_from_points(grid.shape, pts)
        dist = dist_by_region[idx]
        if int(dist[comp_mask].max()) < 2:
            continue
        if len(pts) < 24:
            continue
        eligible.append((idx, comp_mask, dist))

    if not eligible:
        return 0

    rng.shuffle(eligible)
    min_target = min(MIN_BLOCKER_REGIONS, len(eligible))
    if len(eligible) >= 2:
        min_target = max(min_target, 2)
    max_target = min(MAX_BLOCKER_REGIONS, len(eligible))
    target = rng.randint(min_target, max_target)
    placed = 0
    for idx, comp_mask, dist in eligible:
        if placed >= target:
            break
        marker_cells = {(int(rr), int(cc)) for _d, _col, rr, cc in markers_by_region[idx]}
        occupied = set(marker_cells)
        candidates = _blocker_candidates(comp_mask, dist, occupied, marker_cells)
        rng.shuffle(candidates)
        for cells in candidates:
            for r, c in cells:
                grid[r, c] = blocker_color
            placed += 1
            break
    return placed


def _validate_color_order(grid: np.ndarray, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray):
    out_stop = _solve(grid, collision_stop=True)
    out_full = _solve(grid, collision_stop=False)

    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    free = g != wall
    diff = 0
    blocker_cells = 0
    active_regions = 0
    irregular_regions = 0
    shadow_base_cells = 0

    for pts in _cc4(free):
        comp_mask = _mask_from_points((h, w), pts)
        if _bbox_gap(comp_mask) >= IRREGULAR_GAP_MIN:
            irregular_regions += 1
        base, _markers, blocker_mask = _classify_region_colors(g, comp_mask)
        blocker_cells += int(blocker_mask.sum())

        region_diff = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            if blocker_mask[r, c]:
                continue
            if int(out_stop[r, c]) != int(out_full[r, c]):
                diff += 1
                region_diff += 1
            if int(out_stop[r, c]) == int(base) and int(out_full[r, c]) != int(base):
                shadow_base_cells += 1
        if blocker_mask.any() and region_diff > 0:
            active_regions += 1

    return {
        "collision_diff_cells": diff,
        "blocker_cells": blocker_cells,
        "active_regions": active_regions,
        "irregular_regions": irregular_regions,
        "shadow_base_cells": shadow_base_cells,
        "changed_from_input": int(np.count_nonzero(out_stop != g)),
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, blocker, marker_pool = _pick_colors(rng)

    wall_mask = _irregularize_walls(_place_wall_skeleton(H, W, rng), rng)
    free = ~wall_mask
    comps = _cc4(free)
    if not (N_REGIONS_MIN <= len(comps) <= N_REGIONS_MAX):
        return None
    if min(len(pts) for pts in comps) < MIN_REGION_AREA:
        return None
    irregular_count = 0
    for pts in comps:
        comp_mask = _mask_from_points((H, W), pts)
        if _bbox_gap(comp_mask) >= IRREGULAR_GAP_MIN:
            irregular_count += 1
    if irregular_count < MIN_IRREGULAR_REGIONS:
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask] = wall

    used, marker_meta = _stamp_markers(grid, comps, bg, marker_pool, rng)
    if used is None:
        return None
    markers_by_region, _dist_by_region = marker_meta
    if any(len(markers) == 0 for markers in markers_by_region.values()):
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None

    placed_blockers = _stamp_blockers(grid, comps, blocker, marker_meta, rng)
    if placed_blockers < MIN_BLOCKER_REGIONS:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if metrics["irregular_regions"] < MIN_IRREGULAR_REGIONS:
        return None
    if metrics["blocker_cells"] < 3:
        return None
    min_active_regions = 2 if placed_blockers >= 2 else 1
    if metrics["active_regions"] < min_active_regions:
        return None
    if metrics["collision_diff_cells"] < 8:
        return None
    if metrics["shadow_base_cells"] < 4:
        return None
    if metrics["changed_from_input"] < 12:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
