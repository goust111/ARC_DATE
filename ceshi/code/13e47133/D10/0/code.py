# Variant of 13e47133
# Dimension: D10 — 距离度量方式 (meta)
# Change: 使用切比雪夫距离（8-邻域）计算到区域边界的距离... → 使用曼哈顿距离（4-邻域）

import random

import numpy as np

from arc_utils import components_mask

# Design analysis for the D10 variant
# -----------------------------------
# a) Concrete behavior:
#    The input still consists of wall-colored partitions over one dominant base color,
#    plus 0-3 marker pixels inside each non-wall region. The key change is that marker
#    depth and fill depth are now measured by 4-neighbor distance to the region
#    boundary. In a plain rectangle this often matches the old layering, so the
#    generator must create at least one L/T-like region where 4-neighbor and
#    8-neighbor depth differ. Example: in an L-shaped region, a cell tucked behind a
#    concave corner can be depth 2 by 4-neighbor BFS but only depth 1 by 8-neighbor
#    BFS because diagonal escape is allowed in the old metric. The palette position of
#    a marker at that cell therefore changes, and the filled shells bend differently.
# b) Complexity calibration:
#    If every region stayed rectangular, many samples would look unchanged. To keep the
#    metric shift observable without adding unrelated mechanics, walls are still axis
#    aligned and region counts stay small, but at least one region is made irregular.
#    Marker counts remain 0-3 and start distance remains 0 or 1, so the rule is still
#    learnable from a few examples.
# c) Implicit assumptions review:
#    - The original boundary detector assumed 8-neighbor contact; this must change to
#      4-neighbor contact or Manhattan depth is wrong from the first layer onward.
#    - The original BFS also expanded in 8 directions; that must change to 4-direction
#      expansion or interior depths collapse back to Chebyshev depth.
#    - Palette extraction by "one marker per depth, first in reading order" remains
#      valid, but the generator must ensure chosen markers really occupy consecutive
#      Manhattan depths.
#    - The bg/wall inference by modal colors still works only if bg stays global mode
#      and wall stays second mode, so generation keeps that invariant and validates it.
# d) Placement legality:
#    The wall generator draws connected axis-aligned segments whose endpoints touch the
#    border or an existing wall, so every wall pixel has an orthogonal wall neighbor.
#    Region sizes and counts are validated after construction. Marker placement only
#    uses cells inside a component, never overwrites walls, and checks adjacency softly
#    to keep markers visually isolated.
# e) Visual observability:
#    The generator rejects layouts where Manhattan and Chebyshev solvers produce the
#    same output. It also requires at least one marked region containing cells whose
#    Manhattan depth differs from Chebyshev depth, so the changed metric is supported
#    by visible evidence rather than hidden metadata.
# f) Visual cleanliness:
#    Regions are few and fairly large, markers are single cells, and the wall network
#    is only an elbow plus one or two extra splits. This preserves clear partitions and
#    avoids clutter from too many overlapping cues.
# g) Color inference:
#    Colors are never sorted by color id. Marker colors only encode palette order via
#    spatial depth, exactly as in the original task. The number of distinct marker
#    colors remains modest, and repeated use across regions is allowed.

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _pts_to_mask(pts, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask


def _boundary_distance(component_mask: np.ndarray, boundary_dirs, step_dirs) -> np.ndarray:
    """Distance from each True cell to the component boundary under the given neighborhood."""
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in boundary_dirs:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
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
        for dr, dc in step_dirs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _boundary_distance_manhattan(component_mask: np.ndarray) -> np.ndarray:
    return _boundary_distance(component_mask, DIR4, DIR4)


def _boundary_distance_chebyshev(component_mask: np.ndarray) -> np.ndarray:
    return _boundary_distance(component_mask, DIR8, DIR8)


def _solve_with_distance(grid: np.ndarray, dist_fn) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = _pts_to_mask(pts, h, w)
        vals = g[comp_mask]
        base = _mode(vals)
        dist = dist_fn(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in sorted((int(r), int(c)) for r, c in pts):
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            palette = []
            ds = sorted(dist_to_color)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each wall-bounded region using a cyclic palette indexed by Manhattan depth from the region boundary.

    Input structure:
    - The grid is split by one wall color into several 4-connected non-wall regions.
    - Each region is mostly one base color, with a few single-cell markers of other colors embedded at different inward depths.

    Rule:
    - For every region, compute each marker's 4-neighbor (Manhattan) distance to the region boundary and order marker colors by that depth.
    - If the shallowest marker is not on the boundary layer, prepend the region's base color to the palette.
    - Recolor every cell in the region with `palette[distance % len(palette)]`, where `distance` is the same Manhattan boundary distance; walls remain unchanged.

    Key insight:
    The markers do not label absolute positions; they define a repeating palette over taxicab-distance shells, so irregular regions reveal bent, diamond-like layers instead of the old 8-neighbor framing.
    """
    return _solve_with_distance(grid, _boundary_distance_manhattan)


H_MIN, H_MAX = 14, 30
W_MIN, W_MAX = 14, 30
MIN_REGION_DIM = 3
MIN_REGION_CELLS = 18
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
GEN_MAX_ATTEMPTS = 400
MIN_VARIANT_DIFF = 12
EXTRA_SPLIT_PROB = 0.6


def _draw_h(mask: np.ndarray, r: int, c0: int, c1: int):
    c0, c1 = sorted((int(c0), int(c1)))
    mask[int(r), c0:c1 + 1] = 1


def _draw_v(mask: np.ndarray, c: int, r0: int, r1: int):
    r0, r1 = sorted((int(r0), int(r1)))
    mask[r0:r1 + 1, int(c)] = 1


def _random_symmetry(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, rng.randrange(4))
    return np.ascontiguousarray(out)


def _wall_has_no_dangling_pixels(mask: np.ndarray) -> bool:
    h, w = mask.shape
    wall_pts = np.argwhere(mask == 1)
    for r, c in wall_pts:
        if not any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] == 1 for dr, dc in DIR4):
            return False
    return True


def _is_rect_component(pts) -> bool:
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return area == len(pts)


def _component_info(pts, h: int, w: int):
    mask = _pts_to_mask(pts, h, w)
    d4 = _boundary_distance_manhattan(mask)
    d8 = _boundary_distance_chebyshev(mask)
    coords = sorted((int(r), int(c)) for r, c in pts)
    cells_by_d: dict[int, list[tuple[int, int]]] = {}
    gap_cells = 0
    max_d = 0
    for r, c in coords:
        d = int(d4[r, c])
        cells_by_d.setdefault(d, []).append((r, c))
        max_d = max(max_d, d)
        if d != int(d8[r, c]):
            gap_cells += 1
    center_r = sum(r for r, _ in coords) / len(coords)
    center_c = sum(c for _, c in coords) / len(coords)
    return {
        "pts": coords,
        "mask": mask,
        "d4": d4,
        "d8": d8,
        "cells_by_d": cells_by_d,
        "gap_cells": gap_cells,
        "max_d": max_d,
        "center_r": center_r,
        "center_c": center_c,
        "rect": _is_rect_component(coords),
    }


def _choose_marker_cells(info, ds, rng: random.Random, require_gap: bool):
    chosen: list[tuple[int, int]] = []
    used_gap = False
    d8 = info["d8"]

    for d in ds:
        candidates = info["cells_by_d"].get(d, [])
        if not candidates:
            return None

        separated = [
            p for p in candidates
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in chosen)
        ]
        if separated:
            candidates = separated

        candidates = candidates[:]
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda p: (
                0 if (require_gap and not used_gap and int(d8[p[0], p[1]]) != d) else 1,
                -abs(int(d8[p[0], p[1]]) - d),
                abs(p[0] - info["center_r"]) + abs(p[1] - info["center_c"]),
                p[0],
                p[1],
            )
        )

        pick = candidates[0]
        if int(d8[pick[0], pick[1]]) != d:
            used_gap = True
        chosen.append(pick)

    if require_gap and not used_gap:
        return None
    return chosen


def _pick_region_markers(info, rng: random.Random, marker_pool: list[int], force_gap: bool):
    max_d = int(info["max_d"])
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return []

    if force_gap:
        if upper < 2:
            return None
        n = 3 if upper >= 3 and rng.random() < 0.6 else 2
    else:
        n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
        n = min(n, upper)

    if n == 0:
        return []

    start_hi = min(START_DIST_MAX, max_d - (n - 1))
    if start_hi < 0:
        return None if force_gap else []

    starts = list(range(start_hi + 1))
    if force_gap and 1 in starts:
        starts = [1, 0]
    elif rng.random() < 0.5:
        starts.reverse()

    chosen_cells = None
    start_d = 0
    for s in starts:
        ds = list(range(s, s + n))
        chosen_cells = _choose_marker_cells(info, ds, rng, require_gap=force_gap)
        if chosen_cells is not None:
            start_d = s
            break
    if chosen_cells is None:
        return None if force_gap else []

    colors = rng.sample(marker_pool, n)
    return [
        (start_d + k, colors[k], chosen_cells[k][0], chosen_cells[k][1])
        for k in range(n)
    ]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _cheb_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_distance(grid, _boundary_distance_chebyshev)


def _make_wall_layout(H: int, W: int, rng: random.Random):
    for _ in range(80):
        mask = np.zeros((H, W), dtype=int)

        rr = rng.randint(MIN_REGION_DIM, H - MIN_REGION_DIM - 1)
        cc = rng.randint(MIN_REGION_DIM, W - MIN_REGION_DIM - 1)
        c2_lo = cc + MIN_REGION_DIM + 1
        c2_hi = W - MIN_REGION_DIM - 1
        if c2_lo > c2_hi:
            continue

        _draw_h(mask, rr, 0, cc)
        _draw_v(mask, cc, 0, rr)

        c2 = rng.randint(c2_lo, c2_hi)
        _draw_v(mask, c2, rr, H - 1)

        if rng.random() < EXTRA_SPLIT_PROB:
            c1_lo = MIN_REGION_DIM
            c1_hi = cc - MIN_REGION_DIM - 1
            if c1_lo <= c1_hi:
                c1 = rng.randint(c1_lo, c1_hi)
                _draw_v(mask, c1, 0, rr)

        mask = _random_symmetry(mask, rng)
        free = mask == 0
        comps = _cc4(free)

        if not (N_REGIONS_MIN <= len(comps) <= N_REGIONS_MAX):
            continue
        if min(len(pts) for pts in comps) < MIN_REGION_CELLS:
            continue
        if not _wall_has_no_dangling_pixels(mask):
            continue

        infos = [_component_info(pts, *mask.shape) for pts in comps]
        if not any(not info["rect"] for info in infos):
            continue
        if not any(info["gap_cells"] > 0 and info["max_d"] >= 2 for info in infos):
            continue

        return mask, comps, infos
    return None, None, None


def _stamp_markers(grid: np.ndarray, infos, bg: int, marker_pool: list[int], rng: random.Random):
    featured = [i for i, info in enumerate(infos) if info["gap_cells"] > 0 and info["max_d"] >= 2]
    if not featured:
        return None
    featured_idx = rng.choice(featured)

    used_colors = set()
    featured_gap_marked = False

    for i, info in enumerate(infos):
        markers = _pick_region_markers(info, rng, marker_pool, force_gap=(i == featured_idx))
        if markers is None:
            return None
        if i == featured_idx and markers:
            featured_gap_marked = any(info["d4"][r, c] != info["d8"][r, c] for _d, _col, r, c in markers)
        for _d, col, r, c in markers:
            if grid[r, c] != bg:
                return None
            grid[r, c] = col
            used_colors.add(col)

    if not featured_gap_marked:
        return None
    return used_colors


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, comps, infos = _make_wall_layout(H, W, rng)
    if wall_mask is None:
        return None

    grid = np.full(wall_mask.shape, bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, infos, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    out_new = main(grid)
    out_old = _cheb_main(grid)
    if np.array_equal(grid, out_new):
        return None
    if np.count_nonzero(out_new != out_old) < MIN_VARIANT_DIFF:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
