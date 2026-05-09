# Combo variant of 13e47133 (index 19)
# combo_spec: D9/0+D11/0
# Name: 按面积排序+统一标记数量

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the fused variant:
# - Concrete behavior: if three regions have areas 20 < 30 < 42 and marker pairs
#   [2,3], [3,4], [4,5] at distances d=0,1, then the original task fills them
#   independently with palettes [2,3], [3,4], [4,5]. The fused rule processes
#   the smallest region first, consumes 2 and 3 there, then region 2 loses color
#   3 and becomes palette [base,4], and region 3 loses color 4 and becomes
#   palette [base,5]. The visible dependency is therefore "size order controls
#   later color availability".
# - Complexity calibration: every region has exactly two markers, always at
#   consecutive depths d=0,1. The generator alternates among a few simple reuse
#   templates so examples cover the three meaningful post-filter outcomes for a
#   later region: keep only d=0, keep only d=1 (thus reinsert base), or lose
#   both and fall back to base-only. This keeps the logic inferable while still
#   exercising the sequential-consumption rule beyond one narrow branch.
# - Hidden assumptions changed from the original solver: regions are no longer
#   independent, and 0-marker cases are removed. `main()` must sort by area,
#   maintain a global consumed-color set, and reinsert the region base color if
#   filtering removes the boundary-distance marker. Equal-area regions would make
#   the order depend on an invisible tie-break, so the generator forbids them.
# - Placement legality and visual cleanliness: walls still form disjoint
#   rectangular regions, markers stay on one diagonal inside their own rectangle
#   at valid Chebyshev depths, and no marker overlaps another object. Reused
#   colors appear in short, local motifs rather than arbitrary clashes, making
#   the fused behavior visually testable from a few examples.


DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _cc4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _mode(arr) -> int:
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
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _detect_bg_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    return int(bg), int(wall)


def _extract_regions(grid: np.ndarray, wall: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions = []
    for comp in _cc4(g != wall):
        pts = sorted((int(r), int(c)) for r, c in comp)
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

        regions.append(
            {
                "pts": pts,
                "mask": comp_mask,
                "base": int(base),
                "dist": dist,
                "dist_to_color": dist_to_color,
                "area": len(pts),
                "anchor": pts[0],
            }
        )
    return regions


def _build_palette(base: int, dist_to_color: dict[int, int], consumed: set[int]) -> tuple[list[int], list[int]]:
    kept: list[tuple[int, int]] = []
    for d in sorted(dist_to_color):
        color = int(dist_to_color[d])
        if color not in consumed:
            kept.append((int(d), color))

    if not kept:
        return [int(base)], []

    palette: list[int] = []
    if kept[0][0] > 0:
        palette.append(int(base))

    used_now: list[int] = []
    for _d, color in kept:
        palette.append(int(color))
        used_now.append(int(color))
    return palette, used_now


def _solve(grid: np.ndarray, *, sequential: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, wall = _detect_bg_wall(g)
    regions = _extract_regions(g, wall)
    regions.sort(key=lambda info: (int(info["area"]), info["anchor"]))

    out = g.copy()
    consumed: set[int] = set()
    for info in regions:
        active_consumed = consumed if sequential else set()
        palette, used_now = _build_palette(info["base"], info["dist_to_color"], active_consumed)
        m = len(palette)
        for r, c in info["pts"]:
            d = int(info["dist"][r, c])
            out[r, c] = int(palette[d % m])
        if sequential:
            consumed.update(used_now)
    return out


def _independent_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, sequential=False)


def _consumption_effect_count(grid: np.ndarray) -> int:
    combo_out = _solve(grid, sequential=True)
    independent_out = _solve(grid, sequential=False)
    _bg, wall = _detect_bg_wall(np.asarray(grid, dtype=int))
    changed = 0
    for info in _extract_regions(np.asarray(grid, dtype=int), wall):
        if any(int(combo_out[r, c]) != int(independent_out[r, c]) for r, c in info["pts"]):
            changed += 1
    return changed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned regions with concentric bands, but let smaller regions spend colors before larger ones.

    Input structure:
    - The grid is cut into wall-separated regions that share one dominant base color.
    - Every region contains the same number of isolated marker pixels, and those markers sit at specific Chebyshev depths from the region boundary.

    Rule:
    - Order the regions by visible area from smallest to largest.
    - Within each region, read marker colors by increasing boundary distance to form the palette, but discard any marker colors already consumed by earlier regions; if the surviving palette no longer starts at distance 0, prepend the region base color.
    - Paint the region by cycling through that filtered palette with `palette[distance % len(palette)]`, leaving wall cells unchanged.

    Key insight: The marker count is uniform, so the main extra deduction is global rather than local: infer the area order, track which colors smaller regions spend, and then apply the usual concentric-distance fill with the remaining colors.
    """
    return _solve(grid, sequential=True)


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MIN_AREA_GAP = 4

# ---- Markers ----
MARKERS_PER_REGION = 2
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _place_walls(h: int, w: int, rng: random.Random):
    wall_mask = np.zeros((h, w), dtype=int)
    rects = [(0, 0, h - 1, w - 1)]
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
        axis_choices = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axis_choices)

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


def _validate_area_order(rects: list[tuple[int, int, int, int]]) -> bool:
    areas = sorted(_rect_area(rect) for rect in rects)
    if len(set(areas)) != len(areas):
        return False
    if len(areas) >= 2 and min(b - a for a, b in zip(areas, areas[1:])) < MIN_AREA_GAP:
        return False
    return True


def _marker_cells(rect: tuple[int, int, int, int], colors: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    if _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1) < len(colors) - 1:
        return None

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    cells = []
    for d, color in enumerate(colors):
        rr = br + dr * d
        cc = bc + dc * d
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            return None
        cells.append((int(d), int(color), int(rr), int(cc)))
    return cells


def _stamp_markers(
    grid: np.ndarray,
    rect_infos: list[dict],
    bg: int,
    region_pairs: list[tuple[int, int]],
    rng: random.Random,
):
    used = set()
    for i, info in enumerate(rect_infos):
        pair = [int(color) for color in region_pairs[i]]
        cells = _marker_cells(info["rect"], pair, rng)
        if cells is None:
            return None
        for _d, color, rr, cc in cells:
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = int(color)
            used.add(int(color))
    return used


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != int(bg):
        return False
    return len(order) >= 2 and int(vals[order[1]]) == int(wall)


def _region_color_templates(n_regions: int) -> list[list[tuple[int, int]]]:
    templates: list[list[tuple[int, int]]] = []

    chain = [(i, i + 1) for i in range(n_regions)]
    templates.append(chain)

    if n_regions >= 2:
        d0_keep = [(0, 1), (2, 1)]
        next_color = 3
        while len(d0_keep) < n_regions:
            prev = d0_keep[-1][0]
            d0_keep.append((prev, next_color))
            next_color += 1
        templates.append(d0_keep)

    if n_regions >= 3:
        wipeout = [(0, 1), (1, 2), (0, 2)]
        next_color = 3
        while len(wipeout) < n_regions:
            prev = wipeout[-1][1]
            wipeout.append((prev, next_color))
            next_color += 1
        templates.append(wipeout)

    unique_templates: list[list[tuple[int, int]]] = []
    seen = set()
    for template in templates:
        key = tuple(template)
        if key not in seen:
            seen.add(key)
            unique_templates.append(template)
    return unique_templates


def _select_region_pairs(n_regions: int, marker_pool: list[int], rng: random.Random):
    templates = _region_color_templates(n_regions)
    rng.shuffle(templates)
    for template in templates:
        symbol_count = 1 + max(max(pair) for pair in template)
        if not (DISTINCT_MARKER_COLORS_MIN <= symbol_count <= DISTINCT_MARKER_COLORS_MAX):
            continue
        if symbol_count > len(marker_pool):
            continue
        chosen = rng.sample(marker_pool, symbol_count)
        return [tuple(int(chosen[idx]) for idx in pair) for pair in template]
    return None


def _validate_uniform_markers(grid: np.ndarray, wall: int) -> bool:
    counts = [len(info["dist_to_color"]) for info in _extract_regions(grid, wall)]
    return bool(counts) and len(set(counts)) == 1 and counts[0] == MARKERS_PER_REGION


def _gen_attempt(rng: random.Random):
    h = rng.randint(H_MIN, H_MAX)
    w = rng.randint(W_MIN, W_MAX)
    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(h, w, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None
    if not _validate_area_order(rects):
        return None

    rect_infos = [
        {"rect": rect, "area": _rect_area(rect), "anchor": (rect[0], rect[1])}
        for rect in rects
    ]
    rect_infos.sort(key=lambda info: (int(info["area"]), info["anchor"]))

    region_pairs = _select_region_pairs(len(rect_infos), marker_pool, rng)
    if region_pairs is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, rect_infos, bg, region_pairs, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None
    if not _validate_uniform_markers(grid, wall):
        return None

    combo_out = main(grid)
    if np.array_equal(grid, combo_out):
        return None
    if _consumption_effect_count(grid) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
