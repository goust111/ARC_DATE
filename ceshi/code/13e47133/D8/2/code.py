# Variant of 13e47133
# Dimension: D8 — 调色板构建规则 (meta)
# Change: 按标记到边界的切比雪夫距离排序，距离>0时在前面补背景色；调色板循环使用... → 多区域共享全局调色板

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the D8 variant
# a) Concrete behavior and trace:
#    The wall layout and the region-distance geometry stay unchanged: the grid is
#    still cut into 3-6 non-wall regions, and each region is filled by bands based
#    on Chebyshev distance to that region's boundary. The changed step is palette
#    construction. Original behavior: each region inspects only its own markers,
#    sorts them by local boundary depth, optionally prepends the base color, then
#    cycles that local palette inside the region. Variant behavior: markers from
#    all regions are read once in row-major order to build one shared palette, and
#    every region uses that same palette. Example: if the top-left region contains
#    markers [7, 3] and a lower region contains marker [4], the original solver
#    paints the first region with [7, 3] (or [bg, 7, 3]) and the second region
#    with [4] (or [bg, 4]); the variant paints both regions with [7, 3, 4].
# b) Complexity calibration:
#    If every region had the same marker sequence, the new rule would be hard to
#    distinguish from the old one. The generator therefore forces at least two
#    marker-bearing regions and uses globally unique marker colors, so the global
#    palette visibly differs from any local palette in most regions.
# c) Hidden assumptions in the original main():
#    - `dist_to_color` assumed palette membership is local to a component.
#    - Sorting by distance assumed marker depth determines palette order.
#    - `if ds[0] > 0: palette.append(base)` assumed each region may need its own
#      base-prefixed palette.
#    These assumptions are invalid after the variant change, so the solver must
#    collect marker colors globally before processing region fill.
# d) Placement legality:
#    Markers are still placed on consecutive diagonal depths from a rectangle
#    corner, so they stay isolated, in bounds, and on valid non-wall cells.
#    Rectangular recursive splits guarantee regions remain disjoint and solver
#    boundary distances are well-defined.
# e) Visual observability:
#    The palette order is based on spatial reading order, never on color ids.
#    Because the same palette reappears across several separated regions, a human
#    can infer that information must be pooled globally rather than locally.
# f) Visual cleanliness:
#    Only walls, base cells, and isolated single-pixel markers appear in the
#    input. Output remains the same concentric-band structure as the source task,
#    so regions are easy to parse.
# g) Color-rule inferability:
#    Marker colors are globally distinct and limited to 3-6 per sample. This
#    keeps the shared palette short enough to notice, while repeated use across
#    multiple regions provides enough evidence for the global rule.


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


def _bg_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    return int(bg), int(wall)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint wall-separated regions with concentric bands using one palette shared by the whole grid.

    Input structure:
    - Wall-colored line segments partition the canvas into several non-wall regions.
    - Regions are mostly base-colored, but isolated marker pixels of other colors can appear in different regions.

    Rule:
    - Gather all marker colors across the entire grid in row-major order, keeping the first occurrence of each color to form one global palette.
    - For every non-wall region, compute each cell's Chebyshev distance to that region boundary.
    - Fill every region with the same cyclic palette via palette[distance mod palette_length]; if no markers exist, fall back to the base color.

    Key insight: Marker pixels no longer define separate local palettes. They jointly define a single global color sequence that must be collected first and then reused for each region's distance pattern.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg, wall = _bg_wall(g)
    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    region_data: list[tuple[np.ndarray, np.ndarray]] = []
    markers: list[tuple[int, int, int]] = []

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        region_data.append((pts, dist))

        for r, c in pts:
            v = int(g[r, c])
            if v != base:
                markers.append((int(r), int(c), v))

    markers.sort(key=lambda x: (x[0], x[1]))
    palette: list[int] = []
    seen = set()
    for _r, _c, color in markers:
        if color not in seen:
            seen.add(color)
            palette.append(int(color))
    if not palette:
        palette = [bg]

    m = len(palette)
    for pts, dist in region_data:
        for r, c in pts:
            d = int(dist[r, c])
            out[r, c] = palette[d % m]

    return out


def _main_local_baseline(grid: np.ndarray) -> np.ndarray:
    """Original per-region palette rule, kept only to reject degenerate generator samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _bg, wall = _bg_wall(g)

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
        pts_sorted = sorted((int(r), int(c)) for r, c in pts)
        for r, c in pts_sorted:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        palette: list[int] = []
        if dist_to_color:
            ds = sorted(dist_to_color)
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            out[r, c] = palette[int(dist[r, c]) % m]

    return out


# Generator-side guarantees for the variant solver:
# - `main()` treats every non-base cell inside a region as a marker, so generator
#   keeps markers isolated and preserves a single dominant base color in each region.
# - The shared palette is based on first row-major appearance, so generator uses
#   globally distinct marker colors to keep that order unambiguous and visible.
# - The rule is only interesting when multiple regions contribute, so generator
#   enforces at least two marker-bearing regions and rejects samples where the new
#   output matches the old local-palette baseline.
# - `main()` infers bg/wall from global frequencies, so generator preserves bg as
#   the overall mode and wall as the second-most common color.

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Marker placement ----
MARKERS_PER_REGION_MAX = 3
GLOBAL_MARKERS_MIN, GLOBAL_MARKERS_MAX = 3, 6
START_DIST_MAX = 1

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
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
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


def _marker_slots(rect, n: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if n <= 0 or n > upper:
        return None

    max_start = min(START_DIST_MAX, max_d - (n - 1))
    if max_start < 0:
        return None
    start_d = rng.randint(0, max_start)

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    slots = []
    for k in range(n):
        d = start_d + k
        rr = br + dr * d
        cc = bc + dc * d
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            return None
        slots.append((d, rr, cc))
    return slots


def _plan_marker_counts(rects, rng: random.Random):
    uppers = [min(MARKERS_PER_REGION_MAX, _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)) for r0, c0, r1, c1 in rects]
    candidates = [i for i, upper in enumerate(uppers) if upper >= 1]
    if len(candidates) < 2:
        return None

    max_total = min(GLOBAL_MARKERS_MAX, sum(uppers[i] for i in candidates))
    if max_total < GLOBAL_MARKERS_MIN:
        return None

    active_n = rng.randint(2, min(len(candidates), max_total))
    active = rng.sample(candidates, active_n)
    min_total = max(GLOBAL_MARKERS_MIN, active_n)
    total = rng.randint(min_total, max_total)

    counts = [0] * len(rects)
    for idx in active:
        counts[idx] = 1
    extra = total - active_n

    while extra > 0:
        expandable = [idx for idx in active if counts[idx] < uppers[idx]]
        if not expandable:
            return None
        idx = rng.choice(expandable)
        counts[idx] += 1
        extra -= 1

    return counts


def _stamp_markers(grid, rects, bg, marker_pool, rng: random.Random):
    counts = _plan_marker_counts(rects, rng)
    if counts is None:
        return None

    planned = []
    active_regions = 0
    for idx, rect in enumerate(rects):
        slots = _marker_slots(rect, counts[idx], rng) if counts[idx] else []
        if slots is None:
            return None
        if slots:
            active_regions += 1
        for d, rr, cc in slots:
            if grid[rr, cc] != bg:
                return None
            planned.append((rr, cc, idx, d))

    if active_regions < 2:
        return None
    if not (GLOBAL_MARKERS_MIN <= len(planned) <= min(GLOBAL_MARKERS_MAX, len(marker_pool))):
        return None

    planned.sort(key=lambda x: (x[0], x[1]))
    palette = rng.sample(marker_pool, len(planned))
    region_colors = {}
    used = set()
    for color, (rr, cc, idx, _d) in zip(palette, planned):
        grid[rr, cc] = color
        used.add(color)
        region_colors.setdefault(idx, []).append(color)

    if len(region_colors) < 2:
        return None

    local_palettes = {tuple(cols) for cols in region_colors.values()}
    if len(local_palettes) < 2:
        return None

    return used


def _validate_color_order(grid, bg, wall):
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

    used = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (GLOBAL_MARKERS_MIN <= len(used) <= GLOBAL_MARKERS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        baseline = _main_local_baseline(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, baseline):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
