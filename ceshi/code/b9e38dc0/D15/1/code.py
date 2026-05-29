# Variant of b9e38dc0
# Dimension: D15 — 填充颜色规则 (transform)
# Change: 洪水填充使用源像素（内部像素）的颜色填满整个半包围区域... → 棋盘格填充

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D15 variant
#
# a) Concrete behavior:
#    The original task finds one wall component, infers the opening direction,
#    flood-fills all reachable background cells from the interior seed, then
#    trims straight shadows behind obstacles. The D15 change only alters how the
#    newly reached cells inside the semi-enclosed pocket are colored. Instead of
#    painting that pocket solid with the seed color, we keep the outward radiate
#    zone solid and recolor only the enclosed interior into a checkerboard of
#    {fill, bg}. The checkerboard phase is anchored to the seed parity so the
#    seed still looks like the source of the pattern. Example: if the seed is at
#    (8,7), then every interior cell with (r+c)%2 matching (8+7)%2 becomes fill
#    and the alternating interior cells stay bg. The rays past the opening stay
#    solid fill, so the input/output pairs visibly separate "interior pattern"
#    from "outward radiation".
#
# b) Complexity calibration:
#    A fully free checkerboard phase would be too unconstrained. The variant
#    therefore uses exactly one rule: global grid parity anchored by the seed.
#    This is visually strong, repeats many times inside one example, and does
#    not introduce extra latent categories. To keep the rule legible, generator
#    seeds are forced to be singleton pixels; multi-pixel 4-connected blobs
#    would necessarily occupy both parities and would blur the checkerboard cue.
#
# c) Implicit-assumption audit against the original solver:
#    - `filled_mask = (seen & (g == bg))` still means "all newly reached cells".
#      That remains valid; the change is only in the writeback step.
#    - `out[filled_mask] = fill` was safe before because every reached cell used
#      the same color. In the variant this would be wrong for interior cells, so
#      we now split the reached set into `interior_mask`, `checker_fill_mask`,
#      and `checker_bg_mask`.
#    - Shadow logic originally mutated `filled_mask` after painting. That is
#      still useful, but checkerboarding must happen after shadow removal so
#      blocked interior fill cells stay bg rather than being repainted.
#    - The original code never distinguished the inside pocket from the outward
#      radiate zone. The variant requires that distinction, so we build a second
#      flood fill with an added "mouth seal" between the two wall endpoints.
#
# d) Placement legality:
#    The wall geometry, margins, endpoint gap, and obstacle spacing stay the
#    same. The only generator-side compatibility change is singleton seeds plus
#    stricter validation that the interior checkerboard and the outer ray are
#    both large enough to be visible. Obstacles are still placed only on bg
#    cells and never overlap wall/seed pixels.
#
# e) Visual observability:
#    The rule is human-visible because the pocket shows alternating fill/bg
#    cells while the emitted ray remains a solid stripe/funnel. This contrast is
#    visible in 2-3 examples without relying on color ids, counts, or hidden
#    ordering. The phase is tied to the seed location, a visible cue.
#
# f) Visual cleanliness:
#    The pattern is restricted to one region layer only. Walls, obstacles, the
#    seed, the checkerboard pocket, and the solid ray remain visually separable.
#    The generator rejects cramped examples where the pocket is too small to
#    show multiple alternating cells.
#
# g) Color inference:
#    Only two colors participate in the new transform inside the pocket: the
#    existing fill color and the visible background color. No invisible color-id
#    based mapping is introduced. The output therefore exposes a structural
#    alternation, not an arbitrary recoloring.
#
# 9) New assumptions introduced by the modified solver and how generator
#    guarantees them:
#    - The opening endpoints must be axis-aligned so the mouth can be sealed by
#      one straight row/column segment. The original generator already produces
#      that structure.
#    - The fill seed should be a singleton so the parity anchor is unambiguous.
#      Generator enforces this via `SEED_SIZE_WEIGHTS = [1]`.
#    - The interior must contain enough cells on both parities to make the
#      checkerboard inferable. Generator validates minimum visible counts for
#      both checker colors and for the outer solid ray.
# ---------------------------------------------------------------------------


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _seal_opening(mask: np.ndarray, endpoints: list[tuple[int, int]], rad: tuple[int, int]) -> np.ndarray:
    """Close the wall mouth with one straight segment between the two endpoints."""
    sealed = mask.copy()
    if len(endpoints) < 2:
        return sealed

    if rad[0] != 0:
        row = endpoints[0][0]
        c0 = min(c for _, c in endpoints)
        c1 = max(c for _, c in endpoints)
        sealed[row, c0 : c1 + 1] = True
        return sealed

    if rad[1] != 0:
        col = endpoints[0][1]
        r0 = min(r for r, _ in endpoints)
        r1 = max(r for r, _ in endpoints)
        sealed[r0 : r1 + 1, col] = True
        return sealed

    ep0, ep1 = endpoints[:2]
    rr, cc = ep0
    sealed[rr, cc] = True
    while rr != ep1[0]:
        rr += _sgn(ep1[0] - rr)
        sealed[rr, cc] = True
    while cc != ep1[1]:
        cc += _sgn(ep1[1] - cc)
        sealed[rr, cc] = True
    return sealed


def _flood_mask(g: np.ndarray, passable: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    """4-neighbor flood mask from all start cells over a boolean passable mask."""
    q = starts[:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _checker_masks(shape: tuple[int, int], interior_mask: np.ndarray, anchor: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split the interior into checkerboard fill-vs-bg cells using seed parity."""
    rows, cols = np.indices(shape)
    parity = (rows + cols) & 1
    anchor_parity = (anchor[0] + anchor[1]) & 1
    checker_fill = interior_mask & (parity == anchor_parity)
    checker_bg = interior_mask & ~checker_fill
    return checker_fill, checker_bg


def _compute_fill_state(grid: np.ndarray) -> dict:
    """Shared geometric analysis for both solver and generator validation."""
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color (fill color)"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))

    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        empty = np.zeros_like(g, dtype=bool)
        return {
            "grid": g,
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "fill_pos": [],
            "rad": (0, 0),
            "filled_mask": empty,
            "interior_mask": empty,
            "checker_fill_mask": empty,
            "checker_bg_mask": empty,
        }

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))

    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall component to have endpoints (degree-1 pixels)"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            if step % 2 == 1:
                rr += d[0]
                cc += d[1]
            else:
                if rad[0] != 0:
                    rr += d[0]
                else:
                    cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = _flood_mask(g, passable, fill_pos)
    filled_mask = seen & (g == bg)

    sealed_barrier = _seal_opening(barrier, endpoints, rad)
    interior_seen = _flood_mask(g, (~sealed_barrier) & ((g == bg) | (g == fill)), fill_pos)
    interior_mask = interior_seen & (g == bg)

    anchor = min(fill_pos)
    checker_fill_mask, checker_bg_mask = _checker_masks(g.shape, interior_mask, anchor)

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "rad": rad,
        "filled_mask": filled_mask,
        "interior_mask": interior_mask,
        "checker_fill_mask": checker_fill_mask,
        "checker_bg_mask": checker_bg_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a wall-bounded pocket from its seed and radiate outward through the opening.

    Input structure:
    - The most common color is the background, and the most frequent non-background color is a partial wall with one open side.
    - A non-wall seed color sits inside the pocket; other isolated non-background pixels may be obstacles.

    Rule:
    - Flood the reachable area from the seed, but color the enclosed interior pocket as a checkerboard alternating between the seed color and the background color.
    - Past the opening, the outward radiating region stays solid seed-colored, and obstacles still cast background shadows behind them along the radiate direction.

    Key insight: The same geometry as the original task identifies the filled region, but sealing the mouth separates "inside the enclosure" from "outside the opening", letting only the interior switch to a seed-anchored checkerboard.
    """
    state = _compute_fill_state(grid)
    g = state["grid"]
    fill_pos = state["fill_pos"]
    if not fill_pos:
        return g

    filled_mask = state["filled_mask"].copy()
    out = g.copy()
    out[filled_mask] = state["fill"]

    obstacles = (g != state["bg"]) & (g != state["fill"]) & (g != state["wall"])

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = state["bg"]
                filled_mask[r, c] = False

    if state["rad"] != (0, 0):
        dr, dc = state["rad"]
        h, w = g.shape
        if dr != 0 and dc == 0:
            for c in range(w):
                rows = range(h) if dr > 0 else range(h - 1, -1, -1)
                shadow_process([(r, c) for r in rows])
        elif dc != 0 and dr == 0:
            for r in range(h):
                cols = range(w) if dc > 0 else range(w - 1, -1, -1)
                shadow_process([(r, c) for c in cols])
        else:
            if dr == dc:
                for k in range(-(w - 1), h):
                    coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)
            else:
                for k in range(h + w - 1):
                    coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)

    checker_fill = state["checker_fill_mask"] & filled_mask
    checker_bg = state["checker_bg_mask"] & filled_mask
    out[checker_fill] = state["fill"]
    out[checker_bg] = state["bg"]
    return out


# --------------- generator ---------------

# Canvas size bounds
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
OPEN_END_MARGIN = 3

# Fill seed
SEED_SIZE_WEIGHTS = [1]  # singleton seeds keep checkerboard phase visible
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_TOTAL_FILLED = 20
MIN_VISIBLE_FILL = 12
MIN_INTERIOR_CELLS = 10
MIN_CHECKER_FILL = 4
MIN_CHECKER_BG = 4
MIN_RAY_FILL = 6
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


def _degree1_pixels(points):
    """Return degree-1 pixels (exactly one 8-neighbor in the set)."""
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
    """Build one zigzag arm from (0, start_v) toward (depth, target_v)."""
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows = []
    if n_side_steps:
        if tip_12:
            chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
            extra_steps = 0
        else:
            chosen_rows = [1]
            if n_side_steps > 1:
                chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
            chosen_rows.sort()

    first_row = chosen_rows[0] if chosen_rows else 1
    extra_rows = [r for r in chosen_rows if r != first_row]
    random.shuffle(extra_rows)
    orthogonal_rows = set(extra_rows[:extra_steps])

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
    return pts


def _build_local_wall(rad, h, w):
    """Construct wall shape in local (u, v) coordinates. Returns (pts, depth, gap) or None."""
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 6)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        depth = random.randint(DEPTH_HORIZ[0], min(DEPTH_HORIZ[1], w - 6))
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 1))

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    elif gap == 6:
        left_bottom, right_bottom = 2, 4
    else:
        left_bottom, right_bottom = random.choice([(2, 5), (3, 4)])

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, WALL_PIXELS_MIN - base_count)
    max_extra = min(max_extra_l + max_extra_r, WALL_PIXELS_MAX - base_count)
    if min_extra > max_extra:
        return None

    extra = random.randint(min_extra, max_extra)
    extra_l = random.randint(max(0, extra - max_extra_r), min(max_extra_l, extra))
    extra_r = extra - extra_l

    max_n_side = max(left_bottom, gap - right_bottom)
    tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    left = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    while len(pts) < WALL_PIXELS_MIN:
        candidates = [
            (u, v)
            for u in range(max(0, depth - 3), depth + 1)
            for v in range(gap + 1)
            if (u, v) not in pts
            and sum(
                (uu, vv) in pts
                for uu in range(u - 1, u + 2)
                for vv in range(v - 1, v + 2)
                if (uu, vv) != (u, v)
            )
            >= 2
        ]
        random.shuffle(candidates)
        added = False
        for p in candidates:
            trial = pts | {p}
            if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u, v, rad, anchor):
    """Map local (u, v) to grid (r, c) given radiate direction and anchor."""
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    """Choose anchor so the wall fits within margins."""
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth, gap):
    """Return bg cells strictly between the two wall arms in local coords."""
    by_row = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed, interior_set):
    """Grow a small 4-connected blob from seed within interior cells."""
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        p = frontier.pop(random.randrange(len(frontier)))
        u, v = p
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _obstacle_fits(pos, placed):
    """Check that pos is at least Chebyshev OBSTACLE_MIN_SEP from all placed obstacles."""
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
    """Find bg cells far enough from wall for obstacle placement, using distance transform."""
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    """Place up to n_obstacles on the grid, preferring some in the fill zone."""
    placed = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _touches_border(mask):
    """Check if any True cell in mask lies on the grid border."""
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _gen_attempt():
    """Single attempt to generate a valid input grid."""
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth, gap)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_state = _compute_fill_state(grid)
    except Exception:
        return None
    nominal_fill_zone = nominal_state["filled_mask"]

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        state = _compute_fill_state(grid)
        out = main(grid)
    except Exception:
        return None

    visible_fill = (out == state["fill"]) & (grid == bg)
    checker_fill_visible = state["checker_fill_mask"] & (out == state["fill"])
    checker_bg_visible = state["checker_bg_mask"] & (out == state["bg"])
    ray_fill_visible = (state["filled_mask"] & ~state["interior_mask"]) & (out == state["fill"])

    if int(state["filled_mask"].sum()) < MIN_TOTAL_FILLED:
        return None
    if int(visible_fill.sum()) < MIN_VISIBLE_FILL or not _touches_border(visible_fill):
        return None
    if int(state["interior_mask"].sum()) < MIN_INTERIOR_CELLS:
        return None
    if int(checker_fill_visible.sum()) < MIN_CHECKER_FILL:
        return None
    if int(checker_bg_visible.sum()) < MIN_CHECKER_BG:
        return None
    if int(ray_fill_visible.sum()) < MIN_RAY_FILL:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 input")
