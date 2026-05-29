# Variant of b9e38dc0
# Dimension: D12 — 精灵数量与空间排列 (meta)
# Change: 多个半包围精灵分布在网格中，各自独立，开口方向可能不同... → 精灵排列成规则网格

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D12 variant
# ----------------------------------
# source_task: None
#
# a) Concrete behavior and trace
#    - The original code effectively solves one wall + one seed. It finds the wall color,
#      picks the fill color closest to that wall color, then chooses the single closest wall
#      component to all fill pixels and computes one radiate direction from that component.
#    - The variant needs several independent wall/seed sprites, but the key visible change is
#      not merely "more objects": the objects must sit on a regular lattice. I implement a
#      2x2 array because it fits comfortably inside the original size regime while still making
#      the spatial rule obvious at a glance.
#    - The four sprites share one wall prototype and one fill prototype per instance. Their
#      openings are tied to array position:
#        top-left -> up, top-right -> right, bottom-left -> left, bottom-right -> down.
#      This is a small finite pattern that repeats support within one sample and is easy to
#      compare across 2-4 ARC examples.
#    - Small trace of the changed step:
#      old flow: all fill pixels are pooled -> nearest wall component is selected once ->
#      one flood/radiate/shadow pass happens.
#      new flow: split fill pixels into 4-connected blobs -> pair each blob to its nearest wall
#      component -> infer one radiate direction per pair -> run the same flood/radiate/shadow
#      logic component-wise -> union the four results.
#
# b) Complexity calibration
#    - Four sprites are enough that the regular placement is unmistakable, but still small
#      enough to avoid heavy overlap and generator brittleness.
#    - Openings are limited to the four cardinal directions in a fixed outward pattern. This
#      keeps the number of branches low and repeated.
#    - All sprites in one grid reuse the same wall geometry and seed geometry, only rotated.
#      That prevents the scene from becoming noisy and makes the D12 change visually dominant.
#
# c) Hidden assumptions in the original main() that no longer hold
#    - "wall_comp = min(comps, key=...)" assumes exactly one relevant wall component. This fails
#      once multiple sprites exist because only one of them would be solved.
#    - A single global "rad" assumes one opening direction for the whole task. In the variant,
#      each sprite has its own direction, so shadowing must also be applied per sprite.
#    - "passable" previously allowed all fill-colored cells; with multiple seeds, that would let
#      one sprite's search leak through another sprite's seed pixels. The new solver only treats
#      the current seed component as passable seed material.
#
# d) Placement legality
#    - The generator constructs one compact wall prototype (17-25 pixels, 2 endpoints), rotates
#      it, and places the four copies into four non-touching cells of a 2x2 layout.
#    - Each cell is larger than the rotated sprite bbox, so no overlap or border clipping occurs.
#      Cell-facing placement keeps the opening near the correct outer border.
#    - Obstacles are placed only on background cells, away from wall pixels, and optionally in
#      the nominal fill region, so the solver stays unambiguous.
#
# e) Visual observability
#    - The D12 change is visible directly in the input: the walls are no longer scattered, they
#      form a neat array. The output reinforces the same structure because each copy radiates
#      toward the border determined by its lattice position.
#    - No invisible ordering by color id is used. Pairing is geometric only.
#
# f) Visual cleanliness
#    - The array uses four outward-facing directions, so the radiated regions mainly expand into
#      separate quadrants and do not pile up near the center.
#    - Obstacles are sparse; they add shadows without cluttering the lattice.
#
# g) Color inference
#    - The wall color is still the most common non-background color.
#    - The fill color is the closest non-wall color to the walls because every seed blob is
#      4-connected and adjacent to its wall, while obstacles are kept farther away.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _coords_to_set(comp: np.ndarray) -> set[tuple[int, int]]:
    return set((int(r), int(c)) for r, c in comp)


def _component_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in a:
        for rr, cc in b:
            d = abs(r - rr) + abs(c - cc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _comp_neighbors(comp: set[tuple[int, int]], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for rr, cc in neighbors8(r, c):
        if (rr, cc) in comp:
            out.append((rr, cc))
    return out


def _endpoints(comp: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, c) for r, c in comp if len(_comp_neighbors(comp, r, c)) == 1]


def _outward_dir(comp: set[tuple[int, int]], ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    n1 = _comp_neighbors(comp, *ep)[0]
    d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
    d_use = d1
    is_12 = False
    n2s = [p for p in _comp_neighbors(comp, *n1) if p != ep]
    if n2s:
        n2 = n2s[0]
        d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
        if (d1[0] == 0) ^ (d1[1] == 0):
            if d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
    return (-d_use[0], -d_use[1]), is_12


def _component_radiate(comp: set[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]], list[bool]]:
    endpoints = _endpoints(comp)
    assert len(endpoints) >= 2, "expected each wall component to have two endpoints"
    out_info = [_outward_dir(comp, ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    return rad, endpoints, tip_12


def _solve_one_component(
    g: np.ndarray,
    bg: int,
    wall_mask: np.ndarray,
    fill: int,
    obstacles: np.ndarray,
    wall_comp: set[tuple[int, int]],
    seed_comp: set[tuple[int, int]],
) -> np.ndarray:
    h, w = g.shape
    rad, endpoints, tip_12 = _component_radiate(wall_comp)
    out_dirs = [_outward_dir(wall_comp, ep)[0] for ep in endpoints]

    barrier = np.array(wall_mask, copy=True)

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

    seed_mask = np.zeros_like(g, dtype=bool)
    for r, c in seed_comp:
        seed_mask[r, c] = True

    passable = (~barrier) & ((g == bg) | seed_mask)
    q = list(seed_comp)
    seen = np.zeros_like(g, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True

    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)

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
                filled_mask[r, c] = False

    if rad != (0, 0):
        dr, dc = rad
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

    return filled_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve a regular array of half-enclosure sprites by filling each one independently.

    Input structure:
    - The background is the most common color. One wall color forms several separate U/funnel-like wall components arranged on a regular 2x2 lattice.
    - A shared fill color appears as one blob inside each wall component. Any remaining non-background colors are singleton obstacle pixels on the background.

    Rule:
    - For each fill blob, flood-fill the background region reachable through its own wall interior, then continue radiating outward through that wall's opening.
    - Obstacles stay fixed and erase the filled trail behind them, but only along the radiate direction of the affected wall component.

    Key insight:
    - The scene is no longer a single enclosure. Pair every fill blob with its nearest wall component, infer that component's opening from its endpoints, and reuse the original ray-bounded fill logic component by component.
    """
    g = np.array(grid, copy=True)
    vals = np.unique(g)
    bg = get_bg(g)

    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    wall_comps = [_coords_to_set(comp) for comp in components_mask(wall_mask, connectivity=8)]
    assert wall_comps, "expected at least one wall component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in wall_comps for p in comp]

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
    fill_comps = [_coords_to_set(comp) for comp in components_mask(g == fill, connectivity=4)]
    if not fill_comps:
        return g

    obstacles = (g != bg) & (g != fill) & (g != wall)
    final_fill = np.zeros_like(g, dtype=bool)

    unused_walls = set(range(len(wall_comps)))
    ordered_fill_comps = sorted(fill_comps, key=lambda comp: (-len(comp), min(comp)))
    for fill_comp in ordered_fill_comps:
        candidate_ids = sorted(unused_walls) if unused_walls else list(range(len(wall_comps)))
        wall_idx = min(
            candidate_ids,
            key=lambda i: (_component_distance(fill_comp, wall_comps[i]), -len(wall_comps[i])),
        )
        unused_walls.discard(wall_idx)
        local_fill = _solve_one_component(g, bg, wall_mask, fill, obstacles, wall_comps[wall_idx], fill_comp)
        final_fill |= local_fill

    out = np.array(g, copy=True)
    out[final_fill] = fill
    return out


# --------------- generator ---------------

# Prototype wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
PROTOTYPE_DEPTH = (5, 6)
PROTOTYPE_GAP = (5, 6)
TIP_12_PROB = 0.5

# Regular 2x2 layout
OUTER_MARGIN = 2
CELL_GAP = 2
CELL_PAD = 1

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 4
OBSTACLE_IN_FILL_PROB = 0.7
OBSTACLE_MIN_SEP = 2

# Validation
PALETTE_SIZE = (3, 5)
MIN_NEW_FILL = 28
GEN_RETRIES = 400


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[tuple[int, int]]:
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


def _build_prototype_wall() -> tuple[set[tuple[int, int]], int, int]:
    depth = random.randint(*PROTOTYPE_DEPTH)
    gap = random.randint(*PROTOTYPE_GAP)

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    else:
        left_bottom, right_bottom = 2, 4

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, WALL_PIXELS_MIN - base_count)
    max_extra = min(max_extra_l + max_extra_r, WALL_PIXELS_MAX - base_count)
    if min_extra > max_extra:
        raise RuntimeError("invalid prototype wall budget")

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
            ) >= 2
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
            raise RuntimeError("failed to pad prototype wall")

    assert WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX
    assert len(_degree1_pixels(pts)) == 2
    return pts, depth + 1, gap + 1


def _rotate_points(points: set[tuple[int, int]], height: int, width: int, rad: tuple[int, int]) -> tuple[set[tuple[int, int]], int, int]:
    if rad == (-1, 0):
        return set(points), height, width
    if rad == (0, 1):
        return {(c, height - 1 - r) for r, c in points}, width, height
    if rad == (1, 0):
        return {(height - 1 - r, width - 1 - c) for r, c in points}, height, width
    return {(width - 1 - c, r) for r, c in points}, width, height


def _local_interior(wall: set[tuple[int, int]], height: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, height - 1):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        u, v = frontier.pop(random.randrange(len(frontier)))
        neighbors = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        for q in random.sample(neighbors, len(neighbors)):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _choose_seed_blob(proto_wall: set[tuple[int, int]], proto_h: int) -> set[tuple[int, int]]:
    interior = _local_interior(proto_wall, proto_h)
    interior_set = set(interior)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in proto_wall for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, proto_h - 1 - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        raise RuntimeError("no valid seed pool")
    return _grow_seed_blob(random.choice(seed_pool), interior_set)


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid: np.ndarray, bg: int, wall_color: int, nominal_fill_zone: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    fill_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
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


def _count_touched_borders(mask: np.ndarray) -> int:
    return sum(
        [
            bool(mask[0, :].any()),
            bool(mask[-1, :].any()),
            bool(mask[:, 0].any()),
            bool(mask[:, -1].any()),
        ]
    )


def _regular_cell_origins(cell_h: int, cell_w: int) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    h = 2 * cell_h + CELL_GAP + 2 * OUTER_MARGIN
    w = 2 * cell_w + CELL_GAP + 2 * OUTER_MARGIN
    top = OUTER_MARGIN
    left = OUTER_MARGIN
    bottom = OUTER_MARGIN + cell_h + CELL_GAP
    right = OUTER_MARGIN + cell_w + CELL_GAP
    return (top, left), (bottom, right), h, w


def _anchor_in_cell(
    cell_top: int,
    cell_left: int,
    cell_h: int,
    cell_w: int,
    sprite_h: int,
    sprite_w: int,
    rad: tuple[int, int],
) -> tuple[int, int]:
    extra_r = cell_h - sprite_h
    extra_c = cell_w - sprite_w
    assert extra_r >= 0 and extra_c >= 0
    if rad == (-1, 0):
        return cell_top, cell_left + extra_c // 2
    if rad == (1, 0):
        return cell_top + extra_r, cell_left + extra_c // 2
    if rad == (0, -1):
        return cell_top + extra_r // 2, cell_left
    return cell_top + extra_r // 2, cell_left + extra_c


def _render_points(grid: np.ndarray, points: set[tuple[int, int]], anchor: tuple[int, int], color: int) -> None:
    r0, c0 = anchor
    for r, c in points:
        grid[r0 + r, c0 + c] = color


def _gen_attempt() -> np.ndarray | None:
    try:
        proto_wall, proto_h, proto_w = _build_prototype_wall()
        proto_seed = _choose_seed_blob(proto_wall, proto_h)
    except RuntimeError:
        return None

    cell_h = max(proto_h, proto_w) + CELL_PAD
    cell_w = max(proto_h, proto_w) + CELL_PAD
    (top, left), (bottom, right), h, w = _regular_cell_origins(cell_h, cell_w)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]
    grid = np.full((h, w), bg, dtype=int)

    layout = [
        (top, left, (-1, 0)),
        (top, right, (0, 1)),
        (bottom, left, (0, -1)),
        (bottom, right, (1, 0)),
    ]

    for cell_top, cell_left, rad in layout:
        wall_pts, sprite_h, sprite_w = _rotate_points(proto_wall, proto_h, proto_w, rad)
        seed_pts, _, _ = _rotate_points(proto_seed, proto_h, proto_w, rad)
        anchor = _anchor_in_cell(cell_top, cell_left, cell_h, cell_w, sprite_h, sprite_w, rad)
        _render_points(grid, wall_pts, anchor, wall_color)
        for r, c in seed_pts:
            rr = anchor[0] + r
            cc = anchor[1] + c
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = fill_color

    # Validate the intended D12 structure before adding obstacles.
    wall_comps = components_mask(grid == wall_color, connectivity=8)
    fill_comps = components_mask(grid == fill_color, connectivity=4)
    if len(wall_comps) != 4 or len(fill_comps) != 4:
        return None

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None
    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if _count_touched_borders(new_fill) < 3:
        return None

    # Re-check the visible structure after obstacle placement.
    if len(components_mask(grid == wall_color, connectivity=8)) != 4:
        return None
    if len(components_mask(grid == fill_color, connectivity=4)) != 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D12 input")
