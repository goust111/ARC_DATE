# Variant of b9e38dc0
# Dimension: D8 — 变换——填充后精灵的变化 (transform)
# Change: flood_fill变换——半包围区域内部被源像素颜色填满，源像素扩展为整个区域的填充色... → 填充后精灵边框被擦除

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior and trace
#    The input keeps the original ingredients: one dominant wall color forming a single
#    half-enclosure, one seed-color blob inside it near the closed end, and optional
#    obstacle pixels elsewhere. The output still does the same two geometric actions:
#    fill the interior and radiate outward through the opening, then remove shadows
#    behind obstacles along the radiate direction.
#    The variant changes only the last visible step: after the flood/radiation result
#    has been finalized, erase the wall component that defined the enclosure. Example:
#    if the original output would be
#      wall wall wall
#      wall fill wall
#      bg   fill bg
#    the variant first computes that same state, then turns the wall cells to bg:
#      bg bg bg
#      bg fill bg
#      bg fill bg
#    The radiating fill and obstacle shadows stay unchanged; only the frame disappears.
#
# b) Complexity calibration
#    The new signal is visually strong because 17-25 wall pixels vanish, which is larger
#    than the 1-6 seed blob and comparable to the filled interior. We keep the original
#    limited freedom: one wall sprite, one fill color, 0-5 singleton obstacles, four
#    radiate directions, and the two arm-angle modes. That preserves variety without
#    creating extra branches that would obscure the new rule.
#
# c) Implicit-assumption audit against the original main()
#    - Wall identification by "most frequent non-bg color" still works because the wall
#      is erased only after all detection is complete.
#    - Fill-color identification by distance-to-wall still works because the generator
#      still places the seed blob adjacent to the interior wall.
#    - The barrier mask must still include the wall even though the wall disappears in
#      the final output; erasing earlier would let the flood leak through the former
#      border and would change the task, not just the rendering.
#    - Only the selected wall component should be erased. Erasing every pixel of the wall
#      color would add an unstated assumption that no unrelated component shares that color.
#    - Shadow processing must happen before erasure; shadows depend on the true filled
#      area, not on the post-erasure appearance.
#
# d) Placement legality
#    The generator keeps the original wall-construction and anchor-placement logic, so the
#    wall still fits in-bounds, has exactly two endpoints, and leaves enough open-end
#    margin for radiation to reach a border. Obstacles are still placed only on bg cells
#    away from the wall, so they never overlap the erased border in the input.
#
# e) Visual observability
#    A solver looking at 2-4 examples can see that the fill behaves like the original
#    puzzle, but the wall outline is gone in every output. This is visible without using
#    color ids or hidden ordering: the same enclosing shape present in the input is absent
#    in the output while the fill remains exactly where that shape would have constrained it.
#
# f) Visual cleanliness
#    The output is actually cleaner than the original because one whole layer disappears.
#    Obstacles remain sparse singletons, so removing the border does not create ambiguous
#    color overlaps. The filled mass and rays become the only large foreground structure.
#
# g) Color change inferability
#    There is no new color remapping. The wall simply becomes bg after the fill is done.
#    Because the removed cells are exactly the input wall geometry, the bg conversion is
#    supported by strong spatial cues and is easy to infer from few examples.
#
# h) Generator-side anti-degeneration check
#    The generator validates that the new output differs from the legacy output exactly on
#    the chosen wall component, and that at least WALL_PIXELS_MIN cells are erased. This
#    guarantees the D8 behavior is active on generated samples rather than accidentally
#    producing an output identical to the base task.
#
# i) New assumptions introduced by this variant
#    The solver now assumes that the wall component associated with the seed remains a
#    coherent component mask that can be erased at the end. The generator guarantees this
#    by preserving the original single-wall construction and by never placing wall-colored
#    distractor pixels.


def _solve_core(
    grid: np.ndarray,
    *,
    erase_wall: bool,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
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

    def sgn(x: int) -> int:
        return (x > 0) - (x < 0)

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color (fill color)"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = list(zip(*np.where(g == color)))
        pts = [(int(r), int(c)) for r, c in pts]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill_stats = {c: min_dist_to_wall(c) for c in other_colors}
    fill = min(other_colors, key=lambda c: (fill_stats[c][0], fill_stats[c][1]))

    fill_pos = list(zip(*np.where(g == fill)))
    fill_pos = [(int(r), int(c)) for r, c in fill_pos]
    if not fill_pos:
        out = g.copy()
        meta = {
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "wall_comp_mask": np.zeros_like(g, dtype=bool),
            "filled_mask": np.zeros_like(g, dtype=bool),
            "rad": (0, 0),
        }
        return (out, meta) if return_meta else out

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

    endpoints: list[tuple[int, int]] = []
    for r, c in wall_comp:
        if len(comp_neighbors(r, c)) == 1:
            endpoints.append((r, c))

    assert endpoints, "expected wall component to have endpoints (degree-1 pixels)"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (sgn(n1[0] - ep[0]), sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (sgn(n2[0] - n1[0]), sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (sgn(dr_sum), sgn(dc_sum))

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int], skip: int = 0) -> None:
        rr, cc = start
        for _ in range(skip):
            rr += d[0]
            cc += d[1]
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
            add_ray(ep, d, skip=0)

    passable = (~barrier) & ((g == bg) | (g == fill))
    q = fill_pos[:]
    seen = np.zeros_like(passable, dtype=bool)
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

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)

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
                out[r, c] = bg
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

    if erase_wall:
        out[comp_mask] = bg

    meta = {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "wall_comp_mask": comp_mask,
        "filled_mask": filled_mask.copy(),
        "rad": rad,
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the region defined by a partial wall sprite, radiate through its opening, then remove the wall outline.

    Input structure:
    - bg is the most common color. The dominant non-bg color is a single partial enclosure with two endpoints.
    - A seed blob of another color sits inside that enclosure near its closed end; other non-bg pixels are obstacles.

    Rule:
    - Use the wall endpoints to infer the opening direction, flood-fill the reachable interior/radiate zone with the seed color, and apply obstacle shadows along that same direction.
    - After the fill result is complete, erase the wall component itself to background, leaving only the filled mass, rays, and unchanged obstacles.

    Key insight: The wall still controls where fill may travel, but in the final rendering that same wall vanishes, so the solver must infer the hidden frame first and erase it only after using it as a barrier.
    """
    return _solve_core(grid, erase_wall=True)


# --------------- generator (o5x, D8 variant) ---------------

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
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


def _degree1_pixels(points):
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
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
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth, gap):
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
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
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
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _gen_attempt():
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
        p for p in interior
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
        nominal_out = _solve_core(grid, erase_wall=True)
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
        legacy_out, legacy_meta = _solve_core(grid, erase_wall=False, return_meta=True)
        out, meta = _solve_core(grid, erase_wall=True, return_meta=True)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    erased_wall_mask = legacy_meta["wall_comp_mask"]
    diff_mask = out != legacy_out
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if int(erased_wall_mask.sum()) < WALL_PIXELS_MIN:
        return None
    if int(diff_mask.sum()) != int(erased_wall_mask.sum()):
        return None
    if not np.array_equal(diff_mask, erased_wall_mask):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D8-variant input")
