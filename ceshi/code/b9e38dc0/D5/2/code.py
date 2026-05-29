# Variant of b9e38dc0
# Dimension: D5 — 目标/终止条件 (dst)
# Change: 辐射射线延伸到网格边缘终止；孤立像素同样移动到网格边缘... → 辐射到遇到同色射线终止

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import neighbors4, neighbors8, components_mask, get_bg


# Design analysis
# yaml-like notes:
#   concretization:
#     - 原题只有一个半封闭墙体，因此“射线遇到另一条射线”在输入里不可发生。
#       为了让 D5 可观察，生成器最小化地扩成 2 个同向、同墙色、同填充色的墙体精灵。
#     - 每个精灵仍然是原题的 V/U 型半包围墙、内部小填充种子、外部单像素障碍。
#       差异只放在“开口两端发出的边界射线”的终止条件：原题到边界，本变体到另一精灵的同色射线。
#     - 具体小例子（向上辐射）：
#       左精灵右侧边界线向右上走，右精灵左侧边界线向左上走。
#       原逻辑：两条线都继续到顶边，因此两块外放填充会一直开到网格顶端。
#       新逻辑：两条线第一次落到同一格时，都停在该格，后面的尾部被裁掉，
#       因而两块填充在上方更早收口，输出会比原题少一块明显的顶端填充区域。
#   complexity_calibration:
#     - 只生成 2 个精灵，避免多重碰撞导致视觉杂乱，也足够体现“射线交互”。
#     - 两个精灵共享同一 radiate_dir，阴影扫描仍沿同一轴，保留原题结构。
#     - 每个样例至少 1 个碰撞格，且“碰撞版输出”和“原版到边界输出”必须不同，防止退化。
#   implicit_assumptions_review:
#     - 原 main 里 `wall_comp = min(...)` 依赖“只有一个相关墙体”；变体里必须改成“每个 fill blob 绑定最近墙体”。
#     - 原 `add_ray/add_ray_12` 依赖“射线永不相交”；变体里改成同步推进的 ray simulation，
#       并在撞到已存在射线格时截断旧尾部。
#     - 原阴影逻辑隐含单一辐射方向；生成器因此保证两个精灵同向，solver 也断言所有相关墙体方向一致。
#   placement_legality:
#     - 两个墙体按同一开口高度对齐、沿横向排开，保证不重叠、不开到网格外，并给内侧射线留下相遇空间。
#     - 障碍仍只放在离墙至少 2 格的背景格上，避免被误认成靠墙 fill。
#   observability:
#     - 颜色不参与任何排序；只有几何位置决定哪个墙体与哪个 fill blob 对应、哪两条射线会相撞。
#     - 人眼可直接比较“上方/下方是否一直开到边界”来归纳新规则。
#   cleanliness:
#     - 只保留两层语义：墙体/填充区域/少量障碍；不额外引入新标记。
#     - 两个精灵共享颜色但空间分离，碰撞发生在它们之间的空带里，画面整洁。
#   new_main_assumptions:
#     - 所有相关墙体共享同一个 wall color、fill color、radiate_dir。
#     - 不处理几何意义上的“对角线穿叉但不落同一格”的碰撞；生成器专门构造为同格相遇。
#     - 同一墙体自己的两条边界线不会先互撞；生成器通过间距与朝向保证这一点。


# -- Configuration constants --------------------------------------------------
GRID_H = (11, 18)
GRID_W = (17, 22)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
GAP_VERT = (6, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3
SPRITE_COUNT = 2
SPRITE_BBOX_GAP = (1, 3)
ALIGN_JITTER = 0

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 4
OBSTACLE_IN_FILL_PROB = 0.55
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 18
MIN_COLLISION_DIFF = 6
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


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


def _build_local_wall(h, force_tip_12=None):
    max_depth = min(DEPTH_VERT[1], h - 6)
    if max_depth < DEPTH_VERT[0]:
        return None
    depth = random.randint(DEPTH_VERT[0], max_depth)
    gap = random.randint(*GAP_VERT)

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
    if force_tip_12 is None:
        tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    elif force_tip_12:
        if depth < 2 * max_n_side:
            return None
        tip_12 = True
    else:
        tip_12 = False

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
    return pts, depth, gap, tip_12


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _wall_anchor_bounds(rad, h, w, depth, gap):
    if rad == (-1, 0):
        return OPEN_END_MARGIN, h - depth - 3, 0, w - gap - 1
    if rad == (1, 0):
        return depth + 2, h - 4, 0, w - gap - 1
    if rad == (0, -1):
        return 0, h - gap - 1, OPEN_END_MARGIN, w - depth - 3
    return 0, h - gap - 1, depth + 2, w - 4


def _local_interior(wall, depth):
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


def _extract_scene(g):
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(g == wall, connectivity=8)
    ]
    assert wall_comps, "expected wall components"

    wall_points = [p for comp in wall_comps for p in comp]

    def min_dist_to_wall(color):
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
        return None

    fill_comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(g == fill, connectivity=4)
    ]
    assert fill_comps, "expected fill component(s)"

    def comp_min_dist(seed_comp, wall_comp):
        best_d = 10**9
        for r, c in seed_comp:
            for wr, wc in wall_comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    selected_indices = []
    for seed_comp in fill_comps:
        idx = min(
            range(len(wall_comps)),
            key=lambda i: (comp_min_dist(seed_comp, wall_comps[i]), -len(wall_comps[i])),
        )
        selected_indices.append(idx)
    selected_indices = sorted(set(selected_indices))

    wall_mask = np.zeros_like(g, dtype=bool)
    ray_specs = []
    rads = []

    for comp_id in selected_indices:
        wall_comp = wall_comps[comp_id]
        comp_mask = np.zeros_like(g, dtype=bool)
        for r, c in wall_comp:
            comp_mask[r, c] = True
            wall_mask[r, c] = True

        def comp_neighbors(r, c):
            out = []
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                    out.append((rr, cc))
            return out

        endpoints = sorted((r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1)
        assert len(endpoints) == 2, "expected each wall component to have exactly 2 endpoints"

        def outward_dir(ep):
            n1 = comp_neighbors(*ep)[0]
            d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
            d_use = d1
            is_12 = False
            n2s = [p for p in comp_neighbors(*n1) if p != ep]
            if n2s:
                n2 = n2s[0]
                d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
                if (d1[0] == 0) ^ (d1[1] == 0):
                    if d2[0] != 0 and d2[1] != 0:
                        d_use = d2
                        is_12 = True
            return (-d_use[0], -d_use[1]), is_12

        out_info = [outward_dir(ep) for ep in endpoints]
        out_dirs = [d for d, _ in out_info]
        tip_12 = [flag for _, flag in out_info]
        rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
        assert rad != (0, 0), "expected non-zero radiate direction"
        rads.append(rad)

        for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
            ray_specs.append(
                {
                    "comp_id": comp_id,
                    "start": ep,
                    "dir": d,
                    "is_12": is_12,
                }
            )

    assert ray_specs, "expected at least one ray"
    assert len(set(rads)) == 1, "expected all selected wall components to share one radiate direction"

    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_mask": wall_mask,
        "ray_specs": ray_specs,
        "rad": rads[0],
    }


def _build_barrier_to_edge(scene, shape):
    h, w = shape
    barrier = scene["wall_mask"].copy()
    rad = scene["rad"]

    def add_ray(start, d):
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start, d):
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

    for spec in scene["ray_specs"]:
        if spec["is_12"]:
            add_ray_12(spec["start"], spec["dir"])
        else:
            add_ray(spec["start"], spec["dir"])
    return barrier


def _build_barrier_with_collisions(scene, shape):
    h, w = shape
    barrier = scene["wall_mask"].copy()
    rad = scene["rad"]
    cell_rays = {}
    collision_cells = set()
    collided_rays = set()

    rays = {}
    for rid, spec in enumerate(scene["ray_specs"]):
        rays[rid] = {
            "rid": rid,
            "comp_id": spec["comp_id"],
            "pos": spec["start"],
            "dir": spec["dir"],
            "is_12": spec["is_12"],
            "phase": 0,
            "path": [],
            "index": {},
            "active": True,
        }

    def next_step(ray):
        if not ray["is_12"]:
            return ray["dir"]
        if ray["phase"] % 2 == 0:
            return ray["dir"]
        if rad[0] != 0:
            return (ray["dir"][0], 0)
        return (0, ray["dir"][1])

    def advance_phase(ray):
        if ray["is_12"]:
            ray["phase"] = (ray["phase"] + 1) % 2

    def append_cell(ray, cell):
        if cell in ray["index"]:
            return
        ray["index"][cell] = len(ray["path"])
        ray["path"].append(cell)
        barrier[cell[0], cell[1]] = True
        cell_rays.setdefault(cell, set()).add(ray["rid"])

    def truncate_at_cell(ray, cell):
        if cell not in ray["index"]:
            return
        idx = ray["index"][cell]
        tail = ray["path"][idx + 1 :]
        for old in tail:
            owners = cell_rays.get(old)
            if owners is not None:
                owners.discard(ray["rid"])
                if not owners:
                    cell_rays.pop(old, None)
                    barrier[old[0], old[1]] = False
            ray["index"].pop(old, None)
        ray["path"] = ray["path"][: idx + 1]
        ray["pos"] = cell
        ray["active"] = False

    max_iters = h * w * max(4, len(rays))
    for _ in range(max_iters):
        active_ids = [rid for rid, ray in rays.items() if ray["active"]]
        if not active_ids:
            break

        proposals = {}
        by_cell = {}
        for rid in sorted(active_ids):
            ray = rays[rid]
            step = next_step(ray)
            nxt = (ray["pos"][0] + step[0], ray["pos"][1] + step[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                ray["active"] = False
                continue
            if scene["wall_mask"][nxt[0], nxt[1]]:
                ray["active"] = False
                continue
            proposals[rid] = (nxt, step)
            by_cell.setdefault(nxt, []).append(rid)

        if not proposals:
            break

        handled = set()
        for cell in sorted(by_cell):
            bidders = by_cell[cell]
            existing = cell_rays.get(cell, set())
            involved = {rays[rid]["comp_id"] for rid in bidders}
            involved |= {rays[oid]["comp_id"] for oid in existing}
            if len(involved) < 2:
                continue
            collision_cells.add(cell)
            for oid in list(existing):
                truncate_at_cell(rays[oid], cell)
                collided_rays.add(oid)
            for rid in bidders:
                append_cell(rays[rid], cell)
                rays[rid]["pos"] = cell
                rays[rid]["active"] = False
                collided_rays.add(rid)
                handled.add(rid)

        for rid in sorted(proposals):
            if rid in handled or not rays[rid]["active"]:
                continue
            cell, _step = proposals[rid]
            owners = cell_rays.get(cell)
            if owners:
                owner_comps = {rays[oid]["comp_id"] for oid in owners}
                if any(comp != rays[rid]["comp_id"] for comp in owner_comps):
                    collision_cells.add(cell)
                    for oid in list(owners):
                        truncate_at_cell(rays[oid], cell)
                        collided_rays.add(oid)
                    append_cell(rays[rid], cell)
                    rays[rid]["pos"] = cell
                    rays[rid]["active"] = False
                    collided_rays.add(rid)
                    continue
                rays[rid]["active"] = False
                continue
            append_cell(rays[rid], cell)
            rays[rid]["pos"] = cell
            advance_phase(rays[rid])
    else:
        raise AssertionError("ray simulation did not converge")

    return barrier, {
        "collision_cells": len(collision_cells),
        "collided_rays": len(collided_rays),
    }


def _solve_with_mode(grid, stop_on_collision):
    g = np.array(grid, copy=True)
    scene = _extract_scene(g)
    if scene is None:
        return g, {"collision_cells": 0, "collided_rays": 0}

    if stop_on_collision:
        barrier, stats = _build_barrier_with_collisions(scene, g.shape)
    else:
        barrier = _build_barrier_to_edge(scene, g.shape)
        stats = {"collision_cells": 0, "collided_rays": 0}

    bg = scene["bg"]
    fill = scene["fill"]
    wall = scene["wall"]
    h, w = g.shape

    passable = (~barrier) & ((g == bg) | (g == fill))
    q = scene["fill_pos"][:]
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

    def shadow_process(coords):
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

    dr, dc = scene["rad"]
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    else:
        raise AssertionError("expected cardinal radiate direction")

    stats["new_fill"] = int(np.count_nonzero((out == fill) & (g == bg)))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partial enclosure and let its boundary rays stop when they collide with another sprite's rays.

    Input structure:
    - The background is the most common color. The wall color is the most frequent non-background color and may form two separate funnel-like wall components.
    - A single fill color appears as one small blob inside each wall component. Extra non-background colors are singleton obstacles.

    Rule:
    - Flood-fill from every fill blob through the interior and out through the opening, using the endpoint rays as the outer boundary just like the base task.
    - The only changed termination rule is for those rays: instead of always running to the grid edge, a ray halts at the first cell already reached by a ray from a different wall sprite, and the older ray is truncated back to that meeting cell as well.
    - Obstacles still stay fixed and cast background shadows behind themselves along the shared radiate direction.

    Key insight: The fill is still controlled by wall geometry, but now the decisive stopping event is ray-to-ray interaction between separate sprites rather than the outer frame of the grid.
    """
    out, _stats = _solve_with_mode(grid, stop_on_collision=True)
    return out


def _sample_walls(h, force_tip_12):
    walls = []
    for _ in range(SPRITE_COUNT):
        built = _build_local_wall(h, force_tip_12=force_tip_12)
        if built is None:
            return None
        walls.append(built)
    return walls


def _choose_aligned_anchors(rad, h, w, walls):
    row_los = []
    row_his = []
    col_los = []
    col_his = []
    for _pts, depth, gap, _tip_12 in walls:
        rlo, rhi, clo, chi = _wall_anchor_bounds(rad, h, w, depth, gap)
        row_los.append(rlo)
        row_his.append(rhi)
        col_los.append(clo)
        col_his.append(chi)

    if rad[0] != 0:
        shared_lo = max(row_los)
        shared_hi = min(row_his)
        if shared_lo > shared_hi:
            return None
        for _ in range(40):
            base_r = random.randint(shared_lo, shared_hi)
            widths = [gap + 1 for _pts, _depth, gap, _tip_12 in walls]
            spacings = [random.randint(*SPRITE_BBOX_GAP) for _ in range(SPRITE_COUNT - 1)]
            total_w = sum(widths) + sum(spacings)
            if total_w > w:
                continue
            start_c = random.randint(0, w - total_w)
            anchors = []
            cur_c = start_c
            ok = True
            for i, (_pts, depth, gap, _tip_12) in enumerate(walls):
                _rlo, _rhi, clo, chi = _wall_anchor_bounds(rad, h, w, depth, gap)
                r0 = base_r + random.randint(-ALIGN_JITTER, ALIGN_JITTER)
                if not (_rlo <= r0 <= _rhi):
                    ok = False
                    break
                c0 = cur_c
                if not (clo <= c0 <= chi):
                    ok = False
                    break
                anchors.append((r0, c0))
                cur_c += gap + 1
                if i < len(spacings):
                    cur_c += spacings[i]
            if ok:
                return anchors
        return None

    shared_lo = max(col_los)
    shared_hi = min(col_his)
    if shared_lo > shared_hi:
        return None
    for _ in range(40):
        base_c = random.randint(shared_lo, shared_hi)
        heights = [gap + 1 for _pts, _depth, gap, _tip_12 in walls]
        spacings = [random.randint(*SPRITE_BBOX_GAP) for _ in range(SPRITE_COUNT - 1)]
        total_h = sum(heights) + sum(spacings)
        if total_h > h:
            continue
        start_r = random.randint(0, h - total_h)
        anchors = []
        cur_r = start_r
        ok = True
        for i, (_pts, depth, gap, _tip_12) in enumerate(walls):
            rlo, rhi, _clo, _chi = _wall_anchor_bounds(rad, h, w, depth, gap)
            c0 = base_c + random.randint(-ALIGN_JITTER, ALIGN_JITTER)
            if not (_clo <= c0 <= _chi):
                ok = False
                break
            r0 = cur_r
            if not (rlo <= r0 <= rhi):
                ok = False
                break
            anchors.append((r0, c0))
            cur_r += gap + 1
            if i < len(spacings):
                cur_r += spacings[i]
        if ok:
            return anchors
    return None


def _render_core_grid():
    rad = random.choice([(-1, 0), (1, 0)])
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    force_tip_12 = random.random() < TIP_12_PROB

    walls = _sample_walls(h, force_tip_12)
    if walls is None:
        return None
    anchors = _choose_aligned_anchors(rad, h, w, walls)
    if anchors is None:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]

    grid = np.full((h, w), bg, dtype=int)

    for (local_wall, depth, _gap, _tip_12), anchor in zip(walls, anchors):
        for u, v in local_wall:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] != bg:
                return None
            grid[r, c] = wall_color

        interior = _local_interior(local_wall, depth)
        wall_set = set(local_wall)
        adjacent = [
            p
            for p in interior
            if any(
                (p[0] + dr, p[1] + dc) in wall_set
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
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

    return grid, bg, wall_color, fill_color


def _gen_attempt():
    built = _render_core_grid()
    if built is None:
        return None
    grid, bg, wall_color, fill_color = built

    try:
        out_collision, stats_collision = _solve_with_mode(grid, stop_on_collision=True)
        out_edge, _stats_edge = _solve_with_mode(grid, stop_on_collision=False)
    except Exception:
        return None

    diff_pixels = int(np.count_nonzero(out_collision != out_edge))
    if stats_collision["collision_cells"] < 1:
        return None
    if diff_pixels < MIN_COLLISION_DIFF:
        return None
    if stats_collision["new_fill"] < MIN_NEW_FILL:
        return None

    obstacle_colors = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(
            grid,
            bg,
            wall_color,
            (out_collision == fill_color) & (grid == bg),
        )
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out_collision, stats_collision = _solve_with_mode(grid, stop_on_collision=True)
        out_edge, _stats_edge = _solve_with_mode(grid, stop_on_collision=False)
    except Exception:
        return None

    diff_pixels = int(np.count_nonzero(out_collision != out_edge))
    if stats_collision["collision_cells"] < 1:
        return None
    if diff_pixels < MIN_COLLISION_DIFF:
        return None
    if stats_collision["new_fill"] < MIN_NEW_FILL:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D5 input")
