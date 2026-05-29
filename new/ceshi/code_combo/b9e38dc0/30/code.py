# Combo variant of b9e38dc0 (index 30)
# combo_spec: D2/2+D16/1
# Name: 部分精灵无像素+反方向移动

import itertools
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the fused variant
# a) Concrete behavior:
#    Each wall component still behaves like the original half-enclosure: its seed fills the
#    inside and radiates out of the opening. The new part is a second sprite color used only
#    for optional singleton pixels behind the closed end. If a wall has that singleton, the
#    output moves it straight toward the border in direction -radiate_dir; if not, nothing
#    moves for that wall. Example: two upward-radiating walls side by side can both emit the
#    same fill plume, but only the left wall may also send a lone pixel downward to the
#    bottom border, while the right wall has no such extra motion.
# b) Complexity calibration:
#    The generator always places 2-3 walls and forces a mixed presence pattern: at least one
#    wall has a singleton and at least one does not. All walls in one grid share orientation,
#    which keeps the rule readable while still repeating the conditional logic in one sample.
# c) Broken assumptions from the original main():
#    The base code assumed one wall component, one fill seed color region, and no need to
#    pair optional movers. That no longer holds. The new solver therefore works per wall
#    component, computes a separate radiate direction for each, matches seeds by minimum
#    total distance, and looks for optional singleton pixels in the wall's closed-side lane.
# d) Placement legality:
#    Walls are packed side by side or stacked with clear gaps, seed blobs stay inside their
#    own enclosure, optional singleton pixels are placed only on background cells behind the
#    matching closed end, and grid margins guarantee room for the reverse-direction motion.
# e) Visual observability:
#    The output difference is directly visible: some walls produce only plumes, while others
#    also produce an extra border-reaching line on the opposite side. Across examples the
#    global orientation changes, so the singleton motion is inferable as "opposite to the
#    plume" rather than "always downward" or similar.
# f) Clean visuals:
#    The generator uses just four semantic colors (bg, wall, fill, mover) and avoids
#    obstacles, keeping the output legible even with multiple walls.
# g) New solver assumptions reviewed:
#    - every wall has one fill blob -> guaranteed by generator
#    - mover pixels are singletons on the closed side of at most one wall -> guaranteed
#    - there is at least one mover and at least one wall without one -> guaranteed
#    - wall remains the most frequent non-bg color, fill is the closest non-wall color to a
#      wall, mover is any remaining color -> guaranteed by color counts and placement


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _component_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            d = abs(r0 - r1) + abs(c0 - c1)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _analyze_wall_component(comp: set[tuple[int, int]], shape: tuple[int, int]) -> dict:
    h, w = shape
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [p for p in comp if len(comp_neighbors(*p)) == 1]
    assert len(endpoints) == 2, "expected exactly two degree-1 wall endpoints"

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
    assert rad != (0, 0), "expected a cardinal radiate direction"

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

    closed_dir = (-rad[0], -rad[1])
    closed_proj_max = max(r * closed_dir[0] + c * closed_dir[1] for r, c in comp)
    if rad[0] != 0:
        perp_vals = [c for _, c in comp]
    else:
        perp_vals = [r for r, _ in comp]
    perp_lo = min(perp_vals)
    perp_hi = max(perp_vals)
    perp_mid = (perp_lo + perp_hi) / 2.0
    return {
        "comp": comp,
        "barrier": barrier,
        "rad": rad,
        "closed_dir": closed_dir,
        "closed_proj_max": closed_proj_max,
        "perp_lo": perp_lo,
        "perp_hi": perp_hi,
        "perp_mid": perp_mid,
    }


def _assign_fill_components(
    wall_infos: list[dict], fill_comps: list[set[tuple[int, int]]]
) -> dict[int, set[tuple[int, int]]]:
    assert len(fill_comps) >= len(wall_infos), "not enough fill components for the walls"
    best_perm = None
    best_score = None
    indices = range(len(fill_comps))
    for perm in itertools.permutations(indices, len(wall_infos)):
        score = 0
        for wi, fi in enumerate(perm):
            score += _component_distance(wall_infos[wi]["comp"], fill_comps[fi])
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return {wi: fill_comps[fi] for wi, fi in enumerate(best_perm)}


def _pick_mover_for_wall(
    wall_info: dict,
    mover_pixels: list[tuple[int, int, int]],
    used: set[int],
) -> tuple[int, int, int] | None:
    best = None
    for idx, r, c in mover_pixels:
        if idx in used:
            continue
        proj = r * wall_info["closed_dir"][0] + c * wall_info["closed_dir"][1]
        if proj <= wall_info["closed_proj_max"]:
            continue
        perp = c if wall_info["rad"][0] != 0 else r
        if not (wall_info["perp_lo"] - 1 <= perp <= wall_info["perp_hi"] + 1):
            continue
        closed_gap = proj - wall_info["closed_proj_max"]
        offset = abs(perp - wall_info["perp_mid"])
        dist = min(abs(r - wr) + abs(c - wc) for wr, wc in wall_info["comp"])
        score = (closed_gap, offset, dist)
        if best is None or score < best[0]:
            best = (score, (idx, r, c))
    return None if best is None else best[1]


def _shadow_process(out: np.ndarray, filled_mask: np.ndarray, obstacles: np.ndarray, rad: tuple[int, int]) -> None:
    h, w = out.shape

    def scan(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = get_bg(out)
                filled_mask[r, c] = False

    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            scan([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            scan([(r, c) for c in cols])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread each seed through its matching half-enclosure and optionally launch a separate singleton away from the enclosure.

    Input structure:
    - The background is the most common color, and one wall color forms several disjoint U/V-like partial enclosures with a shared orientation.
    - One fill-colored blob sits inside every wall near its closed end.
    - A second non-wall color may appear as isolated singleton pixels behind the closed end of only some walls.

    Rule:
    - For every wall component, flood-fill the reachable interior/background with the fill color and radiate it outward through the opening, using the same endpoint-derived wedge as in the base task.
    - If a wall has a matching singleton on its closed side, move that singleton straight in the direction opposite the radiation until the grid edge, clearing its old path back to background.

    Key insight:
    - The wall endpoints still reveal each component's opening direction, but now that direction must be used twice: once for the fill plume and once, conditionally, reversed to drive the optional singleton pixel.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected non-background colors"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    wall_comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
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
    mover_colors = [c for c in other_colors if c != fill]

    fill_comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(g == fill, connectivity=4)]
    assert fill_comps, "expected at least one fill component"

    wall_infos = [_analyze_wall_component(comp, g.shape) for comp in wall_comps]
    wall_infos.sort(
        key=lambda info: (
            min(r for r, _ in info["comp"]),
            min(c for _, c in info["comp"]),
        )
    )
    fill_assignment = _assign_fill_components(wall_infos, fill_comps)

    mover_pixels: list[tuple[int, int, int]] = []
    mover_masks = []
    for color in mover_colors:
        for pts in components_mask(g == color, connectivity=4):
            pts_list = [(int(r), int(c)) for r, c in pts]
            if len(pts_list) == 1:
                mover_masks.append((color, pts_list[0]))
    for idx, (color, pos) in enumerate(mover_masks):
        mover_pixels.append((idx, pos[0], pos[1]))

    out = g.copy()
    used_movers: set[int] = set()

    obstacle_mask = np.zeros_like(g, dtype=bool)
    for color in mover_colors:
        obstacle_mask |= g == color

    for wi, wall_info in enumerate(wall_infos):
        seed_comp = fill_assignment[wi]
        seed_mask = np.zeros_like(g, dtype=bool)
        for r, c in seed_comp:
            seed_mask[r, c] = True

        passable = (~wall_info["barrier"]) & ((g == bg) | seed_mask)
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
        out[filled_mask] = fill

        extra_obstacles = obstacle_mask & (~filled_mask)
        if extra_obstacles.any():
            _shadow_process(out, filled_mask.copy(), extra_obstacles, wall_info["rad"])

        mover = _pick_mover_for_wall(wall_info, mover_pixels, used_movers)
        if mover is not None:
            idx, r, c = mover
            used_movers.add(idx)
            color = int(g[r, c])
            rr, cc = r, c
            out[rr, cc] = bg
            while True:
                nr = rr + wall_info["closed_dir"][0]
                nc = cc + wall_info["closed_dir"][1]
                if not (0 <= nr < h and 0 <= nc < w):
                    out[rr, cc] = color
                    break
                rr, cc = nr, nc
                out[rr, cc] = bg

    return out


# --------------- generator ---------------

SPRITE_COUNT = (2, 3)
OPEN_END_MARGIN = 3
CLOSED_END_MARGIN = 5
LANE_GAP = (3, 5)
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3
GEN_RETRIES = 300

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 9)
DEPTH_HORIZ = (5, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5
TRACER_OFFSET = 2


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


def _build_local_wall(rad, span_cap):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], span_cap - OPEN_END_MARGIN - CLOSED_END_MARGIN - 1)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], span_cap - OPEN_END_MARGIN - CLOSED_END_MARGIN - 1)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
        gap = random.randint(*GAP_HORIZ)

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

    if len(_degree1_pixels(pts)) != 2:
        return None
    return {
        "wall": pts,
        "depth": depth,
        "gap": gap,
        "bridge_center": (left_bottom + right_bottom) // 2,
    }


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


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
        u, v = frontier.pop(random.randrange(len(frontier)))
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _choose_sprite_count(rad):
    if rad[0]:
        return random.choice([2, 2, 3])
    return random.choice([2, 2, 3])


def _build_sprite_specs(rad):
    n = _choose_sprite_count(rad)
    span_cap = 19 if rad[0] else 20
    sprites = []
    for _ in range(n):
        spec = _build_local_wall(rad, span_cap)
        if spec is None:
            return None
        sprites.append(spec)
    return sprites


def _compute_grid_shape(rad, sprites):
    lane_sizes = [s["gap"] + 1 for s in sprites]
    lane_gaps = [random.randint(*LANE_GAP) for _ in range(len(sprites) - 1)]
    lane_total = sum(lane_sizes) + sum(lane_gaps)
    max_depth = max(s["depth"] for s in sprites)
    span = OPEN_END_MARGIN + max_depth + CLOSED_END_MARGIN + 1
    if rad[0]:
        h = span + random.randint(0, 2)
        w = lane_total + random.randint(2, 5)
    else:
        h = lane_total + random.randint(2, 5)
        w = span + random.randint(0, 2)
    return h, w, lane_gaps


def _anchors_for_layout(rad, sprites, shape, lane_gaps):
    h, w = shape
    anchors = []
    if rad == (-1, 0):
        row0 = OPEN_END_MARGIN
        col = random.randint(1, max(1, w - (sum(s["gap"] + 1 for s in sprites) + sum(lane_gaps)) - 1))
        for i, s in enumerate(sprites):
            anchors.append((row0, col))
            col += s["gap"] + 1
            if i < len(lane_gaps):
                col += lane_gaps[i]
    elif rad == (1, 0):
        row0 = h - 1 - OPEN_END_MARGIN
        col = random.randint(1, max(1, w - (sum(s["gap"] + 1 for s in sprites) + sum(lane_gaps)) - 1))
        for i, s in enumerate(sprites):
            anchors.append((row0, col))
            col += s["gap"] + 1
            if i < len(lane_gaps):
                col += lane_gaps[i]
    elif rad == (0, -1):
        col0 = OPEN_END_MARGIN
        row = random.randint(1, max(1, h - (sum(s["gap"] + 1 for s in sprites) + sum(lane_gaps)) - 1))
        for i, s in enumerate(sprites):
            anchors.append((row, col0))
            row += s["gap"] + 1
            if i < len(lane_gaps):
                row += lane_gaps[i]
    else:
        col0 = w - 1 - OPEN_END_MARGIN
        row = random.randint(1, max(1, h - (sum(s["gap"] + 1 for s in sprites) + sum(lane_gaps)) - 1))
        for i, s in enumerate(sprites):
            anchors.append((row, col0))
            row += s["gap"] + 1
            if i < len(lane_gaps):
                row += lane_gaps[i]
    return anchors


def _tracer_local(spec):
    return spec["depth"] + TRACER_OFFSET, spec["bridge_center"]


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    sprites = _build_sprite_specs(rad)
    if sprites is None:
        return None

    h, w, lane_gaps = _compute_grid_shape(rad, sprites)
    anchors = _anchors_for_layout(rad, sprites, (h, w), lane_gaps)

    if h < 10 or w < 10:
        return None

    palette = random.sample(range(10), 4)
    bg, wall_color, fill_color, mover_color = palette
    grid = np.full((h, w), bg, dtype=int)

    mover_indices = list(range(len(sprites)))
    random.shuffle(mover_indices)
    n_movers = random.randint(1, len(sprites) - 1)
    mover_set = set(mover_indices[:n_movers])

    for idx, (spec, anchor) in enumerate(zip(sprites, anchors)):
        for u, v in spec["wall"]:
            r, c = _map_local(u, v, rad, anchor)
            if not (0 <= r < h and 0 <= c < w) or grid[r, c] != bg:
                return None
            grid[r, c] = wall_color

        interior = _local_interior(spec["wall"], spec["depth"])
        wall_set = set(spec["wall"])
        adjacent = [
            p
            for p in interior
            if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        ]
        seed_pool = [p for p in adjacent if p[0] >= max(1, spec["depth"] - SEED_DEPTH_LOOKBACK)]
        if not seed_pool:
            return None
        blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
        for u, v in blob:
            r, c = _map_local(u, v, rad, anchor)
            if not (0 <= r < h and 0 <= c < w) or grid[r, c] != bg:
                return None
            grid[r, c] = fill_color

        if idx in mover_set:
            u, v = _tracer_local(spec)
            r, c = _map_local(u, v, rad, anchor)
            if not (0 <= r < h and 0 <= c < w) or grid[r, c] != bg:
                return None
            grid[r, c] = mover_color

    try:
        out = main(grid)
    except Exception:
        return None

    moved = (grid == mover_color) & (out != mover_color)
    landed = (out == mover_color) & (grid != mover_color)
    if int(moved.sum()) < 1 or int(landed.sum()) < 1:
        return None

    fill_growth = (out == fill_color) & (grid == bg)
    if int(fill_growth.sum()) < 18:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo b9e38dc0 input")
