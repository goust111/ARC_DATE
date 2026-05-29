# Variant of b9e38dc0
# Dimension: D10 — 多步处理顺序 (multi_step)
# Change: 两步处理：第一步填充半包围区域并辐射，第二步孤立像素直线移动（第一步结果作为障碍物影响第二步）... → 链式反应：辐射触发相邻精灵

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D10 chain-reaction variant
# --------------------------------------------------
# variant_spec:
#   carried_over_geometry:
#     - Each wall sprite keeps the original task's partial-enclosure geometry:
#       one 8-connected wall component with exactly two degree-1 endpoints and
#       one cardinal radiate direction inferred from those endpoints.
#     - The radiated region is still the original "wall + endpoint rays" wedge,
#       and obstacle shadows still cut away fill behind obstacles along the
#       sprite's radiate direction.
#   new_objects_and_trigger:
#     - Instead of one wall sprite, the grid contains 2-3 same-colored wall
#       sprites. Only one sprite contains the actual fill seed blob in the input;
#       this is the start sprite.
#     - Every sprite has a visible opening segment: the bg cells strictly between
#       its two endpoints. When a previously activated sprite's radiated fill
#       reaches any cell of another sprite's opening segment, that second sprite
#       activates.
#     - A triggered sprite uses those reached opening cells as its local seeds,
#       then performs its own fill+shadowed-radiate step. This may activate yet
#       more sprites, so the solver must propagate with a queue rather than a
#       single flood fill.
#   concrete_example:
#     - Original behavior: one upward-facing funnel contains a seed, so the fill
#       floods its pocket and expands to the top border. No second object reacts.
#     - Variant behavior: that first wedge hits the opening row of a second
#       funnel placed higher up. The second funnel had no interior seed in the
#       input, but once its opening is touched, it fills and emits its own wedge,
#       which can hit a third funnel. The visual difference is a staggered,
#       multi-hop spread instead of one independent expansion.
#   complexity_guardrails:
#     - Chain depth is restricted to 2 or 3 activated sprites. That is enough to
#       make the ordering visible without turning the puzzle into an unreadable
#       routing problem.
#     - All activated sprites share the same wall color and fill color. The
#       trigger is therefore geometric and positional only; no hidden ordering by
#       color id is used.
#     - The generator suppresses optional obstacle clutter. The solver still
#       supports obstacle shadows, but the examples prioritize legible chain
#       structure over unrelated noise.
#   old-code_assumption_audit:
#     - The original solver selected exactly one wall component and implicitly
#       assumed all other non-bg pixels were either fill or obstacle. That is no
#       longer safe because multiple wall components must coexist; each component
#       needs its own geometry analysis and its own activation state.
#     - The original flood fill always started from interior fill pixels. In the
#       variant, later sprites start from opening cells that were filled by an
#       earlier sprite, so the solver must accept external opening seeds.
#     - The original shadow phase could directly erase newly filled cells in one
#       global pass. In the variant, each sprite's shadow only limits that
#       sprite's own propagation; earlier fill laid down by previous sprites must
#       remain intact.
#   placement_legality:
#     - Generator placement guarantees that wall components are not 8-touching,
#       all openings stay bg in the input, and each later opening lies inside the
#       predecessor's radiated mask but outside masks from earlier non-parent
#       sprites. That makes the chain causal and solver-detectable.
#   visual_observability:
#     - The start sprite is the only one with an interior seed blob, so the
#       chain origin is obvious.
#     - Later sprites are visibly dormant in the input and visibly become filled
#       in the output only because a predecessor's wedge reaches their openings.
#     - Because the generator validates parent-child activation links, examples
#       do not silently collapse back to "one sprite independently fills itself".


# ----------------------------- shared geometry ------------------------------

GRID_H = (18, 24)
GRID_W = (18, 24)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 9)
DEPTH_HORIZ = (5, 9)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 4
CHAIN_LEN_CHOICES = (2, 3)
GEN_RETRIES = 400
COMPONENT_PLACE_RETRIES = 90
MIN_TOTAL_NEW_FILL = 28
MIN_CHILD_NEW_FILL = 10

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _coords_of_color(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    pts = list(zip(*np.where(grid == color)))
    return [(int(r), int(c)) for r, c in pts]


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, *, tip_12: bool = False) -> list[tuple[int, int]]:
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows: list[int] = []
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


def _build_local_wall(rad: tuple[int, int], h: int, w: int):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 7)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 7)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 2))

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

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _map_points(points: list[tuple[int, int]] | set[tuple[int, int]], rad: tuple[int, int], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    return [_map_local(u, v, rad, anchor) for u, v in points]


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior: list[tuple[int, int]] = []
    for u in range(1, depth):
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
        nbrs = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int) -> tuple[int, int]:
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(1, w - gap - 2)
    if rad == (1, 0):
        return random.randint(depth + 2, h - OPEN_END_MARGIN - 1), random.randint(1, w - gap - 2)
    if rad == (0, -1):
        return random.randint(1, h - gap - 2), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(1, h - gap - 2), random.randint(depth + 2, w - OPEN_END_MARGIN - 1)


def _in_bounds_all(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in points)


def _opening_cells_from_endpoints(endpoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(endpoints) == 2
    (r1, c1), (r2, c2) = sorted(endpoints)
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return [(r1, c) for c in range(lo + 1, hi)]
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        return [(r, c1) for r in range(lo + 1, hi)]
    dr = _sgn(r2 - r1)
    dc = _sgn(c2 - c1)
    out = []
    rr, cc = r1 + dr, c1 + dc
    while (rr, cc) != (r2, c2):
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def _component_info(comp: set[tuple[int, int]], h: int, w: int) -> dict:
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in comp if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) == 2, "expected exactly two endpoints"

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
    tip_12 = [f for _, f in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    assert rad in {(-1, 0), (1, 0), (0, -1), (0, 1)}, "expected cardinal radiate direction"

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

    opening = _opening_cells_from_endpoints(endpoints)
    return {
        "points": frozenset(comp),
        "mask": comp_mask,
        "endpoints": tuple(sorted(endpoints)),
        "opening": tuple(opening),
        "rad": rad,
        "barrier": barrier,
    }


def _reachable_from_seeds(out: np.ndarray, bg: int, fill: int, barrier: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    passable = (~barrier) & ((out == bg) | (out == fill))
    seen = np.zeros_like(passable, dtype=bool)
    q: list[tuple[int, int]] = []
    for r, c in seeds:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and passable[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _apply_shadow(orig: np.ndarray, reach: np.ndarray, *, bg: int, fill: int, wall: int, rad: tuple[int, int]) -> np.ndarray:
    out = reach.copy()
    obstacles = (orig != bg) & (orig != fill) & (orig != wall)
    h, w = orig.shape

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if out[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and out[r, c]:
                out[r, c] = False

    dr, dc = rad
    if dr != 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    else:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    return out


def _simulate_chain(grid: np.ndarray, *, want_details: bool = False):
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = int(get_bg(g))
    vals = [int(v) for v in np.unique(g)]
    non_bg = [v for v in vals if v != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected wall components"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected a fill color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = _coords_of_color(g, color)
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
    fill_pos = _coords_of_color(g, fill)
    if not fill_pos:
        return g if not want_details else (g, {})

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

    comp_infos = [_component_info(comp, h, w) for comp in comps]
    comp_index_by_points = {info["points"]: idx for idx, info in enumerate(comp_infos)}
    start_idx = min(range(len(comps)), key=lambda i: (comp_min_dist(comps[i]), -len(comps[i])))

    out = g.copy()
    activated: set[int] = {start_idx}
    activation_order = [start_idx]
    parent: dict[int, int | None] = {start_idx: None}
    queue: list[tuple[int, list[tuple[int, int]]]] = [(start_idx, fill_pos[:])]
    propagation_masks = [np.zeros((h, w), dtype=bool) for _ in comps]

    while queue:
        idx, seeds = queue.pop(0)
        info = comp_infos[idx]
        reach = _reachable_from_seeds(out, bg, fill, info["barrier"], seeds)
        prop = _apply_shadow(g, reach, bg=bg, fill=fill, wall=wall, rad=info["rad"])
        propagation_masks[idx] = prop
        new_fill = prop & (out == bg)
        out[new_fill] = fill

        for j, other in enumerate(comp_infos):
            if j in activated:
                continue
            hits = [(r, c) for r, c in other["opening"] if prop[r, c]]
            if hits:
                activated.add(j)
                activation_order.append(j)
                parent[j] = idx
                queue.append((j, hits))

    if not want_details:
        return out
    return out, {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "components": comp_infos,
        "component_index_by_points": comp_index_by_points,
        "start_idx": start_idx,
        "activated": activated,
        "activation_order": activation_order,
        "parent": parent,
        "propagation_masks": propagation_masks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate a chain of wall sprites by letting one sprite's radiation ignite another sprite's opening.

    Input structure:
    - bg is the most common color, and the dominant non-bg color forms several separate partial-wall sprites.
    - Only one sprite contains the fill-color seed blob near its closed end; later sprites begin empty but expose visible opening cells between their endpoints.

    Rule:
    - The seeded sprite fills its enclosed region and radiates outward exactly as in the original funnel task, including obstacle shadows.
    - If that radiated fill reaches another sprite's opening, the touched opening cells become that sprite's new seeds; the sprite then fills and radiates too, and this repeats as a chain reaction.

    Key insight:
    - The puzzle is no longer solved by analyzing one funnel in isolation: you must identify the start sprite, track which opening each radiation wedge reaches next, and follow the resulting activation order across multiple sprites.
    """
    return _simulate_chain(grid, want_details=False)


# ------------------------------ generator -----------------------------------

def _make_component(rad: tuple[int, int], h: int, w: int):
    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    interior = _local_interior(local_wall, depth)
    return {
        "rad": rad,
        "wall": local_wall,
        "depth": depth,
        "gap": gap,
        "interior": interior,
        "opening_local": [(0, v) for v in range(1, gap)],
    }


def _mapped_component_cells(comp_def: dict, anchor: tuple[int, int]) -> dict:
    rad = comp_def["rad"]
    wall = _map_points(comp_def["wall"], rad, anchor)
    interior = _map_points(comp_def["interior"], rad, anchor)
    opening = _map_points(comp_def["opening_local"], rad, anchor)
    return {"wall": wall, "interior": interior, "opening": opening}


def _wall_is_separate(existing_walls: set[tuple[int, int]], wall: list[tuple[int, int]]) -> bool:
    for r, c in wall:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in existing_walls:
                    return False
    return True


def _all_bg(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(grid[r, c] == grid[0, 0] or True for r, c in cells)


def _place_start_component(grid: np.ndarray, wall_color: int, fill_color: int):
    h, w = grid.shape
    for _ in range(120):
        rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        comp_def = _make_component(rad, h, w)
        if comp_def is None:
            continue
        anchor = _place_anchor(rad, h, w, comp_def["depth"], comp_def["gap"])
        mapped = _mapped_component_cells(comp_def, anchor)
        footprint = mapped["wall"] + mapped["interior"] + mapped["opening"]
        if not _in_bounds_all(footprint, h, w):
            continue
        if any(grid[r, c] != grid[0, 0] for r, c in footprint):
            continue

        wall_set = set(comp_def["wall"])
        adjacent = [
            p
            for p in comp_def["interior"]
            if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        ]
        seed_pool = [p for p in adjacent if p[0] >= max(1, comp_def["depth"] - SEED_DEPTH_LOOKBACK)]
        if not seed_pool:
            continue
        blob_local = _grow_seed_blob(random.choice(seed_pool), set(comp_def["interior"]))
        blob = _map_points(blob_local, rad, anchor)
        if any((r, c) in mapped["wall"] for r, c in blob):
            continue

        for r, c in mapped["wall"]:
            grid[r, c] = wall_color
        for r, c in blob:
            if grid[r, c] != grid[0, 0]:
                break
            grid[r, c] = fill_color
        else:
            return {
                "rad": rad,
                "anchor": anchor,
                "wall_points": frozenset(mapped["wall"]),
            }

        for r, c in mapped["wall"]:
            grid[r, c] = grid[0, 0]
        for r, c in blob:
            if grid[r, c] == fill_color:
                grid[r, c] = grid[0, 0]
    return None


def _run_positions(mask: np.ndarray, vertical: bool, length: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    if vertical:
        for c in range(w):
            start = 0
            while start < h:
                while start < h and not mask[start, c]:
                    start += 1
                end = start
                while end < h and mask[end, c]:
                    end += 1
                if end - start >= length:
                    for r0 in range(start, end - length + 1):
                        out.append((r0, c))
                start = end + 1
    else:
        for r in range(h):
            start = 0
            while start < w:
                while start < w and not mask[r, start]:
                    start += 1
                end = start
                while end < w and mask[r, end]:
                    end += 1
                if end - start >= length:
                    for c0 in range(start, end - length + 1):
                        out.append((r, c0))
                start = end + 1
    return out


def _anchor_from_opening_start(rad: tuple[int, int], opening_start: tuple[int, int]) -> tuple[int, int]:
    r, c = opening_start
    if rad[0]:
        return r, c - 1
    return r - 1, c


def _candidate_anchors_from_parent_mask(parent_mask: np.ndarray, forbidden_mask: np.ndarray, rad: tuple[int, int], gap: int) -> list[tuple[int, int]]:
    opening_len = gap - 1
    usable = parent_mask & (~forbidden_mask)
    vertical = rad[0] == 0
    starts = _run_positions(usable, vertical, opening_len)
    anchors = [_anchor_from_opening_start(rad, start) for start in starts]
    random.shuffle(anchors)
    return anchors


def _paint_wall(grid: np.ndarray, wall_points: list[tuple[int, int]], color: int) -> None:
    for r, c in wall_points:
        grid[r, c] = color


def _component_index_for_points(details: dict, wall_points: frozenset[tuple[int, int]]) -> int:
    return details["component_index_by_points"][wall_points]


def _try_add_component(grid: np.ndarray, *, wall_color: int, prev_wall_points: frozenset[tuple[int, int]], earlier_wall_points: list[frozenset[tuple[int, int]]]):
    h, w = grid.shape
    current_out, details = _simulate_chain(grid, want_details=True)
    prev_idx = _component_index_for_points(details, prev_wall_points)
    parent_mask = details["propagation_masks"][prev_idx]
    forbidden_mask = np.zeros((h, w), dtype=bool)
    for pts in earlier_wall_points:
        idx = _component_index_for_points(details, pts)
        if idx != prev_idx:
            forbidden_mask |= details["propagation_masks"][idx]

    existing_walls = set()
    for pts in earlier_wall_points:
        existing_walls |= set(pts)

    for _ in range(COMPONENT_PLACE_RETRIES):
        rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        comp_def = _make_component(rad, h, w)
        if comp_def is None:
            continue
        anchors = _candidate_anchors_from_parent_mask(parent_mask, forbidden_mask, rad, comp_def["gap"])
        if not anchors:
            continue
        for anchor in anchors[:20]:
            mapped = _mapped_component_cells(comp_def, anchor)
            footprint = mapped["wall"] + mapped["interior"] + mapped["opening"]
            if not _in_bounds_all(footprint, h, w):
                continue
            if any(grid[r, c] != grid[0, 0] for r, c in footprint):
                continue
            if not _wall_is_separate(existing_walls, mapped["wall"]):
                continue
            opening = mapped["opening"]
            if not all(parent_mask[r, c] and not forbidden_mask[r, c] for r, c in opening):
                continue

            temp = grid.copy()
            _paint_wall(temp, mapped["wall"], wall_color)
            try:
                temp_out, temp_details = _simulate_chain(temp, want_details=True)
            except Exception:
                continue

            new_wall_points = frozenset(mapped["wall"])
            if new_wall_points not in temp_details["component_index_by_points"]:
                continue
            new_idx = _component_index_for_points(temp_details, new_wall_points)
            if new_idx not in temp_details["activated"]:
                continue
            if temp_details["parent"].get(new_idx) != prev_idx:
                continue
            if len(temp_details["activated"]) != len(earlier_wall_points) + 1:
                continue
            child_new_fill = int((temp_details["propagation_masks"][new_idx] & (temp == temp_details["bg"])).sum())
            total_added_vs_current = int((temp_out != current_out).sum())
            if child_new_fill < MIN_CHILD_NEW_FILL or total_added_vs_current < MIN_CHILD_NEW_FILL:
                continue

            _paint_wall(grid, mapped["wall"], wall_color)
            return {"rad": rad, "anchor": anchor, "wall_points": new_wall_points}
    return None


def _gen_attempt():
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    bg, wall_color, fill_color = random.sample(range(10), 3)
    chain_len = random.choice(CHAIN_LEN_CHOICES)

    grid = np.full((h, w), bg, dtype=int)
    start = _place_start_component(grid, wall_color, fill_color)
    if start is None:
        return None

    placed = [start["wall_points"]]
    prev = start["wall_points"]
    for _ in range(chain_len - 1):
        added = _try_add_component(grid, wall_color=wall_color, prev_wall_points=prev, earlier_wall_points=placed)
        if added is None:
            return None
        placed.append(added["wall_points"])
        prev = added["wall_points"]

    try:
        out, details = _simulate_chain(grid, want_details=True)
    except Exception:
        return None

    if len(details["activated"]) != chain_len:
        return None
    new_fill = (out == details["fill"]) & (grid == details["bg"])
    if int(new_fill.sum()) < MIN_TOTAL_NEW_FILL:
        return None
    if not (new_fill[0, :].any() or new_fill[-1, :].any() or new_fill[:, 0].any() or new_fill[:, -1].any()):
        return None
    start_info = details["components"][details["start_idx"]]
    seeds = _coords_of_color(grid, details["fill"])
    base_reach = _reachable_from_seeds(grid.copy(), details["bg"], details["fill"], start_info["barrier"], seeds)
    base_prop = _apply_shadow(grid, base_reach, bg=details["bg"], fill=details["fill"], wall=details["wall"], rad=start_info["rad"])
    base = grid.copy()
    base[base_prop & (base == details["bg"])] = details["fill"]
    if int((out != base).sum()) < MIN_CHILD_NEW_FILL:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid D10 chain-reaction input")
