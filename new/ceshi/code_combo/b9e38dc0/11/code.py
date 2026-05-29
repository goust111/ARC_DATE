# Combo variant of b9e38dc0 (index 11)
# combo_spec: D9/0+D11/2
# Name: 裁剪输出+无孤立像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the fused variant
# a) Behavior trace:
#    The original core mechanic already has two distinct phases: first build a wall-defined
#    funnel and flood/radiate the seed color through the opening, then post-process the
#    filled region using extra non-wall pixels as blockers. This combo removes that second
#    phase entirely and keeps only the wall + seed + flood/radiate step. The output is then
#    cropped to the minimal rectangle that contains every non-background cell of the result.
# b) Complexity calibration:
#    Keeping the 1:1 versus 1:2 wall-arm geometry preserves meaningful variability in the
#    radiating wedge, while removing blockers avoids mixed mechanisms. Cropping supplies a
#    second visible change that is structural rather than color-id based.
# c) Hidden assumptions to revisit:
#    - "Most frequent non-bg color is the wall" becomes safer because inputs only contain
#      wall and fill colors.
#    - "Closest other color to the wall is the fill seed" still holds and is now unique.
#    - The old obstacle-shadow scan assumed distractor pixels exist; that premise is false
#      here, so the whole scan must be deleted rather than disabled piecemeal.
#    - Cropping introduces a new requirement: the solved grid must have a non-empty active
#      region and should not occupy the full canvas, so the generator must enforce margins.
# d) Placement legality:
#    The generator reuses the original local-wall construction because it already guarantees
#    a single 8-connected wall with exactly two endpoints. Anchor placement is tightened so
#    the activity fits fully inside the input while still leaving removable blank borders.
# e) Visual observability:
#    Each example now contains only three roles: background, wall, and fill seed. Humans can
#    read the rule directly from the geometry: the seed expands through the wall opening and
#    the final answer discards surrounding empty space.
# f) Visual cleanliness:
#    No obstacle overlays remain, so the filled wedge is continuous and easy to parse. The
#    cropped output isolates the active sprite cluster instead of leaving large blank margins.
# g) Inferability:
#    No branch depends on the numeric identity of colors. Wall and fill are recovered from
#    relative frequency and proximity, which are visible structural cues across examples.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    return [set((int(r), int(c)) for r, c in pts) for pts in components_mask(mask, connectivity=8)]


def _pick_wall_and_fill(g: np.ndarray, bg: int) -> tuple[int, int]:
    vals = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(vals) >= 2, "expected wall and fill colors"

    wall = max(vals, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    wall_points = [(int(r), int(c)) for r, c in np.argwhere(wall_mask)]
    assert wall_points, "expected wall pixels"

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        best = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                dist = abs(r - wr) + abs(c - wc)
                if dist < best:
                    best = dist
                    if best == 0:
                        return 0, len(pts)
        return best, len(pts)

    other = [c for c in vals if c != wall]
    fill = min(other, key=min_dist_to_wall)
    return wall, fill


def _solve_uncropped(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    wall, fill = _pick_wall_and_fill(g, bg)

    fill_pos = [(int(r), int(c)) for r, c in np.argwhere(g == fill)]
    assert fill_pos, "expected fill seed"

    wall_mask = g == wall
    comps = _components(wall_mask)
    assert comps, "expected at least one wall component"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                dist = abs(r - wr) + abs(c - wc)
                if dist < best:
                    best = dist
                    if best == 0:
                        return 0
        return best

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        pts = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                pts.append((rr, cc))
        return pts

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    def outward_dir(endpoint: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*endpoint)[0]
        d1 = (_sgn(n1[0] - endpoint[0]), _sgn(n1[1] - endpoint[1]))
        d_use = d1
        is_12 = False

        n2s = [p for p in comp_neighbors(*n1) if p != endpoint]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(endpoint) for endpoint in endpoints]
    out_dirs = [direction for direction, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(dr for dr, _ in out_dirs)), _sgn(sum(dc for _, dc in out_dirs)))
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
            elif rad[0] != 0:
                rr += d[0]
            else:
                cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for endpoint, direction, is_12 in zip(endpoints, out_dirs, tip_12, strict=True):
        if is_12:
            add_ray_12(endpoint, direction)
        else:
            add_ray(endpoint, direction)

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = np.zeros_like(passable, dtype=bool)
    stack = []
    for r, c in fill_pos:
        if passable[r, c] and not seen[r, c]:
            seen[r, c] = True
            stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))

    out = g.copy()
    out[seen & (g == bg)] = fill
    return out


def _crop_to_activity(grid: np.ndarray, bg: int) -> np.ndarray:
    active = np.argwhere(grid != bg)
    assert active.size, "expected non-background activity"
    r0, c0 = active.min(axis=0)
    r1, c1 = active.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the seed color through a wall opening and return only the active sprite area.

    Input structure:
    - The background is the most common color.
    - The wall is the dominant non-background color and forms one funnel or U-like 8-connected component.
    - A smaller non-wall color appears inside the enclosure as the fill seed; there are no extra obstacle pixels in this variant.

    Rule:
    - Use the wall endpoints to infer the opening direction and extend two boundary rays outward from them.
    - Flood-fill all background cells reachable from the seed inside that ray-bounded region, preserving the wall and original seed cells.
    - After the fill/radiate step is complete, crop the output to the smallest rectangle containing every non-background pixel.

    Key insight:
    - The task keeps only the wall-guided radiation from the original puzzle; the second-stage blocker logic is gone, so the only extra work beyond the fill is finding the exact non-background bounding box for the cropped answer.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    solved = _solve_uncropped(g)
    return _crop_to_activity(solved, bg)


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
SIDE_MARGIN = 2
CLOSED_END_MARGIN = 2

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Validation
MIN_NEW_FILL = 20
MIN_ROWS_REMOVED = 2
MIN_COLS_REMOVED = 2
GEN_RETRIES = 500


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(
    depth: int,
    start_v: int,
    target_v: int,
    extra_steps: int,
    *,
    tip_12: bool = False,
) -> list[tuple[int, int]]:
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
    extra_rows = [row for row in chosen_rows if row != first_row]
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
        max_depth = min(DEPTH_VERT[1], h - 6)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 6)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
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


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int) -> tuple[int, int] | None:
    if rad == (-1, 0):
        r_lo = OPEN_END_MARGIN
        r_hi = h - depth - CLOSED_END_MARGIN - 1
        c_lo = SIDE_MARGIN
        c_hi = w - gap - SIDE_MARGIN - 1
    elif rad == (1, 0):
        r_lo = depth + CLOSED_END_MARGIN
        r_hi = h - OPEN_END_MARGIN - 1
        c_lo = SIDE_MARGIN
        c_hi = w - gap - SIDE_MARGIN - 1
    elif rad == (0, -1):
        r_lo = SIDE_MARGIN
        r_hi = h - gap - SIDE_MARGIN - 1
        c_lo = OPEN_END_MARGIN
        c_hi = w - depth - CLOSED_END_MARGIN - 1
    else:
        r_lo = SIDE_MARGIN
        r_hi = h - gap - SIDE_MARGIN - 1
        c_lo = depth + CLOSED_END_MARGIN
        c_hi = w - OPEN_END_MARGIN - 1

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
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


def _grow_seed_blob(seed: tuple[int, int], interior: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        u, v = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        for q in random.sample(nbrs, len(nbrs)):
            if q in interior and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built

    anchor = _place_anchor(rad, h, w, depth, gap)
    if anchor is None:
        return None

    bg, wall_color, fill_color = random.sample(range(10), 3)
    grid = np.full((h, w), bg, dtype=int)

    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        if not (0 <= r < h and 0 <= c < w):
            return None
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth)
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
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        full_out = _solve_uncropped(grid)
        cropped = main(grid)
    except Exception:
        return None

    new_fill = (full_out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if not _touches_border(new_fill):
        return None

    rows_removed = h - cropped.shape[0]
    cols_removed = w - cropped.shape[1]
    if rows_removed < MIN_ROWS_REMOVED or cols_removed < MIN_COLS_REMOVED:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo b9e38dc0 input")
