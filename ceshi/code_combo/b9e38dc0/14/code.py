# Combo variant of b9e38dc0 (index 14)
# combo_spec: D3/3+D11/0
# Name: 孤立像素轨迹涂边框色+颜色匹配关联
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    - Keep the original half-enclosure flood/radiate mechanic, but now place two wall sprites
#      in one grid. Each sprite has its own wall color and its own interior seed color.
#    - Add one isolated exterior pixel per sprite. Its color equals that sprite's interior seed color.
#      In the output that isolated pixel slides straight along the sprite's opening direction until
#      it reaches the grid edge. The moved pixel keeps its own color, but every vacated trail cell is
#      repainted with the matched sprite's wall color rather than background.
#    - To make the color-pair rule necessary, the generator swaps the exterior placement lanes:
#      the seed-colored pixel for the left/top sprite is placed near the right/bottom sprite, and vice
#      versa. A nearest-sprite heuristic therefore fails on purpose.
# b) Complexity calibration:
#    - Exactly two sprites are used. That is enough to show repeated color-pair evidence without
#      creating a crowded three-way matching puzzle.
#    - The palette is limited to five colors: background, two wall colors, and two seed/marker colors.
#      This keeps the mapping legible for humans.
# c) Assumption audit against the original solver:
#    - "Most frequent non-bg color is the wall" no longer holds because there are two different walls.
#      The new solver detects wall components geometrically: monochrome 8-connected components of
#      17-25 pixels with exactly two degree-1 endpoints.
#    - "The unique non-wall color closest to the wall is the fill color" no longer holds because the
#      same color also appears as an exterior isolated marker. The new solver identifies the seed blob
#      by restricting attention to the wall component's own bounding box.
#    - "All fill-colored cells can seed BFS" no longer holds because an exterior marker shares the seed
#      color. The new solver seeds the flood only from the in-bbox seed blob.
# d) Placement legality:
#    - The generator places the two walls in separated slots and validates that their nominal radiated
#      fill zones do not overlap. Exterior marker lanes are searched only on clear background paths that
#      stay out of every wall and every fill zone.
# e) Visual observability:
#    - Each sample visibly contains two seed colors repeated twice: once inside a wall and once as an
#      isolated exterior dot. The trail color is a third cue that points back to the correct wall color.
#      Because placements are swapped, the matching cannot be read from proximity alone.
# f) Visual cleanliness:
#    - Marker paths are chosen on lanes that avoid the filled wedges, so border-colored trails remain
#      crisp instead of overwriting the flood output.
# g) Inferability of the color change:
#    - Only one rule changes the trail color: "marker color matches interior seed color, trail color
#      becomes that matched sprite's border color". With two paired examples in every grid, the mapping
#      is repeated enough for a human solver to infer.

from __future__ import annotations

import random
import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25

GRID_H_VERT = (16, 18)
GRID_H_HORIZ = (17, 18)
GRID_W_VERT = (18, 19)
GRID_W_HORIZ = (16, 19)

DEPTH_VERT = (5, 8)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (5, 6)
GAP_HORIZ = (5, 6)
TIP_12_PROB = 0.5
OPEN_END_MARGIN = 3

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4]
SEED_DEPTH_LOOKBACK = 3

MIN_NEW_FILL = 26
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, c in points
        if sum((rr, cc) in points for rr, cc in neighbors8(r, c)) == 1
    ]


def _comp_neighbors(
    points: set[tuple[int, int]], r: int, c: int
) -> list[tuple[int, int]]:
    return [(rr, cc) for rr, cc in neighbors8(r, c) if (rr, cc) in points]


def _analyze_wall_component(
    shape: tuple[int, int], color: int, points: set[tuple[int, int]]
) -> dict[str, object]:
    h, w = shape
    endpoints = sorted(_degree1_pixels(points))
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = _comp_neighbors(points, *ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _comp_neighbors(points, *n1) if p != ep]
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
    assert rad != (0, 0), "wall opening direction must be non-zero"

    mask = np.zeros((h, w), dtype=bool)
    for r, c in points:
        mask[r, c] = True

    r0, c0, r1, c1 = _bbox(points)
    return {
        "color": int(color),
        "points": points,
        "mask": mask,
        "bbox": (r0, c0, r1, c1),
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
        "open_center": ((endpoints[0][0] + endpoints[1][0]) / 2.0, (endpoints[0][1] + endpoints[1][1]) / 2.0),
    }


def _find_wall_infos(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for pts_arr in components_mask(grid == color, connectivity=8):
            pts = {(int(r), int(c)) for r, c in pts_arr}
            if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
                continue
            if len(_degree1_pixels(pts)) != 2:
                continue
            infos.append(_analyze_wall_component(grid.shape, color, pts))
    assert infos, "expected at least one wall component"
    return infos


def _build_barrier(shape: tuple[int, int], wall_info: dict[str, object]) -> np.ndarray:
    h, w = shape
    barrier = np.array(wall_info["mask"], copy=True)
    endpoints = wall_info["endpoints"]
    out_dirs = wall_info["out_dirs"]
    tip_12 = wall_info["tip_12"]
    rad = wall_info["rad"]

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

    return barrier


def _seed_blob_for_wall(
    grid: np.ndarray, wall_info: dict[str, object], bg: int
) -> tuple[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = wall_info["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = [int(v) for v in np.unique(sub) if int(v) not in (bg, wall_info["color"])]
    assert len(colors) == 1, "expected exactly one non-wall seed color inside a wall bbox"
    fill_color = colors[0]
    seeds = [
        (int(r), int(c))
        for r, c in zip(*np.where(grid == fill_color))
        if r0 <= int(r) <= r1 and c0 <= int(c) <= c1
    ]
    assert seeds, "expected seed blob inside wall bbox"
    return fill_color, seeds


def _shadow_trim(
    grid: np.ndarray,
    out: np.ndarray,
    filled_mask: np.ndarray,
    wall_info: dict[str, object],
    fill_color: int,
    bg: int,
) -> None:
    obstacles = (grid != bg) & (grid != fill_color) & (~wall_info["mask"])
    dr, dc = wall_info["rad"]
    if (dr, dc) == (0, 0):
        return

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

    h, w = grid.shape
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    elif dr == dc:
        for k in range(-(w - 1), h):
            coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)
    else:
        for k in range(h + w - 1):
            coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)


def _solve_wall_fill(
    grid: np.ndarray,
    wall_info: dict[str, object],
    fill_color: int,
    seed_positions: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    barrier = _build_barrier(grid.shape, wall_info)
    passable = (~barrier) & (grid == bg)
    seen = np.zeros_like(passable, dtype=bool)
    q = list(seed_positions)
    for r, c in q:
        seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    filled_mask = seen & (grid == bg)
    out = np.array(grid, copy=True)
    out[filled_mask] = fill_color
    _shadow_trim(grid, out, filled_mask, wall_info, fill_color, bg)
    return filled_mask


def _path_to_edge(
    start: tuple[int, int], d: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    rr, cc = start
    cells = [(rr, cc)]
    while True:
        nr, nc = rr + d[0], cc + d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            break
        rr, cc = nr, nc
        cells.append((rr, cc))
    return cells


def _move_marker(
    out: np.ndarray,
    start: tuple[int, int],
    d: tuple[int, int],
    marker_color: int,
    trail_color: int,
) -> None:
    path = _path_to_edge(start, d, out.shape)
    assert len(path) >= 2, "marker must have a non-zero move"
    for r, c in path[:-1]:
        out[r, c] = trail_color
    rr, cc = path[-1]
    out[rr, cc] = marker_color


def _sort_walls_for_layout(wall_infos: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(wall_infos) <= 1:
        return wall_infos
    rad = wall_infos[0]["rad"]
    if rad[0] != 0:
        return sorted(wall_infos, key=lambda info: info["bbox"][1])
    return sorted(wall_infos, key=lambda info: info["bbox"][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each half-enclosed sprite, then move same-colored exterior markers to the border.

    Input structure:
    - bg is the most common color.
    - There are two wall sprites, each a monochrome 8-connected 17-25 pixel partial enclosure with
      its own interior seed blob color.
    - Each seed color appears a second time as one isolated exterior pixel on the opening side of
      some sprite; that exterior placement may be closer to the wrong sprite.

    Rule:
    - For every wall sprite, flood-fill its interior and radiate outward through the opening exactly
      as in the base task, using the seed color found inside that sprite's bounding box.
    - Match each isolated exterior pixel to the sprite whose interior seed has the same color.
    - Slide that isolated pixel straight along the matched sprite's opening direction until the grid
      edge. The landing pixel keeps the marker color, and every vacated trail cell is painted with
      the matched sprite's wall color.

    Key insight:
    - Proximity is unreliable here because the isolated markers are deliberately swapped between the
      two sprite neighborhoods. The stable relation is color pairing: interior seed color identifies
      the correct wall, and that wall color determines the trail.
    """
    g = np.array(grid, dtype=int)
    out = np.array(g, copy=True)
    bg = int(get_bg(g))

    wall_infos = _sort_walls_for_layout(_find_wall_infos(g, bg))

    fill_by_wall: list[tuple[dict[str, object], int, list[tuple[int, int]]]] = []
    for wall_info in wall_infos:
        fill_color, seeds = _seed_blob_for_wall(g, wall_info, bg)
        fill_by_wall.append((wall_info, fill_color, seeds))

    for wall_info, fill_color, seeds in fill_by_wall:
        filled_mask = _solve_wall_fill(g, wall_info, fill_color, seeds, bg)
        out[filled_mask] = fill_color

    for wall_info, fill_color, seeds in fill_by_wall:
        seed_set = set(seeds)
        marker_positions = [
            (int(r), int(c))
            for r, c in zip(*np.where(g == fill_color))
            if (int(r), int(c)) not in seed_set
        ]
        assert len(marker_positions) == 1, "expected exactly one exterior marker per seed color"
        _move_marker(out, marker_positions[0], wall_info["rad"], fill_color, wall_info["color"])

    return out


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[tuple[int, int]]:
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


def _build_local_wall(rad: tuple[int, int], h: int, w: int) -> tuple[set[tuple[int, int]], int, int] | None:
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
        gap = random.randint(*GAP_HORIZ)

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


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
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


def _render_wall(
    grid: np.ndarray,
    local_wall: set[tuple[int, int]],
    rad: tuple[int, int],
    anchor: tuple[int, int],
    wall_color: int,
) -> set[tuple[int, int]]:
    placed = set()
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            raise AssertionError("wall placement out of bounds")
        if grid[r, c] != grid[0, 0]:
            raise AssertionError("wall overlap")
        grid[r, c] = wall_color
        placed.add((r, c))
    return placed


def _choose_layout(
    rad: tuple[int, int], h: int, w: int, wall_specs: list[tuple[set[tuple[int, int]], int, int]]
) -> list[tuple[int, int]] | None:
    (_, depth0, gap0), (_, depth1, gap1) = wall_specs
    if rad[0] != 0:
        left_c = random.randint(1, 2)
        right_c = w - gap1 - 2
        if right_c - (left_c + gap0) < 4:
            return None
        if rad == (-1, 0):
            open_r = random.randint(OPEN_END_MARGIN, h - max(depth0, depth1) - 4)
        else:
            open_r = random.randint(max(depth0, depth1) + 3, h - OPEN_END_MARGIN - 1)
        return [(open_r, left_c), (open_r, right_c)]
    top_r = random.randint(1, 2)
    bottom_r = h - gap1 - 2
    if bottom_r - (top_r + gap0) < 4:
        return None
    if rad == (0, -1):
        open_c = random.randint(OPEN_END_MARGIN, w - max(depth0, depth1) - 4)
    else:
        open_c = random.randint(max(depth0, depth1) + 3, w - OPEN_END_MARGIN - 1)
    return [(top_r, open_c), (bottom_r, open_c)]


def _candidate_marker_start(
    grid: np.ndarray, wall_info: dict[str, object], blocked: np.ndarray, bg: int
) -> tuple[int, int] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = wall_info["bbox"]
    dr, dc = wall_info["rad"]
    open_r, open_c = wall_info["open_center"]

    def path_clear(path: list[tuple[int, int]]) -> bool:
        return all(grid[r, c] == bg and not blocked[r, c] for r, c in path)

    if (dr, dc) == (-1, 0):
        cols = sorted(range(w), key=lambda c: (abs(c - open_c), c))
        for r in range(r0 - 1, 0, -1):
            for c in cols:
                path = [(rr, c) for rr in range(r, -1, -1)]
                if path_clear(path):
                    return (r, c)
    elif (dr, dc) == (1, 0):
        cols = sorted(range(w), key=lambda c: (abs(c - open_c), c))
        for r in range(r1 + 1, h - 1):
            for c in cols:
                path = [(rr, c) for rr in range(r, h)]
                if path_clear(path):
                    return (r, c)
    elif (dr, dc) == (0, -1):
        rows = sorted(range(h), key=lambda r: (abs(r - open_r), r))
        for c in range(c0 - 1, 0, -1):
            for r in rows:
                path = [(r, cc) for cc in range(c, -1, -1)]
                if path_clear(path):
                    return (r, c)
    else:
        rows = sorted(range(h), key=lambda r: (abs(r - open_r), r))
        for c in range(c1 + 1, w - 1):
            for r in rows:
                path = [(r, cc) for cc in range(c, w)]
                if path_clear(path):
                    return (r, c)
    return None


def _distance_to_wall(info: dict[str, object], p: tuple[int, int]) -> int:
    return min(abs(p[0] - r) + abs(p[1] - c) for r, c in info["points"])


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W_VERT) if rad[0] else random.randint(*GRID_W_HORIZ)

    built0 = _build_local_wall(rad, h, w)
    built1 = _build_local_wall(rad, h, w)
    if built0 is None or built1 is None:
        return None
    layout = _choose_layout(rad, h, w, [built0, built1])
    if layout is None:
        return None

    palette = random.sample(range(10), 5)
    bg, wall_color0, wall_color1, fill_color0, fill_color1 = palette
    grid = np.full((h, w), bg, dtype=int)

    walls_local = [built0, built1]
    wall_colors = [wall_color0, wall_color1]
    fill_colors = [fill_color0, fill_color1]

    for (local_wall, _depth, _gap), anchor, wall_color in zip(walls_local, layout, wall_colors):
        try:
            _render_wall(grid, local_wall, rad, anchor, wall_color)
        except AssertionError:
            return None

    for (local_wall, depth, _gap), anchor, fill_color in zip(walls_local, layout, fill_colors):
        interior = _local_interior(local_wall, depth)
        wall_set = set(local_wall)
        adjacent = [
            p
            for p in interior
            if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        ]
        seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
        if not seed_pool:
            seed_pool = adjacent
        if not seed_pool:
            return None
        blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
        for u, v in blob:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] != bg:
                return None
            grid[r, c] = fill_color

    try:
        wall_infos = _sort_walls_for_layout(_find_wall_infos(grid, bg))
    except Exception:
        return None
    if len(wall_infos) != 2:
        return None

    wall_payload: list[tuple[dict[str, object], int, list[tuple[int, int]]]] = []
    union_fill = np.zeros_like(grid, dtype=bool)
    try:
        for wall_info in wall_infos:
            fill_color, seeds = _seed_blob_for_wall(grid, wall_info, bg)
            wall_payload.append((wall_info, fill_color, seeds))
            union_fill |= _solve_wall_fill(grid, wall_info, fill_color, seeds, bg)
    except Exception:
        return None

    blocked = np.zeros_like(grid, dtype=bool)
    for info, _fill, _seeds in wall_payload:
        blocked |= info["mask"]

    starts = []
    for info, _fill, _seeds in wall_payload:
        start = _candidate_marker_start(grid, info, blocked, bg)
        if start is None:
            return None
        starts.append(start)
        blocked[start] = True

    if len(set(starts)) != 2:
        return None

    for idx, (_info, fill_color, _seeds) in enumerate(wall_payload):
        start = starts[1 - idx]
        path = _path_to_edge(start, rad, grid.shape)
        if len(path) < 2 or any(grid[r, c] != bg for r, c in path):
            return None
        grid[start] = fill_color

    for idx, (info, fill_color, _seeds) in enumerate(wall_payload):
        start = starts[1 - idx]
        own_dist = _distance_to_wall(info, start)
        other_dist = _distance_to_wall(wall_payload[1 - idx][0], start)
        if own_dist <= other_dist:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out != grid) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None

    if not (new_fill[0, :].any() or new_fill[-1, :].any() or new_fill[:, 0].any() or new_fill[:, -1].any()):
        return None

    trail_cells = 0
    for info, fill_color, seeds in wall_payload:
        seed_set = set(seeds)
        start = [
            (int(r), int(c))
            for r, c in zip(*np.where(grid == fill_color))
            if (int(r), int(c)) not in seed_set
        ]
        if len(start) != 1:
            return None
        path = _path_to_edge(start[0], info["rad"], grid.shape)
        trail = path[:-1]
        if not trail:
            return None
        if not all(out[r, c] == info["color"] for r, c in trail):
            return None
        trail_cells += len(trail)
    if trail_cells < 6:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")
