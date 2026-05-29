# Variant of b9e38dc0
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: same——输出网格与输入尺寸完全相同... → 输出按精灵数量分区

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
# ---------------------------------------------------------------------------
# variant_spec:
#   core_change:
#     original: one scene in one canvas, output keeps the same canvas shape
#     variant: the input canvas contains multiple independent wall sprites; the
#       output is partitioned into one full-size panel per wall sprite, with
#       solid separator rows between panels
#   concrete_structure:
#     - The input keeps the original visual vocabulary: bg, one wall color,
#       one fill seed per wall, and optional obstacle pixels.
#     - To make D9 observable, the generator places exactly two independent wall
#       scenes in one canvas, arranged left-to-right.
#     - The output keeps the original canvas size but turns the two horizontal
#       bands into explicit display panels. Each panel shows the solve result
#       for exactly one wall scene, computed after erasing the other scene.
#     - The empty gap between the two scene bands becomes a solid wall-colored
#       separator column block. The panel order follows the input's left-to-
#       right order.
#   behavior_trace:
#     - Original flow: detect the single wall component -> infer its fill color
#       -> flood and radiate in-place -> return one solved grid.
#     - Variant flow: detect both wall components -> for each one, keep only the
#       pixels belonging to that scene on a blank copy of the full canvas ->
#       run the original single-scene solve -> stack the two solved full-canvas
#       results with a wall-colored divider row.
#     - The main difference is therefore not the local fill rule; it is the
#       extra decomposition/recomposition layer around that rule.
#   complexity_calibration:
#     - One panel would collapse back to the original "same-shape" behavior, so
#       the generator always creates two scenes.
#     - More than two scenes would either exceed ARC-friendly output height
#       bounds or force the wall geometry to become too tiny and ambiguous.
#     - The two scenes share the wall color but use distinct fill colors so the
#       scene-to-seed correspondence stays visually obvious.
#   implicit_assumptions_audit:
#     - The original code assumes exactly one relevant wall component. Here that
#       is no longer true, so the single-scene logic is moved into _solve_single
#       and only applied after one wall component has been isolated.
#     - The original fill-color selection uses "nearest non-wall color". That
#       remains valid per isolated scene, and the generator guarantees each
#       scene's fill color is unique and closest to its own wall.
#     - Obstacles are still singletons in the same scene; the generator keeps
#       obstacle colors scene-local so assigning them by nearest wall component
#       is safe.
#   placement_legality:
#     - Each scene lives in its own column band; wall, seed, and obstacles are
#       restricted to that band, so the two scenes never overlap in the input.
#     - The solver isolates scenes on the full canvas, so local fill can still
#       extend all the way to the global border exactly as the variant intends.
#   observability_and_cleanliness:
#     - The input clearly shows two wall sprites in left/right order.
#     - The output clearly shows two side-by-side panels, with the middle gap
#       converted into a solid wall-colored divider. This makes the "one scene
#       per panel" rule visually explicit without changing the outer canvas
#       dimensions.
#     - Separator color is not arbitrary: it reuses the visible wall color.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _solve_single(grid: np.ndarray) -> np.ndarray:
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

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))

    fill_pos = list(zip(*np.where(g == fill)))
    fill_pos = [(int(r), int(c)) for r, c in fill_pos]
    if not fill_pos:
        return g

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

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))

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

    return out


def _min_dist_to_points(points_a: list[tuple[int, int]], points_b: list[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in points_a:
        for rr, cc in points_b:
            best = min(best, abs(r - rr) + abs(c - cc))
    return best


def _group_scenes(g: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_components = [
        {
            "points": [(int(r), int(c)) for r, c in pts],
        }
        for pts in components_mask(g == wall, connectivity=8)
    ]
    assert len(wall_components) >= 2, "expected multiple wall sprites for this variant"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected fill colors"

    color_points = {
        c: [(int(r), int(c2)) for r, c2 in zip(*np.where(g == c))]
        for c in other_colors
    }

    color_owner: dict[int, int] = {}
    color_dist: dict[int, int] = {}
    for color, pts in color_points.items():
        best_idx = None
        best_d = 10**9
        for idx, comp in enumerate(wall_components):
            d = _min_dist_to_points(pts, comp["points"])
            if d < best_d:
                best_d = d
                best_idx = idx
        assert best_idx is not None
        color_owner[color] = best_idx
        color_dist[color] = best_d

    scenes: list[dict] = []
    for idx, comp in enumerate(wall_components):
        owned_colors = [c for c in other_colors if color_owner[c] == idx]
        assert owned_colors, "each wall needs at least one non-wall color"
        fill = min(owned_colors, key=lambda c: (color_dist[c], len(color_points[c])))
        scene = {
            "wall_points": comp["points"],
            "fill": fill,
            "fill_points": color_points[fill],
            "obstacle_points": [],
        }
        for color in owned_colors:
            if color == fill:
                continue
            scene["obstacle_points"].extend(color_points[color])
        all_points = scene["wall_points"] + scene["fill_points"] + scene["obstacle_points"]
        scene["col_lo"] = min(c for _, c in all_points)
        scene["col_hi"] = max(c for _, c in all_points)
        scenes.append(scene)

    scenes.sort(key=lambda scene: scene["col_lo"])
    return bg, wall, scenes


def _panel_spans(scenes: list[dict], width: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    assert len(scenes) == N_SCENES, "expected the generated number of scene panels"
    band_w = (width - BAND_GAP) // N_SCENES
    assert N_SCENES * band_w + BAND_GAP == width, "unexpected canvas width for fixed panel layout"
    spans = [(0, band_w), (band_w + BAND_GAP, width)]
    separators = [(band_w, band_w + BAND_GAP)]
    return spans, separators


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a multi-sprite canvas into one in-place output panel per wall sprite and solve each panel independently.

    Input structure:
    - bg is the most common color. One wall color appears in multiple disconnected partial enclosures.
    - Each wall enclosure has its own nearby fill-color seed inside it, plus optional obstacle pixels in the same band of space.

    Rule:
    - Treat each wall enclosure as a separate scene on a blank copy of the full input canvas.
    - For each scene, apply the original flood-fill, outward radiation, and obstacle-shadow logic independently.
    - Recompose the result into same-size left/right panels, separated by a solid wall-colored divider occupying the original gap.

    Key insight: The local fill/radiate behavior stays the same, but the output no longer overlays all sprites together; it isolates each wall sprite into its own panel and uses spatial ordering to decide panel order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg, wall, scenes = _group_scenes(g)

    panels: list[np.ndarray] = []
    for scene in scenes:
        local = np.full((h, w), bg, dtype=int)
        for r, c in scene["wall_points"]:
            local[r, c] = wall
        for r, c in scene["fill_points"]:
            local[r, c] = scene["fill"]
        for r, c in scene["obstacle_points"]:
            local[r, c] = g[r, c]
        panels.append(_solve_single(local))

    spans, separators = _panel_spans(scenes, w)
    out = np.full((h, w), bg, dtype=int)
    for panel, (col_lo, col_hi) in zip(panels, spans):
        out[:, col_lo:col_hi] = panel[:, col_lo:col_hi]
    for col_lo, col_hi in separators:
        out[:, col_lo:col_hi] = wall
    return out


# --------------- generator (variant) ---------------

# Full input canvas bounds
GRID_H = (11, 14)
BAND_W = (10, 12)
BAND_GAP = 2
N_SCENES = 2

# Wall geometry adapted so two scenes fit in one shared canvas
WALL_PIXELS_MIN = 13
WALL_PIXELS_MAX = 20
DEPTH_VERT = (4, 6)
DEPTH_HORIZ = (4, 5)
GAP_VERT = (4, 5)
GAP_HORIZ = (4, 5)
TIP_12_PROB = 0.45
OPEN_END_MARGIN = 2

# Seed and obstacles
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4]
SEED_DEPTH_LOOKBACK = 2
MAX_OBSTACLES_PER_SCENE = 3
OBSTACLE_IN_FILL_PROB = 0.55
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 10
GEN_RETRIES = 500
SCENE_RETRIES = 120


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


def _build_local_wall(rad: tuple[int, int], h: int, w: int):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 5)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(GAP_VERT[0], min(GAP_VERT[1], w - 2))
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 5)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 1))

    if gap < 4:
        return None

    if gap == 4:
        left_bottom, right_bottom = 1, 3
    else:
        left_bottom, right_bottom = random.choice([(1, 4), (2, 3)])

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
            for u in range(max(0, depth - 2), depth + 1)
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


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int):
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 2), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 1, h - 2), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 2)
    return random.randint(0, h - gap - 1), random.randint(depth + 1, w - 2)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        vals = by_row.get(u)
        if vals is None or len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
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


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _find_obstacle_candidates(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    nominal_fill_zone: np.ndarray,
    col_lo: int,
    col_hi: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    band_mask = np.zeros_like(grid, dtype=bool)
    band_mask[:, col_lo:col_hi] = True
    eligible = (grid == bg) & (dist > 1) & band_mask
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


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


def _scene_attempt(
    h: int,
    w: int,
    col_lo: int,
    col_hi: int,
    side: str,
    bg: int,
    wall_color: int,
    fill_color: int,
    obstacle_colors: list[int],
) -> np.ndarray | None:
    band_w = col_hi - col_lo
    dir_choices = [(-1, 0), (1, 0), (0, -1)] if side == "left" else [(-1, 0), (1, 0), (0, 1)]
    rad = random.choice(dir_choices)

    built = _build_local_wall(rad, h, band_w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, band_w, depth, gap)

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        c += col_lo
        if not (0 <= r < h and col_lo <= c < col_hi):
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
        c += col_lo
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_out = _solve_single(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES_PER_SCENE)
    if n_obstacles and obstacle_colors:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone, col_lo, col_hi)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = _solve_single(grid)
    except Exception:
        return None
    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(*GRID_H)
        band_w = random.randint(*BAND_W)
        w = N_SCENES * band_w + BAND_GAP

        palette = random.sample(range(10), 7)
        bg, wall_color, fill_left, fill_right = palette[:4]
        obstacle_pool = palette[4:]
        random.shuffle(obstacle_pool)
        cut = random.randint(1, len(obstacle_pool) - 1)
        left_obstacles = obstacle_pool[:cut]
        right_obstacles = obstacle_pool[cut:]

        left_cols = (0, band_w)
        right_cols = (band_w + BAND_GAP, w)

        left_scene = None
        for _ in range(SCENE_RETRIES):
            left_scene = _scene_attempt(
                h, w, left_cols[0], left_cols[1], "left", bg, wall_color, fill_left, left_obstacles
            )
            if left_scene is not None:
                break
        if left_scene is None:
            continue

        right_scene = None
        for _ in range(SCENE_RETRIES):
            right_scene = _scene_attempt(
                h, w, right_cols[0], right_cols[1], "right", bg, wall_color, fill_right, right_obstacles
            )
            if right_scene is not None:
                break
        if right_scene is None:
            continue

        grid = np.full((h, w), bg, dtype=int)
        mask_left = left_scene != bg
        mask_right = right_scene != bg
        if np.any(mask_left & mask_right):
            continue
        grid[mask_left] = left_scene[mask_left]
        grid[mask_right] = right_scene[mask_right]

        try:
            out = main(grid)
        except Exception:
            continue

        if out.shape != grid.shape:
            continue

        try:
            base_like = _solve_single(grid)
        except Exception:
            continue
        if np.array_equal(out, base_like):
            continue

        spans, separators = _panel_spans(_group_scenes(grid)[2], w)
        if any(not np.all(out[:, col_lo:col_hi] == wall_color) for col_lo, col_hi in separators):
            continue

        left_cols, right_cols = spans[0], spans[1]
        left_panel = out[:, left_cols[0] : left_cols[1]]
        right_panel = out[:, right_cols[0] : right_cols[1]]
        if np.array_equal(left_panel, right_panel):
            continue
        if np.array_equal(left_panel, grid[:, left_cols[0] : left_cols[1]]):
            continue
        if np.array_equal(right_panel, grid[:, right_cols[0] : right_cols[1]]):
            continue
        return grid

    raise RuntimeError("failed to generate valid D9 b9e38dc0 input")
