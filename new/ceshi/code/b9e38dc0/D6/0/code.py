# Variant of b9e38dc0
# Dimension: D6 — 半包围精灵的形状结构 (obstacle)
# Change: 半包围区域由U形或C形边框围成，有一个明确的开口方向（上/下/左/右）... → 多开口的围栏（两个或更多开口）

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import neighbors4, neighbors8, components_mask, get_bg

# Design analysis
# a) 具体化与行为追踪:
#    变体把“单口 U/C 形围栏”改成“薄矩形围栏上有两个或更多缺口”。围栏仍是单个墙色连通块，
#    填充种子仍在围栏内部，障碍仍是其他单像素/少量像素。输入中最显眼的变化是：墙不再靠两端点
#    暗示唯一开口，而是在边框上出现多个缺口。输出中填充不仅灌满内部，还会分别从每个缺口向外
#    形成独立带状辐射区。原题在关键环节上是“找两个端点 -> 合成单一 radiate_dir -> 整体做一次
#    影子扫描”；本变体改成“找边框上的所有缺口 -> 每个缺口单独建立边界射线和扫描方向 -> 结果并集”。
# b) 复杂度校准:
#    如果允许任意多边、任意厚度、任意重叠辐射带，视觉和推理都会变乱。这里把生成器收敛到
#    “1 像素厚矩形围栏 + 两个对向缺口”，solver 仍支持 2+ 缺口，但样例主要展示两条反向辐射，
#    既能清楚看出“多开口”，又避免带状区域大量交叉。
# c) 隐式假设审查:
#    原 main() 中与 D6 强相关的旧假设包括:
#    - 用 degree-1 像素当开口端点: 对矩形缺口完全失效，因为缺口两侧通常不是端点。
#    - 把所有端点 outward_dir 求和成单一 rad: 多缺口时不同缺口可能朝不同边，不能合并。
#    - 阴影逻辑默认只沿一个全局方向扫描: 多缺口时必须按缺口分别扫描，否则会把另一条辐射带误伤。
#    因此新 solver 改为 bbox 边扫描找缺口，并给每个缺口保留自己的法向方向。
# d) 放置合法性:
#    生成器只生成矩形边框，缺口只开在非角点位置，且上下缺口或左右缺口彼此错开，避免两条辐射带
#    在同一列/行内互相覆盖。矩形整体在两个开口朝向的边上都保留至少 3 格外部空间，保证辐射可见。
# e) 视觉可观察性:
#    样例中人类能直接看到“内部先填满，再从两个边上的缺口各自向外冒出”。开口由缺口位置决定，
#    与颜色 id 无关；不同样例只改变缺口位置和围栏朝向，不引入难以从少量样例概括的隐藏条件。
# f) 可视化整洁度:
#    只允许对向双缺口进入生成样本，可避免上/左/右多条带在外部交叉成杂乱块。障碍仍为少量散点，
#    且与墙保持距离，避免与缺口边界粘连。
# g) 颜色变化可推断性:
#    颜色规则不变: 墙色是最显眼非背景色，最近墙的另一种颜色是填充色，障碍色保持原色不变。
#    输出唯一的颜色变化仍是“背景 -> 填充色”，并且变化区域由围栏内部与缺口带共同定义。
# h) 新增 solver 假设审查:
#    新 solver 假设最近填充种子对应的墙组件是薄矩形边框，缺口表现为 bbox 四条边上的连续背景段。
#    生成器通过只画 1 像素厚矩形并只在边框上挖缺口，保证该假设成立；同时缺口不在角上，保证每个
#    缺口都能由两侧墙尖端确定一对平行边界射线。

# -- Configuration constants --------------------------------------------------
GRID_H = (12, 18)
GRID_W = (15, 19)
FENCE_H = (7, 11)
FENCE_W = (8, 12)
OPEN_LEN = (2, 3)
OPEN_END_MARGIN = 3
SIDE_MARGIN = 1

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5, 6]
MAX_OBSTACLES = 2
OBSTACLE_IN_FILL_PROB = 0.35
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 16
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _bbox_from_points(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _select_wall_and_fill(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid)
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0, len(pts)
        return best, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill-color seed pixel"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    return bg, wall, fill, fill_pos, wall_comp


def _false_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(values):
        if values[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(values) and not values[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _find_openings(comp_mask: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int]]:
    coords = np.argwhere(comp_mask)
    assert coords.size, "expected non-empty wall component"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    openings: list[dict] = []

    top = [bool(comp_mask[r0, c]) for c in range(c0, c1 + 1)]
    for a, b in _false_runs(top):
        if a == 0 or b == len(top) - 1:
            continue
        openings.append(
            {
                "side": "top",
                "rad": (-1, 0),
                "span": (c0 + a, c0 + b),
                "tips": ((r0, c0 + a - 1), (r0, c0 + b + 1)),
            }
        )

    bottom = [bool(comp_mask[r1, c]) for c in range(c0, c1 + 1)]
    for a, b in _false_runs(bottom):
        if a == 0 or b == len(bottom) - 1:
            continue
        openings.append(
            {
                "side": "bottom",
                "rad": (1, 0),
                "span": (c0 + a, c0 + b),
                "tips": ((r1, c0 + a - 1), (r1, c0 + b + 1)),
            }
        )

    left = [bool(comp_mask[r, c0]) for r in range(r0, r1 + 1)]
    for a, b in _false_runs(left):
        if a == 0 or b == len(left) - 1:
            continue
        openings.append(
            {
                "side": "left",
                "rad": (0, -1),
                "span": (r0 + a, r0 + b),
                "tips": ((r0 + a - 1, c0), (r0 + b + 1, c0)),
            }
        )

    right = [bool(comp_mask[r, c1]) for r in range(r0, r1 + 1)]
    for a, b in _false_runs(right):
        if a == 0 or b == len(right) - 1:
            continue
        openings.append(
            {
                "side": "right",
                "rad": (0, 1),
                "span": (r0 + a, r0 + b),
                "tips": ((r0 + a - 1, c1), (r0 + b + 1, c1)),
            }
        )

    return openings, (r0, c0, r1, c1)


def _add_cardinal_ray(barrier: np.ndarray, start: tuple[int, int], d: tuple[int, int]) -> None:
    h, w = barrier.shape
    rr, cc = start
    while True:
        rr += d[0]
        cc += d[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        barrier[rr, cc] = True


def _flood_from_seeds(passable: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    seen = np.zeros_like(passable, dtype=bool)
    q = seeds[:]
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _shadow_lines_for_opening(opening: dict, h: int, w: int) -> list[list[tuple[int, int]]]:
    lo, hi = opening["span"]
    d = opening["rad"]
    lines: list[list[tuple[int, int]]] = []
    if d == (-1, 0):
        rows = range(h - 1, -1, -1)
        for c in range(lo, hi + 1):
            lines.append([(r, c) for r in rows])
    elif d == (1, 0):
        rows = range(h)
        for c in range(lo, hi + 1):
            lines.append([(r, c) for r in rows])
    elif d == (0, -1):
        cols = range(w - 1, -1, -1)
        for r in range(lo, hi + 1):
            lines.append([(r, c) for c in cols])
    else:
        cols = range(w)
        for r in range(lo, hi + 1):
            lines.append([(r, c) for c in cols])
    return lines


def _apply_opening_shadows(
    out: np.ndarray,
    filled_mask: np.ndarray,
    obstacles: np.ndarray,
    openings: list[dict],
) -> None:
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
                out[r, c] = get_bg(out)
                filled_mask[r, c] = False

    h, w = out.shape
    for opening in openings:
        for coords in _shadow_lines_for_opening(opening, h, w):
            shadow_process(coords)


def analyze_primary_fence(grid: np.ndarray) -> dict:
    bg, wall, fill, fill_pos, wall_comp = _select_wall_and_fill(grid)
    comp_mask = _component_mask(wall_comp, grid.shape)
    openings, bbox = _find_openings(comp_mask)
    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pixels": len(fill_pos),
        "opening_count": len(openings),
        "opening_sides": tuple(sorted(op["side"] for op in openings)),
        "bbox": bbox,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of a fenced region and leak the color out through every edge gap.

    Input structure:
    - bg is the most common color. The wall is the most frequent non-bg color and forms a thin fence component.
    - A fill-color seed sits inside that fence. Any other non-bg pixels are preserved obstacles.
    Rule:
    - Flood-fill the interior of the fence, then continue outward through each wall gap, using the gap's two wall tips as parallel boundary rays.
    - Obstacles cast background-colored shadows independently along each gap's own outward direction.

    Key insight: Openings are missing runs on the fence bbox edges, not degree-1 endpoints; each detected gap defines its own ray direction and shadow scan.
    """
    g = np.array(grid, copy=True)
    bg, wall, fill, fill_pos, wall_comp = _select_wall_and_fill(g)

    comp_mask = _component_mask(wall_comp, g.shape)
    openings, _bbox = _find_openings(comp_mask)
    assert len(openings) >= 2, "expected a fence with two or more openings"

    barrier = comp_mask.copy()
    for opening in openings:
        for tip in opening["tips"]:
            _add_cardinal_ray(barrier, tip, opening["rad"])

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = _flood_from_seeds(passable, fill_pos)
    filled_mask = seen & (g == bg)

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    _apply_opening_shadows(out, filled_mask, obstacles, openings)
    return out


def _rect_border_points(height: int, width: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(width):
        pts.add((0, c))
        pts.add((height - 1, c))
    for r in range(height):
        pts.add((r, 0))
        pts.add((r, width - 1))
    return pts


def _choose_two_disjoint_intervals(length: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(80):
        len_a = random.randint(*OPEN_LEN)
        len_b = random.randint(*OPEN_LEN)
        a0 = random.randint(1, length - len_a - 2)
        b0 = random.randint(1, length - len_b - 2)
        a1 = a0 + len_a - 1
        b1 = b0 + len_b - 1
        if max(a0, b0) <= min(a1, b1):
            continue
        if abs((a0 + a1) - (b0 + b1)) < 4:
            continue
        return (a0, a1), (b0, b1)
    return None


def _build_local_fence(kind: str, h: int, w: int) -> tuple[set[tuple[int, int]], dict] | None:
    fence_h = random.randint(*FENCE_H)
    fence_w = random.randint(*FENCE_W)
    if fence_h + 2 * OPEN_END_MARGIN > h or fence_w + 2 * OPEN_END_MARGIN > w:
        return None

    wall = _rect_border_points(fence_h, fence_w)
    openings: dict = {"kind": kind, "fence_h": fence_h, "fence_w": fence_w}

    if kind == "vertical":
        chosen = _choose_two_disjoint_intervals(fence_w)
        if chosen is None:
            return None
        top_gap, bottom_gap = chosen
        for c in range(top_gap[0], top_gap[1] + 1):
            wall.remove((0, c))
        for c in range(bottom_gap[0], bottom_gap[1] + 1):
            wall.remove((fence_h - 1, c))
        for c in range(top_gap[0], top_gap[1] + 1):
            wall.add((1, c))
        openings["expected_borders"] = ("top", "bottom")
        openings["gaps"] = {"top": top_gap, "bottom": bottom_gap}
    else:
        chosen = _choose_two_disjoint_intervals(fence_h)
        if chosen is None:
            return None
        left_gap, right_gap = chosen
        for r in range(left_gap[0], left_gap[1] + 1):
            wall.remove((r, 0))
        for r in range(right_gap[0], right_gap[1] + 1):
            wall.remove((r, fence_w - 1))
        for r in range(left_gap[0], left_gap[1] + 1):
            wall.add((r, 1))
        openings["expected_borders"] = ("left", "right")
        openings["gaps"] = {"left": left_gap, "right": right_gap}

    return wall, openings


def _place_anchor(kind: str, grid_h: int, grid_w: int, fence_h: int, fence_w: int) -> tuple[int, int]:
    if kind == "vertical":
        r0 = random.randint(OPEN_END_MARGIN, grid_h - fence_h - OPEN_END_MARGIN)
        c0 = random.randint(SIDE_MARGIN, grid_w - fence_w - SIDE_MARGIN)
    else:
        r0 = random.randint(SIDE_MARGIN, grid_h - fence_h - SIDE_MARGIN)
        c0 = random.randint(OPEN_END_MARGIN, grid_w - fence_w - OPEN_END_MARGIN)
    return r0, c0


def _local_interior(openings: dict) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, openings["fence_h"] - 1) for c in range(1, openings["fence_w"] - 1)]


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target:
                    break
    return blob


def _seed_pool(kind: str, wall: set[tuple[int, int]], openings: dict) -> list[tuple[int, int]]:
    interior = _local_interior(openings)
    pool: list[tuple[int, int]] = []
    for r, c in interior:
        if not any((rr, cc) in wall for rr, cc in neighbors8(r, c)):
            continue
        if kind == "vertical":
            if c <= 2 or c >= openings["fence_w"] - 3:
                pool.append((r, c))
        else:
            if r <= 2 or r >= openings["fence_h"] - 3:
                pool.append((r, c))
    return pool


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(
    grid: np.ndarray, bg: int, wall_color: int, nominal_fill_zone: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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


def _border_fill_count(mask: np.ndarray) -> int:
    count = 0
    count += int(mask[0, :].any())
    count += int(mask[-1, :].any())
    count += int(mask[:, 0].any())
    count += int(mask[:, -1].any())
    return count


def _expected_borders_hit(mask: np.ndarray, expected: tuple[str, str]) -> bool:
    checks = {
        "top": bool(mask[0, :].any()),
        "bottom": bool(mask[-1, :].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
    }
    return all(checks[name] for name in expected)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H)
    grid_w = random.randint(*GRID_W)
    kind = random.choice(["vertical", "horizontal"])

    built = _build_local_fence(kind, grid_h, grid_w)
    if built is None:
        return None
    local_wall, opening_info = built
    r0, c0 = _place_anchor(kind, grid_h, grid_w, opening_info["fence_h"], opening_info["fence_w"])

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for r, c in local_wall:
        grid[r0 + r, c0 + c] = wall_color

    pool = _seed_pool(kind, local_wall, opening_info)
    if not pool:
        return None

    blob = _grow_seed_blob(random.choice(pool), set(_local_interior(opening_info)))
    for r, c in blob:
        gr, gc = r0 + r, c0 + c
        if grid[gr, gc] != bg:
            return None
        grid[gr, gc] = fill_color

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

    metrics = analyze_primary_fence(grid)
    if metrics["opening_count"] < 2:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D6 variant input")
