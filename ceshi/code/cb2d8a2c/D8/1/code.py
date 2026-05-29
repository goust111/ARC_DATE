# Variant of cb2d8a2c
# Dimension: D8 — 障碍条上的蓝色像素数量与分布 (obstacle)
# Change: 每条障碍条有1-4个蓝色像素散布在红色像素之间，蓝色像素数量直接编码该条的最小回避距离... → 蓝色像素替换为其他颜色编码

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

# Design analysis (YAML-like, written before implementation):
# variant_goal:
#   keep:
#     - 原题的网格尺寸范围、单个绿色起点、交替贴墙的红色障碍条、曼哈顿禁区、Dijkstra 最短且尽量少转弯的寻路。
#     - 输出仍然是绿色路径 + 障碍条中的特殊像素全部变红。
#   change:
#     - 原题半径来自“同一条障碍条里蓝色像素的个数”。
#     - 现在半径来自“同一条障碍条里统一的标记颜色”。
#     - 标记像素数量固定为 3 个，位置固定成 C-R-C-R-C 前缀；因此计数和分布都不再提供半径信息。
# concrete_object_form:
#   bg: azure(8)
#   start: unique green(3) on one border
#   bars:
#     - 每条 bar 仍为一格粗、贴一侧墙、朝另一侧延伸。
#     - bar 主体颜色仍为 red(2)。
#     - bar 靠墙端的前 5 格中，第 1/3/5 格放同一种标记色；其余格 red。
#     - 标记色到半径的固定映射:
#         yellow(4) -> 1
#         blue(1)   -> 2
#         pink(6)   -> 3
#         orange(7) -> 4
# behavior_trace:
#   original_local_step:
#     - 组件内有 blue_count 个蓝点。
#     - solver 对该组件做 BFS 膨胀到半径 blue_count。
#   variant_local_step:
#     - 组件内不再数蓝点，而是抽取“唯一非红色标记色”。
#     - 由颜色查表得到 radius，再做同样的 BFS 膨胀。
#   visible_difference:
#     - 两条都有 3 个标记点的 bar，可能因颜色不同而要求不同绕行距离。
#     - 这能直接破坏“数点即可解题”的旧策略。
# complexity_calibration:
#   - 半径仍限制在 1..4，保持原题复杂度，不引入过多离散档位。
#   - 颜色只用 4 种，足够形成映射表，又不会让视觉太杂。
#   - 生成器尽量在 count>1 时使用至少两种半径，避免样例退化成“只有一种颜色”的弱可观察情况。
# hidden_assumption_audit:
#   old_solver_dependencies:
#     - bar_mask 只看 {blue, red}: 变体后必须扩展为 {all marker colors, red}。
#     - radius = count(blue): 变体后失效，必须改为读取唯一标记色。
#     - out[out == BLUE] = RED: 变体后必须把所有标记色都转成 red。
#   new_solver_dependencies:
#     - 每个障碍条组件必须恰有一种非红标记色；generator 通过统一涂色保证。
#     - 每个 bar 必须至少有 5 格长度，才能容纳固定的 C-R-C-R-C 前缀；generator 的长度下界保证。
# placement_legality:
#   - 主轴间距仍按半径值计算:
#       start_to_first >= r1 + 1
#       gap(i,i+1) >= ri + r{i+1} + 1
#       last_to_end >= 1
#   - 横轴剩余空隙仍满足 cross_gap >= radius + 1，确保路径存在绕开空间。
#   - bar 长度还需 >= 5 以容纳固定标记前缀。
# visual_observability:
#   - 所有 bar 的标记数量相同，解题者无法再从数量读出半径。
#   - 输出中的路径与不同颜色 bar 的离距不同，训练对能支持“颜色->距离”归纳。
# neatness:
#   - 颜色层次清晰: azure 背景、green 路径、red 障碍主体、少量 marker 色。
#   - 不引入额外图例区，避免网格被多个说明对象挤乱。
# color_inference:
#   - 颜色映射是固定小表，不依赖颜色 id 的排序，也不依赖隐藏元数据。
#   - 同色 bar 总是同半径，不同色 bar 可在同图中并列出现，便于肉眼比较。
# generator_guarantees_for_main:
#   - 恰有一个绿色起点，且位于边界。
#   - 每个 bar 是单个 4 连通组件，组件内只有 red 和一种 marker 色。
#   - 布局约束保证新 solver 的 goal_state 可达。

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

MARKER_TO_RADIUS = {
    YELLOW: 1,
    BLUE: 2,
    PINK: 3,
    ORANGE: 4,
}
RADIUS_TO_MARKER = {radius: color for color, radius in MARKER_TO_RADIUS.items()}
MARKER_COLORS = tuple(MARKER_TO_RADIUS.keys())
MARKER_PIXELS_PER_BAR = 3
MARKER_PREFIX_LEN = 2 * MARKER_PIXELS_PER_BAR - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path across the grid while red obstacle bars advertise their keep-out radius by marker color instead of marker count.

    Input structure:
    - bg = azure (8). Each obstacle bar is mostly red (2) but begins with a fixed C-R-C-R-C marker prefix using one color from {yellow, blue, pink, orange}.
    - A single green (3) pixel lies on one border edge and marks the path start.
    - All bars keep the original wall-attached geometry; only the encoding of the danger radius changes.

    Rule:
    - Read each bar's radius from its unique marker color: yellow->1, blue->2, pink->3, orange->4.
    - Forbid every cell whose Manhattan distance from that bar is at most the encoded radius, then route the green path from the start edge to the opposite edge through the remaining cells, minimizing steps first and turns second.
    - After the path is drawn, every marker-colored pixel becomes red.

    Key insight: the repeated marker pixels are no longer counted; identical three-marker prefixes can force different detours solely because their color differs.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    obstacle_mask = (g == RED) | np.isin(g, MARKER_COLORS)
    dirs = _DIRS_URDL

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1
    elif sr == h - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", w - 1
    elif sc == w - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    bar_comps = components_mask(obstacle_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        marker_colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != RED}
        assert len(marker_colors) == 1, "each obstacle bar must use exactly one marker color"
        marker_color = marker_colors.pop()
        assert marker_color in MARKER_TO_RADIUS, "unknown obstacle marker color"
        radius = MARKER_TO_RADIUS[marker_color]

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            dist[r, c] = 0
            q.append((int(r), int(c)))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked = obstacle_mask | forbidden
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(dirs):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    out = g.copy()
    assert goal_state is not None, "a valid path to the opposite edge must exist"
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[np.isin(out, MARKER_COLORS)] = RED
    return out


# --------------- generator ---------------

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
RADIUS_LO, RADIUS_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300


def _min_main_len(radii: tuple[int, ...]) -> int:
    """Minimum main-axis size to fit all bars with the required keep-out spacing."""
    last_bar = radii[0] + 2
    for a, b in zip(radii, radii[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Distribute total into parts non-negative integers uniformly."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_radii(count: int, main_hi: int) -> tuple[int, ...]:
    """Pick per-bar radii that fit and usually show at least two distinct colors."""
    choices = [
        combo
        for combo in itertools.product(range(RADIUS_LO, RADIUS_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
        and (count == 1 or len(set(combo)) >= 2)
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, radius: int) -> tuple[int, int] | None:
    """Valid bar length range given cross-axis size and encoded radius."""
    lo = max(
        BAR_LEN_MIN,
        MARKER_PREFIX_LEN,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - radius - 1,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, marker_color: int) -> None:
    """Paint one wall-attached obstacle bar with a fixed-color marker prefix."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[:MARKER_PREFIX_LEN:2] = marker_color
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length:] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation (start on top, path goes downward)."""
    count = len(radii)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += radii[i - 1] + radii[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths = []
    for row, radius, side in zip(rows, radii, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, radius)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, RADIUS_TO_MARKER[radius])

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical top-start grid so the start edge matches edge."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output places green on the opposite edge."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build one valid input and verify it with main()."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    radii = _choose_radii(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, radii, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D8 input")
