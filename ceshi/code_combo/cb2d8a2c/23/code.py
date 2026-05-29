# Combo variant of cb2d8a2c (index 23)
# combo_spec: D8/1+D19/0
# Name: 颜色编码距离+图例说明

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

# Design analysis (YAML-like, written before implementation):
# fusion_goal:
#   keep:
#     - 原题的主干结构不变: azure 背景、单个边界绿色起点、交替贴墙的 1-cell 粗障碍条、
#       以 Manhattan 半径构造禁区、再用 Dijkstra 找“先最短、再最少转弯”的绿路径。
#     - 输出仍把障碍条中的特殊标记像素并入红色主体。
#   change:
#     - D8/1: 半径不再来自“同一条 bar 上有几个蓝点”，而改成“同一条 bar 使用哪一种统一标记色”。
#       为了彻底切断旧规则，所有 bar 的标记像素数量固定成 3 个，图案固定为 C-R-C-R-C。
#     - D19/0: 输入角落新增图例区。图例不是装饰，而是每个样例内部的显式查表说明:
#         一行 = [标记色样本][灰色分隔][若干酒红 tally]
#       tally 的长度就是该行对应的半径值 1..4。
#     - 为了让“读 legend”成为必要步骤，颜色->半径映射在每个样例里随机置换；
#       solver 不能硬编码 yellow=1 之类的全局表，必须先解析图例。
# concrete_object_form:
#   bg: azure(8)
#   path_start: unique green(3) pixel on one border edge
#   bars:
#     - 主体为 red(2)，每条 bar 恰含一种 marker 色，marker 前缀固定为 C-R-C-R-C。
#     - marker 色从 {blue(1), yellow(4), pink(6), orange(7)} 中选。
#     - 3 个 marker 的数量与分布对所有 bar 都相同，因此不再携带半径信息。
#   legend:
#     - 固定 4 行，放在起始边一侧的角落，且与第一条 bar 位于相反侧角，避免覆盖。
#     - 每行结构从角落向内读:
#         marker_color, grey(5), maroon(9) repeated radius times, then bg padding
#     - 四行分别代表 radius 1/2/3/4，但行上的 marker 颜色按样例随机打乱。
# behavior_trace:
#   small_example:
#     - 假设某样例 legend 四行显示:
#         pink + 1 个 maroon
#         yellow + 2 个 maroon
#         blue + 3 个 maroon
#         orange + 4 个 maroon
#       那么 pink-bar 的禁区半径是 1，yellow-bar 是 2，blue-bar 是 3，orange-bar 是 4。
#     - 原题里如果一条 bar 上看到 3 个蓝点，就直接半径=3。
#     - 融合后就算两条 bar 都是 3 个 marker，它们也可能因颜色不同而需要不同绕行距离；
#       而且同一种颜色在不同样例里的半径还能变化，必须去角落 legend 查表。
# complexity_calibration:
#   - 半径档位仍限制在 1..4，保持与原题同级复杂度，不把路径搜索膨胀成过难问题。
#   - 标记色只用 4 种，legend 也只占一个 4x6 小角，不会把谜面挤得过乱。
#   - 生成器在 bar 数 > 1 时强制至少两种不同半径，避免样例退化成“所有颜色都等价”。
#   - 映射按样例置换，使 legend 真正参与推理，同时 legend 自身提供足够明确的结构性线索。
# hidden_assumption_audit:
#   old_main_dependencies:
#     - 原版只把 {blue, red} 当障碍组件: 现在必须改成 {red + marker_colors}，但要排除 legend 区。
#     - 原版 radius = count(blue): 现在必须改成“组件内唯一 marker 色 -> legend 查表 -> radius”。
#     - 原版把所有 blue 变 red: 现在只能把障碍条里的 marker 变 red，legend 上的颜色必须保留。
#   new_main_dependencies:
#     - 图例必须能被唯一识别；因此它固定贴角、固定 4 行、固定分隔色与 tally 色、且这两色不在 bar 中使用。
#     - 每条 bar 必须只有一种 marker 色；generator 统一给整条 bar 的 3 个 marker 用同色。
#     - 每条 bar 长度至少 5，才能容纳 C-R-C-R-C 前缀；生成器显式保证。
# placement_legality:
#   - bar 间主轴间距仍按真实 radius 约束:
#       start_to_first >= r1 + 1
#       gap(i,i+1) >= r_i + r_{i+1} + 1
#       last_to_end >= 1
#   - 横轴空隙仍满足 cross_gap >= radius + 1，确保路径能从 bar 的开口侧绕过去。
#   - legend 放在第一条 bar 相反侧的顶角，因此 row 很早时也不会与 bar 或起点重叠。
#   - 旋转到其他起始边时，legend 连同主谜面一起旋转，仍然留在某个真实角落。
# visual_observability:
#   - bar 上所有 marker 数量一致，肉眼无法再靠“数几个点”得出距离。
#   - legend 的每一行把“颜色样本”和“长度 1..4 的 tally”并排放在同一区域，查表关系明确可见。
#   - 同图常有至少两种 radius，因此输出路径对不同颜色 bar 的绕行距离通常可直接对比。
# neatness:
#   - 主谜面只额外增加一个紧凑角落表格，没有把说明符号散落到路径区域中。
#   - legend 使用 grey 分隔和 maroon tally，和 red bar / green path / azure bg 层次分明。
# inference_check:
#   - 如果忽略 legend，只按“固定颜色表”或“marker 数量=3”来解，很多样例会得到错误路径；
#     因为 generator 会随机置换颜色映射，并生成多半径 bar 组合。
# generator_guarantees_for_main:
#   - 恰有一个合法 legend，且 solver 在 4 个旋转视角中最多只会匹配到一个角落候选。
#   - 恰有一个绿色起点且不落在 legend 上。
#   - 每个障碍 bar 是一个 4 连通组件，组件内仅含 red 与一种 marker 色。
#   - 布局经 main() 验证后才输出，确保 goal_state 可达且不会触发断言。

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

MARKER_COLORS = (BLUE, YELLOW, PINK, ORANGE)
RADIUS_VALUES = (1, 2, 3, 4)
MARKER_PIXELS_PER_BAR = 3
MARKER_PREFIX_LEN = 2 * MARKER_PIXELS_PER_BAR - 1
LEGEND_HEIGHT = len(RADIUS_VALUES)
LEGEND_WIDTH = 2 + max(RADIUS_VALUES)


def _try_extract_corner_legend(rot: np.ndarray, corner: str) -> tuple[dict[int, int], np.ndarray] | None:
    """Parse the fixed-size legend in a rotated view if it matches the expected corner layout."""
    h, w = rot.shape
    if h < LEGEND_HEIGHT or w < LEGEND_WIDTH:
        return None

    mask = np.zeros((h, w), dtype=bool)
    radius_to_marker: dict[int, int] = {}
    used_markers: set[int] = set()

    for row_idx, radius in enumerate(RADIUS_VALUES):
        r = row_idx
        if corner == "tl":
            marker_pos = (r, 0)
            sep_pos = (r, 1)
            tally_slice = slice(2, 2 + radius)
            tail_slice = slice(2 + radius, LEGEND_WIDTH)
            tally = rot[r, tally_slice]
            tail = rot[r, tail_slice]
            row_cells = [(r, c) for c in range(LEGEND_WIDTH)]
        else:
            marker_pos = (r, w - 1)
            sep_pos = (r, w - 2)
            tally_slice = slice(w - 2 - radius, w - 2)
            tail_slice = slice(w - LEGEND_WIDTH, w - 2 - radius)
            tally = rot[r, tally_slice]
            tail = rot[r, tail_slice]
            row_cells = [(r, c) for c in range(w - LEGEND_WIDTH, w)]

        marker = int(rot[marker_pos])
        if marker not in MARKER_COLORS or marker in used_markers:
            return None
        if int(rot[sep_pos]) != GREY:
            return None
        if tally.size != radius or not np.all(tally == MAROON):
            return None
        if tail.size and not np.all(tail == BG):
            return None

        radius_to_marker[radius] = marker
        used_markers.add(marker)
        for rr, cc in row_cells:
            if int(rot[rr, cc]) != BG:
                mask[rr, cc] = True

    return {marker: radius for radius, marker in radius_to_marker.items()}, mask


def _extract_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    """Find the corner legend under some rotation and return marker->radius plus its mask in original coords."""
    matches: list[tuple[dict[int, int], np.ndarray]] = []
    for k in range(4):
        rot = np.rot90(grid, k)
        for corner in ("tl", "tr"):
            parsed = _try_extract_corner_legend(rot, corner)
            if parsed is None:
                continue
            marker_to_radius, mask_rot = parsed
            matches.append((marker_to_radius, np.rot90(mask_rot, -k)))

    assert len(matches) == 1, "grid must contain exactly one valid corner legend"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the green path across the grid while each red obstacle bar reads its keep-out distance from a corner legend rather than from counting markers.

    Input structure:
    - bg = azure (8). Obstacle bars are red (2) with a fixed C-R-C-R-C marker prefix using one color from {blue, yellow, pink, orange}.
    - A single green (3) pixel lies on a border edge and marks the path start.
    - One corner contains a 4-row legend: each row shows a marker color, a grey separator, and 1-4 maroon tally cells that explicitly define that color's distance.

    Rule:
    - Parse the legend to recover the per-instance marker_color -> radius mapping, then find each bar component and assign it the radius of its unique marker color.
    - Forbid every cell whose Manhattan distance from a bar is at most that radius, keep the legend itself blocked, and route the green path from the start edge to the opposite edge through the remaining cells, minimizing steps first and turns second.
    - After the path is drawn, only the marker-colored pixels that belong to obstacle bars become red; the legend remains unchanged.

    Key insight: the bar markers no longer encode distance by how many times they appear, and the color meanings are not globally fixed; the corner legend must be read first, then applied to every bar.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    marker_to_radius, legend_mask = _extract_legend(g)
    obstacle_marker_mask = np.isin(g, MARKER_COLORS) & (~legend_mask)
    obstacle_mask = ((g == RED) | obstacle_marker_mask) & (~legend_mask)

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
        marker_colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) in MARKER_COLORS}
        assert len(marker_colors) == 1, "each obstacle bar must use exactly one marker color"
        marker_color = marker_colors.pop()
        assert marker_color in marker_to_radius, "unknown obstacle marker color"
        radius = marker_to_radius[marker_color]

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            rr, cc = int(r), int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))

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

    blocked = obstacle_mask | forbidden | legend_mask
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
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
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

    out[obstacle_marker_mask] = RED
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
    """Pick per-bar radii that fit and usually expose at least two legend rows in use."""
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
    """Paint one wall-attached obstacle bar with a fixed-color C-R-C-R-C prefix."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[:MARKER_PREFIX_LEN:2] = marker_color
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length:] = colors[::-1]


def _paint_legend(grid: np.ndarray, legend_side: int, radius_to_marker: dict[int, int]) -> None:
    """Paint the 4-row corner legend on the top-left (0) or top-right (1)."""
    h, w = grid.shape
    assert h >= LEGEND_HEIGHT and w >= LEGEND_WIDTH
    for row_idx, radius in enumerate(RADIUS_VALUES):
        marker = radius_to_marker[radius]
        if legend_side == 0:
            grid[row_idx, 0] = marker
            grid[row_idx, 1] = GREY
            grid[row_idx, 2 : 2 + radius] = MAROON
        else:
            grid[row_idx, w - 1] = marker
            grid[row_idx, w - 2] = GREY
            grid[row_idx, w - 2 - radius : w - 2] = MAROON


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    first_side: int,
    radius_to_marker: dict[int, int],
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
    legend_side = 1 - first_side
    _paint_legend(grid, legend_side, radius_to_marker)

    lengths = []
    for row, radius, side in zip(rows, radii, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, radius)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, radius_to_marker[radius])

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
    cross_size = random.randint(max(cross_lo, LEGEND_WIDTH), cross_hi)
    first_side = random.randrange(2)
    shuffled_markers = random.sample(MARKER_COLORS, len(MARKER_COLORS))
    radius_to_marker = {
        radius: shuffled_markers[radius - 1]
        for radius in RADIUS_VALUES
    }

    canonical = _canonical_grid(main_size, cross_size, radii, first_side, radius_to_marker)
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
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
