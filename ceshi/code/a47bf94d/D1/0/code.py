# Variant of a47bf94d
# Dimension: D1 — 管道路径追踪方式 (path)
# Change: 管道追踪（pipeline_trace）——沿青色(color 8)正交管道线段从一个端点追踪到另一个端点，遇到color-9矩形块时穿过
#         → 最短路径搜索：去掉预建管道，改为在背景中绕开 color-9 墙带，用 BFS 搜索源点到目标锚点的最短正交路径

import random
from collections import deque

import numpy as np

from arc_utils import get_bg, neighbors4


# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 保留原题最稳定、最可见的对象：顶部/底部两排 3x3 token 位置、solid / X / diamond 三种局部形状、以及颜色一一对应。
#    - 删除原题输入里的青色管道和灰色 crossing 标记；中部改成若干条横向 maroon(9) 墙带，每条墙带只留一个小缺口。
#    - 每种颜色仍然有两个位置：顶部固定是“diamond 终态位置”，底部固定是“X 终态位置”，底部列顺序仍由一个非恒等置换决定。
#    - 输入中：
#      1. 大多数颜色在顶部放 solid，底部放同色 diamond 锚点；
#      2. 少数颜色在底部放 X，顶部放同色 diamond 锚点。
#    - 输出中：
#      1. 顶部全部变成 diamond；
#      2. 底部全部变成 X；
#      3. 额外把每个颜色从源端点到对侧锚点端点的最短背景路径画成 8。
#    - 具体小例子：原题里“顶端 solid + 管道 + 底端空位”时，solver 只需顺着已有 8 走到另一端；现在同样的顶端 solid
#      和底端同色锚点之间没有预制线路，只有几道 9 墙。solver 需要在 0 背景里 BFS，选出绕墙最短路线，再把这条路线画成 8。
#
# b) 复杂度校准
#    - 若只删掉管道而不把路径显式画出来，样例会退化成“同色上下配对”，D1 不可观察；因此输出显式新增最短路轨迹。
#    - 若让目标也通过最短路从多个候选里竞争，会把任务复杂度抬到“匹配 + 路径规划”，对 2-4 个样例过重；因此目标仍由同色 diamond 锚点直接指定。
#    - 迷宫只使用 2-4 条横向墙带，每条 1 个缺口，控制自由度，保证最短路规则能从少量样例归纳。
#
# c) 隐式假设审查（针对原 main 的相关段落）
#    - 原 `track_mask = g != bg` 假设“所有非背景非 token 的格子都是可走管道”；变体不成立，因为 9 现在是障碍，且输入里根本没有管道。
#    - 原 `_trace_wire(...)` 假设存在度数为 1 的管道端点、以及沿组件局部贪心前进即可到达终点；变体不成立，因为路线要在空白背景中全局搜索。
#    - 原 `comp_colors` / `endpoints` / `9-rect pass-through` 都依赖显式连通管网，必须整体删除。
#    - 新 solver 的核心假设变成：每个活动颜色恰有一个“源 token”(solid 或 X) 和一个“对侧 diamond 锚点”；generator 会强制满足。
#
# d) 放置合法性
#    - token 仍放在距边缘 2-3 行的位置，3x3 完整落盘；顶部列等距，底部列为其置换。
#    - 9 墙带只放在上下 token 中间，避免与 token 重叠，也避免把端点封死。
#    - 墙带横跨整行，仅在单个小窗口留空，因此任意 top↔bottom 配对都必定存在路，但通常需要水平绕行。
#    - 路径端点位于 token 中心向内侧 2 格处，始终是背景格，solver 可以直接起终点 BFS。
#
# e) 视觉可观察性
#    - 人类可直接看到：输入没有 8 管道，输出却新增 8 路径，而且路径被 9 墙的缺口强制弯折，这就是 D1 的核心变化。
#    - token 变换仍与原题一致：solid 本地变 diamond，远端 diamond 被补成 X；X 源则保持 X，远端 diamond 保持 diamond。
#
# f) 可视化整洁度
#    - 障碍只用统一颜色 9，路径只用统一颜色 8；token 颜色仍来自小色池，角色分工清楚。
#    - 墙带是少量粗横条，路径虽然可能共享一段，但整体仍像“穿过缺口的最短路线”，不至于变成噪声。
#
# g) 颜色变化可推断性
#    - 不做颜色映射；同色上下两处始终成对出现，颜色只承担“哪两个位置配对”的可见线索，不参与排序或 tie-break。
#
# 参考任务 7b0280bc 的借鉴点
#    - 采用 BFS 求最短路而不是局部追踪。
#    - 输出中显式显示被选中的路径，使“搜索而非跟线”成为可见规律。

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for isolated 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            isolated = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                isolated = False
                                break
                    if not isolated:
                        break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in _X_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for isolated 3x3 diamonds."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 1])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _stamp_token(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def _clear_3x3(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    grid[cr - 1:cr + 2, cc - 1:cc + 2] = bg


def _endpoint_pair(src: tuple[int, int], dst: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if src[0] < dst[0]:
        return (src[0] + 2, src[1]), (dst[0] - 2, dst[1])
    return (src[0] - 2, src[1]), (dst[0] + 2, dst[1])


def _bfs_shortest_path(g: np.ndarray, bg: int, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Shortest 4-neighbor path through background cells, with deterministic tie-breaking."""
    h, w = g.shape
    passable = g == bg
    passable[start] = True
    passable[goal] = True
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        nbrs = []
        for nr, nc in neighbors4(*cur):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and (nr, nc) not in prev:
                md = abs(nr - goal[0]) + abs(nc - goal[1])
                nbrs.append((md, nr, nc))
        nbrs.sort()
        for _, nr, nc in nbrs:
            prev[(nr, nc)] = cur
            q.append((nr, nc))
    assert goal in prev
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _by_color(found: dict[tuple[int, int], tuple[int, tuple[int, int]]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for center, (col, _) in found.items():
        assert col not in out
        out[col] = center
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace fixed pipe-following with shortest-path routing between paired token sites.

    Input structure:
    - The background is open floor. Color-9 rectangles form walls in the middle of the board.
    - Each active color appears at a top site and a bottom site. One site is a source token (top solid block or bottom X), and the opposite site is a same-colored diamond anchor.

    Rule:
    - Every solid source collapses in place to a diamond.
    - For each color, find the shortest 4-neighbor background path between the source-side endpoint and the opposite anchor endpoint, then draw that path in color 8.
    - At the far site, solid sources turn the diamond anchor into an X; X sources leave the far diamond as a diamond.

    Key insight: Color still tells you which two token sites pair up, but the route is no longer encoded by prebuilt pipes; it must be searched in empty space around the walls.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    solid_by_color = _by_color(solids)
    x_by_color = _by_color(xs)
    diamond_by_color = _by_color(diamonds)

    source_colors = sorted(set(solid_by_color) | set(x_by_color))
    assert source_colors

    out = g.copy()

    for col in source_colors:
        assert not (col in solid_by_color and col in x_by_color)
        assert col in diamond_by_color

        if col in solid_by_color:
            src = solid_by_color[col]
            dst = diamond_by_color[col]
            _clear_3x3(out, src, bg)
            _stamp_token(out, src, col, _DIAMOND_OFF)
            far_shape = _X_OFF
        else:
            src = x_by_color[col]
            dst = diamond_by_color[col]
            far_shape = _DIAMOND_OFF

        start, goal = _endpoint_pair(src, dst)
        path = _bfs_shortest_path(g, bg, start, goal)
        for r, c in path:
            out[r, c] = 8

        _clear_3x3(out, dst, bg)
        _stamp_token(out, dst, col, far_shape)

    return out


# --------------- generator (shortest_path_anchor_maze) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
WALL_COUNT_CHOICES = [2, 3, 4]
WALL_HEIGHT_CHOICES = [1, 2]
GAP_WIDTH_CHOICES = [2, 3]
MIN_WALL_GAP_ROWS = 2
GEN_MAX_RETRIES = 300


def _random_permutation(n: int) -> list[int]:
    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            return perm


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    vals = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return vals


def _place_wall_bands(grid: np.ndarray, top_ep_row: int, bot_ep_row: int) -> list[tuple[int, int, int]]:
    """Place full-width wall bands with a single gap; return [(r0, h, gap_center), ...]."""
    h, w = grid.shape
    span_top = top_ep_row + 2
    span_bot = bot_ep_row - 2
    span_len = span_bot - span_top + 1
    if span_len < 6:
        return []

    max_walls = min(max(WALL_COUNT_CHOICES), max(2, (span_len + MIN_WALL_GAP_ROWS) // (1 + MIN_WALL_GAP_ROWS)))
    choices = [n for n in WALL_COUNT_CHOICES if 2 <= n <= max_walls]
    if not choices:
        return []

    for _ in range(80):
        n_walls = random.choice(choices)
        heights = [random.choice(WALL_HEIGHT_CHOICES) for _ in range(n_walls)]
        required = sum(heights) + MIN_WALL_GAP_ROWS * (n_walls - 1)
        if required > span_len:
            continue
        slack = span_len - required
        extras = _random_composition(slack, n_walls + 1)
        starts = []
        cur = span_top + extras[0]
        for i, hh in enumerate(heights):
            starts.append(cur)
            cur += hh + MIN_WALL_GAP_ROWS + extras[i + 1]

        left_choices = list(range(1, max(2, w // 3)))
        right_choices = list(range(max(1, w - w // 3 - 3), max(2, w - 2)))
        if not left_choices or not right_choices:
            continue
        start_side = random.choice([0, 1])
        bands = []
        ok = True
        for i, (r0, hh) in enumerate(zip(starts, heights)):
            gap_w = random.choice(GAP_WIDTH_CHOICES)
            side = (start_side + i) % 2
            if side == 0:
                g0 = random.choice(left_choices)
            else:
                g0 = random.choice(right_choices)
            g0 = min(max(1, g0), w - gap_w - 1)
            g1 = g0 + gap_w - 1
            if not (1 <= g0 <= g1 < w - 1):
                ok = False
                break
            grid[r0:r0 + hh, :g0] = 9
            grid[r0:r0 + hh, g1 + 1:] = 9
            bands.append((r0, hh, g0 + gap_w // 2))
        if ok:
            return bands
        grid[:, :] = 0
    return []


def _build_specs(
    n: int,
    ch_cols: list[int],
    perm: list[int],
    colors: list[int],
    top_row: int,
    bot_row: int,
    x_channels: set[int],
) -> list[dict]:
    specs = []
    for i in range(n):
        top_center = (top_row, ch_cols[i])
        bot_center = (bot_row, ch_cols[perm[i]])
        specs.append(
            {
                "color": colors[i],
                "top_center": top_center,
                "bot_center": bot_center,
                "source_kind": "x" if i in x_channels else "solid",
                "source_center": bot_center if i in x_channels else top_center,
                "target_center": top_center if i in x_channels else bot_center,
            }
        )
    return specs


def _make_input(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    inp = grid.copy()
    for spec in specs:
        color = spec["color"]
        if spec["source_kind"] == "solid":
            _stamp_token(inp, spec["top_center"], color, _SOLID_OFF)
            _stamp_token(inp, spec["bot_center"], color, _DIAMOND_OFF)
        else:
            _stamp_token(inp, spec["top_center"], color, _DIAMOND_OFF)
            _stamp_token(inp, spec["bot_center"], color, _X_OFF)
    return inp


def _expected_output(inp: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, dict]:
    bg = int(get_bg(inp))
    out = inp.copy()
    total_detour = 0
    total_turns = 0
    for spec in specs:
        color = spec["color"]
        src = spec["source_center"]
        dst = spec["target_center"]
        start, goal = _endpoint_pair(src, dst)
        path = _bfs_shortest_path(inp, bg, start, goal)
        for r, c in path:
            out[r, c] = 8
        if spec["source_kind"] == "solid":
            _clear_3x3(out, src, bg)
            _stamp_token(out, src, color, _DIAMOND_OFF)
            _clear_3x3(out, dst, bg)
            _stamp_token(out, dst, color, _X_OFF)
        else:
            _clear_3x3(out, dst, bg)
            _stamp_token(out, dst, color, _DIAMOND_OFF)
        total_detour += len(path) - 1 - (abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
        dirs = []
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            dirs.append((r1 - r0, c1 - c0))
        total_turns += sum(d0 != d1 for d0, d1 in zip(dirs, dirs[1:]))
    metrics = {"detour": total_detour, "turns": total_turns, "path_cells": int((out == 8).sum())}
    return out, metrics


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    w = max(ch_cols[-1] + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_row + 2
    colors = random.sample(TOKEN_COLORS_POOL, n)
    perm = _random_permutation(n)

    h_min = max(GRID_H_RANGE[0], top_ep_row + 10)
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(h_min, GRID_H_RANGE[1])
    bot_row = h - top_row - 1
    bot_ep_row = bot_row - 2
    if bot_ep_row - top_ep_row < 8:
        return None

    x_count = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(x_count, n)))

    base = np.zeros((h, w), dtype=int)
    bands = _place_wall_bands(base, top_ep_row, bot_ep_row)
    if len(bands) < 2:
        return None

    specs = _build_specs(n, ch_cols, perm, colors, top_row, bot_row, x_channels)
    inp = _make_input(base, specs)

    try:
        expected, metrics = _expected_output(inp, specs)
    except Exception:
        return None

    if metrics["detour"] <= 0 or metrics["turns"] <= 0 or metrics["path_cells"] <= 0:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
