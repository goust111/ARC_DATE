# Combo variant of a47bf94d (index 3)
# combo_spec: D1/0+D8/2
# Name: 最短路径+旋转节点

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 原题：上下两排同色 token 通过预制 8 管道配对，9 号矩形只是管道里的宽节点；solid 在原地变 diamond，
#      远端得到 X，少数反向通道则由 X 源补出 diamond。
#    - 融合后：
#      1. 输入里不再预放任何 8 管道。
#      2. 中部改成若干条横向 5 号障碍带；每条障碍带内部只有一个 9 号“门”可以穿过。
#      3. solver 需要从源 token 的内侧端点到同色 diamond 占位符的内侧端点做 BFS 最短路搜索。
#      4. 路径每穿过一个 9 门，就根据门的宽高比累计一次 90 度旋转：
#         宽>高为顺时针，高>宽为逆时针。
#      5. 为避免 D8/2 退化成“旋转了但结果没变”，远端输出改成带方向 tip 的互补形状：
#         solid 源送出 “X+tip”，X 源送出 “diamond+tip”；tip 会被累计旋转带动。
#    - 具体小例子：顶部实心 3x3 源和底部同色 diamond 占位符之间若有三条障碍带，门型依次是
#      tall / wide / wide，那么净旋转 = -90 + 90 + 90 = +90。最短路被障碍带强制穿过这三个门，
#      输出时底部不是固定 X，而是“向右转过 90 度”的 X+tip。
#
# b) 复杂度校准
#    - 如果只做 BFS 而远端仍是纯 X / 纯 diamond，90 度旋转对最终像素完全不可见，融合会退化。
#    - 因此保留原题的“源端本地归一化 + 远端补形”结构，但让远端补形带一个单格 tip，使 D8 可见且仍然是
#      少量离散朝向（4 档），不会把规则变成连续几何推理。
#    - 迷宫结构仍控制在 2-4 条横向障碍带，且门只在左右两侧交替出现，保证 BFS 结果清晰、可从少量样例归纳。
#
# c) 隐式假设审查（对应原 main 中被改动的前提）
#    - 原 `track_mask = g != bg` 假设“所有非背景都是既有线路”；现在不成立，因为 5 是墙，9 是可通行旋转门，
#      而输入根本没有预制线路。
#    - 原 `_trace_wire(...)` 假设局部贪心沿连通管线追踪即可；现在不成立，因为路线来自空白网格中的全局最短路。
#    - 原 9 矩形是“直通/转弯管节点”；现在 9 只提供可通行门和旋转事件，不再承载局部拓扑。
#    - 原远端输出对 90 度旋转不敏感（X / diamond 都自同构）；现在必须改成非 90 度对称的带 tip 形状。
#
# d) 放置合法性
#    - token 仍放在距离边缘 2-3 行的位置，保持完整 3x3。
#    - 5 号障碍带只放在上下 token 之间，横向铺满整行，不会与 token 重叠。
#    - 每条障碍带中的 9 门高度恰好等于带厚，因此上下区域只有这一块可穿越；这样所有 top↔bottom 路径都必经门。
#    - 9 门只使用非正方形尺寸，避免“宽高相等时方向不明”的歧义。
#
# e) 视觉可观察性
#    - 人类能直接看到：输入没有任何 8 管道，但输出新增了 8 最短路。
#    - 人类也能看到：不同样例里 9 门有横门/竖门两种宽高比，输出远端的 tip 朝向随门序列变化。
#    - 同一样例中多种颜色共享同一套门序列，给出重复支撑，避免只靠单个孤例猜旋转。
#
# f) 可视化整洁度
#    - 角色分工固定：0 地板，5 墙带，9 旋转门，8 输出路径，1/2/3/4/6/7 为 token 颜色。
#    - 墙带整行、门是小矩形、token 仍位于上下边缘，区域层次清楚；路径不会覆盖 9 门本身，门在输出里继续可见。
#
# g) 变化的可推断性
#    - 不做颜色映射；颜色只用于上下配对。
#    - 真正变化的是“同色哪两个位置配对”“路径穿过哪些门”“tip 最终朝哪边”，三者都有明确可见的结构线索。
#
# h) 新 main() 的隐式假设与生成器保证
#    - 假设每个活动颜色恰有一个源 token（solid 或 X）和一个同色 diamond 占位符；generator 强制这一一对应。
#    - 假设所有 source-target 配对是上下相对，因此端点是中心向内 2 格；generator 固定 top/bottom 双层布局。
#    - 假设 9 组件都是实心矩形且非正方形；generator 只生成满矩形门，并拒绝正方形门。
#    - 假设最短路存在；generator 用整行墙带 + 单门结构保证上下可达。
#    - 假设旋转真的影响输出；generator 拒绝净旋转为 0 的样本，并额外验证“有旋转”和“无旋转”两种求解结果不同。


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

# Base orientations for the transported, rotation-sensitive complements.
_X_TIP_OFF = _X_OFF + [(-1, 0)]
_DIAMOND_TIP_OFF = _DIAMOND_OFF + [(-1, -1)]


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


def _by_color(found: dict[tuple[int, int], tuple[int, tuple[int, int]]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for center, (col, _) in found.items():
        assert col not in out
        out[col] = center
    return out


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


def _rotate_offsets(offsets: list[tuple[int, int]], k_cw: int) -> list[tuple[int, int]]:
    k = k_cw % 4
    rotated = list(offsets)
    for _ in range(k):
        rotated = [(dc, -dr) for dr, dc in rotated]
    return rotated


def _find_9rects(g: np.ndarray) -> list[dict]:
    """Return metadata for solid non-square color-9 rectangles."""
    rects: list[dict] = []
    for idx, comp in enumerate(components_mask(g == 9, connectivity=4)):
        cells = {(int(r), int(c)) for r, c in comp}
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert len(cells) == h * w
        assert h != w
        rects.append(
            {
                "id": idx,
                "cells": cells,
                "bbox": (r0, c0, r1, c1),
                "rot": 1 if w > h else -1,
            }
        )
    return rects


def _bfs_shortest_path(g: np.ndarray, bg: int, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Shortest 4-neighbor path through background and 9-gates, with deterministic tie-breaking."""
    h, w = g.shape
    passable = (g == bg) | (g == 9)
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


def _path_rotation(path: list[tuple[int, int]], rect_lookup: dict[tuple[int, int], int], rect_rot: dict[int, int]) -> int:
    touched: set[int] = set()
    total = 0
    for p in path:
        rect_id = rect_lookup.get(p)
        if rect_id is None or rect_id in touched:
            continue
        touched.add(rect_id)
        total += rect_rot[rect_id]
    return total


def _target_offsets(source_kind: str, rot_k: int) -> list[tuple[int, int]]:
    if source_kind == "solid":
        return _rotate_offsets(_X_TIP_OFF, rot_k)
    return _rotate_offsets(_DIAMOND_TIP_OFF, rot_k)


def _solve(grid: np.ndarray, *, apply_rotation: bool) -> np.ndarray:
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

    rects = _find_9rects(g)
    rect_lookup: dict[tuple[int, int], int] = {}
    rect_rot: dict[int, int] = {}
    for rect in rects:
        rect_rot[int(rect["id"])] = int(rect["rot"])
        for cell in rect["cells"]:
            rect_lookup[cell] = int(rect["id"])

    out = g.copy()

    for col in source_colors:
        assert not (col in solid_by_color and col in x_by_color)
        assert col in diamond_by_color

        if col in solid_by_color:
            source_kind = "solid"
            src = solid_by_color[col]
            dst = diamond_by_color[col]
            _clear_3x3(out, src, bg)
            _stamp_token(out, src, col, _DIAMOND_OFF)
        else:
            source_kind = "x"
            src = x_by_color[col]
            dst = diamond_by_color[col]

        start, goal = _endpoint_pair(src, dst)
        path = _bfs_shortest_path(g, bg, start, goal)
        rot_k = _path_rotation(path, rect_lookup, rect_rot) if apply_rotation else 0

        for r, c in path:
            if int(out[r, c]) == bg:
                out[r, c] = 8

        _clear_3x3(out, dst, bg)
        _stamp_token(out, dst, col, _target_offsets(source_kind, rot_k))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each token through an empty maze and rotate its delivered complement at 9-gates.

    Input structure:
    - The board has top and bottom token sites. Each active color appears once as a source token (solid 3x3 or X) and once as a same-colored diamond placeholder.
    - The middle area has no prebuilt 8 pipe. Instead, full-width color-5 barrier bands divide the floor, and each band contains one color-9 rectangular gate.

    Rule:
    - For each color, search the shortest 4-neighbor path from the source-side endpoint to the opposite placeholder endpoint through background cells and 9-gates.
    - Solid sources collapse in place to diamonds. At the far site, the placeholder is replaced by the source's complementary shape with a directional tip.
    - Every 9-gate touched by the path rotates that delivered shape by 90 degrees: wide gates rotate clockwise, tall gates rotate counterclockwise. The chosen path is drawn in color 8, but the 9-gates themselves stay visible.

    Key insight: Color still pairs the two sites, but the actual output depends on two coupled inferences: which shortest maze route is forced by the 5-bands, and what net rotation that route accumulates from the sequence of 9-gates it passes through.
    """
    return _solve(grid, apply_rotation=True)


# --------------- generator (shortest_path_rotation_gates) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
X_INPUT_COUNT_WEIGHTS = [0, 65, 35]
GATE_COUNT_CHOICES = [2, 3, 4]
WIDE_GATE_WIDTHS = [3, 4]
TALL_GATE_SPECS = [(2, 1), (3, 1), (3, 2)]
MIN_BAND_GAP_ROWS = 2
GEN_MAX_RETRIES = 400


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


def _place_gate_bands(grid: np.ndarray, top_ep_row: int, bot_ep_row: int) -> list[dict]:
    """Place full-width 5-bands with one embedded 9-gate each."""
    h, w = grid.shape
    span_top = top_ep_row + 2
    span_bot = bot_ep_row - 2
    span_len = span_bot - span_top + 1
    if span_len < 7:
        return []

    for _ in range(100):
        n_bands = random.choice(GATE_COUNT_CHOICES)
        kinds = [random.choice(["wide", "tall"]) for _ in range(n_bands)]
        if not any(k == "wide" for k in kinds):
            kinds[random.randrange(n_bands)] = "wide"
        if not any(k == "tall" for k in kinds):
            kinds[random.randrange(n_bands)] = "tall"

        heights = [1 if kind == "wide" else random.choice([spec[0] for spec in TALL_GATE_SPECS]) for kind in kinds]
        required = sum(heights) + MIN_BAND_GAP_ROWS * (n_bands - 1)
        if required > span_len:
            continue

        slack = span_len - required
        extras = _random_composition(slack, n_bands + 1)
        starts = []
        cur = span_top + extras[0]
        for hh, extra in zip(heights, extras[1:]):
            starts.append(cur)
            cur += hh + MIN_BAND_GAP_ROWS + extra

        start_side = random.choice([0, 1])
        bands: list[dict] = []
        ok = True
        for idx, (kind, hh, r0) in enumerate(zip(kinds, heights, starts)):
            if kind == "wide":
                gate_h, gate_w = 1, random.choice(WIDE_GATE_WIDTHS)
            else:
                choices = [spec for spec in TALL_GATE_SPECS if spec[0] == hh]
                gate_h, gate_w = random.choice(choices)

            side = (start_side + idx) % 2
            if side == 0:
                lo, hi = 1, max(1, w // 3)
            else:
                lo, hi = max(1, w - w // 3 - gate_w - 1), max(1, w - gate_w - 2)
            if lo > hi:
                ok = False
                break
            c0 = random.randint(lo, hi)
            c1 = c0 + gate_w - 1
            if c0 < 1 or c1 >= w - 1:
                ok = False
                break

            grid[r0:r0 + hh, :] = 5
            grid[r0:r0 + gate_h, c0:c1 + 1] = 9
            bands.append(
                {
                    "r0": r0,
                    "h": hh,
                    "c0": c0,
                    "w": gate_w,
                    "rot": 1 if gate_w > gate_h else -1,
                }
            )

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
            }
        )
    return specs


def _make_input(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    inp = grid.copy()
    for spec in specs:
        color = int(spec["color"])
        top_center = spec["top_center"]
        bot_center = spec["bot_center"]
        if spec["source_kind"] == "solid":
            _stamp_token(inp, top_center, color, _SOLID_OFF)
            _stamp_token(inp, bot_center, color, _DIAMOND_OFF)
        else:
            _stamp_token(inp, top_center, color, _DIAMOND_OFF)
            _stamp_token(inp, bot_center, color, _X_OFF)
    return inp


def _count_rotation_events(inp: np.ndarray) -> tuple[int, int]:
    rects = _find_9rects(inp)
    rect_lookup: dict[tuple[int, int], int] = {}
    rect_rot: dict[int, int] = {}
    for rect in rects:
        rect_rot[int(rect["id"])] = int(rect["rot"])
        for cell in rect["cells"]:
            rect_lookup[cell] = int(rect["id"])

    bg = int(get_bg(inp))
    solids = _by_color(_find_solid_3x3(inp, bg))
    xs = _by_color(_find_x_3x3(inp, bg))
    diamonds = _by_color(_find_diamond_3x3(inp, bg))

    source_colors = sorted(set(solids) | set(xs))
    touched_total = 0
    rot_total = 0
    for col in source_colors:
        src = solids[col] if col in solids else xs[col]
        dst = diamonds[col]
        path = _bfs_shortest_path(inp, bg, *_endpoint_pair(src, dst))
        touched = set()
        for p in path:
            rect_id = rect_lookup.get(p)
            if rect_id is not None:
                touched.add(rect_id)
        touched_total += len(touched)
        rot_total += sum(rect_rot[rid] for rid in touched)
    return touched_total, rot_total


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
    h_min = max(GRID_H_RANGE[0], top_ep_row + 10)
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(h_min, GRID_H_RANGE[1])
    bot_row = h - top_row - 1
    bot_ep_row = bot_row - 2
    if bot_ep_row - top_ep_row < 8:
        return None

    colors = random.sample(TOKEN_COLORS_POOL, n)
    perm = _random_permutation(n)
    x_count = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(x_count, n)))
    if not x_channels or len(x_channels) == n:
        return None

    base = np.zeros((h, w), dtype=int)
    bands = _place_gate_bands(base, top_ep_row, bot_ep_row)
    if len(bands) < 2:
        return None

    net_rot = sum(int(band["rot"]) for band in bands)
    if net_rot % 4 == 0:
        return None

    specs = _build_specs(n, ch_cols, perm, colors, top_row, bot_row, x_channels)
    inp = _make_input(base, specs)

    try:
        expected = main(inp)
        baseline = _solve(inp, apply_rotation=False)
    except Exception:
        return None

    if np.array_equal(expected, baseline):
        return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    touched_total, rot_total = _count_rotation_events(inp)
    if touched_total < n * 2:
        return None
    if rot_total == 0:
        return None
    if int((expected == 8).sum()) <= 0:
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
