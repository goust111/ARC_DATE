# Variant of 332f06d7
# Dimension: D11 — 通道结构复杂度 (obstacle)
# Change: 单条蜿蜒走廊，>=3个转弯，无分叉，通过粗粒度Hamiltonian walk生成... → 带障碍物通道

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D11 obstacle variant
# ============================================
# a) 具体化与行为追踪
# - 保留原题的三类核心对象：背景、蓝色河道、两个同尺寸实心方块。
# - 变化点是：蓝色河道不再是“只有宽窄变化的一条空走廊”，而是宽一些的通道，
#   其中嵌入少量障碍色小多连块。障碍色既不是背景也不是河流色，视觉上像河道里的
#   “岛”或“石块”。黑色移动块仍是 n×n 实心方块，红色目标块仍是同尺寸方块。
# - 小例子追踪：
#   原题在某段横向走廊里，如果整段宽度 >= n，则 BFS 会一直直行；若中间缩到 < n，
#   则在闸口前停住。
#   新变体里，同样一段区域可能整体宽度足够，但中间插入障碍岛。此时“像素连通”
#   不等于“n×n 占位连通”：移动块必须沿着整块可放置的位置绕开障碍；若障碍封死了
#   所有可放置路线，则虽然蓝色像素可能仍连通，方块依然到不了目标。
# - 因而变化真正影响的是“可行占位集合”的构造：从原题的只受边界宽度控制，
#   变成既受边界宽度控制，也受内部障碍切掉的窗口影响。
#
# b) 复杂度校准
# - 为了让 D11 肉眼可见，生成器不再使用 n=1。单像素方块绕障碍的效果太弱，容易
#   退化成普通路径连通；这里固定 n∈{2,3,4}，确保“整块窗口”约束显著。
# - 障碍数量控制在 2-6 个组件，颜色只引入 1 种，避免颜色映射过多。
# - 障碍形状只用非实心方形的小多连块（骨牌、L 形、三连条），避免和 mover/dest
#   的“实心正方形”身份冲突。
#
# c) 隐式假设审查（针对原 main 的关键写法）
# - 原代码 `cand.sort(key=lambda s: int(s["col"]))` 依赖“较小颜色 id 一定是 mover”。
#   这在 ARC 观察上不可见，也被本题额外要求禁止。新 solver 改为：
#   触边的同尺寸方块 = destination；内部近边但不触边的同尺寸方块 = mover。
#   这个差异在输入里可见，而且原生成器本来就满足“dest 贴边、mover 不贴边”。
# - 原代码把“除了 bg 和两方块色之外最常见的颜色”当 river。加入障碍色后这仍可用，
#   但前提是河道面积必须明显大于障碍面积。生成器显式保证这一点。
# - 原代码 `trav = river ∪ mover ∪ dest` 在障碍变体中仍成立；障碍色天然被排除。
# - 原代码的 BFS 是对 n×n top-left placement 做搜索，这正好是本变体需要保留的
#   机制，因为障碍影响的就是“哪些窗口仍完整可放置”。
#
# d) 放置合法性
# - dest 的 top-left 必须贴边，mover 的 top-left 必须位于内圈（距离某边正好 1）。
# - 宽河道先从 placement-path 渲染出来，再在河道内部放障碍；障碍不得覆盖 mover/dest。
# - 障碍组件必须完全落在河道像素上，且不得越界。
# - 生成后用 solver 自检，确保不会触发 assert，且输出与显式构造的 expected 一致。
#
# e) 视觉可观察性
# - 样例中人类能直接看到：蓝河里有别色小块，黑方块走到输出时绕过它们或者被它们挡住。
# - mover/dest 的角色通过“是否贴边”区分，而不是靠颜色编号排序。
# - 生成器还会检查“去掉障碍后”的路径性质：
#   reachable 样例要求去障碍后最短路更短；
#   blocked 样例要求去障碍后原本不可达变成可达。
#   这保证 D11 不是装饰性的。
#
# f) 可视化整洁度
# - 通道由单条蜿蜒 placement 主路径膨胀得到，因此整体仍像一条主河道，而不是满屏噪声。
# - 障碍组件小且数量有限，并避开方块本体，避免对象层叠混乱。
#
# g) 颜色变化可推断性
# - 颜色集合固定为：背景 3、河道 1、移动块 0、目标块 2、障碍 8。
# - 输出只发生“黑块移动、起点变回蓝色”这一种变化；障碍颜色不变化。
# - 没有任何依赖颜色 id 顺序的隐藏规则。
#
# h) 参考任务说明
# - 提示里把 `cb2d8a2c` 作为 pathfinding 灵感来源，但本工作区没有其 solver 源码，
#   只有官方 I/O JSON。这里据其样例可见特征采用“蓝色可通行区中存在内部背景/障碍，
#   需要按可放置占位而非仅像素连通来判断路线”的机制。
#
# i) 新 solver 的输入假设与 generator 的对应保证
# - 假设 1：存在两块同尺寸实心正方形，其中恰有一块贴边。generator 显式构造。
# - 假设 2：河道色是除背景/两方块色之外面积最大的颜色。generator 让河道远大于障碍。
# - 假设 3：障碍组件不会伪装成额外的实心正方形候选。generator 只放非实心方形组件，
#   并额外检查障碍连通块不形成实心正方形。


_BG, _RIVER, _MOVER, _DEST, _OBST = 3, 1, 0, 2, 8
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_OBSTACLE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
]


def _touches_boundary(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _edge_distance(r0: int, c0: int, n: int, H: int, W: int) -> int:
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _find_square_candidates(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        if h != w:
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append(
            {
                "col": col,
                "r0": int(b[0]),
                "c0": int(b[1]),
                "n": h,
                "touch": _touches_boundary(int(b[0]), int(b[1]), h, H, W),
                "edge_dist": _edge_distance(int(b[0]), int(b[1]), h, H, W),
            }
        )
    return squares


def _select_mover_and_dest(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    squares = _find_square_candidates(g, bg)
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    for n in sorted(by_n.keys(), reverse=True):
        group = by_n[n]
        touching = [sq for sq in group if sq["touch"]]
        interior = [sq for sq in group if not sq["touch"]]
        if not touching or not interior:
            continue
        mover = min(interior, key=lambda sq: (int(sq["edge_dist"]), int(sq["r0"]), int(sq["c0"])))
        dest = min(touching, key=lambda sq: (int(sq["r0"]), int(sq["c0"])))
        return mover, dest

    raise AssertionError("No valid mover/destination square pair found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a solid square through a winding river that contains internal obstacle islands.

    Input structure:
    - The corner-majority background surrounds one connected river color.
    - A solid NxN mover square sits on the river just inside an outer edge, and a same-sized destination square touches the border.
    - Small obstacle-color polyominoes can appear inside the river and must be treated as walls.

    Rule:
    - The mover advances by unit top-left steps over positions whose full NxN footprint stays on river, mover, or destination cells.
    - Obstacle cells cut holes inside the river, so the square may need to detour around them.
    - If the destination placement is reachable, the mover covers it; otherwise it stops at the farthest reachable placement and its start becomes river.

    Key insight: You must reason about the feasible NxN placement windows, not just raw blue-pixel connectivity, because obstacle islands can break or reroute the square's path.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _select_mover_and_dest(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2 and 0 <= dr < h2 and 0 <= dc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr2, dc2 in _DIR4:
            rr, cc = r + dr2, c + dc2
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))

    if bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _analyze_grid(grid: np.ndarray, sr: int, sc: int, dr: int, dc: int, n: int) -> dict | None:
    H, W = grid.shape
    h2, w2 = H - n + 1, W - n + 1
    if not (0 <= sr < h2 and 0 <= sc < w2 and 0 <= dr < h2 and 0 <= dc < w2):
        return None

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[sr, sc]):
        return None

    dist = _bfs_placement(feasible, sr, sc)
    dest_ok = bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if dest_ok:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    return {
        "feasible": feasible,
        "dist": dist,
        "dest_ok": dest_ok,
        "dest_dist": int(dist[dr, dc]) if dest_ok else -1,
        "stop": (int(tr), int(tc)),
    }


def _recover_shortest_path(dist: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    if int(dist[dr, dc]) < 0:
        return []
    path = [(dr, dc)]
    cur = (dr, dc)
    while cur != (sr, sc):
        r, c = cur
        found = False
        for dr2, dc2 in _DIR4:
            rr, cc = r + dr2, c + dc2
            if 0 <= rr < dist.shape[0] and 0 <= cc < dist.shape[1] and int(dist[rr, cc]) == int(dist[r, c]) - 1:
                cur = (int(rr), int(cc))
                path.append(cur)
                found = True
                break
        if not found:
            return []
    path.reverse()
    return path


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.7:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in _DIR4:
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in _DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path and best_len >= total * 0.85:
            break
    return best_path


def _compute_offsets(num_positions: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = num_positions - 2
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    offsets = [int(round(first + i * step)) for i in range(num_coarse)]
    for i in range(len(offsets) - 1):
        if offsets[i + 1] - offsets[i] < 2:
            return None
    return offsets


def _expand_waypoints(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = [waypoints[0]]
    for nr, nc in waypoints[1:]:
        r, c = out[-1]
        if r != nr and c != nc:
            raise AssertionError("Waypoints must be axis-aligned")
        while r != nr:
            r += 1 if nr > r else -1
            out.append((r, c))
        while c != nc:
            c += 1 if nc > c else -1
            out.append((r, c))
    return out


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev = None
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        cur = (dr, dc)
        if prev is not None and cur != prev:
            turns += 1
        prev = cur
    return turns


def _placement_mask(h2: int, w2: int, path: list[tuple[int, int]], pad: int) -> np.ndarray:
    mask = np.zeros((h2, w2), dtype=bool)
    for r, c in path:
        for dr in range(-pad, pad + 1):
            span = pad - abs(dr)
            for dc in range(-span, span + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h2 and 0 <= cc < w2:
                    mask[rr, cc] = True
    return mask


def _render_river(H: int, W: int, n: int, placements: np.ndarray) -> np.ndarray:
    grid = np.full((H, W), _BG, dtype=int)
    for r, c in zip(*np.where(placements)):
        grid[r : r + n, c : c + n] = _RIVER
    return grid


def _square_pixels(r0: int, c0: int, n: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + n) for c in range(c0, c0 + n)}


def _obstacles_are_safe(grid: np.ndarray) -> bool:
    for comp in find_components(grid, bg=_BG, connectivity=4, monochromatic=True, return_format="dict"):
        if int(comp["color"]) != _OBST:
            continue
        b = comp["bbox"]
        if int(bbox_h(b)) != int(bbox_w(b)):
            continue
        patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if bool((patch == _OBST).all()):
            return False
    return True


def _place_shape_near_window(
    grid: np.ndarray,
    placement: tuple[int, int],
    n: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    r, c = placement
    candidates: list[list[tuple[int, int]]] = []
    for shape in random.sample(_OBSTACLE_SHAPES, len(_OBSTACLE_SHAPES)):
        max_dr = max(dr for dr, _ in shape)
        max_dc = max(dc for _, dc in shape)
        r_lo = max(1, r - 1)
        r_hi = min(H - 2, r + n)
        c_lo = max(1, c - 1)
        c_hi = min(W - 2, c + n)
        for ar in range(r_lo, r_hi + 1):
            for ac in range(c_lo, c_hi + 1):
                cells = [(ar + dr, ac + dc) for dr, dc in shape]
                if any(rr < 0 or rr >= H or cc < 0 or cc >= W for rr, cc in cells):
                    continue
                if any(rr in (0, H - 1) or cc in (0, W - 1) for rr, cc in cells):
                    continue
                if any((rr, cc) in forbidden for rr, cc in cells):
                    continue
                if any(grid[rr, cc] != _RIVER for rr, cc in cells):
                    continue
                if ar + max_dr >= H or ac + max_dc >= W:
                    continue
                candidates.append(cells)
    random.shuffle(candidates)
    for cells in candidates:
        for rr, cc in cells:
            grid[rr, cc] = _OBST
        if _obstacles_are_safe(grid):
            return True
        for rr, cc in cells:
            grid[rr, cc] = _RIVER
    return False


def _make_expected(grid: np.ndarray, sr: int, sc: int, tr: int, tc: int, n: int) -> np.ndarray:
    out = grid.copy()
    out[sr : sr + n, sc : sc + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.55
    n = random.choice([2, 3, 4])

    min_size = max(12, 4 * n + 4)
    if min_size > 20:
        return None
    H = W = random.randint(min_size, 20)
    h2 = H - n + 1
    w2 = W - n + 1

    coarse_dim_max = min(5, (min(h2, w2) + 1) // 2)
    if coarse_dim_max < 3:
        return None
    coarse_dim = random.randint(3, coarse_dim_max)
    offsets_r = _compute_offsets(h2, coarse_dim)
    offsets_c = _compute_offsets(w2, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None

    edge_cells = []
    for c in range(1, coarse_dim - 1):
        edge_cells.append((0, c))
        edge_cells.append((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, coarse_dim - 1))
    if len(edge_cells) < 2:
        return None

    start_coarse = random.choice(edge_cells)
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]
    walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None

    waypoints = [(offsets_r[r], offsets_c[c]) for r, c in walk]
    end_r, end_c = waypoints[-1]
    if end_coarse[0] == 0:
        dest_waypoint = (0, end_c)
    elif end_coarse[0] == coarse_dim - 1:
        dest_waypoint = (h2 - 1, end_c)
    elif end_coarse[1] == 0:
        dest_waypoint = (end_r, 0)
    else:
        dest_waypoint = (end_r, w2 - 1)
    waypoints.append(dest_waypoint)

    placement_path = _expand_waypoints(waypoints)
    sr, sc = placement_path[0]
    dr, dc = placement_path[-1]
    if _touches_boundary(sr, sc, n, H, W):
        return None
    if not _touches_boundary(dr, dc, n, H, W):
        return None
    if _edge_distance(sr, sc, n, H, W) != 1:
        return None
    if _count_turns(placement_path) < 3:
        return None

    pad = 2 if n == 2 and min(h2, w2) >= 12 and random.random() < 0.4 else 1
    placements = _placement_mask(h2, w2, placement_path, pad)
    if not bool(placements[sr, sc]) or not bool(placements[dr, dc]):
        return None

    base_grid = _render_river(H, W, n, placements)
    if base_grid[0, 0] != _BG or base_grid[0, -1] != _BG or base_grid[-1, 0] != _BG or base_grid[-1, -1] != _BG:
        return None

    clear_grid = base_grid.copy()
    clear_grid[sr : sr + n, sc : sc + n] = _MOVER
    clear_grid[dr : dr + n, dc : dc + n] = _DEST
    clear_info = _analyze_grid(clear_grid, sr, sc, dr, dc, n)
    if clear_info is None or not clear_info["dest_ok"]:
        return None
    clear_path = _recover_shortest_path(clear_info["dist"], sr, sc, dr, dc)
    if len(clear_path) < 8:
        return None

    forbidden = _square_pixels(sr, sc, n) | _square_pixels(dr, dc, n)
    usable_indices = list(range(max(2, len(clear_path) // 5), min(len(clear_path) - 2, len(clear_path) * 4 // 5)))
    if len(usable_indices) < 2:
        return None

    for _obs_attempt in range(140):
        trial = base_grid.copy()
        if reached:
            choose_n = random.randint(2, min(4, len(usable_indices)))
            chosen = sorted(random.sample(usable_indices, choose_n))
        else:
            center = random.choice(usable_indices)
            span = random.randint(2, 4)
            chosen = [idx for idx in range(center, min(center + span, len(clear_path) - 2))]
            if len(chosen) < 2:
                continue

        placed = 0
        for idx in chosen:
            if _place_shape_near_window(trial, clear_path[idx], n, forbidden):
                placed += 1
        extra = random.randint(0, 2)
        for _ in range(extra):
            idx = random.choice(usable_indices)
            if _place_shape_near_window(trial, clear_path[idx], n, forbidden):
                placed += 1
        if placed < 2:
            continue

        trial[sr : sr + n, sc : sc + n] = _MOVER
        trial[dr : dr + n, dc : dc + n] = _DEST
        info = _analyze_grid(trial, sr, sc, dr, dc, n)
        if info is None:
            continue
        cleared = np.where(trial == _OBST, _RIVER, trial)
        clear_again = _analyze_grid(cleared, sr, sc, dr, dc, n)
        if clear_again is None or not clear_again["dest_ok"]:
            continue

        obstacle_cells = int(np.sum(trial == _OBST))
        if obstacle_cells < 4:
            continue

        if reached:
            if not info["dest_ok"]:
                continue
            if int(info["dest_dist"]) < int(clear_again["dest_dist"]) + 2:
                continue
        else:
            if info["dest_ok"]:
                continue
            if tuple(info["stop"]) == (sr, sc):
                continue

        expected = _make_expected(trial, sr, sc, info["stop"][0], info["stop"][1], n)
        if not np.array_equal(main(trial), expected):
            continue
        return trial
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid obstacle-channel input after 400 attempts")
