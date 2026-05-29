# Combo variant of 332f06d7 (index 23)
# combo_spec: D11/0+D18/1
# Name: 颜色推断+图例通行规则

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
# - 原题里，背景固定为绿色、河道固定为蓝色，solver 可直接把“除背景和两方块外的主颜色”
#   当成河道，然后在该颜色上做 n×n placement BFS。
# - 本融合版里，背景色会跨样例变化；同时图中会出现两种“河道候选色”，只有图例标记为
#   pass 的那一种才可通行。图例固定放在某个角落的 4x4 区域，结构是两行，每行都是
#   `[2x2 颜色样本][2x2 状态样本]`。状态色 7 表示可通行，状态色 5 表示不可通行。
# - 具体小例子：样例 A 中，背景可能是 8，主允许河道是 4，伪延伸河道是 1；样例 B 中，
#   背景可能变成 6，允许河道变成 9，伪延伸河道变成 3。原题在这两个样例上都只会看到
#   “一条河”；融合版则要求先识别四角大底色是背景，再读图例，知道哪种河道色能让黑块滑动。
# - 行为差异：原题的卡住原因只有“几何宽度 < n”；融合版保留这个几何卡点，同时在卡点之后
#   用另一种颜色继续铺出一条看似可走的通路到红块。若忽略图例，会误以为能到达终点。
#
# b) 复杂度校准
# - 颜色自由度只给三类角色：背景、允许河道、阻塞河道，并限制它们都从一个小色池里采样。
# - 图例仍然只有两条规则，状态色固定重复出现，这样人类能在 2-4 个样例中稳定归纳
#   “右侧状态块决定左侧颜色是否可通行”，而不会陷入过多颜色映射。
# - mover 和 destination 仍固定为 0/2，避免再引入一层颜色角色推断。
#
# c) 隐式假设审查
# - 原 solver 的隐式假设 “背景固定/河道固定” 已失效，因此不能写死 `_BG/_RIVER`。
#   新 solver 先用四角多数色推断背景，再单独读图例得到 allowed color。
# - 原 solver 的隐式假设 “所有非背景单色方块都是题目对象” 也失效，因为图例中有 2x2
#   单色块。新 solver 必须先识别图例，并在做连通组件/方块检测前把图例区域遮回背景。
# - 原 solver 的隐式假设 “最常见剩余色就是河道” 失效，因为现在会同时出现允许河道色、
#   阻塞河道色和图例状态色。新 solver 完全不依赖频率选河道，而只依赖图例。
# - 新增假设 “起点所在底色就是唯一允许河道色”。generator 明确保证 mover 起点是从
#   allowed 河道上挖出来的，因此输出时把起点恢复为 allowed color 是正确的。
#
# d) 放置合法性
# - generator 先用原题方式构造一条 allowed 主河道，再人为加入几何 chokepoint，使
#   allowed-only 情况下目标不可达。
# - 随后从 allowed-only 最远可达位置到 destination 再铺一条 blocked-color 旁路，
#   并要求该旁路避开图例角落且仅覆盖背景像素，保证不会重叠破坏主河道、方块或图例。
# - 图例只放入完整 4x4 背景角块，因此不会越界，也不会遮住任务主体。
#
# e) 视觉可观察性
# - 解题者能直接看到：四角底色大多相同，但其中一个角落嵌了 4x4 图例；图例列出两种候选
#   河道色及其状态；黑块在输出中只沿其中一种颜色移动。
# - 跨样例时，背景和河道颜色都会换，但图例结构与状态色不变，因此“先读图例，再判颜色”
#   的规则是可见且可归纳的。
#
# f) 可视化整洁度
# - 图例被限制在单一角落小补丁内，状态色不会出现在主体区域。
# - 主河道与伪河道都保持原题那种规则的 n 宽走廊外观，没有额外噪声标记。
# - blocked 旁路只覆盖背景，不会把 allowed 河道染花成多层混叠。
#
# g) 变化的可推断性
# - 关键变化只有两层：背景颜色变、河道颜色是否可通行由图例决定。两层都有强结构线索：
#   背景由四角/大片空地给出，通行性由角落图例给出。
# - 为避免颜色太多，语义色总量控制为 7 类：背景、mover、destination、allowed、
#   blocked、pass-marker、block-marker；其中真正会跨样例变化的只有前三类角色色。
#
# main() 的新隐式假设与 generator 保证
# - 假设四角多数色就是背景：generator 保证图例只占一个角，另外三个角仍是纯背景。
# - 假设存在且仅存在一个合法 4x4 角落图例：generator 精确按该格式放置。
# - 假设遮掉图例后，0 和 2 是仅有的一对同尺寸任务方块：generator 保证这一点。
# - 假设图例里的 allowed 色恰好是 mover 起点下方的河道色：generator 保证起点来自
#   allowed 主河道，而 blocked 色只作为后续旁路。

_MOVER = 0
_DEST = 2
_PASS_MARK = 7
_BLOCK_MARK = 5
_LEGEND_SIZE = 4
_ROLE_COLORS = [1, 3, 4, 6, 8, 9]
_ROLE_TRIPLE_QUEUE: list[tuple[int, int, int]] = []


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _legend_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, w - _LEGEND_SIZE),
        (h - _LEGEND_SIZE, 0),
        (h - _LEGEND_SIZE, w - _LEGEND_SIZE),
    ]


def _detect_legend(grid: np.ndarray) -> dict:
    h, w = grid.shape
    for r0, c0 in _legend_positions(h, w):
        patch = grid[r0 : r0 + _LEGEND_SIZE, c0 : c0 + _LEGEND_SIZE]
        if patch.shape != (_LEGEND_SIZE, _LEGEND_SIZE):
            continue
        q00 = patch[0:2, 0:2]
        q01 = patch[0:2, 2:4]
        q10 = patch[2:4, 0:2]
        q11 = patch[2:4, 2:4]
        quads = [q00, q01, q10, q11]
        if not all(np.all(q == q[0, 0]) for q in quads):
            continue
        key_top = int(q00[0, 0])
        status_top = int(q01[0, 0])
        key_bottom = int(q10[0, 0])
        status_bottom = int(q11[0, 0])
        if {status_top, status_bottom} != {_PASS_MARK, _BLOCK_MARK}:
            continue
        if len({key_top, key_bottom}) != 2:
            continue
        if key_top in (_MOVER, _DEST, _PASS_MARK, _BLOCK_MARK):
            continue
        if key_bottom in (_MOVER, _DEST, _PASS_MARK, _BLOCK_MARK):
            continue
        allowed = []
        blocked = []
        for key, status in ((key_top, status_top), (key_bottom, status_bottom)):
            if status == _PASS_MARK:
                allowed.append(int(key))
            else:
                blocked.append(int(key))
        if len(allowed) == 1 and len(blocked) == 1:
            return {
                "r0": int(r0),
                "c0": int(c0),
                "allowed": allowed,
                "blocked": blocked,
                "all_channels": [int(key_top), int(key_bottom)],
            }
    raise AssertionError("legend not found")


def _mask_legend(grid: np.ndarray, legend: dict, bg: int) -> np.ndarray:
    out = grid.copy()
    out[legend["r0"] : legend["r0"] + _LEGEND_SIZE, legend["c0"] : legend["c0"] + _LEGEND_SIZE] = int(bg)
    return out


def _find_roles(grid: np.ndarray, bg: int) -> tuple[dict, dict]:
    squares: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert len(squares) >= 2
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    mover = next((sq for sq in cand if int(sq["col"]) == _MOVER), None)
    dest = next((sq for sq in cand if int(sq["col"]) == _DEST), None)
    assert mover is not None and dest is not None
    return mover, dest


def _solve_from_allowed(grid: np.ndarray, allowed_colors: list[int]) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    try:
        legend = _detect_legend(g)
    except AssertionError:
        legend = None
    work = _mask_legend(g, legend, bg) if legend is not None else g.copy()

    mover, dest = _find_roles(work, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    traversable = np.isin(work, list(allowed_colors) + [_MOVER, _DEST])
    h2 = int(h - n + 1)
    w2 = int(w - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(traversable[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = int(dr), int(dc)
    else:
        best = -1
        tr, tc = int(sr), int(sc)
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = int(r), int(c)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(allowed_colors[0])
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    legend = _detect_legend(g)
    return _solve_from_allowed(g, legend["allowed"])


def _next_role_triple() -> tuple[int, int, int]:
    global _ROLE_TRIPLE_QUEUE
    if not _ROLE_TRIPLE_QUEUE:
        _ROLE_TRIPLE_QUEUE = [
            (bg, allowed, blocked)
            for bg in _ROLE_COLORS
            for allowed in _ROLE_COLORS
            for blocked in _ROLE_COLORS
            if len({bg, allowed, blocked}) == 3
        ]
        random.shuffle(_ROLE_TRIPLE_QUEUE)
    return _ROLE_TRIPLE_QUEUE.pop()


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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
    color: int,
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = int(color)
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = int(color)
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = int(color)
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    n: int,
    direction: str,
    bg: int,
) -> None:
    h, w = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(6, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if not (0 <= c < w):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < h:
                    grid[rr, c] = int(bg)
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if not (0 <= r < h):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < w:
                    grid[r, cc] = int(bg)


def _river_connected(grid: np.ndarray, colors: list[int]) -> bool:
    mask = np.isin(grid, colors + [_MOVER, _DEST])
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(gsize: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = gsize - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _window_overlaps_rect(r: int, c: int, n: int, rect: tuple[int, int, int, int]) -> bool:
    rr0, cc0, rr1, cc1 = rect
    return not (r + n - 1 < rr0 or rr1 < r or c + n - 1 < cc0 or cc1 < c)


def _choose_legend_corner(grid: np.ndarray, bg: int) -> tuple[int, int] | None:
    h, w = grid.shape
    corners = _legend_positions(h, w)
    random.shuffle(corners)
    for r0, c0 in corners:
        patch = grid[r0 : r0 + _LEGEND_SIZE, c0 : c0 + _LEGEND_SIZE]
        if patch.shape == (_LEGEND_SIZE, _LEGEND_SIZE) and bool((patch == int(bg)).all()):
            return int(r0), int(c0)
    return None


def _find_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    h2: int,
    w2: int,
    n: int,
    forbidden_rect: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    if _window_overlaps_rect(start[0], start[1], n, forbidden_rect):
        return None
    if _window_overlaps_rect(goal[0], goal[1], n, forbidden_rect):
        return None

    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        order = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        random.shuffle(order)
        for dr, dc in order:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h2 and 0 <= cc < w2):
                continue
            if (rr, cc) in prev:
                continue
            if _window_overlaps_rect(rr, cc, n, forbidden_rect):
                continue
            prev[(rr, cc)] = (r, c)
            q.append((rr, cc))
    if goal not in prev:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _render_overlay_on_bg(grid: np.ndarray, path: list[tuple[int, int]], n: int, color: int, bg: int) -> None:
    for r, c in path:
        patch = grid[r : r + n, c : c + n]
        patch[patch == int(bg)] = int(color)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == r2:
            c_lo, c_hi = min(c1, c2), max(c1, c2) + n
            patch = grid[r1 : r1 + n, c_lo:c_hi]
            patch[patch == int(bg)] = int(color)
        else:
            r_lo, r_hi = min(r1, r2), max(r1, r2) + n
            patch = grid[r_lo:r_hi, c1 : c1 + n]
            patch[patch == int(bg)] = int(color)


def _place_legend(grid: np.ndarray, r0: int, c0: int, allowed_color: int, blocked_color: int) -> None:
    rows = [((allowed_color, _PASS_MARK), 0), ((blocked_color, _BLOCK_MARK), 2)]
    random.shuffle(rows)
    for (key, status), rr in rows:
        grid[r0 + rr : r0 + rr + 2, c0 : c0 + 2] = int(key)
        grid[r0 + rr : r0 + rr + 2, c0 + 2 : c0 + 4] = int(status)


def _gen_attempt(bg: int, allowed_color: int, blocked_color: int) -> np.ndarray | None:
    n = random.choice([2, 3, 4])
    min_size = max(10, 5 * n)
    if min_size > 20:
        return None

    gsize = random.randint(min_size, 20)
    h = w = gsize
    min_spacing = n + 1
    max_coarse = max(3, (gsize - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))

    offsets_r = _compute_offsets(gsize, n, coarse_dim)
    offsets_c = _compute_offsets(gsize, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
        return None

    edge_cells = set()
    for c in range(coarse_dim):
        edge_cells.add((0, c))
        edge_cells.add((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_dim - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_dim or len(set(cc for _, cc in walk)) < coarse_dim:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c, allowed_color)

    mask = grid != int(bg)
    if np.any(mask, axis=1).sum() / h < 0.70 or np.any(mask, axis=0).sum() / w < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
        candidates = []
        for direction in ("top", "bottom", "left", "right"):
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                elif all(grid[r, c] in (bg, allowed_color) for r in range(pr) for c in range(pc, pc + n) if c < w):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = h - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (bg, allowed_color) for r in range(pr + n, h) for c in range(pc, pc + n) if c < w):
                    candidates.append((h - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                elif all(grid[r, c] in (bg, allowed_color) for r in range(pr, pr + n) for c in range(pc) if r < h):
                    candidates.append((pr, 0, dist))
            else:
                dist = w - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (bg, allowed_color) for r in range(pr, pr + n) for c in range(pc + n, w) if r < h):
                    candidates.append((pr, w - n, dist))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2])
        return int(candidates[0][0]), int(candidates[0][1])

    result = _extend_to_edge(dest_r, dest_c)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = int(allowed_color)
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = int(allowed_color)
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == h or dest_c == 0 or dest_c + n == w):
        return None

    touching = mover_r == 0 or mover_r + n == h or mover_c == 0 or mover_c + n == w
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = int(allowed_color)
            mover_r = 1
        elif mover_r + n == h:
            grid[h - 1, mover_c : mover_c + n] = int(allowed_color)
            mover_r = h - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = int(allowed_color)
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, w - 1] = int(allowed_color)
            mover_c = w - n - 1
    if mover_r == 0 or mover_r + n == h or mover_c == 0 or mover_c + n == w:
        return None
    if min(mover_r, h - (mover_r + n), mover_c, w - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = int(allowed_color)
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = int(allowed_color)

    num_choke = random.randint(1, min(3, len(walk) - 2))
    min_seg = max(1, len(walk) // 4)
    available = list(range(min_seg, len(walk) - 1))
    random.shuffle(available)
    for seg_idx in available[:num_choke]:
        pr1, pc1 = pixel_pos[seg_idx]
        pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
        direction = "h" if pr1 == pr2 else "v"
        _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction, bg)
    if not _river_connected(grid, [allowed_color]):
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    allowed_output = _solve_from_allowed(grid, [allowed_color])
    if np.array_equal(allowed_output, grid):
        return None

    h2, w2 = h - n + 1, w - n + 1
    allowed_trav = np.isin(grid, [allowed_color, _MOVER, _DEST])
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(allowed_trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None
    dist = _bfs_placement(feasible, mover_r, mover_c)
    if 0 <= dest_r < h2 and 0 <= dest_c < w2 and bool(feasible[dest_r, dest_c]) and int(dist[dest_r, dest_c]) >= 0:
        return None

    tr, tc = mover_r, mover_c
    best_d = -1
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best_d:
                best_d = int(dist[r, c])
                tr, tc = int(r), int(c)
    if best_d <= 0:
        return None

    legend_corner = _choose_legend_corner(grid, bg)
    if legend_corner is None:
        return None
    legend_r0, legend_c0 = legend_corner
    forbidden_rect = (legend_r0, legend_c0, legend_r0 + _LEGEND_SIZE - 1, legend_c0 + _LEGEND_SIZE - 1)

    bypass_path = _find_path((tr, tc), (dest_r, dest_c), h2, w2, n, forbidden_rect)
    if bypass_path is None or len(bypass_path) < 2:
        return None

    grid2 = grid.copy()
    _render_overlay_on_bg(grid2, bypass_path, n, blocked_color, bg)
    _place_legend(grid2, legend_r0, legend_c0, allowed_color, blocked_color)

    new_out = main(grid2)
    expected = _solve_from_allowed(grid2, [allowed_color])
    if not np.array_equal(new_out, expected):
        return None

    all_color_out = _solve_from_allowed(grid2, [allowed_color, blocked_color])
    if np.array_equal(new_out, all_color_out):
        return None
    if not np.array_equal(all_color_out[dest_r : dest_r + n, dest_c : dest_c + n], np.full((n, n), _MOVER, dtype=int)):
        return None
    if np.array_equal(new_out[dest_r : dest_r + n, dest_c : dest_c + n], np.full((n, n), _MOVER, dtype=int)):
        return None

    return grid2


def generate_input() -> np.ndarray:
    for _ in range(400):
        bg, allowed_color, blocked_color = _next_role_triple()
        result = _gen_attempt(bg, allowed_color, blocked_color)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
