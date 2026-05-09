# Variant of 332f06d7
# Dimension: D11 — 通道结构复杂度 (obstacle)
# Change: 单条蜿蜒走廊，>=3个转弯，无分叉，通过粗粒度Hamiltonian walk生成... → 带分叉通道

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 输入仍是方形网格，绿色背景上有蓝色通道，黑色 mover 与红色 destination
#      仍是同尺寸的实心正方形。
#    - 关键变化不再是“单一走廊”，而是“树形通道”：从 mover 出发先经过一段蜿蜒主干，
#      到达唯一分叉点后分成两条支路。一条支路继续通向贴边的 destination，
#      另一条支路在内部终止，是没有目标的死胡同。
#    - 原题在 placement-grid 上的可行位置图近似一条链；这里改成有一个 degree>=3 的节点。
#      例子：原题 BFS 只有一条继续前进的路，变体里 BFS 到分叉点后会同时扩展两条路，
#      只有通向贴边红块的那条能完成任务，另一条只是可到达但无目标。
# b) 复杂度校准
#    - 每个样例固定只有一个主分叉点，避免多分叉导致视觉噪声。
#    - 保留 2 个同尺寸实心块与 1 个蓝色连通通道，不增加额外标记物。
#    - 分叉后的无目标支路必须足够长，确保人眼能明确看见“岔路”而不是小凸起。
# c) 隐式假设审查
#    - 原 solver 用颜色 id 排序来区分 mover/destination，这在视觉上不可见；
#      变体改为用“谁贴边谁是 destination”来区分，generator 也强制保证这一点。
#    - 原 solver 假设通道是否可达由 BFS 自然决定，这在变体里仍成立；
#      因为分叉只改变可行位置图的拓扑，不改变 n×n 整块滑动规则。
#    - 原 solver 不依赖“唯一走廊”的代码段主要是 BFS 本身，因此 solver 的核心搜索逻辑
#      可以保留；需要改的是对象识别假设而不是搜索方式。
# d) 放置合法性
#    - generator 直接在 placement-grid 中构造树形骨架，再把每个可放置位置膨胀成 n×n 河道。
#    - mover 放在距边界 1 格的位置，destination 贴某一条边但不占角，死胡同端点不放对象。
#    - 构造后重新计算 feasible placements，只有在 solver 看到的实际可行图里仍含分叉、
#      destination 可达、且 mover 起点可行时才接受样本。
# e) 视觉可观察性
#    - 不使用颜色编号做任何排序或角色映射；角色线索完全来自空间关系：
#      贴边方块是目标，内侧方块是移动体。
#    - 样例中会稳定出现“三个叶子端点”：mover 起点、destination 终点、死胡同终点。
# f) 可视化整洁度
#    - 只允许一个主要分叉点，避免多个交叉口形成杂乱蓝块。
#    - destination 不放在角上，保证四角仍是背景色且红块更容易辨认。
# g) 颜色变化可推断性
#    - 颜色语义不变：蓝色是通道，黑色是被移动方块，红色是目标方块，绿色是背景。
#
# New main() assumptions checked against generator
# -----------------------------------------------
# - 恰有一组“同尺寸实心非背景方块”同时包含 1 个贴边块和 1 个非贴边块。
# - 通道颜色是除背景和这两个方块颜色外的主要剩余颜色。
# - mover 的 n×n 初始窗口一定是 feasible。
# - 输入中的实际 feasible 图至少有 1 个 reachable junction (degree>=3) 与 3 个叶子，
#   所以分叉行为不是名义上的。


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _touches_border(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _extract_squares(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        r0, c0, r1, c1 = map(int, b)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": r0, "c0": c0, "n": int(bbox_h(b))})
    return squares


def _identify_roles(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares = _extract_squares(g, bg)
    assert len(squares) >= 2

    groups: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        groups[int(sq["n"])].append(sq)

    candidates: list[tuple[int, int, list[dict], list[dict], list[dict]]] = []
    for nside, group in groups.items():
        touching = [sq for sq in group if _touches_border(int(sq["r0"]), int(sq["c0"]), int(sq["n"]), H, W)]
        inner = [sq for sq in group if sq not in touching]
        if len(touching) == 1 and len(inner) >= 1:
            candidates.append((len(group), int(nside), group, touching, inner))

    assert candidates
    candidates.sort(key=lambda x: (-int(x[0]), -int(x[1])))
    _count, _nside, _group, touching, inner = candidates[0]

    dest = touching[0]
    mover = min(
        inner,
        key=lambda sq: min(
            int(sq["r0"]),
            int(sq["c0"]),
            int(H - (sq["r0"] + sq["n"])),
            int(W - (sq["c0"] + sq["n"])),
        ),
    )
    return mover, dest


def _window_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


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


def _pick_terminal(dist: np.ndarray, sr: int, sc: int, dr: int, dc: int, feasible: np.ndarray) -> tuple[int, int]:
    h2, w2 = feasible.shape
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        return dr, dc

    best = -1
    tr, tc = sr, sc
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a branched blue channel toward an equal-sized edge target.

    Input structure:
    - The corner-majority color is background, and a blue connected channel occupies the interior.
    - Two same-sized solid squares sit on the channel: one interior mover and one edge-touching target.
    - The channel contains a single visible branch, so one side path is a dead end.

    Rule:
    - Move the interior square along the channel using 4-neighbor unit slides of its top-left corner.
    - A move is legal only when the entire n×n body stays inside channel/occupied river pixels.
    - If the edge-touching target position is reachable, the mover ends there; otherwise it stops at the
      farthest reachable placement.
    - The mover's starting square is repainted as river after the slide.

    Key insight: Work in the placement grid of feasible n×n windows; the branch only changes the graph shape,
    so BFS naturally explores both branches and selects the one that reaches the edge target.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _identify_roles(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col == bg:
            continue
        if col in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _window_feasible(trav, n)
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    tr, tc = _pick_terminal(dist, sr, sc, dr, dc, feasible)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        r1, c1 = points[i]
        r2, c2 = points[i + 1]
        if r1 == r2:
            step = 1 if c2 >= c1 else -1
            seg = [(r1, c) for c in range(c1, c2 + step, step)]
        else:
            assert c1 == c2
            step = 1 if r2 >= r1 else -1
            seg = [(r, c1) for r in range(r1, r2 + step, step)]
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _render_traversable(H: int, W: int, n: int, placements: set[tuple[int, int]]) -> np.ndarray:
    trav = np.zeros((H, W), dtype=bool)
    for r, c in placements:
        trav[r : r + n, c : c + n] = True
    return trav


def _build_branch_skeleton(P: int) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    if P < 9:
        return None

    dr_hi = max(1, P // 4)
    dr = random.randint(1, dr_hi)

    jr_lo = max(dr + 2, P // 2 - 1)
    jr_hi = min(P - 4, P // 2 + 1)
    if jr_lo > jr_hi:
        return None
    jr = random.randint(jr_lo, jr_hi)

    er_lo = jr + 2
    er_hi = P - 2
    if er_lo > er_hi:
        return None
    er = random.randint(er_lo, er_hi)
    sr = er

    r2_lo = dr + 1
    r2_hi = jr - 1
    if r2_lo > r2_hi:
        return None
    r2 = random.randint(r2_lo, r2_hi)

    jc_lo = max(4, P // 2 - 1)
    jc_hi = min(P - 3, P // 2 + 1)
    if jc_lo > jc_hi:
        return None
    jc = random.randint(jc_lo, jc_hi)

    if jc - 3 < 2:
        return None
    c1 = random.randint(2, jc - 3)

    c2_lo = c1 + 1
    c2_hi = jc - 1
    if c2_lo > c2_hi:
        return None
    c2 = random.randint(c2_lo, c2_hi)

    ec_lo = jc + 1
    ec_hi = P - 2
    if ec_lo > ec_hi:
        return None
    ec = random.randint(ec_lo, ec_hi)

    start = (sr, 1)
    junction = (jr, jc)
    dest = (dr, P - 1)
    dead_end = (er, ec)

    trunk = _polyline_cells([start, (sr, c1), (r2, c1), (r2, c2), (jr, c2), junction])
    goal = _polyline_cells([junction, (dr, jc), dest])
    dead = _polyline_cells([junction, (er, jc), dead_end])

    counts: Counter[tuple[int, int]] = Counter(trunk + goal + dead)
    if any(v > 1 and k != junction for k, v in counts.items()):
        return None

    placements = set(counts.keys())
    return start, dest, placements


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    return out


def _reachable_shape_metrics(feasible: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    dist = _bfs_placement(feasible, sr, sc)
    junctions = 0
    leaves = 0
    h2, w2 = feasible.shape
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) < 0:
                continue
            deg = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and int(dist[rr, cc]) >= 0:
                    deg += 1
            if deg >= 3:
                junctions += 1
            if deg == 1:
                leaves += 1
    return junctions, leaves


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([1, 2, 3, 4])
    min_size = max(10, 4 * n + 4)
    if min_size > 20:
        return None

    G = random.randint(min_size, 20)
    H = W = G
    P = G - n + 1

    built = _build_branch_skeleton(P)
    if built is None:
        return None
    start, dest, placements = built

    trav = _render_traversable(H, W, n, placements)
    area_ratio = float(trav.mean())
    if not (0.18 <= area_ratio <= 0.60):
        return None

    sr, sc = start
    dr, dc = dest
    grid = np.full((H, W), _BG, dtype=int)
    grid[trav] = _RIVER
    grid[sr : sr + n, sc : sc + n] = _MOVER
    grid[dr : dr + n, dc : dc + n] = _DEST
    grid = _random_transform(grid)

    bg = int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])
    mover, dest_sq = _identify_roles(grid, bg)
    if int(mover["n"]) != n or int(dest_sq["n"]) != n:
        return None

    trav2 = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _window_feasible(trav2, n)
    sr2, sc2 = int(mover["r0"]), int(mover["c0"])
    dr2, dc2 = int(dest_sq["r0"]), int(dest_sq["c0"])
    if not bool(feasible[sr2, sc2]):
        return None

    dist = _bfs_placement(feasible, sr2, sc2)
    if int(dist[dr2, dc2]) < 0:
        return None

    junctions, leaves = _reachable_shape_metrics(feasible, sr2, sc2)
    if junctions < 1 or leaves < 3:
        return None

    out = main(grid)
    mover_col = int(mover["col"])
    river_mask = out[sr2 : sr2 + n, sc2 : sc2 + n]
    dest_mask = out[dr2 : dr2 + n, dc2 : dc2 + n]
    if not bool((river_mask == _RIVER).all()):
        return None
    if not bool((dest_mask == mover_col).all()):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
