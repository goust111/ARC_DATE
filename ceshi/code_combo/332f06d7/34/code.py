# Combo variant of 332f06d7 (index 34)
# combo_spec: D6/4+D13/0+D19/2
# Name: 卡住变色+停卡点前方+颜色线索可达

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


"""
design_analysis:
  fused_rule:
    - 河道现在有两种蓝色语义:
      - 深蓝(1) = 该段对任意 n 都足够宽
      - 浅蓝(8) = 只对 n=1 等价可通，对 n>=2 语义上等价“窄于 n 的卡点”
    - 输出也有两种终态:
      - 成功到达红块: 黑块覆盖红块
      - 被卡住: 黑块不会继续滑向其他支路，而是停在通往红块的主路上、第一段浅蓝前的最后一个深蓝停泊位，并变成橙色(7)
  behavior_trace:
    small_example:
      - 设 n=2，黑块沿深蓝主河道前进，主路后半段被整段染成浅蓝，红块在浅蓝尽头
      - 同时在浅蓝之前的某个拐点接出一条深蓝支路
      - 原题:
        - 只看几何宽度；若整条物理河道都够宽，就会继续穿过浅蓝并到达红块
      - 仅做 D19/2:
        - 会把浅蓝当作“窄段”，但若不可达时仍按最远 BFS 点停下，黑块可能跑到支路尽头
      - 本融合版本:
        - 先用“物理河道”找从起点到红块的主路
        - 再用“颜色语义”判断哪些停泊位真的可达
        - 第一次遇到主路上的浅蓝阻断时，停在它前一格，并变橙
      - 差异:
        - 浅蓝既提供“为什么过不去”的线索
        - 停止位置又反推出“卡点就在前方”
        - 橙色进一步显式编码“失败而非成功到达”
  complexity_calibration:
    - 新增颜色只有 2 个语义位:
      - 浅蓝作为宽度线索
      - 橙色作为失败终态
    - 关键自由度保持二元:
      - 深蓝/浅蓝
      - 成功/失败
    - 为了让 D13/0 不退化，stuck 样例强制在卡点上游加入可达支路，并拒绝 “旧 farthest 策略与新前停策略相同” 的样例
  implicit_assumption_audit:
    main_lines:
      - 原版 `river = most common remaining color` 失效，因为现在河道允许两种颜色；solver 必须显式识别深蓝/浅蓝
      - 原版“可通行像素 == 单一河流色 ∪ mover ∪ dest”失效；现在要区分:
        - 物理可通行: 深蓝 ∪ 浅蓝 ∪ mover ∪ dest
        - 语义可通行: n=1 时同上，n>=2 时去掉浅蓝
      - 原版不可达时扫描 `max(dist)` 失效；本题必须沿“忽略颜色时通往红块的物理最短路”找第一段不可语义通过的位置
      - 原版默认起点底色就是唯一河流色；现在新增假设“mover 起点总在深蓝上”，这样输出把原位恢复成深蓝才一致。generator 会保证
      - 输出橙色只出现在结果中，输入绝不预置橙色，避免对象检测冲突
    new_assumptions:
      - 红块在物理层面始终可由整条宽河道连到，只是语义层面可能被浅蓝段阻断
      - stuck 时，通往红块的主路上至少存在一段浅蓝前沿，且其前一格是合法深蓝停泊位
      - 若存在其他可达支路，solver 也不能被它吸走；因此目标选择必须绑定“通往红块的主路”而非“所有可达点里最远”
  placement_legality:
    - 生成器先画一条主路，再把红块延伸到边界，最后可选地在主路上游接一条深蓝支路
    - 所有粗网格停泊位都按固定步长 `n+1` 分开，避免非预期粘连
    - 每个样例都重新计算 n×n placement graph，并验证:
      - 起点语义可行
      - 红块物理可达
      - `main(input)` 与构造出的期望输出完全一致
  visual_observability:
    - 人类可同时读出三层信号:
      - 浅蓝后缀提示“后面那段对大方块不够宽”
      - 停在浅蓝前方提示“卡点就在那里”
      - 橙色提示“这次没到终点”
    - 另外保留部分 n=1 且含浅蓝的成功样例，帮助推断“浅蓝不是绝对墙，而是和方块尺寸联动”
  cleanliness:
    - 所有标记都复用原河道/终态位置，不叠加额外符号
    - 支路只在卡点上游出现，且保持深蓝，避免与浅蓝阻断段混在一起
  inferability:
    - 浅蓝总是成段出现，而不是零散噪点
    - 橙色只在失败输出出现，且位置总紧贴浅蓝前沿
    - 这让少量样例就能归纳出 “浅蓝=宽度线索、橙色=被卡住”
"""


_BG = 3
_WIDE = 1
_DEST = 2
_MOVER = 0
_STUCK = 7
_NARROW = 8
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_mover_and_dest(grid: np.ndarray, bg: int) -> tuple[dict, dict]:
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
    cand = list(by_n[int(nside)])
    assert len(cand) >= 2

    mover = next((sq for sq in cand if int(sq["col"]) == _MOVER), None)
    dest = next((sq for sq in cand if int(sq["col"]) == _DEST), None)
    if mover is not None and dest is not None:
        return mover, dest

    cand.sort(key=lambda sq: int(sq["col"]))
    return cand[0], cand[1]


def _physical_pixel_mask(grid: np.ndarray, mover_col: int, dest_col: int) -> np.ndarray:
    return (grid == _WIDE) | (grid == _NARROW) | (grid == mover_col) | (grid == dest_col)


def _semantic_pixel_mask(grid: np.ndarray, n: int, mover_col: int, dest_col: int) -> np.ndarray:
    mask = (grid == _WIDE) | (grid == mover_col) | (grid == dest_col)
    if n == 1:
        mask |= grid == _NARROW
    return mask


def _compute_feasible(pixel_mask: np.ndarray, n: int) -> np.ndarray:
    H, W = pixel_mask.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(pixel_mask[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs(feasible: np.ndarray, sr: int, sc: int, with_parent: bool = False):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] | None = {(sr, sc): None} if with_parent else None
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                if parent is not None:
                    parent[(rr, cc)] = (r, c)
                q.append((rr, cc))
    if with_parent:
        return dist, parent
    return dist


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None], end: tuple[int, int]
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int, int]:
    best_d = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best_d:
                best_d = int(dist[r, c])
                tr, tc = r, c
    return tr, tc, best_d


def _stop_before_first_blocked(
    physical_path: list[tuple[int, int]], semantic_dist: np.ndarray
) -> tuple[int, int]:
    for idx, (r, c) in enumerate(physical_path):
        if int(semantic_dist[r, c]) < 0:
            prev_r, prev_c = physical_path[max(0, idx - 1)]
            return prev_r, prev_c
    raise AssertionError("expected blocked point on physical path")


def _solve_target(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_mover_and_dest(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    semantic_feasible = _compute_feasible(_semantic_pixel_mask(g, n, int(mover["col"]), int(dest["col"])), n)
    physical_feasible = _compute_feasible(_physical_pixel_mask(g, int(mover["col"]), int(dest["col"])), n)
    h2, w2 = semantic_feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2
    assert 0 <= dr < h2 and 0 <= dc < w2
    assert bool(semantic_feasible[sr, sc])
    assert bool(physical_feasible[sr, sc])

    semantic_dist = _bfs(semantic_feasible, sr, sc)
    reached = bool(semantic_feasible[dr, dc]) and int(semantic_dist[dr, dc]) >= 0
    if reached:
        return {
            "reached": True,
            "n": n,
            "sr": sr,
            "sc": sc,
            "tr": dr,
            "tc": dc,
            "final_color": int(mover["col"]),
            "semantic_dist": semantic_dist,
        }

    physical_dist, parent = _bfs(physical_feasible, sr, sc, with_parent=True)
    assert int(physical_dist[dr, dc]) >= 0
    physical_path = _reconstruct_path(parent, (dr, dc))
    tr, tc = _stop_before_first_blocked(physical_path, semantic_dist)
    return {
        "reached": False,
        "n": n,
        "sr": sr,
        "sc": sc,
        "tr": tr,
        "tc": tc,
        "final_color": _STUCK,
        "semantic_dist": semantic_dist,
        "physical_path": physical_path,
    }


def main(grid: np.ndarray) -> np.ndarray:
    info = _solve_target(grid)
    out = np.array(grid, dtype=int).copy()
    n = int(info["n"])
    sr, sc = int(info["sr"]), int(info["sc"])
    tr, tc = int(info["tr"]), int(info["tc"])
    out[sr : sr + n, sc : sc + n] = _WIDE
    out[tr : tr + n, tc : tc + n] = int(info["final_color"])
    return out


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        d1 = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        d2 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if d1 != d2:
            turns += 1
    return turns


def _free_path_exists(
    rows: int, cols: int, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]
) -> bool:
    if start == goal:
        return True
    if goal in blocked:
        return False
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            nb = (rr, cc)
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if nb in blocked or nb in seen:
                continue
            if nb == goal:
                return True
            seen.add(nb)
            q.append(nb)
    return False


def _random_main_path(
    rows: int, cols: int, start: tuple[int, int], end: tuple[int, int], min_len: int, min_turns: int
) -> list[tuple[int, int]] | None:
    for _attempt in range(240):
        visited = {start}
        path = [start]
        while True:
            cur = path[-1]
            if cur == end and len(path) >= min_len and _path_turns(path) >= min_turns:
                return path

            candidates: list[tuple[int, int]] = []
            for dr, dc in _DIR4:
                nb = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                    continue
                if nb in visited:
                    continue
                if nb == end and len(path) + 1 < min_len:
                    continue
                test_blocked = set(visited)
                test_blocked.add(nb)
                if nb != end and not _free_path_exists(rows, cols, nb, end, test_blocked):
                    continue
                candidates.append(nb)

            if not candidates:
                break

            random.shuffle(candidates)

            def _score(nb: tuple[int, int]) -> tuple[int, int, int, int]:
                turn_bonus = 0
                if len(path) >= 2:
                    prev = path[-2]
                    d1 = (cur[0] - prev[0], cur[1] - prev[1])
                    d2 = (nb[0] - cur[0], nb[1] - cur[1])
                    if d1 != d2:
                        turn_bonus = 2
                free_degree = 0
                for dr2, dc2 in _DIR4:
                    rr2, cc2 = nb[0] + dr2, nb[1] + dc2
                    if 0 <= rr2 < rows and 0 <= cc2 < cols and (rr2, cc2) not in visited:
                        free_degree += 1
                manhattan = abs(nb[0] - end[0]) + abs(nb[1] - end[1])
                if len(path) < min_len - 2:
                    return (turn_bonus, free_degree, manhattan, random.randint(0, 7))
                return (turn_bonus, free_degree, -manhattan, random.randint(0, 7))

            nxt = max(candidates, key=_score)
            visited.add(nxt)
            path.append(nxt)
    return None


def _strict_offsets(G: int, n: int, count: int) -> list[int] | None:
    span = (count - 1) * (n + 1)
    max_start = G - n - 1 - span
    if max_start < 1:
        return None
    start = random.randint(1, max_start)
    return [start + i * (n + 1) for i in range(count)]


def _paint_bridge(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int, color: int) -> None:
    pr1, pc1 = p1
    pr2, pc2 = p2
    if pr1 == pr2:
        c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        grid[pr1 : pr1 + n, c_lo:c_hi] = color
    else:
        r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        grid[r_lo:r_hi, pc1 : pc1 + n] = color


def _paint_pixel_path(grid: np.ndarray, pixel_path: list[tuple[int, int]], n: int, color: int) -> None:
    for pr, pc in pixel_path:
        grid[pr : pr + n, pc : pc + n] = color
    for i in range(len(pixel_path) - 1):
        _paint_bridge(grid, pixel_path[i], pixel_path[i + 1], n, color)


def _edge_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    out = set()
    for c in range(cols):
        out.add((0, c))
        out.add((rows - 1, c))
    for r in range(1, rows - 1):
        out.add((r, 0))
        out.add((r, cols - 1))
    return list(out)


def _branch_cell_ok(
    rows: int,
    cols: int,
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    required_neighbor: tuple[int, int],
) -> bool:
    if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
        return False
    if cell in occupied:
        return False
    occ_neighbors = []
    for dr, dc in _DIR4:
        nb = (cell[0] + dr, cell[1] + dc)
        if nb in occupied:
            occ_neighbors.append(nb)
    return set(occ_neighbors) == {required_neighbor}


def _grow_branch(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    anchor: tuple[int, int],
    min_len: int = 2,
    max_len: int = 4,
) -> list[tuple[int, int]] | None:
    first_steps = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in _DIR4]
    random.shuffle(first_steps)
    for first in first_steps:
        if not _branch_cell_ok(rows, cols, first, occupied, anchor):
            continue
        branch = [first]
        used = set(occupied)
        used.add(first)
        target_len = random.randint(min_len, max_len)
        cur = first
        while len(branch) < target_len:
            next_steps = [(cur[0] + dr, cur[1] + dc) for dr, dc in _DIR4]
            random.shuffle(next_steps)
            picked = None
            for nb in next_steps:
                if _branch_cell_ok(rows, cols, nb, used, cur):
                    picked = nb
                    break
            if picked is None:
                break
            branch.append(picked)
            used.add(picked)
            cur = picked
        if len(branch) >= min_len:
            return branch
    return None


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
    corridor_colors = {_WIDE, _NARROW}
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, 0))
            elif all(grid[r, c] in corridor_colors | {_BG} for r in range(pr) for c in range(pc, pc + n) if c < W):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
            elif all(
                grid[r, c] in corridor_colors | {_BG}
                for r in range(pr + n, H)
                for c in range(pc, pc + n)
                if c < W
            ):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, 0))
            elif all(grid[r, c] in corridor_colors | {_BG} for r in range(pr, pr + n) for c in range(pc) if r < H):
                candidates.append((pr, 0, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
            elif all(
                grid[r, c] in corridor_colors | {_BG}
                for r in range(pr, pr + n)
                for c in range(pc + n, W)
                if r < H
            ):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return int(candidates[0][0]), int(candidates[0][1])


def _color_main_suffix(
    grid: np.ndarray,
    main_pixels: list[tuple[int, int]],
    cut_idx: int,
    n: int,
    dest_pos: tuple[int, int],
) -> None:
    for idx in range(cut_idx, len(main_pixels)):
        pr, pc = main_pixels[idx]
        grid[pr : pr + n, pc : pc + n] = _NARROW
    for idx in range(max(0, cut_idx - 1), len(main_pixels) - 1):
        _paint_bridge(grid, main_pixels[idx], main_pixels[idx + 1], n, _NARROW)
    tail = main_pixels[-1]
    if tail != dest_pos:
        grid[dest_pos[0] : dest_pos[0] + n, dest_pos[1] : dest_pos[1] + n] = _NARROW
        _paint_bridge(grid, tail, dest_pos, n, _NARROW)


def _grid_size_for_n(n: int) -> int:
    if n == 1:
        return random.randint(10, 16)
    if n == 2:
        return random.randint(12, 16)
    if n == 3:
        return random.randint(14, 20)
    return 20


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([1, 2, 3, 4])
    reached = True if n == 1 else random.random() < 0.45
    use_narrow_hint = (n == 1 and random.random() < 0.75) or (n >= 2 and not reached)

    G = _grid_size_for_n(n)
    H = W = G
    max_cells = 1 + (G - n - 2) // (n + 1)
    if max_cells < 4:
        return None
    coarse_dim = random.randint(4, min(6, max_cells))
    offsets_r = _strict_offsets(G, n, coarse_dim)
    offsets_c = _strict_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None

    edge_cells = _edge_cells(coarse_dim, coarse_dim)
    random.shuffle(edge_cells)
    start_cell = edge_cells[0]
    far = sorted(edge_cells, key=lambda cell: -(abs(cell[0] - start_cell[0]) + abs(cell[1] - start_cell[1])))
    end_cell = far[0] if far[0] != start_cell else far[1]

    min_main_len = max(7, coarse_dim + 2)
    main_path = _random_main_path(coarse_dim, coarse_dim, start_cell, end_cell, min_main_len, min_turns=3)
    if main_path is None or len(main_path) < 6:
        return None

    occupied = set(main_path)
    cut_idx = None
    branch: list[tuple[int, int]] | None = None
    branch_anchor_idx: int | None = None
    if n >= 2 and not reached:
        if len(main_path) < 7:
            return None
        cut_lo = max(2, len(main_path) // 2)
        cut_hi = len(main_path) - 2
        if cut_hi < cut_lo:
            return None
        cut_idx = random.randint(cut_lo, cut_hi)
        anchor_lo = max(1, len(main_path) // 4)
        anchor_hi = cut_idx - 1
        if anchor_hi < anchor_lo:
            return None
        anchor_indices = list(range(anchor_lo, anchor_hi + 1))
        random.shuffle(anchor_indices)
        for anchor_idx in anchor_indices:
            branch = _grow_branch(coarse_dim, coarse_dim, occupied, main_path[anchor_idx], min_len=2, max_len=4)
            if branch is not None:
                branch_anchor_idx = anchor_idx
                break
        if branch is None:
            return None

    grid = np.full((H, W), _BG, dtype=int)
    main_pixels = [(offsets_r[r], offsets_c[c]) for r, c in main_path]
    _paint_pixel_path(grid, main_pixels, n, _WIDE)

    if branch is not None:
        assert branch_anchor_idx is not None
        branch_pixels = [main_pixels[branch_anchor_idx]] + [(offsets_r[r], offsets_c[c]) for r, c in branch]
        _paint_pixel_path(grid, branch_pixels, n, _WIDE)

    river_area = float(np.mean(grid != _BG))
    if not (0.20 <= river_area <= 0.60):
        return None

    mover_r, mover_c = main_pixels[0]
    dest_r, dest_c = main_pixels[-1]

    result = _extend_to_edge(grid, dest_r, dest_c, n)
    if result is None:
        return None
    edge_dest_r, edge_dest_c = result
    if edge_dest_r != dest_r:
        r_lo, r_hi = min(edge_dest_r, dest_r), max(edge_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _WIDE
    if edge_dest_c != dest_c:
        c_lo, c_hi = min(edge_dest_c, dest_c), max(edge_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _WIDE
    dest_r, dest_c = edge_dest_r, edge_dest_c
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _WIDE
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = _WIDE
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _WIDE
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, W - 1] = _WIDE
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    if use_narrow_hint and cut_idx is None:
        cut_lo = max(1, len(main_pixels) // 3)
        cut_hi = len(main_pixels) - 2
        if cut_hi < cut_lo:
            return None
        cut_idx = random.randint(cut_lo, cut_hi)

    if cut_idx is not None:
        _color_main_suffix(grid, main_pixels, cut_idx, n, (dest_r, dest_c))

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    try:
        info = _solve_target(grid)
    except Exception:
        return None

    semantic_dist = info["semantic_dist"]
    if n >= 2 and reached:
        if not bool(info["reached"]):
            return None
    if n >= 2 and not reached:
        if bool(info["reached"]):
            return None
        old_tr, old_tc, old_best = _farthest_reachable(semantic_dist, mover_r, mover_c)
        if old_best <= 0:
            return None
        if (old_tr, old_tc) == (int(info["tr"]), int(info["tc"])):
            return None
        if int(grid[int(info["tr"]), int(info["tc"])]) != _WIDE:
            return None
        phys_path = info["physical_path"]
        blocked_idx = next((i for i, pos in enumerate(phys_path) if int(semantic_dist[pos[0], pos[1]]) < 0), None)
        if blocked_idx is None or blocked_idx == 0:
            return None
        blocked_r, blocked_c = phys_path[blocked_idx]
        if int(grid[blocked_r, blocked_c]) != _NARROW:
            return None
    if n == 1 and not bool(info["reached"]):
        return None

    output = grid.copy()
    out_n = int(info["n"])
    output[int(info["sr"]) : int(info["sr"]) + out_n, int(info["sc"]) : int(info["sc"]) + out_n] = _WIDE
    output[int(info["tr"]) : int(info["tr"]) + out_n, int(info["tc"]) : int(info["tc"]) + out_n] = int(
        info["final_color"]
    )
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
