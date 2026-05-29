# Combo variant of 332f06d7 (index 6)
# combo_spec: D3/0+D6/4
# Name: 连续轨迹+卡住变色

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


"""
design_analysis:
  concrete_behavior:
    - 输入仍是原题结构：绿色背景、蓝色蜿蜒河道、黑色 n×n mover、红色 n×n destination。
    - 输出融合两项变化：
      1. mover 经过的每个可行 placement 都留下连续黑色轨迹，不再刷回蓝色。
      2. 若 destination 不可达，则最终停住的 n×n 方块改为橙色，而不是黑色。
    - 因此像素级可见模式是：
      - 可达：从起点到终点出现连续黑轨迹，末端覆盖红块。
      - 卡住：从起点到卡点前出现连续黑轨迹，最前端停点变成橙块，红块仍保留在远处。
  behavior_trace:
    - 例子：n=2，placement path 为 (3,3)->(3,4)->(4,4)，终点不可达，最远可达停点是 (3,4)。
    - 原题输出：起点刷蓝，只在 (3,4) 留一个黑色 2x2，红块保留。
    - D3 单变体：在 (3,3) 和 (3,4) 都刷黑，形成连续黑带，红块保留。
    - 本融合变体：先把路径前缀刷黑，再把最终停点 (3,4) 刷成橙色，因此看到黑色轨迹 + 橙色失败头部 + 远处红色目标。
    - 差异的关键不在 reachability 计算，而在“重建 BFS 路径后如何渲染”。
  complexity_calibration:
    - 只额外引入 1 个状态色橙色，颜色总数仍低。
    - 轨迹与失败标记共用同一条几何推理链：都依赖相同的可行 placement BFS。
    - generator 保持 reachable / stuck 两类样本，避免训练集中只学到轨迹、不学到橙色失败态。
  implicit_assumption_audit:
    - 原 main() 的 solid-square 检测仍成立，因为输入没有轨迹或橙色，输入对象结构不变。
    - 原 main() 靠颜色升序把两个方块分成 mover/dest，这是数值隐式假设；融合版优先显式识别黑色 mover 与红色 dest，只在异常输入回退。
    - 原渲染代码只写 source 和 target 两个矩形，隐含“旧位置会恢复河流色”；D3 打破此假设，所以必须保存 parent 并回溯完整路径。
    - D6 引入橙色后，river 识别不能把橙色当成输入河道颜色；融合版显式把橙色排除在 river 候选外。
    - 新假设：最终目标格或最远可达格一定从起点可回溯出一条 BFS 路径。solver 通过 parent 数组保证；generator 再用 solver 自校验。
  placement_legality:
    - 输入放置逻辑沿用原题 output-first：先造蓝色通道，再放 mover / dest。
    - chokepoint 只在 stuck 且 n>=2 时加入，宽度严格为 1..n-1，因此能保证 destination 不可达。
    - generator 在放置后重算 feasible placement，并要求 solver 输出与内部 oracle 一致，防止重叠、越界或非法通道宽度漏网。
  visual_observability:
    - 人类可从少量样例直接看到两种信号：
      - 黑色不再“瞬移”，而是沿通道连续留下轨迹。
      - 只有卡住时，轨迹最前端会变成橙色，且红色终点仍存在。
    - 规则依赖几何可见条件“能否穿过宽度 >= n 的通道”，不依赖颜色编号排序。
  visual_cleanliness:
    - 轨迹只用黑色，失败态只用橙色，目标仍是红色，图层关系稳定。
    - 橙色只覆盖最终停点，不会在整条路径上混色，因此黑轨迹与失败头部容易区分。
  inference_of_change:
    - 颜色变化不是任意映射，而是结构化条件触发：到不了终点时，只有最后那个方块改成橙色。
    - generator 额外过滤退化样本，要求输出确实不同于原题输出，并且轨迹联合面积超过起始面积。
"""


_BG = 3
_RIVER = 1
_MOVER = 0
_DEST = 2
_STUCK = 7


def _bfs_with_parent(feasible: np.ndarray, sr: int, sc: int):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent_r = -np.ones((h2, w2), dtype=int)
    parent_c = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent_r[rr, cc] = r
                parent_c[rr, cc] = c
                q.append((rr, cc))
    return dist, parent_r, parent_c


def _reconstruct_path(
    parent_r: np.ndarray, parent_c: np.ndarray, sr: int, sc: int, tr: int, tc: int
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    r, c = tr, tc
    while True:
        path.append((int(r), int(c)))
        if r == sr and c == sc:
            break
        pr = int(parent_r[r, c])
        pc = int(parent_c[r, c])
        assert pr >= 0 and pc >= 0
        r, c = pr, pc
    path.reverse()
    return path


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


def _pick_river_color(grid: np.ndarray, excluded: set[int]) -> int:
    counts = Counter(map(int, grid.flatten()))
    for col, _ct in counts.most_common():
        if int(col) not in excluded:
            return int(col)
    raise AssertionError("river color not found")


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int, int]:
    best_d = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best_d:
                best_d = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc, best_d


def _solve_core(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_mover_and_dest(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    river = _pick_river_color(g, excluded={int(bg), int(mover["col"]), int(dest["col"]), _STUCK})
    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))

    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist, parent_r, parent_c = _bfs_with_parent(feasible, sr, sc)

    reached = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if reached:
        tr, tc = int(dr), int(dc)
        final_color = int(mover["col"])
    else:
        tr, tc, best_d = _farthest_reachable(dist, sr, sc)
        assert best_d >= 0
        final_color = _STUCK

    path = _reconstruct_path(parent_r, parent_c, sr, sc, tr, tc)

    out = g.copy()
    if reached:
        for r, c in path:
            out[r : r + n, c : c + n] = int(mover["col"])
    else:
        for r, c in path[:-1]:
            out[r : r + n, c : c + n] = int(mover["col"])
        out[tr : tr + n, tc : tc + n] = _STUCK

    info = {
        "bg": bg,
        "river": int(river),
        "n": n,
        "sr": sr,
        "sc": sc,
        "dr": dr,
        "dc": dc,
        "tr": tr,
        "tc": tc,
        "reached": reached,
        "path": path,
        "path_len": len(path),
        "final_color": final_color,
        "start_area": n * n,
        "black_area": int(np.count_nonzero(out == int(mover["col"]))),
        "orange_area": int(np.count_nonzero(out == _STUCK)),
        "travel_union_area": int(np.count_nonzero((out == int(mover["col"])) | (out == _STUCK))),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the solid square through the river exactly as in the base task.

    Fusion rule:
    - Every visited placement remains black, leaving a continuous trajectory.
    - If the destination is unreachable, the final stopped square is recolored orange.
    """
    out, _info = _solve_core(grid)
    return out


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


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]):
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
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if (
                        0 <= ar < rows
                        and 0 <= ac < cols
                        and (ar, ac) not in test_visited
                        and (ar, ac) in seen
                    ):
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
) -> list[tuple[int, int]]:
    pixel_pos: list[tuple[int, int]] = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    n: int,
    direction: str,
) -> None:
    H, W = grid.shape
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
            if c < 0 or c >= W:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
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
            if r < 0 or r >= H:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < W:
                    grid[r, cc] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _gen_attempt():
    reached = random.random() < 0.45
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, n, coarse_rows)
    offsets_c = _compute_offsets(G, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n:
            return None
    for i in range(len(offsets_c) - 1):
        if offsets_c[i + 1] - offsets_c[i] < n:
            return None

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    def _extend_to_edge(pr: int, pc: int):
        candidates = []
        for direction in ("top", "bottom", "left", "right"):
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(
                    grid[r, c] in (_BG, _RIVER)
                    for r in range(pr + n, H)
                    for c in range(pc, pc + n)
                    if c < W
                ):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            elif direction == "right":
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(
                    grid[r, c] in (_BG, _RIVER)
                    for r in range(pr, pr + n)
                    for c in range(pc + n, W)
                    if r < H
                ):
                    candidates.append((pr, W - n, dist))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2])
        return candidates[0][:2]

    result = _extend_to_edge(dest_r, dest_c)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _RIVER
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = _RIVER
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached and n >= 2:
        num_choke = random.randint(1, min(3, len(walk) - 2))
        min_seg = max(1, len(walk) // 4)
        available = list(range(min_seg, len(walk) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = (
        0 <= dest_r < h2
        and 0 <= dest_c < w2
        and bool(feasible[dest_r, dest_c])
        and int(dist[dest_r, dest_c]) >= 0
    )
    if reached != dest_ok:
        return None

    output, info = _solve_core(grid)

    base_output = grid.copy()
    base_output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    base_output[info["tr"] : info["tr"] + n, info["tc"] : info["tc"] + n] = _MOVER

    if info["path_len"] < 2:
        return None
    if info["travel_union_area"] <= info["start_area"]:
        return None
    if np.array_equal(output, base_output):
        return None

    if info["reached"]:
        if info["black_area"] <= info["start_area"]:
            return None
        if info["orange_area"] != 0:
            return None
    else:
        if info["final_color"] != _STUCK:
            return None
        if info["orange_area"] != n * n:
            return None
        if info["black_area"] <= 0:
            return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
