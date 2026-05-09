# Variant of 332f06d7
# Dimension: D3 — 轨迹涂色 (trajectory)
# Change: 无轨迹涂色——方块移走后原位置填充河流色（蓝色），不留下行进痕迹... → 连续轨迹
#
# Design analysis
# design:
#   a_concretization_and_trace:
#     input:
#       - 保持原题外观: 绿色背景, 蓝色河道, 一个黑色实心正方形 mover, 一个红色同尺寸实心正方形 dest.
#       - 输入中不预先出现轨迹色, 这样输出新增的灰色轨迹是清晰可见的唯一新信息.
#     output:
#       - 方块不是“瞬移”到终点; 输出把滑行过程可视化.
#       - 方块访问过的每个 n×n 位置都会留下灰色块, 最终停下的位置仍是黑色, 未到达的红色目标保持红色.
#       - 例: n=2, 路径 top-left 为 [(8,8),(8,7),(8,6)].
#         原题输出: 只把 (8,8) 复原成蓝色, 在 (8,6) 画黑色.
#         本变体输出: (8,8) 与 (8,7) 对应的 2×2 占位都画成灰色, 再在 (8,6) 画黑色.
#         差异在于“经过区域”的并集变成显式对象, 人能直接看见滑行路线.
#   b_complexity_calibration:
#     - 轨迹色固定为灰色 5, 不引入额外颜色映射表, 复杂度集中在“哪些位置被经过”.
#     - generator 只接受路径长度 >= 3 个 placement 的样本, 避免输出几乎和原题一样.
#     - 仍保留 reachable / stuck 两类, 轨迹在两类中都成立, 不会退化成单一情况.
#   c_implicit_assumptions_in_original_main:
#     - 原代码 `cand.sort(key=lambda s: s["col"])` 依赖颜色编号决定 mover/dest, 这是不可见假设.
#       新版改为用可见空间关系识别:
#       dest 触边, mover 不触边但离边界很近(<=2), 这与原 generator 的可见布局一致.
#     - 原代码最终把起点恢复为河流色; 该写法依赖“无轨迹”.
#       新版不能无条件填河流, 而是必须重建 BFS 路径并把 path[:-1] 画成轨迹色.
#     - 原代码只求 reachability 和终点; 新版还需要 parent 指针来恢复实际路径.
#   d_placement_legality:
#     - generator 仍保证两个方块不重叠、不越界, 河道对 n×n 窗口可行性与原题一致.
#     - 轨迹只出现在输出, 因此不会干扰输入中 solid-square 检测或 river 检测.
#     - 输出时先画灰轨迹, 再画黑色终点, 保证最终停驻方块不会被灰色覆盖掉.
#   e_visual_observability:
#     - 解题者从少量样例可见: 蓝河道不变, 黑块沿可通行河道移动, 灰色连续带准确记录经过区域.
#     - 不使用颜色 id 排序; 推理仅依赖“哪个方块贴边、哪个方块靠边但未贴边”的可见特征.
#   f_visual_cleanliness:
#     - 轨迹只覆盖 mover 经过的占位区域, 不额外画箭头、数字或噪声标记.
#     - 灰色与蓝河道/黑方块/红目标分离明显, 输出层次是“灰轨迹 < 黑终点”这一层, 不会乱.
#   g_color_inference:
#     - 唯一新增颜色是灰色 5, 且它总是紧贴 mover 的滑行路线出现.
#     - 颜色变化有稳定结构线索: 灰色只来自 mover 的历史位置, 红色只在未到达目标时保留.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST, _TRAIL = 3, 1, 0, 2, 5


def _edge_gap(square: dict, H: int, W: int) -> int:
    n = int(square["n"])
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _edge_contacts(square: dict, H: int, W: int) -> int:
    n = int(square["n"])
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    contacts = 0
    if r0 == 0:
        contacts += 1
    if c0 == 0:
        contacts += 1
    if r0 + n == H:
        contacts += 1
    if c0 + n == W:
        contacts += 1
    return contacts


def _choose_mover_and_dest(squares: list[dict], H: int, W: int) -> tuple[dict, dict]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    options: list[tuple[int, int, dict, dict]] = []
    for nside, group in by_n.items():
        edge = [sq for sq in group if _edge_contacts(sq, H, W) > 0]
        near = [sq for sq in group if _edge_contacts(sq, H, W) == 0 and _edge_gap(sq, H, W) <= 2]
        if not edge or not near:
            continue
        dest = sorted(edge, key=lambda sq: (-_edge_contacts(sq, H, W), sq["r0"], sq["c0"]))[0]
        mover = sorted(near, key=lambda sq: (_edge_gap(sq, H, W), sq["r0"], sq["c0"]))[0]
        options.append((len(group), int(nside), mover, dest))

    assert options
    _count, _nside, mover, dest = max(options, key=lambda item: (item[0], item[1]))
    return mover, dest


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent = np.full((h2, w2, 2), -1, dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent[rr, cc] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _recover_path(parent: np.ndarray, tr: int, tc: int) -> list[tuple[int, int]]:
    path = []
    cur = (int(tr), int(tc))
    while True:
        path.append(cur)
        pr, pc = parent[cur[0], cur[1]]
        if pr < 0:
            break
        cur = (int(pr), int(pc))
    path.reverse()
    return path


def _paint_output_with_trail(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    mover_color: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in path[:-1]:
        out[r : r + n, c : c + n] = _TRAIL
    tr, tc = path[-1]
    out[tr : tr + n, tc : tc + n] = int(mover_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding river and visualize the full travel trail.

    Input structure:
    - A dominant background color fills the corners, with a blue river corridor running through it.
    - Two same-sized solid squares sit on the river: the moving square stays near an edge, while the target square touches an edge.

    Rule:
    - The moving square can shift one cell at a time only through placements where its entire NxN footprint stays inside river-or-square cells.
    - If the target placement is reachable, the square stops there; otherwise it stops at the farthest reachable placement.
    - Every earlier placement visited along that route is recolored as a continuous grey trail, while the final placement remains the mover color.

    Key insight:
    - Solve the task on the placement grid of valid NxN top-left positions; a BFS both finds the stopping point and recovers the exact route whose occupied windows become the trail.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert len(squares) >= 2
    mover, dest = _choose_mover_and_dest(squares, H, W)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col == bg or col in (int(mover["col"]), int(dest["col"]), _TRAIL):
            continue
        river = int(col)
        break
    assert river is not None

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

    dist, parent = _bfs_placement(feasible, sr, sc)

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

    path = _recover_path(parent, tr, tc)
    assert path
    return _paint_output_with_trail(g, path, n, int(mover["col"]))


def _hamiltonian_walk(rows, cols, start, end):
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
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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
                    for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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


def _render_channel(grid, path, n, offsets_r, offsets_c):
    pixel_pos = []
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


def _add_chokepoint(grid, pixel_pos_i, pixel_pos_j, n, direction):
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


def _river_connected(grid):
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _compute_offsets(G, n, num_coarse):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _gen_attempt():
    reached = random.random() < 0.5
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

    def _extend_to_edge(pr, pc):
        candidates = []
        for direction in ["top", "bottom", "left", "right"]:
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            else:
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
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

    dist, parent = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and bool(feasible[dest_r, dest_c]) and int(dist[dest_r, dest_c]) >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    if dest_ok:
        tr, tc = dest_r, dest_c
    else:
        best_d = -1
        tr, tc = mover_r, mover_c
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best_d:
                    best_d = int(dist[r, c])
                    tr, tc = r, c
    path = _recover_path(parent, tr, tc)
    if len(path) < 3:
        return None

    output = _paint_output_with_trail(grid, path, n, _MOVER)
    if np.count_nonzero(output == _TRAIL) < max(2, n):
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
