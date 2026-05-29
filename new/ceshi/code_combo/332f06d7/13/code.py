# Combo variant of 332f06d7 (index 13)
# combo_spec: D6/0+D8/2
# Name: 颜色交换+颜色反转识别
#
# Design analysis
# ===============
# 1) 具体化与行为追踪
#    - 画面仍是原题骨架: 绿色背景、蓝色河道、两个同尺寸实心方块。
#    - 但角色识别改了: 颜色值更大的方块才是 mover，因此固定配色下是红块在动、黑块是目标。
#    - 例子: 若 n=2，红块在河道起点，黑块贴边在河道终点。
#      原题: 黑块会被当作 mover，若可达则终点变黑、起点变蓝。
#      本融合: 红块被当作 mover，若可达则终点变红、起点变黑，形成真正的颜色交换。
#      若不可达，则红块停在最远可达处，起点回蓝，黑块继续留在终点。
#    - 因此两处差异是耦合的:
#      先要从样例看出“红块在移动”，再要看出“只有到达黑块时，起点才变成黑块”。
#
# 2) 复杂度校准
#    - 不改河道几何、不改卡点机制、不引入新对象，只改角色识别和到达后的变换。
#    - 颜色种类仍只有 4 种，交换只发生在两个结构角色之间，视觉负担低。
#    - 生成器把可达样例比例调高到约 70%，避免训练样例大多只显示“红块移动”而看不出交换事件。
#
# 3) 隐式假设审查
#    - 方块检测仍依赖“非背景单色实心正方形”；生成器继续只放这两个关键方块。
#    - 选角色的旧假设 `sort ascending` 不再成立，必须改为 `sort descending`。
#    - `out[src]=river` 这个旧结论只在不可达时成立；可达时必须写回 `dest` 颜色完成交换。
#    - 其余可行窗口、BFS、最远点选择只依赖几何通行性，与融合改动兼容。
#
# 4) 放置合法性
#    - 目标方块仍延伸到边界，移动方块仍靠边但不接触边界。
#    - 两方块都只落在已挖出的 n×n 河道可放置位置，不重叠、不越界。
#    - stuck 样例仍只在中后段河道打出宽度 < n 的卡点，并复验连通与可达性。
#
# 5) 视觉可观察性与可推断性
#    - “颜色值大小规律”本身对人类不可见，所以生成器必须把更大颜色值稳定映射成可见的起点移动块。
#    - 这里固定为红块起步、黑块贴边等待；输出中若可达会出现“起点黑、终点红”的成对交换。
#    - 这样 2-4 组样例即可同时支撑:
#      a. 红块才是 mover；b. 只有到达时才发生颜色交换；c. 卡住时不会交换。
#
# 6) 可视化整洁度
#    - 所有对象仍是整块实心方块，没有额外纹理、标记或重叠层。
#    - 交换只发生在原起点和原终点两个清晰位置，画面不会变乱。
#
# 7) 新 solver 假设与 generator 对应保证
#    - 假设 A: 至少存在两个同尺寸实心方块。生成器始终满足。
#    - 假设 B: 更大颜色值方块确实位于河道上游起点。生成器用 `_MOVER=2` 保证。
#    - 假设 C: 更小颜色值方块确实位于边界终点。生成器用 `_DEST=0` 保证。
#    - 假设 D: 除背景和两方块颜色外，河道色是最主要剩余颜色。生成器仍只使用固定四色。

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


def _find_squares(grid: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        bbox = comp["bbox"]
        if bbox_h(bbox) != bbox_w(bbox):
            continue
        patch = grid[int(bbox[0]) : int(bbox[2]) + 1, int(bbox[1]) : int(bbox[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(bbox[0]), "c0": int(bbox[1]), "n": int(bbox_h(bbox))})
    return squares


def _select_roles(squares: list[dict]) -> tuple[dict, dict]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for square in squares:
        by_n[int(square["n"])].append(square)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    candidates = by_n[int(nside)]
    assert not (len(candidates) < 2)

    ordered = sorted(candidates, key=lambda s: int(s["col"]), reverse=True)
    return ordered[0], ordered[1]


def _find_river(grid: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, grid.flatten()))
    for col, _count in counts.most_common():
        if int(col) in (int(bg), int(mover_col), int(dest_col)):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _feasible_windows(traversable: np.ndarray, n: int) -> np.ndarray:
    H, W = traversable.shape
    h2, w2 = H - n + 1, W - n + 1
    assert not (h2 <= 0 or w2 <= 0)
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(traversable[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, bool, tuple[int, int], tuple[int, int], int, dict, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _find_squares(g, bg)
    assert not (len(squares) < 2)
    mover, dest = _select_roles(squares)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _find_river(g, bg, int(mover["col"]), int(dest["col"]))

    traversable = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _feasible_windows(traversable, n)
    h2, w2 = feasible.shape
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))
    assert not (not bool(feasible[sr, sc]))

    dist = _bfs_placement(feasible, sr, sc)
    reached_dest = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if reached_dest:
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
    if reached_dest:
        out[sr : sr + n, sc : sc + n] = int(dest["col"])
        out[tr : tr + n, tc : tc + n] = int(mover["col"])
    else:
        out[sr : sr + n, sc : sc + n] = int(river)
        out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out, reached_dest, (sr, sc), (tr, tc), river, mover, dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the higher-color square through the river. If it reaches the lower-color
    target square, the two endpoint colors swap; otherwise the mover just stops at
    the farthest reachable placement and the target stays unchanged.
    """
    out, _reached_dest, _src, _dst, _river, _mover, _dest = _solve_with_metadata(grid)
    return out


_BG, _RIVER, _MOVER, _DEST = 3, 1, 2, 0


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


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
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
        hold_len = max(1, hold_len)
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
        hold_len = max(1, hold_len)
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
    _labels, num = scipy_label(mask.astype(int))
    return bool(num == 1)


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.7
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
    edge_list = list(edge_cells)
    random.shuffle(edge_list)
    start_coarse = edge_list[0]
    far = sorted(edge_list, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
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

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
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

    traversable = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _feasible_windows(traversable, n)
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = 0 <= dest_r < feasible.shape[0] and 0 <= dest_c < feasible.shape[1] and feasible[dest_r, dest_c] and dist[dest_r, dest_c] >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    expected = grid.copy()
    if dest_ok:
        expected[mover_r : mover_r + n, mover_c : mover_c + n] = _DEST
        expected[dest_r : dest_r + n, dest_c : dest_c + n] = _MOVER
    else:
        best_d = -1
        tr, tc = mover_r, mover_c
        for r in range(feasible.shape[0]):
            for c in range(feasible.shape[1]):
                if dist[r, c] > best_d:
                    best_d = dist[r, c]
                    tr, tc = r, c
        expected[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
        expected[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
