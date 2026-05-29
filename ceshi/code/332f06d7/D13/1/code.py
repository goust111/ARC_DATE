# Variant of 332f06d7
# Dimension: D13 — 方块尺寸与网格尺寸的相关性 (meta)
# Change: n=1~4，网格尺寸与n正相关（n=1→10-16, n=2→12-16, n=3→14-20, n=4→20），网格始终正方形
#   → 方块尺寸由通道编码；黑/红对象本身不再显示 n，而是由其所在端点水垫可容纳的最大正方形边长给出
#
# Design analysis:
# design:
#   concrete_behavior:
#   - 原题里黑/红对象本身就是 n×n 实心方块，因此 solver 直接从对象边长取 n。
#   - 变体里黑色 mover 和红色 destination 都缩成 1×1 标记，但它们各自都放在一个蓝色端点水垫的左上角。
#   - 真正控制运动的 n 不再来自标记尺寸，而来自端点位置处“从该标记向右下扩展，最多能取到多大的全可通行正方形”。
#   - 例子：若黑点位于一个 3×3 蓝色起始水垫左上角，那么虽然黑点只占 1 格，solver 仍把它当作边长 3 的隐形方块来做可达性 BFS。
#   - 这会影响“窄口是否能通过”：原题在同一输入上若把黑点当 1×1 会穿过窄口；变体会因解码出 n=3 而停在窄口前。
#   complexity_calibration:
#   - 只保留一个编码机制：端点水垫的可容纳正方形大小。避免再叠加颜色映射或多种编码位点，否则 2-4 个样例不足以归纳。
#   - 标记始终是 1×1，颜色仍固定为黑/红，河道仍是单一蓝色连通通道，因此视觉差异集中在“物体尺寸来自河道，而不是来自物体自身”。
#   - n 仍限制在 1..4，避免自由度过大；同时提高 n>=2 的采样权重，让变体和原题的行为差异更常出现。
#   implicit_assumption_audit:
#   - 原 `find_components(... solid square ...)` 依赖“对象本身就是方块”，在变体中失效，因为 mover/dest 只剩单像素标记。
#   - 原 `by_n` 与“最常见边长”依赖“至少两块同尺寸方块”，在变体中失效，因为不再从对象面积读 n。
#   - 原 `cand.sort(key=color)` 依赖颜色 id 排序，这是不可见线索；变体中直接使用固定角色颜色黑/红，不再做隐藏排序。
#   - 原 `n = mover['n']` 依赖“mover 边长就是通行尺寸”；变体中改为从河道端点水垫解码 `n = min(start_pad, dest_pad)`。
#   - 原 BFS / farthest-reachable 逻辑与河道几何本身无关，仍然成立，因此保持不变。
#   placement_legality:
#   - generator 仍先按真实 n 画出 n 宽河道，再把单像素标记嵌入端点水垫左上角，因此标记不会越界，也不会离开河道。
#   - chokepoint 仍只在下游段制造，保证起点/终点的端点水垫仍能稳定编码同一个 n。
#   - generator 会显式验证两端点的 anchored-square size 都等于预设 n，防止某次路径形状意外形成更大的可容纳正方形。
#   observability:
#   - 人类能直接看到：黑/红都只是小标记，但它们所在蓝色端点平台大小在不同样例中变化，并且输出运动结果受这个平台大小控制。
#   - 若用原题规则把黑点当 1×1，会错误地让它穿过许多本应挡住大方块的窄口；因此 input/output 对能支持解题者察觉新的控制变量。
#   visual_cleanliness:
#   - 标记只占端点水垫左上角一格，其余仍是整洁的单色河道；没有多层叠印，也没有与 chokepoint 混杂的额外符号。
#   color_inference:
#   - 颜色角色保持原题习惯：蓝=河道、黑=移动者、红=目标、绿=背景。没有新增颜色，也没有依赖颜色 id 的隐藏排序。
#   main_new_assumptions_guaranteed_by_generator:
#   - 恰有一个黑点和一个红点。
#   - 黑点与红点都位于各自端点水垫的左上角。
#   - 从黑点/红点出发向右下取最大可通行正方形，大小一致且等于真实 n。
#   - 河道唯一，背景占四角，`main()` 用这些假设即可稳定恢复规则。

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _anchored_square_size(mask: np.ndarray, r: int, c: int) -> int:
    H, W = mask.shape
    best = 0
    limit = min(H - r, W - c)
    for k in range(1, limit + 1):
        if bool(mask[r : r + k, c : c + k].all()):
            best = k
        else:
            break
    return int(best)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a 1-cell marker through a winding river using a square size decoded from the river itself.

    Input structure:
    - The background is the corner-majority color, and one other color forms a single blue-like river.
    - A black 1x1 mover marker and a red 1x1 destination marker sit on the river.
    - Each marker occupies the top-left cell of an endpoint river pad; that pad's square capacity encodes the hidden side length n.

    Rule:
    - First decode n by measuring the largest all-traversable square anchored at the mover and destination markers.
    - Treat the mover as an invisible n×n square whose top-left corner is the marker location.
    - Slide that invisible square through the river via 4-neighbor top-left moves; if the destination anchor is reachable, move the black marker there, otherwise stop at the farthest reachable anchor.
    - The mover's original cell becomes river, and an unreached red marker remains in place.

    Key insight:
    - The marker size is a decoy: the real motion size comes from the endpoint river pads, so chokepoints narrower than the decoded n still block the move.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    mover_pos = np.argwhere(g == _MOVER)
    dest_pos = np.argwhere(g == _DEST)
    assert len(mover_pos) == 1 and len(dest_pos) == 1
    sr, sc = map(int, mover_pos[0])
    dr, dc = map(int, dest_pos[0])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) not in (int(bg), _MOVER, _DEST):
            river = int(col)
            break
    assert river is not None

    trav = (g == river) | (g == _MOVER) | (g == _DEST)
    n_start = _anchored_square_size(trav, sr, sc)
    n_dest = _anchored_square_size(trav, dr, dc)
    n = min(int(n_start), int(n_dest))
    assert n >= 1

    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2 and 0 <= dr < h2 and 0 <= dc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)

    if feasible[dr, dc] and dist[dr, dc] >= 0:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if dist[r, c] > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr, sc] = river
    out[tr, tc] = _MOVER
    return out


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
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if not (0 <= c < W):
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
            if not (0 <= r < H):
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
    mask = grid == _RIVER
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
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


def _endpoint_code_ok(grid: np.ndarray, mover_r: int, mover_c: int, dest_r: int, dest_c: int, n: int) -> bool:
    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    return _anchored_square_size(trav, mover_r, mover_c) == n and _anchored_square_size(trav, dest_r, dest_c) == n


def _gen_attempt() -> np.ndarray | None:
    n = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 3])[0]
    reached = True if n == 1 else (random.random() < 0.5)

    G = random.randint(17, 20)
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
    if len({cr for cr, _ in walk}) < coarse_rows or len({cc for _, cc in walk}) < coarse_cols:
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
        for direction in ("top", "bottom", "left", "right"):
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
        return int(candidates[0][0]), int(candidates[0][1])

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

    if mover_r == 0:
        grid[0, mover_c : mover_c + n] = _RIVER
        mover_r = 1
    elif mover_r + n == H:
        grid[H - 1, mover_c : mover_c + n] = _RIVER
        mover_r = H - n - 1
    elif mover_c == 0:
        grid[mover_r : mover_r + n, 0] = _RIVER
        mover_c = 1
    elif mover_c + n == W:
        grid[mover_r : mover_r + n, W - 1] = _RIVER
        mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

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

    if _anchored_square_size(grid == _RIVER, mover_r, mover_c) != n:
        return None
    if _anchored_square_size(grid == _RIVER, dest_r, dest_c) != n:
        return None

    grid[mover_r, mover_c] = _MOVER
    grid[dest_r, dest_c] = _DEST
    if not _endpoint_code_ok(grid, mover_r, mover_c, dest_r, dest_c, n):
        return None

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = feasible[dest_r, dest_c] and dist[dest_r, dest_c] >= 0
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
                if dist[r, c] > best_d:
                    best_d = int(dist[r, c])
                    tr, tc = r, c

    output = grid.copy()
    output[mover_r, mover_c] = _RIVER
    output[tr, tc] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
