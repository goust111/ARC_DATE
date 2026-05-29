# Variant of 332f06d7
# Dimension: D4 — 源对象形态 (src)
# Change: 黑色实心n*n正方形，尺寸n=1~4，位于通道内部但不接触网格边缘... → 多色方块移动体

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w


# Design analysis for the D4 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 输入仍是绿色背景 + 蓝色河道 + 终点方块，但起点不再是单色黑块，而是一个完整填满 n*n
#   外接框的多色图案块。图案随位移整体平移，不会旋转、翻转、重着色。
# - 为了让“多色”在像素层面可见，n 需要收紧到 2..4；n=1 无法承载内部图案。
# - 具体样式控制在少量模板：
#   - n=2: 单色底色 + 一个异色角像素。
#   - n=3,4: 外框一种颜色，内部实心另一种颜色。
# - 原逻辑关键环节：输出时把终点 n*n 区域涂成 mover 的单色。
#   变体逻辑：输出时必须把 mover 的整块图案 patch 原样复制到目标位置。
#   例：3x3 的黄框灰心块到达红色 3x3 方块时，输出不能只填黄色，必须保留 8 个黄边框 +
#   1 个灰中心。
#
# b) 复杂度校准
# - 不引入额外对象类别；仍然只有一个 mover 和一个 destination。
# - 图案模板数量很少，避免“每题一个编码方式”；但 mover 至少 2 色，确保视觉上不退化成原题。
# - destination 仍保持单色贴边方块，便于人类区分“被覆盖的目标”和“需要整体追踪的移动体”。
#
# c) 隐式假设审查（原 main 中与 D4 相关的失效点）
# - 原代码按 monochromatic=True 找单色组件；多色 mover 会被拆碎，不能再用。
# - 原代码用颜色升序区分 mover/dest；这里禁止依赖颜色 id 排序，改用可见结构：
#   多色方形组件 = mover，单色贴边方形组件 = destination。
# - 原代码输出时对目标窗整体赋单色 mover["col"]；这会毁掉图案，必须改成复制 mover patch。
# - 原代码把 trav 定义为 {river ∪ mover_color ∪ dest_color}；多色 mover 不能按单个颜色并入，
#   因此改成“river 掩码 + mover bbox + dest bbox”。
#
# d) 放置合法性
# - generator 保证 mover 与 destination 都是完整 n*n 实心方形 bbox，不越界、不重叠。
# - mover 位于河道内部且不接触边界；destination 仍贴边，且与河道连通。
# - 所有 mover 图案颜色都避开 bg/river/destination，solver 才能稳定把对象层分离出来。
#
# e) 视觉可观察性
# - 人类可直接看到：蓝河上有一个多色方块、一个红色目标方块；输出中多色图案整体搬运。
# - mover/destination 的区分完全靠可见结构（是否多色、是否贴边），不靠颜色编号。
#
# f) 可视化整洁度
# - 河道、目标、移动体三层语义清晰：绿色背景，蓝色通道，红色终点，多色 mover。
# - mover 图案限制为 2 色，避免与河道/背景叠加成噪声。
#
# g) 颜色变化可推断性
# - 不存在颜色映射规则；颜色只用于区分区域。输出唯一变化是“同一块多色 patch 的平移”。
#
# 额外的 solver 假设与 generator 保证
# - solver 假设最常见的非背景色就是河道色；generator 用大面积连通蓝河保证这一点。
# - solver 假设对象层 ({非bg且非river}) 里恰有两个完整方形组件：一个多色、一个单色；
#   generator 只放这两个对象并保持间隔。


def _edge_touching(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _detect_objects(g: np.ndarray, bg: int, river: int) -> tuple[dict, dict]:
    H, W = g.shape
    obj_mask = (g != int(bg)) & (g != int(river))
    labeled, num = scipy_label(obj_mask.astype(int))
    objects: list[dict] = []
    for k in range(1, int(num) + 1):
        coords = np.argwhere(labeled == k)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        n = int(bbox_h(bbox))
        if n != int(bbox_w(bbox)):
            continue
        if int(coords.shape[0]) != n * n:
            continue
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        colors = sorted(set(map(int, patch.flatten())))
        objects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "n": n,
                "patch": patch,
                "colors": colors,
                "edge": _edge_touching(int(r0), int(c0), n, H, W),
            }
        )

    patterned = [obj for obj in objects if len(obj["colors"]) >= 2]
    assert patterned
    mover = max(patterned, key=lambda obj: (obj["n"], -obj["r0"], -obj["c0"]))

    solid = [obj for obj in objects if len(obj["colors"]) == 1 and obj["n"] == mover["n"]]
    edge_solid = [obj for obj in solid if obj["edge"]]
    assert edge_solid or solid
    dest = edge_solid[0] if edge_solid else solid[0]
    return mover, dest


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a patterned square block through a winding river toward a same-sized landing square.

    Input structure:
    - The corner-majority color is the background, and one dominant non-background color forms a river-like corridor.
    - On that corridor sit two same-sized square objects: an interior multicolor mover and a boundary-touching solid target.

    Rule:
    - Treat the mover as one rigid n*n patch and slide it through the corridor without changing its internal colors.
    - A move is allowed only when the full n*n footprint fits inside traversable river space.
    - If the target square is reachable, place the full mover pattern there; otherwise stop at the farthest reachable placement.
    - Refill the mover's original location with the river color.

    Key insight: The motion constraint is about the whole square footprint, so narrow bottlenecks can block the patch even when
    individual river pixels continue beyond them.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) != int(bg):
            river = int(col)
            break
    assert river is not None

    mover, dest = _detect_objects(g, bg, river)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    mover_patch = np.array(mover["patch"], dtype=int)

    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    trav = (g == int(river))
    trav[int(sr) : int(sr + n), int(sc) : int(sc + n)] = True
    trav[int(dr) : int(dr + n), int(dc) : int(dc + n)] = True

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    assert bool(feasible[int(sr), int(sc)])

    dist = _bfs_placement(feasible, sr, sc)
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0:
        tr, tc = int(dr), int(dc)
    else:
        best = -1
        tr, tc = int(sr), int(sc)
        for r in range(h2):
            for c in range(w2):
                if int(dist[int(r), int(c)]) > best:
                    best = int(dist[int(r), int(c)])
                    tr, tc = int(r), int(c)

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = mover_patch
    return out


# --- Generator ---

_BG, _RIVER, _DEST = 3, 1, 2
_PATTERN_COLORS = (4, 5, 6, 7, 8, 9)


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
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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


def _render_channel(grid: np.ndarray, path, n: int, offsets_r: list[int], offsets_c: list[int]):
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


def _add_chokepoint(grid: np.ndarray, pixel_pos_i, pixel_pos_j, n: int, direction: str):
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
    mask = grid == _RIVER
    if not bool(mask.any()):
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


def _make_mover_patch(n: int) -> np.ndarray:
    outer, inner = random.sample(list(_PATTERN_COLORS), 2)
    patch = np.full((n, n), int(outer), dtype=int)
    if n == 2:
        rr, cc = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        patch[rr, cc] = int(inner)
    else:
        patch[1 : n - 1, 1 : n - 1] = int(inner)
    return patch


def _gen_attempt():
    reached = random.random() < 0.5
    n = random.choice([2, 3, 4])

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
        for direction in ["top", "bottom", "left", "right"]:
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
                    grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W
                ):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            else:
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(
                    grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H
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
    if not _edge_touching(dest_r, dest_c, n, H, W):
        return None

    touching = _edge_touching(mover_r, mover_c, n, H, W)
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
        elif mover_c + n == W:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if _edge_touching(mover_r, mover_c, n, H, W):
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

    mover_patch = _make_mover_patch(n)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_patch
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = grid == _RIVER
    trav[mover_r : mover_r + n, mover_c : mover_c + n] = True
    trav[dest_r : dest_r + n, dest_c : dest_c + n] = True
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
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

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = mover_patch

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
