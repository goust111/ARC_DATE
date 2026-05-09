# Variant of 332f06d7
# Dimension: D8 — 移动体与目标的配对规则 (meta)
# Change: 按颜色值升序排序，颜色最小的为移动体（黑色0），次小的为目标（红色2），通过相同尺寸n配对... → 按尺寸差异配对
#
# Design analysis:
# spec:
#   variant_core:
#     - 输入里仍然只有一条蓝色水道、一个黑色实心方块、一个红色实心方块。
#     - 黑方块始终比红方块大 1 个像素层：尺寸对固定为 2->1、3->2 或 4->3。
#     - 红方块不再表示“与 mover 同尺寸的终点占位”，而是表示 mover 最终落位时左上角的可见裁剪区。
#   concrete_behavior:
#     - mover 的真实滑行尺寸由黑方块大小决定，BFS 仍在 mover 尺寸的放置网格上进行。
#     - 若 mover 可以把自己的左上角滑到红方块左上角，那么输出只把该位置左上角的
#       红方块同尺寸区域涂成黑色；mover 多出的那一圈不显示，保持为蓝色水道。
#     - 若通道中存在宽度 < mover 尺寸的卡口，则 mover 停在最远可达放置位，并且同样只显示
#       一个“红方块尺寸”的黑色裁剪块；原红色目标保持不变。
#   flow_trace:
#     original:
#       - 检出两个同尺寸方块，按颜色 id 升序挑 mover/dest。
#       - BFS 使用该共同尺寸 n；到达 dest 时输出一个 n×n 黑块。
#     variant:
#       - 检出两个不同尺寸方块，较大的就是 mover，较小的就是 target。
#       - BFS 使用 mover 尺寸 m；dest 的左上角只是 mover 终点锚点。
#       - 输出时不再画 m×m，而是只画 target 尺寸 t×t 的黑块。
#     difference:
#       - “谁在移动”由尺寸决定而不是颜色排序。
#       - “落点是否可达”仍由大方块的完整占位决定，但“最终可见形状”由小方块尺寸决定。
#   hidden_assumptions_replaced:
#     - 原代码默认存在至少两个同尺寸方块；这里改成必须存在唯一最大方块和唯一最小方块。
#     - 原代码把 dest 的左上角直接当成 n×n 落位；这里仍用该锚点，但输出裁剪成 t×t。
#     - 原代码 generator 用同一个 n 同时控制 mover、dest、河道和卡口；这里拆成 mover 尺寸 m
#       与 target 尺寸 t=m-1，其中河道/卡口只服从 m。
#   legality_guarantees:
#     - generator 只把目标放在上边界或左边界，且不允许角落，因此“目标左上角 = mover 最终左上角”
#       这一锚定方式始终合法，不会让 m×m 落位越界。
#     - 目标红块始终嵌在一个已经被渲染好的 m×m 河道终点的左上角，solver 检查可达性时会看到
#       完整的 m×m traversable 区域。
#     - stuck 样例中的卡口只收窄蓝河，不破坏整条河的像素连通性；因此视觉上仍是一条连续水道，
#       但大 mover 的放置 BFS 会被阻断。
#   observability:
#     - 输入里黑块明显大于红块，输出里的黑块又变成红块大小，这一变化在 2-3 个样例里肉眼可见。
#     - 没有用颜色 id 排序；尺寸差异就是可见的配对线索。
#   cleanliness:
#     - 仍只保留四种颜色和两个实心方块，没有额外标记层。
#     - 目标只占终点大平台的左上角，不和别的对象叠成杂乱形状。

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


def _detect_squares(g: np.ndarray, bg: int) -> list[dict]:
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
    return squares


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
    Slide a larger solid square through a river so its anchor reaches a smaller target square.

    Input structure:
    - A corner-dominant background surrounds one connected blue river.
    - One solid black square is the mover; one solid red square is the target anchor.
    - The black square is larger than the red square, and the red square sits in the upper-left corner
      of a river endpoint that is wide enough for the full black square.

    Rule:
    - The black square moves along the river using its own full size as the collision footprint.
    - A landing position is valid only if the entire black-sized window stays inside river/mover/target cells.
    - If the mover can place its upper-left corner on the red square's upper-left corner, it reaches the target.
    - Whether it reaches the target or stops at the farthest reachable placement, the output shows only a
      black patch matching the red square's size, and the original mover area turns back into river.

    Key insight: Reachability is decided by the larger mover footprint, but the visible result is cropped to
    the smaller target size, so narrow river sections can block motion even though the final black mark is small.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _detect_squares(g, bg)
    assert len(squares) >= 2

    sizes = sorted({int(s["n"]) for s in squares})
    assert len(sizes) >= 2
    mover_cands = [s for s in squares if int(s["n"]) == sizes[-1]]
    dest_cands = [s for s in squares if int(s["n"]) == sizes[0]]
    assert len(mover_cands) == 1 and len(dest_cands) == 1
    mover = mover_cands[0]
    dest = dest_cands[0]

    mover_n = int(mover["n"])
    dest_n = int(dest["n"])
    assert mover_n > dest_n

    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col == bg or col in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = H - mover_n + 1
    w2 = W - mover_n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + mover_n, c : c + mover_n].all())
    assert feasible[sr, sc]

    dist = _bfs_placement(feasible, sr, sc)

    if 0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0:
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
    out[sr : sr + mover_n, sc : sc + mover_n] = river
    out[tr : tr + dest_n, tc : tc + dest_n] = int(mover["col"])
    return out


# --- Generator ---

_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


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
    grid: np.ndarray, path: list[tuple[int, int]], mover_n: int, offsets_r: list[int], offsets_c: list[int]
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + mover_n, pc : pc + mover_n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + mover_n
            grid[pr1 : pr1 + mover_n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + mover_n
            grid[r_lo:r_hi, pc1 : pc1 + mover_n] = _RIVER
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], mover_n: int, direction: str
) -> None:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, mover_n - 1)
    remove = mover_n - narrow_w
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + mover_n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + mover_n
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
                rr = pr1 + k if side == 0 else pr1 + mover_n - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + mover_n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + mover_n
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
                cc = pc1 + k if side == 0 else pc1 + mover_n - 1 - k
                if 0 <= cc < W:
                    grid[r, cc] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, mover_n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - mover_n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_top_or_left(grid: np.ndarray, pr: int, pc: int, mover_n: int):
    H, W = grid.shape
    candidates = []
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + mover_n) if c < W):
        candidates.append((0, pc, pr, "top"))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + mover_n) for c in range(pc) if r < H):
        candidates.append((pr, 0, pc, "left"))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x[2])
    for nr, nc, _dist, edge in candidates:
        if not (nr == 0 and nc == 0):
            return nr, nc, edge
    return None


def _gen_attempt():
    reached = random.random() < 0.5
    mover_n = random.choice([2, 3, 4])
    dest_n = mover_n - 1

    min_size = max(10, 5 * mover_n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = mover_n + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, mover_n, coarse_rows)
    offsets_c = _compute_offsets(G, mover_n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < mover_n:
            return None
    for i in range(len(offsets_c) - 1):
        if offsets_c[i + 1] - offsets_c[i] < mover_n:
            return None

    top_left_edges = []
    for c in range(coarse_cols):
        top_left_edges.append((0, c))
    for r in range(1, coarse_rows):
        top_left_edges.append((r, 0))
    random.shuffle(top_left_edges)
    end_coarse = top_left_edges[0]

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells = [cell for cell in edge_cells if cell != end_coarse]
    edge_cells.sort(key=lambda e: -(abs(e[0] - end_coarse[0]) + abs(e[1] - end_coarse[1])))
    if not edge_cells:
        return None
    start_coarse = edge_cells[0]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, mover_n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_anchor_r, dest_anchor_c = pixel_pos[-1]

    result = _extend_to_top_or_left(grid, dest_anchor_r, dest_anchor_c, mover_n)
    if result is None:
        return None
    new_dest_anchor_r, new_dest_anchor_c, edge = result
    if edge == "top" and new_dest_anchor_c == 0:
        return None
    if edge == "left" and new_dest_anchor_r == 0:
        return None
    if new_dest_anchor_r != dest_anchor_r:
        r_lo, r_hi = min(new_dest_anchor_r, dest_anchor_r), max(new_dest_anchor_r, dest_anchor_r) + mover_n
        grid[r_lo:r_hi, dest_anchor_c : dest_anchor_c + mover_n] = _RIVER
    if new_dest_anchor_c != dest_anchor_c:
        c_lo, c_hi = min(new_dest_anchor_c, dest_anchor_c), max(new_dest_anchor_c, dest_anchor_c) + mover_n
        grid[dest_anchor_r : dest_anchor_r + mover_n, c_lo:c_hi] = _RIVER
    dest_anchor_r, dest_anchor_c = new_dest_anchor_r, new_dest_anchor_c

    if not ((dest_anchor_r == 0) ^ (dest_anchor_c == 0)):
        return None

    touching = (
        mover_r == 0
        or mover_r + mover_n == H
        or mover_c == 0
        or mover_c + mover_n == W
    )
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + mover_n] = _RIVER
            mover_r = 1
        elif mover_r + mover_n == H:
            grid[H - 1, mover_c : mover_c + mover_n] = _RIVER
            mover_r = H - mover_n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + mover_n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r : mover_r + mover_n, W - 1] = _RIVER
            mover_c = W - mover_n - 1
    if mover_r == 0 or mover_r + mover_n == H or mover_c == 0 or mover_c + mover_n == W:
        return None
    if min(mover_r, H - (mover_r + mover_n), mover_c, W - (mover_c + mover_n)) > 2:
        return None

    grid[mover_r : mover_r + mover_n, mover_c : mover_c + mover_n] = _RIVER

    if not reached:
        num_choke = random.randint(1, min(3, len(walk) - 2))
        min_seg = max(1, len(walk) // 4)
        available = list(range(min_seg, len(walk) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), mover_n, direction)
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + mover_n, mover_c : mover_c + mover_n] = _MOVER
    grid[dest_anchor_r : dest_anchor_r + dest_n, dest_anchor_c : dest_anchor_c + dest_n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    h2, w2 = H - mover_n + 1, W - mover_n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + mover_n, c : c + mover_n].all())
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = 0 <= dest_anchor_r < h2 and 0 <= dest_anchor_c < w2 and feasible[dest_anchor_r, dest_anchor_c]
    dest_ok = dest_ok and dist[dest_anchor_r, dest_anchor_c] >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    if dest_ok:
        tr, tc = dest_anchor_r, dest_anchor_c
    else:
        best_d = -1
        tr, tc = mover_r, mover_c
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best_d:
                    best_d = int(dist[r, c])
                    tr, tc = r, c
        if best_d <= 0:
            return None

    output = grid.copy()
    output[mover_r : mover_r + mover_n, mover_c : mover_c + mover_n] = _RIVER
    output[tr : tr + dest_n, tc : tc + dest_n] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
