# Variant of 332f06d7
# Dimension: D10 — 阻塞场景处理 (meta)
# Change: 不可达时（stuck）方块停在BFS可达的最远位置（距起点最远的可行放置点），目标方块保持不变... → 阻塞时标记通道

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


"""
design_analysis:
  variant_goal:
    summary: |
      保留原题的“同尺寸方块沿蓝色水道滑行”主体，只改 stuck 分支：
      若红方块不可达，则黑方块和红方块都保持原位，输出把真正过窄的水道像素改成橙色。
  a_specific_behavior_and_trace:
    input_shape:
      - 绿色背景包围一条单连通蓝色河道。
      - 黑色 n×n 方块是 mover，位于边缘附近但不接触边界。
      - 红色 n×n 方块是 destination，接触某一条外边界。
      - stuck 样例里，河道中有且仅有一个连续窄口，像素仍然连通，但宽度小于 n。
    output_shape:
      reachable:
        - 黑块滑到红块位置。
        - 黑块起点恢复成蓝色。
      stuck:
        - 黑块不移动，红块不变。
        - 窄口处所有“蓝色但不属于任何合法 n×n 放置窗口”的河道像素改成橙色。
    traced_example: |
      设 n=3。原题 stuck 时，BFS 只能到窄口前，输出把黑块移动到最远可达放置点。
      本变体中，同样先在 n×n 放置图上做 BFS；若红块不可达，则不再挑最远可达点。
      改为统计所有合法 n×n 放置窗口覆盖过哪些像素。窄口里残留的蓝色像素虽然在像素层面连通，
      但由于任何 3×3 窗口都会碰到背景，所以这些像素的覆盖计数为 0。把这一段蓝像素改成橙色即可。
  b_complexity_calibration:
    - 若 stuck 中允许多个窄口，解题者很难知道该标哪一个，因此生成器强制只造一个窄口。
    - 若窄口过于零碎，橙色标记会像噪声，因此窄口做成单个连续矩形细条。
    - reachable 与 stuck 两类样例都会出现，避免解题者误以为永远只做橙色标记。
  c_implicit_assumptions_audit:
    - 原代码用颜色升序区分 mover/dest；这是不可见先验。新 solver 改为用几何角色区分：
      贴边的同尺寸方块是 destination，不贴边的是 mover。
    - 原代码默认 stuck 时一定要选择一个最远 BFS 点；该前提在本变体失效，故不可达分支改为瓶颈检测。
    - 原代码把输出中的黑块原位置无条件刷成蓝色；本变体 stuck 时不能这样做，否则会伪造移动。
    - 原代码生成器允许多个 chokepoint；本变体会引入“该标哪一个”的歧义，所以改成唯一 chokepoint。
  d_placement_legality:
    - 网格仍为 10..20 的方阵。
    - 方块始终不越界、不重叠，并且红块贴边、黑块不贴边。
    - 窄口只开在两段相邻粗网格节点之间的直线连接段内部，不切到起点块或终点块。
    - generator 在返回前验证 main(input) 与预期输出一致。
  e_visual_observability:
    - stuck 输出中橙色是唯一新增颜色，且只出现在明显变细的位置。
    - mover/dest 的角色不靠颜色排序推断，而靠“内侧起点 vs 贴边终点”的稳定几何线索支撑。
  f_visual_cleanliness:
    - 橙色只覆盖一段连续窄口，不叠到黑块、红块或背景上。
    - 通道仍保持单连通，避免出现多处碎片化标记。
  g_color_inference:
    - 颜色集合保持小而稳定：绿背景、蓝河道、黑 mover、红 dest、橙 bottleneck。
    - 只有 stuck 输出会新增橙色，人类可从配套样例中归纳出“橙色=过窄瓶颈”。
  solver_generator_contract:
    - solver 假设恰有一组同尺寸实心方块满足“一个贴边、一个不贴边”，generator 保证这一点。
    - solver 假设 stuck 时存在至少一段蓝色像素不被任何合法 n×n 窗口覆盖，generator 通过唯一窄口保证这一点。
"""


_BG, _RIVER, _MOVER, _DEST, _MARK = 3, 1, 0, 2, 7
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _touches_border(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _edge_distance(r0: int, c0: int, n: int, H: int, W: int) -> int:
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _detect_roles(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        col = int(comp["color"])
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    candidates: list[tuple[int, int, dict, dict]] = []
    for nside, group in by_n.items():
        edge = [sq for sq in group if _touches_border(int(sq["r0"]), int(sq["c0"]), int(sq["n"]), H, W)]
        inner = [sq for sq in group if sq not in edge]
        if len(edge) == 1 and len(inner) >= 1:
            inner.sort(
                key=lambda sq: (
                    _edge_distance(int(sq["r0"]), int(sq["c0"]), int(sq["n"]), H, W),
                    int(sq["r0"]),
                    int(sq["c0"]),
                )
            )
            candidates.append((len(group), int(nside), inner[0], edge[0]))

    assert not (not candidates)
    candidates.sort(key=lambda item: (item[0], item[1]))
    _count, _nside, mover, dest = candidates[-1]
    return mover, dest


def _find_river_color(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), int(mover_col), int(dest_col)):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _feasible_placements(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    assert not (h2 <= 0 or w2 <= 0)
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
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _placement_cover(feasible: np.ndarray, n: int, shape: tuple[int, int]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    h2, w2 = feasible.shape
    for r in range(h2):
        for c in range(w2):
            if bool(feasible[r, c]):
                cover[r : r + n, c : c + n] += 1
    return cover


def _shortest_pixel_path_mask(trav: np.ndarray, start_mask: np.ndarray, end_mask: np.ndarray) -> np.ndarray:
    H, W = trav.shape
    seen = np.zeros((H, W), dtype=bool)
    prev_r = -np.ones((H, W), dtype=int)
    prev_c = -np.ones((H, W), dtype=int)
    q = deque()

    for r, c in np.argwhere(start_mask):
        rr, cc = int(r), int(c)
        seen[rr, cc] = True
        prev_r[rr, cc] = rr
        prev_c[rr, cc] = cc
        q.append((rr, cc))

    end_rc: tuple[int, int] | None = None
    while q:
        r, c = q.popleft()
        if bool(end_mask[r, c]):
            end_rc = (int(r), int(c))
            break
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and bool(trav[rr, cc]) and not bool(seen[rr, cc]):
                seen[rr, cc] = True
                prev_r[rr, cc] = int(r)
                prev_c[rr, cc] = int(c)
                q.append((int(rr), int(cc)))

    assert not (end_rc is None)
    path_mask = np.zeros((H, W), dtype=bool)
    r, c = end_rc
    while True:
        path_mask[r, c] = True
        pr, pc = int(prev_r[r, c]), int(prev_c[r, c])
        if pr == r and pc == c:
            break
        r, c = pr, pc
    return path_mask


def _find_bottleneck_mask(
    g: np.ndarray,
    trav: np.ndarray,
    feasible: np.ndarray,
    mover: dict,
    dest: dict,
    river: int,
) -> np.ndarray:
    n = int(mover["n"])
    cover = _placement_cover(feasible, n, g.shape)
    unsupported = (g == int(river)) & (cover == 0)
    assert not (not bool(unsupported.any()))

    start_mask = np.zeros_like(trav, dtype=bool)
    end_mask = np.zeros_like(trav, dtype=bool)
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    start_mask[sr : sr + n, sc : sc + n] = True
    end_mask[dr : dr + n, dc : dc + n] = True
    path_mask = _shortest_pixel_path_mask(trav, start_mask, end_mask)

    labeled, num = scipy_label(unsupported.astype(int))
    assert not (num == 0)

    chosen = np.zeros_like(unsupported, dtype=bool)
    for lab in range(1, int(num) + 1):
        comp = labeled == lab
        if bool((comp & path_mask).any()):
            chosen |= comp
    if not bool(chosen.any()):
        best_size = -1
        best_lab = 1
        for lab in range(1, int(num) + 1):
            size = int((labeled == lab).sum())
            if size > best_size:
                best_size = size
                best_lab = lab
        chosen = labeled == best_lab
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a solid square through a winding river and highlight the chokepoint when the fit fails.

    Input structure:
    - The corner-majority color is background. A blue river forms one connected corridor.
    - Two same-sized solid squares sit on that river: an interior mover square and a boundary-touching destination square.

    Rule:
    - The mover can slide only through placements where an entire NxN window stays inside river-or-square pixels.
    - If the destination placement is reachable, move the mover onto the destination and refill its start with river.
    - If the destination is blocked by a too-narrow neck, leave both squares in place and recolor that bottleneck river segment orange.

    Key insight:
    Pixel connectivity alone is not enough: the blocked neck is still blue and connected, but it belongs to no valid NxN placement.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _detect_roles(g, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    river = _find_river_color(g, bg, int(mover["col"]), int(dest["col"]))
    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _feasible_placements(trav, n)
    h2, w2 = feasible.shape

    assert not (not (0 <= sr < h2 and 0 <= sc < w2))
    assert not (not bool(feasible[sr, sc]))

    dist = _bfs_placement(feasible, sr, sc)
    dest_ok = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0

    out = g.copy()
    if dest_ok:
        out[sr : sr + n, sc : sc + n] = int(river)
        out[dr : dr + n, dc : dc + n] = int(mover["col"])
        return out

    bottleneck = _find_bottleneck_mask(g, trav, feasible, mover, dest, river)
    out[bottleneck] = _MARK
    return out


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


def _river_connected(grid):
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G, n, num_coarse):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    vals = [int(round(first + i * step)) for i in range(num_coarse)]
    for i in range(len(vals) - 1):
        if vals[i + 1] - vals[i] <= n:
            return None
    return vals


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int):
    H, W = grid.shape
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
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
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
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2])
    return candidates[0][:2]


def _carve_bottleneck(grid: np.ndarray, pixel_pos_i, pixel_pos_j, n: int, direction: str) -> bool:
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    keep_width = random.randint(1, n - 1)
    side = random.randint(0, 1)

    if direction == "h":
        gap_lo = min(pc1, pc2) + n
        gap_hi = max(pc1, pc2) - 1
        if gap_hi < gap_lo:
            return False
        max_len = min(4, gap_hi - gap_lo + 1)
        choke_len = random.randint(1, max_len)
        c0 = random.randint(gap_lo, gap_hi - choke_len + 1)
        grid[pr1 : pr1 + n, c0 : c0 + choke_len] = _BG
        r0 = pr1 if side == 0 else pr1 + n - keep_width
        grid[r0 : r0 + keep_width, c0 : c0 + choke_len] = _RIVER
        return True

    gap_lo = min(pr1, pr2) + n
    gap_hi = max(pr1, pr2) - 1
    if gap_hi < gap_lo:
        return False
    max_len = min(4, gap_hi - gap_lo + 1)
    choke_len = random.randint(1, max_len)
    r0 = random.randint(gap_lo, gap_hi - choke_len + 1)
    grid[r0 : r0 + choke_len, pc1 : pc1 + n] = _BG
    c0 = pc1 if side == 0 else pc1 + n - keep_width
    grid[r0 : r0 + choke_len, c0 : c0 + keep_width] = _RIVER
    return True


def _gen_attempt():
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    max_coarse = max(3, (G - 2) // (n + 2))
    coarse_dim = random.randint(3, min(6, max_coarse))
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
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

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]
    result = _extend_to_edge(grid, dest_r, dest_c, n)
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
    if not _touches_border(dest_r, dest_c, n, H, W):
        return None

    touching = _touches_border(mover_r, mover_c, n, H, W)
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
    if _touches_border(mover_r, mover_c, n, H, W):
        return None
    if _edge_distance(mover_r, mover_c, n, H, W) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached:
        candidate_segments = []
        for seg_idx in range(1, len(pixel_pos) - 1):
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[seg_idx + 1]
            if pr1 == pr2 and max(pc1, pc2) - min(pc1, pc2) > n:
                candidate_segments.append(seg_idx)
            if pc1 == pc2 and max(pr1, pr2) - min(pr1, pr2) > n:
                candidate_segments.append(seg_idx)
        if not candidate_segments:
            return None
        late = [idx for idx in candidate_segments if idx >= len(pixel_pos) // 3]
        if late:
            candidate_segments = late
        random.shuffle(candidate_segments)
        seg_idx = candidate_segments[0]
        pr1, pc1 = pixel_pos[seg_idx]
        pr2, pc2 = pixel_pos[seg_idx + 1]
        direction = "h" if pr1 == pr2 else "v"
        if not _carve_bottleneck(grid, (pr1, pc1), (pr2, pc2), n, direction):
            return None
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _feasible_placements(trav, n)
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    h2, w2 = feasible.shape
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and bool(feasible[dest_r, dest_c]) and int(dist[dest_r, dest_c]) >= 0
    if reached != dest_ok:
        return None

    output = grid.copy()
    if dest_ok:
        output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
        output[dest_r : dest_r + n, dest_c : dest_c + n] = _MOVER
    else:
        mover = {"col": _MOVER, "r0": mover_r, "c0": mover_c, "n": n}
        dest = {"col": _DEST, "r0": dest_r, "c0": dest_c, "n": n}
        bottleneck = _find_bottleneck_mask(grid, trav, feasible, mover, dest, _RIVER)
        if not bool(bottleneck.any()):
            return None
        _labeled, num = scipy_label(bottleneck.astype(int))
        if int(num) != 1:
            return None
        output[bottleneck] = _MARK

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
