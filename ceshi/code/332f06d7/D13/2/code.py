# Variant of 332f06d7
# Dimension: D13 — 方块尺寸与网格尺寸的相关性 (meta)
# Change: n=1~4，网格尺寸与n正相关（n=1→10-16, n=2→12-16, n=3→14-20, n=4→20），网格始终正方形
#         → 移动体和目标尺寸不同；红色目标是更大的正方形，贴边放置，旁边留有一个与黑色移动体同尺寸的蓝色停泊位

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis
# a) Concrete behavior and trace:
#    原题里黑块和红块同为 n×n，求解器直接把“红块左上角”当成终点位置；到达后黑块整块覆盖红块。
#    这个变体里黑块仍是 n×n，但红块改成更大的 m×m（m>n）并贴住一条边。真正的终点不是红块本身，
#    而是红块内侧紧挨着它的一块蓝色 n×n 停泊位。比如 n=2,m=4、红块在上边界时，输入末端会出现：
#    一块 4×4 红方块；其正下方只有一个 2×2 蓝停泊位接入河道。原逻辑会把 4×4 红块误当成可覆盖终点，
#    变体逻辑则必须识别“尺寸不匹配，所以红块不可进入，只能停进旁边那块 2×2 蓝位”。
# b) Complexity calibration:
#    只引入一个新自由度：m=n+1 或 n+2。这样每个样例里尺寸差异肉眼明显，但不至于出现太多类别。
#    停泊位始终与移动体同尺寸，且每图只出现一处，避免匹配歧义。
# c) Old assumptions that break:
#    - 原 main 依赖“有一个尺寸出现至少两次”，这里不再成立。
#    - 原 main 用颜色 id 排序区分 mover/dest；这里禁用该做法，改为几何角色：
#      贴边的大方块是目标，未贴边的小方块是移动体。
#    - 原 main 把目标色也加入可通行区域；这里红色目标不可通行，只能停在其旁边的蓝色停泊位。
# d) Placement legality:
#    生成器保证：红目标整块在边界内、停泊位与目标整边贴合、移动体不触边、河道与停泊位连通、
#    目标本体不与河道重叠、卡口只收窄河道而不打断蓝色连通分量。
# e) Visual observability:
#    2~4 个样例里，人类会稳定看到“黑块比红块小，而且黑块最终停在红块旁边的蓝槽里，而不是盖住红块”。
#    卡口分支仍保留，因此也能看出“停不进去时停在最远可达处”。
# f) Visual cleanliness:
#    红目标保持纯色实心大方块；蓝停泊位只有一个；黑块最终只覆盖蓝槽，不与红块叠色，层次清楚。
# g) Color inference:
#    不做颜色置换，也不依赖颜色数值排序；颜色只承担可见语义：绿背景、蓝河道、黑移动体、红目标。
#
# New solver assumptions audited against the generator:
# - Exactly one non-background solid square touches the border and is larger than the mover: generator enforces a single edge target.
# - Exactly one non-background solid square stays off the border: generator keeps the mover one cell away from every edge.
# - Exactly one n×n all-blue window is flush with the target's inward side: generator creates a single dock and keeps the rest of that side green.
# - The river color is the only remaining non-background color after removing mover/target colors: generator uses a single blue channel.


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _find_solid_squares(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        r0, c0 = int(b[0]), int(b[1])
        n = int(bbox_h(b))
        patch = g[r0 : r0 + n, c0 : c0 + n]
        if not bool((patch == col).all()):
            continue
        touches_edge = r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W
        squares.append({"col": col, "r0": r0, "c0": c0, "n": n, "touches_edge": touches_edge})
    return squares


def _find_dock(g: np.ndarray, target: dict, mover_n: int, river: int) -> tuple[int, int]:
    H, W = g.shape
    tr, tc, m = int(target["r0"]), int(target["c0"]), int(target["n"])
    target_col = int(target["col"])
    candidates: list[tuple[int, int]] = []

    def all_color(r: int, c: int, h: int, w: int, col: int) -> bool:
        if r < 0 or c < 0 or r + h > H or c + w > W:
            return False
        return bool((g[r : r + h, c : c + w] == col).all())

    # The generator keeps the target off corners, so the border-touching side uniquely
    # determines the inward side where the dock must be found.
    if tr == 0:
        r = tr + m
        if r + mover_n <= H:
            for c in range(tc, tc + m - mover_n + 1):
                if all_color(r, c, mover_n, mover_n, river) and all_color(r - 1, c, 1, mover_n, target_col):
                    candidates.append((r, c))
    elif tr + m == H:
        r = tr - mover_n
        if r >= 0:
            for c in range(tc, tc + m - mover_n + 1):
                if all_color(r, c, mover_n, mover_n, river) and all_color(tr, c, 1, mover_n, target_col):
                    candidates.append((r, c))
    elif tc == 0:
        c = tc + m
        if c + mover_n <= W:
            for r in range(tr, tr + m - mover_n + 1):
                if all_color(r, c, mover_n, mover_n, river) and all_color(r, c - 1, mover_n, 1, target_col):
                    candidates.append((r, c))
    elif tc + m == W:
        c = tc - mover_n
        if c >= 0:
            for r in range(tr, tr + m - mover_n + 1):
                if all_color(r, c, mover_n, mover_n, river) and all_color(r, tc, mover_n, 1, target_col):
                    candidates.append((r, c))
    else:
        assert False

    assert len(candidates) == 1
    return candidates[0]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel into a dock beside a larger target square.

    Input structure:
    - The corner-majority background surrounds one connected blue river.
    - A solid black n×n mover sits on the river.
    - A larger solid red m×m square touches the outer border, and one blue n×n dock is flush against its inward side.

    Rule:
    - The black square glides through the blue river using only placements where its full n×n footprint fits on traversable cells.
    - The red square is not traversable; the intended destination is the blue dock attached to it.
    - If that dock is reachable, the mover occupies the dock while the red square stays unchanged.
    - If a choke in the river is narrower than n, the mover instead stops at the farthest reachable placement, and its start area turns blue.

    Key insight:
    The mover and target have different sizes, so “cover the red square” is no longer valid; only the dock that matches the mover's size can be filled.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _find_solid_squares(g, bg)
    assert len(squares) >= 2
    boundary = [s for s in squares if s["touches_edge"]]
    interior = [s for s in squares if not s["touches_edge"]]
    assert boundary and interior

    target = max(boundary, key=lambda s: s["n"])
    mover = min(interior, key=lambda s: s["n"])
    assert int(target["n"]) > int(mover["n"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover["col"]), int(target["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dock_r, dock_c = _find_dock(g, target, n, river)

    trav = (g == river) | (g == int(mover["col"]))
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

    dist = _bfs_placement(feasible, sr, sc)
    if bool(feasible[dock_r, dock_c]) and int(dist[dock_r, dock_c]) >= 0:
        tr, tc = dock_r, dock_c
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


# --- Generator ---


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


def _render_channel(grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]) -> list[tuple[int, int]]:
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


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str) -> None:
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
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _choose_target_size(n: int, G: int) -> int | None:
    choices = [m for m in (n + 1, n + 2) if m <= 6 and m + n + 1 <= G]
    if not choices:
        return None
    return random.choice(choices)


def _extend_to_edge_target(
    grid: np.ndarray, end_r: int, end_c: int, n: int, m: int
) -> tuple[int, int, int, int] | None:
    H, W = grid.shape
    candidates: list[tuple[int, int, int, int, int]] = []

    def target_is_clear(tr: int, tc: int) -> bool:
        return bool((grid[tr : tr + m, tc : tc + m] == _BG).all())

    def noncorner_options(lo: int, hi: int, limit: int) -> list[int]:
        vals = [v for v in range(lo, hi + 1) if 0 < v < limit]
        if vals:
            return vals
        return list(range(lo, hi + 1))

    # Top edge target, dock directly below it.
    dock_r = m
    if dock_r + n <= H:
        lo = max(0, end_c - (m - n))
        hi = min(end_c, W - m)
        if lo <= hi:
            for tc in noncorner_options(lo, hi, W - m):
                if target_is_clear(0, tc):
                    r_lo, r_hi = min(end_r, dock_r), max(end_r, dock_r) + n
                    if bool(np.isin(grid[r_lo:r_hi, end_c : end_c + n], [_BG, _RIVER]).all()):
                        candidates.append((abs(end_r - dock_r), dock_r, end_c, 0, tc))

    # Bottom edge target, dock directly above it.
    dock_r = H - m - n
    if dock_r >= 0:
        lo = max(0, end_c - (m - n))
        hi = min(end_c, W - m)
        if lo <= hi:
            tr = H - m
            for tc in noncorner_options(lo, hi, W - m):
                if target_is_clear(tr, tc):
                    r_lo, r_hi = min(end_r, dock_r), max(end_r, dock_r) + n
                    if bool(np.isin(grid[r_lo:r_hi, end_c : end_c + n], [_BG, _RIVER]).all()):
                        candidates.append((abs(end_r - dock_r), dock_r, end_c, tr, tc))

    # Left edge target, dock directly right of it.
    dock_c = m
    if dock_c + n <= W:
        lo = max(0, end_r - (m - n))
        hi = min(end_r, H - m)
        if lo <= hi:
            for tr in noncorner_options(lo, hi, H - m):
                if target_is_clear(tr, 0):
                    c_lo, c_hi = min(end_c, dock_c), max(end_c, dock_c) + n
                    if bool(np.isin(grid[end_r : end_r + n, c_lo:c_hi], [_BG, _RIVER]).all()):
                        candidates.append((abs(end_c - dock_c), end_r, dock_c, tr, 0))

    # Right edge target, dock directly left of it.
    dock_c = W - m - n
    if dock_c >= 0:
        lo = max(0, end_r - (m - n))
        hi = min(end_r, H - m)
        if lo <= hi:
            tc = W - m
            for tr in noncorner_options(lo, hi, H - m):
                if target_is_clear(tr, tc):
                    c_lo, c_hi = min(end_c, dock_c), max(end_c, dock_c) + n
                    if bool(np.isin(grid[end_r : end_r + n, c_lo:c_hi], [_BG, _RIVER]).all()):
                        candidates.append((abs(end_c - dock_c), end_r, dock_c, tr, tc))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _dist, dock_r, dock_c, target_r, target_c = candidates[0]

    if dock_r != end_r:
        r_lo, r_hi = min(end_r, dock_r), max(end_r, dock_r) + n
        grid[r_lo:r_hi, dock_c : dock_c + n] = _RIVER
    if dock_c != end_c:
        c_lo, c_hi = min(end_c, dock_c), max(end_c, dock_c) + n
        grid[dock_r : dock_r + n, c_lo:c_hi] = _RIVER
    return dock_r, dock_c, target_r, target_c


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    m = _choose_target_size(n, G)
    if m is None:
        return None

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
    end_r, end_c = pixel_pos[-1]
    extended = _extend_to_edge_target(grid, end_r, end_c, n, m)
    if extended is None:
        return None
    dock_r, dock_c, target_r, target_c = extended

    target_touches = (
        target_r == 0 or target_c == 0 or target_r + m == H or target_c + m == W
    )
    if not target_touches:
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
        elif mover_c + n == W:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[target_r : target_r + m, target_c : target_c + m] = _BG

    if not reached and n >= 2:
        num_choke = random.randint(1, min(3, len(pixel_pos) - 2))
        min_seg = max(1, len(pixel_pos) // 4)
        available = list(range(min_seg, len(pixel_pos) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    # Repaint the final straight extension to the dock in case a chokepoint cut through it.
    if dock_r != end_r:
        r_lo, r_hi = min(end_r, dock_r), max(end_r, dock_r) + n
        grid[r_lo:r_hi, dock_c : dock_c + n] = _RIVER
    if dock_c != end_c:
        c_lo, c_hi = min(end_c, dock_c), max(end_c, dock_c) + n
        grid[dock_r : dock_r + n, c_lo:c_hi] = _RIVER

    grid[target_r : target_r + m, target_c : target_c + m] = _DEST
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER

    trav = (grid == _RIVER) | (grid == _MOVER)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if trav[r : r + n, c : c + n].all():
                feasible[r, c] = True
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dock_ok = feasible[dock_r, dock_c] and dist[dock_r, dock_c] >= 0
    if reached and not dock_ok:
        return None
    if not reached and dock_ok:
        return None

    if dock_ok:
        tr, tc = dock_r, dock_c
    else:
        best_d = -1
        tr, tc = mover_r, mover_c
        for r in range(h2):
            for c in range(w2):
                if dist[r, c] > best_d:
                    best_d = dist[r, c]
                    tr, tc = r, c

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = _MOVER

    try:
        oracle = main(grid)
    except Exception:
        return None
    if not np.array_equal(oracle, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
