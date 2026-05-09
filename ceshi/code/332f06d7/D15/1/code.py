# Variant of 332f06d7
# Dimension: D15 — 可达性判定逻辑 (meta)
# Change: 通过BFS在可行放置位置空间上搜索，判断目标位置是否在起点的连通分量内... → 单向通道可达性

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label as scipy_label

_ROOT = Path(__file__).resolve().parents[5]
for _extra in (_ROOT, _ROOT / "solvers"):
    _extra_s = str(_extra)
    if _extra_s not in sys.path:
        sys.path.insert(0, _extra_s)

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D15 variant
#
# concrete_behavior:
#   visual_form:
#   - 保留原题的绿色背景、蜿蜒水道、黑色 mover 方块、红色 destination 方块。
#   - 新增两种“单向水道”颜色: 青色底色 + 黄色箭头尖端。它们总是组成一个细长矩形桥段，
#     黄色端表示前进方向，方向由“青色主体 -> 黄色尖端”决定。
#   - 普通蓝色河道仍可双向滑动；只有这些特殊桥段是单向的。
#   pipeline_impact:
#   - 原题在可行放置位置图上做无向 BFS；现在仍在同一个 placement 图上搜索，
#     但相邻 placement 是否可走，要额外检查“新进入的那一列/行”是否穿过单向桥段，
#     若穿过，则移动方向必须与桥段方向一致。
#   traced_example:
#   - 设 n=2，某条水平桥段被标成“向右”。原逻辑下 placement (5,4) 与 (5,5) 双向连通。
#   - 新逻辑下，从 (5,4)->(5,5) 时，新进入的右边界列落在该桥段上，因此允许；
#     从 (5,5)->(5,4) 时，新进入的左边界列同样落在该桥段上，但方向相反，因此禁止。
#   - 所以几何上仍连通，但有向可达性可能断开。
#
# complexity_calibration:
# - 不引入多套颜色到方向的任意映射，只固定使用“青色主体 + 黄色尖端”的可视箭头语义。
# - 单向标记只出现在笔直桥段，不出现在拐角块上，避免组件粘连成复杂图案。
# - 每个样例都会有至少一个单向桥段，且通常有多个，保证肉眼能看出“这题不是原题”。
#
# hidden_assumptions_audit:
# - 原 `cand.sort(key=color)` 依赖颜色 id 排序来区分 mover/dest；这里改为利用可见结构：
#   destination 一定贴边，mover 一定不贴边但靠近边界。
# - 原 `trav = river|mover|dest` 只考虑单一河道颜色；这里改为 `trav = grid != bg`，因为
#   单向桥段颜色也必须是可通行水道。
# - 原 BFS 默认所有 4 邻接都是合法边；这里每条边都经过 `_step_allowed` 审核。
#
# placement_legality:
# - 生成器仍先造 placement 路径，再渲染成像素河道，因此 mover/dest 与河道天然对齐。
# - 单向桥段只刷在两个 n×n 落点之间的独占连接矩形内，不覆盖方块，不越界，不与其他标记重叠。
# - 目的地继续贴边，mover 继续近边但不贴边，solver 的结构判别有生成保证。
#
# visual_observability:
# - 人类可以直接看到：有些桥段是蓝色普通水道，有些桥段是“青色长条+黄色端点”的箭头水道。
# - 输出仍然只是黑方块移动；若某个箭头反向，则黑方块会停在箭头前的最后可达位置，
#   与原题“只看宽度”有明显不同。
#
# visual_cleanliness:
# - 箭头桥段只放在线性连接条上，不压在拐角、起点、终点方块上，因此图面分层清楚。
# - 颜色总数控制在 6 种: 背景、普通水道、mover、dest、箭头主体、箭头尖端。
#
# new_solver_assumptions_checked_against_generator:
# - 假设恰有两个同尺寸实心方块，其中一个贴边一个不贴边。生成器保证。
# - 假设单向组件是轴对齐矩形，且厚度正好为 n。生成器保证。
# - 假设普通几何河道本身无宽度瓶颈；若出现“不可达”，应主要来自方向限制。生成器用无向 BFS 复核。


_BG = 3
_RIVER = 1
_MOVER = 0
_DEST = 2
_FLOW_BASE = 8
_FLOW_TIP = 4


def _edge_touching(square: dict, h: int, w: int) -> bool:
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    n = int(square["n"])
    return r0 == 0 or c0 == 0 or r0 + n == h or c0 + n == w


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


def _pick_mover_and_dest(squares: list[dict], h: int, w: int) -> tuple[dict, dict]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    edge = [sq for sq in cand if _edge_touching(sq, h, w)]
    inner = [sq for sq in cand if not _edge_touching(sq, h, w)]
    assert edge and inner

    # Destination is the edge-touching square; mover is the non-edge square.
    dest = edge[0]
    mover = inner[0]
    return mover, dest


def _flow_dir_map(g: np.ndarray, n: int) -> dict[tuple[int, int], tuple[int, int]]:
    mask = (g == _FLOW_BASE) | (g == _FLOW_TIP)
    labeled, num = scipy_label(mask.astype(int))
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for lab in range(1, int(num) + 1):
        pts = np.argwhere(labeled == lab)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        base_pts = np.argwhere((labeled == lab) & (g == _FLOW_BASE))
        tip_pts = np.argwhere((labeled == lab) & (g == _FLOW_TIP))
        assert len(base_pts) > 0 and len(tip_pts) > 0
        base_ctr = base_pts.mean(axis=0)
        tip_ctr = tip_pts.mean(axis=0)

        if h == n and w >= 2:
            dr = 0
            dc = 1 if tip_ctr[1] > base_ctr[1] else -1
        elif w == n and h >= 2:
            dr = 1 if tip_ctr[0] > base_ctr[0] else -1
            dc = 0
        else:
            raise AssertionError("one-way component must be an axis-aligned strip of thickness n")

        for r, c in pts:
            out[(int(r), int(c))] = (int(dr), int(dc))
    return out


def _frontier_cells(r: int, c: int, dr: int, dc: int, n: int) -> list[tuple[int, int]]:
    if dr == 1:
        return [(r + n, cc) for cc in range(c, c + n)]
    if dr == -1:
        return [(r - 1, cc) for cc in range(c, c + n)]
    if dc == 1:
        return [(rr, c + n) for rr in range(r, r + n)]
    if dc == -1:
        return [(rr, c - 1) for rr in range(r, r + n)]
    raise AssertionError("step must be 4-neighbor")


def _step_allowed(
    feasible: np.ndarray,
    flow_dirs: dict[tuple[int, int], tuple[int, int]],
    r: int,
    c: int,
    rr: int,
    cc: int,
    n: int,
) -> bool:
    if not (0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc])):
        return False
    dr = int(rr - r)
    dc = int(cc - c)
    marks = {flow_dirs[cell] for cell in _frontier_cells(r, c, dr, dc, n) if cell in flow_dirs}
    return not marks or all(vec == (dr, dc) for vec in marks)


def _bfs_placement(
    feasible: np.ndarray,
    sr: int,
    sc: int,
    n: int,
    flow_dirs: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h2 and 0 <= cc < w2):
                continue
            if dist[rr, cc] != -1:
                continue
            if flow_dirs is None:
                if not bool(feasible[rr, cc]):
                    continue
            else:
                if not _step_allowed(feasible, flow_dirs, r, c, rr, cc, n):
                    continue
            dist[rr, cc] = dist[r, c] + 1
            q.append((rr, cc))
    return dist


def _feasible_windows(trav: np.ndarray, n: int) -> np.ndarray:
    h, w = trav.shape
    h2, w2 = h - n + 1, w - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel that may include one-way bridge segments.

    Input structure:
    - The corner-majority background surrounds a connected water corridor.
    - Two same-sized solid squares lie on the corridor: an interior mover square and an edge-touching destination square.
    - Some straight corridor segments are recolored as cyan strips with a yellow tip; those strips are one-way, pointing from cyan toward yellow.

    Rule:
    - The mover still travels by unit steps in top-left placement space, and every NxN placement must stay fully on traversable water.
    - Crossing ordinary blue water is bidirectional.
    - Crossing a marked strip is only allowed in the strip's pointing direction, determined by the newly entered frontier row or column.
    - If the destination placement is reachable under those directed constraints, the mover replaces it; otherwise the mover stops at the farthest directed-reachable placement.

    Key insight: Geometric connectivity is no longer enough; the reachable set is a directed subgraph extracted from visible one-way bridge markers.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _detect_squares(g, bg)
    assert len(squares) >= 2
    mover, dest = _pick_mover_and_dest(squares, h, w)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = g != bg
    feasible = _feasible_windows(trav, n)
    assert bool(feasible[sr, sc])

    flow_dirs = _flow_dir_map(g, n)
    dist = _bfs_placement(feasible, sr, sc, n, flow_dirs=flow_dirs)

    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(feasible.shape[0]):
            for c in range(feasible.shape[1]):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(100):
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


def _compute_offsets(gsz: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = gsz - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    vals = [int(round(first + i * step)) for i in range(num_coarse)]
    return vals


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> tuple[list[tuple[int, int]], list[dict]]:
    pixel_pos = []
    segments: list[dict] = []
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
            inner_lo = min(pc1, pc2) + n
            inner_hi = max(pc1, pc2) - 1
            if inner_lo <= inner_hi:
                step = 1 if pc2 > pc1 else -1
                segments.append({"kind": "h", "r0": pr1, "r1": pr1 + n - 1, "c0": inner_lo, "c1": inner_hi, "step": step, "path_idx": i})
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
            inner_lo = min(pr1, pr2) + n
            inner_hi = max(pr1, pr2) - 1
            if inner_lo <= inner_hi:
                step = 1 if pr2 > pr1 else -1
                segments.append({"kind": "v", "r0": inner_lo, "r1": inner_hi, "c0": pc1, "c1": pc1 + n - 1, "step": step, "path_idx": i})
    return pixel_pos, segments


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[tuple[int, int] | None, dict | None]:
    h, w = grid.shape
    candidates = []
    for direction in ["top", "bottom", "left", "right"]:
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append(((0, pc), None, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                seg = {"kind": "v", "r0": 0, "r1": pr - 1, "c0": pc, "c1": pc + n - 1, "step": -1, "path_idx": 10**9}
                candidates.append(((0, pc), seg, dist))
        elif direction == "bottom":
            dist = h - (pr + n)
            if dist == 0:
                candidates.append(((pr, pc), None, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, h) for c in range(pc, pc + n)):
                seg = {"kind": "v", "r0": pr + n, "r1": h - 1, "c0": pc, "c1": pc + n - 1, "step": 1, "path_idx": 10**9}
                candidates.append(((h - n, pc), seg, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append(((pr, 0), None, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                seg = {"kind": "h", "r0": pr, "r1": pr + n - 1, "c0": 0, "c1": pc - 1, "step": -1, "path_idx": 10**9}
                candidates.append(((pr, 0), seg, dist))
        else:
            dist = w - (pc + n)
            if dist == 0:
                candidates.append(((pr, pc), None, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, w)):
                seg = {"kind": "h", "r0": pr, "r1": pr + n - 1, "c0": pc + n, "c1": w - 1, "step": 1, "path_idx": 10**9}
                candidates.append(((pr, w - n), seg, dist))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[2])
    pos, seg, _dist = candidates[0]
    return pos, seg


def _paint_oneway_segment(grid: np.ndarray, seg: dict, forward: bool) -> None:
    if seg["kind"] == "h":
        length = int(seg["c1"] - seg["c0"] + 1)
    else:
        length = int(seg["r1"] - seg["r0"] + 1)
    if length < 2:
        return

    tip_len = 1 if length == 2 else random.randint(1, min(2, length - 1))
    grid[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c0"]) : int(seg["c1"]) + 1] = _FLOW_BASE
    sign = int(seg["step"]) if forward else -int(seg["step"])

    if seg["kind"] == "h":
        if sign == 1:
            grid[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c1"]) - tip_len + 1 : int(seg["c1"]) + 1] = _FLOW_TIP
        else:
            grid[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c0"]) : int(seg["c0"]) + tip_len] = _FLOW_TIP
    else:
        if sign == 1:
            grid[int(seg["r1"]) - tip_len + 1 : int(seg["r1"]) + 1, int(seg["c0"]) : int(seg["c1"]) + 1] = _FLOW_TIP
        else:
            grid[int(seg["r0"]) : int(seg["r0"]) + tip_len, int(seg["c0"]) : int(seg["c1"]) + 1] = _FLOW_TIP


def _directional_segments(segments: list[dict]) -> list[dict]:
    return [seg for seg in segments if ((seg["c1"] - seg["c0"] + 1) if seg["kind"] == "h" else (seg["r1"] - seg["r0"] + 1)) >= 2]


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([2, 3, 4])

    min_size = max(12, 5 * n)
    if min_size > 20:
        return None
    gsz = random.randint(min_size, 20)
    h = w = gsz

    min_spacing = n + 2
    max_coarse = max(3, (gsz - 2) // min_spacing)
    coarse_dim = random.randint(3, min(5, max_coarse))
    offsets_r = _compute_offsets(gsz, n, coarse_dim)
    offsets_c = _compute_offsets(gsz, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < min_spacing for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < min_spacing for i in range(len(offsets_c) - 1)):
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
    if walk is None or len(walk) < 5:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    pixel_pos, segments = _render_channel(grid, walk, n, offsets_r, offsets_c)
    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    edge_pos, edge_seg = _extend_to_edge(grid, dest_r, dest_c, n)
    if edge_pos is None:
        return None
    new_dest_r, new_dest_c = edge_pos
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    if edge_seg is not None:
        length = (edge_seg["c1"] - edge_seg["c0"] + 1) if edge_seg["kind"] == "h" else (edge_seg["r1"] - edge_seg["r0"] + 1)
        if length >= 2:
            segments.append(edge_seg)
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == h or dest_c == 0 or dest_c + n == w):
        return None

    touching = mover_r == 0 or mover_r + n == h or mover_c == 0 or mover_c + n == w
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _RIVER
            mover_r = 1
        elif mover_r + n == h:
            grid[h - 1, mover_c : mover_c + n] = _RIVER
            mover_r = h - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, w - 1] = _RIVER
            mover_c = w - n - 1
    if mover_r == 0 or mover_r + n == h or mover_c == 0 or mover_c + n == w:
        return None
    if min(mover_r, h - (mover_r + n), mover_c, w - (mover_c + n)) > 2:
        return None

    dir_candidates = _directional_segments(segments)
    if len(dir_candidates) < 1:
        return None

    random.shuffle(dir_candidates)
    num_marked = random.randint(1, min(4, len(dir_candidates)))
    marked = dir_candidates[:num_marked]
    barrier = None
    if not reached:
        middle = [seg for seg in marked if seg["path_idx"] >= max(1, len(walk) // 4)]
        if not middle:
            middle = [seg for seg in dir_candidates if seg["path_idx"] >= max(1, len(walk) // 4)]
        if not middle:
            return None
        barrier = random.choice(middle)

    for seg in marked:
        _paint_oneway_segment(grid, seg, forward=(seg is not barrier))
    if barrier is not None and barrier not in marked:
        _paint_oneway_segment(grid, barrier, forward=False)

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = grid != _BG
    feasible = _feasible_windows(trav, n)
    if not bool(feasible[mover_r, mover_c]):
        return None
    if not bool(feasible[dest_r, dest_c]):
        return None

    directed = _bfs_placement(feasible, mover_r, mover_c, n, flow_dirs=_flow_dir_map(grid, n))
    undirected = _bfs_placement(feasible, mover_r, mover_c, n, flow_dirs=None)

    dest_directed = int(directed[dest_r, dest_c]) >= 0
    dest_undirected = int(undirected[dest_r, dest_c]) >= 0
    if not dest_undirected:
        return None
    if reached != dest_directed:
        return None
    if not reached:
        if barrier is None:
            return None
        if np.max(directed) <= 0:
            return None

    if not np.array_equal(main(grid), _expected_output(grid)):
        return None
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _pick_mover_and_dest(_detect_squares(g, bg), h, w)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    feasible = _feasible_windows(g != bg, n)
    dist = _bfs_placement(feasible, sr, sc, n, flow_dirs=_flow_dir_map(g, n))
    if int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(feasible.shape[0]):
            for c in range(feasible.shape[1]):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
