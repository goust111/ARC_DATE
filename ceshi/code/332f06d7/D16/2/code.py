# Variant of 332f06d7
# Dimension: D16 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——角色由固定颜色值隐式编码（黑=移动体，红=目标，蓝=通道，绿=背景）... → 目标方块内编码

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers" / "arc_utils.py"
    if _cand.exists():
        sys.path.append(str(_parent / "solvers"))
        break

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the D16 variant
# design_analysis:
#   concretization:
#     - 保留原题的绿色背景、蓝色水道、黑色 mover、红色目标块与同尺寸滑动机制。
#     - 变化只发生在目标块内部：目标不再是纯红，而是红色 n×n 方块内嵌 1-3 个黄色像素。
#     - 黄色像素数量 k 编码“离原始终点还要保留多少整块步幅”：
#       `back_steps = (k - 1) * n`。1 个标记沿用原题终点；2/3 个标记分别提前 1/2 个整块边长停下。
#   behavior_trace:
#     - 原题流程：先找 mover 与纯色目标，再在可放置网格上 BFS，reachable 时去目标，stuck 时去最远可达点。
#     - 本变体：先找 mover，再把“除背景/河道/mover 外”的完整 n×n 混色块识别为目标；先算出原题终点，
#       再沿 BFS 首次发现的父指针回退 `back_steps`。
#     - 例：n=3，若原题终点 top-left 是 (4,12)，目标里有 3 个黄点，则本题终点改成
#       BFS 路径上往回 6 步的位置；如果原题本来就被卡口挡住，则从“卡口前最远点”往回 6 步。
#   complexity_calibration:
#     - 只用 1 种标记色，避免“颜色含义”额外学习；真正需要解码的是数量。
#     - 标记数量只取 1/2/3 三档，且生成器会强制末端直道足够长，让三档差异可见且不退化。
#     - 为保证目标内部还能明显保留“红色主体 + 黄色标记”，生成器限制 n ∈ {2, 3}。
#   hidden_assumptions_review:
#     - 原代码的“目标必须是纯色连通块”前提失效，因此不能再靠 monochromatic square 检测目标。
#     - 原代码用颜色排序决定 mover/dest，这既依赖旧属性，也违反“不可见颜色 id 不参与规则推断”。
#       新 solver 改成：mover = 唯一实心方块；target = 其余非背景/非河道/非 mover 颜色并集的 n×n 方块。
#     - 原代码的 traversable mask 只包含 mover 色、dest 色、river 色；现在必须允许目标内部的标记色。
#       直接使用 `g != bg` 更稳妥。
#   placement_legality:
#     - 目标并集始终占满一个 n×n bbox，不与 mover 重叠，不越界。
#     - stuck 情况的卡口只挖掉末端直道中的单列/单行部分像素，仍保留 1..n-1 像素宽的蓝色细流，
#       所以河道联通但 n×n mover 无法通过。
#     - 生成器只接受 `main(input)` 成功且与内部 oracle 一致的样本。
#   visual_observability:
#     - 解题者能直接看到：目标块总是红底黄点，黄点数量变化会让黑块离“原本应该到达的位置”更早停下。
#     - 末端直道被刻意拉长，保证 2/3 个黄点时黑块通常与目标完全分离，不会把标记盖得看不清。
#   cleanliness:
#     - 标记只放在目标块左上角的固定 row-major 顺序中，避免误把具体形状当作编码。
#     - 不引入额外对象或额外颜色映射，只在目标块内部叠一层少量像素。
#   generator_guarantees:
#     - reachable / stuck 两类都保留；卡口只放在目标前的直尾段。
#     - 对每个样本都验证：目标可被正确识别，符号数为 1..3，若符号数 > 1，则最终黑块不与目标重叠。

_BG, _RIVER, _MOVER, _DEST, _MARK = 3, 1, 0, 2, 4


def _solid_square_components(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        col = int(comp["color"])
        r0, c0 = int(b[0]), int(b[1])
        n = int(bbox_h(b))
        patch = g[r0 : r0 + n, c0 : c0 + n]
        if bool((patch == col).all()):
            squares.append({"color": col, "r0": r0, "c0": c0, "n": n})
    return squares


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert not (coords.size == 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bfs_with_parent(feasible: np.ndarray, sr: int, sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent_r = -np.ones((h2, w2), dtype=int)
    parent_c = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    for_start = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        r, c = q.popleft()
        for dr, dc in for_start:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent_r[rr, cc] = r
                parent_c[rr, cc] = c
                q.append((rr, cc))
    return dist, parent_r, parent_c


def _rewind(parent_r: np.ndarray, parent_c: np.ndarray, tr: int, tc: int, steps: int) -> tuple[int, int]:
    r, c = int(tr), int(tc)
    left = int(steps)
    while left > 0 and parent_r[r, c] != -1:
        nr, nc = int(parent_r[r, c]), int(parent_c[r, c])
        r, c = nr, nc
        left -= 1
    return int(r), int(c)


def _solve_state(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    solid_squares = [s for s in _solid_square_components(g, bg) if int(s["n"]) >= 2]
    assert not (len(solid_squares) < 1)
    solid_squares.sort(key=lambda s: (-int(s["n"]), int(s["r0"]), int(s["c0"])))
    mover = solid_squares[0]

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), int(mover["color"])):
            continue
        river = int(col)
        break
    assert not (river is None)

    target_mask = (g != int(bg)) & (g != int(river)) & (g != int(mover["color"]))
    dr0, dc0, dr1, dc1 = _mask_bbox(target_mask)
    assert not (dr1 - dr0 + 1 != n or dc1 - dc0 + 1 != n)
    assert not (not bool(target_mask[dr0 : dr1 + 1, dc0 : dc1 + 1].all()))

    dr, dc = int(dr0), int(dc0)
    target_patch = g[dr : dr + n, dc : dc + n]
    patch_counts = Counter(map(int, target_patch.flatten()))
    assert not (len(patch_counts) != 2)
    base_col, base_ct = patch_counts.most_common(1)[0]
    marker_count = int(n * n - base_ct)
    assert not (base_col == int(mover["color"]))
    assert not (marker_count < 1 or marker_count > 3)

    trav = g != int(bg)
    h2, w2 = H - n + 1, W - n + 1
    assert not (h2 <= 0 or w2 <= 0)
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert not (not bool(feasible[sr, sc]))

    dist, parent_r, parent_c = _bfs_with_parent(feasible, sr, sc)

    dest_ok = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if dest_ok:
        end_r, end_c = int(dr), int(dc)
    else:
        best = -1
        end_r, end_c = int(sr), int(sc)
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    end_r, end_c = int(r), int(c)

    back_steps = int((marker_count - 1) * n)
    tr, tc = _rewind(parent_r, parent_c, end_r, end_c, back_steps)

    return {
        "grid": g,
        "bg": int(bg),
        "river": int(river),
        "mover_color": int(mover["color"]),
        "n": int(n),
        "sr": int(sr),
        "sc": int(sc),
        "dr": int(dr),
        "dc": int(dc),
        "marker_count": int(marker_count),
        "back_steps": int(back_steps),
        "dest_ok": bool(dest_ok),
        "end_r": int(end_r),
        "end_c": int(end_c),
        "tr": int(tr),
        "tc": int(tc),
        "dist": dist,
    }


def _render_output(state: dict) -> np.ndarray:
    g = np.array(state["grid"], dtype=int)
    n = int(state["n"])
    sr, sc = int(state["sr"]), int(state["sc"])
    tr, tc = int(state["tr"]), int(state["tc"])
    river = int(state["river"])
    mover_color = int(state["mover_color"])

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = mover_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square along a winding watercourse, but decode the target's internal marks first.

    Input structure:
    - A dominant background color surrounds one connected river.
    - One solid NxN mover square sits on the river.
    - One same-sized target square sits elsewhere on the river; it is mostly one color but contains 1-3 marker pixels.

    Rule:
    - First compute the same placement endpoint as in the base task: the target location if reachable, otherwise the farthest feasible placement before a bottleneck.
    - Then count the target's marker pixels. One marker keeps the base endpoint, while two or three markers pull the stop point back by one or two full square-widths along the BFS path.
    - The mover's starting square is replaced by river color, and the mover is painted at the decoded final stop.

    Key insight:
    The target is now both a place and an instruction. You must read the marker count inside the target square, then rewind the usual sliding destination by whole-square strides.
    """
    state = _solve_state(np.array(grid, dtype=int))
    return _render_output(state)


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


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


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


def _extend_to_edge(
    grid: np.ndarray,
    pr: int,
    pc: int,
    n: int,
    min_tail: int,
) -> tuple[int, int, str, int, int] | None:
    H, W = grid.shape
    candidates = []
    if pr >= min_tail and all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
        candidates.append((0, pc, "v", -1, pr))
    if H - (pr + n) >= min_tail and all(
        grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n)
    ):
        candidates.append((H - n, pc, "v", 1, H - (pr + n)))
    if pc >= min_tail and all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
        candidates.append((pr, 0, "h", -1, pc))
    if W - (pc + n) >= min_tail and all(
        grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W)
    ):
        candidates.append((pr, W - n, "h", 1, W - (pc + n)))
    if not candidates:
        return None
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x[4])
    return candidates[0]


def _adjust_mover_off_edge(grid: np.ndarray, mover_r: int, mover_c: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
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
    if mover_r <= 0 or mover_r + n >= H or mover_c <= 0 or mover_c + n >= W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None
    return int(mover_r), int(mover_c)


def _add_tail_gate(
    grid: np.ndarray,
    tail_start_r: int,
    tail_start_c: int,
    dest_r: int,
    dest_c: int,
    n: int,
    back_steps: int,
) -> bool:
    remove = random.randint(1, n - 1)
    keep_top_or_left = random.randint(0, 1)

    if tail_start_r == dest_r:
        if dest_c > tail_start_c:
            lo = tail_start_c + n + back_steps
            hi = dest_c - 1
        else:
            lo = dest_c + n
            hi = tail_start_c - back_steps - 1
        if lo > hi:
            return False
        gate_col = random.randint(lo, hi)
        if keep_top_or_left == 0:
            grid[tail_start_r : tail_start_r + remove, gate_col] = _BG
        else:
            grid[tail_start_r + n - remove : tail_start_r + n, gate_col] = _BG
        return True

    if dest_r > tail_start_r:
        lo = tail_start_r + n + back_steps
        hi = dest_r - 1
    else:
        lo = dest_r + n
        hi = tail_start_r - back_steps - 1
    if lo > hi:
        return False
    gate_row = random.randint(lo, hi)
    if keep_top_or_left == 0:
        grid[gate_row, tail_start_c : tail_start_c + remove] = _BG
    else:
        grid[gate_row, tail_start_c + n - remove : tail_start_c + n] = _BG
    return True


def _paint_target(grid: np.ndarray, dest_r: int, dest_c: int, n: int, marker_count: int) -> None:
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST
    marker_cells = [(0, 0), (0, 1), (1, 0)]
    for rr, cc in marker_cells[:marker_count]:
        grid[dest_r + rr, dest_c + cc] = _MARK


def _rectangles_overlap(r1: int, c1: int, r2: int, c2: int, n: int) -> bool:
    return not (r1 + n <= r2 or r2 + n <= r1 or c1 + n <= c2 or c2 + n <= c1)


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.randint(1, 3)
    reached = random.random() < 0.5
    n = random.choice([2, 3])

    min_size = max(14, 6 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(4, (G - 2) // min_spacing)
    coarse_dim = random.randint(4, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, n, coarse_rows)
    offsets_c = _compute_offsets(G, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
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
    if walk is None or len(walk) < 5:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mover_r, mover_c = pixel_pos[0]
    raw_dest_r, raw_dest_c = pixel_pos[-1]

    tail_need = (marker_count - 1) * n
    if not reached:
        tail_need += n + 1
    extended = _extend_to_edge(grid, raw_dest_r, raw_dest_c, n, tail_need)
    if extended is None:
        return None
    dest_r, dest_c, _axis, _sign, tail_len = extended
    if dest_r != raw_dest_r:
        r_lo, r_hi = min(dest_r, raw_dest_r), max(dest_r, raw_dest_r) + n
        grid[r_lo:r_hi, raw_dest_c : raw_dest_c + n] = _RIVER
    if dest_c != raw_dest_c:
        c_lo, c_hi = min(dest_c, raw_dest_c), max(dest_c, raw_dest_c) + n
        grid[raw_dest_r : raw_dest_r + n, c_lo:c_hi] = _RIVER

    moved = _adjust_mover_off_edge(grid, mover_r, mover_c, n)
    if moved is None:
        return None
    mover_r, mover_c = moved

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached:
        if not _add_tail_gate(grid, raw_dest_r, raw_dest_c, dest_r, dest_c, n, (marker_count - 1) * n):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    _paint_target(grid, dest_r, dest_c, n, marker_count)

    try:
        state = _solve_state(grid)
    except Exception:
        return None

    if bool(state["dest_ok"]) != reached:
        return None
    if int(state["marker_count"]) != marker_count:
        return None
    if int(state["back_steps"]) != (marker_count - 1) * n:
        return None
    if int(state["dist"][int(state["end_r"]), int(state["end_c"])]) < int(state["back_steps"]):
        return None
    if marker_count > 1 and _rectangles_overlap(int(state["tr"]), int(state["tc"]), dest_r, dest_c, n):
        return None

    out = _render_output(state)
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
