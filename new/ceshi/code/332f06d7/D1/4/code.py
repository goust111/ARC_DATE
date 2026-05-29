from __future__ import annotations

# Variant of 332f06d7
# Dimension: D1 — 通道导航方式 (path)
# Change: 沿蓝色水道走廊（corridor_following），方块在宽度>=n的通道内BFS搜索可行放置位置... → 管道跟踪导航
#
# Design analysis:
# spec:
#   variant_summary:
#     - 仍然保留绿色背景、蓝色通道、黑色移动方块、红色目标方块。
#     - 核心变化不再是“在所有可放置位置上做 BFS 找到终点/最远点”，而是“把可放置位置视为一根有朝向的管道并沿其拓扑跟踪”。
#   concretization:
#     - 输入中的蓝色区域由一条主干和一个 T 形分叉组成，黑色方块位于入口叶子，红色方块位于某个出口叶子。
#     - 方块大小仍为 n×n，蓝色通道也按这个 n 渲染，因此 solver 仍要先恢复“哪些 top-left 位置能放下整个 n×n 方块”。
#     - 但得到 feasible placement 图后，不再允许自由 BFS；黑块只能从入口沿当前朝向前进。
#     - 到达分叉点时，固定执行“右转优先，其次直行，再其次左转”，这让同一张图上可能存在 BFS 可达但不会被选择的支路。
#   behavior_trace:
#     - 原题在分叉图上会把红块视作普通可达目标，只要存在一条宽度足够的路线就会到达红块。
#     - 新题在同样的 placement 图上，先由入口的唯一邻居确定初始朝向；随后每一步只看当前节点的局部拓扑。
#     - 例如：黑块自上向下进入 T 形结点，左右两侧都可通。旧逻辑会因为红块在左/右任意一侧都能 BFS 到达而追到红块；新逻辑只会按“右转优先”进入右手支路。
#     - 如果右手支路中段被收窄到 < n，旧逻辑可能改走左侧到达红块；新逻辑则会在右手支路卡口前停下。
#   complexity_calibration:
#     - 只引入一个稳定、可观察的自由度：分叉时的右转规则。
#     - 不引入颜色重映射、对象内部纹理或多级符号，避免把 D1 变化和其他维度混在一起。
#     - 为了避免退化为原题，生成器会持续产出“红块在未被选中的支路”或“被选中的支路有卡口”的样本。
#   implicit_assumptions_review:
#     - 原 main() 中的 BFS 默认“所有 feasible 邻接都可以任意探索”，这个前提在变体中失效，必须改成单路径 trace。
#     - 原代码用颜色值排序决定 mover/dest，这属于对颜色 id 的隐藏依赖；这里改成直接寻找黑块和红块两个显式角色。
#     - 原题允许通道只是连通即可；新题还隐含要求 mover 的 placement 节点是叶子、分叉结点的度数为 3，生成器负责保证。
#   placement_legality:
#     - 生成器先在 placement 网格上构造一棵无环 T 形树，再把每个 placement 节点渲染为 n×n 蓝块并取并集。
#     - 这样 solver 通过“整块 n×n 都在蓝/黑/红内”的判定时，恢复出来的 feasible placement 图与设计图一致，不会额外长出斜向捷径。
#     - 卡口只在被选中的直线支路中段雕刻，保持蓝色像素连通，但让 n×n 放置图在该处断开。
#   visual_observability:
#     - 人类可以直接看到：黑块从某个边界入口进入、在 T 结点遇到两个出口、输出总是顺着相对运动方向的右手支路前进。
#     - 当红块位于左手支路时，输出会明显“不去追红块”，这正是与原 BFS 规则最可观察的差异。
#   visual_cleanliness:
#     - 仅使用 4 种颜色，结构是粗管道和两个纯色方块，没有叠加标记。
#     - 分叉保持正交、无遮挡，卡口只出现在单一支路中部，因此网格读图干净。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, neighbors4


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR_CYCLE = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


def _turn_right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIR_CYCLE[(_DIR_CYCLE.index(d) + 1) % 4]


def _turn_left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIR_CYCLE[(_DIR_CYCLE.index(d) - 1) % 4]


def _line_positions(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    out: list[tuple[int, int]] = []
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        for c in range(c1, c2 + step, step):
            out.append((int(r1), int(c)))
    else:
        step = 1 if r2 >= r1 else -1
        for r in range(r1, r2 + step, step):
            out.append((int(r), int(c1)))
    return out


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _neighbors_in_feasible(feasible: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    h2, w2 = feasible.shape
    out: list[tuple[int, int]] = []
    for rr, cc in neighbors4(r, c):
        if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]):
            out.append((int(rr), int(cc)))
    return out


def _trace_pipe(
    feasible: np.ndarray,
    start: tuple[int, int],
    dest: tuple[int, int],
) -> tuple[int, int]:
    cur = (int(start[0]), int(start[1]))
    prev: tuple[int, int] | None = None
    heading: tuple[int, int] | None = None

    for _ in range(feasible.size + 5):
        if cur == dest:
            return cur

        forward = [p for p in _neighbors_in_feasible(feasible, cur) if p != prev]
        if not forward:
            return cur

        if prev is None:
            assert len(forward) == 1
            nxt = forward[0]
        elif len(forward) == 1:
            nxt = forward[0]
        else:
            assert heading is not None
            by_dir = {(p[0] - cur[0], p[1] - cur[1]): p for p in forward}
            order = [_turn_right(heading), heading, _turn_left(heading), (-heading[0], -heading[1])]
            nxt = None
            for d in order:
                if d in by_dir:
                    nxt = by_dir[d]
                    break
            assert nxt is not None

        heading = (int(nxt[0] - cur[0]), int(nxt[1] - cur[1]))
        prev, cur = cur, nxt

    assert False, "pipe trace exceeded step budget"


def _find_role_squares(g: np.ndarray, bg: int) -> tuple[dict, dict]:
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

    movers = [s for s in squares if int(s["col"]) == _MOVER]
    dests = [s for s in squares if int(s["col"]) == _DEST]
    assert movers and dests

    mover = max(movers, key=lambda s: int(s["n"]))
    same_size_dests = [s for s in dests if int(s["n"]) == int(mover["n"])]
    assert same_size_dests
    dest = max(same_size_dests, key=lambda s: (int(s["r0"]) + int(s["c0"])))
    return mover, dest


def _find_river_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) not in (int(bg), _MOVER, _DEST):
            return int(col)
    assert False, "river color not found"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push a solid square through a branched pipe, following the pipe's turning rule rather than a global search.

    Input structure:
    - bg dominates the corners. A blue pipe is wide enough for an n×n square and may split into multiple orthogonal branches.
    - One black n×n square marks the entering block; one red n×n square marks a same-sized target chamber on one branch.

    Rule:
    - Convert the blue pipe into feasible n×n top-left placements, then trace the block from its entrance leaf.
    - The initial step is forced by the unique outgoing pipe segment; later, at a junction, the block prefers right turn, then straight, then left.
    - If the chosen branch stays wide enough, the block keeps following it until it reaches the red square or a terminal leaf.
    - If that chosen branch narrows below n, the block stops at the last feasible placement before the pinch point, and the start cell becomes blue.

    Key insight: Movement is controlled by local pipe topology and heading at each junction, so a red square on another BFS-reachable branch can be ignored.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_role_squares(g, bg)
    river = _find_river_color(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == river) | (g == _MOVER) | (g == _DEST)
    feasible = _build_feasible(trav, n)

    assert bool(feasible[sr, sc])
    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1]:
        assert bool(feasible[dr, dc])
    else:
        assert False, "destination placement out of feasible grid"

    tr, tc = _trace_pipe(feasible, (sr, sc), (dr, dc))

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def _placement_bfs(feasible: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = [(sr, sc)]
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        for rr, cc in _neighbors_in_feasible(feasible, (r, c)):
            if int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int) -> None:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)
    direction = "h" if pr1 == pr2 else "v"

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(3, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H and 0 <= c < W:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(3, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= r < H and 0 <= cc < W:
                    grid[r, cc] = _BG


def _render_nodes(grid: np.ndarray, nodes: set[tuple[int, int]], n: int) -> None:
    for r, c in nodes:
        grid[r : r + n, c : c + n] = _RIVER


def _orientation_layout(
    P: int,
) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    side = random.choice(["top", "bottom", "left", "right"])
    margin = 3
    assert P >= 8
    if side == "top":
        jc = random.randint(margin, P - 1 - margin)
        jr = random.randint(margin, P - 1 - margin)
        return side, (0, jc), (jr, jc), (jr, P - 1), (jr, 0)
    if side == "bottom":
        jc = random.randint(margin, P - 1 - margin)
        jr = random.randint(margin, P - 1 - margin)
        return side, (P - 1, jc), (jr, jc), (jr, 0), (jr, P - 1)
    if side == "left":
        jr = random.randint(margin, P - 1 - margin)
        jc = random.randint(margin, P - 1 - margin)
        return side, (jr, 0), (jr, jc), (0, jc), (P - 1, jc)
    jr = random.randint(margin, P - 1 - margin)
    jc = random.randint(margin, P - 1 - margin)
    return side, (jr, P - 1), (jr, jc), (P - 1, jc), (0, jc)


def _build_tree_nodes(
    start: tuple[int, int],
    junction: tuple[int, int],
    left_leaf: tuple[int, int],
    right_leaf: tuple[int, int],
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    trunk = _line_positions(start, junction)
    left_branch = _line_positions(junction, left_leaf)
    right_branch = _line_positions(junction, right_leaf)
    nodes = set(trunk) | set(left_branch) | set(right_branch)
    return nodes, trunk, left_branch, right_branch


def _scenario_choice() -> str:
    return random.choices(
        ["chosen_dest", "other_dest", "blocked_dest"],
        weights=[30, 40, 30],
        k=1,
    )[0]


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3, 4])
    G = random.randint(max(12, 4 * n + 2), 20)
    P = G - n + 1
    if P < 8:
        return None

    _side, start, junction, left_leaf, right_leaf = _orientation_layout(P)
    nodes, trunk, left_branch, right_branch = _build_tree_nodes(start, junction, left_leaf, right_leaf)

    if len(trunk) < 4 or len(left_branch) < 4 or len(right_branch) < 4:
        return None

    grid = np.full((G, G), _BG, dtype=int)
    _render_nodes(grid, nodes, n)

    scenario = _scenario_choice()
    dest = right_leaf if scenario != "other_dest" else left_leaf

    if scenario == "blocked_dest":
        gate_branch = right_branch
        gate_idx = random.randint(2, len(gate_branch) - 2)
        _add_chokepoint(grid, gate_branch[gate_idx], gate_branch[gate_idx - 1], n)

    sr, sc = start
    dr, dc = dest
    grid[sr : sr + n, sc : sc + n] = _MOVER
    grid[dr : dr + n, dc : dc + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _build_feasible(trav, n)
    if not bool(feasible[sr, sc]) or not bool(feasible[dr, dc]):
        return None

    start_deg = len(_neighbors_in_feasible(feasible, start))
    junction_deg = len(_neighbors_in_feasible(feasible, junction))
    if start_deg != 1 or junction_deg != 3:
        return None

    traced = _trace_pipe(feasible, start, dest)
    dist = _placement_bfs(feasible, start)
    bfs_reaches_dest = int(dist[dr, dc]) >= 0

    if scenario == "chosen_dest" and traced != dest:
        return None
    if scenario == "other_dest" and (traced == dest or not bfs_reaches_dest):
        return None
    if scenario == "blocked_dest" and (traced == dest or bfs_reaches_dest):
        return None

    out = grid.copy()
    out[sr : sr + n, sc : sc + n] = _RIVER
    tr, tc = traced
    out[tr : tr + n, tc : tc + n] = _MOVER
    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
