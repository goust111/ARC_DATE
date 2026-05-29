# Combo variant of 88bcf3b4 (index 24)
# combo_spec: D6/0+D9/0+D17/2
# Name: 矩形正交无alongside L形路径
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题的三元组：一条垂直于边界的 bar、terminal 旁边的一段不规则细线 segment、以及 terminal 前方的障碍物。
#    - 但输出不再是原题的 3 相位 FL/FR + F + 反向对角；融合后只剩 2 相位：
#      1) approach：从根部一侧沿 (fwd + side) 的对角线靠近障碍物；
#      2) exit：在矩形障碍物的外角点上方（或左侧）立刻改成纯 side 方向的正交退出。
#    - 具体例子：top-edge bar 且 segment 在 terminal 右侧时，局部坐标里 segment 从 (0,1) 出发，
#      approach 可能走到 (1,2),(2,3)，随后不再有 alongside，而是直接在 turn=(2,3) 处改成
#      (2,4),(2,5)...；矩形障碍物则放在 turn 正前方一格开始的区域里，例如角点在 (3,3)，向前和向内扩张。
#    - 与原题相比，差异不只是“退出方向变了”，而是“沿边阶段被删掉”后，turn 点完全由矩形的近端角点决定；
#      这让输出从包裹式 V 形变成可预测的 A*O* 尖角 L 形。
# b) 复杂度校准
#    - 没有把问题简化成“随便找个矩形然后硬编码尾巴”，仍保留：bar 检测、segment 主路径抽取、最近障碍物匹配、
#      多 triple 共存、output-first 生成与 oracle 回验。
#    - 为避免视觉上退化回原题或退化成只有一两格的小折线，generator 强制：
#      approach >= 1，orth exit >= 2，矩形宽高 >= 2，且 turn 与矩形近端角点关系固定。
#    - approach 长度只取少量档位（1 或 2），这是刻意控制自由度：样例之间会重复同一几何模式，
#      人类能在少量示例中看出“对角靠近矩形角，再横/竖向甩开”的统一规律。
# c) 隐式假设审查
#    - 原 solver 默认 obstacle 可以是不规则 sprite，并通过 skin BFS 走出一段 alongside；本融合版下该前提失效。
#      新 solver 改为先验证 obstacle 是严格实心矩形，再直接从矩形的“近端外角”推导 turn。
#    - 原 solver 用 boundary_path 最后一步决定 diagonal exit；这里改为固定 pure side exit。
#    - 原 solver 的单调性校验是 FL*F*FR* / FR*F*FL*；这里改成 A*O*。
#    - 原 generator 可以任意长障碍物并靠 skin 路径兜住；这里必须保证矩形近端角点正好落在
#      turn+fwd 位置，否则 solver 的角点推理会失真，因此 generator 明确按该关系构造矩形。
# d) 放置合法性
#    - 仍采用 output-first：先构造 bar / rectangle / output 路径，再反推出输入 segment，并用 main(input)==output 校验。
#    - `_place_triple` 继续对 input+output+obstacle 的并集做扩张预留，避免多个 triple 相互干扰。
#    - 矩形宽度受 approach 长度限制，不允许跨过 terminal 的正前轴线；这样 obstacle 始终停留在 turn 的“内侧”。
# e) 视觉可观察性
#    - 人类能在 2-4 个样例里直接看到：障碍物从不规则 blob 变成了矩形块；segment 先斜着靠近，
#      到达矩形近端角附近后立即拐成水平/垂直尾巴，没有任何沿边直行阶段。
#    - 由于 exit 至少 2 格、矩形至少 2x2，这个变化不会被误看成原题的短暂边界扰动。
# f) 可视化整洁度
#    - 矩形始终位于 turn 的内侧和前方，output tail 则位于矩形外侧；两者不重叠，层次明确。
#    - 限制多个 triple 使用不同边并带 padding 放置，减少局部拥挤导致的识别歧义。
# g) 变化的可推断性
#    - 颜色角色维持原题的三色分工：bar / segment / obstacle；不依赖颜色值大小、排序或隐藏标签。
#    - 变化完全由几何关系驱动：谁是 bar、谁是 segment、谁是矩形 obstacle 都能靠形状和位置看出来。
# h) 新 solver 假设与 generator 保障
#    - solver 假设 obstacle component 是严格填满 bbox 的矩形；generator 只产生这种矩形。
#    - solver 假设 terminal 邻接的有效 segment 是 1-cell-thick 简单路径且附着点度数为 1；generator 保持原题输入笔画约束。
#    - solver 假设最近矩形 obstacle 唯一；generator 在多 triple 情况下额外筛掉“别的 obstacle 更近”的样本。

from __future__ import annotations

import random
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        rows = sorted(set(rs))
        return rows == list(range(min(rows), max(rows) + 1))
    return False


def _bar_endpoints(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        c = cs[0]
        return (min(rs), c), (max(rs), c)
    r = rs[0]
    return (r, min(cs)), (r, max(cs))


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bar_forward_dir(comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _dot(a: tuple[int, int], b: tuple[int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1]


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)

    def neighbors(p: tuple[int, int]):
        r, c = p
        for q in neighbors4(r, c):
            if q in sset:
                yield q
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            yield q

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if nxt in prev:
                continue
            prev[nxt] = cur
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    far = max(dist.keys(), key=lambda p: (dist[p], p[0], p[1]))
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    sset = set(seg_pts)
    deg = {p: 0 for p in sset}
    for r, c in sset:
        for q in neighbors4(r, c):
            if q in sset:
                deg[(r, c)] += 1
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            deg[(r, c)] += 1
    return deg


def _rectangle_cells(comp: list[tuple[int, int]]) -> set[tuple[int, int]] | None:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return rect if rect == set(comp) else None


def _approach_turn_from_rectangle(
    obstacle_set: set[tuple[int, int]],
    terminal: tuple[int, int],
    seg_start: tuple[int, int],
    fwd: tuple[int, int],
    side: tuple[int, int],
) -> tuple[tuple[int, int], int]:
    tr, tc = terminal

    def proj_fwd(p: tuple[int, int]) -> int:
        return _dot((p[0] - tr, p[1] - tc), fwd)

    def proj_side(p: tuple[int, int]) -> int:
        return _dot((p[0] - tr, p[1] - tc), side)

    corner = min(obstacle_set, key=lambda p: (proj_fwd(p), -proj_side(p), p[0], p[1]))
    turn = (corner[0] - fwd[0], corner[1] - fwd[1])

    delta = (turn[0] - seg_start[0], turn[1] - seg_start[1])
    approach_f = _dot(delta, fwd)
    approach_s = _dot(delta, side)
    assert approach_f >= 1, "approach must be non-empty"
    assert approach_f == approach_s, "turn must be reachable by diagonal-only approach"
    return turn, approach_f


def _classify_route(route: list[tuple[int, int]], fwd: tuple[int, int], side: tuple[int, int]) -> str:
    classes = []
    for a, b in zip(route, route[1:]):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        step = (dr, dc)
        if step == (fwd[0] + side[0], fwd[1] + side[1]):
            classes.append("A")
        elif step == side:
            classes.append("O")
        else:
            classes.append("X")
    return "".join(classes)


# Overview:
# - Rehang each irregular segment onto the nearest filled rectangle using a two-phase L turn:
#   diagonal approach first, then immediate orthogonal exit.
#
# Input structure:
# - Background plus repeated bar / segment / rectangle triples; the bar touches a border, the segment is a thin
#   irregular stroke attached at the bar terminal, and the obstacle component is a strict rectangle.
#
# Rule:
# - Keep the bar and rectangle fixed; move only the segment so it approaches the rectangle on the root side with
#   forward-diagonal steps and, at the near outer corner, switches directly into a pure horizontal/vertical side exit.
#
# Key insight:
# - Because the obstacle is rectangular and there is no alongside phase, the bend cell is no longer inferred from a
#   skin walk; it is determined by the rectangle's nearest outer corner.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing each irregular segment toward the nearest rectangular obstacle, preserving length,
    but replace the old wraparound path with a two-phase diagonal-then-orthogonal corner turn.

    Input structure:
    - Background plus one or more triples: a straight bar touching a border, a thin irregular segment attached
      orthogonally to the bar terminal, and a nearby obstacle component that is a filled rectangle.

    Rule:
    - Keep every bar and rectangle fixed.
    - For each attached segment, recover its length, find the nearest rectangle, move the segment so it approaches
      the rectangle on the root side by repeated forward-diagonal steps, then immediately exit in the pure side
      direction from the rectangle's near outer corner; there is no along-edge phase.
    - The painted segment keeps the same number of pixels and never overlaps the rectangle.

    Key insight:
    - Once the obstacle is known to be a strict rectangle, the bend point is determined by a single predictable
      corner cell: the segment heads diagonally to the cell just outside that corner and then continues orthogonally,
      yielding an A*O* L-shaped route instead of the original three-phase wrap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue
                terminal_ep = e1 if touch0 else e0
                tr, tc = terminal_ep
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=True):
                if cand_start not in comp:
                    continue
                if len(comp) < 2:
                    break
                deg = _segment_degrees(comp)
                if deg.get(cand_start, 0) != 1:
                    break
                path = _segment_main_path(comp, cand_start)
                seg_len = len(path)
                key = (seg_len, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        fwd = _bar_forward_dir(bar_comp)
        side = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
        assert abs(side[0]) + abs(side[1]) == 1, "segment root must be orthogonally adjacent to terminal"
        assert _dot(side, fwd) == 0, "segment root must lie on a side direction"

        obstacle_candidates = []
        for col in colors:
            if col in (bar_col, seg_col):
                continue
            for comp in _components(g, col, conn8=True):
                rect = _rectangle_cells(comp)
                if rect is None:
                    continue
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in rect)
                mr = min(r for r, _ in rect)
                mc = min(c for _, c in rect)
                obstacle_candidates.append(((d, mr, mc), rect))
        assert obstacle_candidates, "must find at least one rectangular obstacle"
        obstacle_set = min(obstacle_candidates, key=lambda item: item[0])[1]

        turn, approach_len = _approach_turn_from_rectangle(obstacle_set, terminal, seg_start, fwd, side)
        route = [seg_start]
        cur = seg_start
        for _ in range(approach_len):
            cur = (cur[0] + fwd[0] + side[0], cur[1] + fwd[1] + side[1])
            route.append(cur)
        assert route[-1] == turn, "approach must land on the derived turn cell"
        assert len(route) < seg_len, "route must include a non-empty orthogonal exit"

        while len(route) < seg_len:
            nxt = (route[-1][0] + side[0], route[-1][1] + side[1])
            assert 0 <= nxt[0] < h and 0 <= nxt[1] < w, "orthogonal exit left the grid"
            assert nxt not in obstacle_set, "orthogonal exit cannot enter the rectangle"
            assert int(g[nxt]) in (bg, seg_col), "orthogonal exit cannot overwrite another object"
            route.append(nxt)

        assert not any(p in obstacle_set for p in route), "output segment overlaps obstacle"
        step_signature = _classify_route(route, fwd, side)
        assert re.fullmatch(r"A*O*", step_signature), f"unexpected route signature: {step_signature}"
        moves.append((seg_comp, seg_col, route))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
APPROACH_LEN_MIN, APPROACH_LEN_MAX = 1, 2
EXIT_LEN_MIN = 2
RECT_H_MIN, RECT_H_MAX = 2, 4
RECT_W_MIN = 2
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 500
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _build_input_path(seg_len: int, turns: int, side_sign: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side_sign)]
    diag_step = (-1, side_sign)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < DIAG_START_PROB
            elif last_was_diag:
                is_diag = random.random() < DIAG_CONTINUE_PROB
            else:
                is_diag = random.random() < DIAG_START_PROB
            step = diag_step if is_diag else orth_step
            if last_was_diag is not None and is_diag != last_was_diag:
                turns_left -= 1
            max_run = remaining - turns_left
            run = random.randint(1, max(1, min(max_run, remaining)))
        last_was_diag = (step == diag_step)
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _build_output_path(seg_len: int, side_sign: int, approach_len: int) -> list[tuple[int, int]]:
    coords = [(0, side_sign)]
    for _ in range(approach_len):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc + side_sign))
    while len(coords) < seg_len:
        pr, pc = coords[-1]
        coords.append((pr, pc + side_sign))
    return coords


def _build_rectangle(approach_len: int, side_sign: int, height: int, width: int) -> list[tuple[int, int]]:
    corner_r = approach_len + 1
    corner_c = side_sign * (approach_len + 1)
    rect = []
    for dr in range(height):
        for ds in range(width):
            rect.append((corner_r + dr, corner_c - side_sign * ds))
    return rect


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side_sign = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    approach_max = min(APPROACH_LEN_MAX, seg_len - 1 - EXIT_LEN_MIN)
    if approach_max < APPROACH_LEN_MIN:
        return None
    approach_len = random.randint(APPROACH_LEN_MIN, approach_max)
    exit_len = seg_len - 1 - approach_len
    if exit_len < EXIT_LEN_MIN:
        return None

    width_max = min(approach_len + 1, 3)
    if width_max < RECT_W_MIN:
        return None
    rect_w = random.randint(RECT_W_MIN, width_max)
    rect_h = random.randint(RECT_H_MIN, RECT_H_MAX)

    output_seg = _build_output_path(seg_len, side_sign, approach_len)
    seg_set = set(output_seg)
    obstacle = _build_rectangle(approach_len, side_sign, rect_h, rect_w)
    obstacle_set = set(obstacle)
    if obstacle_set & seg_set:
        return None
    if obstacle_set & bar_set:
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    if max_turns < INPUT_TURN_MIN:
        return None
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side_sign)
    if _is_4adj(input_seg, obstacle):
        return None

    signature = _classify_route(output_seg, (1, 0), (0, side_sign))
    if not re.fullmatch(r"A*O*", signature):
        return None
    if "O" not in signature:
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]

    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 11:
        max_triples = 1
    elif n < 15:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_triple(n)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["output_seg"], seg_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if len(placed_triples) > 1:
        for i, placed in enumerate(placed_triples):
            tr, tc = placed["terminal"]
            own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
            other = min(
                abs(r - tr) + abs(c - tc)
                for j, other_placed in enumerate(placed_triples)
                if j != i
                for r, c in other_placed["obstacle"]
            )
            if own >= other:
                return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
