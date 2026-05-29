# Variant of 88bcf3b4
# Dimension: D2 — 边界遍历方向 (path)
# Change: 沿障碍物边界的遍历方向由 bar 末端到障碍物最近点的位置关系隐式决定
#         （原题里基本等价于 root side 决定先走 FL 还是 FR）
#         改为：障碍物颜色决定方向；两种固定障碍物颜色分别对应两种镜像绕行方向。
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original:
#       - 输入线段附着在 terminal 左右哪一侧，会隐式决定输出线段先朝哪一侧斜进，
#         也就决定了“沿障碍物皮肤”走的是哪一个镜像方向。
#       - solver 中最关键的旧假设是：
#         off_c / off_r（即 seg_start 相对 terminal 的侧向偏移）既用于把输入线段平移到障碍物附近，
#         也用于决定 V 形凹口内侧，从而把 path 方向和 root side 绑死。
#     variant:
#       - 保留 bar、输入 irregular segment、输出总长度守恒、obstacle 固定不动。
#       - 新增两种专用 obstacle 颜色，整题固定映射：
#         一个颜色 => clockwise 类的镜像绕行，另一个颜色 => counterclockwise 类的镜像绕行。
#       - 输入线段附着侧 attach_side 仍然随机，但输出绕行侧 turn_side 不再由 attach_side 决定，
#         而是只看 obstacle color。
#   concrete_pixel_behavior:
#     - 线段起点仍是 terminal 的一个 4 邻接像素，可能在 bar 轴线任一侧。
#     - 输出仍是 3-phase 单调路径：
#         phase1: diagonal-forward 重复 p1 次，方向由 obstacle color 决定
#         phase2: straight-forward 重复 p2 次
#         phase3: opposite diagonal-forward 重复 p3 次
#     - obstacle 的“内侧支撑列”放在 apex_inner 上，并沿 forward 方向连续延伸 p2+1 个像素；
#       solver 通过这条支撑列从 obstacle 反推出 p1 / p2。
#   behavior_trace:
#     - 原逻辑例子：
#         attach_side=+1 时，旧生成器会构造 (1,+1) 的 approach，obstacle 也被放在对应 V 内侧，
#         所以无论 obstacle 长什么颜色，输出都是同一侧镜像。
#     - 新逻辑例子：
#         attach_side=+1 但 obstacle 颜色映射到 turn_side=-1，
#         则输出从 (0,+1) 先走到 (1,0)、(2,-1)...，绕行方向与输入附着侧脱钩。
#     - 差异：
#         同样的 attach_side 在不同 obstacle 颜色下会产生左右镜像相反的输出。
#   complexity_calibration:
#     - 只使用两种 obstacle 颜色，避免颜色映射档位过多。
#     - attach_side 仍随机，且与 obstacle color 独立采样，避免退化回“看附着侧就够了”。
#     - 每个 triple 仍只含 bar / segment / obstacle 三类物体，视觉复杂度与原题同级。
#   implicit_assumption_audit:
#     - 旧 main 中 `off_r/off_c` 被拿来定义 entry 和 apex 内侧；在变体中这不再能表示绕行方向。
#     - 因此新 main 明确拆成两个量：
#         attach_side = 输入端点位于 terminal 的哪一侧
#         turn_side   = 由 obstacle color 决定的输出绕行方向
#     - obstacle 的可解性不再依赖动态 BFS 找 contour，而依赖“支撑列”可被唯一识别；
#       generator 必须保证 obstacle 含有这条列，且最近可匹配点对应真实 p1。
#   placement_legality:
#     - output-first 仍然保留：先画 output path，再从 apex_inner 生长 obstacle，再生成 input path。
#     - reserved bbox 仍用于分离多个 triple，避免最近障碍物匹配串组。
#     - obstacle 颜色固定在两色集合内，bar / segment / bg 从其余颜色中采样，防止类型歧义。
#   visual_observability:
#     - 只看输入无法知道 turn_side；必须结合颜色观察输入输出对。
#     - attach_side 与 obstacle color 独立，训练样例中会出现“同附着侧、不同颜色、不同输出方向”的对照。
#   cleanliness:
#     - 仍使用单像素细线、单个小 obstacle、1-3 个 triple，图面不会比原题更拥挤。
#   color_inference:
#     - 颜色映射是固定的二元规则，且 obstacle 是唯一使用这两种颜色的对象，
#       因此人类可从少量样例中归纳“哪个障碍物颜色对应哪种绕行镜像”。

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


CW_OBSTACLE_COLOR = 2
CCW_OBSTACLE_COLOR = 8
OBSTACLE_TURN_BY_COLOR = {
    CW_OBSTACLE_COLOR: 1,
    CCW_OBSTACLE_COLOR: -1,
}


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        row = rs[0]
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        col = cs[0]
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


def _bar_frame(bar_comp: list[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    e0, e1 = _bar_endpoints(bar_comp)
    touch0 = _touches_border(e0, h, w)
    terminal = e1 if touch0 else e0
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        fwd = (1, 0) if min(rs) == 0 else (-1, 0)
        side = (0, 1) if fwd == (1, 0) else (0, -1)
    else:
        fwd = (0, 1) if min(cs) == 0 else (0, -1)
        side = (-1, 0) if fwd == (0, 1) else (1, 0)
    return terminal, fwd, side


def _to_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - terminal[0]
    dc = p[1] - terminal[1]
    lf = dr * fwd[0] + dc * fwd[1]
    ls = dr * side[0] + dc * side[1]
    return int(lf), int(ls)


def _from_local(
    loc: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    lf, ls = loc
    return (
        terminal[0] + lf * fwd[0] + ls * side[0],
        terminal[1] + lf * fwd[1] + ls * side[1],
    )


def _build_output_local(seg_len: int, attach_side: int, turn_side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, attach_side)]
    for phase_len, step in zip([p1, p2, p3], [(1, turn_side), (1, 0), (1, -turn_side)]):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
    assert len(coords) == seg_len
    return coords


def _infer_wrap_lengths(
    obstacle_local: set[tuple[int, int]], seg_len: int, attach_side: int, turn_side: int
) -> tuple[int, int, int]:
    candidates = sorted(
        lf
        for lf, ls in obstacle_local
        if lf >= 1 and ls == attach_side + (lf - 1) * turn_side and lf <= seg_len - 1
    )
    assert candidates, "obstacle must expose an apex_inner candidate on the turn-side support diagonal"
    p1 = candidates[0]
    inner_col = attach_side + (p1 - 1) * turn_side
    remain = seg_len - 1 - p1
    p2 = 0
    while p2 < remain and (p1 + p2 + 1, inner_col) in obstacle_local:
        p2 += 1
    p3 = seg_len - 1 - p1 - p2
    assert p3 >= 0
    return p1, p2, p3


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            if col in OBSTACLE_TURN_BY_COLOR:
                continue
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                vertical = len(set(cs)) == 1
                border_ep = e0 if touch0 else e1
                if vertical:
                    if border_ep[0] not in (0, h - 1):
                        continue
                else:
                    if border_ep[1] not in (0, w - 1):
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
            if cand_col in OBSTACLE_TURN_BY_COLOR:
                continue
            for comp in _components(g, cand_col, conn8=True):
                if cand_start in comp:
                    if len(comp) < 2:
                        break
                    deg = _segment_degrees(comp)
                    if deg.get(cand_start, 0) != 1:
                        break
                    path = _segment_main_path(comp, cand_start)
                    L = len(path)
                    key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                    if best_seg is None or key > best_seg[0]:
                        best_seg = (key, cand_col, cand_start, comp)
                    break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        best_obstacle = None
        for obstacle_col in OBSTACLE_TURN_BY_COLOR:
            for comp in _components(g, obstacle_col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best_obstacle is None or key < best_obstacle[0]:
                    best_obstacle = (key, obstacle_col, set(comp))
        assert best_obstacle is not None, "must find an obstacle component"
        _, obstacle_col, obstacle_set = best_obstacle

        _, fwd, side = _bar_frame(bar_comp, h, w)
        attach_local = _to_local(seg_start, terminal, fwd, side)
        assert attach_local[0] == 0 and abs(attach_local[1]) == 1, "segment must attach laterally to terminal"
        attach_side = attach_local[1]
        turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_col]

        obstacle_local = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
        p1, p2, p3 = _infer_wrap_lengths(obstacle_local, L, attach_side, turn_side)
        new_local = _build_output_local(L, attach_side, turn_side, p1, p2, p3)
        new_pts = [_from_local(p, terminal, fwd, side) for p in new_local]

        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "output segment must stay within grid"
        assert new_pts[0] == seg_start, "rerouted segment must keep the same attached start pixel"
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
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


def _grow_obstacle(
    starts: list[tuple[int, int]], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = set(starts)
    while len(obstacle) < max_size:
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, turns: int, attach_side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, attach_side)]
    diag_step = (-1, attach_side)
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
        last_was_diag = step == diag_step
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _build_output_path(
    seg_len: int, attach_side: int, turn_side: int, p1: int, p2: int, p3: int
) -> list[tuple[int, int]]:
    return _build_output_local(seg_len, attach_side, turn_side, p1, p2, p3)


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _make_triple(n: int, turn_side: int, obstacle_color: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    attach_side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(PHASE_MAX, remaining))
    remaining -= p1
    p2 = random.randint(0, min(PHASE_MAX, remaining))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, attach_side, turn_side, p1, p2, p3)
    seg_set = set(output_seg)
    apex = output_seg[p1]
    inner_col = apex[1] - turn_side
    support = [(p1 + k, inner_col) for k in range(p2 + 1)]
    if any(p in seg_set or p in bar_set for p in support):
        return None

    obs_size = random.randint(max(OBS_SIZE_MIN, len(support)), OBS_SIZE_MAX)
    obstacle = _grow_obstacle(support, obs_size, seg_set | bar_set)
    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, attach_side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "obstacle_color": obstacle_color,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "attach_side": attach_side,
        "turn_side": turn_side,
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


def _place_triple(
    triple: dict, edge: str, n: int, reserved: np.ndarray
) -> dict | None:
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
            "obstacle_color": triple["obstacle_color"],
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "attach_side": triple["attach_side"],
            "turn_side": triple["turn_side"],
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

    non_obstacle_colors = [c for c in range(10) if c not in OBSTACLE_TURN_BY_COLOR]
    bg, bar_color, seg_color = random.sample(non_obstacle_colors, 3)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    obstacle_colors = [CW_OBSTACLE_COLOR, CCW_OBSTACLE_COLOR]
    chosen_edges = random.sample(list(EDGE_DATA), triple_count)
    color_schedule = [random.choice(obstacle_colors) for _ in range(triple_count)]
    if triple_count >= 2:
        color_schedule[0] = CW_OBSTACLE_COLOR
        color_schedule[1] = CCW_OBSTACLE_COLOR
        random.shuffle(color_schedule)

    for edge, obstacle_color in zip(chosen_edges, color_schedule):
        turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_color]
        triple = _make_triple(n, turn_side, obstacle_color)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], placed["obstacle_color"])
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], placed["obstacle_color"])
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
