# Variant of 88bcf3b4
# Dimension: D8 — 匹配条件（障碍物选择） (match_cond)
# Change: 选择离 bar 末端曼哈顿距离最近的障碍物精灵作为绕行目标；当存在多个 triple 时，每个 bar 选择自己最近的障... → 选择最远障碍物
#
# Design analysis
# ---
# variant_spec:
#   core_change:
#     old: 每个 bar 选择离自己 terminal 最近的 obstacle component。
#     new: 每个 bar 改为选择离自己 terminal 最远的 obstacle component。
#     preserved: 一旦目标 obstacle 选定，后续“靠近目标、贴着 skin 绕行、保持线段像素数”的机制保持原样。
#   concrete_pixel_behavior:
#     input:
#       - 仍然是 bar + 输入折线 segment + 若干 obstacle sprite。
#       - 但为了让“最远匹配”可见，单个样本中必须至少有 2 个 triple。
#       - 每个 bar 附近保留一个更近的 obstacle 作为干扰项，同时另一个 obstacle 稍远但仍在 segment 长度可到达范围内。
#     output:
#       - segment 不再绕自己最近的 obstacle，而是跨过去绕更远的那个 obstacle。
#       - 因为目标更远，输出路径通常更长、更跨区域；这正是 D8 的可见效果。
#   behavior_trace:
#     small_example:
#       - top bar 的 own obstacle 在 terminal 下方偏左，距离 3。
#       - left bar 的 obstacle 距离 top bar 的 terminal 为 5-7。
#       - 原规则会让 top bar 的 segment 向下后很快绕 own obstacle。
#       - 新规则会忽略这个更近障碍物，继续朝 left bar 的 obstacle 前进并在其外侧绕行。
#       - 差异集中发生在 main() 的 obstacle 选择处；路径搜索与 skin 绕行逻辑沿用原版。
#   complexity_calibration:
#     - 若只生成 1 个 triple，则“最近/最远”完全退化为同一行为，因此新生成器固定生成 2 个 triple。
#     - 若把目标障碍物放得太远，segment 会在到达前就被长度截断，肉眼看不出“绕远障碍物”；因此将跨 triple 的目标距离收紧到约 5-7。
#     - 若 triple 太多，谁是最远会变得难以肉眼判断，且路径更易互相干扰；因此维持 2 个 triple，重复同一规律即可。
#   implicit_assumptions_review:
#     - 原 solver 的 `best = min((d, mr, mc))` 隐含了“最近匹配”；这里必须改成“距离最大优先，平局仍取左上”。
#     - 原 generator 末尾 `own < other` 明确强制“自己的 obstacle 最近”；这里必须删掉并反过来构造 `own < other <= reachable_far`。
#     - `pick_p_near()` 仍假设：目标 obstacle 外侧存在一个和输入 segment 偏移兼容的可落点。新生成器通过小 obstacle、较空旷布局和 oracle 校验来保证。
#     - segment 颜色在多个 triple 间共享，因此不同输入 segment 不能 8 连通；否则 solver 会把它们当成同一组件。生成器显式避免这一点。
#     - obstacle 颜色也共享，因此不同 obstacle 不能 8 连通；生成器同样显式避免。
#   placement_legality:
#     - 两个 triple 放在相邻边上，使“自己的 obstacle 更近、别人的 obstacle 更远但仍可达”更容易同时满足。
#     - 采用较长的 bar（6-7）与 segment（6-7），保证输入 segment 能完整位于 bar 和边界之间，也保证输出能实际碰到远障碍物。
#     - 生成后统一用 `main_far` 做 oracle；任何越界、重叠、skin 不可走、wrap 非单调的候选都会被丢弃。
#   visual_observability:
#     - 每个样本固定 2 个 bar 和 2 个 obstacle，且两个 bar 都绕“对方的 obstacle”，这样 2-4 个样例足以让人看出“不是绕最近那个”。
#     - 生成器还会过滤 `main_far(input) == main_near(input)` 的样本，避免退化成原题。
#   cleanliness:
#     - obstacle 保持小且稀疏，segment 输入态贴边、输出态进入内区，颜色层次简单，避免大面积互相覆盖。
#   new_main_assumptions_vs_generator:
#     - 新 main() 假设每个 bar 至少存在一个比 own obstacle 更远且可到达的 obstacle。generate_input() 用双 obstacle 布局和 oracle 筛选保证。
#     - 新 main() 仍假设输出 route 不与非背景对象冲突。生成器只接受能通过 `_solve(..., choose_farthest=True)` 的输入。

import random
from collections import deque

import numpy as np

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


def _bfs_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _bfs_path_with_diag_filter(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}

    def can_step(r: int, c: int, nr: int, nc: int) -> bool:
        dr = nr - r
        dc = nc - c
        if dr == 0 or dc == 0:
            return True
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if 0 <= mid1[0] < h and 0 <= mid1[1] < w and bool(walkable[mid1]):
            return False
        if 0 <= mid2[0] < h and 0 <= mid2[1] < w and bool(walkable[mid2]):
            return False
        return True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            if not can_step(r, c, nr, nc):
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    assert goal in prev, "BFS-with-diag-filter must reach goal"
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _greedy_manhattan_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape

    def md(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    cur = start
    path = [cur]
    seen = {cur}
    while cur != goal:
        cur_md = md(cur)
        best = None
        for nr, nc in neighbors8(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            nmd = md(nxt)
            if nmd >= cur_md:
                continue
            is_diag = 1 if (nr != cur[0] and nc != cur[1]) else 0
            key = (nmd, is_diag, nr, nc)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        _, _, nr, nc = best
        cur = (nr, nc)
        path.append(cur)
        seen.add(cur)
    return path


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


def _select_obstacle(
    g: np.ndarray,
    colors: list[int],
    terminal: tuple[int, int],
    bar_col: int,
    seg_col: int,
    choose_farthest: bool,
) -> tuple[set[tuple[int, int]], int]:
    obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
    assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"

    best = None
    obstacle_set = None
    best_dist = None
    for col in obstacle_candidates:
        for comp in _components(g, col, conn8=True):
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            if choose_farthest:
                key = (d, -mr, -mc)
                better = best is None or key > best
            else:
                key = (d, mr, mc)
                better = best is None or key < best
            if better:
                best = key
                best_dist = d
                obstacle_set = set(comp)
    assert obstacle_set is not None and best_dist is not None, "must find at least one obstacle component"
    return obstacle_set, best_dist


def _solve(grid: np.ndarray, *, choose_farthest: bool, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()
    details = []

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

        obstacle_set, chosen_dist = _select_obstacle(
            g, colors, terminal, bar_col, seg_col, choose_farthest=choose_farthest
        )

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

        walkable = (g == bg) | (g == seg_col)
        walkable[terminal] = True

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if int(g[er, ec]) not in (bg, seg_col):
                    continue
                d = abs(pr - terminal[0]) + abs(pc - terminal[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return (pr, pc)

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"
        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted segment path must stay within grid"
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])

        def skin_ok(p: tuple[int, int]) -> bool:
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)
        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]

        assert len(boundary_path) >= 2, "boundary path must have at least 2 cells to derive extension direction"
        dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
        dc_ext = boundary_path[-1][1] - boundary_path[-2][1]

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:L]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

        rs_bar = [r for r, _ in bar_comp]
        cs_bar = [c for _, c in bar_comp]
        if len(set(cs_bar)) == 1:
            fwd = (1, 0) if min(rs_bar) == 0 else (-1, 0)
        else:
            fwd = (0, 1) if min(cs_bar) == 0 else (0, -1)
        step_classes = []
        for j in range(len(new_pts) - 1):
            dr = new_pts[j + 1][0] - new_pts[j][0]
            dc = new_pts[j + 1][1] - new_pts[j][1]
            fwd_comp = dr * fwd[0] + dc * fwd[1]
            perp_comp = dr * fwd[1] + dc * (-fwd[0])
            if abs(dr) + abs(dc) == 1:
                step_classes.append("F" if fwd_comp > 0 else "X")
            elif fwd_comp > 0 and perp_comp > 0:
                step_classes.append("FR")
            elif fwd_comp > 0 and perp_comp < 0:
                step_classes.append("FL")
            else:
                step_classes.append("X")
        s = "".join(step_classes)
        import re

        assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
            f"output segment not monotone wrap: {' '.join(step_classes)}"
        )

        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = new_pts[p1_end]
            perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
            inner = (-perp_dir[0], -perp_dir[1])
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"

        moves.append((seg_comp, seg_col, new_pts))
        details.append(
            {
                "terminal": terminal,
                "seg_start": seg_start,
                "seg_comp": list(seg_comp),
                "seg_color": seg_col,
                "chosen_obstacle": set(obstacle_set),
                "chosen_dist": chosen_dist,
                "new_pts": list(new_pts),
            }
        )

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col

    if return_details:
        return out, details
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, choose_farthest=True, return_details=False)


GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 22
BAR_LEN = 7
SEG_LEN = 7
P1 = 3
P2_CHOICES = [0, 1]
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 4
MAX_RETRIES = 500
OWN_OBS_MAX = 4
OTHER_OBS_MD = 5

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

CORNER_LAYOUTS = [
    {
        "triples": [
            {"edge": "top", "side": -1},
            {"edge": "left", "side": 1},
        ]
    },
    {
        "triples": [
            {"edge": "top", "side": 1},
            {"edge": "right", "side": -1},
        ]
    },
    {
        "triples": [
            {"edge": "bottom", "side": 1},
            {"edge": "left", "side": -1},
        ]
    },
    {
        "triples": [
            {"edge": "bottom", "side": -1},
            {"edge": "right", "side": 1},
        ]
    },
]


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


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side)]
    diag_step = (-1, side)
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
                is_diag = random.random() < 0.5
            elif last_was_diag:
                is_diag = random.random() < 0.35
            else:
                is_diag = random.random() < 0.5
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


def _build_output_path(seg_len: int, side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    phase_steps = [(1, side), (1, 0), (1, -side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    assert len(coords) == seg_len
    return coords


def _valid_positions(edge: str, n: int, bar_len: int, union_local: list[tuple[int, int]]) -> list[int]:
    valid = []
    for pos in range(n):
        pts = _transform(union_local, edge, n, bar_len, pos)
        if all(0 <= r < n and 0 <= c < n for r, c in pts):
            valid.append(pos)
    return valid


def _md(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _min_md_point_to_coords(point: tuple[int, int], coords: list[tuple[int, int]]) -> int:
    return min(_md(point, q) for q in coords)


def _min_md_sets(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> int:
    return min(_md(a, b) for a in coords_a for b in coords_b)


def _touch_or_overlap8(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    aset = set(coords_a)
    for r, c in coords_b:
        if (r, c) in aset:
            return True
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in aset:
                    return True
    return False


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _place_layout(n: int):
    layout = random.choice(CORNER_LAYOUTS)
    triples = []
    for spec in layout["triples"]:
        turns = random.randint(INPUT_TURN_MIN, min(INPUT_TURN_MAX, SEG_LEN - 2))
        p2 = random.choice(P2_CHOICES)
        p3 = SEG_LEN - 1 - P1 - p2
        output_seg = _build_output_path(SEG_LEN, spec["side"], P1, p2, p3)
        apex = output_seg[P1]
        obstacle_anchor = (apex[0], apex[1] - spec["side"])
        obstacle_extra = (obstacle_anchor[0], obstacle_anchor[1] - spec["side"])
        triples.append(
            {
                "edge": spec["edge"],
                "side": spec["side"],
                "bar_len": BAR_LEN,
                "bar": [(-k, 0) for k in range(BAR_LEN)],
                "input_seg": _build_input_path(SEG_LEN, turns, spec["side"]),
                "target_obstacle_shape": [obstacle_anchor, obstacle_extra],
            }
        )
    valid_pos_lists = [
        _valid_positions(triple["edge"], n, triple["bar_len"], triple["bar"] + triple["input_seg"])
        for triple in triples
    ]
    if any(not pos_list for pos_list in valid_pos_lists):
        return None

    pos_pairs = [(p0, p1) for p0 in valid_pos_lists[0] for p1 in valid_pos_lists[1]]
    random.shuffle(pos_pairs)
    for pos0, pos1 in pos_pairs:
        placed = []
        for triple, pos in zip(triples, [pos0, pos1]):
            placed.append(
                {
                    "edge": triple["edge"],
                    "bar_len": triple["bar_len"],
                    "bar": _transform(triple["bar"], triple["edge"], n, triple["bar_len"], pos),
                    "input_seg": _transform(triple["input_seg"], triple["edge"], n, triple["bar_len"], pos),
                    "terminal": _transform([(0, 0)], triple["edge"], n, triple["bar_len"], pos)[0],
                    "target_obstacle_shape": _transform(
                        triple["target_obstacle_shape"], triple["edge"], n, triple["bar_len"], pos
                    ),
                }
            )
        placed[0]["obstacle"] = list(placed[1]["target_obstacle_shape"])
        placed[1]["obstacle"] = list(placed[0]["target_obstacle_shape"])

        all_cells = placed[0]["bar"] + placed[0]["input_seg"] + placed[0]["obstacle"]
        other_cells = placed[1]["bar"] + placed[1]["input_seg"] + placed[1]["obstacle"]
        if set(all_cells) & set(other_cells):
            continue
        if _touch_or_overlap8(placed[0]["obstacle"], placed[1]["obstacle"]):
            continue
        if _touch_or_overlap8(placed[0]["input_seg"], placed[1]["input_seg"]):
            continue
        if _is_4adj(placed[0]["input_seg"], placed[0]["obstacle"]):
            continue
        if _is_4adj(placed[1]["input_seg"], placed[1]["obstacle"]):
            continue
        if _is_4adj(placed[0]["input_seg"], placed[1]["obstacle"]):
            continue
        if _is_4adj(placed[1]["input_seg"], placed[0]["obstacle"]):
            continue

        own0 = _min_md_point_to_coords(placed[0]["terminal"], placed[0]["obstacle"])
        own1 = _min_md_point_to_coords(placed[1]["terminal"], placed[1]["obstacle"])
        other0 = _min_md_point_to_coords(placed[0]["terminal"], placed[1]["obstacle"])
        other1 = _min_md_point_to_coords(placed[1]["terminal"], placed[0]["obstacle"])
        if own0 > OWN_OBS_MAX or own1 > OWN_OBS_MAX:
            continue
        if other0 != OTHER_OBS_MD or other1 != OTHER_OBS_MD:
            continue
        if not (own0 < other0 and own1 < other1):
            continue

        return placed
    return None


def _details_ok(details: list[dict], placed: list[dict]) -> bool:
    by_terminal = {item["terminal"]: item for item in placed}
    for detail in details:
        terminal = detail["terminal"]
        if terminal not in by_terminal:
            return False
        own_obstacle = set(by_terminal[terminal]["obstacle"])
        if detail["chosen_obstacle"] == own_obstacle:
            return False
        if _min_md_sets(detail["new_pts"], list(detail["chosen_obstacle"])) != 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    placed = _place_layout(n)
    if placed is None:
        return None

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    for item in placed:
        paint_coords(inp, item["bar"], bar_color)
        paint_coords(inp, item["input_seg"], seg_color)
        paint_coords(inp, item["obstacle"], obstacle_color)

    try:
        _, details = _solve(inp, choose_farthest=True, return_details=True)
    except Exception:
        return None

    if len(details) != 2:
        return None
    if not _details_ok(details, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
