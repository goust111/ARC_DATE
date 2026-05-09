# Combo variant of 88bcf3b4 (index 22)
# combo_spec: D7/0+D13/0
# Name: 多线段保留复制
#
# Design analysis for the fused variant
# ------------------------------------
# a) 具体化与行为追踪
#    原题是 1 bar -> 1 segment -> 1 obstacle，并且输出会把原 segment 擦掉后重画到 obstacle 周围。
#    这里改成 1 bar -> 2/3 segments -> 1 obstacle，而且输出要同时保留原 segment，再额外画出每条
#    segment 的绕行副本。为了让“保留+复制”在同色条件下仍然肉眼可见，副本不直接复用 raw reroute
#    的起始几格；凡是和原 segment 重叠或 8 邻接到原 segment 的那段前缀，都从可见副本里裁掉，
#    再在远端按退出方向补足同样的像素数。这样输出能看到“原线段还在，障碍物旁又出现了一条同色新线”。
#    小例子：若 terminal 左右各挂一条线，原逻辑只会移动其中一条；融合后两条都保留，各自在 obstacle
#    左右两侧生成独立绕行副本。若再加前向挂点，则前向线也会根据自己原始弯折侧，绕到 obstacle 的对应侧。
# b) 复杂度校准
#    为了让样例里 2-4 个 I/O 对就能看出“一个 bar 同时复制多条 rope”，生成器固定只放 1 个 triple，
#    但把 segment 数控制在两个清晰档位：2 条 = left+right，3 条 = left+front+right。每条 segment
#    颜色都不同，bar 和 obstacle 颜色也唯一。这样不会因为多 triple、多同色重复组件而让匹配关系变脏。
# c) 隐式假设审查
#    - 原 solver 的 `best_seg` 默认 terminal 只需挑 1 条线；这里必须收集全部挂接 segments。
#    - 原 solver 会清空原 segment；这里绝对不能再擦除 source。
#    - 原 solver 选 obstacle 时只排除一个 seg 颜色；这里要排除同一 bar 下面全部 seg 颜色。
#    - D13 的多挂点路由允许第三条 front segment，因此不能保留只适用于左右挂点的 apex-inner 断言。
#    - D7 单变体里副本用第四种颜色，所以副本即使挨着 source 也可区分；这里副本与 source 同色，
#      因此 solver/generator 都要保证“可见副本”和 source 分离，不然会退化成一个大同色组件。
# d) 放置合法性
#    生成器先在局部 top-edge 坐标里构造 bar、obstacle 和 2/3 条输入 segments，再直接调用 solver 的
#    move 规划辅助函数得到可见副本。只有当所有副本都不越界、不撞 bar/obstacle/source、彼此不重叠时，
#    该局部 triple 才会被接受；之后再随机裁成正方形并旋转到四个边之一。
# e) 视觉可观察性
#    解题者在输入里会稳定看到“同一根 bar 挂着多条不同颜色的线”，输出里会稳定看到“原线段没消失，
#    obstacle 附近额外出现同色副本”。颜色不依赖数值大小，真正提供结构线索的是挂点位置和 obstacle 周围
#    新出现的副本组件。
# f) 可视化整洁度
#    只保留一个 triple，obstacle 限在 terminal 前方的小区域，副本按 lane 分流到 obstacle 两侧，
#    front 线只在 3-segment 档位出现。这样输出里虽然每种 segment 颜色会出现两个组件，但层次仍清晰。
# g) 变化的可推断性
#    本融合没有额外颜色映射：每条副本沿用原 segment 自己的颜色。为避免“同色新增”无法辨认，样例中
#    每条副本都被强制做成与 source 分离的独立可见组件，因此人类能从少量样例推断出“保留 source，
#    并复制一条新绕行路径”。

from __future__ import annotations

import random
import re

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

    q = [start]
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    head = 0
    while head < len(q):
        cur = q[head]
        head += 1
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


def _has_2x2(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    sset = set(coords)
    for r, c in sset:
        if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= sset:
            return True
    return False


def _bar_forward(bar_comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _bar_side(fwd: tuple[int, int]) -> tuple[int, int]:
    if fwd == (1, 0):
        return (0, 1)
    if fwd == (-1, 0):
        return (0, -1)
    if fwd == (0, 1):
        return (-1, 0)
    return (1, 0)


def _to_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - terminal[0]
    dc = p[1] - terminal[1]
    return dr * fwd[0] + dc * fwd[1], dr * side[0] + dc * side[1]


def _from_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    return (
        terminal[0] + p[0] * fwd[0] + p[1] * side[0],
        terminal[1] + p[0] * fwd[1] + p[1] * side[1],
    )


def _segment_is_valid(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> bool:
    if len(set(seg_pts)) != len(seg_pts):
        return False
    if _has_2x2(seg_pts):
        return False
    deg = _segment_degrees(seg_pts)
    vals = list(deg.values())
    if any(v > 2 for v in vals):
        return False
    if vals.count(1) != 2:
        return False
    if deg.get(start, 0) != 1:
        return False
    path = _segment_main_path(seg_pts, start)
    return len(path) == len(seg_pts)


def _collect_attached_segments(
    g: np.ndarray, bg: int, bar_col: int, terminal: tuple[int, int]
) -> list[dict[str, object]]:
    h, w = g.shape
    found = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    term_neigh = {(nr, nc) for nr, nc in neighbors4(terminal[0], terminal[1]) if 0 <= nr < h and 0 <= nc < w}
    for nr, nc in term_neigh:
        cand_col = int(g[nr, nc])
        if cand_col in (bg, bar_col):
            continue
        for comp in _components(g, cand_col, conn8=True):
            if (nr, nc) not in comp:
                continue
            key = (cand_col, tuple(sorted(comp)))
            if key in seen:
                break
            seen.add(key)
            start = (nr, nc)
            if not _segment_is_valid(comp, start):
                break
            adj_count = sum(1 for p in comp if p in term_neigh)
            if adj_count != 1:
                break
            path = _segment_main_path(comp, start)
            found.append(
                {
                    "color": cand_col,
                    "start": start,
                    "comp": comp,
                    "path": path,
                }
            )
            break
    found.sort(key=lambda seg: (seg["start"][0] - terminal[0], seg["start"][1] - terminal[1], seg["color"]))  # type: ignore[index]
    return found


def _pick_obstacle_for_bar(
    g: np.ndarray,
    colors: list[int],
    bar_col: int,
    segments: list[dict[str, object]],
    terminal: tuple[int, int],
) -> set[tuple[int, int]]:
    excluded = {bar_col} | {int(seg["color"]) for seg in segments}
    best = None
    obstacle_set = None
    for col in colors:
        if col in excluded:
            continue
        for comp in _components(g, col, conn8=True):
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best is None or key < best:
                best = key
                obstacle_set = set(comp)
    assert obstacle_set is not None, f"no obstacle found for bar at {terminal}"
    return obstacle_set


def _validate_wrap_shape(new_pts: list[tuple[int, int]], bar_comp: list[tuple[int, int]]) -> None:
    fwd = _bar_forward(bar_comp)
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
    assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
        f"output segment not monotone wrap: {' '.join(step_classes)}"
    )


def _segment_branch_local(local_path: list[tuple[int, int]]) -> int:
    for _r, c in local_path[1:]:
        if c < 0:
            return -1
        if c > 0:
            return 1
    return 1


def _plan_local_route(
    start_local: tuple[int, int],
    length: int,
    obstacle_locals: set[tuple[int, int]],
    lane: int,
) -> list[tuple[int, int]]:
    rs = [r for r, _ in obstacle_locals]
    r0 = min(rs)
    r1 = max(rs)
    sr, sc = start_local
    lane_sign = -1 if lane < sc else 1 if lane > sc else (-1 if lane < 0 else 1)
    diag_in = abs(lane - sc)
    straight_in = r0 - sr - diag_in
    assert straight_in >= 0, f"obstacle too close for start {start_local} and lane {lane}"

    route = [start_local]
    cur = start_local
    for _ in range(diag_in):
        cur = (cur[0] + 1, cur[1] + lane_sign)
        route.append(cur)
    for _ in range(straight_in):
        cur = (cur[0] + 1, cur[1])
        route.append(cur)
    for _ in range(max(0, r1 - r0)):
        cur = (cur[0] + 1, cur[1])
        route.append(cur)

    assert len(route) <= length, f"base route too long for start {start_local}"
    exit_dc = -lane_sign
    while len(route) < length:
        cur = (cur[0] + 1, cur[1] + exit_dc)
        route.append(cur)
    return route


def _reroute_segment(
    g: np.ndarray,
    terminal: tuple[int, int],
    bar_comp: list[tuple[int, int]],
    seg: dict[str, object],
    obstacle_set: set[tuple[int, int]],
    lane: int,
) -> list[tuple[int, int]]:
    seg_start = seg["start"]
    seg_path = seg["path"]
    fwd = _bar_forward(bar_comp)
    side = _bar_side(fwd)
    local_path = [_to_local(p, terminal, fwd, side) for p in seg_path]
    local_start = _to_local(seg_start, terminal, fwd, side)
    local_obstacle = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
    local_route = _plan_local_route(local_start, len(local_path), local_obstacle, lane)
    new_pts = [_from_local(p, terminal, fwd, side) for p in local_route]
    h, w = g.shape
    bar_set = set(bar_comp)
    assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), f"reroute exits grid for segment at {seg_start}"
    assert not any(p in bar_set for p in new_pts), f"reroute crosses bar for segment at {seg_start}"
    assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
    _validate_wrap_shape(new_pts, bar_comp)
    return new_pts


def _bar_candidates(g: np.ndarray, bg: int, colors: list[int]) -> list[tuple[int, list[tuple[int, int]], tuple[int, int]]]:
    h, w = g.shape
    bars = []
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
            bars.append((col, comp, terminal_ep))
    bars.sort(key=lambda x: (-len(x[1]), x[0], x[2]))
    return bars


def _halo8(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> set[tuple[int, int]]:
    halo = set(coords)
    for r, c in list(halo):
        for nr, nc in neighbors8(r, c):
            halo.add((nr, nc))
    return halo


def _derive_visible_copy(
    g: np.ndarray,
    raw_route: list[tuple[int, int]],
    source_pts: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    h, w = g.shape
    source_halo = _halo8(source_pts)
    unsafe = [i for i, p in enumerate(raw_route) if p in source_halo]
    assert unsafe, "raw route must start from the source-side attachment"
    assert unsafe == list(range(len(unsafe))), "source-touching cells must form a prefix"
    prefix_len = len(unsafe)
    assert prefix_len < len(raw_route), "visible copy cannot be completely hidden by the source"

    visible = list(raw_route[prefix_len:])
    dr_ext = raw_route[-1][0] - raw_route[-2][0]
    dc_ext = raw_route[-1][1] - raw_route[-2][1]
    cur = raw_route[-1]
    target_len = len(source_pts)
    while len(visible) < target_len:
        cur = (cur[0] + dr_ext, cur[1] + dc_ext)
        if not (0 <= cur[0] < h and 0 <= cur[1] < w):
            break
        if int(g[cur]) != get_bg(g):
            break
        if cur in source_halo:
            break
        visible.append(cur)

    assert len(visible) == target_len, "visible copy must preserve the source pixel count"
    assert len(set(visible)) == len(visible), "visible copy must stay simple"
    assert not any(p in source_halo for p in visible), "visible copy must be visibly separate from the source"
    return visible, prefix_len


def _solve_bar_copy_moves(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    moves: list[dict[str, object]] = []
    for bar_col, bar_comp, terminal in _bar_candidates(g, bg, colors):
        segments = _collect_attached_segments(g, bg, bar_col, terminal)
        assert 2 <= len(segments) <= 3, f"bar at {terminal} must have 2-3 attached segments"
        obstacle_set = _pick_obstacle_for_bar(g, colors, bar_col, segments, terminal)

        fwd = _bar_forward(bar_comp)
        side = _bar_side(fwd)
        local_obstacle = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
        c0 = min(c for _r, c in local_obstacle)
        c1 = max(c for _r, c in local_obstacle)

        front_branch = 0
        for seg in segments:
            local_start = _to_local(seg["start"], terminal, fwd, side)
            if local_start == (1, 0):
                local_path = [_to_local(p, terminal, fwd, side) for p in seg["path"]]
                front_branch = _segment_branch_local(local_path)
                break

        occupied_new: set[tuple[int, int]] = set()
        for seg in segments:
            local_start = _to_local(seg["start"], terminal, fwd, side)
            if local_start == (0, -1):
                lane = c0 - 2 if front_branch < 0 else c0 - 1
            elif local_start == (0, 1):
                lane = c1 + 2 if front_branch > 0 else c1 + 1
            else:
                assert local_start == (1, 0), f"unexpected attachment slot {local_start}"
                lane = c0 - 1 if front_branch < 0 else c1 + 1

            raw_route = _reroute_segment(g, terminal, bar_comp, seg, obstacle_set, lane)
            copy_pts, prefix_len = _derive_visible_copy(g, raw_route, seg["comp"])
            overlap = occupied_new & set(copy_pts)
            assert not overlap, f"copied routes from one bar overlap: {sorted(overlap)}"
            occupied_new |= set(copy_pts)
            moves.append(
                {
                    "terminal": terminal,
                    "bar_comp": bar_comp,
                    "seg_color": int(seg["color"]),
                    "seg_comp": list(seg["comp"]),
                    "raw_route": raw_route,
                    "copy_pts": copy_pts,
                    "prefix_len": prefix_len,
                }
            )
    return moves


def _bar_segment_counts(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    return [len(_collect_attached_segments(g, bg, bar_col, terminal)) for bar_col, _bar_comp, terminal in _bar_candidates(g, bg, colors)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve every original rope-like segment and also add a same-color wrap copy around the shared obstacle.

    Input structure:
    - background
    - one border-attached straight bar
    - one nearby obstacle sprite
    - two or three different-colored irregular segments attached to the same bar terminal

    Rule:
    - Each attached segment is treated independently, but all of them wrap around the same obstacle.
    - The original segments stay exactly where they are.
    - For each original segment, draw an additional same-color routed copy near the obstacle, keeping the copied pixel count equal to the source length.

    Key insight:
    - The task is no longer "move one rope"; it is "for one bar, keep all ropes and clone each one into its own obstacle-hugging route", with the visible copy trimmed away from the source so the same-color duplication can still be seen.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moves = _solve_bar_copy_moves(g)
    for move in moves:
        seg_col = int(move["seg_color"])
        for r, c in move["copy_pts"]:
            assert int(out[r, c]) == get_bg(g), f"copy must paint fresh background at {(r, c)}"
            out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
SEG_LEN_MIN, SEG_LEN_MAX = 5, 8
OBS_SIZE_MIN, OBS_SIZE_MAX = 2, 6
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 4
PLACEMENT_PAD = 2
MAX_RETRIES = 500
TRIPLE_LOCAL_RETRIES = 220
LOCAL_CANVAS = 27

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

ATTACH_LAYOUTS = {
    2: [("left", (0, -1)), ("right", (0, 1))],
    3: [("left", (0, -1)), ("front", (1, 0)), ("right", (0, 1))],
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


def _build_front_input_path(seg_len: int, turns: int, branch_side: int) -> list[tuple[int, int]]:
    coords = [(1, 0), (2, 0), (2, branch_side), (1, 2 * branch_side)]
    remaining = seg_len - 4
    diag_step = (-1, branch_side)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = max(0, turns - 1)
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < 0.6
            elif last_was_diag:
                is_diag = random.random() < 0.45
            else:
                is_diag = random.random() < 0.6
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


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _grow_obstacle(
    start: tuple[int, int],
    max_size: int,
    forbidden: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden and (nr, nc) in allowed
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _make_local_obstacle(seg_count: int, bar: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    bar_set = set(bar)
    obs_dist = random.randint(max(4 if seg_count == 3 else 3, OBS_MD_MIN), OBS_MD_MAX)
    start = (obs_dist, random.choice([-1, 0, 1]))
    side_span = 2 if seg_count == 2 else 1
    allowed = {(r, c) for r in range(2, obs_dist + 4) for c in range(-side_span, side_span + 1)}
    forbidden = bar_set | {(0, -1), (0, 1), (1, 0)}
    max_size = OBS_SIZE_MAX if seg_count == 2 else min(4, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(start, random.randint(OBS_SIZE_MIN, max_size), forbidden, allowed)
    rs = [r for r, _ in obstacle]
    cs = [c for _, c in obstacle]
    if max(abs(c) for c in cs) > side_span:
        return None
    max_h = 4 if seg_count == 2 else 2
    if max(rs) - min(rs) > max_h or max(cs) - min(cs) > 2:
        return None
    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None
    return obstacle


def _paint_local_input(
    bar_len: int,
    bar: list[tuple[int, int]],
    obstacle: list[tuple[int, int]],
    input_segments: list[dict[str, object]],
) -> tuple[np.ndarray, tuple[int, int], list[int]]:
    n = LOCAL_CANVAS
    pos = n // 2
    bg = 0
    bar_color = 1
    obstacle_color = 2
    seg_colors = list(range(3, 3 + len(input_segments)))
    inp = np.full((n, n), bg, dtype=int)
    paint_coords(inp, _transform(bar, "top", n, bar_len, pos), bar_color)
    paint_coords(inp, _transform(obstacle, "top", n, bar_len, pos), obstacle_color)
    for seg, seg_col in zip(input_segments, seg_colors):
        paint_coords(inp, _transform(seg["coords"], "top", n, bar_len, pos), seg_col)
    return inp, _terminal_for("top", n, bar_len, pos), seg_colors


def _make_triple(seg_count: int) -> dict | None:
    for _ in range(TRIPLE_LOCAL_RETRIES):
        bar_len = random.randint(BAR_LEN_MIN, BAR_LEN_MAX)
        bar = [(-k, 0) for k in range(bar_len)]
        obstacle = _make_local_obstacle(seg_count, bar)
        if obstacle is None:
            continue

        r0 = min(r for r, _ in obstacle)
        r1 = max(r for r, _ in obstacle)
        along = r1 - r0
        front_branch = random.choice([-1, 1]) if seg_count == 3 else 0
        min_len = {
            "left": 1 + r0 + along,
            "right": 1 + r0 + along,
            "front": r0 + along,
        }

        input_segments = []
        occupied = set(bar) | set(obstacle)
        ok = True
        for name, start in ATTACH_LAYOUTS[seg_count]:
            need = max(SEG_LEN_MIN, min_len[name])
            if need > SEG_LEN_MAX:
                ok = False
                break
            seg_len = random.randint(need, SEG_LEN_MAX)
            turns = random.randint(INPUT_TURN_MIN, min(INPUT_TURN_MAX, seg_len - 2))
            if name == "front":
                coords = _build_front_input_path(seg_len, turns, front_branch)
            else:
                coords = _build_input_path(seg_len, turns, -1 if name == "left" else 1)
            if coords[0] != start:
                ok = False
                break
            if not _segment_is_valid(coords, start):
                ok = False
                break
            if any(p in occupied for p in coords):
                ok = False
                break
            if _is_4adj(coords, obstacle):
                ok = False
                break
            input_segments.append({"name": name, "start": start, "coords": coords})
            occupied |= set(coords)
        if not ok:
            continue

        inp, terminal, seg_colors = _paint_local_input(bar_len, bar, obstacle, input_segments)
        try:
            moves = _solve_bar_copy_moves(inp)
        except Exception:
            continue
        if len(moves) != seg_count or _bar_segment_counts(inp) != [seg_count]:
            continue

        move_by_color = {int(move["seg_color"]): move for move in moves}
        copy_segments = []
        union = list(bar) + list(obstacle)
        for seg, seg_col in zip(input_segments, seg_colors):
            if seg_col not in move_by_color:
                ok = False
                break
            move = move_by_color[seg_col]
            copy_local = [(r - terminal[0], c - terminal[1]) for r, c in move["copy_pts"]]
            if len(copy_local) != len(seg["coords"]):
                ok = False
                break
            if _has_2x2(copy_local):
                ok = False
                break
            if set(copy_local) & (set(bar) | set(obstacle)):
                ok = False
                break
            if set(copy_local) & set(seg["coords"]):
                ok = False
                break
            copy_segments.append(
                {
                    "coords": copy_local,
                    "prefix_len": int(move["prefix_len"]),
                    "raw_route": [(r - terminal[0], c - terminal[1]) for r, c in move["raw_route"]],
                }
            )
            union.extend(seg["coords"])
            union.extend(copy_local)
        if not ok:
            continue
        if any(a["prefix_len"] <= 0 for a in copy_segments):
            continue
        return {
            "bar_len": bar_len,
            "bar": bar,
            "obstacle": obstacle,
            "input_segments": input_segments,
            "copy_segments": copy_segments,
            "union": union,
        }
    return None


def _gen_attempt(forced_seg_count: int | None = None) -> tuple[np.ndarray, dict] | None:
    seg_count = forced_seg_count if forced_seg_count is not None else (3 if random.random() < 0.3 else 2)
    triple = _make_triple(seg_count)
    if triple is None:
        return None

    bg = random.randrange(10)
    palette = random.sample([c for c in range(10) if c != bg], seg_count + 2)
    bar_color = palette[0]
    obstacle_color = palette[1]
    seg_colors = palette[2:]

    full_in = np.full((LOCAL_CANVAS, LOCAL_CANVAS), bg, dtype=int)
    full_out = np.full((LOCAL_CANVAS, LOCAL_CANVAS), bg, dtype=int)
    pos = LOCAL_CANVAS // 2
    bar_len = triple["bar_len"]

    bar_abs = _transform(triple["bar"], "top", LOCAL_CANVAS, bar_len, pos)
    obstacle_abs = _transform(triple["obstacle"], "top", LOCAL_CANVAS, bar_len, pos)
    paint_coords(full_in, bar_abs, bar_color)
    paint_coords(full_in, obstacle_abs, obstacle_color)
    paint_coords(full_out, bar_abs, bar_color)
    paint_coords(full_out, obstacle_abs, obstacle_color)
    for seg, copy_seg, seg_col in zip(triple["input_segments"], triple["copy_segments"], seg_colors):
        paint_coords(full_in, _transform(seg["coords"], "top", LOCAL_CANVAS, bar_len, pos), seg_col)
        paint_coords(full_out, _transform(seg["coords"], "top", LOCAL_CANVAS, bar_len, pos), seg_col)
        paint_coords(full_out, _transform(copy_seg["coords"], "top", LOCAL_CANVAS, bar_len, pos), seg_col)

    union_abs = _transform(triple["union"], "top", LOCAL_CANVAS, bar_len, pos)
    rs = [r for r, _ in union_abs]
    cs = [c for _, c in union_abs]
    r1 = max(rs)
    c0 = min(cs)
    c1 = max(cs)
    n_min = max(GRID_SIZE_MIN, r1 + 1, c1 - c0 + 1)
    n_max = min(GRID_SIZE_MAX, LOCAL_CANVAS)
    if n_min > n_max:
        return None
    n = random.randint(n_min, n_max)
    left_lo = max(0, c1 - n + 1)
    left_hi = min(c0, LOCAL_CANVAS - n)
    if left_lo > left_hi:
        return None
    left = random.randint(left_lo, left_hi)
    inp = full_in[:n, left : left + n].copy()
    out = full_out[:n, left : left + n].copy()

    k = random.randrange(4)
    inp = np.rot90(inp, k).copy()
    out = np.rot90(out, k).copy()

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    counts = _bar_segment_counts(inp)
    if counts != [seg_count]:
        return None

    meta = {
        "seg_count": seg_count,
        "prefix_lens": [int(seg["prefix_len"]) for seg in triple["copy_segments"]],
        "copy_lengths": [len(seg["coords"]) for seg in triple["copy_segments"]],
    }
    return inp, meta


def generate_input() -> np.ndarray:
    target_seg_count = 3 if random.random() < 0.25 else 2
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt(target_seg_count)
        if attempt is not None:
            grid, _meta = attempt
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")


def _sample_combo_metrics(samples: int = 40) -> dict:
    stats = {
        "samples": 0,
        "seg_counts": [],
        "prefix_lens": [],
        "copy_lengths": [],
    }
    for _ in range(samples):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        _inp, meta = attempt
        stats["samples"] += 1
        stats["seg_counts"].append(meta["seg_count"])
        stats["prefix_lens"].extend(meta["prefix_lens"])
        stats["copy_lengths"].extend(meta["copy_lengths"])
    return stats
