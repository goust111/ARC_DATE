# Combo variant of 88bcf3b4 (index 30)
# combo_spec: D7/1+D18/1
# Name: 图例指定颜色转移涂色
#
# Design analysis
# - 保留原题核心：输入含 1-2 组 `bar + irregular segment + obstacle`，segment 从 bar 内端脱离，
#   绕最近的指定障碍物外皮后重画，且原 segment 消失、像素数保持不变。
# - D7/1 的核心是“输出线段改色”，但本融合不再直接取 obstacle 自身颜色，而是结合 D18/1：
#   角落两格图例的第一格给出“哪种颜色的精灵才算 obstacle”，第二格给出“输出线段的颜色”。
# - 为了让第一格图例真正必要，generator 每个 triple 旁都放一个更近但颜色不匹配的 decoy sprite；
#   若忽略图例，最近精灵会选错。人类可以从 2-4 个样例中观察到：只有和图例首格同色的精灵会被绕行。
# - 为了让第二格图例可归纳，输入中该颜色只出现在 legend，不出现在主体场景；输出里所有新路径统一刷成该色。
# - legend 固定为角落的 2 格条（横或竖，四角任选），顺序始终是“从角落向内”：第一格 obstacle 色，第二格 output 色。

from __future__ import annotations

import random
import re
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 5
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 7
OBS_MD_MIN, OBS_MD_MAX = 3, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
PLACEMENT_PAD = 1
PER_EDGE_RETRIES = 50
MAX_RETRIES = 1200

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    conn = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=conn)]


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
    if goal not in prev:
        return None
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


def _legend_specs(h: int, w: int) -> list[dict[str, object]]:
    return [
        {"cells": [(0, 0), (0, 1)], "box": (0, 0, min(1, h - 1), min(1, w - 1))},
        {"cells": [(0, 0), (1, 0)], "box": (0, 0, min(1, h - 1), min(1, w - 1))},
        {"cells": [(0, w - 1), (0, w - 2)], "box": (0, max(0, w - 2), min(1, h - 1), w - 1)},
        {"cells": [(0, w - 1), (1, w - 1)], "box": (0, max(0, w - 2), min(1, h - 1), w - 1)},
        {"cells": [(h - 1, 0), (h - 1, 1)], "box": (max(0, h - 2), 0, h - 1, min(1, w - 1))},
        {"cells": [(h - 1, 0), (h - 2, 0)], "box": (max(0, h - 2), 0, h - 1, min(1, w - 1))},
        {"cells": [(h - 1, w - 1), (h - 1, w - 2)], "box": (max(0, h - 2), max(0, w - 2), h - 1, w - 1)},
        {"cells": [(h - 1, w - 1), (h - 2, w - 1)], "box": (max(0, h - 2), max(0, w - 2), h - 1, w - 1)},
    ]


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[tuple[int, int]]]:
    h, w = g.shape
    candidates = []
    for spec in _legend_specs(h, w):
        cells = spec["cells"]
        box = spec["box"]
        colors = [int(g[r, c]) for r, c in cells]
        if any(col == bg for col in colors):
            continue
        r0, c0, r1, c1 = box
        non_bg = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(g[r, c]) != bg]
        if sorted(non_bg) != sorted(cells):
            continue
        candidates.append((colors[0], colors[1], cells))
    assert len(candidates) == 1, f"expected unique corner legend, got {candidates}"
    return candidates[0]


def _solve_moves(
    grid: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    obstacle_color, output_color, legend_cells = _find_legend(g, bg)

    scene = g.copy()
    for r, c in legend_cells:
        scene[r, c] = bg

    colors = sorted(set(int(v) for v in np.unique(scene)) - {bg})
    obstacle_components = [set(comp) for comp in _components(scene, obstacle_color, conn8=True)]
    assert obstacle_components, "legend obstacle color must appear in active scene"

    def bar_candidates():
        for col in colors:
            if col == obstacle_color:
                continue
            for comp in _components(scene, col, conn8=False):
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
                    0 <= nr < h and 0 <= nc < w and int(scene[nr, nc]) != bg and int(scene[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(scene[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            if cand_col == obstacle_color:
                continue
            for comp in _components(scene, cand_col, conn8=True):
                if cand_start not in comp or len(comp) < 2:
                    continue
                deg = _segment_degrees(comp)
                if deg.get(cand_start, 0) != 1:
                    continue
                path = _segment_main_path(comp, cand_start)
                L = len(path)
                key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        best_obstacle_key = None
        obstacle_set = None
        for comp in obstacle_components:
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best_obstacle_key is None or key < best_obstacle_key:
                best_obstacle_key = key
                obstacle_set = comp
        assert obstacle_set is not None

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]
        seg_set = set(seg_comp)

        walkable = (scene == bg) | (scene == seg_col)
        walkable[terminal] = True

        def current_seg_walkable(p: tuple[int, int]) -> bool:
            return int(scene[p]) == bg or p in seg_set

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if not current_seg_walkable((er, ec)):
                    continue
                d = abs(pr - terminal[0]) + abs(pc - terminal[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return pr, pc

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"

        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        assert shifted[0] == seg_start, "shifted path must start at seg_start"
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
            return current_seg_walkable(p)

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        skin_walkable = np.zeros((h, w), dtype=bool)
        for p in skin:
            if skin_ok(p):
                skin_walkable[p] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]

        route = list(shifted)
        if boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        if len(boundary_path) >= 2:
            dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
            dc_ext = boundary_path[-1][1] - boundary_path[-2][1]
        elif len(route) >= 2:
            dr_ext = route[-1][0] - route[-2][0]
            dc_ext = route[-1][1] - route[-2][1]
        else:
            dr_ext, dc_ext = off_r, off_c

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if not current_seg_walkable(nxt):
                break
            route.append(nxt)
            cur = nxt

        new_pts = route[:L]
        assert len(new_pts) == L, "rerouted segment must preserve pixel count"
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        assert not any(p in set(bar_comp) for p in new_pts), "output segment overlaps the bar"

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
        assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
            f"output segment not monotone wrap: {' '.join(step_classes)}"
        )

        moves.append((seg_col, seg_comp, bar_comp, obstacle_set, new_pts))

    return bg, obstacle_color, output_color, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner legend, then reroute each segment around the nearest designated obstacle.

    Input structure:
    - One corner contains a 2-cell legend strip. From the corner inward:
      the first cell color marks which sprites count as obstacles;
      the second cell color marks the output color used for every rerouted segment.
    - The active scene otherwise matches the original task: straight border-anchored bars, attached irregular segments,
      and several nearby sprites, some of which may be distractors.

    Rule:
    - Ignore non-legend-colored sprites when selecting the obstacle to wrap.
    - Remove the original segment, keep bars and sprites fixed, and redraw the rerouted segment with the legend's second color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, _, output_color, moves = _solve_moves(g)

    for _, seg_comp, _, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, _, _, _, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = output_color
    return out


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _grow_obstacle(
    start: tuple[int, int],
    max_size: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
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


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    remaining = seg_len - 1
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


def _build_output_path(seg_len: int, side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    phase_steps = [(1, side), (1, 0), (1, -side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
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


def _decoy_templates(side: int) -> list[list[tuple[int, int]]]:
    return [
        [(1, -side)],
        [(1, -side), (2, -side)],
        [(1, -side), (1, -2 * side)],
        [(2, -side)],
    ]


def _make_triple(n: int) -> dict[str, object] | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(remaining - 1, PHASE_MAX))
    remaining -= p1
    p2 = random.randint(0, min(remaining, PHASE_MAX))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set)
    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None

    used = set(bar) | set(input_seg) | set(output_seg) | set(obstacle)
    decoy = None
    for tpl in random.sample(_decoy_templates(side), k=len(_decoy_templates(side))):
        if any(p in used for p in tpl):
            continue
        if _is_4adj(tpl, bar) or _is_4adj(tpl, input_seg) or _is_4adj(tpl, obstacle):
            continue
        decoy = tpl
        break
    if decoy is None:
        return None

    union = bar + obstacle + input_seg + output_seg + decoy
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "decoy": decoy,
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


def _place_triple(triple: dict[str, object], edge: str, n: int, reserved: np.ndarray) -> dict[str, object] | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"] + triple["decoy"]
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
            "decoy": _transform(triple["decoy"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _legend_variants(n: int) -> list[dict[str, object]]:
    variants = []
    for spec in _legend_specs(n, n):
        cells = spec["cells"]
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0 = max(0, min(rs) - 1)
        c0 = max(0, min(cs) - 1)
        r1 = min(n - 1, max(rs) + 1)
        c1 = min(n - 1, max(cs) + 1)
        variants.append({"cells": cells, "reserve": (r0, c0, r1, c1)})
    return variants


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    triple_count = 1 if n < 16 else random.choices([1, 2], weights=[1, 3], k=1)[0]

    palette = random.sample(range(10), 6)
    bg, bar_color, seg_color, obstacle_color, output_color, decoy_color = palette

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples: list[dict[str, object]] = []

    legend = random.choice(_legend_variants(n))
    legend_cells = legend["cells"]
    for idx, cell in enumerate(legend_cells):
        inp[cell] = obstacle_color if idx == 0 else output_color
        out[cell] = obstacle_color if idx == 0 else output_color
    r0, c0, r1, c1 = legend["reserve"]
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    for edge in random.sample(list(EDGE_DATA), triple_count):
        placed = None
        for _ in range(PER_EDGE_RETRIES):
            triple = _make_triple(n)
            if triple is None:
                continue
            placed = _place_triple(triple, edge, n, reserved)
            if placed is not None:
                break
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["input_seg"], seg_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["decoy"], decoy_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["output_seg"], output_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["decoy"], decoy_color)

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
    for placed in placed_triples:
        tr, tc = placed["terminal"]
        own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
        decoy_md = min(abs(r - tr) + abs(c - tc) for r, c in placed["decoy"])
        if decoy_md >= own:
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
