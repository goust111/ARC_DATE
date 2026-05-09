# Variant of 88bcf3b4
# Dimension: D7 — 变换语义 (transform)
# Change: 输入中的原始线段消失（涂为背景色），新路径沿绕行轨迹重新绘制线段色；线段 detach-reattach 但 bar 和障碍物不变。 → 保留原始线段，并额外用第四种颜色画出绕行副本。

# Design analysis for the D7 variant
# ----------------------------------
# spec:
#   core_change:
#     original:
#       the irregular segment is removed from its input position, then redrawn on the wrap route in the same color
#     variant:
#       the original segment stays exactly where it was
#       an additional copied segment is drawn on the wrap route in a fourth color
#   needed_adaptation:
#     because ARC outputs cannot introduce an arbitrary per-instance color with no cue, the input carries
#     one global legend swatch: an isolated 2x2 solid block whose color is reused for every copied route.
#     this is the minimal support structure needed to make the fourth color human-inferable.
#   concrete_pixel_behavior:
#     objects in input:
#       - bg
#       - 1-3 border-attached bars
#       - 1-3 original irregular segments attached to the bars
#       - 1-3 nearby obstacle sprites
#       - exactly one isolated 2x2 legend swatch in the future copy color
#     objects in output:
#       - bars unchanged
#       - obstacles unchanged
#       - legend unchanged
#       - original segments unchanged
#       - each bar also gains one extra wrap-route segment in the legend color
#   flow_trace:
#     original_execution:
#       - detect bar
#       - detect attached segment
#       - compute wrap route of length L
#       - clear the old segment pixels to bg
#       - repaint the route in the original segment color
#     variant_execution:
#       - detect the legend swatch and treat its color as copy_color
#       - detect bar and attached original segment exactly as before
#       - compute the same raw wrap route as before
#       - keep the original segment untouched
#       - convert the raw route into a visible copied route:
#         remove cells already occupied by the original segment, then extend the far end
#         in the same exit direction so the copied route still has L visible pixels
#       - paint only those new bg cells in copy_color
#     key_difference:
#       the geometry engine is unchanged; only the output semantics change from replace to keep+copy,
#       plus one input legend tells the solver which color the copy should use
#   complexity_calibration:
#     - the wrap geometry, obstacle choice, bar detection, and length preservation stay intact
#     - the new legend is small and globally repeated in effect: every copied route in the grid uses it
#     - copy_color is not guessed from numeric ids; it is structurally marked by a unique 2x2 swatch
#   implicit_assumption_audit:
#     - old code cleared seg_comp to bg:
#       invalid now, because source must remain visible
#     - old code repainted only when out[r,c] == bg:
#       still desirable, but now it would silently drop the copied root if the route overlaps the source
#       so the solver must explicitly filter source-overlap cells and compensate by extending further
#     - old obstacle search treated every non-bar/non-segment color as a possible obstacle:
#       invalid now, because the legend color would be misread as an obstacle unless excluded
#     - old generator only needed 4 colors:
#       invalid now, because the copied route needs a visually indicated fourth non-bg color
#   placement_legality:
#     - the legend is placed as an interior 2x2 block with padding, away from all triples
#     - obstacle sprites are rejected if they contain any solid 2x2, so the legend remains uniquely identifiable
#     - the copied route is generated output-first and must stay disjoint from the original segment except
#       for the raw route's shared root cell; this keeps the visible copied route contiguous after dropping that root
#   visual_observability:
#     - in every example, the original segment remains and a second differently colored path appears
#     - the only new color already exists in the input as the 2x2 swatch, so humans can infer the reuse
#     - using the same legend color for all copied routes within one grid reinforces the rule
#   visual_cleanliness:
#     - the legend is compact and isolated
#     - triples are padded apart, so copied routes do not visually tangle with other triples
#     - source and copied routes are distinct colors and occupy disjoint visible cells
#   color_inference:
#     - copy_color is determined solely by the unique 2x2 legend swatch
#     - no ordering by raw color ids is used anywhere in the transformation

from __future__ import annotations

import random
import re
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


def _is_solid_2x2(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 4:
        return False
    rs = sorted(set(r for r, _ in comp))
    cs = sorted(set(c for _, c in comp))
    if len(rs) != 2 or len(cs) != 2:
        return False
    s = set(comp)
    return all((r, c) in s for r in rs for c in cs)


def _has_full_2x2(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    s = set(coords)
    for r, c in s:
        if (r + 1, c) in s and (r, c + 1) in s and (r + 1, c + 1) in s:
            return True
    return False


def _detect_copy_color(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    candidates = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        comps = _components(g, col, conn8=False)
        if len(comps) != 1:
            continue
        comp = comps[0]
        if not _is_solid_2x2(comp):
            continue
        if any(_touches_border(p, h, w) for p in comp):
            continue
        candidates.append(col)
    assert len(candidates) == 1, f"expected unique copy-color legend, got {candidates}"
    return candidates[0]


def _derive_visible_copy(
    g: np.ndarray,
    route: list[tuple[int, int]],
    source_pts: list[tuple[int, int]],
    dr_ext: int,
    dc_ext: int,
    target_len: int,
) -> list[tuple[int, int]]:
    h, w = g.shape
    source_set = set(source_pts)
    visible = [p for p in route if p not in source_set]
    cur = route[-1]
    while len(visible) < target_len:
        nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(g[nr, nc]) != get_bg(g):
            break
        visible.append((nr, nc))
        cur = (nr, nc)
    assert len(visible) == target_len, "copied route must preserve the source pixel count"
    assert len(set(visible)) == len(visible), "copied route must stay simple"
    assert not any(p in source_set for p in visible), "copied route must be visually disjoint from source"
    return visible


def _solve_moves(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    copy_color = _detect_copy_color(g, bg)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg, copy_color})

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
                if v != bg and v not in (bar_col, copy_color):
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
                L = len(path)
                key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"

        best = None
        obstacle_set = None
        for col in obstacle_candidates:
            for comp in _components(g, col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best is None or key < best:
                    best = key
                    obstacle_set = set(comp)
        assert obstacle_set is not None, "must find at least one obstacle component"

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

        raw_route = route[:L]
        assert not any(p in obstacle_set for p in raw_route), "output segment passes through obstacle"

        rs_bar = [r for r, _ in bar_comp]
        cs_bar = [c for _, c in bar_comp]
        if len(set(cs_bar)) == 1:
            fwd = (1, 0) if min(rs_bar) == 0 else (-1, 0)
        else:
            fwd = (0, 1) if min(cs_bar) == 0 else (0, -1)
        step_classes = []
        for j in range(len(raw_route) - 1):
            dr = raw_route[j + 1][0] - raw_route[j][0]
            dc = raw_route[j + 1][1] - raw_route[j][1]
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

        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = raw_route[p1_end]
            perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
            inner = (-perp_dir[0], -perp_dir[1])
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"

        visible_copy = _derive_visible_copy(g, raw_route, seg_comp, dr_ext, dc_ext, L)
        moves.append(
            {
                "bar_comp": bar_comp,
                "terminal": terminal,
                "seg_comp": seg_comp,
                "seg_color": seg_col,
                "raw_route": raw_route,
                "copy_pts": visible_copy,
                "shared_raw_cells": len(set(seg_comp) & set(raw_route)),
            }
        )
    return copy_color, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each original irregular segment in place and add a copied wrap-route segment.

    Input structure:
    - bg
    - bar color: straight line perpendicular to the border
    - segment color: original irregular stroke attached to the bar terminal
    - obstacle color: nearby sprite
    - copy color legend: one isolated 2x2 block whose color will be reused for the copied routes

    Rule:
    - Detect the same wrap route as in the original task.
    - Do not erase the source segment.
    - Instead, draw an extra route in the legend color, preserving the source length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    copy_color, moves = _solve_moves(g)
    for move in moves:
        for r, c in move["copy_pts"]:
            assert int(out[r, c]) == get_bg(g), "copied route must only occupy new background cells"
            out[r, c] = copy_color
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 13, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 2000
PLACEMENT_PAD = 1
MARKER_PAD = 1

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
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]]
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


def _make_copy_path(
    input_seg: list[tuple[int, int]],
    raw_output_seg: list[tuple[int, int]],
    bar: list[tuple[int, int]],
    obstacle: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    root = raw_output_seg[0]
    overlap = set(input_seg) & set(raw_output_seg)
    if overlap != {root}:
        return None
    dr_ext = raw_output_seg[-1][0] - raw_output_seg[-2][0]
    dc_ext = raw_output_seg[-1][1] - raw_output_seg[-2][1]
    blocked = set(input_seg) | set(bar) | set(obstacle)
    copy_seg = list(raw_output_seg[1:])
    cur = raw_output_seg[-1]
    while len(copy_seg) < len(raw_output_seg):
        cur = (cur[0] + dr_ext, cur[1] + dc_ext)
        if cur in blocked:
            return None
        copy_seg.append(cur)
    return copy_seg


def _make_triple(n: int, compact: bool = False) -> dict | None:
    bar_len_max = BAR_LEN_MAX - 1 if compact else BAR_LEN_MAX
    seg_len_max = SEG_LEN_MAX - 1 if compact else SEG_LEN_MAX
    obs_size_max = max(OBS_SIZE_MIN, OBS_SIZE_MAX - 3) if compact else OBS_SIZE_MAX

    max_bar = min(bar_len_max, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(seg_len_max, n - bar_len + 1)
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

    raw_output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(raw_output_seg)

    apex = raw_output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, obs_size_max)
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set)
    if _has_full_2x2(obstacle):
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None

    copy_seg = _make_copy_path(input_seg, raw_output_seg, bar, obstacle)
    if copy_seg is None:
        return None

    union = bar + obstacle + input_seg + copy_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "raw_output_seg": raw_output_seg,
        "copy_seg": copy_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
        "shared_raw_cells": len(set(input_seg) & set(raw_output_seg)),
    }


def _expanded_bbox(coords: list[tuple[int, int]], n: int, pad: int = PLACEMENT_PAD) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - pad),
        max(0, min(cs) - pad),
        min(n - 1, max(rs) + pad),
        min(n - 1, max(cs) + pad),
    )


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["copy_seg"]
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
            "raw_output_seg": _transform(triple["raw_output_seg"], edge, n, bar_len, pos),
            "copy_seg": _transform(triple["copy_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "shared_raw_cells": triple["shared_raw_cells"],
        }
    return None


def _place_marker(n: int, reserved: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    positions = [(r, c) for r in range(1, n - 2) for c in range(1, n - 2)]
    random.shuffle(positions)
    for r, c in positions:
        marker = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
        r0, c0, r1, c1 = _expanded_bbox(marker, n, pad=MARKER_PAD)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return marker, (r0, c0, r1, c1)
    return None


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 17:
        triple_count = 1
    else:
        triple_count = 2 if random.random() < 0.45 else 1

    bg, bar_color, seg_color, obstacle_color, copy_color = random.sample(range(10), 5)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []
    compact = triple_count > 1

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_triple(n, compact=compact)
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
        paint_coords(out, placed["input_seg"], seg_color)
        paint_coords(out, placed["copy_seg"], copy_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    marker_data = _place_marker(n, reserved)
    if marker_data is None:
        return None
    marker, marker_reserve = marker_data
    paint_coords(inp, marker, copy_color)
    paint_coords(out, marker, copy_color)
    r0, c0, r1, c1 = marker_reserve
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True

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
        solver_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(solver_out, out):
        return None

    meta = {
        "triple_count": len(placed_triples),
        "shared_raw_cells": [p["shared_raw_cells"] for p in placed_triples],
        "copy_lengths": [len(p["copy_seg"]) for p in placed_triples],
        "marker_cells": len(marker),
    }
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            grid, _meta = attempt
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")


def _sample_variant_metrics(samples: int = 20) -> dict:
    stats = {
        "samples": 0,
        "triples": [],
        "shared_raw_cells": [],
        "copy_lengths": [],
        "legend_ok": 0,
    }
    for _ in range(samples):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, meta = attempt
        stats["samples"] += 1
        stats["triples"].append(meta["triple_count"])
        stats["shared_raw_cells"].extend(meta["shared_raw_cells"])
        stats["copy_lengths"].extend(meta["copy_lengths"])
        bg = get_bg(inp)
        try:
            _ = _detect_copy_color(inp, bg)
            stats["legend_ok"] += 1
        except AssertionError:
            pass
    return stats
