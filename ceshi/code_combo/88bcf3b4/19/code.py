# Combo variant of 88bcf3b4 (index 19)
# combo_spec: D7/0+D9/2
# Name: 保留原始无退出绕行
#
# Design analysis for the combo variant
# -------------------------------------
# spec:
#   fused_change:
#     original:
#       the attached irregular segment is removed, wrapped around the obstacle, and extended diagonally after leaving the skin
#     combo:
#       the original irregular segment remains in place
#       a fourth-color copy is added, but it contains only the approach and alongside phases
#       there is no exit phase and no attempt to replenish missing length after overlap with the source root
#   concrete_example:
#     example:
#       source_length: 7
#       approach_steps: 2
#       alongside_steps: 2
#     original_flow:
#       - clear the source
#       - draw a 7-pixel rerouted path: approach + alongside + diagonal exit tail
#     combo_flow:
#       - keep the 7-pixel source unchanged
#       - reuse the legend color from a unique 2x2 swatch
#       - draw only the visible prefix near the obstacle: the 2 diagonal approach cells plus the 2 straight alongside cells
#       - stop immediately when the skin traversal ends, so the output visibly becomes "old squiggle + short copied arc"
#   complexity_calibration:
#     - the obstacle-following geometry is still the original task's core reasoning burden
#     - the 2x2 legend remains necessary, otherwise the fourth color would be unconstrained
#     - the copy is intentionally shorter than the source in generated samples, making D9/2 visually observable rather than degenerate
#     - both approach and alongside are forced to be present, so the new arc is not reducible to a single diagonal or a single straight stub
#   implicit_assumption_audit:
#     - old solver assumption "source is erased" is invalid; output must start from a full copy of the input
#     - old D7-only assumption "length must be preserved by extending farther" is invalid; now truncation is the intended behavior
#     - old obstacle search must still exclude the legend color, otherwise the marker could be mistaken for an obstacle
#     - route painting cannot rely on overwriting occupied cells, so the visible copied arc is restricted to background cells only
#     - route validation regex changes from FL*F*FR* / FR*F*FL* to FL*F* / FR*F*, because exit is forbidden
#   placement_legality:
#     - the generator is output-first and places bar, obstacle, source, copied arc, and the legend with padding
#     - the copied arc must intersect the source only at the hidden root cell; the visible copy stays disjoint
#     - obstacles are rejected if they contain any full 2x2 block, preserving the legend as the unique solid-2x2 cue
#     - own obstacle must be closer to its bar terminal than any other obstacle component when multiple triples exist
#   visual_observability:
#     - every sample shows the old source still present and a new fourth-color arc hugging the obstacle
#     - the only place where the fourth color appears in the input is the isolated 2x2 legend, making the color mapping inferable
#     - the copied arc is consistently shorter than the source, so the absence of exit is visible across samples
#   visual_cleanliness:
#     - only 1-2 triples are used, with extra padding and a compact legend
#     - the visible copied arc sits on background cells, avoiding overpaint clutter around the source
#     - both the source and copied arc use distinct colors and remain spatially separated except for the hidden attachment root
#   color_inference:
#     - the fourth color is never guessed from numeric ids
#     - it is identified purely by the unique interior 2x2 marker, and every copied arc in the output reuses that same marker color
#
# New solver assumptions that the generator must guarantee:
# - exactly one non-border solid 2x2 component exists in the legend color
# - each source segment is a simple 8-connected stroke whose terminal-adjacent endpoint has degree 1
# - each copied route overlaps the source only at the shared root cell, so every painted copied cell is background
# - the approach+alongside route is at least 3 cells long in raw form and contains both a diagonal run and a straight run

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


def _forward_dir(bar_comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs_bar = [r for r, _ in bar_comp]
    cs_bar = [c for _, c in bar_comp]
    if len(set(cs_bar)) == 1:
        return (1, 0) if min(rs_bar) == 0 else (-1, 0)
    return (0, 1) if min(cs_bar) == 0 else (0, -1)


def _classify_route_steps(route: list[tuple[int, int]], bar_comp: list[tuple[int, int]]) -> list[str]:
    fwd = _forward_dir(bar_comp)

    step_classes = []
    for j in range(len(route) - 1):
        dr = route[j + 1][0] - route[j][0]
        dc = route[j + 1][1] - route[j][1]
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
    return step_classes


def _truncate_no_exit_route(route: list[tuple[int, int]], bar_comp: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[str]]:
    step_classes = _classify_route_steps(route, bar_comp)
    assert step_classes, "route must contain at least one step"
    first = step_classes[0]
    assert first in ("FL", "FR"), f"approach must start diagonally, got {first}"

    keep_cells = 1
    seen_straight = False
    for sc in step_classes:
        if not seen_straight:
            if sc == first:
                keep_cells += 1
                continue
            if sc == "F":
                seen_straight = True
                keep_cells += 1
                continue
            break
        if sc == "F":
            keep_cells += 1
            continue
        break

    truncated = route[:keep_cells]
    truncated_steps = step_classes[: keep_cells - 1]
    return truncated, truncated_steps


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

        def skin_ok(p: tuple[int, int]) -> bool:
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        route = list(shifted)
        fwd = _forward_dir(bar_comp)
        cur = entry
        while True:
            nxt = (cur[0] + fwd[0], cur[1] + fwd[1])
            if nxt not in skin or not skin_ok(nxt):
                break
            route.append(nxt)
            cur = nxt

        no_exit_route, step_classes = _truncate_no_exit_route(route, bar_comp)
        raw_route = no_exit_route[:L]
        assert raw_route, "raw route must be non-empty"
        assert not any(p in obstacle_set for p in raw_route), "copied route passes through obstacle"

        step_classes = _classify_route_steps(raw_route, bar_comp)
        s = "".join(step_classes)
        assert re.fullmatch(r"(FL)+(F)+", s) or re.fullmatch(r"(FR)+(F)+", s), (
            f"output segment not approach+alongside only: {' '.join(step_classes)}"
        )

        first_step = step_classes[0]
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
        assert apex_inner in obstacle_set, f"obstacle not tight at copied apex: apex={apex}, inner={apex_inner}"

        source_set = set(seg_comp)
        copy_pts = []
        seen = set()
        for p in raw_route:
            if p in source_set or p in seen:
                continue
            seen.add(p)
            copy_pts.append(p)
            if len(copy_pts) == L:
                break
        assert copy_pts, "copied route must contribute visible cells"
        assert all(int(g[r, c]) == bg for r, c in copy_pts), "copied route must occupy fresh background cells"

        moves.append(
            {
                "seg_comp": seg_comp,
                "source_len": L,
                "raw_route_len": len(raw_route),
                "copy_len": len(copy_pts),
                "copy_pts": copy_pts,
                "shared_raw_cells": len(set(raw_route) & source_set),
            }
        )
    return copy_color, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each original irregular segment and add only a short non-exiting wrap copy near the obstacle.

    Input structure:
    - background cells
    - 1-2 border-attached bars, each with one attached irregular segment and one nearby obstacle sprite
    - one isolated interior 2x2 legend block whose color becomes the copied-route color

    Rule:
    - Detect the same bar, source segment, and nearest obstacle structure as in the original task.
    - Reconstruct the reroute toward the obstacle and along its boundary, but stop as soon as the boundary traversal ends; do not add any exit tail.
    - Leave the source segment untouched and paint only the new visible cells of that truncated route in the legend color.

    Key insight:
    - The segment is no longer moved and no longer length-preserved by an exit extension; the output is the original squiggle plus a shorter obstacle-hugging arc copied in the fourth-color legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    copy_color, moves = _solve_moves(g)
    for move in moves:
        for r, c in move["copy_pts"]:
            assert int(out[r, c]) == get_bg(g), "copied route must only occupy background cells"
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


def _build_obstacle_for_wrap(p1: int, p2: int, side: int, max_size: int) -> list[tuple[int, int]]:
    col = side * p1
    min_forward = max(2, p2 + 1)
    max_forward = max(min_forward, min(max_size, p2 + 2))
    forward_len = random.randint(min_forward, max_forward)
    return [(p1 + k, col) for k in range(forward_len)]


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


def _build_short_wrap(side: int, p1: int, p2: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    for _ in range(p1):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc + side))
    for _ in range(p2):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc))
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
    copy_seg = list(raw_output_seg[1:])
    blocked = set(input_seg) | set(bar) | set(obstacle)
    if any(p in blocked for p in copy_seg):
        return None
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

    max_copy_visible = min(2 * PHASE_MAX, seg_len - 2)
    if max_copy_visible < 2:
        return None
    copy_visible_len = random.randint(2, max_copy_visible)
    p1_lo = max(1, copy_visible_len - PHASE_MAX)
    p1_hi = min(PHASE_MAX, copy_visible_len - 1)
    p1 = random.randint(p1_lo, p1_hi)
    p2 = copy_visible_len - p1
    if not (1 <= p2 <= PHASE_MAX):
        return None

    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)
    raw_output_seg = _build_short_wrap(side, p1, p2)
    seg_set = set(raw_output_seg)

    apex = raw_output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(max(OBS_SIZE_MIN, p2 + 1), obs_size_max)
    obstacle = _build_obstacle_for_wrap(p1, p2, side, obs_size)
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
        "source_len": seg_len,
        "copy_len": len(copy_seg),
        "shared_raw_cells": len(set(input_seg) & set(raw_output_seg)),
        "p1": p1,
        "p2": p2,
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
            "source_len": triple["source_len"],
            "copy_len": triple["copy_len"],
            "shared_raw_cells": triple["shared_raw_cells"],
            "p1": triple["p1"],
            "p2": triple["p2"],
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
        "source_lengths": [p["source_len"] for p in placed_triples],
        "copy_lengths": [p["copy_len"] for p in placed_triples],
        "shared_raw_cells": [p["shared_raw_cells"] for p in placed_triples],
        "p1": [p["p1"] for p in placed_triples],
        "p2": [p["p2"] for p in placed_triples],
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


def _sample_variant_metrics(samples: int = 40) -> dict:
    stats = {
        "samples": 0,
        "triples": [],
        "source_lengths": [],
        "copy_lengths": [],
        "shorter_copy_count": 0,
        "shared_root_only_count": 0,
        "legend_ok": 0,
    }
    for _ in range(samples):
        try:
            inp = generate_input()
            _copy_color, moves = _solve_moves(inp)
        except Exception:
            continue
        stats["samples"] += 1
        stats["triples"].append(len(moves))
        source_lengths = [m["source_len"] for m in moves]
        copy_lengths = [m["copy_len"] for m in moves]
        shared_raw_cells = [m["shared_raw_cells"] for m in moves]
        stats["source_lengths"].extend(source_lengths)
        stats["copy_lengths"].extend(copy_lengths)
        stats["shorter_copy_count"] += sum(int(c < s) for s, c in zip(source_lengths, copy_lengths))
        stats["shared_root_only_count"] += sum(int(v == 1) for v in shared_raw_cells)
        bg = get_bg(inp)
        try:
            _ = _detect_copy_color(inp, bg)
            stats["legend_ok"] += 1
        except AssertionError:
            pass
    return stats
