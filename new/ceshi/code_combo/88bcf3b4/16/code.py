# Combo variant of 88bcf3b4 (index 16)
# combo_spec: D7/0+D10/0
# Name: 保留原始裁剪输出

# Design analysis for the combo variant
# -------------------------------------
# spec:
#   fused_change:
#     original:
#       the attached irregular segment is removed, rerouted around the nearest obstacle,
#       and redrawn in the segment color on the full-size grid
#     combo:
#       keep the original irregular segment in place
#       add a second rerouted copy in a fourth color
#       then crop the final output to the minimal rectangle that contains the bars,
#       obstacles, original segments, and copied rerouted segments
#   why_the_fourth_color_needs_support:
#     ARC outputs cannot introduce an unconstrained new color, so the input includes
#     one isolated 2x2 legend swatch whose color is reused for every copied route.
#     In this combo, that legend is only an input cue; it is removed before cropping,
#     so the output crop is driven by the task objects rather than by the marker.
#   concrete_pixel_trace:
#     example_flow:
#       - input: a top-border bar, one attached zig-zag source segment, a nearby obstacle,
#         and an isolated 2x2 legend block elsewhere in the canvas
#       - original task: erase the source, draw the wrap route in the same color, keep canvas size
#       - D7-only: keep the source, draw the wrap route in legend color, keep the legend, keep canvas size
#       - fused combo: keep the source, draw the wrap route in legend color, remove the legend from the
#         output scene, then tight-crop around the bar + obstacle + source + copied route
#     visible_difference:
#       compared with the base task, there are now two line-like objects in output instead of one
#       compared with the D7-only task, the outer blank canvas and the legend vanish in output
#   complexity_calibration:
#     - the rope-around-obstacle geometry is unchanged, so the spatial reasoning remains the main challenge
#     - only one global legend swatch is used, which keeps color inference simple and repeated
#     - triples share one edge in generated examples so the crop is visibly smaller and not drowned out
#       by bars touching opposite borders
#     - 1-2 triples are enough to show repeated behavior without making the crop rectangle or marker
#       placement ambiguous
#   implicit_assumption_audit:
#     - base solver assumed the source segment must be erased:
#       invalid now; source pixels must survive untouched
#     - base solver treated every non-bg/non-bar/non-segment color as an obstacle:
#       invalid now; the legend color must be detected first and excluded from obstacle search
#     - D7-only solver assumed the legend stays in output:
#       invalid under D10; legend cells must be blanked before crop computation
#     - tight crop on all non-bg cells would accidentally keep the legend if it remained painted:
#       therefore cropping is done after explicit legend removal
#     - copied-route painting can no longer rely on raw route length alone because the source remains;
#       overlapping source cells are filtered and the far end is extended to preserve visible copy length
#   placement_legality:
#     - obstacle sprites are rejected if they contain any solid 2x2, keeping the legend uniquely detectable
#     - copied routes are generated output-first and are required to be visibly disjoint from the source
#       except for the hidden raw root cell used by the geometry engine
#     - all triples are placed on the same border edge and kept away from the other borders, guaranteeing
#       that the final crop is smaller than the input and still respects the bar-touching-border rule
#     - the legend is placed outside the future crop bbox with padding, so humans can see it as a separate cue
#   visual_observability:
#     - input shows a compact 2x2 color cue that is absent from output
#     - output shows both the old segment and a new differently colored wrapped copy
#     - the output frame shrinks to the task-relevant region, which is visually obvious across examples
#   visual_cleanliness:
#     - source, obstacle, bar, and copied route stay non-overlapping as colored regions
#     - the legend is isolated from the active scene, so it reads as a key rather than a second obstacle
#     - same-edge placement reduces sprawling full-canvas crops and keeps the examples readable
#   behavior_metrics_to_check:
#     - every valid sample should satisfy: copied_pixels > 0, crop area < input area, legend_outside_crop = True
#     - these metrics distinguish the combo from both the base task and the uncropped D7-only variant

from __future__ import annotations

import random
import re
from collections import deque

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, get_bg, neighbors4, neighbors8, paint_coords


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


def _detect_copy_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
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
        candidates.append((col, comp))
    assert len(candidates) == 1, f"expected unique copy-color legend, got {[(c, len(comp)) for c, comp in candidates]}"
    return candidates[0]


def _derive_visible_copy(
    g: np.ndarray,
    bg: int,
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
        if int(g[nr, nc]) != bg:
            break
        visible.append((nr, nc))
        cur = (nr, nc)
    assert len(visible) == target_len, "copied route must preserve the source pixel count"
    assert len(set(visible)) == len(visible), "copied route must stay simple"
    assert not any(p in source_set for p in visible), "copied route must be visually disjoint from source"
    return visible


def _solve_moves(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    copy_color, legend_cells = _detect_copy_legend(g, bg)
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
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) not in (col, copy_color)
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
            return int(g[r, c]) in (bg, seg_col)

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
            if int(g[nr, nc]) not in (bg, seg_col):
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

        visible_copy = _derive_visible_copy(g, bg, raw_route, seg_comp, dr_ext, dc_ext, L)
        moves.append(
            {
                "bar_comp": bar_comp,
                "terminal": terminal,
                "seg_comp": seg_comp,
                "seg_color": seg_col,
                "copy_pts": visible_copy,
            }
        )
    return copy_color, legend_cells, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each original irregular segment, add a copied wrapped route in a legend color, then crop away the outer canvas.

    Input structure:
    - bg
    - bar color: one or more straight bars attached perpendicularly to a border
    - segment color: irregular 1-cell-thick source strokes attached to the inner bar endpoints
    - obstacle color: nearby sprites that the copied routes must wrap around
    - copy-color legend: exactly one isolated interior 2x2 block indicating the color used for all copied routes

    Rule:
    - Recover the same wrap geometry as in the base task for each valid bar+segment+obstacle group.
    - Do not erase the original source stroke.
    - Draw an additional wrapped copy in the legend color, preserving the source length and avoiding source overlap.
    - Remove the 2x2 legend from the output scene and tight-crop the result to the smallest rectangle containing the bars,
      obstacles, source segments, and copied routes.

    Key insight:
    - The 2x2 swatch is only a color key. The real transformation is still the rope-like rerouting around the nearest obstacle,
      but now it becomes keep-plus-copy, and the final frame is determined only after that copied route has been added.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    copy_color, legend_cells, moves = _solve_moves(g)

    for r, c in legend_cells:
        out[r, c] = bg

    for move in moves:
        for r, c in move["copy_pts"]:
            assert int(out[r, c]) == bg, "copied route must only occupy new background cells"
            out[r, c] = copy_color

    active_bbox = bbox(out != bg)
    assert active_bbox is not None, "cropped output must contain task objects"
    return crop_bbox(out, active_bbox)


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 14, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 3000
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


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(2, n - 2))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["copy_seg"]
    bar_len = triple["bar_len"]

    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        min_r = min(r for r, _ in union_g)
        max_r = max(r for r, _ in union_g)
        min_c = min(c for _, c in union_g)
        max_c = max(c for _, c in union_g)
        if edge in ("top", "bottom") and (min_c == 0 or max_c == n - 1):
            continue
        if edge in ("left", "right") and (min_r == 0 or max_r == n - 1):
            continue
        if edge == "top" and max_r == n - 1:
            continue
        if edge == "bottom" and min_r == 0:
            continue
        if edge == "left" and max_c == n - 1:
            continue
        if edge == "right" and min_c == 0:
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


def _place_marker(
    n: int,
    reserved: np.ndarray,
    active_bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    positions = [(r, c) for r in range(1, n - 2) for c in range(1, n - 2)]
    random.shuffle(positions)
    padded_active = (
        max(0, active_bbox[0] - 1),
        max(0, active_bbox[1] - 1),
        min(n - 1, active_bbox[2] + 1),
        min(n - 1, active_bbox[3] + 1),
    )
    for r, c in positions:
        marker = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
        marker_bbox = (r, c, r + 1, c + 1)
        if _bbox_intersects(marker_bbox, padded_active):
            continue
        r0, c0, r1, c1 = _expanded_bbox(marker, n, pad=MARKER_PAD)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return marker, (r0, c0, r1, c1)
    return None


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 18:
        triple_count = 1
    else:
        triple_count = 2 if random.random() < 0.45 else 1

    bg, bar_color, seg_color, obstacle_color, copy_color = random.sample(range(10), 5)
    inp = np.full((n, n), bg, dtype=int)
    out_full = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []
    compact = triple_count > 1
    edge = random.choice(list(EDGE_DATA))

    for _ in range(triple_count):
        triple = _make_triple(n, compact=compact)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out_full, placed["bar"], bar_color)
        paint_coords(out_full, placed["obstacle"], obstacle_color)
        paint_coords(out_full, placed["input_seg"], seg_color)
        paint_coords(out_full, placed["copy_seg"], copy_color)

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

    active_bbox = bbox(out_full != bg)
    if active_bbox is None:
        return None

    marker_data = _place_marker(n, reserved, active_bbox)
    if marker_data is None:
        return None
    marker, marker_reserve = marker_data
    paint_coords(inp, marker, copy_color)
    r0, c0, r1, c1 = marker_reserve
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    expected = crop_bbox(out_full, active_bbox)

    try:
        solver_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(solver_out, expected):
        return None

    marker_bbox = (min(r for r, _ in marker), min(c for _, c in marker), max(r for r, _ in marker), max(c for _, c in marker))
    meta = {
        "triple_count": len(placed_triples),
        "shared_raw_cells": [p["shared_raw_cells"] for p in placed_triples],
        "copy_lengths": [len(p["copy_seg"]) for p in placed_triples],
        "input_shape": tuple(inp.shape),
        "output_shape": tuple(expected.shape),
        "crop_reduction": inp.size - expected.size,
        "legend_outside_crop": not _bbox_intersects(marker_bbox, active_bbox),
        "copied_pixels": int(np.sum(expected == copy_color)),
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
        "shared_raw_cells": [],
        "copy_lengths": [],
        "crop_reduction": [],
        "input_shapes": [],
        "output_shapes": [],
        "legend_outside_crop": 0,
        "copied_pixels": [],
    }
    for _ in range(samples):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        _inp, meta = attempt
        stats["samples"] += 1
        stats["triples"].append(meta["triple_count"])
        stats["shared_raw_cells"].extend(meta["shared_raw_cells"])
        stats["copy_lengths"].extend(meta["copy_lengths"])
        stats["crop_reduction"].append(meta["crop_reduction"])
        stats["input_shapes"].append(meta["input_shape"])
        stats["output_shapes"].append(meta["output_shape"])
        stats["copied_pixels"].append(meta["copied_pixels"])
        if meta["legend_outside_crop"]:
            stats["legend_outside_crop"] += 1
    return stats
