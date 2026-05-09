# Variant of 88bcf3b4
# Dimension: D5 — 目标/停止条件 (dst)
# Change: 线段绕过离 bar 末端曼哈顿距离最近的障碍物精灵，到达对侧后沿 45 度对角延伸直到像素数耗尽或到达网格边界。... → 堆叠到目标精灵上
#
# Design analysis
# variant_spec:
#   core_visual_change:
#     original: segment wraps around the nearest obstacle and then keeps diverging away
#       on a diagonal tail after it leaves the obstacle skin.
#     variant: segment still detaches, approaches the nearest obstacle, and wraps tightly,
#       but its endpoint now stops on the opposite obstacle face. The output has a
#       convergent "stacked against the surface" ending instead of an outward tail.
#   concrete_pixel_behavior:
#     objects:
#       - bar: unchanged straight border-anchored line.
#       - input segment: unchanged irregular 8-connected stroke attached at the bar terminal.
#       - obstacle: unchanged nearby 4-connected sprite, but generation keeps one clean outer
#         face so the stacked stopping condition is readable.
#     output path:
#       - phase 1: diagonal-forward approach from the segment root side toward the obstacle.
#       - phase 2: straight-forward run along the obstacle's opposite face.
#       - stop: end on that face; no phase-3 diagonal peel-away.
#   behavior_trace:
#     small_example:
#       - if the bar is on the top edge and the segment starts on the bar's right side,
#         the original rule routes diagonally down-right toward the obstacle, then down
#         along its right face, then continues down-left away from it.
#       - the variant keeps the same diagonal approach and same face-following run, but
#         the last painted pixel is the lowest cell still adjacent to the obstacle's right
#         face. The route terminates there even if the original rule could have continued.
#   complexity_calibration:
#     - the generator forces a visible stacked run of at least 2 pixels so the missing tail
#       is obvious in 2-4 examples.
#     - obstacle shapes may grow inward and stay irregular, but they do not break the clean
#       contact face used by the solver; this avoids ambiguous exits.
#     - the approach length stays in a small range (Manhattan distance 2, 4, or 6) so the
#       nearest-obstacle choice remains visually stable and repeated across samples.
#   old_assumptions_audit:
#     - original `exit_cell = max(skin_candidates, ...)` assumed the best endpoint should be
#       any far skin cell because the path would later keep extending away. That assumption
#       is invalid here; we now restrict the target to the obstacle face aligned with the
#       segment's attachment side.
#     - original `while len(route) < L: ... diagonal extension ...` encoded the old dst rule.
#       The variant removes this loop entirely; the generator instead ensures the intended
#       stopping face already consumes the full segment length.
#     - original monotonicity check allowed `FL*F*FR*` or `FR*F*FL*`. The variant must allow
#       only `FL*F*` or `FR*F*`, since the path no longer peels away on the far side.
#   placement_legality:
#     - the obstacle is generated from a spine that sits just inside the stacked face of the
#       output segment; optional extra obstacle pixels may only grow further inward and within
#       the same row range. This guarantees no overlap with the bar/segment, keeps the same
#       nearest obstacle pixel, and leaves the stacked face walkable.
#     - existing reserved-bbox placement is preserved so multiple triples do not interfere.
#   observability:
#     - samples are rejected unless the variant output differs from the original rule by at
#       least one possible extra diagonal continuation cell. This prevents silent degeneration
#       back to the source task.

from __future__ import annotations

from collections import deque
import importlib.util
import random

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


def _bar_forward(bar_comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _step_classes(new_pts: list[tuple[int, int]], fwd: tuple[int, int]) -> list[str]:
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
    return step_classes


def _bfs_path_with_diag_filter_set(
    start: tuple[int, int], goal: tuple[int, int], allowed: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            nxt = (nr, nc)
            if nxt not in allowed or nxt in prev:
                continue
            dr = nr - r
            dc = nc - c
            if dr != 0 and dc != 0:
                if (r, c + dc) in allowed or (r + dr, c) in allowed:
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


def _local_skin(obstacle: set[tuple[int, int]], bounds: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    skin = set()
    for r, c in obstacle:
        for nr, nc in neighbors4(r, c):
            if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in obstacle:
                skin.add((nr, nc))
    return skin


def _local_shifted_path(p1: int, side: int) -> list[tuple[int, int]]:
    return [(k, side * (k + 1)) for k in range(p1 + 1)]


def _local_variant_route(
    obstacle: set[tuple[int, int]], side: int, p1: int, seg_len: int, bounds: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    terminal = (0, 0)
    shifted = _local_shifted_path(p1, side)
    entry = shifted[-1]
    if entry in obstacle:
        return None
    skin = _local_skin(obstacle, bounds)
    if entry not in skin:
        return None

    cr = sum(r for r, _ in obstacle) / len(obstacle)
    cc = sum(c for _, c in obstacle) / len(obstacle)
    away = (cr - terminal[0], cc - terminal[1])
    fwd = (1, 0)

    best = None
    best_boundary = None
    for cand in skin:
        boundary_path = _bfs_path_with_diag_filter_set(entry, cand, skin)
        if boundary_path is None:
            continue
        if set(boundary_path[1:]) & set(shifted[:-1]):
            continue
        total_len = len(shifted) + len(boundary_path) - 1
        if total_len > seg_len:
            continue
        forward_proj = (cand[0] - terminal[0]) * fwd[0] + (cand[1] - terminal[1]) * fwd[1]
        md = abs(cand[0] - terminal[0]) + abs(cand[1] - terminal[1])
        dot = (cand[0] - terminal[0]) * away[0] + (cand[1] - terminal[1]) * away[1]
        key = (total_len, forward_proj, md, dot, -cand[0], -cand[1])
        if best is None or key > best:
            best = key
            best_boundary = boundary_path
    if best_boundary is None:
        return None
    route = list(shifted)
    route.extend(best_boundary[1:])
    return route if len(route) == seg_len else None


def _local_original_route(
    obstacle: set[tuple[int, int]], side: int, p1: int, seg_len: int, bounds: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    terminal = (0, 0)
    shifted = _local_shifted_path(p1, side)
    entry = shifted[-1]
    skin = _local_skin(obstacle, bounds)
    if entry not in skin:
        return None

    cr = sum(r for r, _ in obstacle) / len(obstacle)
    cc = sum(c for _, c in obstacle) / len(obstacle)
    away = (cr - terminal[0], cc - terminal[1])

    def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
        r, c = p
        md = abs(r - terminal[0]) + abs(c - terminal[1])
        dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
        return (md, dot, -r, -c)

    exit_cell = max(skin, key=exit_key)
    boundary_path = _bfs_path_with_diag_filter_set(entry, exit_cell, skin)
    if boundary_path is None or len(boundary_path) < 2:
        return None
    if set(boundary_path[1:]) & set(shifted[:-1]):
        return None

    route = list(shifted)
    route.extend(boundary_path[1:])
    dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
    dc_ext = boundary_path[-1][1] - boundary_path[-2][1]
    r0, c0, r1, c1 = bounds
    cur = route[-1]
    while len(route) < seg_len:
        nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
        if not (r0 <= nr <= r1 and c0 <= nc <= c1):
            break
        if (nr, nc) in obstacle:
            break
        route.append((nr, nc))
        cur = (nr, nc)
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing an irregular line segment around the nearest obstacle and stack it onto the
    obstacle's opposite face, preserving the original segment length.
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

        skin_cells = [p for p in skin if skin_ok(p)]
        assert entry in skin_cells, "entry must lie on the obstacle skin used for stacking"

        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin_cells:
            skin_walkable[r, c] = True

        fwd = _bar_forward(bar_comp)
        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])
        best_exit = None
        best_boundary = None
        for cand in skin_cells:
            boundary_path = _bfs_path_with_diag_filter(entry, cand, skin_walkable)
            if boundary_path is None:
                continue
            if set(boundary_path[1:]) & set(shifted[:-1]):
                continue
            total_len = len(shifted) + len(boundary_path) - 1
            if total_len > L:
                continue
            forward_proj = (cand[0] - terminal[0]) * fwd[0] + (cand[1] - terminal[1]) * fwd[1]
            md = abs(cand[0] - terminal[0]) + abs(cand[1] - terminal[1])
            dot = (cand[0] - terminal[0]) * away[0] + (cand[1] - terminal[1]) * away[1]
            key = (total_len, forward_proj, md, dot, -cand[0], -cand[1])
            if best_exit is None or key > best_exit:
                best_exit = key
                best_boundary = boundary_path
        assert best_boundary is not None, "no obstacle-skin stack route can realize the output path"

        route = list(shifted)
        route.extend(best_boundary[1:])
        new_pts = route[:L]

        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

        step_classes = _step_classes(new_pts, fwd)
        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = new_pts[p1_end]
            inner = (-off_r, -off_c)
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"
        assert all(p in skin_cells for p in new_pts[len(shifted) - 1 :]), "stacked tail must stay on obstacle skin"

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

GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
APPROACH_MAX = 3
STACK_RUN_MIN = 2
OBS_SIZE_MAX = 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 1200
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

_ORIGINAL_SOLVER = None


def _load_original_solver():
    global _ORIGINAL_SOLVER
    if _ORIGINAL_SOLVER is None:
        spec = importlib.util.spec_from_file_location(
            "orig_88bcf3b4", "/home/lixinyu/snap/new/arc2_solver_select/88bcf3b4.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ORIGINAL_SOLVER = mod
    return _ORIGINAL_SOLVER


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


def _grow_inward_obstacle(
    base: list[tuple[int, int]], max_size: int, side: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = set(base)
    row_lo = min(r for r, _ in base)
    row_hi = max(r for r, _ in base)
    face_inner_col = base[0][1]
    while len(obstacle) < max_size:
        cands = []
        for r, c in obstacle:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in obstacle or (nr, nc) in forbidden:
                    continue
                if not (row_lo <= nr <= row_hi):
                    continue
                if side == 1 and nc > face_inner_col:
                    continue
                if side == -1 and nc < face_inner_col:
                    continue
                cands.append((nr, nc))
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


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
    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    max_p1 = min(APPROACH_MAX, seg_len - 1 - STACK_RUN_MIN)
    if max_p1 < 1:
        return None
    p1 = random.randint(1, max_p1)
    spine_len = random.randint(STACK_RUN_MIN + 1, min(seg_len - p1, OBS_SIZE_MAX))
    entry = _local_shifted_path(p1, side)[-1]
    face_inner_col = entry[1] - side
    spine = [(r, face_inner_col) for r in range(entry[0], entry[0] + spine_len)]
    nearest_md = min(abs(r) + abs(c) for r, c in spine)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    obs_size = random.randint(min(len(spine) + 1, OBS_SIZE_MAX), OBS_SIZE_MAX)
    shifted = _local_shifted_path(p1, side)
    obstacle = _grow_inward_obstacle(spine, obs_size, side, set(shifted) | bar_set)
    if len(obstacle) < len(spine) + 1:
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "side": side,
        "seg_len": seg_len,
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
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"]
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
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "seg_start": _transform([(0, triple["side"])], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    triple_count = 1

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
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

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    try:
        pred = main(inp)
        if np.array_equal(pred, inp):
            return None
    except Exception:
        return None

    placed = placed_triples[0]
    terminal = placed["terminal"]
    seg_start = placed["seg_start"]
    off = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
    seg_cells = set(map(tuple, np.argwhere(pred == seg_color)))
    obstacle_cells = set(map(tuple, np.argwhere(inp == obstacle_color)))
    if len(seg_cells) != len(placed["input_seg"]):
        return None
    touch_dirs = set()
    touch_cells = set()
    face_cells = set()
    for r, c in seg_cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in obstacle_cells:
                continue
            touch_cells.add((r, c))
            d = (nr - r, nc - c)
            touch_dirs.add(d)
            if d == (-off[0], -off[1]):
                face_cells.add((r, c))
    if len(touch_cells) < 3:
        return None
    if len(touch_dirs) < 2:
        return None
    if len(face_cells) == len(touch_cells):
        return None
    try:
        old_out = _load_original_solver().main(inp)
        if np.array_equal(old_out, pred):
            return None
    except Exception:
        pass
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
