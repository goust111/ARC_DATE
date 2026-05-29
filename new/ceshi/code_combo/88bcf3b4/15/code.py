# Combo variant of 88bcf3b4 (index 15)
# combo_spec: D5/3+D16/1
# Name: 远距障碍标记终止

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords

# ---------------------------------------------------------------------------
# Design analysis for the combo variant D5/3 + D16/1
#
# spec_variant:
#   base_task: 88bcf3b4
#   fused_change:
#     - D16/1 pushes the chosen obstacle much farther from the bar terminal, so the
#       segment must spend several cells in a long monotone approach before it can wrap.
#     - D5/3 adds a same-color isolated marker; after wrapping, the segment keeps going
#       in a monotone forward ray until it reaches that marker and then stops exactly there.
#
# a) Concrete behavior trace
#   Local coordinates with terminal at (0,0), forward = +r, segment root on the right:
#     input:
#       bar      = [(-4,0),(-3,0),(-2,0),(-1,0),(0,0)]
#       segment  = [(0,1),(-1,1),(-2,1),(-3,2),(-4,2),(-5,3),...]
#       obstacle = a small sprite whose nearest cell is around (3,3) or (4,4), so its
#                  manhattan distance from the terminal is 6-8 instead of 2-6
#       marker   = a lone pixel of the same color as the segment, for example at (10,5)
#     original task:
#       move the segment toward the nearest obstacle, skim its skin, then keep extending
#       along the free exit direction until length runs out or the boundary is hit
#     fused task:
#       still move toward the nearest obstacle and wrap it, but the tail is no longer free:
#       among valid obstacle-skin exits, choose one whose forward ray lands exactly on the
#       same-color marker; the moved segment ends on that marker pixel
#     difference:
#       the obstacle is visibly farther from the bar, so the approach leg is longer, and the
#       singleton marker now controls the post-wrap destination instead of a generic diagonal
#
# b) Complexity calibration
#   - The long-approach change is kept in a narrow range: obstacle md 5-12, mostly produced by
#     approach lengths 3-6. This is visibly different from the base task without exploding the
#     number of spatial cases.
#   - Marker-driven tails use only monotone forward rays {F, FL, FR}; generator mainly emits
#     straight or outward-diagonal tails so the destination clue is easy to read.
#   - Segment colors are unique per triple when several triples appear, so each marker has an
#     unambiguous partner and the user does not need hidden color-order conventions.
#
# c) Implicit assumptions audit
#   - Base solver assumed the segment color formed one component. That fails once a same-color
#     singleton marker is added, so the solver must choose the component attached to the bar and
#     separately locate the isolated singleton.
#   - Base solver assumed the obstacle sat fairly close to the bar. The pathfinding itself does
#     not require that; only the generator constraints did, so those bounds are widened and the
#     segment length / grid size are increased accordingly.
#   - Base solver assumed the tail was determined by the last skin step. That is false here:
#     after the wrap, the solver must search for a skin exit whose monotone forward ray reaches
#     the marker with the exact remaining number of cells.
#
# d) Placement legality
#   - Generator is still output-first: first choose a valid long wrapped route with a reachable
#     marker endpoint, then derive the input irregular stroke plus singleton marker.
#   - Reserved expanded boxes are taken from the union of bar, obstacle, input stroke, and full
#     output route so distant tails do not collide with foreign triples.
#   - Extra validation ensures that, from each terminal, its own obstacle is the nearest foreign
#     non-bar component; otherwise another triple's segment or marker could be mistaken as an
#     obstacle by the solver.
#
# e) Visual observability
#   - The fused behavior is visible in raw pixels: the obstacle is noticeably separated from the
#     bar, creating a long empty corridor before the wrap, and the same-color singleton is the
#     clear endpoint clue reused in the output.
#   - Because the generator rejects cases where the marker route collapses to the old free-tail
#     route, the marker is causally relevant rather than decorative.
#
# f) Visual cleanliness
#   - Only one new clue type is added: a singleton marker.
#   - Large padding and 4/8-isolation checks prevent marker/segment/object clutter.
#   - Obstacles remain compact 4-connected sprites so the wrap region stays readable.
#
# g) Color inference
#   - Colors keep a structural role: shared bar color, shared obstacle color, unique segment color
#     per triple, with the marker sharing only its segment color.
#   - No rule depends on numeric color ids or hidden ordering; matching is visible by equality.
#
# h) Generator oracle strategy
#   - Candidate grids are accepted only when main(inp) exactly reproduces the intended output.
#   - Additional anti-degeneracy filtering compares the marker-driven route with the original
#     free-tail route on the same input and rejects samples where every triple would look the same.
#
# i) New solver assumptions enforced by the generator
#   - Exactly one 4-isolated singleton exists for each segment color.
#   - The chosen marker is reachable from some obstacle-skin exit by a monotone forward ray with
#     the exact remaining length.
#   - Obstacle md from the terminal is always 5-12, and the input stroke stays away from both the
#     obstacle and the marker so object roles remain visually separable.
# ---------------------------------------------------------------------------


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


def _bar_forward(comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _positive_forward_dirs(fwd: tuple[int, int]) -> list[tuple[int, int]]:
    lat = (fwd[1], -fwd[0])
    return [
        fwd,
        (fwd[0] + lat[0], fwd[1] + lat[1]),
        (fwd[0] - lat[0], fwd[1] - lat[1]),
    ]


def _isolated_marker(g: np.ndarray, seg_col: int, bg: int, seg_comp: list[tuple[int, int]]) -> tuple[int, int]:
    h, w = g.shape
    seg_set = set(seg_comp)
    markers = []
    for comp in _components(g, seg_col, conn8=True):
        if set(comp) == seg_set:
            continue
        if len(comp) != 1:
            continue
        r, c = comp[0]
        if any(0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg for nr, nc in neighbors4(r, c)):
            continue
        markers.append((r, c))
    assert len(markers) == 1, f"expected exactly one isolated marker for segment color {seg_col}, got {markers}"
    return markers[0]


def _ray_to_marker(
    start: tuple[int, int],
    marker: tuple[int, int],
    step: tuple[int, int],
    walkable: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    cur = start
    path = [start]
    for _ in range(max(h, w)):
        nr, nc = cur[0] + step[0], cur[1] + step[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        nxt = (nr, nc)
        if nxt != marker and not bool(walkable[nr, nc]):
            return None
        path.append(nxt)
        cur = nxt
        if cur == marker:
            return path
    return None


def _route_to_marker(
    shifted: list[tuple[int, int]],
    entry: tuple[int, int],
    skin: set[tuple[int, int]],
    marker: tuple[int, int],
    walkable: np.ndarray,
    seg_len: int,
    fwd: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = walkable.shape
    skin_walkable = np.zeros((h, w), dtype=bool)
    for r, c in skin:
        if bool(walkable[r, c]) or (r, c) == marker:
            skin_walkable[r, c] = True
    skin_walkable[entry] = True

    candidates = []
    for exit_cell in sorted(skin | {entry}):
        if not skin_walkable[exit_cell]:
            continue
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable)
        if boundary_path is None:
            continue
        prefix = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            prefix.extend(boundary_path[1:])
        if len(prefix) > seg_len:
            continue
        for dir_idx, step in enumerate(_positive_forward_dirs(fwd)):
            tail = _ray_to_marker(exit_cell, marker, step, walkable)
            if tail is None:
                continue
            route = prefix + tail[1:]
            if len(route) != seg_len:
                continue
            if len(set(route)) != len(route):
                continue
            tail_len = len(tail) - 1
            key = (tail_len, -len(boundary_path), dir_idx, exit_cell[0], exit_cell[1])
            candidates.append((key, route))

    assert candidates, f"no obstacle-exit route reaches marker {marker} with segment length {seg_len}"
    return min(candidates, key=lambda x: x[0])[1]


def _route_free_tail(
    shifted: list[tuple[int, int]],
    entry: tuple[int, int],
    skin: set[tuple[int, int]],
    walkable: np.ndarray,
    seg_len: int,
    terminal: tuple[int, int],
    obstacle_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    skin_candidates = [p for p in skin if bool(walkable[p])] + [entry]
    cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
    cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
    away = (cr - terminal[0], cc - terminal[1])

    def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
        r, c = p
        md = abs(r - terminal[0]) + abs(c - terminal[1])
        dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
        return (md, dot, -r, -c)

    exit_cell = max(skin_candidates, key=exit_key)
    h, w = walkable.shape
    skin_walkable = np.zeros((h, w), dtype=bool)
    for r, c in skin:
        if bool(walkable[r, c]):
            skin_walkable[r, c] = True
    skin_walkable[entry] = True
    skin_walkable[exit_cell] = True
    boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]
    assert len(boundary_path) >= 2, "free-tail route needs a nontrivial skin exit"

    dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
    dc_ext = boundary_path[-1][1] - boundary_path[-2][1]
    route = list(shifted)
    if boundary_path and boundary_path[0] == entry:
        route.extend(boundary_path[1:])

    cur = route[-1]
    while len(route) < seg_len:
        nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if not bool(walkable[nr, nc]):
            break
        route.append((nr, nc))
        cur = (nr, nc)
    return route[:seg_len]


def _bar_candidates(g: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]], tuple[int, int]]]:
    h, w = g.shape
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
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
    return sorted(bars, key=lambda x: (-len(x[1]), x[0], x[2]))


def _attached_segment(
    g: np.ndarray, terminal: tuple[int, int], bar_col: int
) -> tuple[int, tuple[int, int], list[tuple[int, int]], int]:
    h, w = g.shape
    neigh = []
    for nr, nc in neighbors4(terminal[0], terminal[1]):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bar_col and v != get_bg(g):
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
    return seg_col, seg_start, seg_comp, seg_len


def _nearest_obstacle(
    g: np.ndarray, terminal: tuple[int, int], excluded: set[int], bg: int
) -> tuple[set[tuple[int, int]], int]:
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg} - excluded)
    assert colors, "must have at least one obstacle color distinct from bar and segment"
    best = None
    obstacle_set = None
    for col in colors:
        for comp in _components(g, col, conn8=True):
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best is None or key < best:
                best = key
                obstacle_set = set(comp)
    assert obstacle_set is not None
    return obstacle_set, int(best[0])


def _routing_context(
    g: np.ndarray,
    bg: int,
    bar_col: int,
    bar_comp: list[tuple[int, int]],
    terminal: tuple[int, int],
    stop_at_marker: bool,
) -> dict:
    h, w = g.shape
    seg_col, seg_start, seg_comp, seg_len = _attached_segment(g, terminal, bar_col)
    marker = _isolated_marker(g, seg_col, bg, seg_comp) if stop_at_marker else None
    obstacle_set, obstacle_md = _nearest_obstacle(g, terminal, {bar_col, seg_col}, bg)

    off_r = seg_start[0] - terminal[0]
    off_c = seg_start[1] - terminal[1]
    walkable = (g == bg) | (g == seg_col)
    walkable[terminal] = True

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
    p_near = (pr, pc)
    walkable[p_near] = True
    path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
    assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"
    shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
    assert shifted and shifted[0] == seg_start
    assert all(0 <= r < h and 0 <= c < w for r, c in shifted)
    entry = shifted[-1]

    skin = set()
    for r, c in obstacle_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                skin.add((nr, nc))

    return {
        "bar_comp": bar_comp,
        "bar_col": bar_col,
        "terminal": terminal,
        "seg_col": seg_col,
        "seg_start": seg_start,
        "seg_comp": seg_comp,
        "seg_len": seg_len,
        "marker": marker,
        "obstacle_set": obstacle_set,
        "obstacle_md": obstacle_md,
        "walkable": walkable,
        "shifted": shifted,
        "entry": entry,
        "skin": skin,
        "fwd": _bar_forward(bar_comp),
    }


def _planned_moves(grid: np.ndarray, stop_at_marker: bool) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    moves = []
    for bar_col, bar_comp, terminal in _bar_candidates(g, bg):
        ctx = _routing_context(g, bg, bar_col, bar_comp, terminal, stop_at_marker)
        if stop_at_marker:
            new_pts = _route_to_marker(
                ctx["shifted"],
                ctx["entry"],
                ctx["skin"],
                ctx["marker"],
                ctx["walkable"],
                ctx["seg_len"],
                ctx["fwd"],
            )
        else:
            new_pts = _route_free_tail(
                ctx["shifted"],
                ctx["entry"],
                ctx["skin"],
                ctx["walkable"],
                ctx["seg_len"],
                ctx["terminal"],
                ctx["obstacle_set"],
            )
        ctx["new_pts"] = new_pts
        moves.append(ctx)
    return moves


def _validate_route(ctx: dict) -> None:
    new_pts = ctx["new_pts"]
    obstacle_set = ctx["obstacle_set"]
    terminal = ctx["terminal"]
    seg_start = ctx["seg_start"]
    fwd = ctx["fwd"]
    marker = ctx["marker"]

    if marker is not None:
        assert new_pts[-1] == marker, "route must terminate at the same-color marker"
    assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

    step_classes = []
    for j in range(len(new_pts) - 1):
        dr = new_pts[j + 1][0] - new_pts[j][0]
        dc = new_pts[j + 1][1] - new_pts[j][1]
        fwd_comp = dr * fwd[0] + dc * fwd[1]
        perp_comp = dr * fwd[1] + dc * (-fwd[0])
        assert fwd_comp > 0, f"tail moved backward or sideways: {(dr, dc)}"
        if abs(dr) + abs(dc) == 1:
            step_classes.append("F")
        elif perp_comp > 0:
            step_classes.append("FR")
        else:
            step_classes.append("FL")
    assert all(sc in {"F", "FL", "FR"} for sc in step_classes)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each terminal-attached stroke toward a farther obstacle, wrap its skin, and stop on a same-color marker.

    Input structure:
    - Background plus one or more triples: a straight bar perpendicular to a border, an irregular 1-cell-thick segment
      attached to the bar's inner endpoint, a small obstacle sprite farther inside the grid, and one isolated singleton
      marker sharing the segment's color.

    Rule:
    - For each bar, keep the nearest foreign obstacle component fixed.
    - Detach the attached segment, translate its root toward that obstacle, follow obstacle-adjacent skin cells, then
      continue only along a monotone forward ray that reaches the same-color isolated marker exactly.
    - The marker becomes the new endpoint and the moved segment keeps the original segment pixel count.

    Key insight:
    - The obstacle still determines where the rope-like wrap happens, but the distant singleton of the same color now
      determines where the wrapped route is allowed to finish; the output is the unique long forward wrap that ends on
      that marker without changing obstacle or bar pixels.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moves = _planned_moves(g, stop_at_marker=True)
    for ctx in moves:
        _validate_route(ctx)

    bg = get_bg(g)
    for ctx in moves:
        for r, c in ctx["seg_comp"]:
            out[r, c] = bg
    for ctx in moves:
        for r, c in ctx["new_pts"]:
            if int(out[r, c]) == bg:
                out[r, c] = ctx["seg_col"]
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 18, 30
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 11
SEG_LEN_MIN, SEG_LEN_MAX = 10, 16
APPROACH_MIN, APPROACH_MAX = 3, 6
ALONG_MIN, ALONG_MAX = 1, 3
TAIL_MIN = 2
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 8
OBS_MD_MIN, OBS_MD_MAX = 5, 12
INPUT_TURN_MIN, INPUT_TURN_MAX = 3, 7
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 1000
PLACEMENT_PAD = 3
TAIL_MODES = ("straight", "outer_diag")

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
    # Keep the input stroke inside the grid by sending it forward into the interior,
    # while biasing diagonal runs away from the obstacle side.
    diag_step = (1, -side)
    orth_step = (1, 0)
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
    seg_len: int, side: int, p1: int, p2: int, tail_mode: str
) -> list[tuple[int, int]]:
    coords = [(0, side)]
    p3 = seg_len - 1 - p1 - p2
    tail_step = (1, 0) if tail_mode == "straight" else (1, side)
    phase_steps = [(1, side), (1, 0), tail_step]
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


def _is_8adj_point(point: tuple[int, int], coords: list[tuple[int, int]]) -> bool:
    pr, pc = point
    for r, c in coords:
        if max(abs(pr - r), abs(pc - c)) <= 1:
            return True
    return False


def _local_marker_isolated(
    marker: tuple[int, int],
    bar: list[tuple[int, int]],
    obstacle: list[tuple[int, int]],
    input_seg: list[tuple[int, int]],
) -> bool:
    occupied = set(bar) | set(obstacle) | set(input_seg)
    if marker in occupied:
        return False
    if _is_8adj_point(marker, input_seg):
        return False
    for nr, nc in neighbors4(marker[0], marker[1]):
        if (nr, nc) in occupied:
            return False
    return True


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None
    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)

    p1_hi = min(APPROACH_MAX, seg_len - 1 - ALONG_MIN - TAIL_MIN)
    if p1_hi < APPROACH_MIN:
        return None
    p1 = random.randint(APPROACH_MIN, p1_hi)
    p2_hi = min(ALONG_MAX, seg_len - 1 - p1 - TAIL_MIN)
    if p2_hi < ALONG_MIN:
        return None
    p2 = random.randint(ALONG_MIN, p2_hi)

    side = random.choice((-1, 1))
    tail_mode = random.choice(TAIL_MODES)
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    output_seg = _build_output_path(seg_len, side, p1, p2, tail_mode)
    seg_set = set(output_seg)
    marker = output_seg[-1]

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
    if max_turns < INPUT_TURN_MIN:
        return None
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)

    if _is_4adj(input_seg, obstacle):
        return None
    if not _local_marker_isolated(marker, bar, obstacle, input_seg):
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
        "marker": [marker],
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
        "tail_mode": tail_mode,
        "obstacle_md": nearest_md,
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
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "marker": _transform(triple["marker"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "obstacle_md": triple["obstacle_md"],
        }
        return placed
    return None


def _nearest_md(origin: tuple[int, int], coords: list[tuple[int, int]]) -> int:
    return min(abs(r - origin[0]) + abs(c - origin[1]) for r, c in coords)


def _sample_moves_differ(inp: np.ndarray) -> bool:
    marker_moves = _planned_moves(inp, stop_at_marker=True)
    for mctx in marker_moves:
        try:
            free_pts = _route_free_tail(
                mctx["shifted"],
                mctx["entry"],
                mctx["skin"],
                mctx["walkable"],
                mctx["seg_len"],
                mctx["terminal"],
                mctx["obstacle_set"],
            )
        except Exception:
            continue
        if mctx["new_pts"] == free_pts:
            return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    max_triples = 1 if n < 24 else 2
    triple_count = random.randint(1, max_triples)

    palette = random.sample(range(10), 3 + triple_count)
    bg = palette[0]
    bar_color = palette[1]
    obstacle_color = palette[2]
    seg_colors = palette[3:]

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge, seg_color in zip(random.sample(list(EDGE_DATA), triple_count), seg_colors):
        triple = _make_triple(n)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)
        paint_coords(inp, placed["marker"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["output_seg"], seg_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed["seg_color"] = seg_color
        placed_triples.append(placed)

    if len(placed_triples) > 1:
        for i, placed in enumerate(placed_triples):
            tr = placed["terminal"]
            own = _nearest_md(tr, placed["obstacle"])
            foreign = []
            for j, other in enumerate(placed_triples):
                if j == i:
                    continue
                foreign.extend(other["obstacle"])
                foreign.extend(other["input_seg"])
                foreign.extend(other["marker"])
            if own >= _nearest_md(tr, foreign):
                return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    if not _sample_moves_differ(inp):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
