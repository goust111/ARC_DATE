# Combo variant of 88bcf3b4 (index 3)
# combo_spec: D5/3+D8/0
# Name: 颜色配对标记终止

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# spec_variant:
#   base_task: 88bcf3b4
#   combo: D5/3 + D8/0
#   chosen_color_pairing_realization:
#     - To keep the color rule visually inferable, this combo uses the clearest
#       legal D8/0 instantiation: each bar and its target obstacle share the
#       same color.
#     - Segment/marker colors stay separate from bar/obstacle colors.
#   visible_objects:
#     - bar: unchanged geometry; straight line from border into the grid.
#     - segment: unchanged input geometry; irregular 1-cell-thick attached path.
#     - obstacle: now intentionally reuses the bar color, so solver cannot rely
#       on "obstacles always have a third color distinct from the bar".
#     - marker: isolated singleton pixel of the segment color; it becomes the
#       segment endpoint in the output.
#   fused_behavior:
#     - Step 1: identify the movable segment and its same-color singleton marker.
#     - Step 2: choose the obstacle whose color matches the bar color, not the
#       nearest obstacle in the whole grid.
#     - Step 3: reroute the segment to that obstacle, travel on obstacle-adjacent
#       skin cells, then choose a monotone forward ray that hits the marker and
#       stop exactly there.
#
# a) Concrete behavior trace
#   Local sketch (origin = bar terminal, forward = +r):
#     input:
#       bar color A at [(-4,0),(-3,0),(-2,0),(-1,0),(0,0)]
#       segment color B starts at (0,1) and meanders backward
#       obstacle of color A sits near the future wrap apex
#       a nearer foreign obstacle of color C exists elsewhere
#       isolated marker of color B sits at (6,2)
#     original task:
#       the segment would pick the nearest obstacle regardless of color, wrap it,
#       then keep extending by the last skin direction until length runs out.
#     combo task:
#       the segment ignores the nearer color-C obstacle, wraps the color-A
#       obstacle because it matches the bar color, and then stops exactly on the
#       color-B marker.
#     difference:
#       both the obstacle selection and the tail destination change. The marker
#       only matters after the correct color-paired obstacle has been chosen.
#
# b) Complexity calibration
#   - The D8 rule is restricted to same-color pairing instead of an arbitrary
#     hidden permutation. This keeps the color logic visible in 2-4 examples.
#   - Each triple uses a unique bar/obstacle color and a unique segment/marker
#     color. That prevents ambiguity while still staying well within the 10-color
#     ARC palette.
#   - Generator uses 2-3 triples, not 1, so there is always at least one foreign
#     obstacle that could have been chosen under a "nearest obstacle" rule.
#
# c) Old implicit assumptions that no longer hold
#   - Old solver assumed obstacle colors were distinct from the bar color. False:
#     obstacle selection must now work at the component level and explicitly
#     exclude the actual bar component.
#   - Old solver assumed the segment color formed one connected component. False:
#     the same color also appears as an isolated singleton marker.
#   - Old solver assumed the free tail direction came from the last boundary
#     step. False: after wrapping, the route is constrained by the marker.
#   - Because bar and obstacle now share a color, generator must prevent them
#     from even diagonal-touching; otherwise 8-connectivity would merge them.
#
# d) Placement legality
#   - Obstacle is grown 4-connected and checked to be 8-separated from the bar.
#   - Marker is 8-separated from every other input object, so it reads as a
#     detached destination cue rather than a pixel incidentally touching an
#     obstacle corner.
#   - Reserved placement boxes include the output route footprint and the marker,
#     so different triples stay spatially separated and do not create accidental
#     same-color merges.
#   - Output validity is oracle-checked by running main() on the generated input.
#
# e) Visual observability
#   - Same-color bar/obstacle pairing is immediately visible from shape roles:
#     a border-attached straight component and an interior sprite reuse the same
#     color.
#   - The marker is a detached singleton in the input and becomes the endpoint in
#     the output, so its role is also visible.
#   - The generator always includes a foreign obstacle because it uses 2 groups.
#     Post-test metrics measure how often color pairing overrides a nearest-
#     obstacle choice; this is tracked statistically rather than enforced on
#     every single sample to keep placement feasible.
#
# f) Visual cleanliness
#   - Shared-color pairing could become messy if components touch, so the
#     generator enforces 8-separation between the bar and obstacle of the same
#     color.
#   - Markers remain single pixels with padding from other objects.
#   - Obstacles are still small sprites, and the moved segment never overwrites
#     them.
#
# g) Change inferability
#   - No invisible ordering on color ids is used. The only color logic is
#     geometric and human-visible: "straight border bar matches interior obstacle
#     of the same color", and "segment matches singleton marker of the same
#     color".
#   - Multi-triple examples repeat this pattern with different colors, so the
#     rule can be generalized from structure rather than from numeric ids.
#
# h) Generator oracle strategy
#   - Output-first: build the wrapped output route and marker endpoint first,
#     then derive an unrelated-looking input stroke of the same length.
#   - Reject attempts unless:
#       1. main(input) reproduces the intended output exactly,
#       2. at least one triple uses a straight marker-directed tail so the marker
#          visibly changes the old free-diagonal continuation.
#   - The D8 effect is then audited over a batch by measuring how often the
#     intended same-color obstacle is not the nearest obstacle.
#
# i) New solver assumptions audited against generator
#   - For each bar color, there is exactly one other 8-connected component of the
#     same color, namely the target obstacle. Generator guarantees unique
#     bar/obstacle colors across triples.
#   - For each segment color, there is exactly one attached stroke component and
#     exactly one isolated singleton marker. Generator enforces this.
#   - There exists a skin exit and forward ray that reaches the marker with the
#     required length. Generator constructs that target output and verifies it by
#     oracle execution.
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
    length: int,
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
        if len(prefix) > length:
            continue
        for dir_idx, step in enumerate(_positive_forward_dirs(fwd)):
            tail = _ray_to_marker(exit_cell, marker, step, walkable)
            if tail is None:
                continue
            route = prefix + tail[1:]
            if len(route) != length:
                continue
            if len(set(route)) != len(route):
                continue
            tail_len = len(tail) - 1
            key = (tail_len, -len(boundary_path), dir_idx, exit_cell[0], exit_cell[1])
            candidates.append((key, route))

    assert candidates, f"no obstacle-exit route reaches marker {marker} with segment length {length}"
    return min(candidates, key=lambda x: x[0])[1]


def _matching_obstacle_component(
    g: np.ndarray,
    bar_col: int,
    bar_comp: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    bar_set = set(bar_comp)
    comps = [set(comp) for comp in _components(g, bar_col, conn8=True) if set(comp) != bar_set]
    assert len(comps) == 1, f"expected exactly one same-color obstacle for bar color {bar_col}, got {len(comps)}"
    return comps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rehang each irregular segment around the obstacle that shares its bar color, then stop on the matching marker.

    Input structure:
    - Background is the most frequent color.
    - Each active group has a straight bar on a border, an attached irregular segment of another color,
      an interior obstacle sprite that reuses the bar color, and one isolated singleton marker that reuses
      the segment color.

    Rule:
    - Identify the movable segment attached to each bar terminal and its same-color singleton marker.
    - Ignore nearer wrong-color obstacles; instead choose the obstacle component whose color matches the bar.
    - Translate the segment toward that obstacle, follow obstacle-adjacent skin cells, then continue along a
      monotone forward ray that reaches the same-color marker and stop exactly there.

    Key insight: Two independent color correspondences must be read together: bar color selects which obstacle
    to wrap, and segment color selects where the rerouted segment ends.
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
                length = len(path)
                key = (length, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (length, _, _, _, _), seg_col, seg_start, seg_comp = best_seg
        marker = _isolated_marker(g, seg_col, bg, seg_comp)
        obstacle_set = _matching_obstacle_component(g, bar_col, bar_comp)

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
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay within grid"
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        fwd = _bar_forward(bar_comp)
        new_pts = _route_to_marker(shifted, entry, skin, marker, walkable, length, fwd)

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

        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _seg_col, _new_pts in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _seg_comp, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 19, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 5
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 8
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 12000
PLACEMENT_PAD = 1
TAIL_MODES = ("straight", "outer_diag")

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

ADJ_EDGE_PAIRS = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
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
        last_was_diag = (step == diag_step)
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _build_output_path(
    seg_len: int, side: int, p1: int, p2: int, p3: int, tail_mode: str
) -> list[tuple[int, int]]:
    coords = [(0, side)]
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


def _is_8_separate(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    for r1, c1 in coords_a:
        for r2, c2 in coords_b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _local_marker_isolated(
    marker: tuple[int, int],
    bar: list[tuple[int, int]],
    obstacle: list[tuple[int, int]],
    input_seg: list[tuple[int, int]],
) -> bool:
    if marker in set(bar) | set(obstacle) | set(input_seg):
        return False
    if _is_8adj_point(marker, bar):
        return False
    if _is_8adj_point(marker, obstacle):
        return False
    if _is_8adj_point(marker, input_seg):
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
    side = random.choice((-1, 1))
    tail_mode = random.choice(TAIL_MODES)

    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    if remaining < 4:
        return None
    p1 = random.randint(2, min(PHASE_MAX, remaining - 2))
    remaining -= p1
    p2 = random.randint(1, min(PHASE_MAX, remaining - 1))
    p3 = remaining - p2
    if p3 < 1:
        return None

    output_seg = _build_output_path(seg_len, side, p1, p2, p3, tail_mode)
    seg_set = set(output_seg)
    marker = output_seg[-1]

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    marker_halo = {(marker[0] + dr, marker[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if apex_inner in marker_halo:
        return None
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set | marker_halo)
    if not _is_8_separate(bar, obstacle):
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
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
        "tail_mode": tail_mode,
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
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "marker": _transform(triple["marker"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "tail_mode": triple["tail_mode"],
        }
        mr, mc = placed["marker"][0]
        if any(0 <= nr < n and 0 <= nc < n and reserved[nr, nc] for nr, nc in neighbors4(mr, mc)):
            continue
        if not _is_8_separate(placed["bar"], placed["obstacle"]):
            continue
        return placed
    return None


def _candidate_placements(triple: dict, edge: str, n: int) -> list[dict]:
    placements = []
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]
    positions = list(range(n))
    random.shuffle(positions)
    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "marker": _transform(triple["marker"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "tail_mode": triple["tail_mode"],
        }
        if not _is_8_separate(placed["bar"], placed["obstacle"]):
            continue
        placements.append(placed)
    return placements


def _reserve_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_pair_placements(triples: list[dict], edges: tuple[str, str], n: int) -> list[dict] | None:
    cand0 = _candidate_placements(triples[0], edges[0], n)
    cand1 = _candidate_placements(triples[1], edges[1], n)
    if not cand0 or not cand1:
        return None
    for placed0 in cand0:
        for placed1 in cand1:
            if _reserve_overlap(placed0["reserve"], placed1["reserve"]):
                continue
            pair = [placed0, placed1]
            if _pairing_override_count(pair) < 1:
                continue
            return pair
    return None


def _nearest_md(origin: tuple[int, int], coords: list[tuple[int, int]]) -> int:
    return min(abs(r - origin[0]) + abs(c - origin[1]) for r, c in coords)


def _pairing_override_count(placed_triples: list[dict]) -> int:
    all_obstacles = [pt["obstacle"] for pt in placed_triples]
    count = 0
    for i, placed in enumerate(placed_triples):
        own = _nearest_md(placed["terminal"], placed["obstacle"])
        nearest_any = min(_nearest_md(placed["terminal"], obs) for obs in all_obstacles)
        if own > nearest_any:
            count += 1
    return count


def _straight_tail_count(placed_triples: list[dict]) -> int:
    return sum(1 for placed in placed_triples if placed["tail_mode"] == "straight")


def _gen_attempt() -> dict | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    triple_count = 2

    palette = random.sample(range(10), 1 + 2 * triple_count)
    bg = palette[0]
    pair_colors = palette[1 : 1 + triple_count]
    seg_colors = palette[1 + triple_count :]

    triples = []
    for _ in range(triple_count):
        triple = _make_triple(n)
        if triple is None:
            return None
        triples.append(triple)

    if all(triple["tail_mode"] != "straight" for triple in triples):
        return None

    edge_pairs = list(ADJ_EDGE_PAIRS)
    random.shuffle(edge_pairs)
    placed_triples = None
    for edges in edge_pairs:
        placed_triples = _find_pair_placements(triples, edges, n)
        if placed_triples is not None:
            break
    if placed_triples is None:
        return None

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)

    for placed, pair_color, seg_color in zip(placed_triples, pair_colors, seg_colors):
        paint_coords(inp, placed["bar"], pair_color)
        paint_coords(inp, placed["obstacle"], pair_color)
        paint_coords(inp, placed["input_seg"], seg_color)
        paint_coords(inp, placed["marker"], seg_color)

        paint_coords(out, placed["bar"], pair_color)
        paint_coords(out, placed["obstacle"], pair_color)
        paint_coords(out, placed["output_seg"], seg_color)

        placed["pair_color"] = pair_color
        placed["seg_color"] = seg_color
    if _straight_tail_count(placed_triples) < 1:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, out):
        return None

    return {
        "inp": inp,
        "out": out,
        "placed_triples": placed_triples,
        "pairing_override": _pairing_override_count(placed_triples),
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pack = _gen_attempt()
        if pack is not None:
            return pack["inp"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
