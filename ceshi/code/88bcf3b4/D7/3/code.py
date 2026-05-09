# Variant of 88bcf3b4
# Dimension: D7 — 变换语义 (transform)
# Change: 输入中的原始线段消失（涂为背景色），新路径沿绕行轨迹重新绘制线段色；线段 detach-reattach 但 bar 和障碍物不变。→ 障碍物被包围涂色：绕行路径保留为外轮廓，线段与障碍物之间的一层空隙也被填成线段色。

# Design analysis for the D7 variant
# - concreteization:
#   input keeps the original object vocabulary: one bar on a border, one 1-cell-thick irregular segment attached at the
#   terminal, one nearby obstacle sprite. Output still removes the input segment and keeps the bar/obstacle unchanged.
#   The new part is that the rerouted segment is no longer just a thin line: after the rope-like route is found, we also
#   paint the one-cell-wide inner band between that route and the obstacle-facing side. Visually, the old output looked
#   like a single bent stroke; the new output looks like a thicker wrapped ribbon with the obstacle left as a hole.
#   Small local example in terminal-relative coordinates for side=+1:
#     original output path:  (0,1),(1,2),(2,3),(3,3),(4,2)
#     inner band cells:      (1,1),(2,2),(3,2),(4,1)   [skip (0,0) because it is the bar terminal]
#     if obstacle occupies some of those inner-band cells, the remaining cells are exactly the visible "filled gap".
# - behavior tracing:
#   old main(): compute reroute path `new_pts`, clear the original segment, repaint only `new_pts`.
#   new main(): compute the same reroute path `new_pts`, then also add the obstacle-facing one-cell lateral offset of
#   that path (when the offset cell is not bar/obstacle and stays in-bounds). This changes the transform stage only.
# - complexity calibration:
#   to keep the new behavior visible, generator forbids degenerate 3-phase paths: p1>=2, p2>=1, p3>=1. That guarantees
#   a readable V/alongside/exit structure and leaves enough room for nontrivial filled pixels. Obstacle growth is also
#   constrained to the inner band of the designed route so the new filled area is consistently adjacent to the obstacle
#   instead of drifting elsewhere and becoming visually ambiguous.
# - implicit assumption audit against the original solver:
#   `_segment_degrees`, `_segment_main_path`, bar detection, nearest-obstacle search, and reroute path search still rely
#   on the same old assumptions, and the generator preserves them.
#   The old final repaint loop assumed "one output path cell per preserved segment pixel"; that assumption no longer
#   holds, so repaint now uses a computed region `fill_pts`, not only the route polyline.
#   The old generator reserved space using `output_seg`; that is no longer enough because the new output is thicker, so
#   the reserved bbox now includes `output_region`.
# - placement legality:
#   generator computes `output_region` before placement, reserves the expanded bbox of bar+obstacle+input_seg+output_region,
#   and validates the whole input by calling `main(inp)` and comparing against the intended `out`.
# - visual observability:
#   the new rule is human-visible because the output gains extra segment-colored pixels next to the obstacle. Generator
#   rejects triples whose extra filled area is too small, so samples do not collapse back to the original thin-line task.
# - cleanliness:
#   obstacle remains a distinct solid color hole inside/against the thicker segment region, bar stays separate, and
#   triples are padded apart. This avoids unreadable overlaps.
# - color inference:
#   no color remapping is introduced; the only color change is structural use of the existing segment color to fill a
#   surrounding band. That is directly inferable from 2-4 examples.

from __future__ import annotations

from collections import deque
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


def _surround_fill_cells(
    route: list[tuple[int, int]],
    terminal: tuple[int, int],
    seg_start: tuple[int, int],
    obstacle_set: set[tuple[int, int]],
    bar_set: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    off_r = seg_start[0] - terminal[0]
    off_c = seg_start[1] - terminal[1]
    assert abs(off_r) + abs(off_c) == 1, "segment must attach orthogonally to the bar terminal"
    inner = (-off_r, -off_c)

    fill = set(route)
    for r, c in route:
        nr, nc = r + inner[0], c + inner[1]
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in obstacle_set or (nr, nc) in bar_set:
            continue
        fill.add((nr, nc))
    return sorted(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing an irregular line segment around a nearby obstacle sprite, then fill the obstacle-facing gap.

    Input structure:
    - bg
    - bar color: straight line perpendicular to a border, one end on the border
    - segment color: irregular 1-cell-thick stroke attached at the bar terminal
    - obstacle color: nearby sprite that the segment reroutes around

    Rule:
    - Keep the original bar and obstacle unchanged.
    - Remove the input segment from its original location.
    - Recompute the same monotone reroute path used by the base task.
    - Paint not only that outer reroute path, but also the one-cell inner band on the obstacle-facing side.

    Key insight: the line still behaves like a rope wrapping the obstacle, but the output is now a wrapped ribbon/filled
    surround rather than a bare polyline.
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
                l_seg = len(path)
                key = (l_seg, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (l_seg, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

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
        while len(route) < l_seg:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:l_seg]
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

        fill_pts = _surround_fill_cells(new_pts, terminal, seg_start, obstacle_set, set(bar_comp), h, w)
        moves.append((seg_comp, seg_col, fill_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, fill_pts in moves:
        for r, c in fill_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
PHASE1_MIN = 2
PHASE2_MIN = 1
PHASE3_MIN = 1
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 5
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PLACEMENT_PAD = 2
MIN_EXTRA_FILL = 2

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


def _grow_obstacle_in_allowed(
    start: tuple[int, int], target_size: int, allowed: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    assert start in allowed, "obstacle seed must be inside the allowed corridor"
    obstacle = {start}
    for _ in range(target_size - 1):
        cands = sorted(
            {
                (nr, nc)
                for r, c in obstacle
                for nr, nc in neighbors4(r, c)
                if (nr, nc) in allowed and (nr, nc) not in obstacle
            }
        )
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _local_inner_band(output_seg: list[tuple[int, int]], side: int, bar: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bar_set = set(bar)
    band = []
    seen = set()
    for r, c in output_seg:
        q = (r, c - side)
        if q in bar_set or q in seen:
            continue
        band.append(q)
        seen.add(q)
    return band


def _local_surround_region(
    output_seg: list[tuple[int, int]],
    side: int,
    obstacle: list[tuple[int, int]],
    bar: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    obstacle_set = set(obstacle)
    bar_set = set(bar)
    region = set(output_seg)
    for r, c in output_seg:
        q = (r, c - side)
        if q in obstacle_set or q in bar_set:
            continue
        region.add(q)
    return sorted(region)


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

    remaining = seg_len - 1
    p1_hi = min(PHASE_MAX, remaining - PHASE2_MIN - PHASE3_MIN)
    if p1_hi < PHASE1_MIN:
        return None
    p1 = random.randint(PHASE1_MIN, p1_hi)
    remaining -= p1
    p2_hi = min(PHASE_MAX, remaining - PHASE3_MIN)
    if p2_hi < PHASE2_MIN:
        return None
    p2 = random.randint(PHASE2_MIN, p2_hi)
    p3 = remaining - p2
    if p3 < PHASE3_MIN:
        return None

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    corridor = set(_local_inner_band(output_seg, side, bar))
    if apex_inner not in corridor:
        return None
    obs_cap = min(OBS_SIZE_MAX, max(1, len(corridor) - 1))
    if obs_cap < OBS_SIZE_MIN:
        return None
    obs_size = random.randint(OBS_SIZE_MIN, obs_cap)
    obstacle = _grow_obstacle_in_allowed(apex_inner, obs_size, corridor)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None

    output_region = _local_surround_region(output_seg, side, obstacle, bar)
    if len(output_region) - len(seg_set) < MIN_EXTRA_FILL:
        return None

    union = bar + obstacle + input_seg + output_region
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "output_region": output_region,
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
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_region"]
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
            "output_region": _transform(triple["output_region"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
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

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
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

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["output_region"], seg_color)

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
