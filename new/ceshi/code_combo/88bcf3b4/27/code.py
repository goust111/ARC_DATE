# Combo variant of 88bcf3b4 (index 27)
# combo_spec: D3/1+D4/0+D9/0
# Name: straight-source orthogonal dashed wrap
#
# Design notes
# - D4/0: the source attached to the bar terminal is a straight 4-connected line.
# - D9/0: the wrapped route has A*F*O* geometry: diagonal approach, forward skin run, then orthogonal side exit.
# - D3/1: only every other cell of that full route is painted, starting from the attachment cell.
# - The generator is output-first and reserves the full hidden route corridor so other triples do not occupy dash gaps.

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


def _bar_forward(comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


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


def _bfs_path4(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
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


def _straight_segment_info(
    comp: list[tuple[int, int]], terminal: tuple[int, int], bar_vertical: bool
) -> tuple[int, tuple[int, int], list[tuple[int, int]]] | None:
    if len(comp) < 2 or not _is_straight_line(comp):
        return None
    e0, e1 = _bar_endpoints(comp)
    endpoints = [p for p in (e0, e1) if abs(p[0] - terminal[0]) + abs(p[1] - terminal[1]) == 1]
    if len(endpoints) != 1:
        return None
    seg_start = endpoints[0]
    off_r = seg_start[0] - terminal[0]
    off_c = seg_start[1] - terminal[1]
    if bar_vertical:
        if off_r != 0 or abs(off_c) != 1:
            return None
    else:
        if off_c != 0 or abs(off_r) != 1:
            return None
    return len(comp), seg_start, comp


def _dashed_points(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [path[i] for i in range(0, len(path), 2)]


def main(grid: np.ndarray) -> np.ndarray:
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
                vertical = len({c for _, c in comp}) == 1
                border_ep = e0 if touch0 else e1
                if vertical and border_ep[0] not in (0, h - 1):
                    continue
                if (not vertical) and border_ep[1] not in (0, w - 1):
                    continue
                terminal = e1 if touch0 else e0
                tr, tc = terminal
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        bar_vertical = len({c for _, c in bar_comp}) == 1
        fwd = _bar_forward(bar_comp)

        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=False):
                if cand_start not in comp:
                    continue
                info = _straight_segment_info(comp, terminal, bar_vertical)
                if info is None:
                    break
                seg_len, seg_start, seg_comp = info
                key = (seg_len, cand_col, seg_start[0], seg_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, seg_start, seg_comp)
                break
        assert best_seg is not None, f"no valid straight segment for bar at {terminal}"
        (seg_len, _, _, _), seg_col, seg_start, seg_comp = best_seg

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
        side = (off_r, off_c)
        assert abs(off_r) + abs(off_c) == 1
        assert off_r * fwd[0] + off_c * fwd[1] == 0

        walkable = g == bg
        walkable[terminal] = True

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if int(g[er, ec]) != bg:
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
        assert shifted and shifted[0] == seg_start
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted)
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        def skin_ok(p: tuple[int, int]) -> bool:
            return int(g[p[0], p[1]]) == bg

        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True

        def exit_key(p: tuple[int, int]) -> tuple[int, int, int, int, int]:
            r, c = p
            forward = (r - terminal[0]) * fwd[0] + (c - terminal[1]) * fwd[1]
            lateral = (r - terminal[0]) * side[0] + (c - terminal[1]) * side[1]
            return (forward, lateral, -abs(lateral), -r, -c)

        reachable = []
        for p in [q for q in skin if skin_ok(q)] + [entry]:
            path = _bfs_path4(entry, p, skin_walkable)
            if path is not None:
                reachable.append((exit_key(p), p, path))
        assert reachable, "no reachable obstacle skin cells"
        _, _, boundary_path = max(reachable, key=lambda item: item[0])

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < seg_len:
            nr, nc = cur[0] + side[0], cur[1] + side[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:seg_len]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        dashed_pts = _dashed_points(new_pts)

        step_classes = []
        for j in range(len(new_pts) - 1):
            dr = new_pts[j + 1][0] - new_pts[j][0]
            dc = new_pts[j + 1][1] - new_pts[j][1]
            fwd_comp = dr * fwd[0] + dc * fwd[1]
            side_comp = dr * side[0] + dc * side[1]
            man = abs(dr) + abs(dc)
            if fwd_comp == 1 and side_comp == 1 and man == 2:
                step_classes.append("A")
            elif fwd_comp == 1 and side_comp == 0 and man == 1:
                step_classes.append("F")
            elif fwd_comp == 0 and side_comp == 1 and man == 1:
                step_classes.append("O")
            else:
                step_classes.append("X")
        assert re.fullmatch(r"A*F*O*", "".join(step_classes or [""])), f"output segment not monotone L-wrap: {step_classes}"
        moves.append((seg_comp, seg_col, dashed_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, dashed_pts in moves:
        for r, c in dashed_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 3, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
MAX_RETRIES = 500
PLACEMENT_PAD = 1

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


def _build_straight_input_path(seg_len: int, side: int) -> list[tuple[int, int]]:
    return [(0, side * k) for k in range(1, seg_len + 1)]


def _build_output_path(seg_len: int, side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    for phase_len, step in zip((p1, p2, p3), ((1, side), (1, 0), (0, side))):
        dr, dc = step
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    assert len(coords) == seg_len
    return coords


def _expanded_4neighbors(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            out.add((nr, nc))
    return out


def _obstacle_is_observable(obstacle: list[tuple[int, int]]) -> bool:
    if len(obstacle) < 3:
        return False
    rs = {r for r, _ in obstacle}
    cs = {c for _, c in obstacle}
    return len(rs) > 1 and len(cs) > 1


def _grow_obstacle(
    base: list[tuple[int, int]], max_size: int, forbidden: set[tuple[int, int]], side: int
) -> list[tuple[int, int]]:
    obstacle = set(base)
    start_r = min(r for r, _ in base)
    end_r = max(r for r, _ in base)
    anchor_c = base[0][1]
    while len(obstacle) < max_size:
        cands = []
        for r, c in list(obstacle):
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in obstacle or (nr, nc) in forbidden:
                    continue
                if not (start_r <= nr <= end_r):
                    continue
                if (nc - anchor_c) * side > 0:
                    continue
                if abs(nc - anchor_c) > 2:
                    continue
                cands.append((nr, nc))
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1, n - 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    if seg_len < 5:
        return None

    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)
    input_seg = _build_straight_input_path(seg_len, side)
    input_buffer = _expanded_4neighbors(input_seg)

    remaining = seg_len - 1
    p1_max = min(PHASE_MAX, remaining - 2)
    if p1_max < 1:
        return None
    p1 = random.randint(1, p1_max)
    remaining -= p1

    p2_max = min(PHASE_MAX, remaining - 1)
    if p2_max < 1:
        return None
    p2 = random.randint(1, p2_max)
    p3 = remaining - p2
    if p3 < 1:
        return None

    output_route = _build_output_path(seg_len, side, p1, p2, p3)
    output_seg = _dashed_points(output_route)
    route_set = set(output_route)

    apex = output_route[p1]
    obstacle_base = [(apex[0] + k, apex[1] - side) for k in range(p2 + 1)]
    if any(p in route_set or p in bar_set or p in input_buffer for p in obstacle_base):
        return None

    obs_size_min = max(OBS_SIZE_MIN, len(obstacle_base))
    if obs_size_min > OBS_SIZE_MAX:
        return None
    obs_size = random.randint(obs_size_min, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(obstacle_base, obs_size, route_set | bar_set | input_buffer, side)
    if not _obstacle_is_observable(obstacle):
        return None
    if any(p in input_buffer for p in obstacle):
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    union = bar + obstacle + input_seg + output_route
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_route": output_route,
        "output_seg": output_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_route"]
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
            "output_route": _transform(triple["output_route"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    max_triples = 1 if n < 11 else 2 if n < 15 else 3
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
        paint_coords(out, placed["output_seg"], seg_color)

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
