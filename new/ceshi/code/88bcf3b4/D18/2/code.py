# Variant of 88bcf3b4
# Dimension: D18 — 符号编码参数 (symbol_instructions)
# Change: 当前任务没有符号编码参数，所有规则由几何关系和固定常数隐式决定。... → 边缘标记指示退出方向

# Design analysis for the D18 variant
# -----------------------------------
# spec_variant:
#   preserved_structure:
#     grid: square, now effectively 10-27 to leave room for a legend in the top-left corner
#     groups: 1-3 bar+segment+obstacle triples
#     bar: same as original, straight, border-anchored, perpendicular to the touched edge
#     segment_input: same as original, irregular 1-cell-thick stroke attached to the bar terminal
#     obstacle: same as original, small nearby irregular sprite that the segment wraps around
#   new_symbol_instruction:
#     location: fixed 3x3 legend patch at the top-left border
#     colors: two extra colors distinct from bg/bar/segment/obstacle
#     shape:
#       - patch is a solid 3x3 rectangle of legend fill color
#       - center ink pixel is always at (1,1)
#       - a second ink pixel indicates the exit mode
#     modes:
#       diag: tip at (2,2)  -> keep the original diagonal exit
#       forward: tip at (2,1) -> exit straight in the bar-forward direction
#       orth: tip at (1,2) -> exit orthogonally using the lateral component that the old diagonal exit used
#   behavior_trace:
#     original:
#       after the segment reaches the obstacle skin, solver extends it using the last skin step, which
#       implicitly fixes a 45-degree diagonal-style departure in the generated data
#     variant:
#       the approach-to-obstacle and skin-following logic stay the same, but the final extension vector
#       is no longer read from the skin geometry; it is selected from the legend mode instead
#     concrete_local_example:
#       assume a top-edge bar, terminal at local (0,0), attached segment starts at (0,+1)
#       original exit phase: ...(p1+p2, p1+1) -> (p1+p2+1, p1)
#       forward mode: ...(p1+p2, p1+1) -> (p1+p2+1, p1+1)
#       orth mode: ...(p1+p2, p1+1) -> (p1+p2, p1)
#       difference: the visible tail after wrapping is now controlled by the legend, not fixed geometry
#   complexity_calibration:
#     - only 3 exit modes are allowed; this is enough to make the symbol visible without making the rule too open-ended
#     - generator forces p3 >= 1 so the legend always changes at least one painted output step
#     - for orth mode generator also forces p2 >= 1 so the orth tail does not degenerate into the obstacle seed cell
#   implicit_assumption_review:
#     - original code assumed "last skin step == desired exit direction"; this is removed
#     - original color scan would treat legend colors as obstacles; solver now reads the legend first and excludes its colors
#     - original monotone-shape assert only allowed FL*F*FR* / FR*F*FL*; orth mode violates that, so validation now focuses
#       on obstacle avoidance and generator-side construction instead of the old regex
#     - extension can now be orthogonal, so solver additionally blocks self-revisits during the extension loop
#   placement_legality:
#     - generator reserves a 4x4 corner region for the legend (3x3 patch + 1-cell moat)
#     - triples are still placed with padded bounding boxes and validated by main(input) == output
#   visual_observability:
#     - all triples in a grid share the same legend, so the symbol has repeated evidence within a sample
#     - the three modes are visible as three different tail behaviors after the obstacle: diagonal / straight / lateral
#     - symbol meaning is geometric, not based on color values; legend colors are arbitrary and only serve visibility

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


LEGEND_SIZE = 3
LEGEND_PAD = 1
LEGEND_TIPS = {
    "diag": (2, 2),
    "forward": (2, 1),
    "orth": (1, 2),
}
TIP_TO_MODE = {v: k for k, v in LEGEND_TIPS.items()}


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


def _bar_frame(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return ((1, 0), (0, 1)) if min(rs) == 0 else ((-1, 0), (0, -1))
    return ((0, 1), (-1, 0)) if min(cs) == 0 else ((0, -1), (1, 0))


def _read_symbol_instruction(grid: np.ndarray, bg: int) -> tuple[str, set[int]]:
    patch = np.asarray(grid[:LEGEND_SIZE, :LEGEND_SIZE], dtype=int)
    vals = set(int(v) for v in np.unique(patch))
    assert bg not in vals, "legend patch must be fully painted"
    assert len(vals) == 2, "legend patch must use exactly fill+ink colors"
    counts = {v: int(np.sum(patch == v)) for v in vals}
    fill = max(counts, key=counts.get)
    ink = min(counts, key=counts.get)
    ink_pts = {(int(r), int(c)) for r, c in zip(*np.where(patch == ink))}
    assert len(ink_pts) == 2 and (1, 1) in ink_pts, "legend must contain a center ink cell and one tip"
    tip = next(p for p in ink_pts if p != (1, 1))
    assert tip in TIP_TO_MODE, f"unsupported legend tip location: {tip}"
    return TIP_TO_MODE[tip], {fill, ink}


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

    def neighs(p: tuple[int, int]):
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
        for nxt in neighs(cur):
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


def _extension_dir(
    exit_mode: str, fwd: tuple[int, int], side_vec: tuple[int, int], side_sign: int
) -> tuple[int, int]:
    if exit_mode == "diag":
        return fwd[0] - side_sign * side_vec[0], fwd[1] - side_sign * side_vec[1]
    if exit_mode == "forward":
        return fwd
    if exit_mode == "orth":
        return -side_sign * side_vec[0], -side_sign * side_vec[1]
    raise AssertionError(f"unknown exit mode: {exit_mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing each irregular segment around its nearest obstacle, but use the border legend to choose
    the post-wrap exit mode.

    The top-left 3x3 legend patch contains a fixed center ink pixel and a second tip pixel:
    - down-right tip: diagonal exit
    - down tip: straight-forward exit
    - right tip: orthogonal exit
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    exit_mode, legend_colors = _read_symbol_instruction(g, bg)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg} - legend_colors)
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
                if v != bg and v != bar_col and v not in legend_colors:
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
        fwd, side_vec = _bar_frame(bar_comp)
        side_sign = off_r * side_vec[0] + off_c * side_vec[1]
        assert side_sign in (-1, 1), f"segment must attach one lateral step from the terminal: {off_r, off_c}"

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

        def exit_key(p: tuple[int, int]):
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

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        dr_ext, dc_ext = _extension_dir(exit_mode, fwd, side_vec, side_sign)
        route_set = set(route)
        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in route_set:
                break
            if int(g[nr, nc]) not in (bg, seg_col):
                break
            route.append((nr, nc))
            route_set.add((nr, nc))
            cur = (nr, nc)

        new_pts = route[:L]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
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
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 300
PLACEMENT_PAD = 1
TRIPLE_BUILD_RETRIES = 40

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _legend_patch(fill: int, ink: int, mode: str) -> np.ndarray:
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), fill, dtype=int)
    patch[1, 1] = ink
    tr, tc = LEGEND_TIPS[mode]
    patch[tr, tc] = ink
    return patch


def _legend_colors_from_patch(grid: np.ndarray) -> tuple[int, int]:
    patch = np.asarray(grid[:LEGEND_SIZE, :LEGEND_SIZE], dtype=int)
    vals = set(int(v) for v in np.unique(patch))
    assert len(vals) == 2
    counts = {v: int(np.sum(patch == v)) for v in vals}
    fill = max(counts, key=counts.get)
    ink = min(counts, key=counts.get)
    return fill, ink


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


def _build_output_path(
    seg_len: int, side: int, p1: int, p2: int, p3: int, exit_mode: str
) -> list[tuple[int, int]]:
    coords = [(0, side)]
    exit_step = {
        "diag": (1, -side),
        "forward": (1, 0),
        "orth": (0, -side),
    }[exit_mode]
    for phase_len, step in zip([p1, p2, p3], [(1, side), (1, 0), exit_step]):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
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


def _make_triple(n: int, exit_mode: str) -> dict | None:
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

    min_p2 = 1 if exit_mode == "orth" else 0
    remaining = seg_len - 1
    if remaining < 1 + min_p2 + 1:
        return None
    p1 = random.randint(1, min(PHASE_MAX, remaining - min_p2 - 1))
    remaining -= p1
    p2 = random.randint(min_p2, min(PHASE_MAX, remaining - 1))
    p3 = remaining - p2
    assert p3 >= 1

    output_seg = _build_output_path(seg_len, side, p1, p2, p3, exit_mode)
    seg_set = set(output_seg)
    if len(seg_set) != len(output_seg):
        return None

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

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
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
    positions = list(range(1, n - 1))
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
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt(exit_mode: str | None = None) -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 14:
        max_triples = 1
    elif n < 20:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)
    if exit_mode is None:
        exit_mode = random.choice(list(LEGEND_TIPS))

    bg, bar_color, seg_color, obstacle_color, legend_fill, legend_ink = random.sample(range(10), 6)
    inp = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    reserved[: LEGEND_SIZE + LEGEND_PAD, : LEGEND_SIZE + LEGEND_PAD] = True
    placed_triples = []

    patch = _legend_patch(legend_fill, legend_ink, exit_mode)
    inp[:LEGEND_SIZE, :LEGEND_SIZE] = patch

    for edge in random.sample(list(EDGE_DATA), triple_count):
        placed = None
        for _ in range(TRIPLE_BUILD_RETRIES):
            triple = _make_triple(n, exit_mode)
            if triple is None:
                continue
            placed = _place_triple(triple, edge, n, reserved)
            if placed is not None:
                break
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

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
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, inp):
        return None
    for placed in placed_triples:
        r0, c0, r1, c1 = placed["reserve"]
        before = set(placed["input_seg"])
        after = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(solved[r, c]) == seg_color
        }
        if len(after) != len(before):
            return None
        if after == before:
            return None
    return inp


def generate_input() -> np.ndarray:
    target_mode = random.choice(list(LEGEND_TIPS))
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(target_mode)
        if grid is None:
            continue
        fill, ink = _legend_colors_from_patch(grid)
        candidate = np.array(grid, copy=True)
        candidate[:LEGEND_SIZE, :LEGEND_SIZE] = _legend_patch(fill, ink, target_mode)
        try:
            out = main(candidate)
        except Exception:
            continue
        alt_mode = "diag" if target_mode != "diag" else "forward"
        alt = np.array(candidate, copy=True)
        alt[:LEGEND_SIZE, :LEGEND_SIZE] = _legend_patch(fill, ink, alt_mode)
        try:
            alt_out = main(alt)
        except Exception:
            alt_out = None
        if alt_out is not None and np.array_equal(out, alt_out):
            continue
        return candidate
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
