# Variant of 88bcf3b4
# Dimension: D9 — 对角延伸退出规则 (path)
# Change: 从障碍物边界退出后，沿最后一步的方向（通常为 45 度对角）继续延伸直到像素数耗尽或到达网格边界。... → 按 bar 前进方向退出

"""
Design analysis
===============

design:
  concrete_behavior:
    - Objects stay the same as the base task: one border-perpendicular bar, one attached 1-cell-thick segment,
      and one nearby obstacle sprite.
    - The changed behavior is only the tail after the segment has wrapped to the far side of the obstacle.
    - Old D9 behavior: the tail keeps the last boundary-traversal step, which is often a 45-degree diagonal.
    - New D9 behavior: once the route leaves the obstacle skin, every remaining pixel advances in the bar-forward
      direction `fwd` (from the border into the grid), regardless of the final skin step.
  behavior_trace:
    - Local top-edge example with terminal at (0,0), segment root at (0,1), obstacle seed at (2,2):
      approach can be (0,1)->(1,2)->(2,3), boundary can pivot through (3,2).
    - Old tail keeps (1,-1), so it continues to (4,1),(5,0)...
    - New tail ignores that corner direction and continues with `fwd=(1,0)`, so it becomes (4,2),(5,2)...
    - The difference appears exactly in the suffix after the wrap, so generator examples must force a visible suffix.
  complexity_calibration:
    - A minimal one-line solver edit is not enough, because the old code also assumed `boundary_path` had length >= 2
      only to derive the tail direction. Under the new rule that assumption is unnecessary and can reject valid wraps.
    - The generator therefore only accepts candidates where:
        1. the new solver succeeds,
        2. the old and new exit rules produce different outputs on the same input,
        3. the new output ends with a straight forward suffix of length >= 2,
        4. there is at least one diagonal wrap step before that suffix.
    - This keeps the variant visually observable and avoids degeneration back to the original task.
  implicit_assumptions_audited:
    - `boundary_path[-1] - boundary_path[-2]` is no longer the exit direction. Replaced by bar-derived `fwd`.
    - `len(boundary_path) >= 2` is no longer required for the variant. If entry==exit, the suffix may still extend
      straight forward from that cell.
    - The original regex `(FL)*F*(FR)* | (FR)*F*(FL)*` was tied to a diagonal tail. The variant can end with an extra
      `F*` tail after the second diagonal run, so the monotonicity check is generalized to “no backward step and at
      most one lateral sign switch”.
    - All obstacle/segment matching logic still assumes: one attached segment endpoint per terminal, nearest obstacle
      remains the intended one, and triples are spatially separated. The generator reserves the union of input/output
      footprints and re-validates the full grid with the solver oracle.
  placement_legality:
    - Triples are constructed in local coordinates around the terminal, then transformed to one of the four edges.
    - Obstacles are anchored inside the V apex and extend mainly along `fwd`; optional inward bulges keep them
      irregular without changing the routing mechanism.
    - The generator paints a candidate triple on an otherwise blank grid, runs both exit rules, and only keeps it if
      the output is legal and the D9-specific straight suffix is visible.
    - Multi-triple grids reserve padded boxes covering both input and predicted output, which keeps obstacle matching
      unambiguous and prevents repaint collisions.
  visual_observability:
    - The only changed rule is directional, so examples must not hide it behind symmetric or too-short routes.
    - The generator biases toward segment lengths 6-9 and requires a straight suffix of at least two steps, making the
      new “wrap then go straight inward” behavior readable from a few examples.
  visual_tidiness:
    - All triples share the original sparse style: no overlap, no 2x2 segment blocks, and padded placement.
    - Because the tail now goes deeper into the grid instead of diagonally peeling away, extra padding is used so
      neighboring triples cannot visually merge.
"""

from __future__ import annotations

import random
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


def _bar_forward(bar_comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
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


def _step_class(dr: int, dc: int, fwd: tuple[int, int]) -> str:
    fwd_comp = dr * fwd[0] + dc * fwd[1]
    perp_comp = dr * fwd[1] + dc * (-fwd[0])
    if abs(dr) + abs(dc) == 1:
        return "F" if fwd_comp > 0 else "X"
    if abs(dr) == 1 and abs(dc) == 1 and fwd_comp > 0:
        if perp_comp > 0:
            return "FR"
        if perp_comp < 0:
            return "FL"
    return "X"


def _step_classes(path: list[tuple[int, int]], fwd: tuple[int, int]) -> list[str]:
    return [_step_class(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1], fwd) for i in range(len(path) - 1)]


def _assert_monotone_wrap(
    path: list[tuple[int, int]],
    fwd: tuple[int, int],
    obstacle_set: set[tuple[int, int]],
    seg_start: tuple[int, int],
    terminal: tuple[int, int],
) -> list[str]:
    step_classes = _step_classes(path, fwd)
    assert all(sc != "X" for sc in step_classes), f"output segment contains non-forward step classes: {' '.join(step_classes)}"

    runs: list[int] = []
    for sc in step_classes:
        if sc == "F":
            continue
        sign = -1 if sc == "FL" else 1
        if not runs or runs[-1] != sign:
            runs.append(sign)
    assert len(runs) <= 2, f"output segment changes lateral wrap side too many times: {' '.join(step_classes)}"

    first_step = step_classes[0] if step_classes else ""
    if first_step in ("FL", "FR"):
        p1_end = 0
        for j, sc in enumerate(step_classes):
            if sc == first_step:
                p1_end = j + 1
            else:
                break
        apex = path[p1_end]
        perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
        inner = (-perp_dir[0], -perp_dir[1])
        apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
        assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"
    return step_classes


def _find_segment_for_bar(
    g: np.ndarray,
    bg: int,
    bar_col: int,
    terminal: tuple[int, int],
) -> tuple[int, tuple[int, int], list[tuple[int, int]], int]:
    neigh = []
    h, w = g.shape
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
    return seg_col, seg_start, seg_comp, length


def _solve(grid: np.ndarray, exit_uses_bar_forward: bool) -> np.ndarray:
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
        seg_col, seg_start, seg_comp, seg_len = _find_segment_for_bar(g, bg, bar_col, terminal)

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
        fwd = _bar_forward(bar_comp)

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

        if exit_uses_bar_forward or len(boundary_path) < 2:
            dr_ext, dc_ext = fwd
        else:
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
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:seg_len]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        _assert_monotone_wrap(new_pts, fwd, obstacle_set, seg_start, terminal)
        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing an irregular segment around the nearest obstacle while preserving segment length.

    Variant D9 change:
    after the obstacle wrap reaches its exit cell, the remaining pixels always continue in the
    bar-forward direction (from the touched border into the grid), not in the final skin-step direction.
    """
    return _solve(grid, exit_uses_bar_forward=True)


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 5
SEG_LEN_MIN, SEG_LEN_MAX = 6, 9
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 6
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 240
PLACEMENT_PAD = 2
TRIPLE_BUILD_RETRIES = 90
TAIL_MIN = 2
BULGE_PROB = 0.35
PHASE_MAX = 3

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


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _build_obstacle_local(p1: int, side: int, obs_len: int) -> list[tuple[int, int]]:
    base = [(p1 + k, side * p1) for k in range(obs_len)]
    obstacle = set(base)
    if p1 >= 2 and len(obstacle) < OBS_SIZE_MAX and random.random() < BULGE_PROB:
        k = random.randint(0, obs_len - 1)
        obstacle.add((p1 + k, side * (p1 - 1)))
    return sorted(obstacle)


def _make_local_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None
    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    if max_turns < INPUT_TURN_MIN:
        return None

    side = random.choice((-1, 1))
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    bar = [(-k, 0) for k in range(bar_len)]

    p1 = random.randint(1, min(PHASE_MAX, seg_len - 3, OBS_MD_MAX // 2))
    obs_len = random.randint(OBS_SIZE_MIN, min(3, OBS_SIZE_MAX, seg_len - p1 - 1))
    obstacle = _build_obstacle_local(p1, side, obs_len)

    if not obstacle:
        return None
    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None
    if _is_4adj(input_seg, obstacle):
        return None

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
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


def _segment_path_from_output(
    out: np.ndarray,
    seg_color: int,
    terminal: tuple[int, int],
) -> list[tuple[int, int]]:
    seg_pts = [(int(r), int(c)) for r, c in np.argwhere(out == seg_color)]
    assert seg_pts, "output must contain a segment"
    start = None
    for nr, nc in neighbors4(terminal[0], terminal[1]):
        if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and int(out[nr, nc]) == seg_color:
            start = (nr, nc)
            break
    assert start is not None, "segment must remain attached to the terminal"
    path = _segment_main_path(seg_pts, start)
    if path[0] != start:
        path.reverse()
    return path


def _tail_metrics(path: list[tuple[int, int]], fwd: tuple[int, int]) -> dict[str, int]:
    step_classes = _step_classes(path, fwd)
    tail_len = 0
    for sc in reversed(step_classes):
        if sc == "F":
            tail_len += 1
        else:
            break
    diag_steps = sum(sc in ("FL", "FR") for sc in step_classes)
    return {"tail_len": tail_len, "diag_steps": diag_steps}


def _place_triple(
    edge: str,
    n: int,
    reserved: np.ndarray,
    bg: int,
    bar_color: int,
    seg_color: int,
    obstacle_color: int,
) -> dict | None:
    fwd = EDGE_DATA[edge]["fwd"]
    for _ in range(TRIPLE_BUILD_RETRIES):
        triple = _make_local_triple(n)
        if triple is None:
            continue
        positions = list(range(n))
        random.shuffle(positions)
        for pos in positions:
            bar = _transform(triple["bar"], edge, n, triple["bar_len"], pos)
            obstacle = _transform(triple["obstacle"], edge, n, triple["bar_len"], pos)
            input_seg = _transform(triple["input_seg"], edge, n, triple["bar_len"], pos)
            union = bar + obstacle + input_seg
            if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union):
                continue

            inp = np.full((n, n), bg, dtype=int)
            paint_coords(inp, bar, bar_color)
            paint_coords(inp, obstacle, obstacle_color)
            paint_coords(inp, input_seg, seg_color)

            terminal = _transform([(0, 0)], edge, n, triple["bar_len"], pos)[0]
            try:
                out_new = _solve(inp, exit_uses_bar_forward=True)
                out_old = _solve(inp, exit_uses_bar_forward=False)
                path = _segment_path_from_output(out_new, seg_color, terminal)
            except Exception:
                continue

            if np.array_equal(out_new, out_old):
                continue

            metrics = _tail_metrics(path, fwd)
            if metrics["tail_len"] < TAIL_MIN or metrics["diag_steps"] == 0:
                continue

            output_seg = [(int(r), int(c)) for r, c in np.argwhere(out_new == seg_color)]
            reserve_coords = bar + obstacle + input_seg + output_seg
            r0, c0, r1, c1 = _expanded_bbox(reserve_coords, n)
            if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            return {
                "bar": bar,
                "obstacle": obstacle,
                "input_seg": input_seg,
                "output_seg": output_seg,
                "terminal": terminal,
                "reserve": (r0, c0, r1, c1),
                "metrics": metrics,
            }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 12:
        max_triples = 1
    elif n < 17:
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
        placed = _place_triple(edge, n, reserved, bg, bar_color, seg_color, obstacle_color)
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
