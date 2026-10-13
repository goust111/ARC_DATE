import numpy as np
from collections import deque

from arc_utils import cc_dirs as _cc_dirs, get_bg as _bg_color


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return _cc_dirs(g == col, tuple(_N8 if conn8 else _N4))


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        row = rs[0]
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        col = cs[0]
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
        for dr, dc in _N8:
            nr, nc = r + dr, c + dc
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
    # Like _bfs_path but disallows diagonal steps when either orthogonal intermediate cell is also walkable.
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
        # diagonal: skip if either intermediate orth cell is in the walkable set
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if 0 <= mid1[0] < h and 0 <= mid1[1] < w and bool(walkable[mid1]):
            return False
        if 0 <= mid2[0] < h and 0 <= mid2[1] < w and bool(walkable[mid2]):
            return False
        return True

    while q:
        r, c = q.popleft()
        for dr, dc in _N8:
            nr, nc = r + dr, c + dc
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

    def md(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    cur = start
    path = [cur]
    seen = {cur}
    while cur != goal:
        cur_md = md(cur)
        best = None  # (md, is_diag, nr, nc)
        for dr, dc in _N8:
            nr, nc = cur[0] + dr, cur[1] + dc
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
            is_diag = 1 if (dr != 0 and dc != 0) else 0
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
    # Extract the "main stroke" of the segment as a simple path.
    #
    # Use 4-neigh edges always. Allow diagonal edges only when they represent a true diagonal stroke,
    # i.e. the diagonal pixels touch only at the corner (no orthogonal intermediate segment pixel),
    # which prevents diagonal shortcuts across bends.
    sset = set(seg_pts)

    def neighbors(p: tuple[int, int]):
        r, c = p
        for dr, dc in _N4:
            q = (r + dr, c + dc)
            if q in sset:
                yield q
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            # If either orthogonal intermediate cell is occupied, treat this as a bend shortcut and skip.
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
        # 4-neigh
        for dr, dc in _N4:
            q = (r + dr, c + dc)
            if q in sset:
                deg[(r, c)] += 1
        # diagonals only if no orth intermediate is present
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


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Swing an irregular line segment around a nearby obstacle sprite, preserving its pixel count.

    Input structure:
    - bg, bar color (straight lines perpendicular to the grid edge, one end on the border), segment color (irregular stroke attached to bar's inner endpoint), obstacle color (sprite near the bar terminal).

    Rule:
    - Each bar is perpendicular to the edge it touches; its inner (terminal) endpoint has an attached irregular segment.
    - The segment detaches and re-routes: it travels from the terminal toward the nearest obstacle, hugs the obstacle's boundary, then exits at a 45° diagonal.
    - Total painted pixels equal the original segment length.

    Key insight: The segment behaves like a rope swung around the obstacle — attach at the nearest point, trace the skin, exit diagonally, conserving length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    out = g.copy()

    def bar_candidates():
        # Yield (bar_color, bar_component, terminal_endpoint).
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue  # require exactly one endpoint on border
                # Orientation must be perpendicular to the edge it touches.
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
                # Must have an attached non-bg neighbor at terminal.
                tr, tc = terminal_ep
                if not any(
                    0 <= tr + dr < h and 0 <= tc + dc < w and int(g[tr + dr, tc + dc]) != bg and int(g[tr + dr, tc + dc]) != col
                    for dr, dc in _N4
                ):
                    continue
                yield col, comp, terminal_ep

    # Process all valid bars; compute moves based on the original grid.
    moves = []  # list of (seg_comp_pts, seg_color, new_pts)
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        # Segment neighbor candidates at terminal.
        neigh = []
        for dr, dc in _N4:
            nr, nc = terminal[0] + dr, terminal[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        if not neigh:
            continue

        # Choose a segment that behaves like a 1-cell-thick stroke attached at an endpoint.
        best_seg = None  # (L, comp_size, seg_col, seg_start, seg_comp)
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=True):
                if cand_start in comp:
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
        if best_seg is None:
            continue
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        # Nearest obstacle component (any other non-bg color) to the terminal.
        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        if not obstacle_candidates:
            continue

        best = None  # (dist, min_r, min_c)
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
        if obstacle_set is None:
            continue

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

        # Path from terminal to obstacle (prefer monotone manhattan decrease).
        walkable = (g == bg) | (g == seg_col)
        walkable[terminal] = True
        # Choose an obstacle pixel so that the translated endpoint (p_near + offset) lies OUTSIDE the obstacle,
        # i.e. the segment connects to the obstacle "skin" rather than landing inside the obstacle and wasting length.
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
            if good:
                _, pr, pc = min(good)
                return (pr, pc)
            return min(obstacle_set, key=lambda p: (abs(p[0] - terminal[0]) + abs(p[1] - terminal[1]), p[0], p[1]))

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        if path_t_to_obs is None:
            continue
        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted):
            continue
        if not shifted or shifted[0] != seg_start:
            continue
        entry = shifted[-1]

        # Obstacle-adjacent skin.
        skin = set()
        for r, c in obstacle_set:
            for dr, dc in _N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        # Pick exit as the farthest walkable skin cell from the terminal (ties resolved by projecting away).
        # Use obstacle centroid direction as an 'away' vector.
        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])

        def skin_ok(p):
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]
        def exit_key(p):
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        # BFS along skin (8-neigh) from entry to exit.
        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]

        # Extension direction is the last step used to reach the exit on the skin path.
        if len(boundary_path) >= 2:
            dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
            dc_ext = boundary_path[-1][1] - boundary_path[-2][1]
        else:
            # Fallback: diagonal in the direction away from the terminal.
            dr_ext = 1 if away[0] > 0 else (-1 if away[0] < 0 else 1)
            dc_ext = 1 if away[1] > 0 else (-1 if away[1] < 0 else 1)

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

        new_pts = route[:L]
        moves.append((seg_comp, seg_col, new_pts))

    # Clear segments first, then repaint.
    for seg_comp, seg_col, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out
