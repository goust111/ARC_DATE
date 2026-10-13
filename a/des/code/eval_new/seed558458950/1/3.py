import numpy as np
from collections import Counter
from collections import deque
from itertools import combinations


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if c + 1 < w:
        yield r, c + 1
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1


def _extract_legend_colors(grid: np.ndarray, bg: int):
    h, w = grid.shape

    def scan_vertical(col: int):
        runs = []
        r = 0
        while r < h:
            if grid[r, col] == bg:
                r += 1
                continue

            start = r
            colors = []
            while r < h and grid[r, col] != bg:
                left_ok = (col == 0 or grid[r, col - 1] == bg)
                right_ok = (col == w - 1 or grid[r, col + 1] == bg)
                if not (left_ok and right_ok):
                    break

                color = int(grid[r, col])
                if color in colors:
                    break

                colors.append(color)
                r += 1

            if colors:
                runs.append((len(colors), colors, start))

            if r == start:
                r += 1

        return runs

    def scan_horizontal(row: int):
        runs = []
        c = 0
        while c < w:
            if grid[row, c] == bg:
                c += 1
                continue

            start = c
            colors = []
            while c < w and grid[row, c] != bg:
                up_ok = (row == 0 or grid[row - 1, c] == bg)
                down_ok = (row == h - 1 or grid[row + 1, c] == bg)
                if not (up_ok and down_ok):
                    break

                color = int(grid[row, c])
                if color in colors:
                    break

                colors.append(color)
                c += 1

            if colors:
                runs.append((len(colors), colors, start))

            if c == start:
                c += 1

        return runs

    candidates = []
    if w > 0:
        candidates.extend(scan_vertical(0))
        if w > 1:
            candidates.extend(scan_vertical(w - 1))
    if h > 0:
        candidates.extend(scan_horizontal(0))
        if h > 1:
            candidates.extend(scan_horizontal(h - 1))

    if not candidates:
        return []

    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]


def _detect_marker_color(grid: np.ndarray, bg: int, legend_colors):
    h, w = grid.shape
    legend_set = set(int(c) for c in legend_colors)

    candidates = []
    for color, cnt in zip(*np.unique(grid, return_counts=True)):
        color = int(color)
        cnt = int(cnt)
        if color == bg or color in legend_set or cnt > 8 or cnt == 1:
            continue

        cells = np.argwhere(grid == color)
        ok = True
        for r, c in cells:
            deg = 0
            for nr, nc in _neighbors(int(r), int(c), h, w):
                if grid[nr, nc] != bg:
                    deg += 1
            if deg != 1:
                ok = False
                break

        if ok:
            candidates.append((cnt, color))

    if candidates:
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][1]

    fallback = [
        (int(color), int(cnt))
        for color, cnt in zip(*np.unique(grid, return_counts=True))
        if int(color) != bg and int(color) not in legend_set
    ]
    if not fallback:
        return None

    fallback.sort(key=lambda t: (t[1], t[0]))
    return fallback[0][0]


def _detect_grey_color(grid: np.ndarray, bg: int, marker: int, legend_colors):
    h, w = grid.shape
    if marker is None:
        return None

    excluded = set(int(c) for c in legend_colors) | {bg, marker}
    best = None

    for color, cnt in zip(*np.unique(grid, return_counts=True)):
        color = int(color)
        if color in excluded or int(cnt) == 0:
            continue

        cells = np.argwhere(grid == color)
        if len(cells) == 0:
            continue

        mixed = 0
        self_like = 0
        for r, c in cells:
            r = int(r)
            c = int(c)
            for nr, nc in _neighbors(r, c, h, w):
                ncolor = int(grid[nr, nc])
                if ncolor == color:
                    self_like += 1
                    break
            for nr, nc in _neighbors(r, c, h, w):
                ncolor = int(grid[nr, nc])
                if ncolor != bg and ncolor != color and ncolor not in excluded:
                    mixed += 1
                    break

        total = len(cells)
        mixed_ratio = mixed / total
        self_ratio = self_like / total
        score = mixed_ratio - 0.8 * self_ratio

        if best is None or score > best[0]:
            best = (score, color)

    if best is None or best[0] <= 0.05:
        return None
    return best[1]


def _propose_next(
    grid: np.ndarray,
    h: int,
    w: int,
    state: dict,
    bg: int,
    marker: int,
    grey: int | None,
    priority,
):
    cr, cc = state["pos"]
    prev = state["prev"]
    path_color = state["path_color"]
    cur_color = int(grid[cr, cc])

    if cur_color == marker:
        for nr, nc in _neighbors(cr, cc, h, w):
            ncolor = int(grid[nr, nc])
            if ncolor == bg or ncolor == marker:
                continue
            if grey is not None and ncolor == grey:
                continue
            if ncolor == path_color:
                return ((nr, nc), path_color)
        return None

    if grey is not None and cur_color == grey:
        options = []
        for nr, nc in _neighbors(cr, cc, h, w):
            ncolor = int(grid[nr, nc])
            if (nr, nc) == prev:
                continue
            if ncolor in (bg, marker, grey):
                continue
            if ncolor == path_color:
                continue
            options.append(((nr, nc), ncolor))

        if not options:
            return None

        options.sort(key=lambda item: (priority.get(item[1], 9999), item[0][0], item[0][1]))
        return options[0]

    cands = []
    for nr, nc in _neighbors(cr, cc, h, w):
        if int(grid[nr, nc]) == path_color and (nr, nc) != prev:
            cands.append((nr, nc))

    if not cands:
        return None
    return (cands[0], path_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate red signals from each red source along its path and resolve crossings by priority.

    Input structure:
    - Background is the dominant color.
    - Thin orthogonal paths are colored; each path has one red endpoint marker and one ordinary endpoint.
    - Grey pixels denote path intersections where two paths cross.
    - A border legend lists path colors in high-to-low priority order.

    Rule:
    - Start one signal at each red source, moving one pixel per step along its current path color.
    - When a signal enters a grey intersection, it switches to the other incident path and keeps moving.
    - If multiple signals reach the same grey intersection in the same step, only the signal whose
      current path color has higher legend priority continues; the others stop.
    - A grey intersection allows passage at most once. Cells visited by passing signals become red;
      untouched path cells remain their original colors, and untouched intersections remain grey.

    Key insight:
    - This is a synchronous propagation problem: correct timing on each branch and tie-breaking at
      shared intersections determine which path segments become red.
    """
    g = np.array(grid, dtype=np.int64)
    h, w = g.shape
    out = g.copy()

    bg = _most_common_color(g)
    legend_colors = _extract_legend_colors(g, bg)
    priority = {c: i for i, c in enumerate(legend_colors)}

    marker = _detect_marker_color(g, bg, legend_colors)
    if marker is None:
        return out

    grey = _detect_grey_color(g, bg, marker, legend_colors)

    states = []
    marker_cells = np.argwhere(g == marker)
    for mr, mc in marker_cells:
        next_color = None
        mr = int(mr)
        mc = int(mc)
        for nr, nc in _neighbors(mr, mc, h, w):
            ncolor = int(g[nr, nc])
            if ncolor in (bg, marker):
                continue
            if grey is not None and ncolor == grey:
                continue
            next_color = ncolor
            break
        if next_color is not None:
            states.append({"pos": (mr, mc), "prev": None, "path_color": next_color})

    if not states:
        return out

    alive = set(range(len(states)))
    used_intersections = set()

    for _ in range(h * w * 2 + 10):
        if not alive:
            break

        proposals = {}
        for sid in list(alive):
            state = states[sid]
            nxt = _propose_next(g, h, w, state, bg, marker, grey, priority)
            if nxt is None:
                alive.discard(sid)
                continue
            proposals.setdefault(nxt[0], []).append((sid, state["path_color"], nxt[1]))

        if not proposals:
            break

        for target, cands in proposals.items():
            tr, tc = target
            if tr < 0 or tr >= h or tc < 0 or tc >= w:
                continue

            is_grey_target = grey is not None and g[target] == grey
            if is_grey_target:
                if target in used_intersections:
                    for sid, _, _ in cands:
                        alive.discard(sid)
                    continue

                if len(cands) > 1:
                    cands.sort(key=lambda item: (priority.get(item[1], 9999), item[0]))
                    winner_sid = cands[0][0]
                    for sid, _, _ in cands[1:]:
                        alive.discard(sid)
                else:
                    winner_sid = cands[0][0]

                if winner_sid not in alive:
                    continue

                _, _, next_color = next(item for item in cands if item[0] == winner_sid)
                st = states[winner_sid]
                st["prev"] = st["pos"]
                st["pos"] = target
                st["path_color"] = next_color
                out[target] = marker
                used_intersections.add(target)
                continue

            if len(cands) > 1:
                cands.sort(key=lambda item: (priority.get(item[1], 9999), item[0]))
                winner_sid = cands[0][0]
                for sid, _, _ in cands[1:]:
                    alive.discard(sid)
            else:
                winner_sid = cands[0][0]

            if winner_sid not in alive:
                continue

            _, _, next_color = cands[0]
            st = states[winner_sid]
            st["prev"] = st["pos"]
            st["pos"] = target
            st["path_color"] = next_color
            out[target] = marker

    return out


def _random_free_cell(occupied, h, w, rng):
    cells = [(r, c) for r in range(1, h - 1) for c in range(2, w - 1) if (r, c) not in occupied]
    if not cells:
        return None
    return cells[int(rng.integers(0, len(cells)))]


def _shortest_path(start, goal, blocked, allow, h: int, w: int):
    if start == goal:
        return [start]

    q = deque([start])
    prev = {start: None}

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, h, w):
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if nxt != goal and nxt in blocked and nxt not in allow:
                continue

            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    if goal not in prev:
        return None

    path = [goal]
    cur = goal
    while prev[cur] is not None:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def _build_route(start, end, waypoints, occupied, h: int, w: int):
    points = [start] + list(waypoints) + [end]
    route = [start]
    used = {start}
    allow = set(waypoints)

    for a, b in zip(points, points[1:]):
        seg = _shortest_path(a, b, occupied | used, allow | {b}, h, w)
        if seg is None:
            return None
        for p in seg[1:]:
            route.append(p)
            used.add(p)

    if len(route) != len(set(route)):
        return None
    return route


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    BG = 0
    MARKER = 2
    GREY = 7
    color_bank = [1, 3, 4, 5, 6, 8, 9]

    for _ in range(260):
        h = int(rng.integers(18, 30))
        w = int(rng.integers(18, 30))

        n_paths = int(rng.integers(2, 5))
        if h < n_paths + 4 or w < 14:
            continue

        path_colors = [int(c) for c in rng.choice(color_bank, size=n_paths, replace=False)]
        rng.shuffle(path_colors)

        legend_colors = list(path_colors)
        g = np.full((h, w), BG, dtype=np.int64)
        for idx, color in enumerate(legend_colors):
            g[1 + idx, 0] = color

        occupied = {(r, 0) for r in range(1, 1 + len(legend_colors))}
        paths = []
        ok_all = True

        for pi, color in enumerate(path_colors):
            built = False
            intersection_mode = pi > 0 and rng.random() < (0.85 if pi == 1 else 0.45)

            for _attempt in range(260):
                waypoints = []
                if intersection_mode and paths:
                    prior = paths[int(rng.integers(0, len(paths)))]
                    if len(prior["coords"]) > 4:
                        wp = prior["coords"][int(rng.integers(1, len(prior["coords"]) - 1))]
                        waypoints = [wp]

                start = _random_free_cell(occupied, h, w, rng)
                end = _random_free_cell(occupied, h, w, rng)
                if start is None or end is None or start == end:
                    continue

                if waypoints:
                    d1 = abs(start[0] - waypoints[0][0]) + abs(start[1] - waypoints[0][1])
                    d2 = abs(end[0] - waypoints[0][0]) + abs(end[1] - waypoints[0][1])
                    if d1 < 2 or d2 < 2:
                        continue

                route = _build_route(start, end, waypoints, occupied, h, w)
                if route is None or len(route) < 8:
                    continue

                paths.append({"color": color, "coords": route})
                occupied.update(route)
                built = True
                break

            if built:
                continue

            # retry without intersection target
            for _attempt2 in range(260):
                waypoints = []
                start = _random_free_cell(occupied, h, w, rng)
                end = _random_free_cell(occupied, h, w, rng)
                if start is None or end is None or start == end:
                    continue
                route = _build_route(start, end, waypoints, occupied, h, w)
                if route is not None and len(route) >= 8:
                    paths.append({"color": color, "coords": route})
                    occupied.update(route)
                    built = True
                    break

            if not built:
                ok_all = False
                break

        if not ok_all:
            continue

        inter_map = {}
        for pi, info in enumerate(paths):
            for pos in info["coords"]:
                inter_map.setdefault(pos, []).append(pi)

        source_side = [int(rng.integers(0, 2)) for _ in paths]
        shared = [p for p, idxs in inter_map.items() if len(idxs) >= 2]

        if shared and rng.random() < 0.7:
            rng.shuffle(shared)
            index_maps = [{coord: idx for idx, coord in enumerate(info["coords"])} for info in paths]
            fixed = False
            for pos in shared:
                ids = inter_map[pos]
                for a, b in combinations(ids, 2):
                    ia = index_maps[a][pos]
                    ib = index_maps[b][pos]
                    da = [ia, len(paths[a]["coords"]) - 1 - ia]
                    db = [ib, len(paths[b]["coords"]) - 1 - ib]
                    for sa in (0, 1):
                        for sb in (0, 1):
                            if da[sa] == db[sb]:
                                source_side[a] = sa
                                source_side[b] = sb
                                fixed = True
                                break
                        if fixed:
                            break
                    if fixed:
                        break
                if fixed:
                    break

            if not fixed and rng.random() < 0.4:
                source_side = [int(rng.integers(0, 2)) for _ in paths]

        for info in paths:
            c = info["color"]
            for r, cc in info["coords"]:
                g[r, cc] = c

        for pos, ids in inter_map.items():
            if len(ids) >= 2:
                g[pos] = GREY

        for i, info in enumerate(paths):
            coords = info["coords"]
            source = coords[0] if source_side[i] == 0 else coords[-1]
            if source in inter_map and len(inter_map[source]) >= 2:
                source = coords[-1] if source_side[i] == 0 else coords[0]
            g[source] = MARKER

        if Counter(g.flatten())[MARKER] != len(paths):
            continue

        return g

    return np.array(
        [
            [0, 3, 0, 0, 0, 0, 0, 0],
            [3, 2, 7, 3, 3, 0, 0, 0],
            [0, 3, 7, 3, 0, 0, 0, 0],
            [0, 4, 0, 0, 4, 4, 0, 0],
            [0, 4, 7, 0, 4, 0, 0, 0],
            [0, 0, 0, 0, 4, 2, 7, 5],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
