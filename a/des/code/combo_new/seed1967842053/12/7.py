import random
from collections import deque

import numpy as np


BLACK = 0
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# ARC color names in alphabetical order, excluding black for path choices:
# azure, blue, gray, green, magenta, maroon, orange, red, yellow.
COLOR_RANK = {8: 0, 1: 1, 5: 2, 3: 3, 6: 4, 9: 5, 7: 6, 2: 7, 4: 8}
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _inside(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def _is_border(pos, h, w):
    r, c = pos
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors(pos, h, w):
    r, c = pos
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _inside(nr, nc, h, w):
            yield nr, nc


def _component_sets(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = set()
        while q:
            p = q.popleft()
            comp.add(p)
            for nr, nc in _neighbors(p, h, w):
                if not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _find_paths(grid):
    h, w = grid.shape
    paths = {}
    for raw in np.unique(grid):
        color = int(raw)
        if color == BLACK:
            continue
        edge_comps = []
        for comp in _component_sets(grid, color):
            if any(_is_border(p, h, w) for p in comp):
                edge_comps.append(comp)
        if edge_comps:
            paths[color] = max(edge_comps, key=len)
    return paths


def _endpoint_for_path(comp, h, w):
    ends = []
    for p in comp:
        deg = sum((n in comp) for n in _neighbors(p, h, w))
        if deg <= 1:
            ends.append(p)
    if not ends:
        ends = list(comp)
    inner = [p for p in ends if not _is_border(p, h, w)]
    pool = inner if inner else ends
    return max(pool, key=lambda p: min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]))


def _find_endpoints(paths, h, w):
    return {color: _endpoint_for_path(comp, h, w) for color, comp in paths.items()}


def _find_signal_links(grid, paths, endpoints):
    h, w = grid.shape
    signal_color = {}
    signal_cell = {}
    for color, end in endpoints.items():
        candidates = []
        for nb in _neighbors(end, h, w):
            val = int(grid[nb])
            if val == BLACK or val == color or val not in paths:
                continue
            if nb in paths[val]:
                continue
            candidates.append((nb, val))
        if not candidates:
            for nb in _neighbors(end, h, w):
                val = int(grid[nb])
                if val != BLACK and val != color and val in paths:
                    candidates.append((nb, val))
        if candidates:
            candidates.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
            cell, val = candidates[0]
            signal_cell[color] = cell
            signal_color[color] = val
    return signal_color, signal_cell


def _route(a, b, horizontal_first=True):
    r1, c1 = a
    r2, c2 = b
    cells = []
    if horizontal_first:
        step_c = 1 if c2 >= c1 else -1
        for c in range(c1, c2 + step_c, step_c):
            cells.append((r1, c))
        step_r = 1 if r2 >= r1 else -1
        for r in range(r1 + step_r, r2 + step_r, step_r):
            cells.append((r, c2))
    else:
        step_r = 1 if r2 >= r1 else -1
        for r in range(r1, r2 + step_r, step_r):
            cells.append((r, c1))
        step_c = 1 if c2 >= c1 else -1
        for c in range(c1 + step_c, c2 + step_c, step_c):
            cells.append((r2, c))
    return cells


def _blocked(grid, cells, allowed):
    for p in cells:
        if p in allowed:
            continue
        if int(grid[p]) != BLACK:
            return True
    return False


def _bridge_color(a, b):
    return a if COLOR_RANK.get(a, 99) <= COLOR_RANK.get(b, 99) else b


def main(grid: np.ndarray) -> np.ndarray:
    """Connect paired edge paths by L-shaped bridges and remove their signal pixels.

    Input structure:
    The background is black. Each colored path is a width-one orthogonal polyline
    of a single color, has one component touching the border, and ends in the
    interior. Next to each interior endpoint is one non-border signal pixel of a
    different color. That signal color is the color of another path, and pairs are
    reciprocal.

    Rule:
    For each reciprocal pair, connect the two interior endpoints with an L-shaped
    bridge. Try the horizontal-then-vertical route first; if it crosses an
    existing non-background pixel after signal removal, use the vertical-then-
    horizontal route. Bridge cells use the earlier color in the ARC color-name
    alphabetical order. Original path cells keep their colors, and every signal
    pixel is erased to black.

    Key insight:
    The neighboring signal pixel is not part of its adjacent path; its color names
    the partner path, so solving requires first decoding these indirect pairings
    and then applying the deterministic L-route rule.
    """
    arr = np.array(grid, copy=True)
    h, w = arr.shape
    paths = _find_paths(arr)
    endpoints = _find_endpoints(paths, h, w)
    signal_color, signal_cell = _find_signal_links(arr, paths, endpoints)

    pairs = []
    used = set()
    for color in sorted(signal_color, key=lambda c: COLOR_RANK.get(c, c + 50)):
        other = signal_color[color]
        if color in used or other in used:
            continue
        if other in signal_color and signal_color[other] == color:
            pairs.append((color, other))
            used.add(color)
            used.add(other)

    out = arr.copy()
    for cell in signal_cell.values():
        out[cell] = BLACK

    for a, b in pairs:
        ea = endpoints[a]
        eb = endpoints[b]
        allowed = {ea, eb}
        h_route = _route(ea, eb, horizontal_first=True)
        if _blocked(out, h_route, allowed):
            chosen = _route(ea, eb, horizontal_first=False)
        else:
            chosen = h_route
        color = _bridge_color(a, b)
        for p in chosen:
            if p in allowed:
                continue
            if int(out[p]) == BLACK:
                out[p] = color
    return out


def _path_is_simple(route, h, w):
    cells = set(route)
    if len(cells) != len(route):
        return False
    degrees = []
    for p in cells:
        degrees.append(sum((n in cells) for n in _neighbors(p, h, w)))
    return degrees.count(1) == 2 and max(degrees) <= 2


def _near_blocked(pos, blocked, h, w):
    if pos in blocked:
        return True
    return any(n in blocked for n in _neighbors(pos, h, w))


def _find_route_for_generator(h, w, start, target, blocked):
    q = deque([start])
    prev = {start: None}
    while q:
        p = q.popleft()
        if p == target:
            break
        dirs = list(DIRS)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = p[0] + dr, p[1] + dc
            npair = (nr, nc)
            if not _inside(nr, nc, h, w) or npair in prev:
                continue
            if npair != start and _is_border(npair, h, w):
                continue
            if npair != target and _near_blocked(npair, blocked, h, w):
                continue
            prev[npair] = p
            q.append(npair)
    if target not in prev:
        return None
    route = []
    p = target
    while p is not None:
        route.append(p)
        p = prev[p]
    route.reverse()
    if len(route) < 5 or not _path_is_simple(route, h, w):
        return None
    return route


def _random_start(h, w):
    side = random.randrange(4)
    if side == 0:
        return 0, random.randint(2, w - 3)
    if side == 1:
        return h - 1, random.randint(2, w - 3)
    if side == 2:
        return random.randint(2, h - 3), 0
    return random.randint(2, h - 3), w - 1


def _route_flags_for_pair(grid, endpoints, a, b):
    ea, eb = endpoints[a], endpoints[b]
    allowed = {ea, eb}
    h_route = _route(ea, eb, True)
    v_route = _route(ea, eb, False)
    h_blocked = _blocked(grid, h_route, allowed)
    v_blocked = _blocked(grid, v_route, allowed)
    return h_blocked, v_blocked


def _all_matchings(items):
    items = tuple(items)
    if not items:
        yield []
        return
    first = items[0]
    for i in range(1, len(items)):
        second = items[i]
        rest = items[1:i] + items[i + 1:]
        for tail in _all_matchings(rest):
            yield [(first, second)] + tail


def _choose_pairs(grid, endpoints, colors, want_blocked):
    candidates = list(_all_matchings(colors))
    random.shuffle(candidates)
    backup = None
    for pairs in candidates:
        flags = [_route_flags_for_pair(grid, endpoints, a, b) for a, b in pairs]
        if any(hb and vb for hb, vb in flags):
            continue
        has_blocked = any(hb for hb, _ in flags)
        if backup is None:
            backup = pairs
        if has_blocked == want_blocked:
            return pairs
    return backup


def _place_signals(grid, paths, endpoints, pairs):
    h, w = grid.shape
    signal_cells = []
    order = []
    for a, b in pairs:
        order.append((a, b))
        order.append((b, a))
    random.shuffle(order)

    for owner, sig_color in order:
        end = endpoints[owner]
        dirs = list(DIRS)
        random.shuffle(dirs)
        placed = False
        for dr, dc in dirs:
            cell = (end[0] + dr, end[1] + dc)
            if not _inside(cell[0], cell[1], h, w) or _is_border(cell, h, w):
                continue
            if int(grid[cell]) != BLACK:
                continue
            ok = True
            for nb in _neighbors(cell, h, w):
                if nb == end:
                    continue
                if int(grid[nb]) != BLACK:
                    ok = False
                    break
            if not ok:
                continue
            if any(nb in paths[sig_color] for nb in _neighbors(cell, h, w)):
                continue
            grid[cell] = sig_color
            signal_cells.append(cell)
            placed = True
            break
        if not placed:
            return False
    return True


def _try_generate_once(want_blocked):
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    n_paths = random.choice((4, 6))
    if min(h, w) < 22:
        n_paths = 4
    colors = random.sample(PATH_COLORS, n_paths)
    grid = np.zeros((h, w), dtype=int)
    blocked = set()
    paths = {}
    endpoints = {}

    for color in colors:
        route = None
        for _ in range(160):
            start = _random_start(h, w)
            if _near_blocked(start, blocked, h, w):
                continue
            target = (random.randint(3, h - 4), random.randint(3, w - 4))
            if _near_blocked(target, blocked, h, w):
                continue
            if abs(start[0] - target[0]) + abs(start[1] - target[1]) < 7:
                continue
            candidate = _find_route_for_generator(h, w, start, target, blocked)
            if candidate is not None:
                route = candidate
                break
        if route is None:
            return None
        for cell in route:
            grid[cell] = color
            blocked.add(cell)
        paths[color] = set(route)
        endpoints[color] = route[-1]

    pairs = _choose_pairs(grid, endpoints, colors, want_blocked)
    if pairs is None:
        return None
    flags = [_route_flags_for_pair(grid, endpoints, a, b) for a, b in pairs]
    if any(hb and vb for hb, vb in flags):
        return None
    if any(hb for hb, _ in flags) != want_blocked:
        return None
    if not _place_signals(grid, paths, endpoints, pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    want_blocked = bool(random.getrandbits(1))
    for _ in range(90):
        grid = _try_generate_once(want_blocked)
        if grid is not None:
            return grid
    for _ in range(90):
        grid = _try_generate_once(not want_blocked)
        if grid is not None:
            return grid
    # Conservative fallback with the same rules but only requiring validity.
    while True:
        grid = _try_generate_once(False)
        if grid is not None:
            return grid
