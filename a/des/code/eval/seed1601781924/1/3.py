import numpy as np
from collections import Counter


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BG = 0
CHANNEL = 5
MARKER = 9
ROUTE_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_MIN = 15
GRID_MAX = 22
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        return (0, 1) if c1 > c0 else (0, -1)
    if c0 == c1:
        return (1, 0) if r1 > r0 else (-1, 0)
    raise AssertionError("segments must be axis-aligned")


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[0], -d[1])


def _edge_start(shape: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = cell
    if r == 1:
        return (0, c)
    if r == h - 2:
        return (h - 1, c)
    if c == 1:
        return (r, 0)
    if c == w - 2:
        return (r, w - 1)
    raise AssertionError("endpoint must be adjacent to an edge")


def _inflate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _center_distance(shape: tuple[int, int], cell: tuple[int, int]) -> float:
    h, w = shape
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    r, c = cell
    return (r - cr) ** 2 + (c - cc) ** 2


def _detect_structure(
    grid: np.ndarray,
) -> tuple[int, int, int | None, dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    starts_by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            v = int(g[r, c])
            if v == bg:
                continue
            starts_by_color.setdefault(v, []).append((r, c))

    assert starts_by_color, "expected edge start pixels"
    assert all(len(v) == 2 for v in starts_by_color.values()), "each color must appear twice on the edge"

    interior_counts: Counter[int] = Counter()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(g[r, c])
            if v != bg:
                interior_counts[v] += 1

    assert interior_counts, "expected interior channels"
    channel_color = max(interior_counts, key=interior_counts.get)
    marker_candidates = [color for color in interior_counts if color != channel_color]
    assert len(marker_candidates) <= 1, "expected at most one marker color"
    marker_color = marker_candidates[0] if marker_candidates else None
    if marker_color is not None:
        assert interior_counts[channel_color] > interior_counts[marker_color], "channel color should dominate markers"

    return bg, int(channel_color), None if marker_color is None else int(marker_color), starts_by_color


def _trace_from_start(
    grid: np.ndarray,
    start: tuple[int, int],
    bg: int,
    channel_color: int,
    marker_color: int | None,
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    network_colors = {int(channel_color)}
    if marker_color is not None:
        network_colors.add(int(marker_color))

    def is_start(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not _in_bounds(g.shape, r, c):
            return False
        if r not in (0, h - 1) and c not in (0, w - 1):
            return False
        return int(g[r, c]) != bg and int(g[r, c]) not in network_colors

    def traversable(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not _in_bounds(g.shape, r, c):
            return False
        return int(g[r, c]) in network_colors or is_start(cell)

    neighbors = []
    sr, sc = start
    for dr, dc in DIRS4:
        nxt = (sr + dr, sc + dc)
        if _in_bounds(g.shape, nxt[0], nxt[1]) and int(g[nxt]) in network_colors:
            neighbors.append(nxt)
    assert len(neighbors) == 1, "each start must touch exactly one channel endpoint"

    cur = neighbors[0]
    direction = (cur[0] - sr, cur[1] - sc)
    path = [start, cur]
    seen = {(cur[0], cur[1], direction[0], direction[1])}

    while True:
        r, c = cur
        dr, dc = direction
        val = int(g[r, c])
        next_dir: tuple[int, int] | None = None

        if marker_color is not None and val == marker_color:
            options = []
            for ndr, ndc in DIRS4:
                if dr * ndr + dc * ndc != 0:
                    continue
                nxt = (r + ndr, c + ndc)
                if traversable(nxt):
                    options.append((ndr, ndc))
            if options:
                next_dir = min(
                    options,
                    key=lambda d: (_center_distance(g.shape, (r + d[0], c + d[1])), d),
                )
            else:
                straight = (r + dr, c + dc)
                assert traversable(straight), "marked intersection must allow a turn or a valid fallback"
                next_dir = (dr, dc)
        else:
            straight = (r + dr, c + dc)
            if traversable(straight):
                next_dir = (dr, dc)
            else:
                options = []
                for ndr, ndc in DIRS4:
                    if (ndr, ndc) == (-dr, -dc):
                        continue
                    nxt = (r + ndr, c + ndc)
                    if traversable(nxt):
                        options.append((ndr, ndc))
                assert len(options) == 1, "ordinary channel step should be unambiguous"
                next_dir = options[0]

        nxt = (r + next_dir[0], c + next_dir[1])
        assert traversable(nxt), "path stepped off the network"
        path.append(nxt)
        if is_start(nxt):
            assert nxt != start, "path must terminate at the opposite edge start"
            return path

        state = (nxt[0], nxt[1], next_dir[0], next_dir[1])
        assert state not in seen, "path should not loop"
        seen.add(state)
        cur = nxt
        direction = next_dir


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, channel_color, marker_color, starts_by_color = _detect_structure(g)
    out = g.copy()
    if marker_color is not None:
        out[out == marker_color] = channel_color

    painted: dict[tuple[int, int], int] = {}
    for color in sorted(starts_by_color):
        for start in sorted(starts_by_color[color]):
            path = _trace_from_start(g, start, bg, channel_color, marker_color)
            for cell in path:
                old = painted.get(cell)
                if old is not None:
                    assert old == color, "different rays should not overlap"
                painted[cell] = color
                out[cell] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays along a gray orthogonal channel network.

    Plain intersections keep the current direction. Marked intersections turn
    ninety degrees, preferring the branch whose next step is closer to the
    grid center.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _eligible_cross_cells(
    a: tuple[int, int],
    b: tuple[int, int],
    margin: int = 2,
) -> list[tuple[int, int]]:
    cells = _segment_cells(a, b)
    if len(cells) <= 2 * margin + 1:
        return []
    return cells[margin:-margin]


def _route_from_points(
    shape: tuple[int, int],
    points: list[tuple[int, int]],
    color: int,
    add_ordinary_cross: bool,
) -> dict[str, object] | None:
    h, w = shape
    channel: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    markers: set[tuple[int, int]] = set()

    segment_dirs: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        seg = _segment_cells(a, b)
        segment_dirs.append(_direction(a, b))
        path_cells.update(seg)
        channel.update(seg)

    for i in range(1, len(points) - 1):
        turn = points[i]
        markers.add(turn)
        incoming = _direction(points[i - 1], points[i])
        outgoing = _direction(points[i], points[i + 1])
        used = {_opp(incoming), outgoing}
        missing = [d for d in DIRS4 if d not in used]
        for dr, dc in missing:
            stub = (turn[0] + dr, turn[1] + dc)
            if not (1 <= stub[0] < h - 1 and 1 <= stub[1] < w - 1):
                return None
            channel.add(stub)

    if add_ordinary_cross:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(len(points) - 1):
            for cell in _eligible_cross_cells(points[i], points[i + 1]):
                candidates.append((cell, segment_dirs[i]))
        if not candidates:
            return None
        center, seg_dir = candidates[np.random.randint(len(candidates))]
        if seg_dir[0] == 0:
            missing = [(-1, 0), (1, 0)]
        else:
            missing = [(0, -1), (0, 1)]
        for dr, dc in missing:
            stub = (center[0] + dr, center[1] + dc)
            if not (1 <= stub[0] < h - 1 and 1 <= stub[1] < w - 1):
                return None
            channel.add(stub)

    starts = (_edge_start(shape, points[0]), _edge_start(shape, points[-1]))
    occupied = set(channel) | set(starts)

    if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in channel):
        return None

    if len(starts) != 2 or starts[0] == starts[1]:
        return None

    route = {
        "color": int(color),
        "channel": channel,
        "markers": markers,
        "starts": starts,
        "occupied": occupied,
        "path_cells": path_cells,
    }
    return route if _validate_route(shape, route) else None


def _render_routes(shape: tuple[int, int], routes: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for route in routes:
        for r, c in route["channel"]:
            grid[r, c] = CHANNEL
        for r, c in route["markers"]:
            grid[r, c] = MARKER
        start_a, start_b = route["starts"]
        grid[start_a] = int(route["color"])
        grid[start_b] = int(route["color"])
    return grid


def _validate_route(shape: tuple[int, int], route: dict[str, object]) -> bool:
    grid = _render_routes(shape, [route])
    color = int(route["color"])
    start_a, start_b = route["starts"]
    expected = set(route["path_cells"]) | {start_a, start_b}

    try:
        path_a = _trace_from_start(grid, start_a, BG, CHANNEL, MARKER)
        path_b = _trace_from_start(grid, start_b, BG, CHANNEL, MARKER)
    except AssertionError:
        return False

    if path_a[-1] != start_b or path_b[-1] != start_a:
        return False
    if set(path_a) != expected or set(path_b) != expected:
        return False

    solved = _solve_grid(grid)
    actual = {tuple(p) for p in np.argwhere(solved == color)}
    return actual == expected


def _make_straight_route(shape: tuple[int, int], color: int, vertical: bool) -> dict[str, object] | None:
    h, w = shape
    if vertical:
        cols = list(range(2, w - 2))
        if not cols:
            return None
        c = int(np.random.choice(cols))
        points = [(1, c), (h - 2, c)]
    else:
        rows = list(range(2, h - 2))
        if not rows:
            return None
        r = int(np.random.choice(rows))
        points = [(r, 1), (r, w - 2)]
    return _route_from_points(shape, points, color, add_ordinary_cross=True)


def _make_turn_route(shape: tuple[int, int], color: int, add_ordinary_cross: bool) -> dict[str, object] | None:
    h, w = shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0

    top_rows = [r for r in range(2, h - 2) if r < center_r]
    bottom_rows = [r for r in range(2, h - 2) if r > center_r]
    left_cols = [c for c in range(2, w - 2) if c < center_c]
    right_cols = [c for c in range(2, w - 2) if c > center_c]
    if not top_rows or not bottom_rows or not left_cols or not right_cols:
        return None

    quadrant = np.random.choice(["tl", "tr", "bl", "br"])
    if quadrant == "tl":
        r = int(np.random.choice(top_rows))
        c = int(np.random.choice(left_cols))
        points = [(h - 2, c), (r, c), (r, w - 2)]
    elif quadrant == "tr":
        r = int(np.random.choice(top_rows))
        c = int(np.random.choice(right_cols))
        points = [(h - 2, c), (r, c), (r, 1)]
    elif quadrant == "bl":
        r = int(np.random.choice(bottom_rows))
        c = int(np.random.choice(left_cols))
        points = [(1, c), (r, c), (r, w - 2)]
    else:
        r = int(np.random.choice(bottom_rows))
        c = int(np.random.choice(right_cols))
        points = [(1, c), (r, c), (r, 1)]
    return _route_from_points(shape, points, color, add_ordinary_cross=add_ordinary_cross)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    shape = (h, w)

    route_count = 2 + int(rng.random() < 0.5)
    colors = list(rng.choice(np.array(ROUTE_COLORS, dtype=int), size=route_count, replace=False))

    plans = [
        ("turn", bool(rng.random() < 0.5)),
        ("straight", True),
    ]
    if route_count == 3:
        if rng.random() < 0.5:
            plans.append(("turn", bool(rng.random() < 0.6)))
        else:
            plans.append(("straight", True))

    routes: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()

    for color, (kind, ordinary) in zip(colors, plans):
        placed = False
        for _ in range(120):
            if kind == "turn":
                candidate = _make_turn_route(shape, int(color), add_ordinary_cross=ordinary)
            else:
                candidate = _make_straight_route(shape, int(color), vertical=bool(rng.integers(0, 2)))
            if candidate is None:
                continue
            occupied = candidate["occupied"]
            if occupied & blocked:
                continue
            routes.append(candidate)
            blocked |= _inflate(occupied, shape)
            placed = True
            break
        if not placed:
            return None

    grid = _render_routes(shape, routes)

    counts = Counter(map(int, grid.flatten()))
    if counts[BG] <= counts[CHANNEL]:
        return None

    try:
        solved = _solve_grid(grid)
    except AssertionError:
        return None

    for route in routes:
        color = int(route["color"])
        expected = set(route["path_cells"]) | set(route["starts"])
        actual = {tuple(p) for p in np.argwhere(solved == color)}
        if actual != expected:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-switch puzzle")
