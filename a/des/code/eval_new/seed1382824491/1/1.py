import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CONTESTED = -2
MAX_GEN_ATTEMPTS = 400


def _line_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segments must be axis-aligned")


def _corner_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _seed_positions(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    seeds: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid != bg):
        color = int(grid[r, c])
        if color not in (GREY, ORANGE):
            seeds[(int(r), int(c))] = color
    return seeds


def _network_neighbors(grid: np.ndarray, bg: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != bg):
        pos = (int(r), int(c))
        cur: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
                cur.append((rr, cc))
        neighbors[pos] = cur
    return neighbors


def _no_valve_seed_dist(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    seeds: dict[tuple[int, int], int],
    valve_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()
    for pos in seeds:
        if pos in valve_cells:
            continue
        dist[pos] = 0
        q.append(pos)
    while q:
        pos = q.popleft()
        d = dist[pos]
        for nxt in neighbors.get(pos, []):
            if nxt in valve_cells or nxt in dist:
                continue
            dist[nxt] = d + 1
            q.append(nxt)
    return dist


def _valve_allowed_entries(
    grid: np.ndarray,
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    seeds: dict[tuple[int, int], int],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    valve_cells = {pos for pos in neighbors if int(grid[pos]) == ORANGE}
    dist = _no_valve_seed_dist(neighbors, seeds, valve_cells)
    allowed: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for valve in valve_cells:
        neigh = neighbors[valve]
        finite = [dist[n] for n in neigh if n in dist]
        if not finite:
            allowed[valve] = set()
            continue
        best = min(finite)
        allowed[valve] = {n for n in neigh if dist.get(n) == best}
    return allowed


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _corner_bg(g)
    seeds = _seed_positions(g, bg)
    neighbors = _network_neighbors(g, bg)
    valve_entry = _valve_allowed_entries(g, neighbors, seeds)
    seed_cells = set(seeds)

    arrival = -np.ones((h, w), dtype=int)
    owner = -np.ones((h, w), dtype=int)
    frontier: list[tuple[tuple[int, int], int]] = []

    for pos, color in seeds.items():
        r, c = pos
        arrival[r, c] = 0
        owner[r, c] = color
        frontier.append((pos, color))

    time = 0
    while frontier:
        proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
        for pos, color in frontier:
            for nxt in neighbors.get(pos, []):
                nr, nc = nxt
                if nxt in seed_cells:
                    continue
                if arrival[nr, nc] != -1:
                    continue
                if int(g[nr, nc]) == ORANGE and pos not in valve_entry.get(nxt, set()):
                    continue
                proposals[nxt].add(int(color))

        next_frontier: list[tuple[tuple[int, int], int]] = []
        for nxt, colors in proposals.items():
            nr, nc = nxt
            arrival[nr, nc] = time + 1
            if len(colors) == 1:
                color = next(iter(colors))
                owner[nr, nc] = color
                next_frontier.append((nxt, color))
            else:
                owner[nr, nc] = CONTESTED
        frontier = next_frontier
        time += 1

    out = g.copy()
    contested_mask = owner == CONTESTED
    for r, c in np.argwhere(owner >= 0):
        rr, cc = int(r), int(c)
        if (rr, cc) in seed_cells:
            continue
        out[rr, cc] = int(owner[rr, cc])
    out[contested_mask & (g == GREY)] = GREY
    out[contested_mask & (g == ORANGE)] = ORANGE

    if not return_meta:
        return out

    valve_mask = g == ORANGE
    meta = {
        "contested_count": int(np.count_nonzero(contested_mask)),
        "passed_valves": int(np.count_nonzero(valve_mask & (out != ORANGE))),
        "untouched_valves": int(np.count_nonzero(valve_mask & (out == ORANGE))),
        "seed_count": len(seeds),
        "pipe_count": int(np.count_nonzero(np.isin(g, (GREY, ORANGE)))),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate endpoint seed colors through a grey pipe network while freezing collisions.

    Input structure:
    - The background is black.
    - Grey pixels form a 4-connected pipe network made of one-pixel horizontal and
      vertical segments, with some degree-3/4 junctions where segments intersect.
    - Distinct non-grey, non-orange endpoint pixels are colored seeds attached to
      pipe leaves.
    - Orange pixels are valves inserted on pipe segments instead of grey pixels.

    Rule:
    - Each seed expands step-by-step through adjacent pipe cells, recoloring visited
      grey pixels to the seed color.
    - A valve can only be entered from the side that has the nearest seed without
      another valve in between; if no seed can reach a side without crossing a valve,
      that side is blocked. A passed valve recolors to the passing color.
    - If different colors try to enter the same pipe pixel on the same expansion
      step, that pixel freezes at its original structural color and propagation does
      not continue through it.
    - Seed pixels stay fixed, untouched valves stay orange, and all other unreachable
      structural pixels remain grey.

    Key insight:
    - The task is a simultaneous multi-source flood on an orthogonal graph, with
      valve direction inferred from seed reachability after removing other valves,
      and collision points acting as neutral sinks that stop further spread.
    """
    return _solve(np.asarray(grid, dtype=int), return_meta=False)


def _paint_paths(
    shape: tuple[int, int],
    draw_paths: list[list[tuple[int, int]]],
    branch_paths: dict[str, list[tuple[int, int]]],
    seed_labels: list[str],
    valve_cells: list[tuple[int, int]],
    seed_palette: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in draw_paths:
        for r, c in path:
            grid[r, c] = GREY
    for r, c in valve_cells:
        grid[r, c] = ORANGE
    for label, color in zip(seed_labels, seed_palette):
        sr, sc = branch_paths[label][0]
        grid[sr, sc] = color
    return grid


def _branch_internal_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return path[1:-1]


def _choose_passed_valve(
    rng: random.Random,
    branch_paths: dict[str, list[tuple[int, int]]],
    seed_labels: list[str],
) -> tuple[str, tuple[int, int]] | None:
    candidates = [label for label in seed_labels if len(_branch_internal_cells(branch_paths[label])) >= 1]
    if not candidates:
        return None
    label = rng.choice(candidates)
    internal = _branch_internal_cells(branch_paths[label])
    cell = internal[min(len(internal) - 1, rng.randint(0, min(1, len(internal) - 1)))]
    return label, cell


def _build_cross(rng: random.Random, want_collision: bool) -> np.ndarray:
    dirs = ["U", "D", "L", "R"]
    pair = rng.choice([("U", "D"), ("L", "R"), ("U", "L"), ("U", "R"), ("D", "L"), ("D", "R")])
    lengths = {d: rng.randint(3, 8) for d in dirs}
    seed_labels = [pair[0], pair[1]]
    free_dirs = [d for d in dirs if d not in seed_labels]
    if want_collision:
        shared = rng.randint(4, 8)
        lengths[pair[0]] = shared
        lengths[pair[1]] = shared
        if free_dirs and rng.random() < 0.35:
            extra = rng.choice(free_dirs)
            lengths[extra] = shared
            seed_labels.append(extra)
    else:
        a = rng.randint(3, 6)
        b = rng.randint(a + 1, 8)
        lengths[pair[0]] = a
        lengths[pair[1]] = b
        if free_dirs and rng.random() < 0.25:
            extra = rng.choice(free_dirs)
            choices = [v for v in range(3, 9) if v not in {a, b}]
            lengths[extra] = rng.choice(choices)
            seed_labels.append(extra)

    pad_t = rng.randint(2, 4)
    pad_b = rng.randint(2, 4)
    pad_l = rng.randint(2, 4)
    pad_r = rng.randint(2, 4)
    h = pad_t + lengths["U"] + 1 + lengths["D"] + pad_b
    w = pad_l + lengths["L"] + 1 + lengths["R"] + pad_r
    center = (pad_t + lengths["U"], pad_l + lengths["L"])

    branch_paths = {
        "U": _line_path((center[0] - lengths["U"], center[1]), center),
        "D": _line_path((center[0] + lengths["D"], center[1]), center),
        "L": _line_path((center[0], center[1] - lengths["L"]), center),
        "R": _line_path((center[0], center[1] + lengths["R"]), center),
    }

    valve_cells: list[tuple[int, int]] = []
    picked = _choose_passed_valve(rng, branch_paths, seed_labels)
    if picked is not None:
        valve_cells.append(picked[1])

    if want_collision:
        untouched_candidates = [d for d in dirs if d not in seed_labels and len(_branch_internal_cells(branch_paths[d])) >= 1]
        if untouched_candidates:
            label = rng.choice(untouched_candidates)
            internal = _branch_internal_cells(branch_paths[label])
            valve_cells.append(internal[rng.randrange(len(internal))])

    palette = rng.sample(SEED_COLORS, len(seed_labels))
    draw_paths = list(branch_paths.values())
    return _paint_paths((h, w), draw_paths, branch_paths, seed_labels, valve_cells, palette)


def _build_double_v(rng: random.Random, want_collision: bool) -> np.ndarray:
    target = rng.choice(["upper", "lower"])

    top = rng.randint(3, 7)
    bottom = rng.randint(3, 7)
    gap = rng.randint(4, 7)
    left_u = rng.randint(3, 7)
    right_u = rng.randint(3, 7)
    left_l = rng.randint(3, 7)
    right_l = rng.randint(3, 7)

    if target == "upper":
        if want_collision:
            shared = rng.randint(4, 7)
            top = shared
            if rng.random() < 0.5:
                left_u = shared
                seed_labels = ["TOP", "UL"]
            else:
                right_u = shared
                seed_labels = ["TOP", "UR"]
        else:
            top, left_u = sorted(rng.sample(range(3, 8), 2))
            if rng.random() < 0.5:
                seed_labels = ["TOP", "UL"]
            else:
                top, right_u = sorted(rng.sample(range(3, 8), 2))
                seed_labels = ["TOP", "UR"]
    else:
        if want_collision:
            shared = rng.randint(4, 7)
            bottom = shared
            if rng.random() < 0.5:
                left_l = shared
                seed_labels = ["BOT", "LL"]
            else:
                right_l = shared
                seed_labels = ["BOT", "LR"]
        else:
            bottom, left_l = sorted(rng.sample(range(3, 8), 2))
            if rng.random() < 0.5:
                seed_labels = ["BOT", "LL"]
            else:
                bottom, right_l = sorted(rng.sample(range(3, 8), 2))
                seed_labels = ["BOT", "LR"]

    left_span = max(left_u, left_l)
    right_span = max(right_u, right_l)
    pad_t = rng.randint(2, 4)
    pad_b = rng.randint(2, 4)
    pad_l = rng.randint(2, 4)
    pad_r = rng.randint(2, 4)
    h = pad_t + top + gap + bottom + 1 + pad_b
    w = pad_l + left_span + 1 + right_span + pad_r

    col = pad_l + left_span
    upper = (pad_t + top, col)
    lower = (upper[0] + gap, col)

    branch_paths = {
        "TOP": _line_path((upper[0] - top, col), upper),
        "UL": _line_path((upper[0], col - left_u), upper),
        "UR": _line_path((upper[0], col + right_u), upper),
        "LL": _line_path((lower[0], col - left_l), lower),
        "LR": _line_path((lower[0], col + right_l), lower),
        "BOT": _line_path((lower[0] + bottom, col), lower),
    }
    connector = _line_path(upper, lower)

    valve_cells: list[tuple[int, int]] = []
    picked = _choose_passed_valve(rng, branch_paths, seed_labels)
    if picked is not None:
        valve_cells.append(picked[1])

    if want_collision:
        if target == "upper":
            connector_internal = connector[1:-1]
            if connector_internal:
                valve_cells.append(rng.choice(connector_internal))
        else:
            connector_internal = connector[1:-1]
            if connector_internal:
                valve_cells.append(rng.choice(connector_internal))

    palette = rng.sample(SEED_COLORS, len(seed_labels))
    draw_paths = list(branch_paths.values()) + [connector]
    return _paint_paths((h, w), draw_paths, branch_paths, seed_labels, valve_cells, palette)


def _build_double_h(rng: random.Random, want_collision: bool) -> np.ndarray:
    target = rng.choice(["left", "right"])

    left = rng.randint(3, 7)
    right = rng.randint(3, 7)
    gap = rng.randint(4, 7)
    up_l = rng.randint(3, 7)
    down_l = rng.randint(3, 7)
    up_r = rng.randint(3, 7)
    down_r = rng.randint(3, 7)

    if target == "left":
        if want_collision:
            shared = rng.randint(4, 7)
            left = shared
            if rng.random() < 0.5:
                up_l = shared
                seed_labels = ["LFT", "LU"]
            else:
                down_l = shared
                seed_labels = ["LFT", "LD"]
        else:
            left, up_l = sorted(rng.sample(range(3, 8), 2))
            if rng.random() < 0.5:
                seed_labels = ["LFT", "LU"]
            else:
                left, down_l = sorted(rng.sample(range(3, 8), 2))
                seed_labels = ["LFT", "LD"]
    else:
        if want_collision:
            shared = rng.randint(4, 7)
            right = shared
            if rng.random() < 0.5:
                up_r = shared
                seed_labels = ["RGT", "RU"]
            else:
                down_r = shared
                seed_labels = ["RGT", "RD"]
        else:
            right, up_r = sorted(rng.sample(range(3, 8), 2))
            if rng.random() < 0.5:
                seed_labels = ["RGT", "RU"]
            else:
                right, down_r = sorted(rng.sample(range(3, 8), 2))
                seed_labels = ["RGT", "RD"]

    top_span = max(up_l, up_r)
    bottom_span = max(down_l, down_r)
    pad_t = rng.randint(2, 4)
    pad_b = rng.randint(2, 4)
    pad_l = rng.randint(2, 4)
    pad_r = rng.randint(2, 4)
    h = pad_t + top_span + 1 + bottom_span + pad_b
    w = pad_l + left + gap + right + 1 + pad_r

    row = pad_t + top_span
    left_j = (row, pad_l + left)
    right_j = (row, left_j[1] + gap)

    branch_paths = {
        "LFT": _line_path((row, left_j[1] - left), left_j),
        "LU": _line_path((row - up_l, left_j[1]), left_j),
        "LD": _line_path((row + down_l, left_j[1]), left_j),
        "RU": _line_path((row - up_r, right_j[1]), right_j),
        "RD": _line_path((row + down_r, right_j[1]), right_j),
        "RGT": _line_path((row, right_j[1] + right), right_j),
    }
    connector = _line_path(left_j, right_j)

    valve_cells: list[tuple[int, int]] = []
    picked = _choose_passed_valve(rng, branch_paths, seed_labels)
    if picked is not None:
        valve_cells.append(picked[1])

    if want_collision:
        connector_internal = connector[1:-1]
        if connector_internal:
            valve_cells.append(rng.choice(connector_internal))

    palette = rng.sample(SEED_COLORS, len(seed_labels))
    draw_paths = list(branch_paths.values()) + [connector]
    return _paint_paths((h, w), draw_paths, branch_paths, seed_labels, valve_cells, palette)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_collision = rng.random() < 0.5
    builder = rng.choice([_build_cross, _build_double_v, _build_double_h])
    grid = builder(rng, want_collision)
    if max(grid.shape) > 30:
        return None

    out, meta = _solve(grid, return_meta=True)
    if meta["seed_count"] < 2:
        return None
    if meta["pipe_count"] < 12:
        return None
    if meta["passed_valves"] < 1:
        return None
    if want_collision and meta["contested_count"] < 1:
        return None
    if (not want_collision) and meta["contested_count"] != 0:
        return None
    if want_collision and meta["untouched_valves"] < 1:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
