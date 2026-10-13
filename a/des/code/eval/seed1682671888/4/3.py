from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
YELLOW = 4
ORANGE = 7
MAROON = 9
INDICATORS = {YELLOW, ORANGE}
CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_HEIGHT_RANGE = (28, 36)
PATH_COUNT_CHOICES = [1, 2, 3]
BOX_WIDTH = 16
BOX_GAP = 3
LOCAL_ROWS_CHOICES = [6, 7, 8]
LOCAL_COLS_CHOICES = [4, 5]
MAIN_EDGE_RANGE = (4, 6)
EXTRA_EDGE_RANGE = (3, 5)
MAX_ATTEMPTS = 300


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return -d[1], d[0]


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return d[1], -d[0]


def _opposite(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], pos: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for d in CARDINALS:
        nxt = _add(pos, d)
        if _in_bounds(shape, nxt):
            out.append(nxt)
    return out


def _relation(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str:
    if out_dir == in_dir:
        return "straight"
    if out_dir == _left(in_dir):
        return "left"
    if out_dir == _right(in_dir):
        return "right"
    raise ValueError(f"invalid turn from {in_dir} to {out_dir}")


def _trace_path(grid: np.ndarray, start: tuple[int, int], out: np.ndarray) -> tuple[int, tuple[int, int]]:
    color = int(grid[start])
    grey_neighbors = [p for p in _neighbors4(grid.shape, start) if int(grid[p]) == GREY]
    assert len(grey_neighbors) == 1, "start must touch exactly one grey path cell"

    prev = start
    cur = grey_neighbors[0]
    direction = _sub(cur, start)
    count = 2
    out[cur] = color

    for _ in range(grid.size):
        options: list[tuple[tuple[int, int], tuple[int, int]]] = []
        grey_deg = 0
        for d in CARDINALS:
            nxt = _add(cur, d)
            if nxt == prev or not _in_bounds(grid.shape, nxt):
                continue
            val = int(grid[nxt])
            if val == GREY or val == MAROON:
                options.append((d, nxt))
            if val == GREY:
                grey_deg += 1

        assert options, "path terminated before reaching the endpoint"

        if grey_deg >= 2:
            chosen_dir = _choose_direction(grid, cur, prev, direction)
            nxt = _add(cur, chosen_dir)
            if not _in_bounds(grid.shape, nxt) or int(grid[nxt]) not in (GREY, MAROON):
                by_dir = {d: p for d, p in options}
                prefs = [direction, _left(direction), _right(direction)]
                nxt = None
                for cand in prefs:
                    if cand in by_dir:
                        chosen_dir = cand
                        nxt = by_dir[cand]
                        break
                if nxt is None:
                    chosen_dir, nxt = options[0]
            else:
                chosen_dir = chosen_dir
        else:
            if len(options) == 1:
                chosen_dir, nxt = options[0]
            else:
                by_dir = {d: p for d, p in options}
                for cand in (direction, _left(direction), _right(direction)):
                    if cand in by_dir:
                        chosen_dir, nxt = cand, by_dir[cand]
                        break
                else:
                    chosen_dir, nxt = options[0]

        if int(grid[nxt]) == MAROON:
            count += 1
            return count, nxt

        prev, cur, direction = cur, nxt, chosen_dir
        count += 1
        out[cur] = color

    raise AssertionError("path traversal exceeded the step budget")


def _find_indicator(
    grid: np.ndarray,
    cur: tuple[int, int],
    prev: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[str, int] | None:
    for anchor in (cur, prev):
        probes = [
            ("left", _add(anchor, _left(direction))),
            ("right", _add(anchor, _right(direction))),
            ("back", _add(anchor, _opposite(direction))),
        ]
        for rel, pos in probes:
            if _in_bounds(grid.shape, pos) and int(grid[pos]) in INDICATORS:
                return rel, int(grid[pos])
    return None


def _choose_direction(
    grid: np.ndarray,
    cur: tuple[int, int],
    prev: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    info = _find_indicator(grid, cur, prev, direction)
    if info is None:
        prefs = [direction, _left(direction), _right(direction)]
    else:
        rel, color = info
        if color == ORANGE or rel == "back":
            prefs = [direction, _left(direction), _right(direction)]
        elif rel == "left":
            prefs = [_left(direction), direction, _right(direction)]
        else:
            prefs = [_right(direction), direction, _left(direction)]

    for cand in prefs:
        nxt = _add(cur, cand)
        if nxt != prev and _in_bounds(grid.shape, nxt) and int(grid[nxt]) in (GREY, MAROON):
            return cand

    raise AssertionError("no valid outgoing branch at an intersection")


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each colored start along the grey network and raise a pillar at its maroon endpoint."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _mode_color(g)

    reserved = {bg, GREY, MAROON, YELLOW, ORANGE}
    starts: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            val = int(g[r, c])
            if val in reserved:
                continue
            if sum(int(g[p]) == GREY for p in _neighbors4(g.shape, (r, c))) == 1:
                starts.append((r, c))

    assert starts, "expected at least one colored start marker"

    out = g.copy()
    for start in starts:
        color = int(g[start])
        length, end = _trace_path(g, start, out)
        er, ec = end
        for rr in range(er, er - length, -1):
            if rr < 0:
                break
            if int(g[rr, ec]) in INDICATORS:
                continue
            out[rr, ec] = color

    return out


def _lattice_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = node
    for dr, dc in CARDINALS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _random_walk(
    rows: int,
    cols: int,
    edges: int,
    required: set[str],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[str]] | None:
    for _ in range(200):
        nodes = [(
            rng.randrange(rows),
            rng.randrange(cols),
        )]
        used = {nodes[0]}
        while len(nodes) < edges + 1:
            cur = nodes[-1]
            choices = [p for p in _lattice_neighbors(cur, rows, cols) if p not in used]
            if not choices:
                break
            nxt = rng.choice(choices)
            nodes.append(nxt)
            used.add(nxt)
        if len(nodes) != edges + 1:
            continue

        dirs = []
        for a, b in zip(nodes, nodes[1:]):
            dirs.append((b[0] - a[0], b[1] - a[1]))
        actions = [_relation(a, b) for a, b in zip(dirs, dirs[1:])]
        if required <= set(actions):
            return nodes, actions
    return None


def _route_cells(nodes: list[tuple[int, int]], spacing: int = 3) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cell_nodes = [(r * spacing, c * spacing) for r, c in nodes]
    route = [cell_nodes[0]]
    for a, b in zip(cell_nodes, cell_nodes[1:]):
        d = ((b[0] - a[0]) // spacing, (b[1] - a[1]) // spacing)
        for k in range(1, spacing + 1):
            route.append((a[0] + d[0] * k, a[1] + d[1] * k))
    return cell_nodes, route


def _build_network(
    rows: int,
    cols: int,
    edges: int,
    required: set[str],
    start_color: int,
    rng: random.Random,
) -> dict | None:
    walk = _random_walk(rows, cols, edges, required, rng)
    if walk is None:
        return None
    nodes, actions = walk
    cell_nodes, route = _route_cells(nodes)
    route_set = set(route)

    first_dir = _sub(cell_nodes[1], cell_nodes[0])
    first_dir = (np.sign(first_dir[0]).item(), np.sign(first_dir[1]).item())
    last_dir = _sub(cell_nodes[-1], cell_nodes[-2])
    last_dir = (np.sign(last_dir[0]).item(), np.sign(last_dir[1]).item())
    start = _sub(cell_nodes[0], first_dir)
    end = _add(cell_nodes[-1], last_dir)

    occupied = set(route_set)
    indicators: dict[tuple[int, int], int] = {}
    stubs: set[tuple[int, int]] = set()

    if start in occupied or end in occupied:
        return None
    occupied.add(start)
    occupied.add(end)

    for i, center in enumerate(cell_nodes[1:-1], start=1):
        incoming = _sub(cell_nodes[i], cell_nodes[i - 1])
        outgoing = _sub(cell_nodes[i + 1], cell_nodes[i])
        incoming = (np.sign(incoming[0]).item(), np.sign(incoming[1]).item())
        outgoing = (np.sign(outgoing[0]).item(), np.sign(outgoing[1]).item())
        action = _relation(incoming, outgoing)
        prev = _sub(center, incoming)

        if action == "straight":
            side = rng.choice([_left(incoming), _right(incoming)])
            indicator_color = ORANGE
        elif action == "left":
            side = _left(incoming)
            indicator_color = YELLOW
        else:
            side = _right(incoming)
            indicator_color = YELLOW

        indicator_pos = _add(prev, side)
        if indicator_pos in occupied or indicator_pos in indicators:
            return None
        indicators[indicator_pos] = indicator_color
        occupied.add(indicator_pos)

        unused = [
            d
            for d in CARDINALS
            if d != _opposite(incoming) and d != outgoing
        ]
        assert unused
        chosen = unused if rng.random() < 0.5 else [rng.choice(unused)]
        added = 0
        for d in chosen:
            stub = _add(center, d)
            if stub in occupied:
                continue
            stubs.add(stub)
            occupied.add(stub)
            added += 1
        if added == 0:
            return None

    length = len(route) + 2
    pillar = {(end[0] - k, end[1]) for k in range(length)}
    if any((cell != end and cell in occupied) for cell in pillar):
        return None

    return {
        "start_color": start_color,
        "route": route,
        "route_set": route_set,
        "start": start,
        "end": end,
        "indicators": indicators,
        "stubs": stubs,
        "length": length,
        "actions": actions,
    }


def _place_network(
    grid: np.ndarray,
    box_col0: int,
    box_w: int,
    network: dict,
    bg: int,
    rng: random.Random,
) -> bool:
    occupied = (
        set(network["route"])
        | set(network["stubs"])
        | set(network["indicators"])
        | {network["start"], network["end"]}
    )
    min_r = min(r for r, _ in occupied)
    max_r = max(r for r, _ in occupied)
    min_c = min(c for _, c in occupied)
    max_c = max(c for _, c in occupied)
    end_r, end_c = network["end"]
    length = int(network["length"])

    low_r = max(-min_r, length - 1 - end_r)
    high_r = grid.shape[0] - 1 - max_r
    low_c = box_col0 - min_c
    high_c = box_col0 + box_w - 1 - max_c
    if low_r > high_r or low_c > high_c:
        return False

    off_r = rng.randint(low_r, high_r)
    off_c = rng.randint(low_c, high_c)
    translate = lambda p: (p[0] + off_r, p[1] + off_c)

    translated = {translate(p) for p in occupied}
    if any(int(grid[p]) != bg for p in translated):
        return False

    end_abs = translate(network["end"])
    pillar = [(end_abs[0] - k, end_abs[1]) for k in range(length)]
    if any(not _in_bounds(grid.shape, p) for p in pillar):
        return False
    if any((p != end_abs and p in translated) for p in pillar):
        return False

    for p in network["route"]:
        grid[translate(p)] = GREY
    for p in network["stubs"]:
        grid[translate(p)] = GREY
    for p, color in network["indicators"].items():
        grid[translate(p)] = color
    grid[translate(network["start"])] = int(network["start_color"])
    grid[translate(network["end"])] = MAROON
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    n_paths = rng.choice(PATH_COUNT_CHOICES)
    width = 2 + n_paths * BOX_WIDTH + (n_paths - 1) * BOX_GAP

    reserved = {GREY, YELLOW, ORANGE, MAROON}
    palette = [c for c in range(10) if c not in reserved]
    bg = rng.choice(palette)
    start_pool = [c for c in palette if c != bg]
    rng.shuffle(start_pool)
    if len(start_pool) < n_paths:
        return None

    grid = np.full((height, width), bg, dtype=int)
    all_actions: list[str] = []
    for idx in range(n_paths):
        rows = rng.choice(LOCAL_ROWS_CHOICES)
        cols = rng.choice(LOCAL_COLS_CHOICES)
        edges = rng.randint(*(MAIN_EDGE_RANGE if idx == 0 else EXTRA_EDGE_RANGE))
        required = {"left", "right", "straight"} if idx == 0 else set()
        network = _build_network(rows, cols, edges, required, start_pool[idx], rng)
        if network is None:
            return None
        all_actions.extend(network["actions"])
        box_col0 = 1 + idx * (BOX_WIDTH + BOX_GAP)
        if not _place_network(grid, box_col0, BOX_WIDTH, network, bg, rng):
            return None

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] <= max(ct for color, ct in counts.items() if color != bg):
        return None
    if not {"left", "right", "straight"} <= set(all_actions):
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
