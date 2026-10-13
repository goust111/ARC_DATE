from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

TOOTH_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)
CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

CW = 1
CCW = -1

MAX_SIDE = 30
MAX_GEARS = 8
STEP_CHOICES = (4, 5, 6)
MARGIN_CHOICES = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _is_tooth_color(value: int) -> bool:
    return int(value) in TOOTH_COLORS


def _gear_at(grid: np.ndarray, r: int, c: int) -> dict | None:
    h, w = grid.shape
    if not (0 < r < h - 1 and 0 < c < w - 1):
        return None

    center = int(grid[r, c])
    if center not in (GREY, MAROON):
        return None

    teeth = []
    for dr, dc in CARDINAL_DIRS:
        value = int(grid[r + dr, c + dc])
        if not _is_tooth_color(value):
            return None
        teeth.append(value)

    nw = int(grid[r - 1, c - 1])
    ne = int(grid[r - 1, c + 1])
    sw = int(grid[r + 1, c - 1])
    se = int(grid[r + 1, c + 1])

    if sw != BLACK or se != BLACK:
        return None

    if center == GREY:
        if nw != BLACK or ne != BLACK:
            return None
        spin = None
    else:
        if (nw, ne) == (ORANGE, BLACK):
            spin = CW
        elif (nw, ne) == (BLACK, ORANGE):
            spin = CCW
        else:
            return None

    return {
        "center": (int(r), int(c)),
        "driver": center == MAROON,
        "input_spin": spin,
        "teeth": tuple(map(int, teeth)),
    }


def _find_gears(grid: np.ndarray) -> tuple[dict[tuple[int, int], dict], tuple[int, int], int]:
    gears: dict[tuple[int, int], dict] = {}
    mask = (grid == GREY) | (grid == MAROON)
    for r, c in np.argwhere(mask):
        gear = _gear_at(grid, int(r), int(c))
        if gear is not None:
            gears[(int(r), int(c))] = gear

    assert gears, "expected at least one gear"
    drivers = [center for center, gear in gears.items() if gear["driver"]]
    assert len(drivers) == 1, f"expected one driver gear, got {len(drivers)}"
    driver = drivers[0]
    root_spin = gears[driver]["input_spin"]
    assert root_spin in (CW, CCW)
    return gears, driver, int(root_spin)


def _trace_neighbor(
    grid: np.ndarray,
    center: tuple[int, int],
    dr: int,
    dc: int,
    centers: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = center
    rr = r + 2 * dr
    cc = c + 2 * dc
    if not (0 <= rr < h and 0 <= cc < w):
        return None
    if int(grid[rr, cc]) != AZURE:
        return None

    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == AZURE:
        rr += dr
        cc += dc

    assert 0 <= rr < h and 0 <= cc < w, "connection exits the grid"
    assert _is_tooth_color(int(grid[rr, cc])), "connection must terminate at a tooth"

    nr = rr + dr
    nc = cc + dc
    neighbor = (int(nr), int(nc))
    assert neighbor in centers, "connection must terminate at another gear"
    return neighbor


def _build_graph(grid: np.ndarray, gears: dict[tuple[int, int], dict]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph = {center: set() for center in gears}
    centers = set(gears)
    for center in gears:
        r, c = center
        for dr, dc in CARDINAL_DIRS:
            neighbor = _trace_neighbor(grid, (r, c), dr, dc, centers)
            if neighbor is None:
                continue
            nr, nc = neighbor
            assert (nr - r, nc - c) != (0, 0)
            graph[center].add(neighbor)
            graph[neighbor].add(center)

    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    assert edge_count == len(gears) - 1, "gear links must form a tree"
    return graph


def _propagate_spins(
    driver: tuple[int, int],
    root_spin: int,
    graph: dict[tuple[int, int], set[tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    spins = {driver: int(root_spin)}
    queue = [driver]
    idx = 0
    while idx < len(queue):
        center = queue[idx]
        idx += 1
        for neighbor in graph[center]:
            expected = -spins[center]
            if neighbor in spins:
                assert spins[neighbor] == expected, "inconsistent spin propagation"
                continue
            spins[neighbor] = expected
            queue.append(neighbor)

    assert len(spins) == len(graph), "all gears must be connected to the driver"
    return spins


def _rotate_teeth(teeth: tuple[int, int, int, int], spin: int) -> tuple[int, int, int, int]:
    up, right, down, left = map(int, teeth)
    if spin == CW:
        return (left, up, right, down)
    assert spin == CCW
    return (right, down, left, up)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the colored teeth of every connected gear by propagating the driver direction across the tree.

    Input structure
    - Each gear is a 3x3 cross: a grey or maroon center, four colored teeth, and black corners.
    - Exactly one maroon center is the driver gear.
    - An orange pixel at the driver's upper-left corner means clockwise; at the upper-right corner means counterclockwise.
    - Azure horizontal/vertical line segments connect opposing teeth and form a tree.

    Rule
    - Start from the driver direction and flip the rotation direction across every connection.
    - For each gear, read the teeth in [up, right, down, left] order.
    - Clockwise gears shift that cycle one step clockwise; counterclockwise gears shift it one step the other way.
    - The driver center becomes grey and the orange marker disappears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gears, driver, root_spin = _find_gears(g)
    graph = _build_graph(g, gears)
    spins = _propagate_spins(driver, root_spin, graph)

    out = g.copy()
    for center, gear in gears.items():
        r, c = center
        rotated = _rotate_teeth(gear["teeth"], spins[center])
        out[r - 1, c] = rotated[0]
        out[r, c + 1] = rotated[1]
        out[r + 1, c] = rotated[2]
        out[r, c - 1] = rotated[3]
        out[r, c] = GREY

    dr, dc = driver
    out[dr - 1, dc - 1] = BLACK
    out[dr - 1, dc + 1] = BLACK
    return out


def _sample_tooth_cycle() -> tuple[int, int, int, int]:
    while True:
        teeth = tuple(int(random.choice(TOOTH_COLORS)) for _ in range(4))
        if len(set(teeth)) >= 2:
            return teeth


def _feasible_margin_pairs(span: int) -> list[tuple[int, int]]:
    pairs = []
    for before in MARGIN_CHOICES:
        for after in MARGIN_CHOICES:
            if span + before + after <= MAX_SIDE:
                pairs.append((before, after))
    return pairs


def _sample_lattice() -> tuple[int, int, int, tuple[int, int], tuple[int, int]] | None:
    step = random.choice(STEP_CHOICES)
    shapes = []
    for rows in range(1, 6):
        for cols in range(1, 6):
            if rows * cols < 2:
                continue
            row_span = 3 + (rows - 1) * step
            col_span = 3 + (cols - 1) * step
            if not _feasible_margin_pairs(row_span) or not _feasible_margin_pairs(col_span):
                continue
            shapes.append((rows, cols, row_span, col_span))
    if not shapes:
        return None

    rows, cols, row_span, col_span = random.choice(shapes)
    top_bottom = random.choice(_feasible_margin_pairs(row_span))
    left_right = random.choice(_feasible_margin_pairs(col_span))
    return rows, cols, step, top_bottom, left_right


def _grow_tree(rows: int, cols: int, count: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    root = (random.randrange(rows), random.randrange(cols))
    nodes = [root]
    occupied = {root}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(nodes) < count:
        candidates = []
        for r, c in nodes:
            for dr, dc in CARDINAL_DIRS:
                nr = r + dr
                nc = c + dc
                pos = (nr, nc)
                if not (0 <= nr < rows and 0 <= nc < cols) or pos in occupied:
                    continue
                occupied_neighbors = 0
                for adr, adc in CARDINAL_DIRS:
                    ar = nr + adr
                    ac = nc + adc
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) in occupied:
                        occupied_neighbors += 1
                if occupied_neighbors == 1:
                    candidates.append(((r, c), pos))
        if not candidates:
            return None
        parent, child = random.choice(candidates)
        occupied.add(child)
        nodes.append(child)
        edges.append((parent, child))

    return nodes, edges


def _grid_center(
    lattice_pos: tuple[int, int],
    step: int,
    top_margin: int,
    left_margin: int,
) -> tuple[int, int]:
    lr, lc = lattice_pos
    return (int(top_margin + 1 + lr * step), int(left_margin + 1 + lc * step))


def _render_gear(grid: np.ndarray, center: tuple[int, int], teeth: tuple[int, int, int, int], *, driver: bool, spin: int) -> None:
    r, c = center
    grid[r, c] = MAROON if driver else GREY
    grid[r - 1, c] = teeth[0]
    grid[r, c + 1] = teeth[1]
    grid[r + 1, c] = teeth[2]
    grid[r, c - 1] = teeth[3]
    if driver:
        if spin == CW:
            grid[r - 1, c - 1] = ORANGE
        else:
            grid[r - 1, c + 1] = ORANGE


def _render_edge(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        left = min(ac, bc)
        right = max(ac, bc)
        grid[ar, left + 2 : right - 1] = AZURE
        return
    assert ac == bc
    top = min(ar, br)
    bottom = max(ar, br)
    grid[top + 2 : bottom - 1, ac] = AZURE


def _gen_attempt() -> np.ndarray | None:
    lattice = _sample_lattice()
    if lattice is None:
        return None
    rows, cols, step, (top_margin, bottom_margin), (left_margin, right_margin) = lattice

    max_gears = min(MAX_GEARS, rows * cols)
    gear_count = random.randint(2, max_gears)
    grown = _grow_tree(rows, cols, gear_count)
    if grown is None:
        return None
    nodes, edges = grown

    height = top_margin + (3 + (rows - 1) * step) + bottom_margin
    width = left_margin + (3 + (cols - 1) * step) + right_margin
    grid = np.full((height, width), BLACK, dtype=int)

    root = nodes[0]
    root_spin = random.choice((CW, CCW))
    centers = {}
    for node in nodes:
        centers[node] = _grid_center(node, step, top_margin, left_margin)

    for parent, child in edges:
        _render_edge(grid, centers[parent], centers[child])

    for node in nodes:
        teeth = _sample_tooth_cycle()
        _render_gear(grid, centers[node], teeth, driver=node == root, spin=root_spin)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid gear-transmission input")
