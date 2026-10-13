import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5

ORTHO_OFFSETS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEAR_COLORS = (1, 2, 3, 6, 7, 8, 9)

GEN_RETRIES = 400
LATTICE_ROWS_RANGE = (2, 5)
LATTICE_COLS_RANGE = (2, 5)
PADDING_RANGE = (0, 2)


def _rotate_teeth(teeth: tuple[int, int, int, int], turn: int) -> tuple[int, int, int, int]:
    if turn > 0:
        return (teeth[3], teeth[0], teeth[1], teeth[2])
    if turn < 0:
        return (teeth[1], teeth[2], teeth[3], teeth[0])
    return teeth


def _lattice_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _gear_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    cells = [(r, c)]
    cells.extend((r + dr, c + dc) for dr, dc in ORTHO_OFFSETS)
    return cells


def _find_gears(grid: np.ndarray) -> dict[tuple[int, int], dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gears: dict[tuple[int, int], dict] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = int(g[r, c])
            if center_color in (BLACK, GREY, YELLOW):
                continue

            diag_vals = [int(g[r + dr, c + dc]) for dr in (-1, 1) for dc in (-1, 1)]
            if any(val != BLACK for val in diag_vals):
                continue

            teeth = tuple(int(g[r + dr, c + dc]) for dr, dc in ORTHO_OFFSETS)
            if any(val not in (GREY, YELLOW) for val in teeth):
                continue
            if teeth.count(YELLOW) > 1:
                continue

            gears[(r, c)] = {
                "center": (r, c),
                "center_color": center_color,
                "teeth": teeth,
                "is_active": teeth.count(YELLOW) == 1,
            }

    assert gears, "expected at least one gear"
    active_gears = [gear for gear in gears.values() if gear["is_active"]]
    assert len(active_gears) == 1, "expected exactly one active gear"
    return gears


def _mesh_graph(centers: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    center_set = set(centers)
    graph = {center: [] for center in centers}
    for r, c in centers:
        for dr, dc in ((-4, 0), (4, 0), (0, -4), (0, 4)):
            nxt = (r + dr, c + dc)
            if nxt in center_set:
                graph[(r, c)].append(nxt)
    return graph


def _rotation_map(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    active_center: tuple[int, int],
) -> dict[tuple[int, int], int]:
    turns = {active_center: 1}
    queue = deque([active_center])
    while queue:
        current = queue.popleft()
        for nxt in graph[current]:
            if nxt in turns:
                continue
            turns[nxt] = -turns[current]
            queue.append(nxt)
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify 3x3 gears, build the meshing graph, and rotate every reachable gear
    from the unique active gear with alternating directions along the chain.
    """
    g = np.asarray(grid, dtype=int)
    gears = _find_gears(g)
    graph = _mesh_graph(list(gears))

    active_center = next(center for center, gear in gears.items() if gear["is_active"])
    turns = _rotation_map(graph, active_center)

    out = g.copy()
    for r, c in gears:
        out[r - 1 : r + 2, c - 1 : c + 2] = BLACK

    for center, gear in gears.items():
        r, c = center
        teeth = _rotate_teeth(gear["teeth"], turns.get(center, 0))
        out[r, c] = gear["center_color"]
        for (dr, dc), color in zip(ORTHO_OFFSETS, teeth):
            out[r + dr, c + dc] = color

    return out


def _grow_path(rows: int, cols: int, size: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    path = [start]
    occupied = {start}

    while len(occupied) < size:
        choices: list[tuple[int, tuple[int, int]]] = []
        endpoints = [(0, path[0]), (1, path[-1])]
        for side, cell in endpoints:
            for nxt in _lattice_neighbors(cell, rows, cols):
                if nxt in occupied:
                    continue
                touch_count = sum(nb in occupied for nb in _lattice_neighbors(nxt, rows, cols))
                if touch_count == 1:
                    choices.append((side, nxt))
        if not choices:
            return None
        side, nxt = random.choice(choices)
        if side == 0:
            path.insert(0, nxt)
        else:
            path.append(nxt)
        occupied.add(nxt)

    return occupied


def _expand_cluster(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    while len(occupied) < target_size:
        frontier = {
            nxt
            for cell in occupied
            for nxt in _lattice_neighbors(cell, rows, cols)
            if nxt not in occupied
        }
        if not frontier:
            return None
        occupied.add(random.choice(list(frontier)))
    return occupied


def _grow_cluster(rows: int, cols: int, size: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    return _expand_cluster(rows, cols, {start}, size)


def _grow_cycle(rows: int, cols: int, size: int) -> set[tuple[int, int]] | None:
    if rows < 2 or cols < 2 or size < 4:
        return None
    r0 = random.randrange(rows - 1)
    c0 = random.randrange(cols - 1)
    occupied = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
    return _expand_cluster(rows, cols, occupied, size)


def _sample_active_component(rows: int, cols: int) -> set[tuple[int, int]] | None:
    modes = ["path", "cluster"]
    if rows >= 2 and cols >= 2:
        modes.append("cycle")
    mode = random.choice(modes)

    max_size = min(len(GEAR_COLORS), rows * cols)
    if mode == "path":
        size = random.randint(2, max_size)
        return _grow_path(rows, cols, size)
    if mode == "cycle":
        size = random.randint(4, max_size)
        return _grow_cycle(rows, cols, size)

    size = random.randint(2, max_size)
    return _grow_cluster(rows, cols, size)


def _add_isolated_gears(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    extras: set[tuple[int, int]] = set()
    max_extra = len(GEAR_COLORS) - len(occupied)
    extra_goal = random.randint(0, max(0, max_extra))

    for _ in range(extra_goal):
        blocked = occupied | extras
        candidates = []
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in blocked:
                    continue
                if any(nb in blocked for nb in _lattice_neighbors(cell, rows, cols)):
                    continue
                candidates.append(cell)
        if not candidates:
            break
        extras.add(random.choice(candidates))

    return extras


def _lattice_to_grid(
    cell: tuple[int, int],
    pad_top: int,
    pad_left: int,
) -> tuple[int, int]:
    r, c = cell
    return (pad_top + 1 + 4 * r, pad_left + 1 + 4 * c)


def _render_from_layout(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    color_map: dict[tuple[int, int], int],
    active_cell: tuple[int, int],
    active_dir: int,
    turns: dict[tuple[int, int], int] | None,
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    pad_top, pad_bottom, pad_left, pad_right = pads
    h = 4 * rows - 1 + pad_top + pad_bottom
    w = 4 * cols - 1 + pad_left + pad_right
    grid = np.full((h, w), BLACK, dtype=int)

    for cell in occupied:
        gr, gc = _lattice_to_grid(cell, pad_top, pad_left)
        teeth = [GREY, GREY, GREY, GREY]
        if cell == active_cell:
            teeth[active_dir] = YELLOW
        if turns is not None:
            teeth = list(_rotate_teeth(tuple(teeth), turns.get(cell, 0)))

        grid[gr, gc] = color_map[cell]
        for (dr, dc), color in zip(ORTHO_OFFSETS, teeth):
            grid[gr + dr, gc + dc] = color

    return grid


def _expected_output(
    occupied: set[tuple[int, int]],
    color_map: dict[tuple[int, int], int],
    active_cell: tuple[int, int],
    active_dir: int,
    rows: int,
    cols: int,
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    graph = {cell: [] for cell in occupied}
    for cell in occupied:
        for nxt in _lattice_neighbors(cell, rows, cols):
            if nxt in occupied:
                graph[cell].append(nxt)
    turns = _rotation_map(graph, active_cell)
    return _render_from_layout(rows, cols, occupied, color_map, active_cell, active_dir, turns, pads)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*LATTICE_ROWS_RANGE)
    cols = random.randint(*LATTICE_COLS_RANGE)

    active_component = _sample_active_component(rows, cols)
    if active_component is None:
        return None

    occupied = set(active_component)
    occupied |= _add_isolated_gears(rows, cols, occupied)
    if not (2 <= len(occupied) <= len(GEAR_COLORS)):
        return None

    cells = list(occupied)
    random.shuffle(cells)
    colors = random.sample(GEAR_COLORS, len(cells))
    color_map = {cell: color for cell, color in zip(cells, colors)}

    active_cell = random.choice(list(active_component))
    active_dir = random.randrange(4)
    pads = tuple(random.randint(*PADDING_RANGE) for _ in range(4))

    inp = _render_from_layout(rows, cols, occupied, color_map, active_cell, active_dir, None, pads)
    expected = _expected_output(occupied, color_map, active_cell, active_dir, rows, cols, pads)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-rotation input")
