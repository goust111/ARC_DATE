from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
COLOR_VALUES = tuple(range(10))

GRID_MIN = 14
GRID_MAX = 28
TURRET_COUNT_RANGE = (2, 6)
GEN_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != bg)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _steps_to_edge(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    row_steps = h - 1 - r if dr > 0 else r
    col_steps = w - 1 - c if dc > 0 else c
    return int(min(row_steps, col_steps))


def _line_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _quadrant_room(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = pos
    dr, dc = direction
    row_room = h - 1 - r if dr > 0 else r
    col_room = w - 1 - c if dc > 0 else c
    return int(row_room), int(col_room)


def _can_place_cross_satellites(
    shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]
) -> bool:
    row_room, col_room = _quadrant_room(shape, pos, direction)
    return row_room >= 2 and col_room >= 2


def _identify_turrets(grid: np.ndarray) -> tuple[int, dict[tuple[int, int], dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    turrets: dict[tuple[int, int], dict[str, object]] = {}

    for color in _colors_present(g, bg):
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        valid: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for pr, pc in coords:
            others = [(r, c) for r, c in coords if (r, c) != (pr, pc)]
            if len(others) < 2:
                continue
            drs = [r - pr for r, _ in others]
            dcs = [c - pc for _, c in others]
            if any(dr == 0 for dr in drs) or any(dc == 0 for dc in dcs):
                continue
            row_signs = {1 if dr > 0 else -1 for dr in drs}
            col_signs = {1 if dc > 0 else -1 for dc in dcs}
            if len(row_signs) == 1 and len(col_signs) == 1:
                valid.append(((pr, pc), (next(iter(row_signs)), next(iter(col_signs)))))
        assert len(valid) == 1, (color, coords, valid)
        primary, direction = valid[0]
        turrets[primary] = {"color": color, "direction": direction}

    assert turrets, "expected at least one turret"
    return bg, turrets


def _build_successors(
    shape: tuple[int, int], turrets: dict[tuple[int, int], dict[str, object]]
) -> dict[tuple[int, int], tuple[int, int] | None]:
    turret_positions = set(turrets)
    successors: dict[tuple[int, int], tuple[int, int] | None] = {}
    for pos, info in turrets.items():
        dr, dc = info["direction"]
        r, c = pos
        target = None
        while True:
            r += dr
            c += dc
            if not _in_bounds(shape, r, c):
                break
            if (r, c) in turret_positions:
                target = (r, c)
                break
        successors[pos] = target
    return successors


def _ordered_chain(
    shape: tuple[int, int], turrets: dict[tuple[int, int], dict[str, object]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    successors = _build_successors(shape, turrets)
    indegree = {pos: 0 for pos in turrets}
    for dst in successors.values():
        if dst is not None:
            indegree[dst] += 1

    starts = [pos for pos, deg in indegree.items() if deg == 0]
    assert len(starts) == 1, indegree
    assert all(deg == 1 for pos, deg in indegree.items() if pos != starts[0]), indegree

    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    cur = starts[0]
    while cur is not None:
        assert cur not in seen, "turret relay must not cycle"
        seen.add(cur)
        order.append(cur)
        cur = successors[cur]

    assert len(order) == len(turrets), (order, turrets)
    directions = [tuple(turrets[pos]["direction"]) for pos in order]
    colors = [int(turrets[pos]["color"]) for pos in order]
    return order, directions, colors


def _draw_path(
    shape: tuple[int, int],
    bg: int,
    order: list[tuple[int, int]],
    directions: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    turret_index = {pos: idx for idx, pos in enumerate(order)}
    cur_idx = 0
    pos = order[0]

    while True:
        out[pos] = colors[cur_idx]
        dr, dc = directions[cur_idx]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not _in_bounds(shape, nr, nc):
            break
        pos = (nr, nc)
        if pos in turret_index:
            cur_idx = turret_index[pos]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the relay of diagonal turrets and keep only the multicolor broken-line path.

    Input structure:
    - The background is the most common color.
    - Each foreground color has exactly one primary pixel; all other pixels of
      that color lie strictly inside one diagonal quadrant of the primary.
    - From each primary, the quadrant indicated by its satellites points along a
      diagonal that either hits another primary first or exits the grid.

    Rule:
    - Find the unique primary that is not hit by any other primary's diagonal.
    - Starting there, move cell by cell along its diagonal direction.
    - Whenever the path lands on another primary, switch immediately to that
      primary's color and diagonal direction, then continue until the path exits.

    Key insight:
    - The same-colored satellite cluster is not output content; it is a local
      direction cue that turns each primary into the next segment source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, turrets = _identify_turrets(g)
    order, directions, colors = _ordered_chain(g.shape, turrets)
    return _draw_path(g.shape, bg, order, directions, colors)


def _candidate_moves(
    shape: tuple[int, int],
    pos: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    options: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    dirs = DIAGONALS.copy()
    random.shuffle(dirs)
    for direction in dirs:
        max_steps = _steps_to_edge(shape, pos, direction)
        if max_steps < 2:
            continue
        distances = list(range(2, max_steps + 1))
        random.shuffle(distances)
        for dist in distances:
            segment = _line_cells(pos, direction, dist)
            if any(cell in occupied for cell in segment):
                continue
            nxt = segment[-1]
            if not any(_can_place_cross_satellites(shape, nxt, d) for d in DIAGONALS):
                continue
            options.append((direction, nxt, segment))
    random.shuffle(options)
    return options


def _choose_final_direction(
    shape: tuple[int, int],
    pos: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    options: list[tuple[int, int]] = []
    dirs = DIAGONALS.copy()
    random.shuffle(dirs)
    for direction in dirs:
        if not _can_place_cross_satellites(shape, pos, direction):
            continue
        final_length = _steps_to_edge(shape, pos, direction)
        if final_length < 2:
            continue
        segment = _line_cells(pos, direction, final_length)
        if any(cell in occupied for cell in segment):
            continue
        options.append(direction)
    return random.choice(options) if options else None


def _build_path(shape: tuple[int, int], turret_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape

    def dfs(
        vertices: list[tuple[int, int]],
        directions: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
        if len(vertices) == turret_count:
            final_dir = _choose_final_direction(shape, vertices[-1], occupied)
            if final_dir is None:
                return None
            return vertices.copy(), directions + [final_dir]

        for direction, nxt, segment in _candidate_moves(shape, vertices[-1], occupied):
            for cell in segment:
                occupied.add(cell)
            vertices.append(nxt)
            directions.append(direction)
            result = dfs(vertices, directions, occupied)
            if result is not None:
                return result
            directions.pop()
            vertices.pop()
            for cell in segment:
                occupied.remove(cell)
        return None

    for _ in range(PATH_BUILD_ATTEMPTS):
        start = (random.randint(2, h - 3), random.randint(2, w - 3))
        occupied = {start}
        result = dfs([start], [], occupied)
        if result is not None:
            return result
    return None


def _path_cells(
    shape: tuple[int, int], order: list[tuple[int, int]], directions: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    cells = {order[0]}
    for idx in range(len(order) - 1):
        cur = order[idx]
        nxt = order[idx + 1]
        length = abs(nxt[0] - cur[0])
        cells.update(_line_cells(cur, directions[idx], length))
    final_len = _steps_to_edge(shape, order[-1], directions[-1])
    cells.update(_line_cells(order[-1], directions[-1], final_len))
    return cells


def _choose_cross_satellite_pair(
    shape: tuple[int, int],
    primary: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    row_room, col_room = _quadrant_room(shape, primary, direction)
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    pr, pc = primary
    dr, dc = direction
    for roff in range(1, row_room + 1):
        for coff in range(1, col_room + 1):
            cell = (pr + dr * roff, pc + dc * coff)
            if cell in blocked:
                continue
            candidates.append((roff, coff, cell))

    pairs: list[list[tuple[int, int]]] = []
    for i, (r1, c1, cell1) in enumerate(candidates):
        for r2, c2, cell2 in candidates[i + 1 :]:
            if (r1 < r2 and c1 > c2) or (r1 > r2 and c1 < c2):
                pairs.append([cell1, cell2])

    if not pairs:
        return None
    return random.choice(pairs)


def _render_input(
    shape: tuple[int, int],
    bg: int,
    order: list[tuple[int, int]],
    directions: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray | None:
    grid = np.full(shape, bg, dtype=int)
    blocked = _path_cells(shape, order, directions)

    for primary, direction, color in zip(order, directions, colors):
        pair = _choose_cross_satellite_pair(shape, primary, direction, blocked)
        if pair is None:
            return None
        r, c = primary
        grid[r, c] = color
        for sr, sc in pair:
            grid[sr, sc] = color
            blocked.add((sr, sc))

    return grid


def _gen_attempt() -> np.ndarray | None:
    turret_count = random.randint(*TURRET_COUNT_RANGE)
    min_side = max(GRID_MIN, 2 * turret_count + 8)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    shape = (h, w)

    built = _build_path(shape, turret_count)
    if built is None:
        return None
    order, directions = built

    palette = random.sample(COLOR_VALUES, turret_count + 1)
    bg = palette[0]
    colors = palette[1:]

    grid = _render_input(shape, bg, order, directions, colors)
    if grid is None:
        return None

    expected = _draw_path(shape, bg, order, directions, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recursive-turret input")
