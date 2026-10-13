from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9
SOURCE_COLORS = (1, 2, 3, 6, 7, 8)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _edge_allowed(g: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    ra, ca = a
    rb, cb = b
    if abs(ra - rb) + abs(ca - cb) != 1:
        return False
    if ca == cb and (int(g[ra, ca]) == MAROON or int(g[rb, cb]) == MAROON):
        return False
    return True


def _component_labels(g: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == bg or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(g[nr, nc]) == bg or labels[nr, nc] != -1:
                        continue
                    if not _edge_allowed(g, (cr, cc), (nr, nc)):
                        continue
                    labels[nr, nc] = comp_id
                    q.append((nr, nc))
            comp_id += 1
    return labels, comp_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through an orthogonal pipe network with directional valves.

    Input structure
    - Background is the dominant color.
    - Black pixels form a one-cell-wide horizontal/vertical pipe network.
    - Grey singleton endpoints are terminals.
    - Non-grey, non-black, non-background singleton endpoints are sources.
    - Maroon valve pixels block vertical connectivity at that crossing while still
      allowing horizontal passage.

    Rule
    - Each source color spreads through all reachable pipe cells and terminals.
    - Vertical movement into or out of a maroon valve is forbidden.
    - A terminal becomes the unique reaching color, yellow if multiple colors
      reach it, or stays grey if none do.
    - Pipes, valves, and sources stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    labels, n_comp = _component_labels(g, bg)

    comp_sources: list[set[int]] = [set() for _ in range(n_comp)]
    terminals: list[tuple[int, int, int]] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == bg:
                continue
            comp = int(labels[r, c])
            assert comp >= 0
            if color == GREY:
                terminals.append((r, c, comp))
            elif color not in (BLACK, MAROON):
                comp_sources[comp].add(color)

    out = g.copy()
    for r, c, comp in terminals:
        colors = comp_sources[comp]
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        elif len(colors) >= 2:
            out[r, c] = YELLOW
    return out


Coord = tuple[int, int]

GEN_HEIGHT = (16, 24)
GEN_WIDTH = (20, 30)
GEN_BUS_SOURCES = (2, 4)
GEN_BUS_TERMINALS = (1, 3)
GEN_BRANCH_DIST = (2, 5)
GEN_ARM_LEN = (2, 3)
GEN_RETRIES = 400


def _claim(used: dict[Coord, str], coord: Coord, role: str, h: int, w: int) -> None:
    r, c = coord
    assert 0 <= r < h and 0 <= c < w, (coord, h, w)
    previous = used.get(coord)
    assert previous is None, (coord, previous, role)
    used[coord] = role


def _add_pipe(
    pipes: set[Coord],
    used: dict[Coord, str],
    coord: Coord,
    h: int,
    w: int,
) -> None:
    _claim(used, coord, "pipe", h, w)
    pipes.add(coord)


def _add_valve(
    valves: set[Coord],
    used: dict[Coord, str],
    coord: Coord,
    h: int,
    w: int,
) -> None:
    _claim(used, coord, "valve", h, w)
    valves.add(coord)


def _add_source(
    sources: dict[Coord, int],
    used: dict[Coord, str],
    coord: Coord,
    color: int,
    h: int,
    w: int,
) -> None:
    _claim(used, coord, "source", h, w)
    sources[coord] = int(color)


def _add_terminal(
    terminals: dict[Coord, int],
    used: dict[Coord, str],
    coord: Coord,
    expected: int,
    h: int,
    w: int,
) -> None:
    _claim(used, coord, "terminal", h, w)
    terminals[coord] = int(expected)


def _random_dist(low: int, high: int, cap: int) -> int | None:
    lo = max(low, 1)
    hi = min(high, cap)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _arm_options(col: int, width: int, count: int) -> dict[int, int] | None:
    directions = random.sample([-1, 1], count)
    lengths: dict[int, int] = {}
    for direction in directions:
        cap = col - 1 if direction < 0 else width - 2 - col
        length = _random_dist(GEN_ARM_LEN[0], GEN_ARM_LEN[1], cap)
        if length is None:
            return None
        lengths[direction] = length
    return lengths


def _add_vertical_leaf(
    pipes: set[Coord],
    sources: dict[Coord, int],
    terminals: dict[Coord, int],
    used: dict[Coord, str],
    root_row: int,
    col: int,
    direction: int,
    dist: int,
    color: int | None,
    expected_terminal: int | None,
    h: int,
    w: int,
) -> None:
    assert direction in (-1, 1)
    for step in range(1, dist):
        _add_pipe(pipes, used, (root_row + direction * step, col), h, w)
    endpoint = (root_row + direction * dist, col)
    if color is not None:
        _add_source(sources, used, endpoint, color, h, w)
    else:
        assert expected_terminal is not None
        _add_terminal(terminals, used, endpoint, expected_terminal, h, w)


def _add_side_terminals(
    pipes: set[Coord],
    terminals: dict[Coord, int],
    used: dict[Coord, str],
    row: int,
    col: int,
    arm_lengths: dict[int, int],
    expected: int,
    h: int,
    w: int,
) -> None:
    for direction, length in arm_lengths.items():
        for step in range(1, length):
            _add_pipe(pipes, used, (row, col + direction * step), h, w)
        _add_terminal(terminals, used, (row, col + direction * length), expected, h, w)


def _render_grid(
    h: int,
    w: int,
    bg: int,
    pipes: set[Coord],
    valves: set[Coord],
    sources: dict[Coord, int],
    terminals: dict[Coord, int],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for r, c in pipes:
        grid[r, c] = BLACK
    for r, c in valves:
        grid[r, c] = MAROON
    for (r, c), color in sources.items():
        grid[r, c] = int(color)
    for r, c in terminals:
        grid[r, c] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT)
    w = random.randint(*GEN_WIDTH)
    bg = random.choice(SOURCE_COLORS)
    palette = [color for color in SOURCE_COLORS if color != bg]

    n_bus_sources = random.randint(*GEN_BUS_SOURCES)
    n_bus_terms = random.randint(*GEN_BUS_TERMINALS)
    if n_bus_sources + 1 > len(palette):
        return None

    bus_leaf_count = n_bus_sources + n_bus_terms
    bus_branch_count = bus_leaf_count - 2

    bus_row = random.randint(6, h - 7)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    c_left = left_margin
    c_right = w - 1 - right_margin
    if c_right - c_left + 1 < 12:
        return None

    valve_buffer = 3
    valve_min = max(c_left + valve_buffer + 1, 4)
    valve_max = min(c_right - valve_buffer - 1, w - 5)
    if valve_min > valve_max:
        return None
    valve_col = random.randint(valve_min, valve_max)

    available_cols = [
        c
        for c in range(c_left + 1, c_right)
        if abs(c - valve_col) > valve_buffer
    ]
    if len(available_cols) < bus_branch_count:
        return None
    branch_cols = random.sample(available_cols, bus_branch_count)

    single_side = random.choice([-1, 1])
    blocked_side = -single_side
    single_space = bus_row - 1 if single_side < 0 else h - bus_row - 2
    blocked_space = bus_row - 1 if blocked_side < 0 else h - bus_row - 2
    if single_space < 3 or blocked_space < 2:
        return None

    source_colors = random.sample(palette, n_bus_sources + 1)
    bus_source_colors = source_colors[:-1]
    isolated_source_color = source_colors[-1]

    pipes: set[Coord] = set()
    valves: set[Coord] = set()
    sources: dict[Coord, int] = {}
    terminals: dict[Coord, int] = {}
    used: dict[Coord, str] = {}

    for c in range(c_left + 1, c_right):
        coord = (bus_row, c)
        if c == valve_col:
            _add_valve(valves, used, coord, h, w)
        else:
            _add_pipe(pipes, used, coord, h, w)

    bus_positions: list[tuple[str, int]] = [("end", c_left), ("end", c_right)]
    bus_positions.extend(("branch", c) for c in branch_cols)
    random.shuffle(bus_positions)

    source_indices = set(random.sample(range(len(bus_positions)), n_bus_sources))
    bus_source_by_index = {
        idx: color
        for idx, color in zip(sorted(source_indices), bus_source_colors)
    }

    for idx, (kind, col) in enumerate(bus_positions):
        source_color = bus_source_by_index.get(idx)
        if kind == "end":
            if source_color is not None:
                _add_source(sources, used, (bus_row, col), source_color, h, w)
            else:
                _add_terminal(terminals, used, (bus_row, col), YELLOW, h, w)
            continue

        direction = random.choice([-1, 1])
        space = bus_row if direction < 0 else h - bus_row - 1
        dist = _random_dist(GEN_BRANCH_DIST[0], GEN_BRANCH_DIST[1], space)
        if dist is None:
            return None
        _add_vertical_leaf(
            pipes,
            sources,
            terminals,
            used,
            bus_row,
            col,
            direction,
            dist,
            source_color,
            None if source_color is not None else YELLOW,
            h,
            w,
        )

    single_src_dist = _random_dist(2, 5, single_space)
    if single_src_dist is None:
        return None
    junction_dist = random.randint(1, single_src_dist - 1)
    single_arm_count = random.choice([1, 2])
    single_arms = _arm_options(valve_col, w, single_arm_count)
    if single_arms is None:
        return None
    for step in range(1, single_src_dist):
        _add_pipe(pipes, used, (bus_row + single_side * step, valve_col), h, w)
    _add_source(
        sources,
        used,
        (bus_row + single_side * single_src_dist, valve_col),
        isolated_source_color,
        h,
        w,
    )
    single_junction_row = bus_row + single_side * junction_dist
    _add_side_terminals(
        pipes,
        terminals,
        used,
        single_junction_row,
        valve_col,
        single_arms,
        isolated_source_color,
        h,
        w,
    )

    blocked_style = random.choice(["line", "tee"])
    if blocked_style == "line":
        blocked_dist = _random_dist(2, 5, blocked_space)
        if blocked_dist is None:
            return None
        _add_vertical_leaf(
            pipes,
            sources,
            terminals,
            used,
            bus_row,
            valve_col,
            blocked_side,
            blocked_dist,
            None,
            GREY,
            h,
            w,
        )
    else:
        blocked_junction_dist = _random_dist(1, 4, blocked_space)
        if blocked_junction_dist is None:
            return None
        blocked_arm_count = random.choice([1, 2])
        blocked_arms = _arm_options(valve_col, w, blocked_arm_count)
        if blocked_arms is None:
            return None
        for step in range(1, blocked_junction_dist + 1):
            _add_pipe(pipes, used, (bus_row + blocked_side * step, valve_col), h, w)
        blocked_junction_row = bus_row + blocked_side * blocked_junction_dist
        _add_side_terminals(
            pipes,
            terminals,
            used,
            blocked_junction_row,
            valve_col,
            blocked_arms,
            GREY,
            h,
            w,
        )

    grid = _render_grid(h, w, bg, pipes, valves, sources, terminals)
    expected = grid.copy()
    categories = {"mixed": 0, "single": 0, "blocked": 0}
    for (r, c), color in terminals.items():
        expected[r, c] = color
        if color == YELLOW:
            categories["mixed"] += 1
        elif color == GREY:
            categories["blocked"] += 1
        else:
            categories["single"] += 1

    if min(categories.values()) == 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid valve-pipe input")
