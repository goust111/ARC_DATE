from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7
BG_CHOICES = (0, 6, 8, 9)
SWITCH_COLORS = (RED, BLUE, GREEN)
ORTH_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

MIN_ARM = 2
MAX_ARM = 5
MIN_GAP = 4
MAX_GAP = 8
MARGIN_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400

Coord = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rot_cw(direction: Coord) -> Coord:
    dr, dc = direction
    return dc, -dr


def _rot_ccw(direction: Coord) -> Coord:
    dr, dc = direction
    return -dc, dr


def _switch_direction(color: int, move: Coord) -> Coord:
    if color == RED:
        return _rot_ccw(move)
    if color == BLUE:
        return _rot_cw(move)
    if color == GREEN:
        return move
    raise AssertionError(f"unexpected switch color {color}")


def _network_neighbors(mask: np.ndarray) -> dict[Coord, list[Coord]]:
    neighbors: dict[Coord, list[Coord]] = {}
    for r, c in np.argwhere(mask):
        pos = (int(r), int(c))
        adj: list[Coord] = []
        for dr, dc in ORTH_DIRS:
            rr = int(r) + dr
            cc = int(c) + dc
            if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
                adj.append((rr, cc))
        neighbors[pos] = adj
    return neighbors


def _analyze_network(grid: np.ndarray) -> tuple[int, dict[Coord, list[Coord]], set[Coord], set[Coord], set[Coord]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    allowed = {bg, YELLOW, GRAY, RED, BLUE, GREEN}
    colors = set(map(int, np.unique(g)))
    assert colors <= allowed

    mask = g != bg
    neighbors = _network_neighbors(mask)
    bulbs: set[Coord] = set()
    switches: set[Coord] = set()
    gray_cells: set[Coord] = set()

    for pos, adj in neighbors.items():
        color = int(g[pos])
        degree = len(adj)
        if color == YELLOW:
            assert degree == 1
            bulbs.add(pos)
        elif color == GRAY:
            assert degree == 2
            gray_cells.add(pos)
        elif color in SWITCH_COLORS:
            assert degree in {3, 4}
            switches.add(pos)
        else:
            raise AssertionError(f"unexpected non-background color {color}")

    assert len(bulbs) in {2, 4}
    assert switches
    assert gray_cells
    return bg, neighbors, bulbs, switches, gray_cells


def _trace_from_bulb(
    grid: np.ndarray,
    start: Coord,
    neighbors: dict[Coord, list[Coord]],
    bulbs: set[Coord],
) -> tuple[Coord, set[Coord]] | None:
    prev = start
    adj = neighbors[start]
    if len(adj) != 1:
        return None

    cur = adj[0]
    move = (cur[0] - prev[0], cur[1] - prev[1])
    seen_states: set[tuple[Coord, Coord]] = set()
    gray_path: set[Coord] = set()

    while True:
        state = (cur, move)
        if state in seen_states:
            return None
        seen_states.add(state)

        if cur in bulbs:
            return (cur, gray_path) if cur != start else None

        color = int(grid[cur])
        if color == GRAY:
            gray_path.add(cur)
            options = [nxt for nxt in neighbors[cur] if nxt != prev]
            if len(options) != 1:
                return None
            nxt = options[0]
        elif color in SWITCH_COLORS:
            out_dir = _switch_direction(color, move)
            nxt = (cur[0] + out_dir[0], cur[1] + out_dir[1])
            if nxt not in neighbors[cur]:
                return None
        else:
            return None

        prev, cur = cur, nxt
        move = (cur[0] - prev[0], cur[1] - prev[1])


def _successful_gray_cells(
    grid: np.ndarray,
    neighbors: dict[Coord, list[Coord]],
    bulbs: set[Coord],
) -> set[Coord]:
    successful: set[Coord] = set()
    for bulb in bulbs:
        traced = _trace_from_bulb(grid, bulb, neighbors, bulbs)
        if traced is None:
            continue
        _end_bulb, gray_path = traced
        successful.update(gray_path)
    return successful


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow bulb signals through a gray path network whose colored switches redirect
    the signal, keep only gray path pixels that belong to successful bulb-to-bulb
    routes, and erase all other gray path pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, neighbors, bulbs, switches, _gray_cells = _analyze_network(g)
    successful = _successful_gray_cells(g, neighbors, bulbs)

    out = np.full_like(g, bg)
    out[g == YELLOW] = YELLOW
    for color in SWITCH_COLORS:
        out[g == color] = color
    for r, c in successful:
        out[r, c] = ORANGE
    return out


def _segment_points(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _draw_polyline(cells: dict[Coord, int], points: list[Coord], reserved: set[Coord]) -> None:
    new_gray: set[Coord] = set()
    for i in range(len(points) - 1):
        for pos in _segment_points(points[i], points[i + 1]):
            if pos in reserved:
                continue
            if pos in new_gray:
                continue
            if pos in cells:
                raise ValueError(f"unexpected overlap at {pos}")
            new_gray.add(pos)
    for pos in new_gray:
        cells[pos] = GRAY


def _normalize_cells(cells: dict[Coord, int]) -> dict[Coord, int]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c): color for (r, c), color in cells.items()}


def _rotate_cells(cells: dict[Coord, int], quarter_turns: int) -> dict[Coord, int]:
    rotated: dict[Coord, int] = {}
    for (r, c), color in cells.items():
        rr, cc = r, c
        for _ in range(quarter_turns % 4):
            rr, cc = -cc, rr
        rotated[(rr, cc)] = color
    return _normalize_cells(rotated)


def _render_component(cells: dict[Coord, int], bg: int) -> np.ndarray:
    cells = _rotate_cells(cells, random.randrange(4))
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    grid = np.full((top + height + bottom, left + width + right), bg, dtype=int)
    for (r, c), color in cells.items():
        grid[top + r, left + c] = color
    return grid


def _build_cross4() -> dict[Coord, int]:
    up = random.randint(MIN_ARM, MAX_ARM)
    down = random.randint(MIN_ARM, MAX_ARM)
    left = random.randint(MIN_ARM, MAX_ARM)
    right = random.randint(MIN_ARM, MAX_ARM)
    switch = (0, 0)
    bulbs = {
        (-up, 0),
        (down, 0),
        (0, -left),
        (0, right),
    }
    reserved = {switch} | bulbs
    cells: dict[Coord, int] = {}
    for bulb in bulbs:
        _draw_polyline(cells, [switch, bulb], reserved)
    cells[switch] = random.choice(SWITCH_COLORS)
    for bulb in bulbs:
        cells[bulb] = YELLOW
    return cells


def _build_tree4() -> dict[Coord, int]:
    gap = random.randint(MIN_GAP, MAX_GAP)
    s1 = (0, 0)
    s2 = (0, gap)
    bulbs = {
        (-random.randint(MIN_ARM, MAX_ARM), 0),
        (random.randint(MIN_ARM, MAX_ARM), 0),
        (-random.randint(MIN_ARM, MAX_ARM), gap),
        (random.randint(MIN_ARM, MAX_ARM), gap),
    }
    reserved = {s1, s2} | bulbs
    cells: dict[Coord, int] = {}
    for bulb in bulbs:
        anchor = s1 if bulb[1] == 0 else s2
        _draw_polyline(cells, [anchor, bulb], reserved)
    _draw_polyline(cells, [s1, s2], reserved)
    cells[s1] = random.choice(SWITCH_COLORS)
    cells[s2] = random.choice(SWITCH_COLORS)
    for bulb in bulbs:
        cells[bulb] = YELLOW
    return cells


def _build_cycle2() -> dict[Coord, int]:
    gap = random.randint(MIN_GAP, MAX_GAP)
    top = random.randint(MIN_ARM, MAX_ARM)
    bottom = random.randint(MIN_ARM, MAX_ARM)
    left = random.randint(MIN_ARM, MAX_ARM)
    right = random.randint(MIN_ARM, MAX_ARM)

    s1 = (0, 0)
    s2 = (0, gap)
    left_bulb = (0, -left)
    right_bulb = (0, gap + right)
    switch_color = random.choice((RED, BLUE))

    reserved = {s1, s2, left_bulb, right_bulb}
    cells: dict[Coord, int] = {}
    _draw_polyline(cells, [left_bulb, s1], reserved)
    _draw_polyline(cells, [s2, right_bulb], reserved)
    _draw_polyline(cells, [s1, (-top, 0), (-top, gap), s2], reserved)
    _draw_polyline(cells, [s1, (bottom, 0), (bottom, gap), s2], reserved)
    cells[s1] = switch_color
    cells[s2] = switch_color
    cells[left_bulb] = YELLOW
    cells[right_bulb] = YELLOW
    return cells


def _count_successful_routes(grid: np.ndarray) -> int:
    bg, neighbors, bulbs, _switches, _gray = _analyze_network(grid)
    count = 0
    for bulb in bulbs:
        if _trace_from_bulb(grid, bulb, neighbors, bulbs) is not None:
            count += 1
    return count


def _generated_grid_is_useful(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        bg = _most_common_color(grid)
        _count_successful_routes(grid)
    except AssertionError:
        return False

    orange_count = int(np.count_nonzero(out == ORANGE))
    gray_count = int(np.count_nonzero(grid == GRAY))
    if orange_count == 0 or gray_count < 4:
        return False

    # Require enough structure change in a substantial fraction of samples.
    erased_gray = bool(np.any((grid == GRAY) & (out == bg)))
    if np.count_nonzero(grid == YELLOW) == 4 and random.random() < 0.5 and not erased_gray:
        return False
    return True


def _build_component() -> dict[Coord, int]:
    bulb_count = random.choice((2, 4))
    if bulb_count == 2:
        return _build_cycle2()
    family = random.choice(("tree4", "cross4"))
    if family == "tree4":
        return _build_tree4()
    return _build_cross4()


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    try:
        component = _build_component()
        grid = _render_component(component, bg)
    except ValueError:
        return None
    return grid if _generated_grid_is_useful(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid switch-network ARC input")
