import random
from typing import Dict, Iterable, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9
SOURCE_PALETTE = (3, 4, 6, 7, 8)

DIR_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]
EDGE_TO_INCOMING = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
GEN_MIN_SIZE = 15
GEN_MAX_SIZE = 26
GEN_MIN_SOURCES = 2
GEN_MAX_SOURCES = 4
GEN_MIN_EDGE_RUN = 3
GEN_MIN_STUB = 1
GEN_MAX_STUB = 4
GEN_MAX_ATTEMPTS = 300
GEN_GADGET_ATTEMPTS = 200


def _add(a: Tuple[int, int], b: Tuple[int, int], steps: int = 1) -> Tuple[int, int]:
    return a[0] + b[0] * steps, a[1] + b[1] * steps


def _opposite(direction: Tuple[int, int]) -> Tuple[int, int]:
    return -direction[0], -direction[1]


def _turn(direction: Tuple[int, int], node_color: int) -> Tuple[int, int]:
    idx = DIR_ORDER.index(direction)
    if node_color == RED:
        return DIR_ORDER[(idx - 1) % 4]
    assert node_color == BLUE
    return DIR_ORDER[(idx + 1) % 4]


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _track_neighbors(grid: np.ndarray, r: int, c: int) -> list[Tuple[int, int]]:
    neighbors: list[Tuple[int, int]] = []
    for dr, dc in DIR_ORDER:
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BLACK:
            neighbors.append((rr, cc))
    return neighbors


def _source_cells(grid: np.ndarray) -> list[Tuple[int, int]]:
    reserved = {BLACK, GREY, RED, BLUE, MAROON}
    sources = [
        (int(r), int(c))
        for r, c in np.argwhere(~np.isin(grid, list(reserved)))
    ]
    assert sources, "expected at least one unique source color"
    return sources


def _trace_signal(grid: np.ndarray, source: Tuple[int, int]) -> Tuple[int, int]:
    neighbors = _track_neighbors(grid, *source)
    assert len(neighbors) == 1, "source must occupy a rail endpoint"
    direction = (neighbors[0][0] - source[0], neighbors[0][1] - source[1])
    r, c = source
    seen = {(r, c, direction[0], direction[1])}

    while True:
        r += direction[0]
        c += direction[1]
        assert _in_bounds(grid.shape, r, c), "signal left the grid"
        color = int(grid[r, c])
        assert color != BLACK, "signal left the rail network"

        if color == MAROON:
            return r, c
        if color in (RED, BLUE):
            direction = _turn(direction, color)
        else:
            assert color == GREY, "signals may only traverse grey rail cells and turn nodes"

        state = (r, c, direction[0], direction[1])
        assert state not in seen, "signal path formed a loop"
        seen.add(state)


def main(grid: np.ndarray) -> np.ndarray:
    """Route each source color through the rail network and write it into the maroon terminal it reaches.

    Input structure:
    - The background is black.
    - Grey single-pixel rails run horizontally or vertically.
    - Red and blue pixels sit on rail crossings and act as turn nodes.
    - Unique source colors occupy some rail endpoints, and maroon pixels mark empty terminal slots on the grid edge.

    Rule:
    - Each source begins moving inward along its only attached rail direction.
    - At a red node the signal turns left relative to its current heading; at a blue node it turns right.
    - The signal keeps following the rail until it reaches a maroon terminal slot.
    - Replace each reached maroon slot with the color of the source that arrived there, while leaving every other pixel unchanged.

    Key insight:
    - Node colors encode relative turns rather than absolute directions, so the same red or blue node sends different sources to different edges depending on how they enter the crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    filled: Dict[Tuple[int, int], int] = {}
    for source in _source_cells(g):
        source_color = int(g[source])
        terminal = _trace_signal(g, source)
        assert terminal not in filled, "two sources reached the same terminal"
        filled[terminal] = source_color

    for terminal, color in filled.items():
        out[terminal] = color
    return out


def _distance_to_edge(shape: Tuple[int, int], cell: Tuple[int, int], direction: Tuple[int, int]) -> int:
    r, c = cell
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return shape[0] - 1 - r
    if dc == -1:
        return c
    return shape[1] - 1 - c


def _candidate_nodes(
    shape: Tuple[int, int],
    incoming: Tuple[int, int],
    node_color: int,
) -> list[Tuple[int, int]]:
    chosen_turn = _turn(incoming, node_color)
    straight = incoming
    back = _opposite(incoming)
    unused_turn = _opposite(chosen_turn)
    candidates: list[Tuple[int, int]] = []

    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            cell = (r, c)
            if _distance_to_edge(shape, cell, back) < GEN_MIN_EDGE_RUN:
                continue
            if _distance_to_edge(shape, cell, chosen_turn) < GEN_MIN_EDGE_RUN:
                continue
            if _distance_to_edge(shape, cell, straight) < GEN_MIN_STUB + 1:
                continue
            if _distance_to_edge(shape, cell, unused_turn) < GEN_MIN_STUB + 1:
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _paint_ray(
    cells: Dict[Tuple[int, int], int],
    origin: Tuple[int, int],
    direction: Tuple[int, int],
    length: int,
    tip_color: int,
) -> None:
    for step in range(1, length):
        cells[_add(origin, direction, step)] = GREY
    cells[_add(origin, direction, length)] = tip_color


def _make_gadget(
    shape: Tuple[int, int],
    source_color: int,
    source_edge: str,
    node_color: int,
    node: Tuple[int, int],
) -> Dict[Tuple[int, int], int]:
    incoming = EDGE_TO_INCOMING[source_edge]
    chosen_turn = _turn(incoming, node_color)
    back = _opposite(incoming)
    unused_turn = _opposite(chosen_turn)

    straight_max = min(GEN_MAX_STUB, _distance_to_edge(shape, node, incoming) - 1)
    unused_max = min(GEN_MAX_STUB, _distance_to_edge(shape, node, unused_turn) - 1)
    straight_len = random.randint(GEN_MIN_STUB, straight_max)
    unused_len = random.randint(GEN_MIN_STUB, unused_max)

    cells: Dict[Tuple[int, int], int] = {node: node_color}
    _paint_ray(cells, node, back, _distance_to_edge(shape, node, back), source_color)
    _paint_ray(cells, node, chosen_turn, _distance_to_edge(shape, node, chosen_turn), MAROON)
    for step in range(1, straight_len + 1):
        cells[_add(node, incoming, step)] = GREY
    for step in range(1, unused_len + 1):
        cells[_add(node, unused_turn, step)] = GREY
    return cells


def _can_place(cells: Iterable[Tuple[int, int]], occupied: set[Tuple[int, int]]) -> bool:
    new_cells = set(cells)
    for cell in new_cells:
        if cell in occupied:
            return False
    for r, c in new_cells:
        for dr, dc in DIR_ORDER:
            neighbor = (r + dr, c + dc)
            if neighbor in occupied and neighbor not in new_cells:
                return False
    return True


def _place_gadgets(
    grid: np.ndarray,
    source_colors: list[int],
    node_colors: list[int],
) -> bool:
    occupied: set[Tuple[int, int]] = set()
    source_edges = list(EDGE_TO_INCOMING)

    for source_color, node_color in zip(source_colors, node_colors):
        placed = False
        edge_order = random.sample(source_edges, len(source_edges))
        for source_edge in edge_order:
            incoming = EDGE_TO_INCOMING[source_edge]
            candidates = _candidate_nodes(grid.shape, incoming, node_color)
            for node in candidates[:GEN_GADGET_ATTEMPTS]:
                gadget = _make_gadget(grid.shape, source_color, source_edge, node_color, node)
                if not _can_place(gadget, occupied):
                    continue
                for cell, color in gadget.items():
                    grid[cell] = color
                occupied.update(gadget)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _node_color_plan(count: int) -> list[int]:
    colors = [RED, BLUE]
    while len(colors) < count:
        colors.append(random.choice((RED, BLUE)))
    random.shuffle(colors)
    return colors[:count]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    w = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    count = random.randint(GEN_MIN_SOURCES, GEN_MAX_SOURCES)
    grid = np.full((h, w), BLACK, dtype=int)

    source_colors = random.sample(list(SOURCE_PALETTE), count)
    node_colors = _node_color_plan(count)
    if not _place_gadgets(grid, source_colors, node_colors):
        return None

    if np.count_nonzero(np.isin(grid, list(source_colors))) != count:
        return None
    if np.count_nonzero(grid == MAROON) != count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out == MAROON) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-routing ARC input")
