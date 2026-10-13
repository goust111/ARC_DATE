import random
from typing import Iterable

import numpy as np


# Fixed ARC palette roles from the puzzle statement.
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
MAROON = 9
SOURCE_COLORS = (0, 3, 4, 6, 8)

# Cardinal directions in NESW order.
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RIGHT_TURN = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
LEFT_TURN = {direction: turn for turn, direction in RIGHT_TURN.items()}
ARM_TO_VECTOR = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
VECTOR_TO_ARM = {vector: arm for arm, vector in ARM_TO_VECTOR.items()}
OPPOSITE_ARM = {"N": "S", "E": "W", "S": "N", "W": "E"}

# Generator controls.
GRID_SIZE_RANGE = (14, 24)
GADGET_COUNT_CHOICES = (1, 2)
GADGET_COUNT_WEIGHTS = (3, 2)
ARM_LENGTH_RANGE = (2, 6)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(grid: np.ndarray, coord: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = coord
    result = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != ORANGE:
            result.append((nr, nc))
    return result


def _source_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(grid != ORANGE)
    sources: list[tuple[int, int]] = []
    for r, c in coords:
        color = int(grid[r, c])
        if color in {BLUE, RED, GREY, MAROON}:
            continue
        position = (int(r), int(c))
        degree = len(_neighbors(grid, position))
        assert degree == 1, "each source terminal must be a degree-1 endpoint"
        sources.append(position)
    assert sources, "expected at least one signal source"
    return sources


def _next_coord(
    grid: np.ndarray,
    prev_coord: tuple[int, int] | None,
    coord: tuple[int, int],
) -> tuple[int, int] | None:
    color = int(grid[coord])
    neighbors = _neighbors(grid, coord)

    if prev_coord is None:
        assert len(neighbors) == 1, "source terminal must touch exactly one wire cell"
        return neighbors[0]

    if color == MAROON:
        assert len(neighbors) == 1, "receiver terminal must be a degree-1 endpoint"
        return None

    candidates = [neighbor for neighbor in neighbors if neighbor != prev_coord]
    assert candidates, "signal path cannot terminate on a non-receiver wire cell"

    if color in {RED, BLUE}:
        dr = coord[0] - prev_coord[0]
        dc = coord[1] - prev_coord[1]
        incoming = (dr, dc)
        turn_map = RIGHT_TURN if color == RED else LEFT_TURN
        out_dir = turn_map[incoming]
        target = (coord[0] + out_dir[0], coord[1] + out_dir[1])
        assert target in candidates, "junction marker must define a valid turn"
        return target

    assert len(candidates) == 1, "non-junction wire cells must have a unique continuation"
    return candidates[0]


def _trace_signal(grid: np.ndarray, source: tuple[int, int]) -> tuple[int, int]:
    prev_coord: tuple[int, int] | None = None
    coord = source
    seen: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    while True:
        state = (prev_coord, coord)
        assert state not in seen, "signal path should not loop"
        seen.add(state)

        if prev_coord is not None and int(grid[coord]) == MAROON:
            return coord

        next_coord = _next_coord(grid, prev_coord, coord)
        assert next_coord is not None, "every source must reach a receiver"
        prev_coord, coord = coord, next_coord


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source signal through the grey wire network and recolor the receiver it reaches.

    The background is orange. Grey wires connect colored source endpoints to maroon receiver
    endpoints. At every red/blue junction marker, the signal turns right/left relative to its
    incoming direction. Wires, markers, and sources stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    used_receivers: set[tuple[int, int]] = set()

    for source in _source_positions(g):
        receiver = _trace_signal(g, source)
        assert receiver not in used_receivers, "receivers should be uniquely targeted"
        used_receivers.add(receiver)
        out[receiver] = int(g[source])

    return out


def _turn_for_marker(direction: tuple[int, int], marker_color: int) -> tuple[int, int]:
    return (RIGHT_TURN if marker_color == RED else LEFT_TURN)[direction]


def _destination_arm(source_arm: str, marker_color: int) -> str:
    incoming = ARM_TO_VECTOR[OPPOSITE_ARM[source_arm]]
    return VECTOR_TO_ARM[_turn_for_marker(incoming, marker_color)]


def _iter_arm_cells(center: tuple[int, int], arm: str, length: int) -> Iterable[tuple[int, int]]:
    dr, dc = ARM_TO_VECTOR[arm]
    r0, c0 = center
    for step in range(1, length + 1):
        yield (r0 + dr * step, c0 + dc * step)


def _gadget_cells(center: tuple[int, int], lengths: dict[str, int]) -> set[tuple[int, int]]:
    cells = {center}
    for arm, length in lengths.items():
        cells.update(_iter_arm_cells(center, arm, length))
    return cells


def _expanded_bbox(cells: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return (
        max(0, min(rows) - 1),
        min(shape[0] - 1, max(rows) + 1),
        max(0, min(cols) - 1),
        min(shape[1] - 1, max(cols) + 1),
    )


def _can_place(
    blocked: np.ndarray,
    shape: tuple[int, int],
    center: tuple[int, int],
    lengths: dict[str, int],
) -> bool:
    cells = _gadget_cells(center, lengths)
    if any(not _in_bounds(shape, r, c) for r, c in cells):
        return False

    top, bottom, left, right = _expanded_bbox(cells, shape)
    return not np.any(blocked[top : bottom + 1, left : right + 1])


def _place_gadget(
    grid: np.ndarray,
    blocked: np.ndarray,
    center: tuple[int, int],
    lengths: dict[str, int],
    marker_color: int,
    source_arms: tuple[str, str],
    source_colors: tuple[int, int],
) -> None:
    receiver_arms = tuple(_destination_arm(arm, marker_color) for arm in source_arms)
    assert len(set(receiver_arms)) == 2
    assert set(source_arms).isdisjoint(receiver_arms)

    grid[center] = marker_color
    for arm, length in lengths.items():
        cells = list(_iter_arm_cells(center, arm, length))
        for r, c in cells[:-1]:
            grid[r, c] = GREY

    for arm, color in zip(source_arms, source_colors):
        terminal = list(_iter_arm_cells(center, arm, lengths[arm]))[-1]
        grid[terminal] = color

    for arm in receiver_arms:
        terminal = list(_iter_arm_cells(center, arm, lengths[arm]))[-1]
        grid[terminal] = MAROON

    cells = _gadget_cells(center, lengths)
    top, bottom, left, right = _expanded_bbox(cells, grid.shape)
    blocked[top : bottom + 1, left : right + 1] = True


def _validate_generated_grid(grid: np.ndarray, expected_sources: int) -> bool:
    try:
        sources = _source_positions(grid)
        if len(sources) != expected_sources:
            return False
        receivers = [_trace_signal(grid, source) for source in sources]
        if len(set(receivers)) != expected_sources:
            return False
        out = main(grid)
    except AssertionError:
        return False

    return int(np.count_nonzero(out == MAROON)) + expected_sources == int(np.count_nonzero(grid == MAROON))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    gadget_count = random.choices(GADGET_COUNT_CHOICES, weights=GADGET_COUNT_WEIGHTS, k=1)[0]
    grid = np.full((height, width), ORANGE, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    total_sources = 2 * gadget_count
    source_palette = random.sample(SOURCE_COLORS, total_sources)
    color_index = 0

    for _ in range(gadget_count):
        placed = False
        for _attempt in range(PLACEMENT_RETRIES):
            lengths = {
                arm: random.randint(*ARM_LENGTH_RANGE)
                for arm in ("N", "E", "S", "W")
            }
            min_center_r = lengths["N"] + 1
            max_center_r = height - lengths["S"] - 2
            min_center_c = lengths["W"] + 1
            max_center_c = width - lengths["E"] - 2
            if min_center_r > max_center_r or min_center_c > max_center_c:
                continue
            center = (
                random.randint(min_center_r, max_center_r),
                random.randint(min_center_c, max_center_c),
            )
            if not _can_place(blocked, grid.shape, center, lengths):
                continue

            marker_color = random.choice((RED, BLUE))
            source_arms = random.choice((("N", "S"), ("E", "W")))
            source_colors = (source_palette[color_index], source_palette[color_index + 1])
            color_index += 2
            _place_gadget(grid, blocked, center, lengths, marker_color, source_arms, source_colors)
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated_grid(grid, expected_sources=total_sources):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
