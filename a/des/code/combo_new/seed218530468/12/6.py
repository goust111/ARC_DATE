import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
DIRECTION_NAMES = (
    "horizontal",
    "vertical",
    "main_diagonal",
    "anti_diagonal",
)
DIRECTION_STEPS = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "main_diagonal": (1, 1),
    "anti_diagonal": (1, -1),
}
COUNT_TEMPLATES = (
    (2, 1, 1, 1),
    (3, 1, 1, 1),
    (3, 2, 1, 1),
    (3, 2, 2, 1),
    (4, 1, 1, 1),
    (4, 2, 1, 1),
    (4, 2, 2, 1),
)
EDGE_MARGIN = 1
HALO = 1
MIN_SEGMENT_LEN = 3
MAX_SEGMENT_LEN = 8
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 500


def _classify_pair(p0: tuple[int, int], p1: tuple[int, int]) -> str:
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    if dr == 0 and dc != 0:
        return "horizontal"
    if dc == 0 and dr != 0:
        return "vertical"
    if dr == dc and dr != 0:
        return "main_diagonal"
    if dr == -dc and dr != 0:
        return "anti_diagonal"
    raise ValueError(f"pair {p0}, {p1} is not aligned to a supported direction")


def _line_cells_between(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(np.sign(p1[0] - p0[0]))
    dc = int(np.sign(p1[1] - p0[1]))
    length = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])))
    cells = [(int(p0[0] + step * dr), int(p0[1] + step * dc)) for step in range(length + 1)]
    assert cells[-1] == p1
    return cells


def _extract_pairs(grid: np.ndarray) -> dict[str, list[tuple[int, tuple[int, int], tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    directions = {name: [] for name in DIRECTION_NAMES}
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BACKGROUND):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        p0 = tuple(map(int, coords[0]))
        p1 = tuple(map(int, coords[1]))
        direction = _classify_pair(p0, p1)
        directions[direction].append((color, p0, p1))
    return directions


def _winner_direction(
    pairs_by_direction: dict[str, list[tuple[int, tuple[int, int], tuple[int, int]]]]
) -> str:
    counts = {name: len(pairs) for name, pairs in pairs_by_direction.items()}
    assert all(count >= 1 for count in counts.values()), "every direction must be represented"
    winner = max(DIRECTION_NAMES, key=lambda name: counts[name])
    assert list(counts.values()).count(counts[winner]) == 1, "expected a unique majority direction"
    return winner


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pairs_by_direction = _extract_pairs(g)
    winner = _winner_direction(pairs_by_direction)

    out = g.copy()
    for color, p0, p1 in pairs_by_direction[winner]:
        for r, c in _line_cells_between(p0, p1):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect only the pixel pairs whose shared direction category has the unique highest count.

    Input structure:
    - The background is black.
    - Each non-black color appears exactly twice, forming one same-color endpoint pair.
    - Every pair is aligned in exactly one of four ways: horizontal, vertical,
      main diagonal, or anti diagonal.
    - All four directions are present, and exactly one direction contains more pairs
      than any other direction.

    Rule:
    - Group the same-color endpoint pairs by their alignment direction.
    - Count how many pairs belong to each of the four direction groups.
    - Identify the unique majority direction.
    - For every pair in that majority group, fill the full straight segment between
      its two endpoints with the pair color.
    - Leave all non-majority pairs unchanged as isolated endpoints.

    Key insight:
    The puzzle is not about individual colors; it first hides a geometric
    classification by direction, then applies the drawing action only to the
    direction class that wins the pair-count vote.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _transform_grid(g)


def _sample_counts(rng: np.random.Generator) -> tuple[dict[str, int], str]:
    template = COUNT_TEMPLATES[int(rng.integers(0, len(COUNT_TEMPLATES)))]
    winner = DIRECTION_NAMES[int(rng.integers(0, len(DIRECTION_NAMES)))]
    remaining_dirs = [name for name in DIRECTION_NAMES if name != winner]
    remaining_counts = list(template[1:])
    rng.shuffle(remaining_dirs)
    rng.shuffle(remaining_counts)

    counts = {winner: int(template[0])}
    for name, count in zip(remaining_dirs, remaining_counts):
        counts[name] = int(count)
    return counts, winner


def _max_length_for(direction: str, height: int, width: int) -> int:
    if direction == "horizontal":
        return min(MAX_SEGMENT_LEN, width - 2 * EDGE_MARGIN)
    if direction == "vertical":
        return min(MAX_SEGMENT_LEN, height - 2 * EDGE_MARGIN)
    return min(MAX_SEGMENT_LEN, height - 2 * EDGE_MARGIN, width - 2 * EDGE_MARGIN)


def _random_segment_cells(
    direction: str, length: int, height: int, width: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    if direction == "horizontal":
        row = int(rng.integers(EDGE_MARGIN, height - EDGE_MARGIN))
        col0 = int(rng.integers(EDGE_MARGIN, width - EDGE_MARGIN - length + 1))
        return [(row, int(col0 + k)) for k in range(length)]
    if direction == "vertical":
        row0 = int(rng.integers(EDGE_MARGIN, height - EDGE_MARGIN - length + 1))
        col = int(rng.integers(EDGE_MARGIN, width - EDGE_MARGIN))
        return [(int(row0 + k), col) for k in range(length)]
    if direction == "main_diagonal":
        row0 = int(rng.integers(EDGE_MARGIN, height - EDGE_MARGIN - length + 1))
        col0 = int(rng.integers(EDGE_MARGIN, width - EDGE_MARGIN - length + 1))
        return [(int(row0 + k), int(col0 + k)) for k in range(length)]

    assert direction == "anti_diagonal"
    row0 = int(rng.integers(EDGE_MARGIN, height - EDGE_MARGIN - length + 1))
    col0 = int(rng.integers(EDGE_MARGIN + length - 1, width - EDGE_MARGIN))
    return [(int(row0 + k), int(col0 - k)) for k in range(length)]


def _cells_are_clear(cells: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    return all(not bool(blocked[r, c]) for r, c in cells)


def _reserve_cells(cells: list[tuple[int, int]], blocked: np.ndarray) -> None:
    height, width = blocked.shape
    for r, c in cells:
        r0 = max(0, r - HALO)
        r1 = min(height, r + HALO + 1)
        c0 = max(0, c - HALO)
        c1 = min(width, c + HALO + 1)
        blocked[r0:r1, c0:c1] = True


def _intended_cell_count(color: int, winner: str, pairs_by_direction: dict[str, list[tuple[int, tuple[int, int], tuple[int, int]]]]) -> int:
    for direction, pairs in pairs_by_direction.items():
        for pair_color, p0, p1 in pairs:
            if pair_color != color:
                continue
            if direction != winner:
                return 2
            return len(_line_cells_between(p0, p1))
    raise AssertionError(f"unknown color {color}")


def _is_valid_candidate(grid: np.ndarray, target_counts: dict[str, int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        pairs_by_direction = _extract_pairs(grid)
        winner = _winner_direction(pairs_by_direction)
        out = _transform_grid(grid)
    except Exception:
        return False

    if {name: len(pairs_by_direction[name]) for name in DIRECTION_NAMES} != target_counts:
        return False

    for color in (int(c) for c in np.unique(grid) if int(c) != BACKGROUND):
        if int(np.count_nonzero(out == color)) != _intended_cell_count(color, winner, pairs_by_direction):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    counts, _winner = _sample_counts(rng)
    total_pairs = int(sum(counts.values()))

    min_side = max(12, 10 + total_pairs)
    height = int(rng.integers(min_side, 31))
    width = int(rng.integers(min_side, 31))

    specs: list[dict[str, object]] = []
    for direction in DIRECTION_NAMES:
        for _ in range(counts[direction]):
            max_len = _max_length_for(direction, height, width)
            if max_len < MIN_SEGMENT_LEN:
                return None
            length = int(rng.integers(MIN_SEGMENT_LEN, max_len + 1))
            specs.append({"direction": direction, "length": length})
    specs.sort(key=lambda spec: int(spec["length"]), reverse=True)

    blocked = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []
    for spec in specs:
        direction = str(spec["direction"])
        length = int(spec["length"])
        cells = None
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = _random_segment_cells(direction, length, height, width, rng)
            if _cells_are_clear(candidate, blocked):
                cells = candidate
                break
        if cells is None:
            return None
        _reserve_cells(cells, blocked)
        placed.append({"direction": direction, "cells": cells})

    colors = [int(c) for c in rng.permutation(np.array(COLOR_POOL))[:total_pairs]]
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for color, item in zip(colors, placed):
        cells = item["cells"]
        grid[cells[0]] = color
        grid[cells[-1]] = color

    if not _is_valid_candidate(grid, counts):
        return None
    if not np.array_equal(main(grid), _transform_grid(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
