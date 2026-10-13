import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BLACK = 0
WALL = 1
PALETTE_SAMPLES = 5


def _longest_true_run(values: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in values:
        if bool(value):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _detect_wall_color(maze: np.ndarray) -> int:
    colors, counts = np.unique(maze, return_counts=True)
    non_black = [(int(color), int(count)) for color, count in zip(colors, counts) if int(color) != BLACK]
    assert non_black, "maze must contain walls"
    return max(non_black, key=lambda item: item[1])[0]


def _read_palette(grid: np.ndarray, wall_color: int) -> Dict[int, int]:
    palette = [int(value) for value in grid[-1] if int(value) not in (BLACK, wall_color)]
    assert len(palette) == PALETTE_SAMPLES, "expected five palette samples"
    assert len(set(palette)) == PALETTE_SAMPLES, "palette colors must be distinct"
    return {idx + 1: color for idx, color in enumerate(palette)}


def _detect_grid_lines(maze: np.ndarray, wall_color: int) -> Tuple[List[int], List[int]]:
    wall_mask = maze == int(wall_color)
    rows = [
        r
        for r in range(maze.shape[0])
        if int(np.count_nonzero(wall_mask[r])) >= 2 and _longest_true_run(wall_mask[r]) >= 2
    ]
    cols = [
        c
        for c in range(maze.shape[1])
        if int(np.count_nonzero(wall_mask[:, c])) >= 2 and _longest_true_run(wall_mask[:, c]) >= 2
    ]
    assert rows and cols, "failed to detect wall grid"
    return rows, cols


def _room_boxes(row_lines: Sequence[int], col_lines: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for r0, r1 in zip(row_lines, row_lines[1:]):
        if r1 - r0 <= 1:
            continue
        for c0, c1 in zip(col_lines, col_lines[1:]):
            if c1 - c0 <= 1:
                continue
            boxes.append((int(r0), int(r1), int(c0), int(c1)))
    return boxes


def _count_room_exits(maze: np.ndarray, box: Tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = box
    total = 0
    total += int(np.count_nonzero(maze[r0, c0 + 1 : c1] == BLACK))
    total += int(np.count_nonzero(maze[r1, c0 + 1 : c1] == BLACK))
    total += int(np.count_nonzero(maze[r0 + 1 : r1, c0] == BLACK))
    total += int(np.count_nonzero(maze[r0 + 1 : r1, c1] == BLACK))
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room using the palette color indexed by its exit count.

    The bottom row contains five single-cell palette samples. In the wall maze
    above, every room is the rectangle between two consecutive horizontal wall
    lines and two consecutive vertical wall lines. A black pixel on one of the
    room's four wall segments is an exit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 9

    maze = g[:-1]
    wall_color = _detect_wall_color(maze)
    palette = _read_palette(g, wall_color)
    row_lines, col_lines = _detect_grid_lines(maze, wall_color)

    out = g.copy()
    for box in _room_boxes(row_lines, col_lines):
        r0, r1, c0, c1 = box
        interior = maze[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == wall_color):
            continue
        exit_count = _count_room_exits(maze, box)
        if exit_count not in palette:
            continue
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[patch == BLACK] = int(palette[exit_count])
    return out


GEN_ATTEMPTS = 400
ROOM_ROW_RANGE = (2, 4)
ROOM_COL_RANGE = (2, 4)
ROOM_H_RANGE = (2, 6)
ROOM_W_RANGE = (2, 6)
TOTAL_H_RANGE = (12, 24)
TOTAL_W_RANGE = (12, 24)
MODE_CHOICES = ("sparse", "mixed", "dense")
MODE_WEIGHTS = (3, 4, 3)
SEGMENT_HOLE_WEIGHTS = {
    "sparse": ((0, 1, 2), (12, 4, 1)),
    "mixed": ((0, 1, 2), (8, 6, 2)),
    "dense": ((0, 1, 2), (5, 6, 3)),
}
MODE_MIN_PEAK = {"sparse": 1, "mixed": 2, "dense": 3}


def _line_positions(sizes: Sequence[int]) -> List[int]:
    positions = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        positions.append(cur)
    return positions


def _sample_layout(rng: random.Random) -> Tuple[List[int], List[int]] | None:
    room_rows = rng.randint(*ROOM_ROW_RANGE)
    room_cols = rng.randint(*ROOM_COL_RANGE)
    heights = [rng.randint(*ROOM_H_RANGE) for _ in range(room_rows)]
    widths = [rng.randint(*ROOM_W_RANGE) for _ in range(room_cols)]

    total_h = sum(heights) + room_rows + 2
    total_w = sum(widths) + room_cols + 1
    if not (TOTAL_H_RANGE[0] <= total_h <= TOTAL_H_RANGE[1]):
        return None
    if not (TOTAL_W_RANGE[0] <= total_w <= TOTAL_W_RANGE[1]):
        return None
    if total_w < 2 * PALETTE_SAMPLES - 1:
        return None
    return heights, widths


def _base_grid(heights: Sequence[int], widths: Sequence[int]) -> Tuple[np.ndarray, List[int], List[int]]:
    row_lines = _line_positions(heights)
    col_lines = _line_positions(widths)
    maze_h = row_lines[-1] + 1
    maze_w = col_lines[-1] + 1
    grid = np.zeros((maze_h + 1, maze_w), dtype=int)
    grid[row_lines, :] = WALL
    grid[:maze_h, col_lines] = WALL
    return grid, row_lines, col_lines


def _palette_colors(rng: random.Random) -> List[int]:
    return rng.sample([2, 3, 4, 5, 6, 7, 8, 9], PALETTE_SAMPLES)


def _place_palette(grid: np.ndarray, colors: Sequence[int], rng: random.Random) -> Dict[int, int]:
    start = rng.randint(0, grid.shape[1] - (2 * PALETTE_SAMPLES - 1))
    for idx, color in enumerate(colors):
        grid[-1, start + 2 * idx] = int(color)
    return {idx + 1: int(color) for idx, color in enumerate(colors)}


def _segment_records(row_lines: Sequence[int], col_lines: Sequence[int]) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    room_rows = len(row_lines) - 1
    room_cols = len(col_lines) - 1
    segments: List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = []

    for k, row in enumerate(row_lines):
        for j in range(room_cols):
            left = int(col_lines[j])
            right = int(col_lines[j + 1])
            coords = [(int(row), c) for c in range(left + 1, right)]
            rooms: List[Tuple[int, int]] = []
            if k > 0:
                rooms.append((k - 1, j))
            if k < room_rows:
                rooms.append((k, j))
            segments.append((coords, rooms))

    for i in range(room_rows):
        top = int(row_lines[i])
        bottom = int(row_lines[i + 1])
        for k, col in enumerate(col_lines):
            coords = [(r, int(col)) for r in range(top + 1, bottom)]
            rooms = []
            if k > 0:
                rooms.append((i, k - 1))
            if k < room_cols:
                rooms.append((i, k))
            segments.append((coords, rooms))

    return segments


def _pick_non_adjacent_indices(length: int, count: int, rng: random.Random) -> List[int]:
    available = list(range(length))
    chosen: List[int] = []
    while available and len(chosen) < count:
        idx = rng.choice(available)
        chosen.append(int(idx))
        available = [j for j in available if abs(j - idx) > 1]
    chosen.sort()
    return chosen


def _carve_gaps(
    grid: np.ndarray,
    row_lines: Sequence[int],
    col_lines: Sequence[int],
    mode: str,
    rng: random.Random,
) -> None:
    hole_values, weights = SEGMENT_HOLE_WEIGHTS[mode]
    for coords, _rooms in _segment_records(row_lines, col_lines):
        if not coords:
            continue
        max_holes = min(2, (len(coords) + 1) // 2)
        wanted = rng.choices(hole_values, weights=weights, k=1)[0]
        hole_count = min(int(wanted), int(max_holes))
        for idx in _pick_non_adjacent_indices(len(coords), hole_count, rng):
            r, c = coords[idx]
            grid[r, c] = BLACK


def _room_exit_counts(grid: np.ndarray, row_lines: Sequence[int], col_lines: Sequence[int]) -> np.ndarray:
    maze = grid[:-1]
    counts = np.zeros((len(row_lines) - 1, len(col_lines) - 1), dtype=int)
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            counts[i, j] = _count_room_exits(
                maze,
                (int(row_lines[i]), int(row_lines[i + 1]), int(col_lines[j]), int(col_lines[j + 1])),
            )
    return counts


def _expected_output(
    grid: np.ndarray,
    row_lines: Sequence[int],
    col_lines: Sequence[int],
    palette: Dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    maze = grid[:-1]
    for box in _room_boxes(row_lines, col_lines):
        exits = _count_room_exits(maze, box)
        if exits not in palette:
            continue
        r0, r1, c0, c1 = box
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[patch == BLACK] = int(palette[exits])
    return out


def _valid_counts(counts: np.ndarray, mode: str, rng: random.Random) -> bool:
    max_count = int(counts.max())
    if max_count == 0 or max_count > 5:
        return False
    positive = {int(value) for value in counts.flat if int(value) > 0}
    if len(positive) < 2:
        return False
    if max_count < MODE_MIN_PEAK[mode]:
        return False
    if rng.random() < 0.5 and not bool(np.any(counts == 0)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    layout = _sample_layout(rng)
    if layout is None:
        return None
    heights, widths = layout
    mode = rng.choices(MODE_CHOICES, weights=MODE_WEIGHTS, k=1)[0]

    grid, row_lines, col_lines = _base_grid(heights, widths)
    palette = _place_palette(grid, _palette_colors(rng), rng)
    _carve_gaps(grid, row_lines, col_lines, mode, rng)

    counts = _room_exit_counts(grid, row_lines, col_lines)
    if not _valid_counts(counts, mode, rng):
        return None

    detected_rows, detected_cols = _detect_grid_lines(grid[:-1], WALL)
    if detected_rows != list(row_lines) or detected_cols != list(col_lines):
        return None

    expected = _expected_output(grid, row_lines, col_lines, palette)
    if np.array_equal(expected, grid):
        return None
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
    raise RuntimeError("failed to generate a valid room-exit palette puzzle")
