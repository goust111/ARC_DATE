from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5


def _ordered_marker_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in grid.flat:
        color = int(value)
        if color in (BLACK, GREY) or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("marker pair must share a row or a column")


def _marker_segments(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    segments: list[tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    for color in _ordered_marker_colors(grid):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 2, "each marker color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        cells = _segment_cells(a, b)
        segments.append((color, a, b, cells))
    assert segments, "expected at least one colored marker pair"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned same-color endpoints, turning the interior red when grey
    obstacles lie on the segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color, a, b, cells in _marker_segments(g):
        interior = cells[1:-1]
        blocked = False
        for r, c in interior:
            value = int(g[r, c])
            assert value in (BLACK, GREY), "segment interiors may only contain background or grey"
            if value == GREY:
                blocked = True

        if blocked:
            for r, c in interior:
                if int(g[r, c]) != GREY:
                    out[r, c] = RED
            out[a[0], a[1]] = color
            out[b[0], b[1]] = color
        else:
            for r, c in cells:
                out[r, c] = color

    return out


# --------------- generator ---------------

MARKER_POOL = [1, 3, 4, 6, 7, 8, 9]
ORIENTATIONS = ("horizontal", "vertical")

PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 5
GRID_SIDE_MIN = 9
GRID_SIDE_MAX = 18
MIN_SEPARATION = 4
MIN_LANE_GAP = 2
MAX_ON_PATH_OBSTACLES = 3
MAX_EXTRA_OBSTACLES = 5
GEN_RETRIES = 300
POSITION_RETRIES = 100


def _sample_spaced_positions(
    count: int,
    lo: int,
    hi: int,
    min_gap: int,
    rng: random.Random,
) -> list[int] | None:
    if hi - lo + 1 < count:
        return None
    candidates = list(range(lo, hi + 1))
    for _ in range(POSITION_RETRIES):
        positions = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _blocked_flags(count: int, rng: random.Random) -> list[bool]:
    blocked_count = count // 2
    if count % 2 == 1 and rng.random() < 0.5:
        blocked_count += 1
    indices = set(rng.sample(range(count), blocked_count))
    return [idx in indices for idx in range(count)]


def _place_horizontal_pair(
    grid: np.ndarray,
    row: int,
    color: int,
    blocked: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    width = grid.shape[1]
    max_separation = width - 3
    if max_separation < MIN_SEPARATION:
        return None

    separation = rng.randint(MIN_SEPARATION, max_separation)
    c0 = rng.randint(1, width - 2 - separation)
    c1 = c0 + separation
    cells = [(row, c) for c in range(c0, c1 + 1)]
    interior = cells[1:-1]

    grid[row, c0] = color
    grid[row, c1] = color
    if blocked:
        obstacle_cap = min(MAX_ON_PATH_OBSTACLES, len(interior) - 1)
        if obstacle_cap < 1:
            return None
        obstacle_count = rng.randint(1, obstacle_cap)
        for r, c in rng.sample(interior, obstacle_count):
            grid[r, c] = GREY
    return cells


def _place_vertical_pair(
    grid: np.ndarray,
    col: int,
    color: int,
    blocked: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    height = grid.shape[0]
    max_separation = height - 3
    if max_separation < MIN_SEPARATION:
        return None

    separation = rng.randint(MIN_SEPARATION, max_separation)
    r0 = rng.randint(1, height - 2 - separation)
    r1 = r0 + separation
    cells = [(r, col) for r in range(r0, r1 + 1)]
    interior = cells[1:-1]

    grid[r0, col] = color
    grid[r1, col] = color
    if blocked:
        obstacle_cap = min(MAX_ON_PATH_OBSTACLES, len(interior) - 1)
        if obstacle_cap < 1:
            return None
        obstacle_count = rng.randint(1, obstacle_cap)
        for r, c in rng.sample(interior, obstacle_count):
            grid[r, c] = GREY
    return cells


def _validate_generated_grid(grid: np.ndarray) -> None:
    assert grid.ndim == 2
    seen_path_cells: set[tuple[int, int]] = set()
    blocked_seen = False
    clear_seen = False

    for color, _a, _b, cells in _marker_segments(grid):
        assert color in MARKER_POOL or color == RED
        for cell in cells:
            assert cell not in seen_path_cells, "segments must not overlap"
            seen_path_cells.add(cell)

        interior = cells[1:-1]
        assert interior, "each segment must have a non-empty interior"
        has_grey = False
        for r, c in interior:
            value = int(grid[r, c])
            assert value in (BLACK, GREY), "unexpected color on a segment interior"
            has_grey |= value == GREY
        blocked_seen |= has_grey
        clear_seen |= not has_grey

    assert blocked_seen and clear_seen, "generator must cover both rule branches"
    assert np.count_nonzero(grid == GREY) >= 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    orientation = rng.choice(ORIENTATIONS)

    if orientation == "horizontal":
        height_min = max(GRID_SIDE_MIN, 2 * pair_count + 1)
        width_min = max(GRID_SIDE_MIN, MIN_SEPARATION + 3)
    else:
        height_min = max(GRID_SIDE_MIN, MIN_SEPARATION + 3)
        width_min = max(GRID_SIDE_MIN, 2 * pair_count + 1)

    height = rng.randint(height_min, GRID_SIDE_MAX)
    width = rng.randint(width_min, GRID_SIDE_MAX)
    grid = np.full((height, width), BLACK, dtype=int)

    colors = rng.sample(MARKER_POOL, pair_count)
    blocked_flags = _blocked_flags(pair_count, rng)
    path_cells: set[tuple[int, int]] = set()

    if orientation == "horizontal":
        rows = _sample_spaced_positions(pair_count, 1, height - 2, MIN_LANE_GAP, rng)
        if rows is None:
            return None
        for row, color, blocked in zip(rows, colors, blocked_flags):
            cells = _place_horizontal_pair(grid, row, color, blocked, rng)
            if cells is None:
                return None
            path_cells.update(cells)
    else:
        cols = _sample_spaced_positions(pair_count, 1, width - 2, MIN_LANE_GAP, rng)
        if cols is None:
            return None
        for col, color, blocked in zip(cols, colors, blocked_flags):
            cells = _place_vertical_pair(grid, col, color, blocked, rng)
            if cells is None:
                return None
            path_cells.update(cells)

    extra_candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if int(grid[r, c]) == BLACK and (r, c) not in path_cells
    ]
    if extra_candidates:
        extra_count = rng.randint(0, min(MAX_EXTRA_OBSTACLES, len(extra_candidates)))
        for r, c in rng.sample(extra_candidates, extra_count):
            grid[r, c] = GREY

    _validate_generated_grid(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "output should differ from input"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
