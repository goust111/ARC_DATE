import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

RAY_DIRECTIONS = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
MAX_GEN_ATTEMPTS = 400
RNG = random.Random()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _axis_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert w % 2 == 1, "expected an odd-width grid"
    axis = w // 2
    assert np.all(grid[:, axis] == GREY), "expected a full grey vertical axis"
    assert sum(bool(np.all(grid[:, c] == GREY)) for c in range(w)) == 1, "expected exactly one grey axis column"
    return axis


def _path_without_walls(
    shape: tuple[int, int], start: tuple[int, int], axis: int, direction: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc) or nc == axis:
            return cells
        cells.append((nr, nc))
        r, c = nr, nc


def _extract_objects(grid: np.ndarray) -> tuple[int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    axis = _axis_col(g)

    blue_coords = np.argwhere(g == BLUE)
    assert blue_coords.shape[0] == 1, "expected exactly one blue start pixel"
    blue = tuple(map(int, blue_coords[0]))
    assert blue[1] > axis, "blue start must be in the right half"

    left_reds = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    assert 3 <= len(left_reds) <= 7, "expected 3-7 red input pixels"
    for r, c in left_reds:
        assert 0 < r < h - 1, "left red pixels must avoid the top and bottom borders"
        assert 0 < c < axis, "left red pixels must stay strictly inside the left half"

    left_allowed = {BLACK, RED}
    right_allowed = {BLACK, BLUE}
    assert set(map(int, np.unique(g[:, :axis]))).issubset(left_allowed), "unexpected color in the left half"
    assert set(map(int, np.unique(g[:, axis + 1 :]))) .issubset(right_allowed), "unexpected color in the right half"

    walls = {(r, 2 * axis - c) for r, c in left_reds}
    assert blue not in walls, "the blue start must not coincide with a mirrored wall"
    return axis, blue, left_reds, walls


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    axis, blue, _left_reds, walls = _extract_objects(g)
    out = g.copy()
    ray_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for direction in RAY_DIRECTIONS:
        dr, dc = direction
        r, c = blue
        painted: list[tuple[int, int]] = []
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc) or nc == axis or (nr, nc) in walls:
                break
            out[nr, nc] = BLUE
            painted.append((nr, nc))
            r, c = nr, nc
        ray_cells[direction] = painted

    for r, c in walls:
        out[r, c] = RED
    return out, ray_cells


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the left pattern into hidden right-side walls, then stop four blue diagonal rays against those walls.

    Input structure:
    - The grid has a black background and one full grey vertical line exactly at the center column.
    - The left half contains 3-7 interior red pixels, while the right half contains exactly one isolated blue pixel.
    - No visible right-half red pixels are present in the input; they must be inferred by reflecting the left reds across the grey axis.

    Rule:
    - Reflect every left-half red pixel across the grey line to obtain a set of virtual wall positions in the right half.
    - Starting from the blue pixel, cast one ray in each diagonal direction in the fixed order right-up, right-down, left-up, left-down.
    - For each step, stop before the next cell if that cell would be outside the grid, on the grey axis, or on a virtual wall; already drawn ray pixels never block later rays.
    - After all four rays are drawn in blue, reveal every virtual wall as a red pixel while keeping the original left half and grey axis unchanged.

    Key insight:
    - The red pixels that matter for ray length are not visible on the right side until the end, so the solver must first mirror the left pattern mentally and only then simulate the diagonal stopping points.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _interior_right_cells(shape: tuple[int, int], axis: int) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(1, h - 1) for c in range(axis + 1, w - 1)]


def _choose_extra_wall(
    shape: tuple[int, int],
    axis: int,
    used: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    interior = _interior_right_cells(shape, axis)
    candidates = [cell for cell in interior if cell not in used and cell not in forbidden]
    if not candidates:
        return None

    if used and RNG.random() < 0.55:
        adjacent = [
            cell
            for cell in candidates
            if any(abs(cell[0] - ur) + abs(cell[1] - uc) == 1 for ur, uc in used)
        ]
        if adjacent:
            return RNG.choice(adjacent)
    return RNG.choice(candidates)


def _nearest_wall_distance(
    path: list[tuple[int, int]], walls: Iterable[tuple[int, int]]
) -> int | None:
    wall_set = set(walls)
    for idx, cell in enumerate(path):
        if cell in wall_set:
            return idx
    return None


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    axis, blue, left_reds, walls = _extract_objects(g)
    out, ray_cells = _simulate(g)
    assert out.shape == g.shape

    structural_paths = {
        direction: _path_without_walls(g.shape, blue, axis, direction) for direction in RAY_DIRECTIONS
    }
    union_structural = {cell for cells in structural_paths.values() for cell in cells}

    blocker_dirs = []
    free_dirs = []
    for direction in RAY_DIRECTIONS:
        path = structural_paths[direction]
        stop_idx = _nearest_wall_distance(path, walls)
        if stop_idx is None:
            free_dirs.append(direction)
            assert ray_cells[direction] == path, "unblocked rays must reach their structural limit"
        else:
            blocker_dirs.append(direction)
            assert ray_cells[direction] == path[:stop_idx], "blocked rays must stop before the first wall"
            assert stop_idx < len(path), "wall stop must happen within the structural path"

    assert blocker_dirs, "at least one ray must be shortened by a mirrored wall"
    assert free_dirs, "at least one ray must stop by axis or boundary instead of a wall"
    assert any(cell not in union_structural for cell in walls), "need at least one mirrored wall off every ray path"
    assert any(cell in union_structural for cell in walls), "need at least one mirrored wall on a ray path"
    assert blue not in walls
    assert len(left_reds) == len(walls)
    assert np.count_nonzero(g == BLUE) == 1

    left = g[:, :axis]
    right = g[:, axis + 1 :]
    assert np.all((left == BLACK) | (left == RED))
    assert np.all((right == BLACK) | (right == BLUE))

    naive = g.copy()
    for direction, path in structural_paths.items():
        for r, c in path:
            naive[r, c] = BLUE
    for r, c in walls:
        naive[r, c] = RED
    assert not np.array_equal(out, naive), "mirrored walls must change at least one ray length"


def _attempt_generate() -> np.ndarray | None:
    h = RNG.randint(13, 19)
    half_w = RNG.randint(6, 9)
    w = half_w * 2 + 1
    axis = w // 2

    blue_r = RNG.randint(2, max(2, h // 2))
    blue_c = RNG.randint(axis + 2, w - 3)
    blue = (blue_r, blue_c)

    structural_paths = {
        direction: _path_without_walls((h, w), blue, axis, direction) for direction in RAY_DIRECTIONS
    }
    blocker_candidates = {
        direction: [
            cell
            for cell in path
            if 0 < cell[0] < h - 1 and axis < cell[1] < w - 1
        ]
        for direction, path in structural_paths.items()
    }
    viable_dirs = [direction for direction, cells in blocker_candidates.items() if cells]
    if not viable_dirs:
        return None

    total_red = RNG.randint(3, 7)
    max_blockers = min(len(viable_dirs), total_red - 1, 3)
    if max_blockers <= 0:
        return None
    blocked_count = RNG.randint(1, max_blockers)
    blocked_dirs = RNG.sample(viable_dirs, blocked_count)

    walls: set[tuple[int, int]] = set()
    union_structural = {cell for path in structural_paths.values() for cell in path}
    for direction in blocked_dirs:
        choices = blocker_candidates[direction][:]
        RNG.shuffle(choices)
        preferred = [cell for cell in choices if cell != structural_paths[direction][0]]
        chosen_pool = preferred if preferred and RNG.random() < 0.75 else choices
        walls.add(chosen_pool[0])

    while len(walls) < total_red:
        extra = _choose_extra_wall((h, w), axis, walls, union_structural | {blue})
        if extra is None:
            return None
        walls.add(extra)

    if not any(cell in union_structural for cell in walls):
        return None
    if not any(cell not in union_structural for cell in walls):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[:, axis] = GREY
    grid[blue] = BLUE
    for r, c in walls:
        grid[r, 2 * axis - c] = RED

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
