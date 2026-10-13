import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9

BACKGROUND = GREY
OBSTACLE = BLACK
CENTER_COLOR = ORANGE
MARKER_COLORS = (BLUE, RED, GREEN, MAROON)
GRID_SIZE = 11
GEN_MAX_ATTEMPTS = 500


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _neighbors8(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _extract_markers(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1

    center = (h // 2, w // 2)
    assert int(g[center]) == CENTER_COLOR
    assert int(np.count_nonzero(g == CENTER_COLOR)) == 1

    marker_cells = np.argwhere(~np.isin(g, [BACKGROUND, OBSTACLE, CENTER_COLOR]))
    assert 3 <= len(marker_cells) <= 4

    markers: list[tuple[int, int, int, int]] = []
    seen_distances: set[int] = set()
    seen_colors: set[int] = set()
    for r0, c0 in marker_cells:
        r, c = int(r0), int(c0)
        color = int(g[r, c])
        assert color in MARKER_COLORS
        assert color not in seen_colors
        assert r != center[0] and c != center[1]
        d = _chebyshev((r, c), center)
        assert d > 0
        assert d not in seen_distances
        seen_distances.add(d)
        seen_colors.add(color)
        markers.append((d, color, r, c))

    markers.sort()
    return center, markers


def _cross_cells(
    grid: np.ndarray,
    marker: tuple[int, int, int, int],
    *,
    ignore_obstacles: bool = False,
) -> list[tuple[int, int]]:
    d, _color, r, c = marker
    cells: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for step in range(1, d + 1):
            rr = r + dr * step
            cc = c + dc * step
            if not _in_bounds(grid.shape, rr, cc):
                break
            if not ignore_obstacles and int(grid[rr, cc]) == OBSTACLE:
                break
            cells.append((rr, cc))
    return cells


def _solve(
    grid: np.ndarray,
    *,
    prefer_near: bool,
    ignore_obstacles: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _center, markers = _extract_markers(g)
    out = g.copy()

    best: dict[tuple[int, int], tuple[int, int]] = {}
    for marker in markers:
        d, color, _r, _c = marker
        for cell in _cross_cells(g, marker, ignore_obstacles=ignore_obstacles):
            if int(g[cell]) != BACKGROUND:
                continue
            if cell not in best:
                best[cell] = (d, color)
                continue
            prev_d, _prev_color = best[cell]
            if prefer_near:
                if d < prev_d:
                    best[cell] = (d, color)
            else:
                if d > prev_d:
                    best[cell] = (d, color)

    for (r, c), (_d, color) in best.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Paint crosses whose arm lengths and overlap priority both come from Chebyshev distance to the orange center.

    Input structure:
    - The grid is an odd square with a grey background, one orange center pixel at the exact grid center, several black obstacles, and 3 to 4 single-pixel colored markers.
    - Every marker lies off the center row and center column, and each marker has a distinct Chebyshev distance to the center.

    Rule:
    - For each marker at `(r, c)`, compute `d`, its Chebyshev distance to the center.
    - Draw a four-direction orthogonal cross centered on that marker, extending at most `d` cells per arm; an arm stops before a black obstacle or the grid edge, but other non-grey cells do not block it.
    - Recolor only the grey cells covered by these crosses, leaving the orange center, the black obstacles, and the marker pixels themselves unchanged.
    - If multiple markers cover the same grey cell, the marker with the smaller Chebyshev distance wins that overlap.

    Key insight:
    - The same distance-to-center value controls two different things at once: how far each cross reaches and which cross has priority where overlaps occur.
    """
    return _solve(grid, prefer_near=True)


def _empty_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    center = GRID_SIZE // 2
    grid[center, center] = CENTER_COLOR
    return grid


def _positions_at_distance(distance: int) -> list[tuple[int, int]]:
    center = GRID_SIZE // 2
    cells: list[tuple[int, int]] = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if r == center and c == center:
                continue
            if r == center or c == center:
                continue
            if _chebyshev((r, c), (center, center)) == distance:
                cells.append((r, c))
    return cells


def _markers_are_clear(markers: list[tuple[int, int, int, int]]) -> bool:
    cells = [(r, c) for _d, _color, r, c in markers]
    for i, (r1, c1) in enumerate(cells):
        for r2, c2 in cells[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _overlap_cells_for_markers(
    grid: np.ndarray, markers: list[tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for marker in markers:
        covered = set(_cross_cells(grid, marker))
        for cell in covered:
            if int(grid[cell]) != BACKGROUND:
                continue
            counts[cell] = counts.get(cell, 0) + 1
    return {cell for cell, count in counts.items() if count >= 2}


def _blocked_arm_candidates(
    grid: np.ndarray, markers: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for d, _color, r, c in markers:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for step in range(2, d + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not _in_bounds(grid.shape, rr, cc):
                    break
                if int(grid[rr, cc]) != BACKGROUND:
                    continue
                candidates.append((rr, cc))
    return candidates


def _forbidden_obstacle_cells(markers: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = {(GRID_SIZE // 2, GRID_SIZE // 2)}
    for _d, _color, r, c in markers:
        forbidden.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if _in_bounds((GRID_SIZE, GRID_SIZE), rr, cc):
                forbidden.add((rr, cc))
    return forbidden


def _can_place_obstacle(
    cell: tuple[int, int],
    obstacles: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    if cell in obstacles or cell in forbidden:
        return False
    return all(neigh not in obstacles for neigh in _neighbors8(cell))


def _sample_markers(rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    marker_count = rng.choice((3, 4))
    colors = rng.sample(MARKER_COLORS, marker_count)

    for _ in range(300):
        distances = rng.sample(range(1, GRID_SIZE // 2 + 1), marker_count)
        markers = []
        for distance, color in zip(distances, colors):
            pos = rng.choice(_positions_at_distance(distance))
            markers.append((distance, color, pos[0], pos[1]))
        if not _markers_are_clear(markers):
            continue

        trial = _empty_grid()
        for _d, color, r, c in markers:
            trial[r, c] = color
        if _overlap_cells_for_markers(trial, markers):
            return markers
    return None


def _render_input(
    markers: list[tuple[int, int, int, int]],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    grid = _empty_grid()
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    for _d, color, r, c in markers:
        grid[r, c] = color
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    markers = _sample_markers(rng)
    if markers is None:
        return None

    target_obstacles = rng.randint(3, 6)
    force_block = rng.random() < 0.75
    forbidden = _forbidden_obstacle_cells(markers)
    obstacles: set[tuple[int, int]] = set()

    blank = _render_input(markers, obstacles)
    if force_block:
        candidates = _blocked_arm_candidates(blank, markers)
        rng.shuffle(candidates)
        for cell in candidates:
            if _can_place_obstacle(cell, obstacles, forbidden):
                obstacles.add(cell)
                break
        if not obstacles:
            return None

    all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    rng.shuffle(all_cells)
    for cell in all_cells:
        if len(obstacles) >= target_obstacles:
            break
        if _can_place_obstacle(cell, obstacles, forbidden):
            obstacles.add(cell)
    if not (3 <= len(obstacles) <= 6):
        return None

    grid = _render_input(markers, obstacles)
    overlap_cells = _overlap_cells_for_markers(grid, markers)
    if not overlap_cells:
        return None

    correct = main(grid)
    reverse_priority = _solve(grid, prefer_near=False)
    if np.array_equal(correct, reverse_priority):
        return None

    if force_block:
        ignore_obstacles = _solve(grid, prefer_near=True, ignore_obstacles=True)
        if np.array_equal(correct, ignore_obstacles):
            return None

    if int(np.count_nonzero(correct != grid)) < 6:
        return None

    center, parsed_markers = _extract_markers(grid)
    assert center == (GRID_SIZE // 2, GRID_SIZE // 2)
    assert len(parsed_markers) == len(markers)
    assert len(obstacles) == int(np.count_nonzero(grid == OBSTACLE))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
