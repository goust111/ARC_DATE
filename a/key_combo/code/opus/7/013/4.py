from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_4_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    cells = set(coords)
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _wall_axis(row: np.ndarray) -> int | None:
    zeros = np.flatnonzero(row == BLACK)
    if len(zeros) != 3:
        return None
    if not np.array_equal(zeros, np.arange(zeros[0], zeros[0] + 3)):
        return None
    if not np.all((row == GREY) | (row == BLACK)):
        return None
    if int(np.count_nonzero(row == GREY)) != len(row) - 3:
        return None
    return int(zeros[1])


def _find_walls_and_axes(grid: np.ndarray) -> tuple[list[int], list[int]]:
    walls: list[int] = []
    axes: list[int] = []
    for r in range(grid.shape[0]):
        axis = _wall_axis(grid[r])
        if axis is not None:
            walls.append(r)
            axes.append(axis)
    assert len(walls) >= 2, "expected multiple horizontal walls with 3-cell doors"
    return walls, axes


def _corridor_specs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    walls, axes = _find_walls_and_axes(grid)
    specs: list[tuple[int, int, int]] = []
    if walls[0] > 0:
        specs.append((0, walls[0] - 1, axes[0]))
    for i in range(len(walls) - 1):
        specs.append((walls[i] + 1, walls[i + 1] - 1, axes[i]))
    return [(r0, r1, axis) for r0, r1, axis in specs if r0 <= r1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete each corridor's red motif using the upper wall door center as that corridor's vertical axis.

    Input structure:
    - The grid has a black background, several full-width horizontal grey wall rows, and one 3-cell black door in each wall.
    - The top corridor or each corridor between adjacent walls contains one small 4-connected red motif on exactly one side of its corridor axis.
    - A corridor's axis is not global: it is the center column of the door cut into that corridor's upper wall.

    Rule:
    - Detect every horizontal grey wall row and the center column of its 3-cell black door.
    - For the top corridor, use the first wall's door center as the mirror axis; for each corridor between two walls, use the upper wall's door center.
    - Preserve the original grid and add red pixels at the left-right mirror positions of every red corridor pixel, drawing only when the mirrored column stays inside the grid.

    Key insight:
    - The doors do not mark where to paint directly; their center columns parameterize a different mirror axis for each corridor, so the scene must be solved corridor by corridor rather than with one fixed symmetry line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    width = g.shape[1]
    for r0, r1, axis in _corridor_specs(g):
        red_coords = np.argwhere(g[r0 : r1 + 1] == RED)
        for local_r, c in red_coords:
            rr = r0 + int(local_r)
            cc = int(c)
            mirror_c = 2 * axis - cc
            if 0 <= mirror_c < width:
                out[rr, mirror_c] = RED
    return out


GRID_H_RANGE = (11, 20)
GRID_W_RANGE = (14, 22)
CORRIDOR_COUNT_CHOICES = (2, 3)
CORRIDOR_HEIGHT_RANGE = (3, 5)
PATTERN_SIZE_RANGE = (2, 5)
GEN_RETRIES = 400


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(sorted(coords), dtype=int)
    r0, c0, _r1, _c1 = _bbox_coords(arr)
    out = [(int(r - r0), int(c - c0)) for r, c in arr]
    return sorted(out)


def _random_polyomino(area: int, max_h: int, max_w: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        shape = _normalize(cells)
        arr = np.array(shape, dtype=int)
        r0, c0, r1, c1 = _bbox_coords(arr)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h <= max_h and w <= max_w:
            return shape
    return None


def _safe_col_band(width: int, axis: int, side: str) -> tuple[int, int]:
    if side == "left":
        return max(0, 2 * axis - (width - 1)), axis - 1
    return axis + 1, min(width - 1, 2 * axis)


def _door_centers(width: int, wall_count: int) -> list[int] | None:
    centers = list(range(3, width - 3))
    if len(centers) < wall_count:
        return None
    return random.sample(centers, wall_count)


def _choose_sides(corridor_count: int) -> list[str]:
    if corridor_count == 2:
        sides = ["left", "right"]
    else:
        sides = ["left", "right", random.choice(["left", "right"])]
    random.shuffle(sides)
    return sides


def _draw_wall_row(row: np.ndarray, axis: int) -> None:
    row[:] = GREY
    row[axis - 1 : axis + 2] = BLACK


def _place_corridor_pattern(
    grid: np.ndarray,
    r0: int,
    r1: int,
    axis: int,
    side: str,
) -> bool:
    height = r1 - r0 + 1
    width = grid.shape[1]
    col_lo, col_hi = _safe_col_band(width, axis, side)
    if col_lo > col_hi:
        return False

    max_shape_w = min(5, col_hi - col_lo + 1)
    if max_shape_w <= 0:
        return False

    area_choices = list(range(PATTERN_SIZE_RANGE[0], PATTERN_SIZE_RANGE[1] + 1))
    random.shuffle(area_choices)
    for area in area_choices:
        shape = _random_polyomino(area, height, max_shape_w)
        if shape is None:
            continue
        arr = np.array(shape, dtype=int)
        _sr0, _sc0, sr1, sc1 = _bbox_coords(arr)
        sh = sr1 + 1
        sw = sc1 + 1
        if sh > height or sw > (col_hi - col_lo + 1):
            continue

        top = r0 + random.randint(0, height - sh)
        left = random.randint(col_lo, col_hi - sw + 1)
        painted: list[tuple[int, int]] = []
        for dr, dc in shape:
            rr = top + dr
            cc = left + dc
            mirror_c = 2 * axis - cc
            if not (0 <= mirror_c < width):
                painted = []
                break
            painted.append((rr, cc))
        if not painted:
            continue
        for rr, cc in painted:
            grid[rr, cc] = RED
        return True
    return False


def _build_input() -> np.ndarray | None:
    corridor_count = random.choice(CORRIDOR_COUNT_CHOICES)
    corridor_heights = [random.randint(*CORRIDOR_HEIGHT_RANGE) for _ in range(corridor_count)]
    wall_count = corridor_count + 1
    height = sum(corridor_heights) + wall_count
    if not (GRID_H_RANGE[0] <= height <= GRID_H_RANGE[1]):
        return None

    width = random.randint(*GRID_W_RANGE)
    centers = _door_centers(width, wall_count)
    if centers is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    wall_rows = [0]
    for gap in corridor_heights:
        wall_rows.append(wall_rows[-1] + gap + 1)
    assert wall_rows[-1] == height - 1

    for wall_r, axis in zip(wall_rows, centers):
        _draw_wall_row(grid[wall_r], axis)

    sides = _choose_sides(corridor_count)
    for i in range(corridor_count):
        if not _place_corridor_pattern(grid, wall_rows[i] + 1, wall_rows[i + 1] - 1, centers[i], sides[i]):
            return None
    return grid


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    out = grid.copy()
    sides: list[str] = []
    width = grid.shape[1]
    for r0, r1, axis in _corridor_specs(grid):
        coords = np.argwhere(grid[r0 : r1 + 1] == RED)
        cols = [int(c) for _rr, c in coords]
        if cols:
            sides.append("left" if max(cols) < axis else "right")
        for local_r, c in coords:
            rr = r0 + int(local_r)
            cc = int(c)
            mirror_c = 2 * axis - cc
            if 0 <= mirror_c < width:
                out[rr, mirror_c] = RED
    return out, sides


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    h, w = grid.shape
    assert h <= 30 and w <= 30

    walls, axes = _find_walls_and_axes(grid)
    assert len(walls) in (3, 4)
    assert walls[0] == 0 and walls[-1] == h - 1
    assert all(b - a >= 4 for a, b in zip(walls, walls[1:]))

    corridor_specs = _corridor_specs(grid)
    assert len(corridor_specs) in (2, 3)
    assert len(set(axis for _r0, _r1, axis in corridor_specs)) == len(corridor_specs)

    sides = []
    for r0, r1, axis in corridor_specs:
        assert 3 <= (r1 - r0 + 1) <= 5
        band = grid[r0 : r1 + 1]
        red_coords = [(r0 + int(r), int(c)) for r, c in np.argwhere(band == RED)]
        assert PATTERN_SIZE_RANGE[0] <= len(red_coords) <= PATTERN_SIZE_RANGE[1]
        assert _is_4_connected(red_coords)
        cols = [c for _r, c in red_coords]
        all_left = all(c < axis for c in cols)
        all_right = all(c > axis for c in cols)
        assert all_left ^ all_right
        side = "left" if all_left else "right"
        sides.append(side)
        for rr, cc in red_coords:
            mirror_c = 2 * axis - cc
            assert 0 <= mirror_c < w
            assert r0 <= rr <= r1
        assert not np.any(grid[r0 : r1 + 1, axis] == RED)

    assert "left" in sides and "right" in sides


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is None:
            continue
        try:
            _validate_input(grid)
            expected, _sides = _expected_output(grid)
            actual = main(grid)
            if np.array_equal(actual, expected):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid mirrored-corridor puzzle")
