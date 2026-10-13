from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GRAY = 5

MIN_GRID_HEIGHT = 9
MAX_GRID_HEIGHT = 14
MIN_GRID_WIDTH = 10
MAX_GRID_WIDTH = 16
NUM_OBJECTS = 3
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _connected_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(cells)

    return components


def _normalize_shape(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _crop_component(cells: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    tile = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        tile[r - r0, c - c0] = BLUE
    return tile


def _diagonal_offset_copy(tile: np.ndarray) -> np.ndarray:
    height, width = tile.shape
    out = np.zeros((2 * height, 2 * width), dtype=int)
    for r, c in np.argwhere(tile == BLUE):
        max_steps = min(2 * height - int(r), 2 * width - int(c))
        for step in range(max_steps):
            out[int(r) + step, int(c) + step] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the blue object touched by the gray marker, crop it, and repeat it along down-right diagonals on a doubled canvas.

    Input structure:
    - A black background contains several small blue 4-connected objects.
    - Exactly one gray singleton marker is present.
    - The gray marker has 4-neighbor contact with exactly one blue object and no contact with the others.

    Rule:
    - Find the unique gray marker and the blue connected component that is 4-neighbor adjacent to it.
    - Crop that selected blue object to its minimal bounding box, keeping interior black cells if any.
    - Create a black output canvas of size 2H x 2W for the cropped tile and copy every blue tile pixel along its down-right diagonal until the canvas edge.

    Key insight:
    - The marker only chooses which object to keep; the final pattern comes from diagonal self-copying of the selected cropped shape, so the object's geometry controls the wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gray_positions = np.argwhere(g == GRAY)
    assert len(gray_positions) == 1, "expected exactly one gray marker"
    gray_r, gray_c = map(int, gray_positions[0])

    components = _connected_components(g, BLUE)
    assert components, "expected at least one blue object"

    selected: list[tuple[int, int]] | None = None
    for cells in components:
        cell_set = set(cells)
        contact_count = 0
        for nr, nc in _neighbors4(gray_r, gray_c):
            if (nr, nc) in cell_set:
                contact_count += 1
        if contact_count > 0:
            assert selected is None, "gray marker should select exactly one object"
            assert contact_count == 1, "gray marker should touch the selected object at exactly one side"
            selected = cells

    assert selected is not None, "no blue object touched the gray marker"
    tile = _crop_component(selected)
    return _diagonal_offset_copy(tile)


def _random_connected_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target_size = random.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        shape = _normalize_shape(cells)
        height, width = _shape_dims(shape)
        if height < 2 or width < 2:
            continue
        if len(shape) == height * width:
            continue
        return shape
    raise RuntimeError("failed to generate an irregular connected shape")


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height, width = _shape_dims(shape)
    transformed = list(shape)
    if random.random() < 0.5:
        transformed = [(r, width - 1 - c) for r, c in transformed]
    rotation = random.randrange(4)
    for _ in range(rotation):
        transformed = [(c, height - 1 - r) for r, c in transformed]
        height, width = width, height
    return _normalize_shape(transformed)


def _can_place(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    cells = {(top + r, left + c) for r, c in shape}
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                if (nr, nc) in cells:
                    continue
                if int(grid[nr, nc]) != BLACK:
                    return False
    return True


def _stamp_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int):
    for r, c in shape:
        grid[top + r, left + c] = BLUE


def _marker_candidates(shape_cells: set[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    candidates = set()
    for r, c in shape_cells:
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) != BLACK:
                continue
            contact_count = sum((ar, ac) in shape_cells for ar, ac in _neighbors4(nr, nc))
            if contact_count == 1:
                candidates.add((nr, nc))
    return sorted(candidates)


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(MIN_GRID_HEIGHT, MAX_GRID_HEIGHT)
    width = random.randint(MIN_GRID_WIDTH, MAX_GRID_WIDTH)
    grid = np.zeros((height, width), dtype=int)

    shapes: list[tuple[tuple[int, int], ...]] = []
    shape_keys: set[tuple[tuple[int, int], ...]] = set()
    while len(shapes) < NUM_OBJECTS:
        shape = _transform_shape(_random_connected_shape())
        if shape in shape_keys:
            continue
        shapes.append(shape)
        shape_keys.add(shape)

    placement_order = list(range(NUM_OBJECTS))
    random.shuffle(placement_order)
    placed_cells: dict[int, set[tuple[int, int]]] = {}

    for index in placement_order:
        shape = shapes[index]
        shape_h, shape_w = _shape_dims(shape)
        if height - shape_h - 1 < 1 or width - shape_w - 1 < 1:
            return None

        candidates = [
            (top, left)
            for top in range(1, height - shape_h)
            for left in range(1, width - shape_w)
            if _can_place(grid, shape, top, left)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _stamp_shape(grid, shape, top, left)
        placed_cells[index] = {(top + r, left + c) for r, c in shape}

    selected_index = random.randrange(NUM_OBJECTS)
    marker_options = _marker_candidates(placed_cells[selected_index], grid)
    if not marker_options:
        return None
    marker_r, marker_c = random.choice(marker_options)
    grid[marker_r, marker_c] = GRAY

    blue_components = _connected_components(grid, BLUE)
    if len(blue_components) != NUM_OBJECTS:
        return None

    gray_positions = np.argwhere(grid == GRAY)
    if len(gray_positions) != 1:
        return None
    gray_r, gray_c = map(int, gray_positions[0])
    touched = 0
    for cells in blue_components:
        contact_count = sum((nr, nc) in set(cells) for nr, nc in _neighbors4(gray_r, gray_c))
        if contact_count > 0:
            if contact_count != 1:
                return None
            touched += 1
    if touched != 1:
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
