from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
CORNERS = ("tl", "tr", "bl", "br")
MAIN_DIAGONAL_CORNERS = {"tl", "br"}

# Generator bounds.
SIZE_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (2, 4)
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_SHAPE_SIDE = 5
MAX_SHAPE_AREA = 8
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))

    return components


def _bbox_center(coords: np.ndarray) -> tuple[float, float]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (float(rows.min() + rows.max()) / 2.0, float(cols.min() + cols.max()) / 2.0)


def _nearest_corner(coords: np.ndarray, size: int) -> str:
    center_r, center_c = _bbox_center(coords)
    distances = [
        ((center_r - 0.0) ** 2 + (center_c - 0.0) ** 2, "tl"),
        ((center_r - 0.0) ** 2 + (center_c - (size - 1)) ** 2, "tr"),
        ((center_r - (size - 1)) ** 2 + (center_c - 0.0) ** 2, "bl"),
        ((center_r - (size - 1)) ** 2 + (center_c - (size - 1)) ** 2, "br"),
    ]
    distances.sort(key=lambda item: item[0])
    assert distances[0][0] < distances[1][0], "expected a unique nearest corner"
    return distances[0][1]


def _reflect_coords(coords: np.ndarray, size: int, corner: str) -> np.ndarray:
    if corner in MAIN_DIAGONAL_CORNERS:
        return np.column_stack((coords[:, 1], coords[:, 0]))
    return np.column_stack((size - 1 - coords[:, 1], size - 1 - coords[:, 0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each foreground object across the diagonal selected by its nearest corner.

    Left-top and right-bottom objects use the main diagonal; right-top and
    left-bottom objects use the anti-diagonal. The original objects are erased
    and only the reflected copies remain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height == width, "diagonal reflection is defined on square grids"
    size = height

    components = _connected_components(g != BACKGROUND)
    out = np.full_like(g, BACKGROUND)
    occupied = np.zeros_like(g, dtype=bool)

    for coords in components:
        corner = _nearest_corner(coords, size)
        target = _reflect_coords(coords, size, corner)
        rows = target[:, 0]
        cols = target[:, 1]
        assert np.all((0 <= rows) & (rows < size) & (0 <= cols) & (cols < size))
        assert not np.any(occupied[rows, cols]), "reflected objects must not overlap"
        out[rows, cols] = g[coords[:, 0], coords[:, 1]]
        occupied[rows, cols] = True

    return out


def _zone_origin(size: int, corner: str) -> tuple[int, int, int]:
    half = size // 2
    top = 0 if corner in {"tl", "tr"} else size - half
    left = 0 if corner in {"tl", "bl"} else size - half
    return top, left, half


def _random_shape(zone_size: int) -> np.ndarray | None:
    max_side = min(MAX_SHAPE_SIDE, zone_size)
    if max_side <= 0:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(1, max_side)
        width = random.randint(1, max_side)
        max_area = min(MAX_SHAPE_AREA, height * width)
        if max_area < 2:
            continue
        area = random.randint(2, max_area)

        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], height, width))
        while len(cells) < area:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], height, width):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        mask = mask[rows][:, cols]
        return mask

    return None


def _place_shape(mask: np.ndarray, size: int, corner: str) -> np.ndarray | None:
    top, left, zone = _zone_origin(size, corner)
    height, width = mask.shape
    if height > zone or width > zone:
        return None

    offsets = [
        (dr, dc)
        for dr in range(zone - height + 1)
        for dc in range(zone - width + 1)
    ]
    random.shuffle(offsets)

    local = np.argwhere(mask)
    for dr, dc in offsets[:PLACEMENT_ATTEMPTS]:
        coords = local + np.array([top + dr, left + dc], dtype=int)
        if _nearest_corner(coords, size) != corner:
            continue
        reflected = _reflect_coords(coords, size, corner)
        source_cells = {tuple(cell) for cell in coords}
        target_cells = {tuple(cell) for cell in reflected}
        if source_cells & target_cells:
            continue
        if source_cells == target_cells:
            continue
        return coords

    return None


def _generate_object(size: int, corner: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        mask = _random_shape(size // 2)
        if mask is None:
            continue
        coords = _place_shape(mask, size, corner)
        if coords is None:
            continue
        reflected = _reflect_coords(coords, size, corner)
        return coords, reflected
    return None


def _touches_any(coords: np.ndarray, occupied: np.ndarray) -> bool:
    height, width = occupied.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                if occupied[rr, cc]:
                    return True
    return False


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*SIZE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    corners = random.sample(CORNERS, count)
    colors = random.sample(PALETTE, count)

    grid = np.zeros((size, size), dtype=int)
    input_occupied = np.zeros((size, size), dtype=bool)
    output_occupied = np.zeros((size, size), dtype=bool)

    for corner, color in zip(corners, colors):
        generated = _generate_object(size, corner)
        if generated is None:
            return None
        coords, reflected = generated

        if _touches_any(coords, input_occupied):
            return None
        if _touches_any(reflected, output_occupied):
            return None

        grid[coords[:, 0], coords[:, 1]] = color
        input_occupied[coords[:, 0], coords[:, 1]] = True
        output_occupied[reflected[:, 0], reflected[:, 1]] = True

    if len(_connected_components(grid != BACKGROUND)) != count:
        return None

    out = main(grid)
    if np.any((grid != BACKGROUND) & (out != BACKGROUND)):
        return None
    if len(_connected_components(out != BACKGROUND)) != count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
