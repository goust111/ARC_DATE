from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 8)

MIN_OBJECTS = 3
MAX_OBJECTS = 5
AREA_CHOICES = (3, 4, 5, 6, 7, 8)
GRID_ROWS_RANGE = (11, 16)
GRID_COLS_RANGE = (13, 18)
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    return coords


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        coords = _normalize_cells(cells)
        mask = _coords_to_mask(coords)
        h, w = mask.shape

        if h == 1 or w == 1:
            continue
        if h * w == area:
            continue
        if len(_components_mask(mask)) != 1:
            continue
        if np.count_nonzero(mask[0]) == 0 or np.count_nonzero(mask[-1]) == 0:
            continue
        if np.count_nonzero(mask[:, 0]) == 0 or np.count_nonzero(mask[:, -1]) == 0:
            continue
        return coords
    return None


def _paint_with_gap(
    grid: np.ndarray,
    forbidden: np.ndarray,
    coords: np.ndarray,
    color: int,
    top_left: tuple[int, int],
) -> None:
    r0, c0 = map(int, top_left)
    placed = coords + np.array([r0, c0], dtype=int)
    grid[placed[:, 0], placed[:, 1]] = int(color)
    for rr, cc in placed:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    forbidden[nr, nc] = True


def _place_objects(
    rows: int,
    cols: int,
    items: list[tuple[int, int, np.ndarray]],
) -> np.ndarray | None:
    grid = np.zeros((rows, cols), dtype=int)
    forbidden = np.zeros((rows, cols), dtype=bool)

    # Place larger shapes first to reduce dead ends.
    for color, _area, coords in sorted(items, key=lambda item: -len(item[2])):
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        locations = [
            (r0, c0)
            for r0 in range(rows - h + 1)
            for c0 in range(cols - w + 1)
            if not np.any(forbidden[(coords[:, 0] + r0), (coords[:, 1] + c0)])
        ]
        if not locations:
            return None
        r0, c0 = random.choice(locations)
        _paint_with_gap(grid, forbidden, coords, color, (r0, c0))

    return grid


def _object_specs(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    specs = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        comps = _components_mask(grid == color)
        assert len(comps) == 1, "each foreground color must form one 4-connected component"
        coords = comps[0]
        top_row = int(coords[:, 0].min())
        left_col = int(coords[:, 1].min())
        specs.append((color, int(len(coords)), top_row, left_col, int(coords[:, 0].max())))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert single-color connected components into an area-sorted mirrored histogram.

    Input structure:
    - The grid has a black background and 3-5 non-black colors.
    - Each foreground color forms exactly one 4-connected component.
    - Component shapes are irregular and their pixel areas are mutually distinct.

    Rule:
    - Measure the pixel area of each foreground component.
    - Sort components by decreasing area; if two areas tie, break the tie by the
      row of the component's topmost pixel.
    - Build a one-pixel-wide bottom-aligned histogram whose column height equals
      each component area and whose column color is the component color.
    - Mirror that histogram left-right and concatenate the mirror to its right.

    Key insight:
    - The output ignores the original shapes and positions; only per-component
      area determines the bar height and column order, and mirroring turns that
      sorted histogram into a symmetric mountain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    specs = _object_specs(g)
    assert MIN_OBJECTS <= len(specs) <= len(PALETTE)

    ordered = sorted(specs, key=lambda item: (-item[1], item[2]))
    max_area = max(area for _color, area, _top, _left, _bottom in ordered)
    hist = np.zeros((max_area, len(ordered)), dtype=int)
    for col, (color, area, _top, _left, _bottom) in enumerate(ordered):
        hist[max_area - area :, col] = int(color)
    return np.concatenate([hist, np.fliplr(hist)], axis=1)


def _expected_output_from_pairs(color_area_pairs: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(color_area_pairs, key=lambda item: -item[1])
    max_area = max(area for _color, area in ordered)
    hist = np.zeros((max_area, len(ordered)), dtype=int)
    for col, (color, area) in enumerate(ordered):
        hist[max_area - area :, col] = int(color)
    return np.concatenate([hist, np.fliplr(hist)], axis=1)


def _sample_items() -> list[tuple[int, int, np.ndarray]] | None:
    n_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    colors = random.sample(PALETTE, n_objects)
    areas = random.sample(AREA_CHOICES, n_objects)
    random.shuffle(colors)
    random.shuffle(areas)

    # Reject accidental agreement with invisible color-id ordering.
    if [color for color, _area in sorted(zip(colors, areas), key=lambda x: -x[1])] == sorted(colors):
        return None

    items = []
    for color, area in zip(colors, areas):
        coords = _random_polyomino(int(area))
        if coords is None:
            return None
        items.append((int(color), int(area), coords))
    return items


def _is_informative(grid: np.ndarray) -> bool:
    specs = _object_specs(grid)
    area_order = [color for color, _area, _top, _left, _bottom in sorted(specs, key=lambda item: -item[1])]
    color_order = sorted(area_order)
    top_order = [color for color, _area, _top, _left, _bottom in sorted(specs, key=lambda item: (item[2], item[3]))]
    left_order = [color for color, _area, _top, _left, _bottom in sorted(specs, key=lambda item: (item[3], item[2]))]

    if area_order == color_order or area_order == color_order[::-1]:
        return False
    if area_order == top_order or area_order == left_order:
        return False

    areas = [area for _color, area, _top, _left, _bottom in specs]
    return len(areas) == len(set(areas))


def _generate_attempt() -> np.ndarray | None:
    items = _sample_items()
    if items is None:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        rows = random.randint(*GRID_ROWS_RANGE)
        cols = random.randint(*GRID_COLS_RANGE)
        grid = _place_objects(rows, cols, items)
        if grid is None:
            continue
        if not _is_informative(grid):
            continue

        pairs = [(color, area) for color, area, _coords in items]
        expected = _expected_output_from_pairs(pairs)
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-area-histogram input")
