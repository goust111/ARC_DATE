from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=int)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, k=random.randrange(4))
    coords = np.argwhere(out)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict[str, object]] = []
    for coords in _components_mask(g != 0):
        r0, c0, r1, c1 = _bbox_coords(coords)
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each foreground region must be monochrome"
        regions.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": int(next(iter(colors))),
                "area": int(coords.shape[0]),
            }
        )
    assert regions, "expected at least one non-black region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each scattered connected region into a full-width horizontal stripe whose height equals the region area.

    Input structure:
    - A black background contains several separated non-black 4-connected regions.
    - Each region is monochrome and uses only a few cells, while regions are vertically separated so their top-to-bottom order is unambiguous.
    - Only each region's color, area, and vertical position matter; shape and horizontal placement are incidental.

    Rule:
    - Find every non-black 4-connected region and measure its pixel area.
    - Sort the regions by the top row of their bounding box, from highest to lowest in the input.
    - Create an output whose width matches the input width and whose height is the sum of all region areas.
    - Fill the output from top to bottom with solid full-width stripes, one per region in sorted order, using that region's color for exactly as many rows as its area.

    Key insight:
    - The input regions act as area-coded tokens: vertical position determines stripe order, and region size alone determines stripe thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = sorted(_extract_regions(g), key=lambda region: (int(region["bbox"][0]), int(region["bbox"][1])))
    total_height = sum(int(region["area"]) for region in regions)
    out = np.zeros((total_height, g.shape[1]), dtype=int)

    row = 0
    for region in regions:
        height = int(region["area"])
        out[row : row + height, :] = int(region["color"])
        row += height
    return out


_GRID_RANGE = (10, 14)
_COUNT_CHOICES = (3, 4)
_AREAS = (1, 2, 3, 4)
_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_MAX_ATTEMPTS = 400

_BASE_SHAPES = {
    1: [np.array([[1]], dtype=int)],
    2: [np.array([[1, 1]], dtype=int)],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    ],
}


def _scatter_total(total: int, parts: int) -> list[int]:
    slots = [0] * parts
    for _ in range(total):
        slots[random.randrange(parts)] += 1
    return slots


def _expected_output(objects: list[dict[str, object]], width: int) -> np.ndarray:
    height = sum(int(obj["area"]) for obj in objects)
    out = np.zeros((height, width), dtype=int)
    row = 0
    for obj in objects:
        area = int(obj["area"])
        out[row : row + area, :] = int(obj["color"])
        row += area
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(_COUNT_CHOICES)
    areas = random.sample(_AREAS, k=count)
    colors = random.sample(_PALETTE, k=count)

    objects: list[dict[str, object]] = []
    for area, color in zip(areas, colors):
        mask = _random_transform(random.choice(_BASE_SHAPES[area]))
        objects.append(
            {
                "area": int(area),
                "color": int(color),
                "mask": mask,
                "shape": mask.shape,
            }
        )

    random.shuffle(objects)
    heights = [int(obj["shape"][0]) for obj in objects]
    min_height = sum(heights) + (count - 1)
    if min_height > _GRID_RANGE[1]:
        return None

    grid_h = random.randint(max(_GRID_RANGE[0], min_height), _GRID_RANGE[1])
    max_width_needed = max(int(obj["shape"][1]) for obj in objects)
    grid_w = random.randint(max(_GRID_RANGE[0], max_width_needed), _GRID_RANGE[1])

    slack = grid_h - min_height
    gaps = _scatter_total(slack, count + 1)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    row = gaps[0]
    placed: list[dict[str, object]] = []
    for idx, obj in enumerate(objects):
        mask = np.asarray(obj["mask"], dtype=int)
        h, w = map(int, mask.shape)
        col = random.randint(0, grid_w - w)
        color = int(obj["color"])
        grid[row : row + h, col : col + w][mask == 1] = color
        placed.append(
            {
                "area": int(obj["area"]),
                "color": color,
                "bbox": (row, col, row + h - 1, col + w - 1),
            }
        )
        row += h
        if idx < count - 1:
            row += 1 + gaps[idx + 1]

    try:
        regions = _extract_regions(grid)
        if len(regions) != count:
            return None
        if len({int(region["area"]) for region in regions}) != count:
            return None
        if any(int(region["area"]) > 4 for region in regions):
            return None
        expected = _expected_output(
            sorted(placed, key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1]))),
            grid_w,
        )
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
