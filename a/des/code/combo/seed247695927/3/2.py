from __future__ import annotations

import random

import numpy as np


GRID_SIZE_RANGE = (8, 15)
OBJECT_COUNT_RANGE = (2, 6)
BG_COLORS = list(range(10))
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 80
SHAPE_RETRIES = 60


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if seen[r0, c0] or color == bg:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            rmin = int(coords[:, 0].min())
            rmax = int(coords[:, 0].max())
            cmin = int(coords[:, 1].min())
            cmax = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (rmin, cmin, rmax, cmax),
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    out = g.copy()
    for component in _find_components(g, bg):
        cmin = component["bbox"][1]
        erase = component["coords"][component["coords"][:, 1] < cmin + 2]
        out[erase[:, 0], erase[:, 1]] = bg
    return out


def _shape_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, cells[0]))]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        height = random.randint(1, 5)
        width = random.randint(2, 6)
        area = height * width
        target = random.randint(max(2, area // 2), area)

        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        frontier = [start]
        while int(mask.sum()) < target:
            r, c = random.choice(frontier)
            neighbors = [
                (nr, nc)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]
            ]
            if not neighbors:
                frontier = [tuple(x) for x in np.argwhere(mask)]
                continue
            nr, nc = random.choice(neighbors)
            mask[nr, nc] = True
            frontier.append((nr, nc))

        if mask[:, 0].any() and mask[:, -1].any() and _shape_connected(mask):
            return mask
    return None


def _can_place(grid: np.ndarray, top: int, left: int, mask: np.ndarray, bg: int) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    r0 = max(0, top - 1)
    r1 = min(H, top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(W, left + w + 1)
    return np.all(grid[r0:r1, c0:c1] == bg)


def _place_shape(grid: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    spots = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(spots)
    for top, left in spots[:PLACEMENT_RETRIES]:
        if not _can_place(grid, top, left, mask, bg):
            continue
        patch = grid[top : top + h, left : left + w]
        patch[mask] = color
        return True
    return False


def _valid_generated(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    components = _find_components(grid, bg)
    if not components:
        return False
    for component in components:
        cmin = component["bbox"][1]
        cmax = component["bbox"][3]
        if cmax - cmin + 1 < 2:
            return False
    if np.sum(grid == bg) * 2 < grid.size:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    bg = random.choice(BG_COLORS)
    grid = np.full((height, width), bg, dtype=int)

    color_choices = [c for c in range(10) if c != bg]
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    colors = [random.choice(color_choices) for _ in range(object_count)]
    random.shuffle(colors)

    for color in colors:
        mask = _random_shape()
        if mask is None or not _place_shape(grid, mask, color, bg):
            return None

    return grid if _valid_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
