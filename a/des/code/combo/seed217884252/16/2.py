from __future__ import annotations

from collections import deque
import random

import numpy as np


ORANGE = 7
GREY = 5
BLUE = 1
YELLOW = 4
RED = 2
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LAYER_COLORS = {1: YELLOW, 2: RED, 3: MAROON}

HALO_PADDING = 3
GRID_MAX = 28
GRID_MIN_BY_OBJECT_COUNT = {1: 14, 2: 18, 3: 22}
SHAPE_HEIGHT_RANGE = (4, 8)
SHAPE_WIDTH_RANGE = (4, 8)
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 160
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        queue = deque([(r, c)])
        while queue:
            rr, cc = queue.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _exterior_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if grid[r, c] == ORANGE and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == ORANGE and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return exterior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each grey object's outer contour into a colored halo.

    Blue markers inside an object count the number of expansion layers:
    1 -> yellow, 2 -> yellow+red, 3 -> yellow+red+maroon.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    foreground = g != ORANGE
    exterior_bg = _exterior_background(g)

    out = g.copy()
    out[out == BLUE] = GREY
    best_layer = np.full((h, w), HALO_PADDING + 1, dtype=int)

    for comp in _components(foreground):
        layer_count = sum(int(g[r, c] == BLUE) for r, c in comp)
        assert 1 <= layer_count <= 3, "each object must contain 1 to 3 blue markers"

        edge_cells = []
        for r, c in comp:
            if any(
                0 <= nr < h and 0 <= nc < w and exterior_bg[nr, nc]
                for nr, nc in _neighbors4(r, c)
            ):
                edge_cells.append((r, c))

        dist: dict[tuple[int, int], int] = {}
        queue: deque[tuple[int, int]] = deque()
        for r, c in edge_cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and exterior_bg[nr, nc] and (nr, nc) not in dist:
                    dist[(nr, nc)] = 1
                    queue.append((nr, nc))

        while queue:
            r, c = queue.popleft()
            layer = dist[(r, c)]
            if layer < best_layer[r, c]:
                best_layer[r, c] = layer
            if layer == layer_count:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and exterior_bg[nr, nc] and (nr, nc) not in dist:
                    dist[(nr, nc)] = layer + 1
                    queue.append((nr, nc))

    for layer, color in LAYER_COLORS.items():
        out[best_layer == layer] = color
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if all(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
            cells.append((r, c))
    return cells


def _make_row_convex_shape(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    span = random.randint(3, width)
    left = random.randint(0, width - span)
    for r in range(height):
        mask[r, left : left + span] = True
        if r == height - 1:
            break
        next_span = max(3, min(width, span + random.choice([-1, 0, 0, 1])))
        lo = max(0, left - next_span + 1)
        hi = min(width - next_span, left + span - 1)
        left = random.randint(lo, hi)
        span = next_span
    return mask


def _make_notched_rect(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        max_depth = min(3, height - 2)
        depth = random.randint(1, max_depth)
        max_span = width - 2
        span = random.randint(1, max_span)
        start = random.randint(1, width - span - 1)
        rows = slice(0, depth) if side == "top" else slice(height - depth, height)
        mask[rows, start : start + span] = False
    else:
        max_depth = min(3, width - 2)
        depth = random.randint(1, max_depth)
        max_span = height - 2
        span = random.randint(1, max_span)
        start = random.randint(1, height - span - 1)
        cols = slice(0, depth) if side == "left" else slice(width - depth, width)
        mask[start : start + span, cols] = False
    return mask


def _make_shape(min_interior: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*SHAPE_HEIGHT_RANGE)
        width = random.randint(*SHAPE_WIDTH_RANGE)
        if random.random() < 0.5:
            mask = _make_row_convex_shape(height, width)
        else:
            mask = _make_notched_rect(height, width)
        if random.random() < 0.5:
            mask = mask[::-1, :]
        if random.random() < 0.5:
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            mask = mask.T
        mask = _trim_mask(mask)
        interior = _interior_cells(mask)
        if len(interior) >= min_interior and int(mask.sum()) >= min_interior + 6:
            return mask, interior
    return None


def _dilate_mask(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        for r, c in np.argwhere(out):
            r = int(r)
            c = int(c)
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                    grown[nr, nc] = True
        out = grown
    return out


def _reserve_patch(shape: np.ndarray, pad: int) -> np.ndarray:
    h, w = shape.shape
    patch = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    patch[pad : pad + h, pad : pad + w] = shape
    return _dilate_mask(patch, pad)


def _sample_marker_counts(object_count: int) -> list[int]:
    counts = [1, 2, 3]
    random.shuffle(counts)
    return counts[:object_count]


def _try_place_object(
    grid: np.ndarray,
    reserve: np.ndarray,
    marker_count: int,
) -> bool:
    shape_data = _make_shape(marker_count)
    if shape_data is None:
        return False
    shape, interior = shape_data
    h, w = shape.shape
    gh, gw = grid.shape
    if h + 2 * HALO_PADDING > gh or w + 2 * HALO_PADDING > gw:
        return False

    reserve_patch = _reserve_patch(shape, HALO_PADDING)
    ph, pw = reserve_patch.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(HALO_PADDING, gh - h - HALO_PADDING)
        left = random.randint(HALO_PADDING, gw - w - HALO_PADDING)
        patch_top = top - HALO_PADDING
        patch_left = left - HALO_PADDING
        current = reserve[patch_top : patch_top + ph, patch_left : patch_left + pw]
        if np.any(current & reserve_patch):
            continue

        window = grid[top : top + h, left : left + w]
        window[shape] = GREY
        for r, c in random.sample(interior, marker_count):
            window[r, c] = BLUE
        current |= reserve_patch
        return True
    return False


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if not np.all(np.isin(grid, [ORANGE, GREY, BLUE])):
        return False

    comps = _components(grid != ORANGE)
    if not (1 <= len(comps) <= 3):
        return False

    expected_layers: set[int] = set()
    for comp in comps:
        blue_cells = [(r, c) for r, c in comp if grid[r, c] == BLUE]
        if len(blue_cells) not in (1, 2, 3):
            return False
        expected_layers.update(range(1, len(blue_cells) + 1))
        for r, c in blue_cells:
            if any(
                not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] == ORANGE
                for nr, nc in _neighbors4(r, c)
            ):
                return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out == BLUE):
        return False
    if not np.all(out[grid != ORANGE] == GREY):
        return False

    for layer, color in LAYER_COLORS.items():
        present = bool(np.any(out == color))
        if (layer in expected_layers) != present:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choices([1, 2, 3], weights=[1, 2, 3], k=1)[0]
    min_size = GRID_MIN_BY_OBJECT_COUNT[object_count]
    height = random.randint(min_size, GRID_MAX)
    width = random.randint(min_size, GRID_MAX)

    grid = np.full((height, width), ORANGE, dtype=int)
    reserve = np.zeros((height, width), dtype=bool)
    for marker_count in _sample_marker_counts(object_count):
        if not _try_place_object(grid, reserve, marker_count):
            return None

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
