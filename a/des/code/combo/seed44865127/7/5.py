from __future__ import annotations

import random
import numpy as np


LEGEND_LENGTHS = (3, 4)
LEGEND_CORNERS = ("tl", "tr", "bl", "br")
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (14, 20)
COLOR_POOL = tuple(range(10))
BASE_SHAPE_COUNT_RANGE = (2, 4)
OBJECT_EXTRA_RANGE = (1, 4)
SHAPE_SIDE_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 7)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_run(row: np.ndarray, bg: int, from_left: bool) -> tuple[int, int, list[int]] | None:
    width = row.shape[0]
    if from_left:
        stop = 0
        while stop < width and int(row[stop]) != bg:
            stop += 1
        if stop == 0 or np.any(row[stop:] != bg):
            return None
        c0, c1 = 0, stop - 1
    else:
        stop = width - 1
        while stop >= 0 and int(row[stop]) != bg:
            stop -= 1
        run = width - 1 - stop
        if run == 0 or np.any(row[: width - run] != bg):
            return None
        c0, c1 = width - run, width - 1

    colors = [int(x) for x in row[c0 : c1 + 1]]
    if len(colors) not in LEGEND_LENGTHS or len(set(colors)) != len(colors):
        return None
    return c0, c1, colors


def _legend_is_isolated(grid: np.ndarray, bg: int, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    height, width = grid.shape
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cell_set:
                if int(grid[rr, cc]) != bg:
                    return False
    return True


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[list[tuple[int, int]], list[int]]] = []
    for row_idx in (0, height - 1):
        for from_left in (True, False):
            run = _corner_run(grid[row_idx], bg, from_left)
            if run is None:
                continue
            c0, c1, colors = run
            cells = [(row_idx, c) for c in range(c0, c1 + 1)]
            if _legend_is_isolated(grid, bg, cells):
                candidates.append((cells, colors))
    assert len(candidates) == 1, "expected exactly one isolated corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Cycle each non-legend object color to the next color defined by the corner legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    legend_cells, cycle = _find_legend(g, bg)
    mapping = {color: cycle[(i + 1) % len(cycle)] for i, color in enumerate(cycle)}

    legend_mask = np.zeros(g.shape, dtype=bool)
    for r, c in legend_cells:
        legend_mask[r, c] = True

    out = g.copy()
    target_mask = (g != bg) & (~legend_mask)
    for src, dst in mapping.items():
        out[(g == src) & target_mask] = dst
    return out


def _shape_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return int(mask.shape[0]), int(mask.shape[1]), tuple(int(x) for x in mask.ravel())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_shape() -> np.ndarray:
    for _ in range(100):
        height = random.randint(*SHAPE_SIDE_RANGE)
        width = random.randint(*SHAPE_SIDE_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], height * width)
        area = random.randint(SHAPE_AREA_RANGE[0], max_area)

        mask = np.zeros((height, width), dtype=bool)
        seed = (random.randrange(height), random.randrange(width))
        mask[seed] = True

        while int(mask.sum()) < area:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                        frontier.append((rr, cc))
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        cropped = _crop_mask(mask)
        if int(cropped.sum()) >= SHAPE_AREA_RANGE[0]:
            return cropped
    raise RuntimeError("failed to sample a connected shape")


def _build_shape_library(count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    while len(shapes) < count:
        shape = _random_connected_shape()
        key = _shape_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
    return shapes


def _legend_start(height: int, width: int, length: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - length
    if corner == "bl":
        return height - 1, 0
    return height - 1, width - length


def _paint_legend(grid: np.ndarray, colors: list[int], corner: str) -> list[tuple[int, int]]:
    row, col0 = _legend_start(grid.shape[0], grid.shape[1], len(colors), corner)
    cells = []
    for offset, color in enumerate(colors):
        c = col0 + offset
        grid[row, c] = color
        cells.append((row, c))
    return cells


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], padding: int) -> None:
    height, width = blocked.shape
    for r, c in cells:
        r0 = max(0, r - padding)
        r1 = min(height, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(width, c + padding + 1)
        blocked[r0:r1, c0:c1] = True


def _free_positions(blocked: np.ndarray, shape: np.ndarray) -> list[tuple[int, int]]:
    height, width = blocked.shape
    sh, sw = shape.shape
    positions = []
    for r in range(1, height - sh):
        for c in range(1, width - sw):
            r0 = max(0, r - PLACEMENT_PADDING)
            r1 = min(height, r + sh + PLACEMENT_PADDING)
            c0 = max(0, c - PLACEMENT_PADDING)
            c1 = min(width, c + sw + PLACEMENT_PADDING)
            if not blocked[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _place_shape(grid: np.ndarray, blocked: np.ndarray, shape: np.ndarray, color: int) -> bool:
    positions = _free_positions(blocked, shape)
    if not positions:
        return False

    r, c = random.choice(positions)
    sh, sw = shape.shape
    patch = grid[r : r + sh, c : c + sw]
    patch[shape] = color

    r0 = max(0, r - PLACEMENT_PADDING)
    r1 = min(grid.shape[0], r + sh + PLACEMENT_PADDING)
    c0 = max(0, c - PLACEMENT_PADDING)
    c1 = min(grid.shape[1], c + sw + PLACEMENT_PADDING)
    blocked[r0:r1, c0:c1] = True
    return True


def _build_object_specs(cycle_colors: list[int], shapes: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    total = len(cycle_colors) + random.randint(*OBJECT_EXTRA_RANGE)
    repeated_shape = random.choice(shapes)
    repeated_colors = random.sample(cycle_colors, 2)

    specs: list[tuple[np.ndarray, int]] = [(repeated_shape, color) for color in repeated_colors]
    used_colors = set(repeated_colors)
    for color in cycle_colors:
        if color not in used_colors:
            specs.append((random.choice(shapes), color))

    while len(specs) < total:
        specs.append((random.choice(shapes), random.choice(cycle_colors)))

    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(COLOR_POOL)

    cycle_len = random.choice(LEGEND_LENGTHS)
    cycle_colors = random.sample([c for c in COLOR_POOL if c != bg], cycle_len)
    corner = random.choice(LEGEND_CORNERS)

    base_shape_count = random.randint(BASE_SHAPE_COUNT_RANGE[0], BASE_SHAPE_COUNT_RANGE[1])
    shapes = _build_shape_library(base_shape_count)
    specs = _build_object_specs(cycle_colors, shapes)

    grid = np.full((height, width), bg, dtype=int)
    legend_cells = _paint_legend(grid, cycle_colors, corner)

    blocked = np.zeros((height, width), dtype=bool)
    _mark_blocked(blocked, legend_cells, padding=1)

    for shape, color in specs:
        if not _place_shape(grid, blocked, shape, color):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
