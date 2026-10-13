from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))


def _legend_entries(grid: np.ndarray) -> list[tuple[int, int]]:
    rows = np.flatnonzero(grid[:, 0] != BG)
    assert rows.size >= 2, "expected at least two legend colors in the leftmost column"
    entries = [(int(r), int(grid[int(r), 0])) for r in rows]
    colors = [color for _row, color in entries]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange each legend color's scattered object pixels into a bottom-aligned vertical bar.

    Input structure:
    - The background is black.
    - The leftmost column contains a vertical legend of distinct non-black colors, read from top to bottom.
    - Outside the legend column, monochrome 4-connected objects use only legend colors and may appear multiple times per color.

    Rule:
    - Create a same-sized black output canvas and copy the legend column unchanged.
    - For each legend color at rank K, count all pixels of that color outside the legend column.
    - Draw those pixels as one bottom-aligned vertical bar in output column 1+K, leaving all other cells black.
    - The legend pixels themselves are not part of the counted objects.

    Key insight:
    - Object shape and location are discarded entirely; only each legend color's total non-legend pixel count remains, and legend order fixes the bar columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    legend = _legend_entries(g)
    legend_colors = [color for _row, color in legend]
    assert w >= len(legend_colors) + 1, "grid must have room to the right of the legend"

    body = g[:, 1:]
    body_colors = set(map(int, np.unique(body))) - {BG}
    assert body_colors <= set(legend_colors), "all non-legend colors must come from the legend"

    out = np.full((h, w), BG, dtype=int)
    out[:, 0] = g[:, 0]

    for rank, color in enumerate(legend_colors):
        count = int(np.count_nonzero(body == color))
        assert count <= h, "bar height must fit inside the canvas"
        if count:
            out[h - count :, rank + 1] = color

    return out


# --------------- generator (o5x) ---------------

MIN_HEIGHT = 12
MAX_HEIGHT = 22
MIN_LEGEND_COLORS = 3
MAX_LEGEND_COLORS = 6
MIN_OBJECT_ZONE_WIDTH = 7
MAX_WIDTH = 30

MAX_OBJECT_AREA = 5
MAX_OBJECTS_PER_COLOR = 3
MAX_TOTAL_PIXELS_PER_COLOR = 10

PLACEMENT_TRIES = 160
PLAN_TRIES = 80
GENERATION_TRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_areas(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    areas: list[int] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            area = 0
            while q:
                rr, cc = q.popleft()
                area += 1
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            areas.append(area)

    return areas


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}

    while len(cells) < area:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        for nbr in _neighbors4(*pick):
            if nbr not in cells:
                frontier.add(nbr)

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    return np.asarray(mask, dtype=bool)


def _sample_plan(height: int, legend_count: int) -> list[list[int]] | None:
    if legend_count > MIN_LEGEND_COLORS and random.random() < 0.5:
        active_count = random.randint(2, legend_count - 1)
    else:
        active_count = random.randint(2, legend_count)
    active_indices = set(random.sample(range(legend_count), active_count))
    split_index = random.choice(tuple(active_indices))

    plans: list[list[int]] = []
    totals: list[int] = []
    for idx in range(legend_count):
        if idx not in active_indices:
            plans.append([])
            totals.append(0)
            continue

        if idx == split_index:
            object_count = random.randint(2, MAX_OBJECTS_PER_COLOR)
        else:
            object_count = 1 + int(random.random() < 0.35)

        for _ in range(PLAN_TRIES):
            areas = [random.randint(1, MAX_OBJECT_AREA) for _ in range(object_count)]
            total = sum(areas)
            if total <= min(height, MAX_TOTAL_PIXELS_PER_COLOR):
                plans.append(areas)
                totals.append(total)
                break
        else:
            return None

    positive_totals = [total for total in totals if total > 0]
    if len(positive_totals) < 2 or len(set(positive_totals)) == 1:
        return None
    if all(all(area == 1 for area in areas) for areas in plans if areas):
        return None
    return plans


def _place_objects(grid: np.ndarray, objects: list[tuple[int, np.ndarray]], start_col: int) -> bool:
    h, w = grid.shape
    occupied = grid != BG
    blocked = _dilate8(occupied)

    random.shuffle(objects)
    for color, mask in objects:
        oh, ow = map(int, mask.shape)
        if start_col + ow > w:
            return False

        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(0, h - oh)
            c0 = random.randint(start_col, w - ow)
            if np.any(blocked[r0 : r0 + oh, c0 : c0 + ow] & mask):
                continue

            patch = grid[r0 : r0 + oh, c0 : c0 + ow]
            patch[mask] = color
            occupied[r0 : r0 + oh, c0 : c0 + ow] |= mask
            blocked = _dilate8(occupied)
            break
        else:
            return False

    return True


def _expected_output(
    shape: tuple[int, int],
    legend_rows: list[int],
    legend_colors: list[int],
    plans: list[list[int]],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    for row, color in zip(legend_rows, legend_colors):
        out[int(row), 0] = int(color)
    for rank, (color, areas) in enumerate(zip(legend_colors, plans)):
        total = sum(areas)
        if total:
            out[h - total :, rank + 1] = int(color)
    return out


def _plan_matches_grid(grid: np.ndarray, legend_colors: list[int], plans: list[list[int]]) -> bool:
    body = grid[:, 1:]
    for color, areas in zip(legend_colors, plans):
        expected_total = sum(areas)
        if int(np.count_nonzero(body == color)) != expected_total:
            return False
        if sorted(_component_areas(body == color)) != sorted(areas):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(MIN_LEGEND_COLORS, MAX_LEGEND_COLORS)
    height = random.randint(max(MIN_HEIGHT, legend_count + 1), MAX_HEIGHT)
    object_start_col = legend_count + 2
    min_width = object_start_col + MIN_OBJECT_ZONE_WIDTH
    if min_width > MAX_WIDTH:
        return None
    width = random.randint(min_width, MAX_WIDTH)

    legend_colors = random.sample(PALETTE, legend_count)
    legend_r0 = random.randint(0, height - legend_count)
    legend_rows = [legend_r0 + i for i in range(legend_count)]

    plans = _sample_plan(height, legend_count)
    if plans is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for row, color in zip(legend_rows, legend_colors):
        grid[int(row), 0] = int(color)

    objects: list[tuple[int, np.ndarray]] = []
    for color, areas in zip(legend_colors, plans):
        for area in areas:
            objects.append((int(color), _random_polyomino(int(area))))

    if not _place_objects(grid, objects, object_start_col):
        return None
    if not _plan_matches_grid(grid, legend_colors, plans):
        return None

    expected = _expected_output(grid.shape, legend_rows, legend_colors, plans)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
