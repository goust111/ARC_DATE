from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw foreground objects as nested square rings ordered by object size.

    Input structure:
    - The grid has one dominant background color.
    - Every foreground object is a 4-connected monochromatic component.
    - Foreground object areas are distinct.

    Rule:
    - Sort the foreground objects by decreasing pixel count.
    - Create a square output with side length 2 * number_of_objects - 1.
    - The largest object color fills the outer ring, each next object color fills
      the next inner ring, and the smallest object color fills the center pixel.

    Key insight: The objects' original shapes do not get copied; their areas
    determine only the outside-to-inside order of the nested square pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert objects, "expected at least one foreground object"

    areas = [int(obj["area"]) for obj in objects]
    assert len(areas) == len(set(areas)), "foreground object areas must be unique"

    ordered = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    colors = [int(obj["color"]) for obj in ordered]
    side = 2 * len(colors) - 1

    out = np.full((side, side), colors[0], dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out

# --------------- generator (att2, o5x) ---------------

# Color domain.
BG, COLOR_COUNT = 7, 10
PALETTE = [c for c in range(COLOR_COUNT) if c != BG]

# Canvas and object bounds.
OBJECT_COUNT_RANGE = (2, 6)
OBJECT_AREAS = range(1, 10)
CANVAS_SIDE_RANGE, CANVAS_AREA_FIT_FACTOR = (6, 10), 2

# Blob growth, orientation, placement, and retries.
ORTHOGONAL_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ROTATION_STEPS, FLIP_CHOICES = 4, [False, True]
PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 1, 4
GENERATION_RETRIES = 300


def _ring_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _connected_blob(area: int, color: int, max_dim: int) -> np.ndarray:
    area = int(area)
    limit = min(int(max_dim), area)
    feasible = [(h, w) for h in range(1, limit + 1) for w in range(1, limit + 1) if h * w >= area]
    h, w = random.choice(feasible)

    cells = {(random.randrange(h), random.randrange(w))}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ORTHOGONAL_NEIGHBORS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*next(iter(cells)))
    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        add_frontier(r, c)

    coords = np.array(list(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)

    patch = np.rot90(patch, random.randrange(ROTATION_STEPS))
    if random.choice(FLIP_CHOICES):
        patch = np.flipud(patch)
    if random.choice(FLIP_CHOICES):
        patch = np.fliplr(patch)
    return np.asarray(patch, dtype=int)


def _make_ordered_blobs(areas: list[int], colors: list[int], side: int) -> list[tuple[int, np.ndarray]]:
    blobs = [(area, _connected_blob(area, color, side)) for area, color in zip(areas, colors)]
    random.shuffle(blobs)
    blobs.sort(key=lambda item: item[1].size, reverse=True)
    return blobs


def _place_blobs(canvas: np.ndarray, blobs: list[tuple[int, np.ndarray]]) -> bool:
    for _area, patch in blobs:
        try:
            r, c = random_free_location(canvas, patch, transparent=BG, padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_CONNECTIVITY)
        except ValueError:
            return False
        paste(canvas, patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n)
    areas = random.sample(OBJECT_AREAS, n)

    ordered = sorted(zip(areas, colors), reverse=True)
    expected = _ring_output([color for _area, color in ordered])

    total_area = sum(areas)
    min_side = max(CANVAS_SIDE_RANGE[0], math.ceil(math.sqrt(CANVAS_AREA_FIT_FACTOR * total_area)))
    side = random.randint(min_side, CANVAS_SIDE_RANGE[1])

    canvas = np.full((side, side), BG, dtype=int)
    if not _place_blobs(canvas, _make_ordered_blobs(areas, colors, side)):
        return None

    try:
        actual = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
