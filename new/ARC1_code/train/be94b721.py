from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the tight crop of the uniquely largest foreground object.

    Input structure:
    - The background is the most common color.
    - The scene contains several separated monochrome foreground objects.
    - Exactly one 4-connected monochrome foreground object has the largest pixel area.

    Rule:
    - Find all 4-connected monochrome foreground components.
    - Select the unique largest component by area.
    - Output that component in its own bounding box, with background in all non-object cells.

    Key insight:
    - The desired crop is identified by connected foreground area, while the smaller objects are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected at least one foreground component"

    max_area = max(int(component["area"]) for component in components)
    largest = [component for component in components if int(component["area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest foreground component"

    component = largest[0]
    color = int(component["color"])
    assert color != bg, "selected component must be foreground"

    coords = component["coords"]
    assert coords.ndim == 2 and coords.shape[1] == 2, "component coords must be row/col pairs"
    r0, c0, r1, c1 = map(int, component["bbox"])

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and object counts.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (5, 9), (9, 13)
OBJECT_COUNT_RANGE, FOREGROUND_COLORS, BACKGROUND_COLOR = (3, 4), range(1, 10), 0

# Blob size, density, and area bounds.
LARGEST_AREA_RANGE, DISTRACTOR_AREA_RANGE = (6, 10), (3, 6)
LARGEST_HEIGHT_RANGE, DISTRACTOR_HEIGHT_RANGE = (3, 4), (2, 4)
BLOB_WIDTH_RANGE, MIN_FILL_NUMERATOR, MIN_FILL_DENOMINATOR = (2, 3), 14, 25

# Placement and retry behavior.
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY, MAX_GENERATION_ATTEMPTS = 1, 8, 300
FOUR_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_be94b721_min_fill_area(height: int, width: int) -> int:
    return (MIN_FILL_NUMERATOR * int(height) * int(width) + MIN_FILL_DENOMINATOR - 1) // MIN_FILL_DENOMINATOR


def _gen_be94b721_dim_choices(area: int, *, largest: bool) -> list[tuple[int, int]]:
    row_min, row_max = LARGEST_HEIGHT_RANGE if largest else DISTRACTOR_HEIGHT_RANGE
    choices: list[tuple[int, int]] = []
    for height in range(row_min, row_max + 1):
        for width in range(BLOB_WIDTH_RANGE[0], BLOB_WIDTH_RANGE[1] + 1):
            if area > height * width:
                continue
            if area < _gen_be94b721_min_fill_area(height, width):
                continue
            if area < height + width - 1:
                continue
            choices.append((height, width))
    return choices


def _gen_be94b721_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.choice([False, True]):
        r, c = 0, random.choice([0, width - 1])
        end_r, end_c = height - 1, width - 1 - c
    else:
        r, c = random.choice([0, height - 1]), 0
        end_r, end_c = height - 1 - r, width - 1

    mask[r, c] = True
    while (r, c) != (end_r, end_c):
        moves = []
        if r != end_r:
            moves.append((r + (1 if end_r > r else -1), c))
        if c != end_c:
            moves.append((r, c + (1 if end_c > c else -1)))
        r, c = random.choice(moves)
        mask[r, c] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in FOUR_NEIGHBOR_OFFSETS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return mask
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True
    return mask


def _gen_be94b721_blob(color: int, area: int, *, largest: bool) -> np.ndarray:
    height, width = random.choice(_gen_be94b721_dim_choices(area, largest=largest))
    mask = _gen_be94b721_connected_mask(height, width, area)
    patch = np.zeros((height, width), dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_be94b721_components(colors: list[int]) -> tuple[list[tuple[np.ndarray, int]], np.ndarray]:
    """Build patches in sampling order; the first color is the expected answer."""
    largest_area = random.randint(*LARGEST_AREA_RANGE)
    largest_patch = _gen_be94b721_blob(colors[0], largest_area, largest=True)
    components: list[tuple[np.ndarray, int]] = [(largest_patch, largest_area)]

    max_distractor_area = min(DISTRACTOR_AREA_RANGE[1], largest_area - 1)
    for color in colors[1:]:
        area = random.randint(DISTRACTOR_AREA_RANGE[0], max_distractor_area)
        components.append((_gen_be94b721_blob(color, area, largest=False), area))
    return components, largest_patch


def _gen_be94b721_place_components(canvas: np.ndarray, components: list[tuple[np.ndarray, int]]) -> bool:
    """Place larger patches first; shuffle randomizes same-area tie order."""
    order = list(range(len(components)))
    random.shuffle(order)
    order.sort(key=lambda i: components[i][1], reverse=True)
    for index in order:
        patch, _area = components[index]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=BACKGROUND_COLOR,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return False
        paste(canvas, patch, r0, c0, transparent=BACKGROUND_COLOR)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(FOREGROUND_COLORS, n_objects)
    components, expected = _gen_be94b721_components(colors)

    canvas = np.zeros((height, width), dtype=int)
    if not _gen_be94b721_place_components(canvas, components):
        return None
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid be94b721 input")
