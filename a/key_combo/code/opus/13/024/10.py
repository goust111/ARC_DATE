from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location, tile_fill


TEMPLATE_LIBRARY: dict[int, tuple[np.ndarray, ...]] = {
    3: (
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    ),
    4: (
        np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=bool),
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]], dtype=bool),
    ),
}

FOREGROUND_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY)
MARKER_CORNERS = ("tl", "tr", "bl", "br")
GRID_SIDE_RANGE = (15, 19)
OBJECT_COUNT_RANGE = (3, 5)
GEN_RETRY_LIMIT = 400


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _random_connected_patch(area: int, color: int) -> np.ndarray:
    options = [(h, w) for h in range(1, 5) for w in range(1, 5) if area <= h * w]
    height, width = random.choice(options)
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.append((rr, cc))
        frontier = list(dict.fromkeys(frontier))
        mask[random.choice(frontier)] = True

    r0, c0, r1, c1 = _mask_bbox(mask)
    return _render_patch(mask[r0 : r1 + 1, c0 : c1 + 1], color)


def _corner_coord(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    rows, cols = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, cols - 1
    if corner == "bl":
        return rows - 1, 0
    return rows - 1, cols - 1


def _expected_output(template_mask: np.ndarray) -> np.ndarray:
    k = int(template_mask.shape[0])
    return tile_fill((4 * k, 4 * k), np.where(template_mask, Color.MAROON, Color.BLACK).astype(int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique largest foreground component and use its silhouette as a periodic maroon tile.

    Input structure:
    - The background is black, with 3 to 5 separated monochrome 4-connected foreground objects in distinct colors.
    - Exactly one foreground component has the largest area, and its tight bounding box is a 3x3 or 4x4 square.
    - A single maroon pixel marks the output paint color and does not belong to the selected largest object.

    Rule:
    - Find all monochrome 4-connected foreground components and select the unique largest one by pixel area.
    - Crop that component to its bounding box and convert it to a binary kxk template, where object cells are 1 and background cells are 0.
    - Read the maroon marker, create a black 12x12 canvas when k=3 or a black 16x16 canvas when k=4, and tile the binary template periodically in both directions.
    - Paint the template's 1-cells with maroon and leave the 0-cells black.

    Key insight:
    - The largest object is not copied into the answer; its cropped silhouette becomes the phase pattern that generates the whole output lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    marker_coords = np.argwhere(g == Color.MAROON)
    assert marker_coords.shape[0] == 1, "expected exactly one maroon marker pixel"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected at least one foreground component"

    max_area = max(int(component["area"]) for component in components)
    largest = [component for component in components if int(component["area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest foreground component"

    component = largest[0]
    coords = component["coords"]
    mask = np.zeros_like(g, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    template = crop_bbox(mask.astype(int), component["bbox"]).astype(bool)

    k_rows, k_cols = template.shape
    assert k_rows == k_cols and k_rows in (3, 4), "largest component crop must be 3x3 or 4x4"
    return _expected_output(template)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(
        (
            random.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]),
            random.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]),
        ),
        dtype=int,
    )

    corner = random.choice(MARKER_CORNERS)
    mr, mc = _corner_coord(grid.shape, corner)
    grid[mr, mc] = Color.MAROON

    k = random.choice((3, 4))
    template_mask = random.choice(TEMPLATE_LIBRARY[k]).copy()
    largest_color = random.choice(FOREGROUND_COLORS)
    largest_patch = _render_patch(template_mask, largest_color)
    largest_area = int(template_mask.sum())

    n_objects = random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])
    distractor_areas = random.sample(list(range(1, largest_area)), n_objects - 1)
    distractor_colors = random.sample([c for c in FOREGROUND_COLORS if c != largest_color], n_objects - 1)
    distractor_patches = [
        _random_connected_patch(area=area, color=color) for area, color in zip(distractor_areas, distractor_colors)
    ]

    patches = [(largest_patch, largest_area)] + [
        (patch, int(np.count_nonzero(patch))) for patch in distractor_patches
    ]
    random.shuffle(patches)
    patches.sort(key=lambda item: item[1], reverse=True)

    for patch, _area in patches:
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=Color.BLACK)

    expected = _expected_output(template_mask)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
