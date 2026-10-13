from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location
import itertools
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-sized filled rectangular objects by overlaying their internal marker pixels.

    Input structure:
    - A dominant background surrounds two or more separated 4-connected foreground objects.
    - Every object is a filled rectangle of the same size.
    - Each rectangle has the same majority base color plus one or more non-base marker pixels.

    Rule:
    - Create one output rectangle with the shared object size, filled with the shared base color.
    - For each input rectangle, copy every non-base marker pixel to the same relative location
      in the output.
    - Marker overlays must either land on base color or agree in color with an existing marker.

    Key insight:
    The separated rectangles are layers of the same template; their colored marker pixels
    combine by local coordinates while the repeated base fill is just the output canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    objects: list[tuple[np.ndarray, int]] = []
    for coords in components_mask(g != bg, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = crop_bbox(g, (r0, c0, r1, c1))
        assert np.all(patch != bg)

        counts = Counter(map(int, patch.ravel()))
        ranked = counts.most_common()
        assert len(ranked) >= 2
        assert ranked[0][1] > ranked[1][1]
        objects.append((patch, int(ranked[0][0])))

    assert len(objects) >= 2

    shapes = {patch.shape for patch, _base in objects}
    assert len(shapes) == 1

    bases = {base for _patch, base in objects}
    assert len(bases) == 1
    base = next(iter(bases))

    h, w = next(iter(shapes))
    out = np.full((h, w), int(base), dtype=int)

    has_marker = False
    for patch, _base in objects:
        marker_mask = patch != int(base)
        has_marker = has_marker or bool(marker_mask.any())
        rows, cols = np.where(marker_mask)
        for r, c in zip(rows, cols):
            value = int(patch[int(r), int(c)])
            assert int(out[int(r), int(c)]) in {int(base), value}
            out[int(r), int(c)] = value

    assert has_marker
    return out

# --------------- generator (o5x) ---------------

# Canvas/object bounds.
RECT_SIDE_OPTIONS = (4, 5, 6)
RECT_COUNT_OPTIONS = (2, 3, 4)
GRID_SIDE_OPTIONS = tuple(range(14, 19))

# Palette and marker construction.
PALETTE = range(10)
NON_MARKER_COLOR_COUNT = 2
MIN_BLOB_SIZE = 1
MAX_BLOB_SIZE = 4
INTERIOR_MARGIN = 1
MIN_MARKER_COVERAGE = 0.25
MAX_MARKER_COVERAGE = 0.32
FALLBACK_MARKER_COVERAGE = 0.22
MARKER_CONNECTIVITY = 4

# Placement and retry controls.
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
PLACEMENT_PADDING_CONNECTIVITY = 8
MAX_GENERATION_ATTEMPTS = 300


def _is_connected(cells: tuple[tuple[int, int], ...], side: int) -> bool:
    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=MARKER_CONNECTIVITY)) == 1


def _blob_candidates(side: int, size: int, taken: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    interior = [
        (r, c)
        for r in range(INTERIOR_MARGIN, side - INTERIOR_MARGIN)
        for c in range(INTERIOR_MARGIN, side - INTERIOR_MARGIN)
        if (r, c) not in taken
    ]
    return [
        combo
        for combo in itertools.combinations(interior, size)
        if _is_connected(combo, side)
    ]


def _choose_blob_sizes(count: int, side: int) -> list[int]:
    area = side * side
    interior_side = side - 2 * INTERIOR_MARGIN
    max_total = min(MAX_BLOB_SIZE * count, interior_side * interior_side)
    low = max(count, math.ceil(MIN_MARKER_COVERAGE * area))
    high = min(max_total, math.floor(MAX_MARKER_COVERAGE * area))
    if low > high:
        low = max(count, min(max_total, round(FALLBACK_MARKER_COVERAGE * area)))
        high = max_total

    total = random.randint(low, high)
    sizes = [MIN_BLOB_SIZE] * count
    remaining = total - count
    while remaining:
        choices = [i for i, size in enumerate(sizes) if size < MAX_BLOB_SIZE]
        i = random.choice(choices)
        sizes[i] += 1
        remaining -= 1
    random.shuffle(sizes)
    return sizes


def _make_layers(side: int, base: int, marker_colors: list[int]):
    output = np.full((side, side), base, dtype=int)
    sizes = _choose_blob_sizes(len(marker_colors), side)
    blobs: list[tuple[tuple[int, int], ...] | None] = [None] * len(marker_colors)
    taken: set[tuple[int, int]] = set()

    order = sorted(range(len(marker_colors)), key=lambda i: (-sizes[i], random.random()))
    for i in order:
        candidates = _blob_candidates(side, sizes[i], taken)
        if not candidates:
            return None
        blob = random.choice(candidates)
        blobs[i] = blob
        taken.update(blob)

    layers = []
    for color, blob in zip(marker_colors, blobs):
        if blob is None:
            return None
        layer = np.full((side, side), base, dtype=int)
        for r, c in blob:
            layer[r, c] = color
            output[r, c] = color
        layers.append(layer)
    return output, layers, len(taken)


def _grid_size(side: int, count: int, marker_total: int) -> int | None:
    area = side * side
    feasible = [
        s
        for s in GRID_SIDE_OPTIONS
        if s * s - count * area > count * area - marker_total
    ]
    if not feasible:
        return None
    return random.choice(feasible)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(RECT_SIDE_OPTIONS)
    count = random.choice(RECT_COUNT_OPTIONS)
    colors = random.sample(PALETTE, count + NON_MARKER_COLOR_COUNT)
    bg, base = colors[0], colors[1]
    marker_colors = colors[2:]

    made = _make_layers(side, base, marker_colors)
    if made is None:
        return None
    expected, layers, marker_total = made

    grid_side = _grid_size(side, count, marker_total)
    if grid_side is None:
        return None

    grid = np.full((grid_side, grid_side), bg, dtype=int)
    random.shuffle(layers)
    try:
        for layer in layers:
            r, c = random_free_location(
                grid,
                layer,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
            paste(grid, layer, r, c)
        actual = main(grid)
    except (AssertionError, ValueError):
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 25e02866 input")
