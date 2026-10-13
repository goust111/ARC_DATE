import numpy as np

from arc_utils import Color, components_mask, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every non-isolated black object in the blue field to teal.

    Input structure:
    - The grid contains only blue background pixels and black foreground pixels.
    - Black pixels form 4-connected objects, including singleton noise pixels.

    Rule:
    - Keep all blue pixels unchanged.
    - Keep every singleton black object unchanged.
    - Recolor every black object with area greater than one to teal.

    Key insight:
    - The distinction is 4-connectivity among black pixels: touching black pixels are
      real objects to mark, while isolated black pixels are preserved as noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, Color.BLUE}), "expected only black and blue"

    out = g.copy()
    for coords in components_mask(g == Color.BLACK, connectivity=4):
        if len(coords) > 1:
            out[coords[:, 0], coords[:, 1]] = Color.TEAL

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GRID_SHAPE = (13, 13)
BACKGROUND_COLOR, OBJECT_COLOR, RECOLOR_COLOR = Color.BLUE, Color.BLACK, Color.TEAL

# Structural target ranges.
BLACK_TOTAL_RANGE = (27, 52)
COMPONENT_TOTAL_RANGE, MULTI_COUNT_RANGE, SINGLETON_COUNT_RANGE = (18, 30), (5, 7), (13, 24)
GENERATION_RETRIES = 300

# Multi-cell object size plan.
SMALL_BASE_SIZE, SMALL_EXTRA_CAPACITY = 2, 1
MEDIUM_BASE_SIZE, MEDIUM_EXTRA_CAPACITY = 4, 0
LARGE_BASE_SIZE, LARGE_EXTRA_CAPACITY = 5, 3
LARGE_COUNT_OPTIONS = (1, 2)

# Polyomino shape and placement constraints.
DIRS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PATCH_LONG_SIDE_MAX, PATCH_SHORT_SIDE_MAX = 5, 3
ROTATION_COUNT, FLIP_PROBABILITY = 4, 0.5
PLACEMENT_PADDING, CONNECTIVITY_4 = 1, 4


def _closed_range(bounds: tuple[int, int]) -> range:
    return range(bounds[0], bounds[1] + 1)


def _gen_size_count_plans(obj_count: int, pixel_total: int) -> list[tuple[int, int, int]]:
    plans = []
    min_small = obj_count // 2 + 1
    for small_count in range(min_small, obj_count + 1):
        for large_count in LARGE_COUNT_OPTIONS:
            medium_count = obj_count - small_count - large_count
            if medium_count < 0:
                continue
            base = SMALL_BASE_SIZE * small_count + MEDIUM_BASE_SIZE * medium_count + LARGE_BASE_SIZE * large_count
            capacity = SMALL_EXTRA_CAPACITY * small_count + MEDIUM_EXTRA_CAPACITY * medium_count + LARGE_EXTRA_CAPACITY * large_count
            if base <= pixel_total <= base + capacity:
                plans.append((small_count, medium_count, large_count))
    return plans


def _gen_object_sizes(obj_count: int, pixel_total: int) -> list[int]:
    small_count, medium_count, large_count = random.choice(
        _gen_size_count_plans(obj_count, pixel_total)
    )
    sizes = [SMALL_BASE_SIZE] * small_count + [MEDIUM_BASE_SIZE] * medium_count + [LARGE_BASE_SIZE] * large_count
    capacities = [SMALL_EXTRA_CAPACITY] * small_count + [MEDIUM_EXTRA_CAPACITY] * medium_count + [LARGE_EXTRA_CAPACITY] * large_count
    extra = pixel_total - sum(sizes)
    while extra:
        expandable = [i for i, cap in enumerate(capacities) if cap > 0]
        i = random.choice(expandable)
        sizes[i] += 1
        capacities[i] -= 1
        extra -= 1
    random.shuffle(sizes)
    return sizes


def _gen_polyomino_patch(area: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier = []
        for r, c in coords:
            for dr, dc in DIRS_4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coords:
                    continue
                trial = coords | {(nr, nc)}
                rows = [p[0] for p in trial]
                cols = [p[1] for p in trial]
                h = max(rows) - min(rows) + 1
                w = max(cols) - min(cols) + 1
                if max(h, w) <= PATCH_LONG_SIDE_MAX and min(h, w) <= PATCH_SHORT_SIDE_MAX:
                    frontier.append((nr, nc))
        if not frontier:
            return np.full((1, area), OBJECT_COLOR, dtype=int)
        coords.add(random.choice(frontier))

    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = [(r - min_r, c - min_c) for r, c in coords]
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    for r, c in norm:
        patch[r, c] = OBJECT_COLOR

    patch = np.rot90(patch, random.randrange(ROTATION_COUNT))
    if random.random() < FLIP_PROBABILITY:
        patch = np.flipud(patch)
    if random.random() < FLIP_PROBABILITY:
        patch = np.fliplr(patch)
    return patch.copy()


def _gen_structural_choices(black_total: int) -> list[tuple[int, int, int, int]]:
    choices = []
    for component_total in _closed_range(COMPONENT_TOTAL_RANGE):
        for multi_count in _closed_range(MULTI_COUNT_RANGE):
            singleton_count = component_total - multi_count
            if not SINGLETON_COUNT_RANGE[0] <= singleton_count <= SINGLETON_COUNT_RANGE[1]:
                continue
            multi_pixels = black_total - singleton_count
            if _gen_size_count_plans(multi_count, multi_pixels):
                choices.append(
                    (component_total, multi_count, singleton_count, multi_pixels)
                )
    return choices


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    """Place a patch while preserving 4-neighbor separation between black components."""
    try:
        r, c = random_free_location(
            grid,
            patch,
            transparent=BACKGROUND_COLOR,
            padding=PLACEMENT_PADDING,
            padding_connectivity=CONNECTIVITY_4,
        )
    except ValueError:
        return False
    paste(grid, patch, r, c, transparent=BACKGROUND_COLOR)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    black_total = random.randint(*BLACK_TOTAL_RANGE)
    choices = _gen_structural_choices(black_total)
    if not choices:
        return None

    component_total, multi_count, singleton_count, multi_pixels = random.choice(choices)
    multi_sizes = _gen_object_sizes(multi_count, multi_pixels)

    for size in sorted(multi_sizes, reverse=True):
        if not _place_patch(grid, _gen_polyomino_patch(size)):
            return None

    singleton = np.array([[OBJECT_COLOR]], dtype=int)
    for _ in range(singleton_count):
        if not _place_patch(grid, singleton):
            return None

    if int(np.sum(grid == OBJECT_COLOR)) != black_total:
        return None

    comps = components_mask(grid == OBJECT_COLOR, connectivity=CONNECTIVITY_4)
    if len(comps) != component_total:
        return None
    found_multi = sorted(len(coords) for coords in comps if len(coords) > 1)
    if len(found_multi) != multi_count or found_multi != sorted(multi_sizes):
        return None
    if sum(1 for coords in comps if len(coords) == 1) != singleton_count:
        return None

    expected = grid.copy()
    for coords in comps:
        if len(coords) > 1:
            expected[coords[:, 0], coords[:, 1]] = RECOLOR_COLOR
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e0fb7511 input")
