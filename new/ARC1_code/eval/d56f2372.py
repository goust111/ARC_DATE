from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste, random_free_location
import random


def _cropped_color_mask(g: np.ndarray, color: int, bg: int) -> np.ndarray:
    mask = g == int(color)
    b = bbox(mask)
    assert b is not None, "expected color to be present"
    r0, c0, r1, c1 = b

    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    crop[mask[r0 : r1 + 1, c0 : c1 + 1]] = int(color)
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique foreground color mask whose cropped shape is symmetric across a vertical axis.

    Input structure:
    - The grid has a dominant background and several colored masks.
    - Each non-background color forms one candidate mask, possibly split into multiple 4-connected objects.
    - Exactly one candidate's bounding-box crop is invariant under left-right reflection.

    Rule:
    - For each non-background color, crop all pixels of that color to their bounding box.
    - Select the unique crop that equals its left-right mirror image.
    - Return that crop, with non-selected cells in the crop set to the background color.

    Key insight:
    - The target is not identified by its color or size; it is the only color mask with vertical mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    symmetric_crops: list[np.ndarray] = []
    for color in colors_present(g, exclude={bg}):
        crop = _cropped_color_mask(g, color, bg)
        if np.array_equal(crop, np.fliplr(crop)):
            symmetric_crops.append(crop)

    assert len(symmetric_crops) == 1, "expected exactly one vertically symmetric color mask"
    return symmetric_crops[0]

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry bounds.
BG_COLOR = 0
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (21, 22), (16, 20)
COLOR_COUNT_RANGE, COLOR_POOL = (4, 5), range(1, 10)
GENERATOR_MAX_ATTEMPTS = 300
# Shared mask density limits.
MASK_PIXEL_RANGE, MIN_FILL_RATIO = (5, 25), (4, 10)
# Symmetric target mask bounds.
SYMMETRIC_HEIGHT_RANGE, SYMMETRIC_WIDTH_CHOICES = (3, 6), (5, 7, 9)
SYMMETRIC_MAX_FILL_RATIO = (55, 100)
# Asymmetric distractor mask bounds.
ASYMMETRIC_HEIGHT_RANGE, ASYMMETRIC_WIDTH_RANGE = (2, 8), (4, 9)
ASYMMETRIC_MAX_FILL_RATIO = (3, 4)
ASYMMETRY_BREAK_PIXELS = 1
ASYMMETRY_LR_DIFF = 2
# Placement constraints between color bounding boxes.
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 8


def _gen_pixel_count_bounds(area: int, max_fill_ratio: tuple[int, int]) -> tuple[int, int]:
    min_pixels, max_pixels = MASK_PIXEL_RANGE
    min_num, min_den = MIN_FILL_RATIO
    max_num, max_den = max_fill_ratio
    return max(min_pixels, (min_num * area + min_den - 1) // min_den), min(max_pixels, (max_num * area) // max_den)


def _gen_units(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range((w + 1) // 2)]


def _gen_unit_weight(w: int, unit: tuple[int, int]) -> int:
    _r, c = unit
    return 1 if w % 2 == 1 and c == w // 2 else 2


def _gen_count_units(w: int, units: set[tuple[int, int]]) -> int:
    return sum(_gen_unit_weight(w, unit) for unit in units)


def _gen_unit_cells(w: int, unit: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = unit
    cc = w - 1 - c
    if c == cc:
        return [(r, c)]
    return [(r, c), (r, cc)]


def _gen_choose_unit(options: list[tuple[int, int]], forbidden: set[tuple[int, int]]) -> tuple[int, int] | None:
    usable = [unit for unit in options if unit not in forbidden]
    if not usable:
        return None
    return random.choice(usable)


def _gen_complete_units(
    h: int,
    w: int,
    selected: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    min_count: int,
    max_count: int,
) -> set[tuple[int, int]] | None:
    selected = set(selected)
    base_count = _gen_count_units(w, selected)
    if base_count > max_count:
        return None

    available = [unit for unit in _gen_units(h, w) if unit not in selected and unit not in forbidden]
    ones = [unit for unit in available if _gen_unit_weight(w, unit) == 1]
    twos = [unit for unit in available if _gen_unit_weight(w, unit) == 2]

    choices: list[tuple[int, int, int]] = []
    for target in range(max(min_count, base_count), max_count + 1):
        need = target - base_count
        for n1 in range(min(len(ones), need) + 1):
            rem = need - n1
            if rem % 2 == 0 and 0 <= rem // 2 <= len(twos):
                choices.append((target, n1, rem // 2))

    if not choices:
        return None

    _, n1, n2 = random.choice(choices)
    selected.update(random.sample(ones, n1))
    selected.update(random.sample(twos, n2))
    return selected


def _gen_render_units(h: int, w: int, units: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for unit in units:
        for r, c in _gen_unit_cells(w, unit):
            mask[r, c] = True
    return mask


def _gen_symmetric_mask() -> np.ndarray | None:
    h = random.randint(*SYMMETRIC_HEIGHT_RANGE)
    w = random.choice(SYMMETRIC_WIDTH_CHOICES)
    reps = list(range((w + 1) // 2))

    selected: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    breaker_c = random.choice(reps)
    selected.add((0, breaker_c))
    forbidden.add((h - 1, breaker_c))

    bottom = _gen_choose_unit([(h - 1, c) for c in reps], forbidden)
    side = _gen_choose_unit([(r, 0) for r in range(h)], forbidden)
    if bottom is None or side is None:
        return None
    selected.add(bottom)
    selected.add(side)

    min_count, max_count = _gen_pixel_count_bounds(h * w, SYMMETRIC_MAX_FILL_RATIO)
    min_count = max(min_count, _gen_count_units(w, selected))
    completed = _gen_complete_units(h, w, selected, forbidden, min_count, max_count)
    if completed is None:
        return None

    mask = _gen_render_units(h, w, completed)
    if np.array_equal(mask, np.flipud(mask)):
        return None
    return mask


def _gen_asymmetric_mask() -> np.ndarray | None:
    h = random.randint(*ASYMMETRIC_HEIGHT_RANGE)
    w = random.randint(*ASYMMETRIC_WIDTH_RANGE)
    reps = list(range((w + 1) // 2))
    pair_reps = list(range(w // 2))
    break_unit = (random.randrange(h), random.choice(pair_reps))

    selected: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = {break_unit}

    top = _gen_choose_unit([(0, c) for c in reps], forbidden)
    if top is None:
        return None
    selected.add(top)

    ud_counterpart = (h - 1, top[1])
    if ud_counterpart != top and ud_counterpart != break_unit:
        forbidden.add(ud_counterpart)

    bottom = _gen_choose_unit([(h - 1, c) for c in reps], forbidden)
    side = _gen_choose_unit([(r, 0) for r in range(h)], forbidden)
    if bottom is None or side is None:
        return None
    selected.add(bottom)
    selected.add(side)

    final_min, final_max = _gen_pixel_count_bounds(h * w, ASYMMETRIC_MAX_FILL_RATIO)
    base_min = max(final_min - ASYMMETRY_BREAK_PIXELS, _gen_count_units(w, selected))
    base_max = final_max - ASYMMETRY_BREAK_PIXELS
    completed = _gen_complete_units(h, w, selected, forbidden, base_min, base_max)
    if completed is None:
        return None

    mask = _gen_render_units(h, w, completed)
    r, c = random.choice(_gen_unit_cells(w, break_unit))
    mask[r, c] = True

    if int(np.count_nonzero(mask != np.fliplr(mask))) != ASYMMETRY_LR_DIFF:
        return None
    return mask


def _gen_colored(mask: np.ndarray, color: int) -> np.ndarray:
    crop = np.zeros(mask.shape, dtype=int)
    crop[mask] = int(color)
    return crop


def _gen_place_crops(h: int, w: int, crops: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=int)
    order = [(crop, random.random()) for crop in crops]
    for crop, _ in sorted(order, key=lambda item: (-item[0].size, item[1])):
        footprint = np.ones(crop.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied,
                footprint,
                transparent=BG_COLOR,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, crop, r0, c0, transparent=BG_COLOR)
        paste(occupied, footprint, r0, c0)
    return grid


def _gen_masks(n_colors: int) -> list[np.ndarray] | None:
    sym_mask = _gen_symmetric_mask()
    if sym_mask is None:
        return None

    masks = [sym_mask]
    for _ in range(n_colors - 1):
        mask = _gen_asymmetric_mask()
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _gen_render_valid_grid(h: int, w: int, crops: list[np.ndarray]) -> np.ndarray | None:
    expected = crops[0]
    placed = _gen_place_crops(h, w, crops)
    if placed is None:
        return None

    try:
        actual = main(placed)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_colors = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, n_colors)

    masks = _gen_masks(n_colors)
    if masks is None:
        return None

    crops = [_gen_colored(mask, color) for mask, color in zip(masks, colors)]
    if sum(np.array_equal(crop, np.fliplr(crop)) for crop in crops) != 1:
        return None

    return _gen_render_valid_grid(h, w, crops)


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d56f2372 input")
