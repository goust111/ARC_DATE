from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste, random_free_location
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every red foreground object of area at least four to pink.

    Input structure
    - The grid has a black background and a single red foreground color.
    - The foreground is split into one or more 4-connected red objects.
    - Pink is absent from the input and is used only for selected output objects.

    Rule
    - Preserve black background pixels and red objects with area one, two, or three.
    - Recolor every 4-connected red object with area at least four to pink.

    Key insight
    - The changed objects are exactly the connected red objects that are large
      enough; their position and shape do not otherwise matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    foreground = colors_present(g, exclude={bg})
    assert foreground == [Color.RED], "expected red to be the only foreground color"
    assert not np.any(g == Color.PINK), "pink must not appear in the input"

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.RED],
        return_format="dict",
    )
    assert objects, "expected at least one red object"
    large_objects = [obj for obj in objects if int(obj["area"]) >= 4]
    assert large_objects, "expected at least one red object of area at least four"

    out = g.copy()
    for obj in large_objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = Color.PINK
    return out

# --------------- generator (o5x) ---------------

# Canvas and density bounds.
GRID_HEIGHT_RANGE = (6, 10)
GRID_WIDTH_RANGE = (6, 12)
SPEC_DENSITY_RANGE = (0.10, 0.40)
OBSERVED_DENSITY_RANGE = (0.14, 0.36)

# Object count and area bounds.
LARGE_COUNT_RANGE = (1, 3)
SMALL_COUNT_RANGE = (4, 6)
TOTAL_COUNT_RANGE = (6, 9)
LARGE_AREA_RANGE = (4, 8)
SMALL_AREA_RANGE = (1, 3)

# Placement, retry, and polyomino controls.
GENERATION_RETRIES = 300
PLACEMENT_PADDING = 1
PLACEMENT_PADDING_CONNECTIVITY = 4
NEIGHBOR_OFFSETS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATION_ROTATIONS = 4


def _gen_polyomino_patch(area: int, max_h: int, max_w: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        candidates = []
        for r, c in coords:
            for dr, dc in NEIGHBOR_OFFSETS_4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coords:
                    continue
                trial = coords | {(nr, nc)}
                rs = [p[0] for p in trial]
                cs = [p[1] for p in trial]
                if max(rs) - min(rs) + 1 <= max_h and max(cs) - min(cs) + 1 <= max_w:
                    candidates.append((nr, nc))
        coords.add(random.choice(candidates))

    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    h = max(r for r, _ in coords) - min_r + 1
    w = max(c for _, c in coords) - min_c + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in coords:
        patch[r - min_r, c - min_c] = Color.RED
    variants = []
    for k in range(ORIENTATION_ROTATIONS):
        rotated = np.rot90(patch, k)
        for variant in (rotated, np.fliplr(rotated)):
            if variant.shape[0] <= max_h and variant.shape[1] <= max_w:
                variants.append(variant.copy())
    return random.choice(variants)


def _gen_distribute_areas(large_count: int, small_count: int, target_area: int) -> tuple[list[int], list[int]]:
    areas = [LARGE_AREA_RANGE[0]] * large_count + [SMALL_AREA_RANGE[0]] * small_count
    caps = [LARGE_AREA_RANGE[1]] * large_count + [SMALL_AREA_RANGE[1]] * small_count
    extra = target_area - sum(areas)

    while extra > 0:
        choices = [i for i, area in enumerate(areas) if area < caps[i]]
        i = random.choice(choices)
        areas[i] += 1
        extra -= 1
    return areas[:large_count], areas[large_count:]


def _gen_count_pair(cell_count: int) -> tuple[int, int] | None:
    density_low = math.ceil(SPEC_DENSITY_RANGE[0] * cell_count)
    density_high = math.floor(SPEC_DENSITY_RANGE[1] * cell_count)
    pairs = []
    for large_count in range(LARGE_COUNT_RANGE[0], LARGE_COUNT_RANGE[1] + 1):
        for small_count in range(SMALL_COUNT_RANGE[0], SMALL_COUNT_RANGE[1] + 1):
            total_count = large_count + small_count
            if not TOTAL_COUNT_RANGE[0] <= total_count <= TOTAL_COUNT_RANGE[1]:
                continue
            min_area = LARGE_AREA_RANGE[0] * large_count + SMALL_AREA_RANGE[0] * small_count
            max_area = LARGE_AREA_RANGE[1] * large_count + SMALL_AREA_RANGE[1] * small_count
            if max(min_area, density_low) <= min(max_area, density_high):
                pairs.append((large_count, small_count))
    if not pairs:
        return None
    return random.choice(pairs)


def _choose_target_area(cell_count: int, large_count: int, small_count: int) -> int:
    """Choose total red area within spec density, preferring observed eval density."""
    min_area = LARGE_AREA_RANGE[0] * large_count + SMALL_AREA_RANGE[0] * small_count
    max_area = LARGE_AREA_RANGE[1] * large_count + SMALL_AREA_RANGE[1] * small_count
    spec_low = max(min_area, math.ceil(SPEC_DENSITY_RANGE[0] * cell_count))
    spec_high = min(max_area, math.floor(SPEC_DENSITY_RANGE[1] * cell_count))
    observed_low = max(spec_low, math.ceil(OBSERVED_DENSITY_RANGE[0] * cell_count))
    observed_high = min(spec_high, math.floor(OBSERVED_DENSITY_RANGE[1] * cell_count))
    low, high = (observed_low, observed_high) if observed_low <= observed_high else (spec_low, spec_high)
    return random.randint(low, high)


def _place_objects(canvas: np.ndarray, large_patches: list[np.ndarray], small_patches: list[np.ndarray]) -> np.ndarray | None:
    """Place patches with 4-connected separation and return the expected output."""
    expected = canvas.copy()
    objects = [(True, patch) for patch in large_patches] + [(False, patch) for patch in small_patches]

    for is_large, patch in objects:
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=Color.BLACK,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=Color.BLACK)
        out_patch = np.where(patch == Color.RED, Color.PINK if is_large else Color.RED, Color.BLACK)
        paste(expected, out_patch, r0, c0, transparent=Color.BLACK)
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    cell_count = height * width

    count_pair = _gen_count_pair(cell_count)
    if count_pair is None:
        return None
    large_count, small_count = count_pair

    target_area = _choose_target_area(cell_count, large_count, small_count)
    large_areas, small_areas = _gen_distribute_areas(large_count, small_count, target_area)
    large_patches = [_gen_polyomino_patch(area, height, width) for area in large_areas]
    small_patches = [_gen_polyomino_patch(area, height, width) for area in small_areas]
    random.shuffle(large_patches)
    random.shuffle(small_patches)

    canvas = np.zeros((height, width), dtype=int)
    expected = _place_objects(canvas, large_patches, small_patches)
    if expected is None:
        return None
    return canvas if np.array_equal(main(canvas), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ae58858e input")
