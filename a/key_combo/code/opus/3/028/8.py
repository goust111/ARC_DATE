from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            return SourceFileLoader("_arc_utils_028_8", str(candidate)).load_module()
    raise ImportError("could not locate new/solvers/arc_utils.py")


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(int)


def _sprite_from_component(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    orange_components = components_mask(grid == Color.ORANGE, connectivity=4)
    assert orange_components, "expected at least one orange component"

    areas = [int(len(coords)) for coords in orange_components]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "expected a unique largest orange component"

    coords = orange_components[areas.index(max_area)]
    assert 3 <= int(coords.shape[0]) <= 6, "sprite area must be between 3 and 6 pixels"

    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    size = int(r1 - r0 + 1)
    assert size == int(c1 - c0 + 1), "sprite bbox must be square"
    assert size in (2, 3), "sprite bbox side must be 2 or 3"

    clean = np.zeros_like(grid, dtype=int)
    for r, c in coords:
        clean[int(r), int(c)] = Color.ORANGE
    sprite = crop_bbox(clean, box)
    assert not any(np.array_equal(sprite != 0, np.rot90(sprite != 0, k)) for k in (1, 2, 3))
    return sprite, coords, box


def _build_pinwheel(sprite: np.ndarray) -> np.ndarray:
    size = int(sprite.shape[0])
    pinwheel = np.zeros((2 * size, 2 * size), dtype=int)
    paste(pinwheel, sprite, 0, 0)
    paste(pinwheel, np.rot90(sprite, 1), 0, size)
    paste(pinwheel, np.rot90(sprite, 2), size, 0)
    paste(pinwheel, np.rot90(sprite, -1), size, size)
    return pinwheel


def _tile_pinwheel(pinwheel: np.ndarray) -> np.ndarray:
    tile_h, tile_w = map(int, pinwheel.shape)
    out = np.zeros((2 * tile_h, 2 * tile_w), dtype=int)
    for r0 in (0, tile_h):
        for c0 in (0, tile_w):
            paste(out, pinwheel, r0, c0)
    return out


def _recolor_by_depth(pattern: np.ndarray, inner_color: int, outer_color: int, radius: int) -> np.ndarray:
    active = pattern != Color.BLACK
    center = (int(pattern.shape[0]) - 1) / 2.0
    rr, cc = np.indices(pattern.shape)
    chebyshev = np.maximum(np.abs(rr - center), np.abs(cc - center))
    layers = _round_half_up(chebyshev)

    out = np.zeros_like(pattern)
    out[active & (layers <= int(radius))] = int(inner_color)
    out[active & (layers > int(radius))] = int(outer_color)
    return out


def _expected_from_sprite(sprite: np.ndarray, inner_color: int, outer_color: int) -> np.ndarray:
    size = int(sprite.shape[0])
    pinwheel = _build_pinwheel(sprite)
    tiled = _tile_pinwheel(pinwheel)
    return _recolor_by_depth(tiled, inner_color, outer_color, size)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise the orange sprite, turn it into a four-rotation pinwheel, tile that pinwheel, then color the tiled pixels by depth from the output center.

    Input structure:
    - The grid has black background, one irregular orange sprite, several isolated orange noise pixels, and one or two grey distractor pixels.
    - The sprite is the unique largest orange 4-connected component and its bbox is a 2x2 or 3x3 square.
    - The bottom-right two cells form a 1x2 color bar whose first color is the inner-layer color and whose second color is the outer-layer color.

    Rule:
    - Keep only the largest orange 4-connected component, erase the other orange pixels and all grey distractors, and crop the kept component to its square bbox.
    - Build a 2x2 pinwheel from that cropped sprite: original in the upper-left, 90-degree counterclockwise in the upper-right, 180-degree in the lower-left, and 90-degree clockwise in the lower-right.
    - Tile that 2N x 2N pinwheel into a 4N x 4N output, then recolor each active pixel by its rounded Chebyshev distance from the output center: distance at most N gets the first bar color, distance greater than N gets the second bar color.

    Key insight:
    - The largest orange component fixes the only valid sprite, and that sprite controls every later stage: its exact asymmetry determines the pinwheel support, the tiling footprint, and which repeated pixels fall into the inner versus outer depth layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a 1x2 color bar"

    sprite, _coords, _box = _sprite_from_component(g)
    inner_color = int(g[-1, -2])
    outer_color = int(g[-1, -1])
    return _expected_from_sprite(sprite, inner_color, outer_color)


def _enumerate_unique_sprites() -> tuple[tuple[np.ndarray, ...], dict[int, tuple[int, ...]]]:
    entries: list[np.ndarray] = []
    ids_by_size: dict[int, list[int]] = {2: [], 3: []}
    seen_outputs: set[tuple[tuple[int, int], bytes]] = set()

    for size in (2, 3):
        for bits in range(1 << (size * size)):
            area = bits.bit_count()
            if not 3 <= area <= 6:
                continue

            sprite = np.zeros((size, size), dtype=int)
            for idx in range(size * size):
                if bits & (1 << idx):
                    sprite[idx // size, idx % size] = Color.ORANGE

            mask = sprite != 0
            if len(components_mask(mask, connectivity=4)) != 1:
                continue
            if bbox_coords(np.argwhere(mask)) != (0, 0, size - 1, size - 1):
                continue
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue

            expected = _expected_from_sprite(sprite, Color.BLUE, Color.RED)
            if not ((expected == Color.BLUE).any() and (expected == Color.RED).any()):
                continue

            signature = (expected.shape, expected.tobytes())
            if signature in seen_outputs:
                continue
            seen_outputs.add(signature)

            catalog_id = len(entries)
            entries.append(sprite)
            ids_by_size[size].append(catalog_id)

    assert ids_by_size[2] and ids_by_size[3], "expected both 2x2 and 3x3 sprite variants"
    return tuple(entries), {size: tuple(ids) for size, ids in ids_by_size.items()}


def _is_4_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _ordered_cells(height: int, width: int, seed: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    start = seed % len(cells)
    return cells[start:] + cells[:start]


def _pick_isolated_cells(
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    banned_neighbors: set[tuple[int, int]],
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    for cell in _ordered_cells(height, width, seed):
        if cell in occupied:
            continue
        if any(_is_4_adjacent(cell, other) for other in banned_neighbors):
            continue
        if any(_is_4_adjacent(cell, other) for other in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("failed to place isolated cells")


def _build_input_for_sprite(sprite: np.ndarray, catalog_id: int) -> np.ndarray:
    size = int(sprite.shape[0])
    area = int(np.count_nonzero(sprite))
    height = size + 5 + (catalog_id % 3)
    width = size + 5 + ((catalog_id // 3) % 3)

    row_choices = list(range(1, height - size - 1))
    col_choices = list(range(1, width - size - 1))
    r0 = row_choices[(catalog_id // 9) % len(row_choices)]
    c0 = col_choices[(catalog_id // 27) % len(col_choices)]

    grid = np.zeros((height, width), dtype=int)
    grid[-1, -2] = Color.BLUE
    grid[-1, -1] = Color.RED

    sprite_cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(sprite != 0)}
    for r, c in sprite_cells:
        grid[r, c] = Color.ORANGE

    bbox_cells = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
    occupied = set(bbox_cells)
    occupied.update({(height - 1, width - 2), (height - 1, width - 1)})

    critical_noise_candidates = []
    for dr, dc in [(-1, -1), (-1, size), (size, -1), (size, size)]:
        cell = (r0 + dr, c0 + dc)
        if 0 <= cell[0] < height and 0 <= cell[1] < width and cell not in occupied:
            critical_noise_candidates.append(cell)
    assert critical_noise_candidates, "expected at least one diagonal noise position"

    noise_count = 2 + ((catalog_id + area) % 3)
    grey_count = 1 + ((catalog_id + size) % 2)

    noise_cells = [critical_noise_candidates[catalog_id % len(critical_noise_candidates)]]
    noise_occupied = occupied | sprite_cells | set(noise_cells)
    extra_noise = _pick_isolated_cells(
        height,
        width,
        noise_occupied,
        sprite_cells | set(noise_cells),
        noise_count - 1,
        seed=7 * catalog_id + area,
    )
    noise_cells.extend(extra_noise)

    for r, c in noise_cells:
        grid[r, c] = Color.ORANGE

    grey_occupied = noise_occupied | set(extra_noise)
    grey_cells = _pick_isolated_cells(
        height,
        width,
        grey_occupied,
        sprite_cells | set(noise_cells),
        grey_count,
        seed=11 * catalog_id + size,
    )
    for r, c in grey_cells:
        grid[r, c] = Color.GREY

    return grid


def _validate_generated_grid(grid: np.ndarray, sprite: np.ndarray) -> None:
    size = int(sprite.shape[0])
    orange_components = components_mask(grid == Color.ORANGE, connectivity=4)
    areas = [int(len(coords)) for coords in orange_components]
    max_area = max(areas)
    assert areas.count(max_area) == 1
    assert max_area == int(np.count_nonzero(sprite))

    largest = orange_components[areas.index(max_area)]
    box = bbox_coords(largest)
    assert box[2] - box[0] + 1 == size
    assert box[3] - box[1] + 1 == size

    rebuilt_sprite, _coords, _rebuilt_box = _sprite_from_component(grid)
    assert np.array_equal(rebuilt_sprite, sprite)

    full_orange = (grid == Color.ORANGE).astype(int)
    all_box = bbox_coords(np.argwhere(full_orange != 0))
    assert all_box != box, "noise should make the all-orange bbox differ from the true sprite bbox"
    assert crop_bbox(full_orange, all_box).shape != sprite.shape or not np.array_equal(
        crop_bbox(full_orange, all_box) != 0,
        sprite != 0,
    )

    expected = _expected_from_sprite(sprite, Color.BLUE, Color.RED)
    actual = main(grid)
    assert actual.shape == (4 * size, 4 * size)
    assert np.array_equal(actual, expected)
    assert (actual == Color.BLUE).any() and (actual == Color.RED).any()

    support = actual != 0
    pinwheel_side = 2 * size
    tile = support[:pinwheel_side, :pinwheel_side]
    assert np.array_equal(support[:pinwheel_side, pinwheel_side:], tile)
    assert np.array_equal(support[pinwheel_side:, :pinwheel_side], tile)
    assert np.array_equal(support[pinwheel_side:, pinwheel_side:], tile)

    assert int(np.sum(grid == Color.GREY)) in (1, 2)
    noise_pixels = int(np.sum(grid == Color.ORANGE)) - int(np.count_nonzero(sprite))
    assert 2 <= noise_pixels <= 4
    assert int(grid[-1, -2]) == Color.BLUE and int(grid[-1, -1]) == Color.RED


_SPRITE_CATALOG, _CATALOG_IDS_BY_SIZE = _enumerate_unique_sprites()
_INPUT_CATALOG: tuple[np.ndarray, ...] = tuple(
    _build_input_for_sprite(sprite, catalog_id) for catalog_id, sprite in enumerate(_SPRITE_CATALOG)
)
for catalog_id, sprite in enumerate(_SPRITE_CATALOG):
    _validate_generated_grid(_INPUT_CATALOG[catalog_id], sprite)


def generate_input() -> np.ndarray:
    size = random.choice((2, 3))
    catalog_id = random.choice(_CATALOG_IDS_BY_SIZE[size])
    return _INPUT_CATALOG[catalog_id].copy()
