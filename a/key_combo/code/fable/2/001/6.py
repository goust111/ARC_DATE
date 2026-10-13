from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, paste, random_free_location, random_sprite, scale_sprite


BG = Color.BLACK
GIANT_COLOR = Color.GREY
BLOCK_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200


def _component_patch(grid: np.ndarray, component: dict) -> np.ndarray:
    return crop_bbox(grid, component["bbox"])


def _downscale_scaled_mask(giant_mask: np.ndarray, small_side: int) -> tuple[np.ndarray, int] | None:
    big_h, big_w = giant_mask.shape
    if big_h != big_w or small_side <= 0 or big_h % small_side != 0:
        return None
    scale = big_h // small_side
    if scale < 1:
        return None

    small_mask = np.zeros((small_side, small_side), dtype=bool)
    for r in range(small_side):
        for c in range(small_side):
            block = giant_mask[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            filled = bool(block[0, 0])
            if not np.all(block == filled):
                return None
            small_mask[r, c] = filled
    return small_mask, scale


def _giant_component(grid: np.ndarray) -> dict:
    gray_components = find_components(
        grid,
        bg=BG,
        colors=[GIANT_COLOR],
        connectivity=8,
        monochromatic=True,
    )
    square_gray = [
        component
        for component in gray_components
        if bbox_h(component["bbox"]) == bbox_w(component["bbox"])
    ]
    assert square_gray, "expected a square gray giant component"
    return max(square_gray, key=lambda component: (bbox_h(component["bbox"]), int(component["area"])))


def _block_components(grid: np.ndarray, giant_bbox: tuple[int, int, int, int]) -> list[dict]:
    components = find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=False,
    )

    blocks: list[dict] = []
    for component in components:
        if component["bbox"] == giant_bbox:
            continue
        patch = _component_patch(grid, component)
        non_black = set(map(int, np.unique(patch))) - {BG}
        if len(non_black) < 2:
            continue
        if patch.shape[0] != patch.shape[1]:
            continue
        blocks.append(component)
    return blocks


def _all_matches(grid: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    giant = _giant_component(grid)
    giant_patch = _component_patch(grid, giant)
    giant_mask = giant_patch == GIANT_COLOR
    big_side = giant_patch.shape[0]

    downscaled_cache: dict[int, tuple[np.ndarray, int] | None] = {}
    matches: list[tuple[np.ndarray, int, int]] = []
    for component in _block_components(grid, giant["bbox"]):
        patch = _component_patch(grid, component)
        small_side = patch.shape[0]
        if small_side >= big_side:
            continue

        if small_side not in downscaled_cache:
            downscaled_cache[small_side] = _downscale_scaled_mask(giant_mask, small_side)
        scaled = downscaled_cache[small_side]
        if scaled is None:
            continue

        target_mask, scale = scaled
        if scale < 2:
            continue

        block_mask = patch != BG
        for rot in range(4):
            if np.array_equal(np.rot90(block_mask, rot), target_mask):
                matches.append((patch, rot, scale))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray giant by finding the uniquely silhouette-matching small multicolor block.

    Input structure:
    - A black background contains one gray giant shape whose square bounding box is an integer upscale of an n x n binary silhouette.
    - Several separate n x n multicolor blocks also appear; black cells inside each block are part of its pattern.
    - Exactly one block has a non-black silhouette that matches the giant's downscaled silhouette after a quarter-turn rotation.

    Rule:
    - Isolate the square gray giant and the small square multicolor blocks.
    - Downscale the giant silhouette by the integer factor given by giant-side divided by block-side, and test all four rotations of every block silhouette.
    - Return the giant bounding-box canvas recolored by the unique matching block rotated into the matching pose and then upscaled back by the same integer factor, keeping black cells black.

    Key insight:
    - The giant contributes only shape and scale; all color information lives in the one small block whose silhouette is isomorphic to the giant after rotation.
    """
    g = np.asarray(grid, dtype=int)
    matches = _all_matches(g)
    assert len(matches) == 1, "expected exactly one matching block/rotation"
    patch, rot, scale = matches[0]
    return scale_sprite(np.rot90(patch, rot), scale)


def _mask_has_unique_rotation(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, rot)) for rot in (1, 2, 3))


def _silhouette_matches_under_rotation(mask: np.ndarray, target: np.ndarray) -> bool:
    return any(np.array_equal(np.rot90(mask, rot), target) for rot in range(4))


def _generate_mask(side: int, forbidden: np.ndarray | None = None) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        density = random.uniform(0.45, 0.75)
        sprite = random_sprite(
            side,
            side,
            density=density,
            symmetry="not_symmetric",
            color_palette=[Color.BLUE],
            connectivity=8,
            background=BG,
        )
        mask = sprite != BG
        if not mask.any() or mask.all():
            continue
        if not _mask_has_unique_rotation(mask):
            continue
        if forbidden is not None and _silhouette_matches_under_rotation(mask, forbidden):
            continue
        return mask
    raise RuntimeError("failed to generate a valid silhouette")


def _paint_block(mask: np.ndarray) -> np.ndarray:
    side = mask.shape[0]
    coords = list(map(tuple, np.argwhere(mask)))
    palette_size = random.randint(2, min(5, len(coords)))
    palette = random.sample(BLOCK_COLORS, palette_size)

    patch = np.zeros((side, side), dtype=int)
    random.shuffle(coords)
    for i, (r, c) in enumerate(coords):
        color = palette[i] if i < palette_size else random.choice(palette)
        patch[r, c] = int(color)
    return patch


def _reserve_square(occupied: np.ndarray, side: int, *, border: int = 1, padding: int = 1) -> tuple[int, int]:
    footprint = np.ones((side, side), dtype=int)
    return random_free_location(
        occupied,
        footprint,
        transparent=0,
        border=border,
        padding=padding,
        padding_connectivity=8,
    )


def _place_square_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> None:
    side = patch.shape[0]
    r, c = _reserve_square(occupied, side)
    occupied[r : r + side, c : c + side] = 1
    paste(grid, patch, r, c, transparent=BG)


def _construct_scene() -> tuple[np.ndarray, np.ndarray]:
    side = random.randint(3, 6)
    scale = random.randint(2, min(4, 30 // side))
    giant_side = side * scale
    distractor_count = random.randint(1, 4)

    base_mask = _generate_mask(side)
    target_block = _paint_block(base_mask)
    rotation = random.randint(0, 3)
    giant_mask = np.rot90(base_mask, rotation)
    giant_patch = np.where(scale_sprite(giant_mask.astype(int), scale), GIANT_COLOR, BG)
    expected = scale_sprite(np.rot90(target_block, rotation), scale)

    footprint_area = giant_side * giant_side + (distractor_count + 1) * side * side
    min_extent = max(giant_side + 2, int(np.ceil(np.sqrt(footprint_area))) + 5, 18)
    if min_extent > 30:
        raise RuntimeError("scene cannot fit within ARC limits")

    height = random.randint(min_extent, 30)
    width = random.randint(min_extent, 30)
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=int)

    patches = [target_block]
    for _ in range(distractor_count):
        distractor_mask = _generate_mask(side, forbidden=giant_mask)
        patches.append(_paint_block(distractor_mask))
    random.shuffle(patches)

    _place_square_patch(grid, occupied, giant_patch)
    for patch in patches:
        _place_square_patch(grid, occupied, patch)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _construct_scene()
        except (RuntimeError, ValueError):
            continue

        matches = _all_matches(grid)
        if len(matches) != 1:
            continue
        patch, rot, scale = matches[0]
        if not np.array_equal(scale_sprite(np.rot90(patch, rot), scale), expected):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid giant recoloring task")
