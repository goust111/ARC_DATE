from __future__ import annotations

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the four colored 2x2 L sprites into a compact 4x4 frame around a central hole.

    Input structure
    - The background is the dominant color.
    - There are exactly four monochrome 4-connected foreground objects.
    - Each object is a three-pixel L shape occupying a 2x2 bounding box with one background corner.

    Rule
    - Crop each L sprite to its 2x2 bounding box.
    - Place it in the 4x4 output quadrant opposite its missing corner, preserving its orientation and color.
    - The missing corner of each sprite becomes one cell of the central 2x2 background hole.

    Key insight
    - The object orientation, not its absolute input position or color id, determines which output quadrant it occupies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(objects) == 4, "expected exactly four foreground L objects"

    out = np.full((4, 4), bg, dtype=int)
    occupied_quadrants: set[tuple[int, int]] = set()
    for obj in objects:
        assert obj["area"] == 3, "each object must contain three pixels"
        r0, c0, r1, c1 = obj["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each object must fit a 2x2 box"

        sprite = crop_bbox(g, (r0, c0, r1, c1))
        color = int(obj["color"])
        assert int(np.count_nonzero(sprite == color)) == 3, "sprite must contain three same-color pixels"
        missing = np.argwhere(sprite == bg)
        assert missing.shape == (1, 2), "sprite must have one background corner"

        missing_r, missing_c = map(int, missing[0])
        dst_r = 2 * (1 - missing_r)
        dst_c = 2 * (1 - missing_c)
        assert (dst_r, dst_c) not in occupied_quadrants, "orientations must cover distinct output quadrants"
        occupied_quadrants.add((dst_r, dst_c))
        paste(out, sprite, dst_r, dst_c)

    assert len(occupied_quadrants) == 4, "expected all four output quadrants to be filled"
    return out

# --------------- generator (o5x) ---------------
from arc_utils import random_free_location

# Canvas, palette, and orientation constants.
BG = 0
INPUT_SHAPE, OUTPUT_SHAPE, SPRITE_SHAPE = (13, 13), (4, 4), (2, 2)
COLOR_POOL = np.arange(1, 10)
MISSING_CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Placement and retry constants.
PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY, MAX_ATTEMPTS = 3, 8, 300

def _build_output_and_sprites() -> tuple[np.ndarray, list[np.ndarray]]:
    """Sample the expected output and the four 2x2 L-sprites to scatter."""
    out = np.full(OUTPUT_SHAPE, BG, dtype=int)
    colors = np.random.choice(COLOR_POOL, size=len(MISSING_CORNERS), replace=False)
    missing_corners = list(MISSING_CORNERS)
    np.random.shuffle(missing_corners)
    sprites: list[np.ndarray] = []
    for color, (missing_r, missing_c) in zip(colors, missing_corners):
        sprite = np.full(SPRITE_SHAPE, int(color), dtype=int)
        sprite[missing_r, missing_c] = BG
        dst_r = SPRITE_SHAPE[0] * (1 - missing_r)
        dst_c = SPRITE_SHAPE[1] * (1 - missing_c)
        paste(out, sprite, dst_r, dst_c)
        sprites.append(sprite.copy())
    return out, sprites


def _gen_attempt() -> np.ndarray | None:
    out, sprites = _build_output_and_sprites()
    inp = np.full(INPUT_SHAPE, BG, dtype=int)
    occupied = np.full(INPUT_SHAPE, BG, dtype=int)
    marker = np.ones(SPRITE_SHAPE, dtype=int)
    try:
        for idx in np.random.permutation(len(sprites)):
            sprite = sprites[int(idx)]
            r, c = random_free_location(
                occupied,
                marker,
                transparent=BG,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
            paste(occupied, marker, r, c)
            paste(inp, sprite, r, c, transparent=BG)
    except ValueError:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a61ba2ce input")
