import numpy as np

from arc_utils import Color, bbox, bbox_h, bbox_w, crop_bbox, get_bg, label_mask, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the teal-marked sprite as a 3x3 mask recolored to that sprite's color.

    Input structure
    - A background grid contains several small foreground sprites.
    - Exactly one teal control pixel is 4-connected into one sprite.
    - The marked multicolor component has a 3x3 bounding box and exactly one
      non-teal foreground color.

    Rule
    - Find the 4-connected foreground component containing the teal marker.
    - Crop that component's bounding box.
    - Output the component mask inside the crop, replacing both the marker and
      the sprite pixels with the component's non-teal color.

    Key insight
    - The teal pixel selects the sprite to summarize; the other sprites are
      distractors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    marker_coords = np.argwhere(g == Color.TEAL)
    assert marker_coords.shape[0] == 1
    mr, mc = map(int, marker_coords[0])

    labels, _n = label_mask(g != bg, connectivity=4)
    component_id = int(labels[mr, mc])
    assert component_id > 0
    component = labels == component_id

    component_bbox = bbox(component)
    assert component_bbox is not None
    assert (bbox_h(component_bbox), bbox_w(component_bbox)) == (3, 3)

    component_colors = set(map(int, g[component]))
    component_colors.discard(bg)
    component_colors.discard(Color.TEAL)
    assert len(component_colors) == 1
    sprite_color = component_colors.pop()

    component_crop = crop_bbox(component, component_bbox)
    out = np.full(component_crop.shape, bg, dtype=int)
    out[component_crop] = sprite_color
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and sprite bounds.
BG, TEAL = 0, 8
CANVAS_SHAPE = (13, 13)
SPRITE_SHAPE, SPRITE_BBOX = (3, 3), (0, 0, 2, 2)
SPRITE_COUNT_BOUNDS, PIXEL_COUNT_BOUNDS = (3, 4), (5, 8)
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Center marker, placement, and retry controls.
CENTER = (1, 1)
CENTER_NEIGHBORS = ((0, 1), (1, 0), (1, 2), (2, 1))
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 8
MAX_GENERATION_ATTEMPTS = 300

def _gen_mask_options(center_required: bool) -> dict[int, list[np.ndarray]]:
    options = {k: [] for k in range(PIXEL_COUNT_BOUNDS[0], PIXEL_COUNT_BOUNDS[1] + 1)}
    for bits in range(1 << (SPRITE_SHAPE[0] * SPRITE_SHAPE[1])):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(SPRITE_SHAPE)
        count = int(mask.sum())
        if count not in options:
            continue
        if center_required and not bool(mask[CENTER]):
            continue
        if bbox(mask) != SPRITE_BBOX:
            continue
        _, n_components = label_mask(mask, connectivity=4)
        if n_components != 1:
            continue
        if center_required and not any(mask[p] for p in CENTER_NEIGHBORS):
            continue
        options[count].append(mask)
    return options

MARKED_MASK_OPTIONS = _gen_mask_options(center_required=True)
PLAIN_MASK_OPTIONS = _gen_mask_options(center_required=False)
SOLID_BOX = np.ones(SPRITE_SHAPE, dtype=int)

def _sample_sprites(colors: list[int], marked_idx: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Render patches and the expected marked-sprite output."""
    sprites = []
    expected = None
    for i, color in enumerate(colors):
        options = MARKED_MASK_OPTIONS if i == marked_idx else PLAIN_MASK_OPTIONS
        pixel_count = random.randint(*PIXEL_COUNT_BOUNDS)
        mask = random.choice(options[pixel_count]).copy()
        patch = np.full(SPRITE_SHAPE, BG, dtype=int)
        patch[mask] = int(color)
        if i == marked_idx:
            patch[CENTER] = TEAL
            expected = np.full(SPRITE_SHAPE, BG, dtype=int)
            expected[mask] = int(color)
        sprites.append(patch)
    assert expected is not None
    return sprites, expected

def _place_sprites(sprites: list[np.ndarray]) -> np.ndarray | None:
    """Place sprite boxes with one-cell separation."""
    grid = np.full(CANVAS_SHAPE, BG, dtype=int)
    occupied = np.full(CANVAS_SHAPE, BG, dtype=int)
    order = list(range(len(sprites)))
    random.shuffle(order)
    try:
        for i in order:
            r, c = random_free_location(
                occupied, SOLID_BOX, transparent=BG,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
            paste(occupied, SOLID_BOX, r, c)
            paste(grid, sprites[i], r, c, transparent=BG)
    except ValueError:
        return None
    return grid

def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.randint(*SPRITE_COUNT_BOUNDS)
    colors = random.sample(FOREGROUND_COLORS, n_sprites)
    marked_idx = random.randrange(n_sprites)
    sprites, expected = _sample_sprites(colors, marked_idx)
    grid = _place_sprites(sprites)
    if grid is None:
        return None
    try:
        return grid if np.array_equal(main(grid), expected) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid sample")
