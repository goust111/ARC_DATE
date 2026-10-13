import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each small foreground sprite horizontally across the canvas without flipping the sprite itself.

    Input structure
    - The dominant color is the background.
    - All foreground pixels form 4-connected multicolor sprites.
    - Each sprite has a 2x2 bounding box, possibly with background cells inside that box.

    Rule
    - Start with an empty background canvas of the same shape.
    - For each sprite, keep its row position and internal pixel layout unchanged.
    - Move its bounding box to the horizontally mirrored column: new_c0 = grid_width - sprite_width - old_c0.

    Key insight
    - The diagonal arrangement changes by mirroring sprite positions, not by reflecting individual rows or color order inside each sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = int(get_bg(g))
    sprites = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert sprites, "expected foreground sprites"

    out = np.full_like(g, bg)
    for sprite in sprites:
        bbox = sprite["bbox"]
        r0, c0, _r1, _c1 = map(int, bbox)
        h = bbox_h(bbox)
        w = bbox_w(bbox)
        assert h == 2 and w == 2, "expected each foreground sprite to occupy a 2x2 bbox"

        patch = crop_bbox(g, bbox)
        mask = patch != bg
        assert np.any(mask), "sprite bbox must contain foreground"

        dst_c0 = width - w - c0
        assert 0 <= dst_c0 <= width - w, "mirrored sprite must stay in bounds"
        target = out[r0 : r0 + h, dst_c0 : dst_c0 + w]
        assert not np.any((target != bg) & mask), "mirrored sprites must not overlap"
        paste(out, patch, r0, dst_c0, transparent=bg)

    return out

# --------------- generator (o5x) ---------------
# Canvas and repeated sprite geometry.
BG, SPRITE_SIZE = 0, 2
SPRITE_COUNT_RANGE = (5, 8)

# Palette and patch sampling.
FG_COLORS = np.arange(1, 10)
SPRITE_COLOR_COUNT, BINARY_CHOICE_COUNT = 2, 2
SPRITE_CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Oracle validation retry budget.
MAX_ATTEMPTS = 300


def _gen_sprite_patch(rng, colors):
    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), BG, dtype=int)
    if rng.integers(BINARY_CHOICE_COUNT) == 0:
        missing = int(rng.integers(len(SPRITE_CELLS)))
        fg_coords = [coord for i, coord in enumerate(SPRITE_CELLS) if i != missing]
    else:
        fg_coords = list(SPRITE_CELLS)

    color_slots = list(colors)
    color_slots.extend(rng.choice(colors, size=len(fg_coords) - len(colors)).tolist())
    rng.shuffle(color_slots)

    for (r, c), color in zip(fg_coords, color_slots):
        patch[r, c] = int(color)

    return patch


def _gen_attempt():
    rng = np.random.default_rng()
    n = int(rng.integers(*SPRITE_COUNT_RANGE))
    size = SPRITE_SIZE * n
    inp = np.full((size, size), BG, dtype=int)
    expected = np.full_like(inp, BG)

    colors = tuple(map(int, rng.choice(FG_COLORS, size=SPRITE_COLOR_COUNT, replace=False).tolist()))
    patch = _gen_sprite_patch(rng, colors)
    starts_left = bool(rng.integers(BINARY_CHOICE_COUNT))

    for k in range(n):
        r0 = SPRITE_SIZE * k
        c0 = SPRITE_SIZE * k if starts_left else size - SPRITE_SIZE - SPRITE_SIZE * k
        paste(inp, patch, r0, c0, transparent=None)
        paste(expected, patch, r0, size - SPRITE_SIZE - c0, transparent=BG)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8ee62060")
