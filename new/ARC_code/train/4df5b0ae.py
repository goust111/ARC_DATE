import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste, random_free_location, random_sprite
import random


def _is_full_grid_frame(component: dict, shape: tuple[int, int]) -> bool:
    H, W = shape
    r0, c0, r1, c1 = component["bbox"]
    if (r0, c0, r1, c1) != (0, 0, H - 1, W - 1):
        return False
    perimeter_area = 2 * H + 2 * W - 4
    return int(component["area"]) == perimeter_area and all(
        r in (0, H - 1) or c in (0, W - 1) for r, c in component["coords"]
    )


def _component_sprite(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in component["coords"]:
        sprite[r - r0, c - c0] = int(component["color"])
    return sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Pack the colored sprites along the bottom edge from smallest to largest.

    Input structure
    The grid has a dominant background color and several monochrome 4-connected
    foreground sprites; a full-grid one-cell frame may surround the scene.

    Rule
    Remove any full-grid rectangular frame, sort the remaining sprites by pixel
    area from smallest to largest, and paste their bounding-box sprites left to
    right with each sprite bottom-aligned.

    Key insight
    The original locations do not determine the final slots; each sprite's area
    gives the packing order, while its shape and color are preserved.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    sprites: list[tuple[int, np.ndarray]] = []
    for component in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if _is_full_grid_frame(component, g.shape):
            continue
        sprites.append((int(component["area"]), _component_sprite(component, bg)))

    assert sprites
    areas = [area for area, _sprite in sprites]
    assert len(areas) == len(set(areas))

    out = np.full_like(g, bg)
    c = 0
    for _area, sprite in sorted(sprites, key=lambda item: item[0]):
        h, w = sprite.shape
        assert h <= H and c + w <= W
        paste(out, sprite, H - h, c, transparent=bg)
        c += w
    return out

# --------------- generator (att1, Claude) ---------------

BG = 7
ALL_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# Canvas and sprite bounds.
GRID_HEIGHT, GRID_WIDTH = 10, 10
SPRITE_MIN_AREA, SPRITE_MAX_AREA = 1, 12
SPRITE_MAX_HEIGHT, SPRITE_MAX_WIDTH = 6, 4
MIN_SPRITE_DENSITY = 0.5

# Sprite sampling and validation.
SPRITE_COUNT_CHOICES = [3, 4]
SPRITE_CONNECTIVITY = 4
SPRITE_SYMMETRY = 'not_symmetric'
SPRITE_RETRY_LIMIT = 20

# Frame, placement, and generation retries.
FRAME_PROBABILITY = 0.3
FRAME_BORDER, NO_FRAME_BORDER = 1, 0
GENERATION_RETRY_LIMIT = 300

VALID_SPRITE_SHAPES = {
    area: [(h, w) for h in range(1, SPRITE_MAX_HEIGHT + 1)
           for w in range(1, SPRITE_MAX_WIDTH + 1)
           if h * w >= area and area / (h * w) >= MIN_SPRITE_DENSITY]
    for area in range(SPRITE_MIN_AREA, SPRITE_MAX_AREA + 1)
}


def _make_sprite(area, color):
    valid_shapes = list(VALID_SPRITE_SHAPES.get(area, []))
    if not valid_shapes:
        return None

    random.shuffle(valid_shapes)
    for h, w in valid_shapes:
        bbox_area = h * w
        if bbox_area == area:
            return np.full((h, w), color, dtype=int)
        try:
            sprite = random_sprite(h, w, density=area / bbox_area,
                                   color_palette=[color], connectivity=SPRITE_CONNECTIVITY,
                                   background=BG, symmetry=SPRITE_SYMMETRY)
            actual_area = int(np.sum(sprite != BG))
            if actual_area == area and is_contiguous(sprite, background=BG,
                                                     connectivity=SPRITE_CONNECTIVITY):
                return sprite
        except Exception:
            continue
    return None


def _sample_sprite_specs():
    n_sprites = random.choice(SPRITE_COUNT_CHOICES)
    possible_areas = list(range(SPRITE_MIN_AREA, SPRITE_MAX_AREA + 1))
    random.shuffle(possible_areas)
    areas = sorted(possible_areas[:n_sprites])
    colors = [random.choice(ALL_COLORS) for _ in range(n_sprites)]
    return areas, colors


def _make_sprites(areas, colors):
    """Build all sprites, preserving area order; return None if any shape fails."""
    sprites = []
    for area, color in zip(areas, colors):
        sprite = None
        for _ in range(SPRITE_RETRY_LIMIT):
            sprite = _make_sprite(area, color)
            if sprite is not None:
                break
        if sprite is None:
            return None
        sprites.append(sprite)
    return sprites


def _fits_packed_output(sprites):
    total_width = sum(sprite.shape[1] for sprite in sprites)
    max_height = max(sprite.shape[0] for sprite in sprites)
    return total_width <= GRID_WIDTH and max_height <= GRID_HEIGHT


def _pack_output(sprites):
    output = np.full((GRID_HEIGHT, GRID_WIDTH), BG, dtype=int)
    col = 0
    for sprite in sprites:
        sprite_height, sprite_width = sprite.shape
        paste(output, sprite, GRID_HEIGHT - sprite_height, col, transparent=BG)
        col += sprite_width
    return output


def _maybe_add_frame(grid, sprites):
    """Mutate grid with an optional perimeter frame; return the active border."""
    use_frame = random.random() < FRAME_PROBABILITY
    if not use_frame:
        return NO_FRAME_BORDER

    sprite_colors = set(int(sprite[sprite != BG].flat[0]) for sprite in sprites)
    frame_candidates = [color for color in ALL_COLORS if color not in sprite_colors]
    if not frame_candidates:
        return NO_FRAME_BORDER

    frame_color = random.choice(frame_candidates)
    grid[0, :] = frame_color
    grid[-1, :] = frame_color
    grid[:, 0] = frame_color
    grid[:, -1] = frame_color
    return FRAME_BORDER


def _place_sprites(grid, sprites, border):
    """Place sprites in-place; return False if no legal location is found."""
    for sprite in sprites:
        try:
            row, col = random_free_location(grid, sprite, transparent=BG, border=border)
            paste(grid, sprite, row, col, transparent=BG)
        except ValueError:
            return False
    return True


def _passes_oracle(inp, output):
    try:
        return np.array_equal(main(inp), output)
    except Exception:
        return False


def _gen_attempt():
    areas, colors = _sample_sprite_specs()
    sprites = _make_sprites(areas, colors)
    if sprites is None or not _fits_packed_output(sprites):
        return None

    output = _pack_output(sprites)
    inp = np.full((GRID_HEIGHT, GRID_WIDTH), BG, dtype=int)
    border = _maybe_add_frame(inp, sprites)
    if not _place_sprites(inp, sprites, border):
        return None
    if _passes_oracle(inp, output):
        return inp
    return None


def generate_input():
    for _ in range(GENERATION_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRY_LIMIT} attempts")
