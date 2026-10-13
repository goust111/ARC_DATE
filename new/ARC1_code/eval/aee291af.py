from collections import Counter

import numpy as np

from arc_utils import colors_present, get_bg, paste, random_free_location
import math
import random


def _framed_rectangles_for_color(g: np.ndarray, bg: int, frame_color: int) -> list[np.ndarray]:
    height, width = g.shape
    sprites: list[np.ndarray] = []

    for r0 in range(height - 2):
        for c0 in range(width - 2):
            for r1 in range(r0 + 3, height + 1):
                for c1 in range(c0 + 3, width + 1):
                    patch = g[r0:r1, c0:c1]
                    border = np.concatenate(
                        (
                            patch[0, :],
                            patch[-1, :],
                            patch[1:-1, 0],
                            patch[1:-1, -1],
                        )
                    )
                    if not np.all(border == frame_color):
                        continue

                    interior = patch[1:-1, 1:-1]
                    if np.all(interior != bg) and np.any(interior != frame_color):
                        sprites.append(patch.copy())

    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the largest clean framed sprite whose foreground pattern is not duplicated.

    Input structure
    - The grid has one dominant background color and two foreground colors.
    - The foreground contains several same-size rectangular sprites with a one-cell frame.
    - Extra foreground clutter may touch the outside of a sprite, and smaller framed subpatterns may occur inside repeated sprites.

    Rule
    - Find clean rectangles whose border is one foreground color and whose interior contains no background.
    - Keep only the largest such rectangles; these are the full source sprites.
    - Among those sprites, return the single full sprite pattern that occurs exactly once.

    Key insight
    The surrounding clutter is irrelevant once the maximal intact frames are detected; the task is an odd-one-out selection among the clean sprite patterns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    by_frame = {}
    for color in colors:
        sprites = _framed_rectangles_for_color(g, bg, int(color))
        if sprites:
            by_frame[int(color)] = sprites
    assert len(by_frame) == 1, "expected exactly one foreground color to form frames"

    sprites = next(iter(by_frame.values()))
    assert len(sprites) >= 3, "expected multiple framed sprites"

    max_area = max(sprite.shape[0] * sprite.shape[1] for sprite in sprites)
    sprites = [sprite for sprite in sprites if sprite.shape[0] * sprite.shape[1] == max_area]
    assert len(sprites) >= 3, "expected multiple largest framed sprites"
    assert len({sprite.shape for sprite in sprites}) == 1, "expected largest sprites to have one shape"

    keys = [tuple(map(tuple, sprite.tolist())) for sprite in sprites]
    counts = Counter(keys)
    assert any(count > 1 for count in counts.values()), "expected a repeated sprite pattern"

    unique_keys = [key for key, count in counts.items() if count == 1]
    assert len(unique_keys) == 1, "expected exactly one unique sprite pattern"

    matches = [sprite for sprite, key in zip(sprites, keys) if key == unique_keys[0]]
    assert len(matches) == 1, "unique pattern should correspond to one sprite"
    return matches[0].copy()

# --------------- generator (o5x) ---------------

# Canvas and object size constraints.
GRID_SIDE_OPTIONS = (16, 17, 18)
SPRITE_COUNT_RANGE = (3, 5)
SPRITE_SIZE_OPTIONS = (4, 5)
THREE_SPRITE_SIZE = 5
SPRITE_PADDING = 1
# Pattern, density, and palette controls.
MIN_FILL_CELLS = 2
DISTINCT_MASK_RETRIES = 20
CLUTTER_DENSITY_MIN, CLUTTER_DENSITY_MAX = 0.09, 0.14
FOREGROUND_DENSITY_MIN, FOREGROUND_DENSITY_MAX = 0.35, 0.50
COLOR_COUNT = 10
# Clutter placement and retry controls.
TOUCH_CLUSTER_COUNT_RANGE = (1, 2)
TOUCH_CLUSTER_SIZE_WEIGHTS = (1, 1, 2)
REQUIRED_ISOLATED_CLUSTERS = 2
BULK_CLUSTER_SIZE_ORDER = (3, 2, 1)
ALT_BULK_CLUSTER_SIZE_ORDER = (2, 3, 1)
ALT_BULK_CLUSTER_PROB = 0.2
GENERATION_RETRIES = 300


def _gen_neighbors4(r, c, h, w):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _gen_neighbors8(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _gen_random_interior_mask(d):
    n = d * d
    fill_count = random.randint(MIN_FILL_CELLS, n - 1)
    mask = np.zeros(n, dtype=bool)
    mask[random.sample(range(n), fill_count)] = True
    return mask.reshape((d, d))


def _gen_distinct_interior_masks(d):
    repeated = _gen_random_interior_mask(d)
    for _ in range(DISTINCT_MASK_RETRIES):
        unique = _gen_random_interior_mask(d)
        if not np.array_equal(repeated, unique):
            return repeated, unique

    unique = repeated.copy()
    frame_cells = list(map(tuple, np.argwhere(~unique)))
    fill_cells = list(map(tuple, np.argwhere(unique)))
    r, c = random.choice(frame_cells)
    unique[r, c] = True
    if unique.all():
        r, c = random.choice(fill_cells)
        unique[r, c] = False
    return repeated, unique


def _gen_make_sprite(size, frame_color, fill_color, interior_mask):
    sprite = np.full((size, size), frame_color, dtype=int)
    sprite[1:-1, 1:-1][interior_mask] = fill_color
    return sprite


def _gen_scene_parameters():
    """Choose sprite count/size, canvas side, and a feasible clutter count."""
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    sprite_size = THREE_SPRITE_SIZE if sprite_count == SPRITE_COUNT_RANGE[0] else random.choice(SPRITE_SIZE_OPTIONS)
    sprite_cells = sprite_count * sprite_size * sprite_size

    feasible = []
    for side in GRID_SIDE_OPTIONS:
        total = side * side
        min_clutter = math.ceil(CLUTTER_DENSITY_MIN * total)
        max_clutter = math.floor(CLUTTER_DENSITY_MAX * total)
        min_clutter = max(min_clutter, math.ceil(FOREGROUND_DENSITY_MIN * total - sprite_cells))
        max_clutter = min(max_clutter, math.floor(FOREGROUND_DENSITY_MAX * total - sprite_cells))
        if min_clutter <= max_clutter:
            feasible.append((side, min_clutter, max_clutter))

    if not feasible:
        return None

    side, min_clutter, max_clutter = random.choice(feasible)
    clutter_cells = random.randint(min_clutter, max_clutter)
    return side, sprite_size, sprite_count, clutter_cells


def _gen_valid_cluster(grid, coords, bg, frame_color, sprite_mask, touch_sprite):
    h, w = grid.shape
    coord_set = set(coords)
    touches_sprite = False

    if len(coord_set) != len(coords):
        return False

    for r, c in coords:
        if grid[r, c] != bg:
            return False
        for nr, nc in _gen_neighbors8(r, c, h, w):
            if (nr, nc) in coord_set:
                continue
            if sprite_mask[nr, nc]:
                touches_sprite = True
                if touch_sprite is False:
                    return False
            elif grid[nr, nc] == frame_color:
                return False

    if touch_sprite is True and not touches_sprite:
        return False
    return True


def _gen_cluster_options(grid, bg, frame_color, sprite_mask, size, touch_sprite):
    h, w = grid.shape
    options = set()
    bg_cells = [tuple(x) for x in np.argwhere(grid == bg)]

    if size == 1:
        options = {(cell,) for cell in bg_cells}
    elif size == 2:
        for r, c in bg_cells:
            for nr, nc in _gen_neighbors4(r, c, h, w):
                if grid[nr, nc] == bg:
                    options.add(tuple(sorted(((r, c), (nr, nc)))))
    else:
        for r, c in bg_cells:
            for nr, nc in _gen_neighbors4(r, c, h, w):
                if grid[nr, nc] != bg:
                    continue
                base = {(r, c), (nr, nc)}
                frontier = set()
                for br, bc in base:
                    frontier.update(_gen_neighbors4(br, bc, h, w))
                for tr, tc in frontier:
                    if (tr, tc) not in base and grid[tr, tc] == bg:
                        options.add(tuple(sorted((*base, (tr, tc)))))

    return [
        coords
        for coords in options
        if _gen_valid_cluster(grid, coords, bg, frame_color, sprite_mask, touch_sprite)
    ]


def _gen_place_clutter_cluster(grid, coords, frame_color):
    rows, cols = zip(*coords)
    grid[rows, cols] = frame_color


def _gen_add_clutter(grid, bg, frame_color, sprite_mask, target_cells):
    """Place exact-count frame-color clutter while preserving sprite detectability."""
    placed = 0
    touch_clusters = random.randint(*TOUCH_CLUSTER_COUNT_RANGE)

    for _ in range(touch_clusters):
        if placed >= target_cells:
            break
        size = min(random.choice(TOUCH_CLUSTER_SIZE_WEIGHTS), target_cells - placed)
        options = _gen_cluster_options(grid, bg, frame_color, sprite_mask, size, True)
        if not options:
            continue
        coords = random.choice(options)
        _gen_place_clutter_cluster(grid, coords, frame_color)
        placed += len(coords)

    if placed == 0:
        return False

    for _ in range(REQUIRED_ISOLATED_CLUSTERS):
        if placed >= target_cells:
            break
        options = _gen_cluster_options(grid, bg, frame_color, sprite_mask, 1, False)
        if not options:
            return False
        coords = random.choice(options)
        _gen_place_clutter_cluster(grid, coords, frame_color)
        placed += 1

    while placed < target_cells:
        remaining = target_cells - placed
        if remaining >= max(BULK_CLUSTER_SIZE_ORDER) and random.random() < ALT_BULK_CLUSTER_PROB:
            sizes = ALT_BULK_CLUSTER_SIZE_ORDER
        else:
            sizes = BULK_CLUSTER_SIZE_ORDER
        sizes = [s for s in sizes if s <= remaining]

        chosen = None
        for size in sizes:
            options = _gen_cluster_options(grid, bg, frame_color, sprite_mask, size, False)
            if options:
                chosen = random.choice(options)
                break
        if chosen is None:
            return False

        _gen_place_clutter_cluster(grid, chosen, frame_color)
        placed += len(chosen)

    return True


def _gen_sprite_variants(sprite_size, frame_color, fill_color):
    repeated_mask, unique_mask = _gen_distinct_interior_masks(sprite_size - 2)
    repeated_sprite = _gen_make_sprite(sprite_size, frame_color, fill_color, repeated_mask)
    unique_sprite = _gen_make_sprite(sprite_size, frame_color, fill_color, unique_mask)
    return repeated_sprite, unique_sprite


def _gen_place_sprites(grid, sprites, bg, sprite_size):
    """Paste non-overlapping sprites and return their occupied mask, or None."""
    sprite_mask = np.zeros(grid.shape, dtype=bool)
    for sprite in sprites:
        try:
            r, c = random_free_location(grid, sprite, transparent=bg, padding=SPRITE_PADDING)
        except ValueError:
            return None
        paste(grid, sprite, r, c)
        sprite_mask[r : r + sprite_size, c : c + sprite_size] = True
    return sprite_mask


def _gen_is_valid_solution(grid, bg, frame_color, fill_color, unique_sprite):
    """Mirror the original rejection checks and solver-oracle validation."""
    if np.sum(grid == bg) <= max(np.sum(grid == frame_color), np.sum(grid == fill_color)):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, unique_sprite)


def _gen_attempt():
    params = _gen_scene_parameters()
    if params is None:
        return None
    side, sprite_size, sprite_count, clutter_cells = params

    bg, frame_color, fill_color = random.sample(range(COLOR_COUNT), 3)
    repeated_sprite, unique_sprite = _gen_sprite_variants(sprite_size, frame_color, fill_color)

    grid = np.full((side, side), bg, dtype=int)
    sprites = [repeated_sprite.copy() for _ in range(sprite_count - 1)]
    sprites.insert(random.randrange(sprite_count), unique_sprite.copy())
    sprite_mask = _gen_place_sprites(grid, sprites, bg, sprite_size)
    if sprite_mask is None:
        return None
    if not _gen_add_clutter(grid, bg, frame_color, sprite_mask, clutter_cells):
        return None

    if not _gen_is_valid_solution(grid, bg, frame_color, fill_color, unique_sprite):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid aee291af input")
