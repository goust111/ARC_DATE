from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous, paste, random_free_location
import itertools
import random


def _component_mask(grid: np.ndarray, component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    color = int(component["color"])
    return grid[r0 : r1 + 1, c0 : c1 + 1] == color


def _candidate_for_placeholder(
    g: np.ndarray, bg: int, color: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    targets = find_components(g, bg=bg, colors=[color], connectivity=4, monochromatic=True)
    if len(targets) < 2:
        return None

    source_mask = (g != bg) & (g != color)
    source_components = components_mask(source_mask, connectivity=4)
    if len(source_components) != 1:
        return None

    r0, c0, r1, c1 = bbox_coords(source_components[0])
    sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    sprite_mask = sprite != bg
    assert np.any(sprite_mask), "source sprite must be nonempty"
    assert not np.any(sprite == color), "source sprite cannot contain the placeholder color"

    for target in targets:
        target_mask = _component_mask(g, target)
        if target_mask.shape != sprite_mask.shape:
            return None
        if not np.array_equal(target_mask, sprite_mask):
            return None

    return sprite, sprite_mask, source_mask, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace every repeated placeholder object with the matching colored sprite and erase the original sprite.

    Input structure:
    - The dominant background surrounds one connected polychromatic source sprite.
    - A separate monochrome placeholder color forms multiple 4-connected target objects.
    - Each target object has the same normalized mask as the source sprite.

    Rule:
    - Infer the placeholder color as the unique foreground color whose components
      match the mask of the remaining connected source sprite.
    - Clear the original source sprite to background.
    - Paint the source sprite's colors onto each matching placeholder object at
      the target object's bbox position.

    Key insight:
    The placeholder objects are unlabeled copies of the source mask; their pixels inherit colors from the corresponding cells of the source sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]] = []
    for color in set(map(int, g.flatten())) - {bg}:
        candidate = _candidate_for_placeholder(g, bg, int(color))
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected a unique placeholder color"

    sprite, sprite_mask, source_mask, targets = candidates[0]
    out = g.copy()
    out[source_mask] = bg

    for target in targets:
        r0, c0, r1, c1 = map(int, target["bbox"])
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == sprite_mask.shape, "target bbox must match source sprite bbox"
        patch[sprite_mask] = sprite[sprite_mask]

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GRID_SHAPE, GEN_BG, PLACEHOLDER_COLOR = (10, 10), 0, 8
SPRITE_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]

# Sprite mask bounds.
SPRITE_HEIGHTS, SPRITE_WIDTHS = (2, 3), (2, 3, 4)
SPRITE_MIN_PIXELS, SPRITE_MAX_PIXELS = 4, 7
SPRITE_MIN_DENSITY = 0.58
SPRITE_MIN_COLORS, SPRITE_MAX_COLORS = 2, 4

# Object placement and retry limits.
OBJECT_COUNT = 4
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 4
MAX_GENERATION_ATTEMPTS = 300

def _gen_mask_options(h: int, w: int, count: int) -> list[np.ndarray]:
    out = []
    cells = [(r, c) for r in range(h) for c in range(w)]
    for combo in itertools.combinations(cells, count):
        rows = {r for r, _ in combo}
        cols = {c for _, c in combo}
        if rows != set(range(h)) or cols != set(range(w)):
            continue
        mask = np.zeros((h, w), dtype=bool)
        rr, cc = zip(*combo)
        mask[rr, cc] = True
        if is_contiguous(mask.astype(int), background=0, connectivity=4):
            out.append(mask)
    return out


def _build_sprite_choices() -> list[tuple[int, list[np.ndarray]]]:
    choices = []
    for h in SPRITE_HEIGHTS:
        for w in SPRITE_WIDTHS:
            lo = max(SPRITE_MIN_PIXELS, int(np.ceil(SPRITE_MIN_DENSITY * h * w)))
            hi = min(SPRITE_MAX_PIXELS, h * w)
            for count in range(lo, hi + 1):
                options = _gen_mask_options(h, w, count)
                if options:
                    choices.append((count, options))
    return choices


SPRITE_CHOICES = _build_sprite_choices()


def _gen_sprite() -> np.ndarray:
    count, options = random.choice(SPRITE_CHOICES)
    mask = random.choice(options)
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    random.shuffle(coords)

    color_count = random.randint(SPRITE_MIN_COLORS, min(SPRITE_MAX_COLORS, count))
    palette = random.sample(SPRITE_COLOR_POOL, color_count)
    sprite = np.zeros(mask.shape, dtype=int)
    for i, (r, c) in enumerate(coords):
        sprite[r, c] = palette[i] if i < len(palette) else random.choice(palette)
    return sprite


def _gen_attempt() -> np.ndarray | None:
    sprite = _gen_sprite()
    h, w = sprite.shape
    full_box = np.ones((h, w), dtype=int)

    bbox_canvas = np.zeros(GRID_SHAPE, dtype=int)
    positions = []
    try:
        for _ in range(OBJECT_COUNT):
            r, c = random_free_location(
                bbox_canvas, full_box, transparent=GEN_BG, padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
            paste(bbox_canvas, full_box, r, c, transparent=GEN_BG)
            positions.append((r, c))
    except ValueError:
        return None

    random.shuffle(positions)
    source, targets = positions[0], positions[1:]

    colored_scene = np.zeros(GRID_SHAPE, dtype=int)
    for r, c in positions:
        paste(colored_scene, sprite, r, c, transparent=GEN_BG)

    out = colored_scene.copy()
    sr, sc = source
    out[sr : sr + h, sc : sc + w][sprite != GEN_BG] = GEN_BG

    inp = colored_scene.copy()
    placeholder = np.where(sprite != GEN_BG, PLACEHOLDER_COLOR, GEN_BG)
    for r, c in targets:
        paste(inp, placeholder, r, c, transparent=GEN_BG)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 321b1fc6 input")
