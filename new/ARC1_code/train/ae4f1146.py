from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, crop_bbox, find_components, get_bg, paste, random_free_location
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the 3x3 sprite with the most pixels of the rare foreground color.

    Input structure
    - A dominant background contains several separated 4-connected foreground sprites.
    - Exactly two foreground colors fill each sprite, and every sprite is a solid 3x3 rectangle.
    - One foreground color is globally rarer; it acts as the marked color.

    Rule
    - Infer the marked color as the unique least frequent non-background color.
    - Extract each foreground component as its 3x3 sprite.
    - Return the unique sprite with the largest count of marked-color pixels.

    Key insight
    - The output is a crop of one existing sprite, selected by marker density rather than by location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    fg_counts = Counter(map(int, g[g != bg].reshape(-1)))
    min_count = min(fg_counts[color] for color in fg_colors)
    marker_colors = [color for color in fg_colors if fg_counts[color] == min_count]
    assert len(marker_colors) == 1, "expected a unique rare foreground color"
    marker = int(marker_colors[0])

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(components) >= 2, "expected multiple candidate sprites"

    scored: list[tuple[int, np.ndarray]] = []
    for component in components:
        box = component["bbox"]
        assert (bbox_h(box), bbox_w(box)) == (3, 3), "each sprite must have a 3x3 bbox"
        assert int(component["area"]) == 9, "each sprite must fill its 3x3 bbox"

        sprite = crop_bbox(g, box)
        assert np.all(sprite != bg), "sprite bbox must contain only foreground colors"
        scored.append((int(np.count_nonzero(sprite == marker)), sprite))

    best_score = max(score for score, _sprite in scored)
    winners = [sprite for score, sprite in scored if score == best_score]
    assert len(winners) == 1, "expected a unique sprite with the most marked pixels"
    return winners[0]

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SHAPE, BACKGROUND_COLOR = (9, 9), 0
DOMINANT_COLOR, MARKER_COLOR = 8, 1

# Sprite and marker-count constraints.
SPRITE_SHAPE, SPRITE_COUNT = (3, 3), 4
SPRITE_CELLS = SPRITE_SHAPE[0] * SPRITE_SHAPE[1]
WINNER_MARKER_COUNTS = (4, 5, 6)
RUNNER_MIN_MARKERS, TOTAL_MARKER_MIN, TOTAL_MARKER_MAX = 3, 9, 14

# Placement and retry behavior.
FREE_LOCATION_KWARGS = {"transparent": BACKGROUND_COLOR, "padding": 1, "padding_connectivity": 4}
MAX_GENERATION_ATTEMPTS = 300


def _build_marker_count_sets():
    valid = []
    for winner in WINNER_MARKER_COUNTS:
        for combo in itertools.combinations(range(winner), SPRITE_COUNT - 1):
            counts = list(combo) + [winner]
            total = sum(counts)
            if max(combo) >= RUNNER_MIN_MARKERS and TOTAL_MARKER_MIN <= total <= TOTAL_MARKER_MAX:
                valid.append(counts)
    return valid


MARKER_COUNT_SETS = _build_marker_count_sets()


def _gen_sprite_with_marker_count(marker_count):
    sprite = np.full(SPRITE_SHAPE, DOMINANT_COLOR, dtype=int)
    sprite.flat[random.sample(range(SPRITE_CELLS), int(marker_count))] = MARKER_COLOR
    return sprite


def _gen_sprites():
    counts = random.choice(MARKER_COUNT_SETS)
    winner_count = max(counts)
    target = _gen_sprite_with_marker_count(winner_count)

    sprites = [target] + [_gen_sprite_with_marker_count(count) for count in counts if count != winner_count]
    random.shuffle(sprites)
    return sprites, target


def _gen_attempt():
    sprites, target = _gen_sprites()
    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    for sprite in sprites:
        try:
            r0, c0 = random_free_location(grid, sprite, **FREE_LOCATION_KWARGS)
        except ValueError:
            return None
        paste(grid, sprite, r0, c0)

    if not np.array_equal(main(grid), target):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ae4f1146 input")
