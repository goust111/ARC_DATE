# Variant of 58490d8a
# Dimension: D5 — 圆点放置间距 (transform)
# Change: 相邻圆点之间间隔1个背景像素（步长为2），即放置在关键列、关键列+2、关键列+4等位置... → 间距逐渐增大

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the D5 variant
# a) Concretization and behavior trace:
#    - The input still contains one black legend rectangle with colored key pixels in column 1,
#      plus scattered monochrome sprites outside the box.
#    - The output is still a black patch with one tally row per key color, but the tally no longer
#      uses a fixed "+2" stride. If a row needs 4 dots and the key column is 1, the old positions
#      were 1,3,5,7. The new positions are 1,3,6,10 because the background gaps grow as 1,2,3.
#    - This changes only the rendering stage of the counted tallies. Object detection, legend
#      detection, and color-to-row matching remain unchanged.
# b) Complexity calibration:
#    - If every sampled count were only 1 or 2, the new rule would look identical to the old rule.
#      The generator therefore guarantees at least one count >= 3 in every sample.
#    - Counts still stay in the original 1..5 range and the number of key colors stays 2..4, so the
#      puzzle remains the same family instead of introducing a second concept.
# c) Hidden assumptions review against the original main():
#    - Legend detection via the largest black component with interior non-black pixels still works,
#      because the input legend drawing is unchanged.
#    - The original `key_col + 2 * i` assumed constant spacing. That assumption is false now and is
#      replaced with a cumulative offset helper.
#    - The original generator width `2*max_count + 1 + slack` relied on the fixed stride as well.
#      This must be expanded so the rightmost dot of the growing-gap sequence always fits.
# d) Placement legality:
#    - Output dots are drawn only inside the legend patch. The new width formula reserves one black
#      trailing column after the farthest possible dot, plus the original random slack.
#    - Sprite placement outside the legend is unchanged: random_free_location still enforces
#      non-overlap and an 8-connected buffer.
# e) Visual observability:
#    - The generator enforces `max(counts) >= 3`, so every sample has at least one row where the
#      third dot lands farther right than it would under the old rule.
#    - With 3, 4, 5 dots the visible column patterns become 1,3,6 / 1,3,6,10 / 1,3,6,10,15, which
#      is easy to distinguish from a uniform arithmetic spacing.
# f) Visual cleanliness:
#    - Only one black legend area is used for the tally, so the wider spacing does not create new
#      overlaps. The rest of the grid keeps the same sparse sprite layout.
# g) Color inference:
#    - Colors are not reordered or transformed. Each row is still anchored by its key pixel, so the
#      new reasoning burden is only the spacing pattern, not an extra color mapping.
# h) New solver assumptions and generator support:
#    - The solver now assumes the legend is wide enough for the cumulative positions implied by the
#      true count. `_legend_shape()` guarantees this for the sampled maximum count.
#    - The solver assumes dots start at the key pixel column and then move strictly rightward by a
#      deterministic count-index rule. `_build_expected()` and the width formula use the same helper.


def _tally_offset(index: int) -> int:
    """Column offset of the zero-based tally dot under growing inter-dot gaps."""
    idx = int(index)
    return idx * (idx + 3) // 2


def _tally_col(key_col: int, index: int) -> int:
    """Absolute column for the zero-based tally dot."""
    return int(key_col) + _tally_offset(index)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and encode the totals inside a legend box.

    Input structure:
    - A black rectangular legend patch contains one colored key pixel per tally row in its first non-black column.
    - Outside that patch, separate monochrome sprites of the key colors are scattered across the background.

    Rule:
    - Count the connected components of each key color outside the legend box.
    - Repaint the legend area as black and place one dot per counted object on the matching key row, starting at the key column and then shifting right with increasing gaps: +2, then +3, then +4, ...

    Key insight: The legend still tells you which row belongs to which color; the new twist is that the tally columns follow a cumulative spacing sequence rather than a constant stride.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    best = None  # (area, bbox)
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            cc = _tally_col(key_col, i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
VISIBLE_MAX_COUNT_MIN = 3
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 300


def _randint(bounds: tuple[int, int]) -> int:
    """Return an inclusive randint from a configured (low, high) pair."""
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    """Return connected masks that span all four sides of the sprite bbox."""
    masks = []
    for cells in combinations(range(h * w), area):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        spans_edges = mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
        if not spans_edges:
            continue
        _, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n == 1:
            masks.append(mask)
    return tuple(masks)


def _make_sprite(color: int, bg: int) -> np.ndarray:
    """Sample one connected same-color sprite drawn on the background color."""
    area = _randint(SPRITE_AREA_RANGE)
    dims = []
    for h, w in SPRITE_BBOXES:
        if h * w >= area and _valid_masks(h, w, area):
            dims.append((h, w))
    h, w = random.choice(dims)
    mask = random.choice(_valid_masks(h, w, area))
    sprite = np.full((h, w), int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _sample_palette() -> tuple[int, list[int], int | None]:
    """Choose background, key colors, and optional distractor color."""
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    """Sample outside-component counts while forcing at least one visible growing-gap row."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= VISIBLE_MAX_COUNT_MIN:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    last_dot_col = _tally_col(KEY_COLUMN, max_count - 1)
    bw = last_dot_col + LEGEND_WIDTH_TAIL + 1 + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, _tally_col(KEY_COLUMN, j)] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the key-column pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
    """Sample a grid and paste the legend away from all canvas edges."""
    bh, bw = map(int, legend.shape)
    H = _randint(GRID_HEIGHT_RANGE)
    W = _randint(GRID_WIDTH_RANGE)
    min_h = bh + 2 * LEGEND_EDGE_MARGIN
    min_w = bw + 2 * LEGEND_EDGE_MARGIN
    if H < min_h or W < min_w:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    r0 = random.randint(LEGEND_EDGE_MARGIN, H - bh - LEGEND_EDGE_MARGIN)
    c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
    paste(grid, legend, r0, c0)
    return grid


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    """Create canonical key sprites, optional distractor sprite, and shuffle them."""
    sprites = []
    for color, count in zip(key_colors, counts):
        sprite = _make_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        sprites.append(_make_sprite(distractor_color, bg))
    random.shuffle(sprites)
    return sprites


def _place_sprites(grid: np.ndarray, sprites: list[np.ndarray], bg: int) -> None:
    """Paste sprites into random non-touching free locations in queue order."""
    for sprite in sprites:
        sr, sc = random_free_location(
            grid,
            sprite,
            transparent=bg,
            padding=SPRITE_PADDING,
            padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
        )
        paste(grid, sprite, sr, sc, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    """Run one output-first generation attempt and reject solver mismatches."""
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a D5 input")
