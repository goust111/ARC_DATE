# Variant of 58490d8a
# Dimension: D3 — 目标位置：图例框的结构 (dst)
# Change: 黑色矩形框，内部第2列（index=1）每隔一行放置一个关键颜色像素作为图例条目，输出画布即为该框的裁剪区域... → 图例条目水平排列替代垂直排列

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis for the D3 variant
# a) Concrete behavior:
#    - Original legend: key pixels sit at (row 1,3,5..., col 1) and each row grows rightward.
#    - Variant legend: key pixels sit at (row 1, col 1,3,5,...) and each column grows downward.
#    - Example with key colors [red, blue] and counts [3, 1]:
#      original output would mark red on one row at cols 1/3/5 and blue on another row;
#      variant output marks red in one column at rows 1/3/5 and blue in another column at row 1 only.
#    - The counting target is unchanged: 8-connected components outside the legend box.
# b) Complexity calibration:
#    - Keep K in 2-4 and counts in 1-5 so multiple columns are visible and each sample still has repeated evidence.
#    - Only the legend orientation changes; sprite shapes, distractors, and palette rules stay intact to avoid adding new reasoning axes.
# c) Implicit-assumption audit of the original solver:
#    - "leftmost non-black column" no longer holds; we now need the topmost non-black row.
#    - Keys are no longer indexed by row, so the `(row, color)` list becomes `(col, color)`.
#    - Tally placement can no longer use `key_col + 2*i`; it must use `key_row + 2*i`.
#    - The largest-black-component legend detection still holds because the legend remains a single black patch with colored holes.
# d) Placement legality:
#    - Rotating the legend swaps which dimension depends on K versus max_count.
#    - New generator therefore uses width = 2*K+1 and height = 2*max_count+1 plus 0-3 cells of slack below the tallies.
#    - Existing sprite placement remains valid because the legend is still pasted as one opaque rectangle and sprites still use padding=1.
# e) Visual observability:
#    - Every generated legend has one populated row and at least two populated columns, so the horizontal-key layout is visible in a few examples.
#    - The output columns visibly extend downward, which disagrees with the source task's row-wise pattern.
# f) Visual cleanliness:
#    - Keys and tally dots stay on every-other coordinates, preserving black gaps and preventing adjacent columns from merging visually.
# g) Color inferability:
#    - No color remapping is introduced; colors only serve as direct identity links between outside sprites and legend entries.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and encode the tallies in a rotated legend box.

    Input structure:
    - A black rectangular legend box contains one row of key-color pixels across its topmost non-black row.
    - Outside the box, separate same-colored sprites for those key colors are scattered on the background.

    Rule:
    - For each key color, count the number of outside connected components of that color.
    - Repaint the cropped legend box in black, then place one dot per counted object every other row downward from the matching key pixel.

    Key insight:
    - The legend is still both the lookup table and the output canvas, but the lookup is read left-to-right while each tally grows top-to-bottom.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    _bg = int(get_bg(g))

    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
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

    assert not (best is None)

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key row:
    # topmost row with any non-black pixels (the "key pixels").
    key_row = None
    for rr in range(bh):
        if (box[int(rr), :] != int(black)).any():
            key_row = int(rr)
            break
    if key_row is None:
        return np.zeros_like(box)

    keys = [(int(cc), int(box[int(key_row), int(cc)])) for cc in range(bw) if int(box[int(key_row), int(cc)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3/4) Count CCs of each key color outside the legend, then draw 1 dot per CC downward.
    for cc, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            rr = int(key_row + 2 * i)
            if 0 <= rr < bh:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

# Key columns and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW = 1
KEY_COLUMN_FIRST = 1
KEY_COLUMN_STEP = 2
TALLY_ROW_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

# Canvas size and legend placement bounds.
GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

# Sprite shape and placement constraints.
SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

# Retry budget for rejection sampling.
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
    """Sample the target outside-component count for each key color."""
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return rotated legend height and width for the key count and largest tally."""
    bh = TALLY_ROW_STEP * max_count + LEGEND_HEIGHT_TAIL + _randint(LEGEND_SLACK_RANGE)
    bw = KEY_COLUMN_STEP * k + LEGEND_WIDTH_TAIL
    return bh, bw


def _key_col(i: int) -> int:
    """Map a zero-based key index to its column inside the legend/output box."""
    return KEY_COLUMN_FIRST + KEY_COLUMN_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        cc = _key_col(i)
        for j in range(count):
            rr = KEY_ROW + TALLY_ROW_STEP * j
            expected[rr, cc] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the top key-row pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[KEY_ROW, _key_col(i)] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a D3 input")
