# Variant of 58490d8a
# Dimension: D9 — 输出画布范围 (grid_shape)
# Change: 输出为图例框的裁剪区域（与框同尺寸的黑色画布，仅包含圆点标记）
# ... → 输出为精灵消失后的网格

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    get_bg,
    label_mask,
    paste,
    random_free_location,
)

# Design analysis for the D9 variant
# a) Concrete behavior and trace:
#    - Input stays structurally identical to the source task: a black legend rectangle with
#      one key-color pixel per odd row, plus several disconnected outside sprites in those
#      same colors, optionally one distractor sprite in another color.
#    - Original execution: count outside components of each key color, then return only the
#      cropped legend-sized black patch with same-row tally dots.
#    - Variant execution: count exactly the same way, but the output canvas is the full grid.
#      Every outside sprite whose color appears in the legend is erased to the background
#      color, and the legend rectangle itself is repainted as the tally patch.
#    - Example trace on one key color:
#      input row in legend has one red key pixel at column 1; outside the box there are
#      three disconnected red sprites. Original output keeps only the legend crop and draws
#      red dots at columns 1,3,5 on that row. Variant output keeps the whole input grid,
#      removes all three red sprites from their original positions, and also draws the same
#      1,3,5 tally inside the legend.
# b) Complexity calibration:
#    - The counting rule is unchanged, so the puzzle still requires matching legend colors
#      to outside component counts.
#    - The erase rule adds a second visible transformation on the same keyed colors.
#      To avoid weak samples where the legend barely changes, generation forces at least one
#      count > 1, so at least one row gains extra dots beyond the input's single key pixel.
#    - Optional distractors are preserved so some non-key colored content survives outside
#      the legend; this keeps the erase rule visually specific instead of looking like a
#      blanket cleanup of the whole canvas.
# c) Implicit assumption audit for the original main():
#    - Legend detection by "largest black CC whose bbox contains non-black pixels" still
#      holds, because the input format is unchanged.
#    - Key-column detection via the leftmost non-black legend column still holds because the
#      variant changes only the output, not the input.
#    - The original `out = np.zeros((bh, bw))` assumes the result is a cropped legend patch.
#      This is the D9-dependent line that must change; now the correct base canvas is the
#      full input grid.
#    - The original dot drawing is harmless because black is the whole output background.
#      In the variant, drawing into a full-grid copy would be wrong unless keyed sprites are
#      erased first and the legend bbox is explicitly reset to black before pasting dots.
# d) Placement legality:
#    - Generator keeps the original 8-neighborhood gap between every sprite and between
#      sprites and the legend box, so components remain unambiguous for counting.
#    - The legend width still depends on `max_count`, so all tally dots fit after repainting.
#    - Erasure is implemented by color restricted to the outside-of-legend mask; therefore
#      legend key pixels are never accidentally removed before the tally patch is pasted.
# e) Visual observability:
#    - In 2-4 examples a solver can see two coupled effects on the same keyed colors:
#      their outside instances vanish, and their counts appear as dot runs in the legend.
#    - Because distractors sometimes remain, it is visible that only legend-indexed colors
#      are erased.
# f) Visual cleanliness:
#    - The output keeps the spacious original layout; only the small legend box and the
#      keyed sprites change, so there is no crowding or overlapping annotation layer.
#    - Non-touching sprite placement prevents erasure from creating ambiguous merged holes.
# g) Color inference:
#    - No hidden ordering by color id is introduced. Rows are read directly from legend
#      positions, and the same visible colors both disappear outside and reappear as tally
#      dots inside the box.


def _find_legend_bbox(grid: np.ndarray, black: int) -> tuple[int, int, int, int]:
    """Return the legend bbox: the largest black component whose bbox encloses non-black cells."""
    best = None  # (area, bbox)
    for coords in components_mask(grid == int(black), connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(grid, bbox)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return tuple(map(int, best[1]))


def _extract_keys(box: np.ndarray, black: int) -> tuple[int | None, list[tuple[int, int]]]:
    """Return the leftmost key column and ordered (row, color) legend entries."""
    bh, bw = map(int, box.shape)
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return None, []
    keys = [
        (int(rr), int(box[int(rr), int(key_col)]))
        for rr in range(bh)
        if int(box[int(rr), int(key_col)]) != int(black)
    ]
    keys.sort(key=lambda t: int(t[0]))
    return key_col, keys


def _build_tally_patch(shape: tuple[int, int], keys: list[tuple[int, int]], key_col: int, counts: dict[int, int]) -> np.ndarray:
    """Render the black legend patch with one every-other-cell dot per counted component."""
    bh, bw = map(int, shape)
    patch = np.zeros((bh, bw), dtype=int)
    for rr, col in keys:
        for i in range(int(counts.get(int(col), 0))):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                patch[int(rr), int(cc)] = int(col)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed sprites, erase them from the scene, and rewrite the legend in place.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on each active row.
    - Outside the box, disconnected sprites of those key colors are scattered on a single background, plus an optional non-key distractor.

    Rule:
    - For each key color, count the number of separate same-colored components outside the legend.
    - Output the full grid: repaint the legend box as a black tally panel with one same-colored dot every other cell, and erase all outside sprites of key colors back to the background.

    Key insight: The legend still encodes the count mapping, but its result is no longer a cropped patch — the counts are written back into the original scene while the counted sprites themselves disappear.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    r0, c0, r1, c1 = _find_legend_bbox(g, black)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key column:
    # leftmost column with any non-black pixels (the "key pixels").
    key_col, keys = _extract_keys(box, black)
    if key_col is None:
        return g.copy()

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    # STEP 3) Count CCs of each key color outside the legend.
    counts: dict[int, int] = {}
    for _rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        counts[int(col)] = int(nc)

    # STEP 4) Keep the whole scene, erase keyed sprites outside the legend, then repaint the legend tally.
    out = g.copy()
    for col in counts:
        out[(out == int(col)) & outside] = int(bg)

    tally = _build_tally_patch((bh, bw), keys, int(key_col), counts)
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(black)
    paste(out, tally, int(r0), int(c0))
    return out


# --------------- generator (o5x) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
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
    """Sample outside-component counts, forcing at least one row to extend beyond the key pixel."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if any(count > 1 for count in counts):
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_tally_canvas(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black legend tally patch implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the key-column pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Sample a grid, paste the legend away from all canvas edges, and return grid plus legend bbox."""
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
    return grid, (int(r0), int(c0), int(r0 + bh - 1), int(c0 + bw - 1))


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


def _build_expected_full(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], tally: np.ndarray, key_colors: list[int], bg: int) -> np.ndarray:
    """Apply the D9 transform explicitly: erase key colors outside the legend and repaint the legend tally."""
    r0, c0, r1, c1 = map(int, legend_bbox)
    expected = np.array(grid, dtype=int, copy=True)
    outside = np.ones(expected.shape, dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    for color in key_colors:
        expected[(expected == int(color)) & outside] = int(bg)

    expected[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(BLACK)
    paste(expected, tally, int(r0), int(c0))
    return expected


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Run one output-first generation attempt and reject solver mismatches."""
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    tally = _build_tally_canvas(key_colors, counts)
    legend = _build_legend(tally.shape, key_colors)
    placed = _new_grid_with_legend(bg, legend)
    if placed is None:
        return None
    grid, legend_bbox = placed

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    expected = _build_expected_full(grid, legend_bbox, tally, key_colors, bg)

    # Reject visually weak cases only if the D9 effect somehow vanishes.
    outside = np.ones(grid.shape, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    erased_pixels = int(sum(((grid == int(color)) & outside).sum() for color in key_colors))
    if erased_pixels <= 0:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            grid, _expected = pair
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a D9 input")
