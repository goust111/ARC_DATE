# Combo variant of 58490d8a (index 5)
# combo_spec: D3/1+D6/0
# Name: 水平图例+底行关键色

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the D3/1 + D6/0 combo
# a) Concrete behavior and trace:
#    - Original task: key pixels sit in one internal column at rows 1,3,5,... and each matching tally grows rightward.
#    - D3/1 rotates the legend logic: keys move onto one row at columns 1,3,5,... and each tally grows downward.
#    - This combo then mirrors the rotated anchor side: the horizontal key row moves from the top to the bottom,
#      so each tally must start at the bottom key pixel and grow upward every other row.
#    - Small trace with key colors [red, blue] and counts [3, 1]:
#      original output marks red at (row 1, cols 1/3/5) and blue at (row 3, col 1);
#      rotated-only output marks red at (rows 1/3/5, col 1) and blue at (row 1, col 3);
#      fused output marks red at (bottom row, col 1), (bottom-2, col 1), (bottom-4, col 1),
#      while blue remains only at (bottom row, col 3).
# b) Complexity calibration:
#    - Keep 2-4 key colors and 1-5 external components per color so multiple legend columns remain visible.
#    - Force at least one sampled count to be >= 2; otherwise the upward growth could be invisible and the task
#      would look like a static bottom legend with no directional evidence.
#    - Preserve the original sprite palette, shape family, and single optional distractor so the only new reasoning
#      burden is the fused orientation/anchor change.
# c) Implicit-assumption audit of the original solver:
#    - "leftmost non-black column" is no longer the key detector; we now need the bottommost non-black row.
#    - Keys are indexed by column rather than row, so the legend lookup list becomes `(col, color)`.
#    - Tally placement can no longer use `key_col +/- 2*i`; the combo needs `key_row - 2*i`.
#    - The legend finder based on the largest black component still works because the legend remains one black
#      rectangle with a few colored holes in its bottom row.
#    - The generator must ensure there are no other colored pixels inside the legend, otherwise the bottom-row scan
#      would be ambiguous.
# d) Placement legality:
#    - The legend box is still pasted as one solid black patch before any sprites are placed.
#    - Sprites are still placed by `random_free_location(..., padding=1, padding_connectivity=8)`, so they cannot
#      overlap or touch each other or the legend.
#    - Setting `height = 2*max_count + top_slack` with the key row at `height-1` guarantees all upward tally rows
#      stay in bounds.
# e) Visual observability:
#    - In the input, the colored key pixels visibly lie along the bottom border instead of the left column or top row.
#    - In the output, at least one column extends upward for multiple dots, making the fused direction change legible.
# f) Visual cleanliness:
#    - Keys and tallies still occupy every-other coordinates, preserving black separators between active columns and dots.
#    - Extra slack is placed above the tallies only, which keeps the bottom anchoring easy to read.
# g) Color inferability:
#    - No recoloring rule is introduced; colors remain direct identifiers linking outside sprites to legend entries.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count exterior colored objects and write the totals into a bottom-anchored horizontal legend.

    Input structure:
    - A black rectangular legend box contains one row of key-color pixels across its bottommost non-black row.
    - Outside the box, separated same-colored sprites of those key colors are scattered on a uniform background.

    Rule:
    - For each key color, count the 8-connected components of that color outside the legend box.
    - Output the cropped legend box repainted in black, then place one dot per counted component in the matching legend column, stepping upward every other row from the bottom key pixel.

    Key insight:
    - The legend still acts as both the color index and the output canvas, but the key row is now read left-to-right along the bottom edge and each tally grows vertically upward from that anchor.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key row:
    # bottommost row with any non-black pixels (the key pixels).
    key_row = None
    for rr in range(bh - 1, -1, -1):
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

    # STEP 3/4) Count CCs of each key color outside the legend, then draw 1 dot per CC upward.
    for cc, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            rr = int(key_row - 2 * i)
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
KEY_COLUMN_FIRST = 1
KEY_COLUMN_STEP = 2
TALLY_ROW_STEP = 2
LEGEND_WIDTH_TAIL = 1
LEGEND_TOP_SLACK_RANGE = (0, 3)

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
    """Sample target counts and force at least one visible upward multi-dot column."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= 2:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the bottom-anchored vertical tallies."""
    bh = TALLY_ROW_STEP * max_count + _randint(LEGEND_TOP_SLACK_RANGE)
    bw = KEY_COLUMN_STEP * k + LEGEND_WIDTH_TAIL
    return bh, bw


def _key_col(i: int) -> int:
    """Map a zero-based key index to its column inside the legend/output box."""
    return KEY_COLUMN_FIRST + KEY_COLUMN_STEP * i


def _key_row(height: int) -> int:
    """The key row is always the legend's bottom row."""
    return int(height - 1)


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    key_row = _key_row(bh)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        cc = _key_col(i)
        for j in range(count):
            expected[key_row - TALLY_ROW_STEP * j, cc] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the bottom key-row pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    key_row = _key_row(shape[0])
    for i, color in enumerate(key_colors):
        legend[key_row, _key_col(i)] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
