# Combo variant of 58490d8a (index 19)
# combo_spec: D6/0+D9/1
# Name: 右侧关键列+精灵消失

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

"""
Design analysis for the combo variant D6/0 + D9/1.

a) Concrete behavior and trace
spec:
  preserved:
    - The input still has one black legend rectangle and 2-4 keyed rows.
    - Each key color still indexes 1-5 disconnected outside sprites of the same color.
    - Counting is still based on 8-connected components outside the legend.
  changed:
    - The legend key pixels are now anchored in the legend's rightmost column.
    - Tally dots therefore extend leftward in steps of 2 cells from that right edge.
    - The output is no longer the cropped legend patch; it is the full grid.
    - All outside sprites whose colors appear in the legend are erased to the background.

Small trace:
  original:
    - Suppose a red key pixel is at legend column 1 and there are 3 red outside components.
    - The output is only the legend crop, with red dots at columns 1, 3, 5.
  combo:
    - Suppose the legend width is 8, so the red key pixel is at column 7.
    - With the same 3 outside components, the full-grid output erases the red sprites,
      repaints the legend area black, and draws red dots at columns 7, 5, 3.
  difference:
    - Position and direction both flip inside the legend, and the counted sprites vanish
      from the rest of the scene instead of remaining visible.

b) Complexity calibration
  - Keep 2-4 key rows and 1-5 sprites per key color so each sample has repeated evidence.
  - Force at least one sampled count to be >= 2; otherwise the leftward growth could be
    visually weak even though the key column is on the right.
  - Keep at most one distractor sprite so the erase rule is visible without clutter.
  - Preserve the original odd-row legend layout to keep row-color correspondence obvious.

c) Hidden-assumption audit of the original main()
  - Legend detection by "largest black CC whose bbox encloses non-black pixels" still holds.
    Right-edge key pixels puncture the border, but the black legend remains one dominant CC.
  - The original scan from left to right for the first non-black legend column is invalid.
    The combo must scan from right to left because the key column moved to the far right.
  - The original tally placement `key_col + 2*i` is invalid. It must become `key_col - 2*i`.
  - The original cropped output canvas is invalid. The combo must start from a full-grid copy,
    erase keyed colors only outside the legend, black out the legend bbox, then paste the tally.
  - Returning a cropped black box when no key column is found is no longer coherent with the
    output shape; the safest fallback is the unchanged full grid.

d) Placement legality
  - The generator pastes the legend first, then uses `random_free_location(..., padding=1,
    padding_connectivity=8)` so sprites cannot overlap or touch each other or the legend.
  - With a right-edge anchor, the width formula becomes `2 * max_count + left_slack`.
    This guarantees the leftward positions `bw-1, bw-3, ...` stay in bounds.
  - Erasure is always masked to the outside-of-legend region, so legend pixels are never
    accidentally deleted before the tally patch is written.

e) Visual observability
  - A solver can directly see that the key pixels sit on the right boundary of the legend.
  - The output then shows dots growing left from that edge while the matching outside sprites
    disappear; these two effects reinforce each other on the same colors.
  - Keeping an optional distractor means some outside content sometimes survives, which makes
    it clear that only legend-indexed colors are erased.

f) Visual cleanliness
  - The layout stays sparse: one legend area, several small isolated sprites, and at most one
    distractor, so the full-grid output does not become crowded.
  - Non-touching placement avoids accidental component merges and keeps counted objects legible.
  - Alternating key rows prevent leftward tally runs from colliding vertically.

g) Change inferability
  - No color remapping is introduced. The same visible key colors both mark the legend rows and
    identify which outside sprites disappear.
  - The rule depends only on visible structure: right-edge key placement, repeated object counts,
    and disappearance of those keyed colors from the scene.
"""


def _find_legend_bbox(grid: np.ndarray, black: int) -> tuple[int, int, int, int]:
    """Return the legend bbox: the largest black CC whose bbox encloses non-black cells."""
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


def _extract_right_keys(box: np.ndarray, black: int) -> tuple[int | None, list[tuple[int, int]]]:
    """Return the rightmost key column and ordered (row, color) legend entries."""
    bh, bw = map(int, box.shape)
    key_col = None
    for cc in range(bw - 1, -1, -1):
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


def _build_leftward_tally(
    shape: tuple[int, int],
    keys: list[tuple[int, int]],
    key_col: int,
    counts: dict[int, int],
) -> np.ndarray:
    """Render the black legend patch with right-anchored dots stepping leftward."""
    bh, bw = map(int, shape)
    patch = np.zeros((bh, bw), dtype=int)
    for rr, col in keys:
        for i in range(int(counts.get(int(col), 0))):
            cc = int(key_col - 2 * i)
            if 0 <= cc < bw:
                patch[int(rr), int(cc)] = int(col)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed sprites, erase them from the scene, and rewrite a right-anchored legend.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on each active row, and those key pixels sit in the box's rightmost column.
    - Outside the box, disconnected sprites of those key colors are scattered on a single background, plus an optional non-key distractor sprite.

    Rule:
    - For each key color, count the number of separate same-colored components outside the legend box.
    - Output the full grid: erase all outside sprites of key colors back to the background, repaint the legend box black, and draw one same-colored dot per counted component starting at the right-edge key column and stepping left every other cell.

    Key insight:
    - The legend no longer just names the colors; its right-edge key placement also tells you the tally direction, while the full-grid output reveals that the counted keyed sprites themselves disappear from the scene.
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
    # rightmost column with any non-black pixels (the key pixels).
    key_col, keys = _extract_right_keys(box, black)
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

    tally = _build_leftward_tally((bh, bw), keys, int(key_col), counts)
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
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_LEFT_SLACK_RANGE = (0, 3)

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
    """Sample outside-component counts and force at least one visibly leftward tally run."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= 2:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest leftward tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + _randint(LEGEND_LEFT_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _key_col(width: int) -> int:
    """The key column is always the legend's rightmost column."""
    return int(width - 1)


def _build_tally_canvas(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black legend tally patch implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    key_col = _key_col(bw)

    tally = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            tally[rr, key_col - TALLY_COL_STEP * j] = int(color)
    return tally


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the rightmost key-column pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    key_col = _key_col(shape[1])
    for i, color in enumerate(key_colors):
        legend[_key_row(i), key_col] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Sample a grid, paste the legend away from all canvas edges, and return grid plus bbox."""
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


def _build_expected_full(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    tally: np.ndarray,
    key_colors: list[int],
    bg: int,
) -> np.ndarray:
    """Erase keyed sprites outside the legend and repaint the legend with the tally."""
    r0, c0, r1, c1 = map(int, legend_bbox)
    expected = np.array(grid, dtype=int, copy=True)
    outside = np.ones(expected.shape, dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    for color in key_colors:
        expected[(expected == int(color)) & outside] = int(bg)

    expected[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(BLACK)
    paste(expected, tally, int(r0), int(c0))
    return expected


def _combo_behavior_metrics(
    grid: np.ndarray,
    expected: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    key_colors: list[int],
) -> tuple[int, int]:
    """Return simple observability metrics for the fused behavior."""
    r0, c0, r1, c1 = map(int, legend_bbox)
    outside = np.ones(grid.shape, dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    erased_pixels = int(sum(((grid == int(color)) & outside).sum() for color in key_colors))
    changed_pixels = int((grid != expected).sum())
    return erased_pixels, changed_pixels


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Run one output-first generation attempt and reject weak or mismatched samples."""
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
    erased_pixels, changed_pixels = _combo_behavior_metrics(grid, expected, legend_bbox, key_colors)
    if erased_pixels <= 0 or changed_pixels <= 0:
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
