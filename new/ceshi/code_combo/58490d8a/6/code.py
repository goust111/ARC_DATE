# Combo variant of 58490d8a (index 6)
# combo_spec: D4/0+D9/1
# Name: 条形填充+精灵消失

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
Design analysis for the D4/0 + D9/1 combo variant

a) Concrete behavior and trace
   - Input keeps the source-task layout: a black legend rectangle contains one colored key
     pixel on each active row, and outside the box there are disconnected sprites in those
     key colors, plus an optional distractor color.
   - Original task:
     count n outside components for a key color -> return only the cropped legend patch ->
     place tally pixels at key_col, key_col+2, key_col+4, ...
   - Combo task:
     count n outside components for a key color -> keep the full scene as output ->
     erase every outside sprite whose color is indexed by the legend ->
     repaint the legend row as a contiguous bar from key_col through key_col+n-1.
   - Small trace with key_col=1 and a row whose color appears on three outside sprites:
     original cropped row: [0, C, 0, C, 0, C, 0]
     combo legend row  : [0, C, C, C, 0, 0, 0]
     and all outside C sprites disappear to background in the full-grid output.
   - The two changes interact on the same counted colors: those colors vanish outside and
     reappear inside the legend as solid length-coded bars.

b) Complexity calibration
   - If every count were 1, the bar rule would visually collapse to a single key pixel.
     Generation therefore forces at least one key color to have count >= 2 in every sample.
   - The task still uses only 2-4 key colors and 1-5 objects per key color, so counting
     remains readable and does not introduce unnecessary freedom.
   - Optional distractors are preserved so that some outside content can survive the erase
     step; this keeps the D9 part specific to legend-indexed colors instead of reading like
     whole-scene cleanup.

c) Implicit assumption audit
   - Legend detection by "largest black component whose bbox contains non-black cells"
     remains valid because the input format itself is unchanged.
   - Key-column detection by the leftmost non-black column inside the legend also remains
     valid for the same reason.
   - The original solver assumed a cropped black output canvas (`out = zeros((bh, bw))`).
     The combo breaks that assumption: output must now start from the full input grid.
   - The original tally placement `key_col + 2 * i` assumes spaced dots. The combo breaks
     that assumption too: rows must be filled contiguously from `key_col` onward.
   - Erasing keyed colors is safe only when restricted to the outside-of-legend mask;
     otherwise the legend key pixels would be destroyed before the bar patch is rebuilt.
   - Width logic also changes: the original dot layout needed about `2*max_count` columns,
     but the bar layout only needs room for `max_count` consecutive cells after `key_col`.

d) Placement legality
   - The generator still uses `random_free_location(..., padding=1, connectivity=8)`, so
     sprites do not overlap and do not touch each other or the legend.
   - The legend is placed with a margin from the grid edges, preserving room for sprites.
   - Legend width is sampled as `KEY_COLUMN + max_count + tail + slack`, guaranteeing the
     longest bar fits inside the box after the legend is repainted.

e) Visual observability
   - Across examples, a solver can see two coupled effects: legend-keyed sprites vanish from
     the outer scene, and the legend row for that same color grows into a solid bar whose
     length matches the number of vanished sprites.
   - Forcing at least one count > 1 prevents the D4 part from hiding inside length-1 cases.
   - Preserved distractors give direct evidence that only indexed colors are erased.

f) Visual cleanliness
   - The output keeps the original spacious grid and updates only two clearly separated
     regions: outside keyed sprites are erased, and the compact legend box is rewritten.
   - Bars occupy only the odd legend rows, leaving blank rows between them, so no dense
     overlay or ambiguous overlap is introduced.

g) Change inferability
   - No hidden ordering by numeric color id is used. Rows are matched purely by visible
     legend position and color identity.
   - The same color serves three visible roles: one key pixel in the input legend, several
     outside sprites to be counted/erased, and one output bar inside the legend. That shared
     structure is sufficient for humans to infer the color correspondence from few examples.
"""


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
    keys.sort(key=lambda item: int(item[0]))
    return key_col, keys


def _build_bar_patch(
    shape: tuple[int, int],
    keys: list[tuple[int, int]],
    key_col: int,
    counts: dict[int, int],
) -> np.ndarray:
    """Render the black legend patch with contiguous same-color bars."""
    bh, bw = map(int, shape)
    patch = np.zeros((bh, bw), dtype=int)
    for rr, col in keys:
        count = int(counts.get(int(col), 0))
        end = min(int(bw), int(key_col + count))
        patch[int(rr), int(key_col) : end] = int(col)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count legend-indexed sprites, erase them from the scene, and rewrite the legend as bars.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on each active row in its leftmost non-black column.
    - Outside that box, disconnected sprites of the key colors are scattered on a single background, with an optional extra distractor color.

    Rule:
    - For each key color, count the number of same-colored connected components outside the legend box.
    - Output the full grid: erase all outside sprites of legend colors back to the background, clear the legend to black, and draw a contiguous same-color bar of length n starting at the key column, where n is the count for that row.

    Key insight: The legend still provides the color-to-row index, but each indexed color now undergoes a fused transformation: its outside instances disappear and their count is rewritten inside the box as a solid horizontal bar.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    r0, c0, r1, c1 = _find_legend_bbox(g, black)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col, keys = _extract_keys(box, black)
    if key_col is None:
        return g.copy()

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    counts: dict[int, int] = {}
    for _rr, col in keys:
        mask = (g == int(col)) & outside
        _labels, nc = label_mask(mask, connectivity=8)
        counts[int(col)] = int(nc)

    out = g.copy()
    for col in counts:
        out[(out == int(col)) & outside] = int(bg)

    tally = _build_bar_patch((bh, bw), keys, int(key_col), counts)
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(black)
    paste(out, tally, int(r0), int(c0))
    return out


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
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
    """Sample outside-component counts, forcing at least one visible bar extension."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if any(count > 1 for count in counts):
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest bar."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = KEY_COLUMN + max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
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
        expected[rr, KEY_COLUMN : KEY_COLUMN + count] = int(color)
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


def _build_expected_full(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    tally: np.ndarray,
    key_colors: list[int],
    bg: int,
) -> np.ndarray:
    """Apply the combo transform explicitly: erase keyed sprites and repaint the legend as bars."""
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

    r0, c0, r1, c1 = legend_bbox
    outside = np.ones(grid.shape, dtype=bool)
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
    raise RuntimeError("failed to generate a valid 58490d8a combo variant input")
