# Variant of 58490d8a
# Dimension: D14 — 多步处理：计数与排序 (multi_step)
# Change: 单步处理 — 直接统计各颜色连通分量数量并填入对应行... → 分阶段处理：先处理偶数行再处理奇数行

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# Design analysis for the D14 variant
# a) Concrete behavior:
#    - The input still contains one black legend rectangle whose left marker column lists the key colors.
#    - Outside the legend, each key color appears as 1-5 disconnected sprites; an optional distractor color may appear once.
#    - Key colors are ordered by their row inside the legend. The 0th, 2nd, ... keys form the "even-position" phase;
#      the 1st, 3rd, ... keys form the "odd-position" phase.
#    - Output keeps the same black canvas size as the legend box, but the two phases write dots differently:
#      even-position keys place dots at columns key_col, key_col+2, ...; odd-position keys place dots at the
#      parity-matching columns counted from the right edge of the box.
#    - Example with bw=13 and counts [2,3,1,4]:
#      row 0 key -> cols [1,3]; row 1 key -> cols [11,9,7]; row 2 key -> [1]; row 3 key -> [11,9,7,5].
# b) Complexity calibration:
#    - The task still requires object counting, but now the solver must also classify key rows by top-to-bottom parity.
#    - The legend is widened by at least two extra cells compared with the original minimum so right-aligned rows do not
#      collapse into the same pattern as left-aligned rows.
# c) Old assumptions that break:
#    - Original code assumed every key row uses `key_col + 2*i` for dot placement. That is no longer valid.
#    - Row parity must be computed from the sorted key order, not from raw grid row parity.
#    - Right alignment needs a parity-compatible anchor column near the right edge; the generator guarantees enough room.
# d) Placement legality:
#    - Sprites remain isolated by one-cell 8-neighborhood padding, so connected-component counting outside the legend is stable.
#    - The legend box stays away from the image border, and the wider legend still fits in the configured canvas ranges.
# e) Visual observability:
#    - The new rule is visible from position, not color ids: alternating key rows produce left-anchored versus right-anchored tallies.
#    - Multiple examples will show that the split always follows key order (0th/1st/2nd/3rd), not any specific color.
# f) Visual cleanliness:
#    - Rows never overlap because each color owns exactly one legend row, and dots stay on that row only.
#    - The black legend canvas remains uncluttered: one row, one color, evenly spaced dots.
# g) Color inference:
#    - No hidden color ranking is used. Colors only serve as identity labels already shown in the legend.
# h) Generator anti-degeneration rule:
#    - Width includes at least one extra right-side slot beyond the largest count so odd rows visibly differ from the original task.
# i) New solver assumptions audited:
#    - The largest black connected component with internal non-black pixels is still the legend.
#    - Keys are read from the leftmost non-black column and sorted top-to-bottom before parity grouping.
#    - Because even and odd phases write on disjoint rows, the staged processing order is safe and the generator does not need
#      extra collision control beyond the original sprite spacing rules.


def _find_legend_bbox(grid: np.ndarray, black: int) -> tuple[int, int, int, int]:
    """Return the bbox of the legend box: the largest black CC that encloses non-black pixels."""
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


def _right_anchor(key_col: int, width: int) -> int:
    """Return the rightmost in-bounds column whose parity matches the key column."""
    anchor = int(width) - 1
    if (anchor - int(key_col)) % 2 != 0:
        anchor -= 1
    return int(anchor)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count exterior colored objects and redraw the legend as a two-phase tally chart.

    Input structure:
    - A black legend rectangle contains one vertical key column of colored marker pixels.
    - Outside the legend, disconnected sprites of those key colors are scattered on a single background.

    Rule:
    - Count the outside connected components for each key color, using the top-to-bottom key order to split rows into two phases.
    - Even-position keys (0th, 2nd, ...) place one dot per component from the left; odd-position keys (1st, 3rd, ...) place one dot per component from the right, always stepping every other cell.

    Key insight: The legend rows are not all filled the same way; first identify the row order, then apply the parity-specific left/right tally rule to the matching color counts.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    r0, c0, r1, c1 = _find_legend_bbox(g, black)
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
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)
    right_anchor = _right_anchor(key_col, bw)

    for parity in (0, 1):
        for key_index, (rr, col) in enumerate(keys):
            if key_index % 2 != parity:
                continue
            mask = (g == int(col)) & outside
            _labels, count = label_mask(mask, connectivity=8)
            for i in range(int(count)):
                if parity == 0:
                    cc = int(key_col + 2 * i)
                else:
                    cc = int(right_anchor - 2 * i)
                if 0 <= cc < bw:
                    out[int(rr), int(cc)] = int(col)

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
# Keep at least one extra parity-matching slot to make right-aligned rows visibly different.
LEGEND_SLACK_RANGE = (2, 4)

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
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + 1 + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    right_anchor = _right_anchor(KEY_COLUMN, bw)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for parity in (0, 1):
        for i, (color, count) in enumerate(zip(key_colors, counts)):
            if i % 2 != parity:
                continue
            rr = _key_row(i)
            for j in range(count):
                if parity == 0:
                    cc = KEY_COLUMN + TALLY_COL_STEP * j
                else:
                    cc = right_anchor - TALLY_COL_STEP * j
                expected[rr, cc] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a input")
