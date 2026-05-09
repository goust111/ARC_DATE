# Combo variant of 58490d8a (index 23)
# combo_spec: D4/0+D9/1+D14/0
# Name: 条形填充+精灵消失+排序

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


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - The input keeps the original ingredients: one black legend rectangle with
#      colored key pixels on alternating rows, plus disconnected same-color
#      sprites outside the box.
#    - The fused output now combines three changes on the same count data:
#      1. D4: each count becomes a contiguous bar starting at the key column.
#      2. D14: legend rows are reassigned by count rank, largest first.
#      3. D9: the output is the full grid, and all keyed outside sprites are
#         erased back to the background.
#    - Example trace with input legend rows [blue, red, yellow] and outside
#      counts [2, 5, 3]:
#        original task -> cropped legend with blue dots x2, red dots x5,
#                         yellow dots x3 on their original rows.
#        combo variant -> full scene, all blue/red/yellow sprites disappear,
#                         and the legend becomes sorted bars:
#                         row 1 = red bar length 5,
#                         row 3 = yellow bar length 3,
#                         row 5 = blue bar length 2.
#      The same counting stage now drives both spatial erasure and ranked bar
#      rendering.
#
# b) Complexity calibration
#    - If counts tied, sorting would be less visible. If every count were 1,
#      bar fill would collapse to single pixels. The generator therefore uses
#      distinct counts and rejects already-descending input order.
#    - Key count stays in 2-4 and counts stay in 1-5, so the task remains small
#      enough to infer from examples without introducing extra object types.
#    - Optional distractors remain allowed so the outside scene is not reduced to
#      pure background after erasure in every sample.
#
# c) Implicit assumption audit
#    - Original assumption: output canvas is only the legend crop. Broken by D9;
#      the solver must instead keep a full-grid copy and repaint only the legend
#      bbox after erasing keyed colors outside it.
#    - Original assumption: every key color draws back onto its own input row.
#      Broken by D14; the solver must separate "rows used to read colors" from
#      "rows used as ranked output slots".
#    - Original assumption: tally marks advance by +2 columns. Broken by D4;
#      rendering must fill a continuous slice from key_col through key_col+count-1.
#    - Still-valid assumptions: the legend is the largest black component whose
#      bbox encloses non-black cells; the leftmost non-black legend column is the
#      key column; outside sprites remain non-touching so component counts are
#      stable under 8-connectivity.
#
# d) Placement legality
#    - Generator keeps the original padding=1, 8-neighborhood placement rule for
#      sprites, so keyed copies stay as separate components and never touch the
#      legend or each other.
#    - Legend width is sized for contiguous bars:
#        bw = KEY_COLUMN + max_count + 1 + slack
#      which guarantees even the longest ranked bar fits fully.
#    - Legend placement still respects a margin from grid edges, so sprite
#      placement has enough space and no object crosses the canvas boundary.
#
# e) Visual observability
#    - The ranking key is purely visible: longer bars move upward in the legend.
#    - The erase rule is also visible because the same colors that are counted
#      disappear from the outside scene.
#    - Distinct counts and unsorted input order ensure that at least one row
#      moves and at least one bar extends beyond a single key pixel in each
#      generated sample.
#
# f) Visual cleanliness
#    - The output uses only one transformed region inside the legend and one
#      cleanup action outside it; there are no overlapping annotation layers.
#    - Bars remain on sparse alternating rows separated by black rows, which
#      keeps the ranked legend easy to read after reordering.
#
# g) Color-rule inferability
#    - No hidden color remapping is introduced. A color is tracked directly from
#      its input key pixel to its outside sprites, then back to its ranked output
#      bar. The only change is row position, which is explained by visible count
#      differences rather than by color ids or any non-visual property.
# ---------------------------------------------------------------------------


def _find_legend_bbox(grid: np.ndarray, black: int) -> tuple[int, int, int, int]:
    """Return the legend bbox: the largest black component whose bbox contains non-black cells."""
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


def _count_records(grid: np.ndarray, outside: np.ndarray, keys: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Return (input_row, color, outside_component_count) for each key color."""
    records: list[tuple[int, int, int]] = []
    for rr, color in keys:
        mask = (grid == int(color)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(rr), int(color), int(count)))
    return records


def _sorted_records(records: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Sort by descending count, then by original top-to-bottom legend order."""
    ranked = list(records)
    ranked.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return ranked


def _build_sorted_bar_patch(
    shape: tuple[int, int],
    slot_rows: list[int],
    key_col: int,
    ranked_records: list[tuple[int, int, int]],
) -> np.ndarray:
    """Render a black legend patch whose rows are ranked and whose tallies are solid bars."""
    bh, bw = map(int, shape)
    patch = np.zeros((bh, bw), dtype=int)
    for out_idx, (_orig_rr, color, count) in enumerate(ranked_records):
        rr = int(slot_rows[out_idx])
        end = min(int(bw), int(key_col + count))
        patch[int(rr), int(key_col) : end] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed sprites, erase them from the scene, and rewrite the legend as ranked bars.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on each alternating active row, all in its leftmost non-black column.
    - Outside the box, disconnected sprites of those key colors are scattered on one background color, with an optional extra distractor color.

    Rule:
    - Count the number of outside connected components for each key color.
    - Output the full grid: erase all outside sprites of key colors back to the background, then repaint the legend box black and fill its active rows with contiguous same-color bars whose lengths equal the counts, ordered from largest count to smallest.

    Key insight: The legend no longer preserves its input row order or its cropped output size; first read it as a color index, then rank those colors by how many outside sprites they own, write the ranked bar chart back into the same legend area, and remove the counted sprites from the rest of the scene.
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
    slot_rows = [int(rr) for rr, _color in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    # STEP 3) Count outside CCs for each key color, then sort the key records by count.
    records = _count_records(g, outside, keys)
    ranked_records = _sorted_records(records)

    # STEP 4) Keep the full scene, erase keyed sprites outside the legend, then repaint the legend as sorted bars.
    out = g.copy()
    for _rr, color, _count in records:
        out[(out == int(color)) & outside] = int(bg)

    patch = _build_sorted_bar_patch((bh, bw), slot_rows, int(key_col), ranked_records)
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(black)
    paste(out, patch, int(r0), int(c0))
    return out


# --------------- generator (combo variant) ---------------

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
    """Sample distinct counts and force the input legend order to differ from the ranked output."""
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and longest contiguous bar."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = KEY_COLUMN + max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch whose rows preserve the unsorted input order."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _build_sorted_bar_canvas(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the ranked black legend patch implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    ranked = _sorted_records(
        [(int(_key_row(i)), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    )
    slot_rows = [_key_row(i) for i in range(len(key_colors))]
    return _build_sorted_bar_patch((bh, bw), slot_rows, KEY_COLUMN, ranked)


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
    """Erase keyed outside sprites and repaint the legend with the ranked bar patch."""
    r0, c0, r1, c1 = map(int, legend_bbox)
    expected = np.array(grid, dtype=int, copy=True)
    outside = np.ones(expected.shape, dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    for color in key_colors:
        expected[(expected == int(color)) & outside] = int(bg)

    expected[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(BLACK)
    paste(expected, tally, int(r0), int(c0))
    return expected


def _fusion_metrics(grid: np.ndarray, expected: np.ndarray) -> dict[str, int]:
    """Return simple observability metrics for the fused behavior."""
    bg = int(get_bg(grid))
    r0, c0, r1, c1 = _find_legend_bbox(grid, BLACK)
    box = crop_bbox(grid, (r0, c0, r1, c1))
    key_col, keys = _extract_keys(box, BLACK)
    assert key_col is not None
    outside = np.ones(grid.shape, dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    records = _count_records(grid, outside, keys)
    ranked = _sorted_records(records)

    erased_pixels = int(sum(((grid == int(color)) & outside).sum() for _rr, color, _count in records))
    moved_rows = int(sum(int(orig_rr != _key_row(i)) for i, (orig_rr, _color, _count) in enumerate(ranked)))
    long_bars = int(sum(int(count > 1) for _rr, _color, count in ranked))
    outside_survivors = int(((expected != int(bg)) & outside).sum())
    return {
        "erased_pixels": erased_pixels,
        "moved_rows": moved_rows,
        "long_bars": long_bars,
        "outside_survivors": outside_survivors,
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    """Run one output-first generation attempt and reject weak or incorrect samples."""
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    tally = _build_sorted_bar_canvas(key_colors, counts)
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
    metrics = _fusion_metrics(grid, expected)

    if metrics["erased_pixels"] <= 0:
        return None
    if metrics["moved_rows"] <= 0:
        return None
    if metrics["long_bars"] <= 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected, metrics


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        bundle = _gen_attempt()
        if bundle is not None:
            grid, _expected, _metrics = bundle
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
