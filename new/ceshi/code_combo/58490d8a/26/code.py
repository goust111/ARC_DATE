# Combo variant of 58490d8a (index 26)
# combo_spec: D3/2+D7/1+D15/0
# Name: 分散图例+多干扰+统一圆点色

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# combo_variant:
#   summary:
#     base_task: count same-color connected sprites and turn the counts into evenly
#       spaced dots inside a black legend structure
#     fused_changes:
#       - D3/2: one big legend box becomes 2-4 scattered mini-legends
#       - D7/1: add 2-3 distractor colors, each repeated 1-3 times
#       - D15/0: every output dot uses one shared tally-only color
#   concrete_pixel_behavior:
#     input:
#       background: one dominant non-black color
#       legends:
#         count: 2-4
#         form: each legend is an isolated black 3xW box with one colored key pixel
#           at row 1, col 1
#         width: 2*target_count + 1 + slack, slack in 0..2
#         placement: legends are separate black connected components spread across
#           the grid and padded away from all sprites
#       source_sprites:
#         key colors: each legend's color appears outside the legends as 1-5
#           disconnected same-color sprites
#         distractors: 2-3 extra colors, each with 1-3 disconnected sprites
#           placed with the same spacing rules
#       reserved_tally_color:
#         the requested "uniform white dots" cannot be literal white in ARC's 0..9
#         palette, so one reserved tally-only color is used as the shared output dot
#         color and is excluded from the input palette
#     output:
#       canvas: same size as the full input grid
#       outside legends: reset to background
#       each legend: repainted in-place as a black box whose middle row contains
#         one dot per counted key-color component at col 1, 3, 5, ...
#       dot_color: all dots use the same reserved tally color, including the first
#         dot that overwrites the input key pixel position
#   behavior_trace:
#     small_example:
#       original:
#         - one large box contains key rows [blue, red, green]
#         - outside counts are [3, 1, 2]
#         - output is a cropped black panel with colored rows
#       combo:
#         - three separate 3xW black boxes sit in different places, keyed by
#           [blue], [red], [green]
#         - outside the boxes there are the same key sprites plus several orange,
#           cyan, and gray distractor sprites
#         - output keeps the whole canvas, clears non-legend areas back to the
#           background, and fills each box with a row of shared-color dots whose
#           lengths are [3], [1], [2]
#       difference:
#         - "largest black component" no longer identifies the task structure
#         - color matching still matters for counting, but not for dot rendering
#         - extra non-key colors force explicit filtering by the legend key colors
#   complexity_calibration:
#     - Multiple legends are visually obvious because they stay separated and are
#       repeated 2-4 times in every sample.
#     - Distractors are rich enough to matter: the generator enforces 2-3 distractor
#       colors and at least 4 total distractor components, so ignoring them is not
#       an edge case.
#     - The shared tally color is also obvious because it never appears in the input
#       and replaces every per-row color cue in the output.
#     - Freedom is still controlled: legends keep the same 3-row template, tally
#       columns stay on odd offsets, and sprite sizes remain in the original range.
#   solver_assumption_audit:
#     - Original assumption: the largest black CC is the only legend. Broken by D3/2.
#     - Original assumption: output is the crop of that one legend bbox. Broken by
#       D3/2 because the repaint must happen in-place on the full grid.
#     - Original assumption: drawing color equals key color. Broken by D15/0.
#     - Original assumption: non-key colors are rare and can be ignored casually.
#       Weakened by D7/1, so solver must count only colors explicitly named by
#       detected legend keys.
#     - New assumption: every legend is a 3-row black CC with exactly one non-black
#       key pixel at (1,1). The generator guarantees this exact template.
#     - New assumption: legends do not touch each other or any sprite in 8-connectivity,
#       so each legend remains its own black connected component. The generator
#       enforces this with border and padding in placement.
#   placement_legality_and_cleanliness:
#     - Legends are placed first, sorted by width, with border and mutual padding.
#     - Sprites are placed after legends with 8-connected padding, preventing overlap
#       and accidental attachment to legends or to each other.
#     - Each legend width is derived from its own target count, so every tally fits.
#     - Output is clean because non-legend clutter disappears and the black boxes stay
#       isolated on the background instead of stacking multiple markings in one area.
#   observability_check:
#     - A human can infer the rule from a few examples: detect the repeated black mini-box
#       motif, match each box's lone key color to same-color sprites elsewhere, ignore
#       other colors, then notice that every answer row uses the same dot color.
#     - No hidden ordering by numeric color value is used. Matching relies only on
#       visible legend-pixel colors and visible object counts.
#   fusion_validation_metric:
#     - During generation we measure: number of detected legends, distractor-color
#       count, total distractor components, whether the reserved tally color leaked
#       into the input, and the spread gap between the legends' union bbox and their
#       total area.
#     - Samples are rejected unless they truly show multiple scattered legends plus
#       multi-color distractors, which prevents regression to the original task.
# -----------------------------------------------------------------------------


BLACK = 0
TALLY_COLOR = 9
INPUT_COLOR_POOL = tuple(range(1, TALLY_COLOR))

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)

DISTRACTOR_COLOR_COUNT_RANGE = (2, 3)
DISTRACTOR_COUNT_RANGE = (1, 3)
MIN_TOTAL_DISTRACTOR_COMPONENTS = 4

LEGEND_HEIGHT = 3
LEGEND_KEY_ROW = 1
LEGEND_KEY_COL = 1
TALLY_COL_STEP = 2
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 2)

GRID_HEIGHT_RANGE = (28, 34)
GRID_WIDTH_RANGE = (28, 34)
LEGEND_BORDER = 1
LEGEND_PADDING = 2

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 500


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    masks = []
    for cells in combinations(range(h * w), area):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        spans_edges = mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
        if not spans_edges:
            continue
        _lab, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n == 1:
            masks.append(mask)
    return tuple(masks)


def _make_sprite(color: int, bg: int) -> np.ndarray:
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


def _detect_legends(grid: np.ndarray) -> tuple[list[tuple[tuple[int, int, int, int], int, int]], np.ndarray]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    legends: list[tuple[tuple[int, int, int, int], int, int]] = []
    legends_mask = np.zeros((H, W), dtype=bool)

    for coords in components_mask(g == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        bbox = tuple(map(int, bbox_coords(coords)))
        patch = crop_bbox(g, bbox)
        ph, pw = map(int, patch.shape)
        if ph != LEGEND_HEIGHT or pw < 3:
            continue

        nonblack = np.argwhere(patch != BLACK)
        if nonblack.shape[0] != 1:
            continue

        rr, cc = map(int, nonblack[0])
        color = int(patch[rr, cc])
        if rr != LEGEND_KEY_ROW or cc != LEGEND_KEY_COL:
            continue
        if color in {BLACK, TALLY_COLOR}:
            continue

        legends.append((bbox, pw, color))
        r0, c0, r1, c1 = bbox
        legends_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    legends.sort(key=lambda item: (item[0][0], item[0][1]))
    return legends, legends_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed objects and write the totals into many scattered legend boxes.

    Input structure:
    - Several isolated black mini-legends are dispersed around the grid, each one a 3-row box with a single key-color pixel on its middle row.
    - Outside those boxes, disconnected sprites of the key colors appear alongside several extra distractor colors.

    Rule:
    - For each mini-legend, count the connected components of its key color outside the union of all legend boxes.
    - Output the full grid: restore the background everywhere else, keep each mini-legend in place as a black box, and fill its middle row with evenly spaced dots using one shared tally color.

    Key insight: The black boxes are no longer one crop target but multiple local templates, so solve by locating every legend first, filtering counts by the visible key colors only, and then rendering all answers with the same dot color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    legends, legends_mask = _detect_legends(g)
    assert legends

    out = np.full((H, W), int(bg), dtype=int)
    color_counts: dict[int, int] = {}

    for bbox, pw, color in legends:
        if color not in color_counts:
            mask = (g == int(color)) & (~legends_mask)
            _lab, n = label_mask(mask, connectivity=8)
            color_counts[color] = int(n)

        patch = np.full((LEGEND_HEIGHT, pw), int(BLACK), dtype=int)
        for i in range(color_counts[color]):
            cc = int(LEGEND_KEY_COL + TALLY_COL_STEP * i)
            if 0 <= cc < pw:
                patch[LEGEND_KEY_ROW, cc] = int(TALLY_COLOR)

        r0, c0, _r1, _c1 = bbox
        paste(out, patch, r0, c0)

    return out


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(INPUT_COLOR_POOL)
    bg = random.choice(colors)

    remaining = [c for c in colors if c != bg]
    k = _randint(KEY_COUNT_RANGE)
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]

    dmax = min(DISTRACTOR_COLOR_COUNT_RANGE[1], len(remaining))
    dmin = min(DISTRACTOR_COLOR_COUNT_RANGE[0], dmax)
    nd = random.randint(dmin, dmax)
    distractor_colors = random.sample(remaining, nd)
    return bg, key_colors, distractor_colors


def _sample_key_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _sample_distractor_counts(n: int, minimum_total: int) -> list[int]:
    counts = [_randint(DISTRACTOR_COUNT_RANGE) for _ in range(n)]
    cap = DISTRACTOR_COUNT_RANGE[1]
    minimum_total = min(int(minimum_total), n * cap)
    while sum(counts) < minimum_total:
        growable = [i for i, count in enumerate(counts) if count < cap]
        if not growable:
            break
        idx = random.choice(growable)
        counts[idx] += 1
    return counts


def _legend_width(count: int) -> int:
    return TALLY_COL_STEP * count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)


def _build_input_legend(color: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), BLACK, dtype=int)
    patch[LEGEND_KEY_ROW, LEGEND_KEY_COL] = int(color)
    return patch


def _build_output_legend(count: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), BLACK, dtype=int)
    for j in range(count):
        patch[LEGEND_KEY_ROW, LEGEND_KEY_COL + TALLY_COL_STEP * j] = int(TALLY_COLOR)
    return patch


def _build_sprite_queue(
    key_colors: list[int],
    key_counts: list[int],
    distractor_colors: list[int],
    distractor_counts: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, count in zip(key_colors, key_counts):
        sprite = _make_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    for color, count in zip(distractor_colors, distractor_counts):
        sprite = _make_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    sprites.sort(key=lambda patch: int(np.count_nonzero(patch != bg)), reverse=True)
    return sprites


def _new_grids(bg: int) -> tuple[np.ndarray, np.ndarray]:
    H = _randint(GRID_HEIGHT_RANGE)
    W = _randint(GRID_WIDTH_RANGE)
    empty = np.full((H, W), int(bg), dtype=int)
    return empty.copy(), empty.copy()


def _place_legends(
    inp: np.ndarray,
    expected: np.ndarray,
    legend_specs: list[tuple[np.ndarray, np.ndarray]],
    bg: int,
) -> bool:
    for in_patch, out_patch in sorted(legend_specs, key=lambda item: item[0].shape[1], reverse=True):
        try:
            r0, c0 = random_free_location(
                inp,
                in_patch,
                transparent=bg,
                border=LEGEND_BORDER,
                padding=LEGEND_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(inp, in_patch, r0, c0)
        paste(expected, out_patch, r0, c0)
    return True


def _place_sprites(grid: np.ndarray, sprites: list[np.ndarray], bg: int) -> bool:
    for sprite in sprites:
        try:
            sr, sc = random_free_location(
                grid,
                sprite,
                transparent=bg,
                border=1,
                padding=SPRITE_PADDING,
                padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return False
        paste(grid, sprite, sr, sc, transparent=bg)
    return True


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legends, legends_mask = _detect_legends(g)
    key_colors = {int(color) for _bbox, _pw, color in legends}

    distractor_colors = []
    for color in sorted(set(map(int, g.flatten()))):
        if color in {bg, BLACK, TALLY_COLOR} or color in key_colors:
            continue
        distractor_colors.append(int(color))

    distractor_components = 0
    for color in distractor_colors:
        mask = (g == int(color)) & (~legends_mask)
        _lab, n = label_mask(mask, connectivity=8)
        distractor_components += int(n)

    if legends:
        r0 = min(bbox[0] for bbox, _pw, _color in legends)
        c0 = min(bbox[1] for bbox, _pw, _color in legends)
        r1 = max(bbox[2] for bbox, _pw, _color in legends)
        c1 = max(bbox[3] for bbox, _pw, _color in legends)
        union_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        occupied_area = sum(LEGEND_HEIGHT * pw for _bbox, pw, _color in legends)
        spread_gap = int(union_area - occupied_area)
    else:
        spread_gap = 0

    return {
        "legend_count": int(len(legends)),
        "distractor_color_count": int(len(distractor_colors)),
        "distractor_component_count": int(distractor_components),
        "spread_gap": int(spread_gap),
        "input_has_tally_color": int((g == TALLY_COLOR).any()),
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, key_colors, distractor_colors = _sample_palette()
    key_counts = _sample_key_counts(len(key_colors))
    min_distractors = max(MIN_TOTAL_DISTRACTOR_COMPONENTS, len(key_colors))
    distractor_counts = _sample_distractor_counts(len(distractor_colors), min_distractors)

    inp, expected = _new_grids(bg)

    legend_specs = []
    for color, count in zip(key_colors, key_counts):
        width = _legend_width(count)
        legend_specs.append((_build_input_legend(color, width), _build_output_legend(count, width)))

    if not _place_legends(inp, expected, legend_specs, bg):
        return None

    sprites = _build_sprite_queue(key_colors, key_counts, distractor_colors, distractor_counts, bg)
    if not _place_sprites(inp, sprites, bg):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    metrics = _fusion_metrics(inp)
    if metrics["legend_count"] != len(key_colors):
        return None
    if metrics["legend_count"] < 2:
        return None
    if metrics["distractor_color_count"] != len(distractor_colors):
        return None
    if metrics["distractor_color_count"] < DISTRACTOR_COLOR_COUNT_RANGE[0]:
        return None
    if metrics["distractor_component_count"] != sum(distractor_counts):
        return None
    if metrics["distractor_component_count"] < min_distractors:
        return None
    if metrics["input_has_tally_color"] != 0:
        return None
    if metrics["spread_gap"] <= 0:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid combo input for 58490d8a")
