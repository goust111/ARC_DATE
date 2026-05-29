# Combo variant of 58490d8a (index 11)
# combo_spec: D3/2+D9/1
# Name: 分散图例框+精灵消失

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
#    - Source task:
#      one tall black legend box lists several key colors in one shared column;
#      the solver counts outside connected components of each key color and returns only
#      the cropped legend patch with rightward every-other-cell tally dots.
#    - D3/2 changes the destination structure:
#      there is no longer one shared legend. Instead, each key color owns its own 3-row
#      black mini-box placed elsewhere in the grid. Each mini-box starts with exactly one
#      key pixel at row 1, col 1, and later receives a tally on that same row.
#    - D9/1 changes the output canvas:
#      the output is the whole grid, not a crop. Key-colored sprites outside legends are
#      erased to background, while the mini-boxes are repainted in place with their tallies.
#    - Example trace:
#      suppose there are three mini-boxes: red, green, yellow. Outside them sit
#      3 red sprites, 1 green sprite, 4 yellow sprites, plus an optional blue distractor.
#      Original task logic would fail immediately because there is no single biggest legend
#      crop to isolate. The combo solver must instead detect all three mini-boxes, union
#      their bboxes into an "inside legends" mask, count red/green/yellow components only
#      outside that mask, erase those counted sprites from the full scene, and repaint the
#      three boxes as red dots at cols 1,3,5; green at col 1; yellow at cols 1,3,5,7.
#    - Visible difference from the source task:
#      multiple separate black boxes remain in place across the full output, and the counted
#      sprites disappear from the background rather than merely being summarized in a crop.
#
# b) Complexity calibration:
#    - Keep only 2-4 legend boxes so the relationship "one box per key color" is repeated
#      but not overcrowded.
#    - Force at least one count > 1 so some mini-box visibly grows beyond its lone key pixel.
#    - Use 3-row legends with a fixed key row/col to keep detection simple and human-visible.
#    - Allow at most one distractor color so preserved non-key content can demonstrate that
#      erasure is selective, without introducing too many extra object classes.
#
# c) Implicit assumption audit against the original main():
#    - "largest black CC is the legend" no longer holds: there are several separate legend CCs.
#    - "all keys live in one crop and one column" no longer holds: each legend is independent.
#    - "output is black crop-sized canvas" no longer holds: the output must preserve the full
#      grid layout while selectively erasing counted sprites.
#    - New combo assumptions introduced by this solver:
#      every true legend is a 3-row black CC whose non-black cells all lie on row 1, share a
#      single color, start at col 1, and continue with step 2 spacing. The generator enforces
#      exactly that template and keeps legends isolated by padding.
#
# d) Placement legality:
#    - Legends are placed first with border and padding so they are separate black CCs and
#      never touch each other or later sprites under 8-connectivity.
#    - Sprites are then placed with the same padding rule, preventing merges that would change
#      connected-component counts.
#    - Each mini-legend width is derived from its own target count, so all dots fit locally.
#
# e) Visual observability:
#    - Across examples, a human can directly see two coupled regularities:
#      1) several separated black mini-boxes act as independent local legends;
#      2) keyed sprites vanish from the scene and reappear as dots inside the matching box.
#    - The mapping is structural, not numeric: box color determines which outside sprites are
#      counted and erased.
#
# f) Visual cleanliness:
#    - The output retains the spacious original grid, but only keyed sprites disappear and only
#      the existing mini-box interiors change. No extra overlay is introduced outside the boxes.
#    - Non-touching placement and small 3-row legends keep the scene readable.
#
# g) Change inferability:
#    - No ordering by color id or other hidden feature is used. The only match signal is the
#      visible key color inside each mini-box.
#    - Because every sample repeats the same row geometry (row 1, cols 1,3,5,...) and the same
#      erase behavior on the matched color, the combined rule is recoverable from few examples.
# ---------------------------------------------------------------------------

BLACK = 0
COLOR_POOL = tuple(range(1, 10))
DISTRACTOR_PROB = 0.6

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
LEGEND_HEIGHT = 3
LEGEND_KEY_ROW = 1
LEGEND_KEY_COL = 1
TALLY_COL_STEP = 2
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 2)

GRID_HEIGHT_RANGE = (24, 30)
GRID_WIDTH_RANGE = (24, 30)
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


def _legend_width(count: int) -> int:
    return TALLY_COL_STEP * count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)


def _build_input_legend(color: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), BLACK, dtype=int)
    patch[LEGEND_KEY_ROW, LEGEND_KEY_COL] = int(color)
    return patch


def _build_output_legend(color: int, count: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), BLACK, dtype=int)
    for j in range(count):
        patch[LEGEND_KEY_ROW, LEGEND_KEY_COL + TALLY_COL_STEP * j] = int(color)
    return patch


def _detect_legends(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int, int, int]]:
    g = np.array(grid, dtype=int)
    legends: list[tuple[tuple[int, int, int, int], int, int, int]] = []
    for coords in components_mask(g == int(BLACK), connectivity=8):
        if coords.size == 0:
            continue
        bbox = tuple(map(int, bbox_coords(coords)))
        patch = crop_bbox(g, bbox)
        ph, pw = map(int, patch.shape)
        if ph != LEGEND_HEIGHT or pw < 3:
            continue
        nonblack = np.argwhere(patch != int(BLACK))
        if nonblack.size == 0:
            continue

        rows = {int(r) for r, _c in nonblack}
        colors = {int(patch[int(r), int(c)]) for r, c in nonblack}
        cols = sorted(int(c) for _r, c in nonblack)
        if rows != {LEGEND_KEY_ROW} or len(colors) != 1:
            continue
        if cols[0] != LEGEND_KEY_COL:
            continue
        if any((cc - LEGEND_KEY_COL) % TALLY_COL_STEP != 0 for cc in cols):
            continue

        legends.append((bbox, ph, pw, next(iter(colors))))
    legends.sort(key=lambda item: item[0])
    return legends


def _legend_mask(shape: tuple[int, int], legends: list[tuple[tuple[int, int, int, int], int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for bbox, _ph, _pw, _color in legends:
        r0, c0, r1, c1 = bbox
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    return mask


def _build_expected_full(
    grid: np.ndarray,
    legends: list[tuple[tuple[int, int, int, int], int, int, int]],
    counts: dict[int, int],
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    legends_mask = _legend_mask(out.shape, legends)

    for _bbox, _ph, _pw, color in legends:
        out[(out == int(color)) & (~legends_mask)] = int(bg)

    for bbox, ph, pw, color in legends:
        r0, c0, _r1, _c1 = bbox
        patch = np.full((ph, pw), int(BLACK), dtype=int)
        for i in range(int(counts.get(int(color), 0))):
            patch[LEGEND_KEY_ROW, LEGEND_KEY_COL + TALLY_COL_STEP * i] = int(color)
        paste(out, patch, int(r0), int(c0))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed sprites across several scattered mini-legends, erase them, and rewrite each legend locally.

    Input structure:
    - Multiple separate black 3-row mini-boxes are scattered around the grid, each containing one key-color pixel on its middle row.
    - Outside those boxes, disconnected sprites of the keyed colors appear on a common background, with an optional extra distractor color.

    Rule:
    - For each mini-box, count the number of same-colored connected components outside the union of all legend boxes.
    - Output the full grid: erase those keyed sprites back to the background and repaint each mini-box as a black tally strip with evenly spaced same-color dots at columns 1,3,5,...

    Key insight:
    - Each black mini-box is an independent local legend, but all of them read counts from the shared outside scene; the answer keeps the whole scene layout while moving the counted information into the boxes and removing the counted sprites themselves.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    legends = _detect_legends(g)
    if not legends:
        return g.copy()

    legends_mask = _legend_mask(g.shape, legends)
    counts: dict[int, int] = {}
    for _bbox, _ph, _pw, color in legends:
        if int(color) in counts:
            continue
        mask = (g == int(color)) & (~legends_mask)
        _lab, n = label_mask(mask, connectivity=8)
        counts[int(color)] = int(n)

    return _build_expected_full(g, legends, counts, bg)


def _sample_palette() -> tuple[int, list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)
    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor


def _sample_counts(k: int) -> list[int]:
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if any(count > 1 for count in counts):
            return counts


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, count in zip(key_colors, counts):
        sprite = _make_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        sprites.append(_make_sprite(distractor_color, bg))
    random.shuffle(sprites)
    return sprites


def _new_grid(bg: int) -> np.ndarray:
    H = _randint(GRID_HEIGHT_RANGE)
    W = _randint(GRID_WIDTH_RANGE)
    return np.full((H, W), int(bg), dtype=int)


def _place_legends(
    grid: np.ndarray,
    legend_specs: list[tuple[int, int, np.ndarray, np.ndarray]],
    bg: int,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]] | None:
    placements: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for _color, _count, in_patch, out_patch in sorted(legend_specs, key=lambda item: item[2].shape[1], reverse=True):
        try:
            r0, c0 = random_free_location(
                grid,
                in_patch,
                transparent=bg,
                border=LEGEND_BORDER,
                padding=LEGEND_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, in_patch, r0, c0)
        h, w = map(int, in_patch.shape)
        placements.append(((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)), out_patch.copy()))
    return placements


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


def _expected_from_input(
    grid: np.ndarray,
    placements: list[tuple[tuple[int, int, int, int], np.ndarray]],
    key_colors: list[int],
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    legends_mask = np.zeros(out.shape, dtype=bool)
    for bbox, _patch in placements:
        r0, c0, r1, c1 = bbox
        legends_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    for color in key_colors:
        out[(out == int(color)) & (~legends_mask)] = int(bg)

    for bbox, patch in placements:
        r0, c0, _r1, _c1 = bbox
        paste(out, patch, int(r0), int(c0))
    return out


def _legend_layout_metrics(grid: np.ndarray) -> tuple[int, int, int, int]:
    bboxes = [bbox for bbox, _ph, _pw, _color in _detect_legends(grid)]
    if not bboxes:
        return 0, 0, 0, 0
    total_area = sum((b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in bboxes)
    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    union_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    row_span = int(r1 - r0)
    col_span = int(c1 - c0)
    return len(bboxes), int(union_area - total_area), row_span, col_span


def _fusion_metrics(inp: np.ndarray, out: np.ndarray, key_colors: list[int]) -> tuple[int, int, int]:
    changed_outside = 0
    remaining_key_pixels = 0
    preserved_nonkey = 0
    legends_mask = _legend_mask(inp.shape, _detect_legends(inp))
    for color in key_colors:
        diff = (inp == int(color)) & (out != int(color)) & (~legends_mask)
        changed_outside += int(diff.sum())
        remaining_key_pixels += int(((out == int(color)) & (~legends_mask)).sum())
    preserved_nonkey = int(((inp == out) & (~legends_mask) & (inp != get_bg(inp))).sum())
    return changed_outside, remaining_key_pixels, preserved_nonkey


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, key_colors, distractor = _sample_palette()
    counts = _sample_counts(len(key_colors))
    grid = _new_grid(bg)

    legend_specs = []
    for color, count in zip(key_colors, counts):
        width = _legend_width(count)
        legend_specs.append(
            (
                int(color),
                int(count),
                _build_input_legend(color, width),
                _build_output_legend(color, count, width),
            )
        )

    placements = _place_legends(grid, legend_specs, bg)
    if placements is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor, bg)
    if not _place_sprites(grid, sprites, bg):
        return None

    expected = _expected_from_input(grid, placements, key_colors, bg)
    if not np.array_equal(main(grid), expected):
        return None

    n_legends, spread_gap, row_span, col_span = _legend_layout_metrics(grid)
    if n_legends < 2 or spread_gap <= 0:
        return None
    if row_span < LEGEND_HEIGHT + 1 or col_span < 4:
        return None

    changed_outside, remaining_key_pixels, preserved_nonkey = _fusion_metrics(grid, expected, key_colors)
    if changed_outside <= 0:
        return None
    if remaining_key_pixels != 0:
        return None
    if distractor is not None and preserved_nonkey <= 0:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            grid, _expected = pair
            return grid
    raise RuntimeError("failed to generate a valid combo input for 58490d8a")
