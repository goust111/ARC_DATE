# Combo variant of 58490d8a (index 4)
# combo_spec: D2/1+D7/1
# Name: 异形同色精灵+多干扰色

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   legend_box:
#     The legend remains one black rectangle with one colored key pixel on each odd key row.
#     The output is still a black tally panel of the same size.
#   outside_sprites:
#     Counted objects are still monochrome 8-connected components with bbox in {2x2, 2x3, 3x2, 3x3}
#     and area 4-7. The D2 change is that repeated key-color components are no longer forced to share
#     one mask; for one visible anchor color, multiple same-color shapes must coexist.
#   distractors:
#     The D7 change adds 2-3 extra colors outside the legend, each with 1-3 components. They never
#     appear in the legend and therefore must be ignored by the solver even though their total count is
#     kept in the same rough range as the key-color components.
# flow_trace_example:
#   original_generation:
#     Suppose the legend keys are red and green with target counts [3, 2]. Red might appear as A, A, A;
#     green as B, B; and there may be at most one blue distractor C. The solver counts red CCs and green
#     CCs, then writes three red dots and two green dots.
#   combo_generation:
#     Red can now appear as A, D, A while green stays B, B. In addition, there may be blue C, C and
#     yellow E and orange F, F outside the box. The solver must still output only the three red dots and
#     two green dots because grouping is by legend color, not by shape and not by mere presence outside.
#   difference:
#     The same-color / different-shape cue and the multi-color distractor clutter happen in the same grid,
#     so the human has to isolate the invariant: count only outside CCs whose color is listed in the box.
# complexity_calibration:
#   We keep the original box geometry, sprite bbox options, area range, and 8-connectivity. To make the
#   combo visible without overcrowding the 20-25 canvas, generation restricts the total number of key
#   components to a midrange band and then samples distractor totals with a 0.5-1.25 ratio to that key
#   total. This keeps the non-key clutter substantial but still placeable.
# solver_audit:
#   - Finding the legend as the largest black component with embedded non-black cells still works because
#     added distractors are non-black and remain separated from the legend by a one-cell 8-neighborhood moat.
#   - Detecting the key column as the leftmost non-black column inside that bbox still works because the
#     legend template is unchanged.
#   - Counting components with label_mask((grid == color) & outside, connectivity=8) is exactly the desired
#     invariant: it ignores shape differences and automatically ignores colors absent from the legend rows.
#   - Therefore the solver should not add any shape matching or distractor classification logic; only the
#     generator changes to enforce richer inputs.
# placement_legality:
#   Every sprite is placed with random_free_location(..., padding=1, padding_connectivity=8), so no two
#   components touch each other or the legend box. This guarantees that same-color components never merge
#   and that distractors cannot accidentally alter the legend's black connected component.
# visual_observability:
#   At least one key color is forced to have count >= 2 and at least two distinct masks, so D2 is visible
#   in nearly every sample. Separately, distractors always use 2-3 distinct non-key colors, so D7 is also
#   visible from a few examples instead of appearing as a rare edge case.
# visual_cleanliness:
#   All objects remain small monochrome blobs on a uniform background, with a one-cell moat between them.
#   That keeps the grid readable even when several distractor colors are present.
# inferability_of_change:
#   No hidden ordering uses numeric color ids. The only usable clue is structural: colors listed in the box
#   define which outside components count, and the output stays an evenly spaced dot sequence per key row.
# main_assumptions_after_change:
#   - Exactly one legend-like black component contains non-black key pixels.
#   - Counted and distractor sprites are separate monochrome 8-connected components.
#   - At least one repeated key color uses multiple shapes, and distractor colors are never reused as keys.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the legend box, ignore same-color shape variation and extra distractor colors, and tally only the keyed colors.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on each odd row of its leftmost non-black column.
    - Outside the box, each key color appears as 1-5 separate 8-connected sprites whose same-color shapes may differ, while 2-3 additional non-key colors also appear as extra sprites.

    Rule:
    - For every legend color, count the outside 8-connected components of that color only; do not merge or classify by geometry, and ignore all colors absent from the legend.
    - Return a black box of the same size, drawing one dot per counted component on the matching key row at columns key_col, key_col+2, key_col+4, ...

    Key insight:
    - The stable signal is the legend's color list: outside shape diversity and multi-color clutter are both distractions, so the task reduces to color-filtered connected-component counting.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    best = None
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
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, color in keys:
        color_mask = (g == int(color)) & outside
        _labels, n_components = label_mask(color_mask, connectivity=8)
        for i in range(int(n_components)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)

    return out


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_TOTAL_RANGE = (4, 9)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
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
MAX_SHAPE_VARIANTS_PER_COLOR = 3

DISTRACTOR_COLOR_COUNT_RANGE = (2, 3)
DISTRACTOR_SPRITE_COUNT_RANGE = (1, 3)
DISTRACTOR_TOTAL_RATIO_RANGE = (0.5, 1.25)

GENERATION_ATTEMPTS = 400


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
        _, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n == 1:
            masks.append(mask)
    return tuple(masks)


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    flat = tuple(int(v) for v in mask.astype(int).flatten())
    return (int(mask.shape[0]), int(mask.shape[1]), *flat)


@lru_cache(maxsize=1)
def _all_mask_options() -> tuple[tuple[np.ndarray, tuple[int, ...]], ...]:
    options: list[tuple[np.ndarray, tuple[int, ...]]] = []
    for area in range(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1):
        for h, w in SPRITE_BBOXES:
            if h * w < area:
                continue
            for mask in _valid_masks(h, w, area):
                options.append((mask, _mask_signature(mask)))
    return tuple(options)


def _make_sprite_variant(
    color: int,
    bg: int,
    exclude: set[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    exclude = exclude or set()
    choices = [(mask, sig) for mask, sig in _all_mask_options() if sig not in exclude]
    if not choices:
        raise ValueError("no unused sprite masks available")
    mask, sig = random.choice(choices)
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite, sig


def _make_canonical_sprite(color: int, bg: int) -> np.ndarray:
    sprite, _sig = _make_sprite_variant(color, bg)
    return sprite


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]

    distractor_color_count = random.randint(
        DISTRACTOR_COLOR_COUNT_RANGE[0],
        min(DISTRACTOR_COLOR_COUNT_RANGE[1], len(remaining)),
    )
    distractor_colors = random.sample(remaining, distractor_color_count)
    return bg, key_colors, distractor_colors


def _sample_counts(k: int) -> list[int]:
    for _ in range(200):
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) == 1:
            counts[random.randrange(k)] = random.randint(2, TARGET_COUNT_RANGE[1])
        total = sum(counts)
        if KEY_TOTAL_RANGE[0] <= total <= KEY_TOTAL_RANGE[1]:
            return counts
    raise ValueError("failed to sample key counts in the target range")


def _sample_distractor_counts(n_colors: int, key_total: int) -> list[int]:
    options = []
    for counts in product(
        range(DISTRACTOR_SPRITE_COUNT_RANGE[0], DISTRACTOR_SPRITE_COUNT_RANGE[1] + 1),
        repeat=int(n_colors),
    ):
        total = sum(counts)
        ratio = total / float(key_total)
        if DISTRACTOR_TOTAL_RATIO_RANGE[0] <= ratio <= DISTRACTOR_TOTAL_RATIO_RANGE[1]:
            options.append(list(counts))
    if not options:
        raise ValueError("no distractor-count configuration matches the ratio constraint")
    return random.choice(options)


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
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


def _split_count(total: int, n_parts: int) -> list[int]:
    counts = [1] * n_parts
    for _ in range(total - n_parts):
        counts[random.randrange(n_parts)] += 1
    random.shuffle(counts)
    return counts


def _build_key_color_sprites(
    color: int,
    count: int,
    bg: int,
    force_multiple_shapes: bool,
) -> tuple[list[np.ndarray], int]:
    max_variants = min(int(count), MAX_SHAPE_VARIANTS_PER_COLOR, len(_all_mask_options()))
    min_variants = 2 if force_multiple_shapes else 1
    if max_variants < min_variants:
        raise ValueError("cannot satisfy requested key-color shape diversity")

    variant_count = random.randint(min_variants, max_variants)
    repetitions = _split_count(int(count), variant_count)

    used_signatures: set[tuple[int, ...]] = set()
    variants = []
    for _ in range(variant_count):
        sprite, sig = _make_sprite_variant(color, bg, exclude=used_signatures)
        used_signatures.add(sig)
        variants.append(sprite)

    sprites: list[np.ndarray] = []
    for sprite, reps in zip(variants, repetitions):
        for _ in range(reps):
            sprites.append(sprite.copy())
    return sprites, variant_count


def _build_distractor_sprites(colors: list[int], counts: list[int], bg: int) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    for color, count in zip(colors, counts):
        sprite = _make_canonical_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    return sprites


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_colors: list[int],
    distractor_counts: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    eligible = [i for i, count in enumerate(counts) if count >= 2]
    if not eligible:
        raise ValueError("need at least one repeated key color to show the D2 change")
    anchor = random.choice(eligible)

    for i, (color, count) in enumerate(zip(key_colors, counts)):
        force_multiple = i == anchor
        color_sprites, _variant_count = _build_key_color_sprites(color, count, bg, force_multiple_shapes=force_multiple)
        sprites.extend(color_sprites)

    sprites.extend(_build_distractor_sprites(distractor_colors, distractor_counts, bg))
    random.shuffle(sprites)
    return sprites


def _place_sprites(grid: np.ndarray, sprites: list[np.ndarray], bg: int) -> None:
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
    try:
        bg, key_colors, distractor_colors = _sample_palette()
        counts = _sample_counts(len(key_colors))
        distractor_counts = _sample_distractor_counts(len(distractor_colors), sum(counts))
    except ValueError:
        return None

    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    try:
        sprites = _build_sprite_queue(key_colors, counts, distractor_colors, distractor_counts, bg)
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
