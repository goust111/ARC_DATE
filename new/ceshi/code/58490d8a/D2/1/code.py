# Variant of 58490d8a
# Dimension: D2 — 源对象：框外精灵的定义 (src)
# Change: 框外各关键颜色的独立连通分量（8连通），每种颜色的所有精灵形状相同，外接矩形2x2至3x3，面积4-7像素... → 不同形状的同色精灵

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the D2 variant
# ----------------------------------
# concrete_behavior:
#   legend_box:
#     identical to the source task: one black rectangle, one key-color pixel on each odd key row,
#     output is still a black tally box with same-sized colored dots.
#   outside_sprites:
#     still monochrome, 8-connected, bbox in {2x2, 2x3, 3x2, 3x3}, area 4-7, and mutually non-touching.
#     The changed part is intra-color diversity: if color 2 appears three times, those three components may
#     now use masks A/B/C instead of forced copies of A.
# flow_trace_example:
#   original_generation:
#     key color red, count=3 -> sample one 3x3 mask A -> place A, A, A -> solver counts 3 red CCs.
#   variant_generation:
#     key color red, count=3 -> sample masks A, B, C -> place A, B, C -> solver still counts 3 red CCs.
#   difference:
#     the input now contains a visible same-color / different-shape pattern, while the output remains the
#     same tally because grouping is by color only.
# complexity_calibration:
#   We keep the original palette size, object count range, and bbox/area constraints so the task does not
#   drift into another dimension. To make D2 visible in 2-4 examples, generation guarantees at least one
#   key color has count >= 2 and uses >= 2 distinct shapes.
# solver_audit:
#   - legend detection by the largest black connected component remains valid because sprites stay non-black.
#   - key-column detection remains valid because the legend structure is unchanged.
#   - counting per key color via label_mask((grid == color) & outside, connectivity=8) already ignores
#     shape identity, so it is exactly the invariant we want to preserve.
#   - therefore no shape-matching code should be introduced; the solver must keep using only color + CCs.
# placement_legality:
#   random_free_location(..., padding=1, padding_connectivity=8) still enforces a one-cell moat between all
#   sprites and between sprites and the legend box, so different same-color shapes never merge accidentally.
# visual_observability:
#   forcing at least one repeated key color with multiple distinct masks prevents degeneration into the
#   original regime where every color appears once or repeated copies look identical.
# cleanliness_and_color_inference:
#   no new color remapping is introduced; colors keep their original role as legend keys, so the only new
#   burden on the solver is to ignore shape variation when grouping components.
# main_assumptions_after_change:
#   - there is exactly one legend-like black component with embedded non-black key pixels;
#   - each counted outside object is a separate 8-connected monochrome component;
#   - same-color components may differ in geometry, and generate_input() explicitly constructs such cases.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a legend box and tally outside components by color, even when same-color sprites use different shapes.

    Input structure:
    - A black legend rectangle contains one colored key pixel on every other row of its leftmost non-black column.
    - Outside the box, each key color appears as 1-5 separate 8-connected sprites; matching colors may have different 2x2..3x3 shapes with area 4-7.

    Rule:
    - For each key color, ignore geometry and count only how many outside 8-connected components use that color.
    - Return a black box of the same size, marking that many evenly spaced dots on the corresponding key row at key_col, key_col+2, key_col+4, ...

    Key insight:
    - The distracting part of the variant is shape diversity inside one color class, but the stable signal is still color membership: each legend row names a color, and the answer is that color's outside component count.
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
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
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

GENERATION_ATTEMPTS = 300


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


def _make_sprite_variant(color: int, bg: int, exclude: set[tuple[int, ...]] | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    exclude = exclude or set()
    choices = [(mask, sig) for mask, sig in _all_mask_options() if sig not in exclude]
    if not choices:
        raise ValueError("no unused sprite masks available")
    mask, sig = random.choice(choices)
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite, sig


def _sample_palette() -> tuple[int, list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
    if max(counts) == 1:
        idx = random.randrange(k)
        counts[idx] = random.randint(2, TARGET_COUNT_RANGE[1])
    return counts


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


def _build_color_sprites(color: int, count: int, bg: int, force_multiple_shapes: bool) -> tuple[list[np.ndarray], int]:
    max_variants = min(int(count), MAX_SHAPE_VARIANTS_PER_COLOR, len(_all_mask_options()))
    min_variants = 2 if force_multiple_shapes else 1
    if max_variants < min_variants:
        raise ValueError("cannot satisfy requested shape diversity")

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


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    eligible = [i for i, count in enumerate(counts) if count >= 2]
    anchor = random.choice(eligible)

    for i, (color, count) in enumerate(zip(key_colors, counts)):
        force_multiple = i == anchor
        color_sprites, _variant_count = _build_color_sprites(color, count, bg, force_multiple_shapes=force_multiple)
        sprites.extend(color_sprites)

    if distractor_color is not None:
        distractor_sprite, _sig = _make_sprite_variant(distractor_color, bg)
        sprites.append(distractor_sprite)

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
    raise RuntimeError("failed to generate a valid 58490d8a D2 variant input")
