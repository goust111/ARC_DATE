# Combo variant of 58490d8a (index 18)
# combo_spec: D2/1+D11/0
# Name: 异形同色精灵+4连通计数

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   - The legend box stays structurally identical to the source task: one black rectangle, one key pixel on
#     each odd legend row, and the output is still a black tally canvas with evenly spaced dots.
#   - The change happens in the outside sprites. A single visible same-color sprite may now be an 8-connected
#     motif that splits into multiple 4-connected pieces because some links are only diagonal.
#   - Same-color visible sprites are no longer forced to reuse one canonical mask. For one color with total
#     count 4, the original task could place A+A+A+A and count four 8-connected objects. Here we can place
#     a unit-3 diagonal motif plus a unit-1 block; visually that is two different same-color shapes, but the
#     tally row still shows four dots because the first visible sprite contributes three 4-components.
# complexity_calibration:
#   - To keep the task inferable, masks are still tiny (bbox 2x2..3x3, area 4..7) and all objects remain
#     monochrome and well separated by an 8-neighbor moat.
#   - We restrict the per-sprite 4-component contribution to 1..5, matching the original tally scale, and
#     only use short partitions of the target count so the scene does not become cluttered.
#   - Each generated sample forces one anchor color with count >= 3 whose visible sprites use at least two
#     different masks and whose total 4-count is larger than its visible sprite count, making both dimensions
#     observable in the same color family.
# hidden_assumptions_audit:
#   - `main()` finds the legend as the largest black 8-connected component whose bbox contains non-black
#     pixels. This still works because all outside sprites are non-black and never touch the box.
#   - `main()` uses the leftmost non-black legend column as the key column. The generator preserves that by
#     only painting one colored key pixel per odd row at column 1 inside the legend patch.
#   - The only solver line that changes semantically is the per-color component count: the old 8-connectivity
#     assumption is invalid once diagonal-only bridges are allowed inside a sprite, so counting must switch
#     to 4-connectivity.
#   - Excluding the legend bbox from the outside mask assumes no sprite overlaps the box. The placement code
#     guarantees a strict gap via `random_free_location(..., padding=1, padding_connectivity=8)`.
# placement_legality:
#   - Every sampled mask is prevalidated to be 8-connected as a whole, have bbox in {2x2, 2x3, 3x2, 3x3},
#     use area 4..7, and span the bbox so it reads as a compact sprite instead of a floating fragment.
#   - Sprite placement uses the background color as transparency plus one-cell 8-neighbor padding, so no two
#     placed sprites can merge under either 4- or 8-connectivity and no sprite can touch the legend.
# visual_observability_and_cleanliness:
#   - Human-solvable evidence comes from repeated color rows: one row shows two or three same-color shapes
#     that visibly differ, yet its output has more dots than visible objects because diagonal links split.
#   - We avoid hidden ordering cues or color-value tricks; grouping is only by the legend colors and the
#     count evidence comes directly from visible motifs versus tally length.
# main_assumptions_after_change:
#   - There is exactly one legend-like black rectangle with embedded key pixels.
#   - Outside sprites are pairwise separated by at least one 8-neighbor cell.
#   - A color's answer equals the number of 4-connected components in all of that color's outside pixels,
#     and `generate_input()` explicitly constructs at least one sample where this differs from 8-connectivity.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a legend box and tally same-color fragments with 4-connectivity, even when one color uses mixed shapes.

    Input structure:
    - A black legend rectangle contains one key-colored pixel on every other row of its first non-black column.
    - Outside the box, each key color appears as 1-5 small monochrome sprites; sprites of the same color may
      use different 2x2..3x3 masks, and some masks stay visually connected only through diagonal contact.

    Rule:
    - For each key color, look only outside the legend box and count same-colored connected components using
      4-connectivity, so corner-touching pixels must be split apart.
    - Return a black box of the same size, placing one dot of that color at key_col, key_col+2, key_col+4,
      ... for every counted 4-connected component on the matching key row.

    Key insight:
    - Shape identity is now a distraction twice over: same-color sprites need not match each other, and one
      visible sprite can contribute multiple tally dots if its pixels are joined only diagonally.
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
        _labels, n_components = label_mask(color_mask, connectivity=4)
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

GENERATION_ATTEMPTS = 300

PARTITIONS: dict[int, tuple[tuple[int, ...], ...]] = {
    1: ((1,),),
    2: ((2,), (1, 1)),
    3: ((3,), (2, 1), (1, 1, 1)),
    4: ((4,), (3, 1), (2, 2), (2, 1, 1)),
    5: ((5,), (4, 1), (3, 2), (3, 1, 1), (2, 2, 1)),
}


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
        _, n8 = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n8 == 1:
            masks.append(mask)
    return tuple(masks)


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return (int(mask.shape[0]), int(mask.shape[1]), *(int(v) for v in mask.astype(int).flatten()))


@lru_cache(maxsize=1)
def _mask_records() -> tuple[tuple[np.ndarray, tuple[int, ...], int], ...]:
    records: list[tuple[np.ndarray, tuple[int, ...], int]] = []
    for area in range(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1):
        for h, w in SPRITE_BBOXES:
            if h * w < area:
                continue
            for mask in _valid_masks(h, w, area):
                sig = _mask_signature(mask)
                _lab4, n4 = label_mask(mask, connectivity=4)
                records.append((mask.copy(), sig, int(n4)))
    return tuple(records)


@lru_cache(maxsize=1)
def _mask_options_by_four_count() -> dict[int, tuple[tuple[np.ndarray, tuple[int, ...]], ...]]:
    grouped: dict[int, list[tuple[np.ndarray, tuple[int, ...]]]] = {}
    for mask, sig, four_count in _mask_records():
        grouped.setdefault(int(four_count), []).append((mask, sig))
    return {k: tuple(v) for k, v in grouped.items()}


def _make_sprite_from_mask(color: int, bg: int, mask: np.ndarray) -> np.ndarray:
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


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
    if max(counts) < 3:
        counts[random.randrange(k)] = random.randint(3, TARGET_COUNT_RANGE[1])
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


def _partition_options(total: int, require_split: bool) -> list[tuple[int, ...]]:
    options = list(PARTITIONS[int(total)])
    if require_split:
        options = [parts for parts in options if any(part > 1 for part in parts)]
    return options


def _distinct_mask_count(unit: int) -> int:
    return len(_mask_options_by_four_count().get(int(unit), ()))


def _can_realize_shape_diversity(total: int, require_split: bool) -> bool:
    for parts in _partition_options(int(total), require_split=require_split):
        if len(parts) < 2:
            continue
        repeated_units = {unit for unit in parts if parts.count(unit) >= 2}
        if len(set(parts)) >= 2:
            return True
        if any(_distinct_mask_count(unit) >= 2 for unit in repeated_units):
            return True
    return False


def _sample_partition(total: int, require_split: bool) -> tuple[int, ...]:
    options = _partition_options(int(total), require_split=require_split)
    if not options:
        raise ValueError("no valid partition options")
    weighted = []
    for parts in options:
        score = 1
        if len(parts) >= 2:
            score += 2
        if any(part > 1 for part in parts):
            score += 3
        weighted.extend([parts] * score)
    return random.choice(weighted)


def _choose_sprite_masks(parts: tuple[int, ...], require_shape_diversity: bool) -> list[tuple[np.ndarray, tuple[int, ...], int]]:
    selections: list[tuple[np.ndarray, tuple[int, ...], int]] = []
    used_signatures: set[tuple[int, ...]] = set()
    distinct_signatures: set[tuple[int, ...]] = set()

    for idx, unit in enumerate(parts):
        options = list(_mask_options_by_four_count()[int(unit)])
        prefer_fresh = require_shape_diversity and (idx < len(parts) - 1 or len(distinct_signatures) < 1)
        if prefer_fresh:
            fresh = [(mask, sig) for mask, sig in options if sig not in used_signatures]
            if fresh:
                options = fresh
        mask, sig = random.choice(options)
        selections.append((mask.copy(), sig, int(unit)))
        used_signatures.add(sig)
        distinct_signatures.add(sig)

    if require_shape_diversity and len(distinct_signatures) < 2:
        raise ValueError("failed to realize visible shape diversity")
    return selections


def _build_color_sprites(
    color: int,
    total_count: int,
    bg: int,
    require_split: bool,
    require_shape_diversity: bool,
) -> list[np.ndarray]:
    for _ in range(40):
        parts = _sample_partition(int(total_count), require_split=require_split)
        if require_shape_diversity:
            if len(parts) < 2:
                continue
            repeated_units = {unit for unit in parts if parts.count(unit) >= 2}
            possible = len(set(parts)) >= 2 or any(_distinct_mask_count(unit) >= 2 for unit in repeated_units)
            if not possible:
                continue
        chosen = _choose_sprite_masks(parts, require_shape_diversity=require_shape_diversity)
        sprites = [_make_sprite_from_mask(color, bg, mask) for mask, _sig, _unit in chosen]
        return sprites
    raise ValueError("unable to build color sprites with requested constraints")


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    eligible = [i for i, total in enumerate(counts) if total >= 3]
    anchor = random.choice(eligible)

    for i, (color, total) in enumerate(zip(key_colors, counts)):
        if i == anchor:
            color_sprites = _build_color_sprites(
                color,
                total,
                bg,
                require_split=True,
                require_shape_diversity=True,
            )
        else:
            require_split = random.random() < 0.6 and total >= 2
            require_shape_diversity = require_split and random.random() < 0.4 and _can_realize_shape_diversity(total, require_split=True)
            color_sprites = _build_color_sprites(
                color,
                total,
                bg,
                require_split=require_split,
                require_shape_diversity=require_shape_diversity,
            )
        sprites.extend(color_sprites)

    if distractor_color is not None:
        distractor_unit = random.choice(sorted(_mask_options_by_four_count()))
        distractor_mask, _sig = random.choice(_mask_options_by_four_count()[int(distractor_unit)])
        sprites.append(_make_sprite_from_mask(distractor_color, bg, distractor_mask))

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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
