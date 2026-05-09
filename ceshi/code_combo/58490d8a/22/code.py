# Combo variant of 58490d8a (index 22)
# combo_spec: D2/1+D7/1+D11/0
# Name: 异形精灵+多干扰+4连通

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   - The legend box stays structurally identical to the source task: one black rectangle with one key pixel
#     per odd row in the left key column, and the output is still a black tally box with evenly spaced dots.
#   - Outside the box, key colors now mix two kinds of difficulty at once:
#       1) same-color sprites may use different 2x2..3x3 masks rather than repeated copies;
#       2) some sprites are only 8-connected internally, so one visible sprite can split into multiple
#          4-connected counted pieces.
#   - Additional 2-3 distractor colors, each with 1-3 sprites, are placed outside as well. They look similar
#     in scale and complexity but never appear in the legend, so they must be ignored.
# flow_trace_example:
#   - Original: red count=3 -> make three identical 8-connected red sprites -> solver counts 3 red objects.
#   - Combo: red tally target=3 -> make two red sprites of different shapes, e.g. one mask with 4-count 2 and
#     one mask with 4-count 1 -> visually there are only two red blobs, but the correct tally is 3 because
#     the first blob splits under 4-connectivity.
#   - Add distractors: e.g. teal and orange sprites appear in similar numbers, but since they are absent from
#     the legend rows they contribute nothing to the output.
# complexity_calibration:
#   - To keep the fusion visible but not chaotic, every sprite still respects the original bbox/area envelope:
#     bbox in {2x2, 2x3, 3x2, 3x3}, area 4-7, whole-mask 8-connectivity.
#   - The 4-connectivity effect is made observable by forcing at least one key color to use a plan with a
#     sprite whose internal 4-component count exceeds 1.
#   - The D2 effect is made observable by forcing that same anchor key color to use at least two distinct
#     visible sprites with different signatures. Other key colors may also vary, but the generator does not
#     rely on too many free choices.
#   - Distractors are limited to 2-3 colors and 1-3 sprites each so they are numerous enough to matter yet
#     still separable by human inspection.
# solver_audit:
#   - Legend detection via the largest black 8-connected component whose bbox contains non-black pixels still
#     holds: neither key nor distractor sprites use black.
#   - Key-column detection via the leftmost non-black column inside the legend still holds because the legend
#     layout is unchanged.
#   - The only semantic change inside main() is the counting step: label the outside mask per key color with
#     connectivity=4 instead of 8.
#   - No shape matching is introduced. D2 explicitly requires grouping by color rather than geometry, so the
#     solver must remain indifferent to which mask variant a same-color sprite uses.
# placement_legality:
#   - All visible sprites are placed with random_free_location(..., padding=1, padding_connectivity=8), so no
#     two sprites and no sprite/legend pair can touch even diagonally.
#   - Because different visible sprites are separated by an 8-neighborhood moat, the total 4-connected count
#     for one color equals the sum of the chosen per-sprite 4-counts. This keeps the output-first generator
#     exact and rejection-checkable.
# visual_observability:
#   - A solver sees repeated evidence that shape identity is unstable inside a color class, so counting by
#     silhouette is unreliable.
#   - A solver also sees at least one color where the number of dots exceeds the number of visible same-color
#     blobs, giving direct evidence that diagonal-only contacts do not merge.
#   - Distractor colors are visually similar in density, so the legend must be used as the filter for which
#     colors matter.
# cleanliness_and_inference:
#   - The legend region remains isolated and regular, while all clutter is confined to the outside field.
#   - No color remapping is introduced: legend rows still directly name the colors to tally, which keeps the
#     fused task inferable from a few examples despite the higher visual load.
# main_assumptions_after_change:
#   - There is exactly one legend-like black component with embedded non-black key pixels.
#   - Outside sprites are monochrome and mutually non-touching under 8-connectivity.
#   - Any same-color 4-connected count increase comes only from internal diagonal-only contacts, and the
#     generator explicitly constructs such masks.

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

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

DISTRACTOR_COLOR_COUNT_RANGE = (2, 3)
DISTRACTOR_SPRITES_PER_COLOR_RANGE = (1, 3)
MAX_VISIBLE_SPRITES_PER_KEY = 3
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
        _, n8 = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n8 == 1:
            masks.append(mask)
    return tuple(masks)


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return (int(mask.shape[0]), int(mask.shape[1]), *tuple(int(v) for v in mask.astype(int).flatten()))


@lru_cache(maxsize=1)
def _mask_catalog() -> tuple[tuple[np.ndarray, tuple[int, ...], int], ...]:
    options: list[tuple[np.ndarray, tuple[int, ...], int]] = []
    for area in range(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1):
        for h, w in SPRITE_BBOXES:
            if h * w < area:
                continue
            for mask in _valid_masks(h, w, area):
                sig = _mask_signature(mask)
                _, n4 = label_mask(mask, connectivity=4)
                options.append((mask, sig, int(n4)))
    return tuple(options)


@lru_cache(maxsize=1)
def _masks_by_four_count() -> dict[int, tuple[tuple[np.ndarray, tuple[int, ...]], ...]]:
    grouped: dict[int, list[tuple[np.ndarray, tuple[int, ...]]]] = {}
    for mask, sig, n4 in _mask_catalog():
        grouped.setdefault(int(n4), []).append((mask, sig))
    return {count: tuple(items) for count, items in grouped.items()}


def _make_sprite_from_mask(color: int, bg: int, mask: np.ndarray) -> np.ndarray:
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _choose_mask_for_part(
    four_count: int,
    color: int,
    bg: int,
    exclude: set[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    exclude = exclude or set()
    choices = [(mask, sig) for mask, sig in _masks_by_four_count()[int(four_count)] if sig not in exclude]
    if not choices and exclude:
        choices = list(_masks_by_four_count()[int(four_count)])
    if not choices:
        raise ValueError(f"no mask available for four_count={four_count}")
    mask, sig = random.choice(choices)
    return _make_sprite_from_mask(color, bg, mask), sig


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _preferred_partitions(total: int) -> tuple[tuple[int, ...], ...]:
    if total == 1:
        return ((1,),)
    if total == 2:
        return ((2,), (1, 1))
    if total == 3:
        return ((2, 1), (3,), (1, 1, 1))
    if total == 4:
        return ((3, 1), (2, 2), (4,), (2, 1, 1))
    if total == 5:
        return ((3, 2), (4, 1), (5,), (2, 2, 1), (3, 1, 1))
    raise ValueError(f"unsupported total count: {total}")


def _choose_key_parts(total: int, *, force_diverse: bool, force_multi_fragment: bool) -> tuple[int, ...]:
    candidates = []
    for parts in _preferred_partitions(int(total)):
        if len(parts) > MAX_VISIBLE_SPRITES_PER_KEY:
            continue
        if force_diverse and len(parts) < 2:
            continue
        if force_multi_fragment and not any(part > 1 for part in parts):
            continue
        candidates.append(parts)
    if not candidates:
        raise ValueError(f"no valid partition for total={total}")
    return random.choice(candidates)


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    key_count = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, key_count)
    remaining = [c for c in remaining if c not in key_colors]

    max_distractors = min(DISTRACTOR_COLOR_COUNT_RANGE[1], len(remaining))
    min_distractors = min(DISTRACTOR_COLOR_COUNT_RANGE[0], max_distractors)
    if min_distractors < DISTRACTOR_COLOR_COUNT_RANGE[0]:
        raise ValueError("not enough colors for the required distractor set")
    distractor_count = random.randint(min_distractors, max_distractors)
    distractor_colors = random.sample(remaining, distractor_count)
    return bg, key_colors, distractor_colors


def _sample_counts(k: int) -> list[int]:
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= 3:
            return counts


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    bh, bw = _legend_shape(len(key_colors), max(counts))
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, total) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(total):
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


def _build_key_sprite_plan(key_colors: list[int], counts: list[int], bg: int) -> list[np.ndarray]:
    anchor_candidates = [i for i, total in enumerate(counts) if int(total) >= 3]
    if not anchor_candidates:
        raise ValueError("need at least one anchor key color with count >= 3")
    anchor = random.choice(anchor_candidates)

    sprites: list[np.ndarray] = []
    for i, (color, total) in enumerate(zip(key_colors, counts)):
        force_diverse = i == anchor
        force_multi_fragment = i == anchor
        parts = _choose_key_parts(int(total), force_diverse=force_diverse, force_multi_fragment=force_multi_fragment)

        used_signatures: set[tuple[int, ...]] = set()
        for part in parts:
            sprite, sig = _choose_mask_for_part(int(part), int(color), int(bg), exclude=used_signatures)
            used_signatures.add(sig)
            sprites.append(sprite)
    return sprites


def _build_distractor_sprite_plan(distractor_colors: list[int], bg: int) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    for color in distractor_colors:
        n_sprites = _randint(DISTRACTOR_SPRITES_PER_COLOR_RANGE)
        used_signatures: set[tuple[int, ...]] = set()
        for _ in range(n_sprites):
            four_count = random.choices((1, 2, 3, 4, 5), weights=(4, 4, 3, 2, 1), k=1)[0]
            sprite, sig = _choose_mask_for_part(int(four_count), int(color), int(bg), exclude=used_signatures)
            used_signatures.add(sig)
            sprites.append(sprite)
    return sprites


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_colors: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites = _build_key_sprite_plan(key_colors, counts, bg)
    sprites.extend(_build_distractor_sprite_plan(distractor_colors, bg))
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


def _find_legend_info(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, int, list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    best = None
    for coords in components_mask(g == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(g, bbox)
        if (sub != BLACK).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        raise AssertionError("legend box not found")

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != BLACK).any():
            key_col = int(cc)
            break
    if key_col is None:
        raise AssertionError("key column not found")

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != BLACK]
    keys.sort(key=lambda item: int(item[0]))
    return (r0, c0, r1, c1), box, int(key_col), keys


def _component_signature(mask: np.ndarray, coords: np.ndarray) -> tuple[int, ...]:
    bbox = bbox_coords(coords)
    cropped = crop_bbox(mask.astype(int), bbox)
    return _mask_signature(cropped.astype(bool))


def fused_behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    legend_bbox, _box, _key_col, keys = _find_legend_info(g)
    r0, c0, r1, c1 = legend_bbox

    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    key_colors = [int(color) for _rr, color in keys]

    diverse_key_colors = 0
    four_minus_eight_total = 0
    for color in key_colors:
        mask = (g == int(color)) & outside
        comps8 = components_mask(mask, connectivity=8)
        signatures = {_component_signature(mask, coords) for coords in comps8}
        if len(comps8) >= 2 and len(signatures) >= 2:
            diverse_key_colors += 1
        _lab4, n4 = label_mask(mask, connectivity=4)
        _lab8, n8 = label_mask(mask, connectivity=8)
        four_minus_eight_total += int(n4) - int(n8)

    outside_colors = sorted(set(int(v) for v in g[outside].flatten()))
    distractor_colors = [color for color in outside_colors if color not in key_colors and color not in (bg, BLACK)]
    distractor_sprite_counts = {
        int(color): int(label_mask((g == int(color)) & outside, connectivity=8)[1]) for color in distractor_colors
    }

    return {
        "diverse_key_colors": int(diverse_key_colors),
        "four_minus_eight_total": int(four_minus_eight_total),
        "distractor_color_count": int(len(distractor_colors)),
        "distractor_sprite_counts": distractor_sprite_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a legend box, ignore non-key clutter, and tally outside fragments by 4-connectivity even when same-color sprites vary in shape.

    Input structure:
    - A black legend rectangle contains one colored key pixel on every other row of its leftmost non-black column.
    - Outside the box, each key color appears as one or more small monochrome sprites in 2x2..3x3 boxes, and same-color sprites may differ in geometry.
    - There are also 2-3 extra distractor colors outside the box; they can have several sprites too, but none of those colors appear in the legend.

    Rule:
    - For each legend color, look only outside the box and count same-colored connected pieces using 4-connectivity, so diagonal-only contacts inside one sprite split into multiple counted parts.
    - Return a black canvas matching the legend size, placing one dot of that color at key_col, key_col+2, key_col+4, ... for each counted 4-connected piece.

    Key insight:
    - The stable signal is the legend's color list, not sprite shape or raw blob count: first filter to the key colors, then split each color's outside pixels with 4-neighbor connectivity before writing the tally dots.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    legend_bbox, box, key_col, keys = _find_legend_info(g)
    r0, c0, r1, c1 = legend_bbox
    bh, bw = map(int, box.shape)

    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, color in keys:
        mask = (g == int(color)) & outside
        _labels, n_components = label_mask(mask, connectivity=4)
        for i in range(int(n_components)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    bg, key_colors, distractor_colors = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    try:
        sprites = _build_sprite_queue(key_colors, counts, distractor_colors, bg)
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    metrics = fused_behavior_metrics(grid)
    if metrics["diverse_key_colors"] < 1:
        return None
    if metrics["four_minus_eight_total"] < 1:
        return None
    if not (2 <= int(metrics["distractor_color_count"]) <= 3):
        return None
    if any(not (1 <= int(v) <= 3) for v in dict(metrics["distractor_sprite_counts"]).values()):
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
