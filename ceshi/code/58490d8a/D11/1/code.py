# Variant of 58490d8a
# Dimension: D11 — 计数对象定义 (transform)
# Change: 统计各关键颜色的8连通独立连通分量数量... → 统计精灵的不同形状种类数

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# Design analysis for the D11 variant
# a) Concretization and trace:
#    The legend box stays unchanged: a black rectangle with one key pixel per row.
#    The outside area still contains disconnected monochrome sprites of each key color,
#    but same-color sprites are no longer forced to be identical.
#    Example trace for one key row:
#    - Original behavior: blue outside sprites = [shape A, shape A, shape B] -> 3 CCs -> 3 dots.
#    - Variant behavior: crop each blue CC to its tight bbox, compare the cropped masks,
#      deduplicate [A, A, B] -> {A, B} -> 2 dots.
#    The changed step is exactly the old "count CCs" stage; legend finding and dot drawing stay intact.
# b) Complexity calibration:
#    Shapes still come from 2x2..3x3 connected masks of area 4..7, so they remain readable.
#    Distinct-shape counts are limited to 1..4 and total object counts to 1..5 per color,
#    preserving the original task scale while making repeats and multi-shape colors visible.
#    Every generated puzzle contains at least one key color with >=2 distinct shapes and one repeated copy,
#    so the variant does not collapse back to object counting.
# c) Implicit-assumption audit of the old main():
#    - Legend detection via the largest black CC with colored pixels inside still holds.
#    - Key-column detection still holds because the legend format is unchanged.
#    - The old line `_labc, nc = label_mask(mask, connectivity=8)` assumed "one component = one tally dot".
#      That assumption is false now; we must keep the component split only as an intermediate step and then
#      compare component shapes after translation-normalizing them by tight cropping.
#    - Writing dots left-to-right still works because the output encodes only a count, not the identities.
# d) Placement legality:
#    Random placement still uses 8-neighborhood padding around each sprite and the legend box, so
#    components never merge accidentally, never overlap, and never touch the legend.
#    Distinct shapes are instantiated as exact copies of sampled masks, so solver-side deduplication is stable.
# e) Visual observability:
#    The changed rule is visible because some same-color sprites repeat while others differ.
#    A human can inspect one color, see e.g. three objects but only two shape patterns, and match that to two dots.
#    No hidden ordering by color id or other non-visual metadata is used.
# f) Visual cleanliness:
#    Objects remain small, monochrome, and separated by at least one cell under 8-connectivity.
#    The legend remains isolated, so there is no ambiguity between source sprites and destination tally dots.
# g) Color inference:
#    Colors still only act as keys: each row counts shapes of the matching outside color.
#    No extra color remapping is introduced, so the new reasoning burden is only shape deduplication.
# h) New solver assumptions introduced here:
#    Exact cropped-mask equality defines "same shape"; rotations/reflections count as different types.
#    The generator enforces this by sampling exact masks from a fixed pool and duplicating them verbatim when needed.


BLACK = 0


def _component_shape_signature(mask: np.ndarray, coords: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    """Return a translation-normalized exact shape signature for one boolean component."""
    sub = crop_bbox(mask, bbox_coords(coords))
    return int(sub.shape[0]), int(sub.shape[1]), tuple(int(x) for x in sub.flat)


def _component_and_shape_count(mask: np.ndarray) -> tuple[int, int]:
    """Return (number of components, number of distinct cropped shapes) for a boolean mask."""
    comps = components_mask(mask, connectivity=8)
    sigs = {_component_shape_signature(mask, coords) for coords in comps}
    return len(comps), len(sigs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count distinct sprite shapes per key color and encode the totals inside the legend box.

    Input structure:
    - A black rectangular legend patch contains one colored key pixel on each active row.
    - Outside that patch, each key color may appear in several disconnected sprites, with repeats and shape variants mixed together.

    Rule:
    - For each key color, collect its outside 8-connected sprites, crop each sprite to its tight bbox, and deduplicate exact cropped shapes.
    - Repaint the legend area as black, then place one dot of the key color every other column for each distinct shape type.

    Key insight:
    - The task is not about how many objects exist, but how many different silhouettes exist after removing translation by tight cropping.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None  # (area, bbox)
    for coords in components_mask(g == int(BLACK), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(BLACK)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Find the legend key column: the leftmost column that contains non-black key pixels.
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(BLACK)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(BLACK)]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3/4) Split by CC, deduplicate same-color cropped shapes, then draw 1 dot per unique shape.
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _components, distinct_shapes = _component_and_shape_count(mask)
        for i in range(int(distinct_shapes)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x, D11 variant) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_KIND_RANGE = (1, 4)
TARGET_OBJECT_CAP = 5
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

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


@lru_cache(maxsize=1)
def _shape_pool() -> tuple[np.ndarray, ...]:
    """Pool of exact sprite masks used to build same-color duplicates and variants."""
    pool = []
    for area in range(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1):
        for h, w in SPRITE_BBOXES:
            if h * w >= area:
                pool.extend(mask.copy() for mask in _valid_masks(h, w, area))
    return tuple(pool)


def _make_sprite_from_mask(color: int, bg: int, mask: np.ndarray) -> np.ndarray:
    """Paint one monochrome sprite from a boolean mask."""
    sprite = np.full(mask.shape, int(bg), dtype=int)
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


def _sample_shape_plan(k: int) -> tuple[list[int], list[int], int]:
    """
    Sample distinct-shape counts and total object counts per key color.

    One emphasis color is forced to have both multiple shape types and a repeated copy,
    guaranteeing that the variant differs from raw component counting.
    """
    focus = random.randrange(k)
    kind_counts = []
    object_counts = []
    for i in range(k):
        if i == focus:
            kind = random.randint(2, min(TARGET_KIND_RANGE[1], TARGET_OBJECT_CAP - 1))
            total = random.randint(kind + 1, TARGET_OBJECT_CAP)
        else:
            kind = _randint(TARGET_KIND_RANGE)
            total = random.randint(kind, TARGET_OBJECT_CAP)
        kind_counts.append(kind)
        object_counts.append(total)
    return kind_counts, object_counts, focus


def _legend_shape(k: int, max_count: int, min_slack: int = 0) -> tuple[int, int]:
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    slack = max(int(min_slack), _randint(LEGEND_SLACK_RANGE))
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + slack
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], kind_counts: list[int], min_slack: int = 0) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and distinct-shape counts."""
    max_count = max(kind_counts)
    bh, bw = _legend_shape(len(key_colors), max_count, min_slack=min_slack)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, kind_counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
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


def _shape_multiplicities(kind_count: int, object_count: int) -> list[int]:
    """Split a total number of objects across shape kinds, keeping at least one per kind."""
    counts = [1] * kind_count
    for _ in range(object_count - kind_count):
        counts[random.randrange(kind_count)] += 1
    return counts


def _build_sprite_queue(
    key_colors: list[int],
    kind_counts: list[int],
    object_counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    """Create same-color repeated sprites plus exact-shape variants, then shuffle them."""
    sprites = []
    shape_pool = _shape_pool()
    for color, kind_count, object_count in zip(key_colors, kind_counts, object_counts):
        chosen_masks = random.sample(shape_pool, kind_count)
        multiplicities = _shape_multiplicities(kind_count, object_count)
        for mask, copies in zip(chosen_masks, multiplicities):
            sprite = _make_sprite_from_mask(color, bg, mask)
            for _ in range(copies):
                sprites.append(sprite.copy())
    if distractor_color is not None:
        mask = random.choice(shape_pool)
        sprites.append(_make_sprite_from_mask(distractor_color, bg, mask))
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
    kind_counts, object_counts, focus = _sample_shape_plan(len(key_colors))
    min_slack = 1 if kind_counts[focus] == max(kind_counts) else 0
    expected = _build_expected(key_colors, kind_counts, min_slack=min_slack)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, kind_counts, object_counts, distractor_color, bg)
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
    raise RuntimeError("failed to generate a valid 58490d8a D11 input")
