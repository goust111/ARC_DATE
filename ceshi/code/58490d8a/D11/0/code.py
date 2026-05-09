# Variant of 58490d8a
# Dimension: D11 — 计数对象定义 (transform)
# Change: 统计各关键颜色的8连通独立连通分量数量... → 改为4连通计数

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis
# spec:
#   concreteization:
#     - 输入仍然是一个黑色图例矩形和若干散落在外部的关键颜色图案。
#     - 变化只发生在“外部图案如何被切分”为独立对象：现在只允许上下左右连通，
#       因此同色像素若只是对角接触，视觉上还是一团，但计数时要拆开。
#     - 例如 3x3 模板
#         110
#         001
#         001
#       在原题里是 1 个 8 连通对象，在本变体里是 2 个 4 连通对象。
#   flow_impact:
#     - solver 的 legend 定位、key 列定位、输出绘制全都可以保留。
#     - 真正需要改的是“按颜色统计外部连通分量”这一步，必须从 8 连通切到 4 连通。
#     - generator 也必须改；如果仍然只生成 4 连通实心图案，那么新旧规则在视觉上没有差异。
#   complexity_calibration:
#     - 为了让规律可见，同时避免形状自由度过大，外部图案限制为一小组固定模板。
#     - 每个模板预先知道自己的 4 连通分量数，取值为 1..5；同一颜色仍重复同一种模板，
#       保持原题“同色精灵外观一致”的稳定感。
#     - 每个样本至少强制一个关键颜色选用分量数 > 1 的模板，这样不会退化回原题。
#   hidden_assumptions_audit:
#     - 原 main() 用“最大黑色 8 连通块且 bbox 内含非黑像素”找图例；这里依然成立，
#       因为背景仍是非黑单色，外部图案也不使用黑色。
#     - 原 main() 假设图例内最左非黑列就是 key 列；这里依然成立，因为图例结构没变。
#     - 原 main() 的 `label_mask(mask, connectivity=8)` 依赖旧属性；在新规则下会把对角接触
#       的子块误并为 1 个对象，必须替换为 `connectivity=4`。
#     - 新 generator 必须继续用 8 邻域间隔放置图案，否则不同图案之间可能在对角上碰到，
#       导致 4 连通 / 8 连通之外又引入跨图案合并的歧义。
#   placement_legality:
#     - 所有图案都在 2x2 或 3x3 bbox 内，面积 4 或 5，接近原题尺度。
#     - `random_free_location(..., padding=1, padding_connectivity=8)` 保证图案彼此以及图例框都不接触。
#     - 因此同色的总 4 连通计数 = 该颜色每个拷贝的已知 4 连通分量数之和，solver 可稳定恢复。
#   observability:
#     - 解题者会看到某些同色“单个图案”内部只靠角相连，但输出点数比可见图案个数更多。
#     - 同一种对角模板在同色中重复出现，提供重复证据，便于从少量样例推断“对角不算连通”。
#   cleanliness:
#     - 图例区仍是独立的黑矩形，输出仍是整齐的等间距点列。
#     - 外部图案尺寸小且不重叠，不会和图例或其他颜色混杂。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a color legend and tally 4-connected fragments outside it.

    Input structure:
    - A solid black legend rectangle contains one key-colored pixel on every other row of its first non-black column.
    - Outside that rectangle, each key color appears as one or more small same-colored motifs, and some motifs only corner-touch internally.

    Rule:
    - For each key color, count outside same-colored components using 4-connectivity, so diagonal-only contact does not merge pieces.
    - Return a black canvas the size of the legend box, placing one evenly spaced dot of that color per counted component on the matching key row.

    Key insight:
    - The ambiguous part is not which colors matter, but how to split their motifs: corner-linked pixels must be separated before writing the tally dots.
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
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=4)
        for i in range(int(nc)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

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

SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8
GENERATION_ATTEMPTS = 300

# Base motif per 4-component count. Each mask is 8-connected as a whole, but the
# same-color 4-connected component count equals the dict key.
BASE_MASKS: dict[int, np.ndarray] = {
    1: np.array([[1, 1], [1, 1]], dtype=bool),
    2: np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool),
    3: np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=bool),
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
    5: np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
}


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


@lru_cache(maxsize=None)
def _mask_variants(four_count: int) -> tuple[np.ndarray, ...]:
    base = BASE_MASKS[int(four_count)]
    seen = set()
    variants = []
    for flip in (False, True):
        arr = np.fliplr(base) if flip else base
        for k in range(4):
            rot = np.rot90(arr, k)
            key = tuple(map(int, rot.flatten()))
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return tuple(variants)


def _make_sprite(color: int, bg: int, four_count: int) -> np.ndarray:
    mask = random.choice(_mask_variants(int(four_count)))
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
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) > 1:
            return counts


def _choose_units(counts: list[int]) -> list[int] | None:
    options = []
    eligible = []
    for i, total in enumerate(counts):
        divisors = [unit for unit in sorted(BASE_MASKS) if total % unit == 0]
        options.append(divisors)
        if any(unit > 1 for unit in divisors):
            eligible.append(i)
    if not eligible:
        return None

    forced = random.choice(eligible)
    units = []
    for i, divisors in enumerate(options):
        split_divisors = [unit for unit in divisors if unit > 1]
        if i == forced:
            units.append(random.choice(split_divisors))
        elif split_divisors and random.random() < 0.6:
            units.append(random.choice(split_divisors))
        else:
            units.append(random.choice(divisors))
    return units


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


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    units: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, total, unit in zip(key_colors, counts, units):
        copies = total // unit
        sprite = _make_sprite(color, bg, unit)
        for _ in range(copies):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        distractor_unit = random.choice(list(BASE_MASKS))
        sprites.append(_make_sprite(distractor_color, bg, distractor_unit))
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
    units = _choose_units(counts)
    if units is None:
        return None

    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, units, distractor_color, bg)
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
