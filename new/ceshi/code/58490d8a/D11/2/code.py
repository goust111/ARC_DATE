# Variant of 58490d8a
# Dimension: D11 — 计数对象定义 (transform)
# Change: 统计各关键颜色的8连通独立连通分量数量... → 统计精灵中最大的那个的面积

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis
# a) 具体化与行为追踪:
#    - 仍然保留一个黑色图例框；框内第一个含非黑像素的列是键列，奇数行放关键颜色像素。
#    - 框外依然散落若干同色精灵，但同一种关键颜色现在至少有两个8连通精灵，而且面积刻意不同。
#    - 输出仍是一个纯黑画布，只在对应键行上画等间隔圆点；不同之处是圆点数不再等于该颜色精灵个数，
#      而是等于该颜色所有框外精灵里“面积最大的那个”的像素数。
#    - 小例子: 某键色在框外有三个8连通精灵，面积分别为4、6、5。
#      原逻辑: 统计到3个连通分量，于是在该行画3个点。
#      变体逻辑: 先找最大面积6，再在该行画6个点。
#      差异落在“按颜色聚合框外精灵之后”的统计步骤，框定位、键列读取、输出排布都不变。
# b) 复杂度校准:
#    - 如果同色精灵仍全部同形同面积，人类很容易误读为“随便选一个面积”或退化回“个数”。
#      因此生成器让每个关键颜色拥有2-4个精灵，且至少出现两个不同面积，保证“取最大值”是必要步骤。
#    - 面积仍限制在4-7，保持原任务的紧凑小精灵风格；输出点列宽度因此改为按最大面积而不是按精灵个数预留。
# c) 隐式假设审查:
#    - 原 `main()` 中 `label_mask(mask)` 只取 `nc`，默认“每个连通分量贡献1个点”。这个前提在变体中失效，
#      必须改为遍历每个连通分量并读取 `coords.shape[0]`。
#    - 原代码无条件从 `key_col + 2*i` 画点，在“点数=连通分量数”时安全；变体里点数可能到7，
#      所以生成器必须同步把图例宽度建立在最大面积上，否则会截断正确答案。
#    - 原生成器为同色颜色复制同一 sprite；这在原题无害，但在变体中会掩盖“最大面积选择”的存在，
#      所以这里必须取消这个旧假设，让同色精灵面积可变。
# d) 放置合法性:
#    - 所有精灵仍通过 `random_free_location(..., padding=1, connectivity=8)` 放置，保证彼此和图例框都不接触。
#    - 每个 sprite 仍来自 2x2~3x3 的8连通掩码，像素面积 4-7，确保不越界且 solver 的按颜色连通域分析稳定。
# e) 视觉可观察性:
#    - 同色出现多个不同大小的精灵时，输出只匹配其中最大的那一个；这在2-4个样例中能肉眼看到，
#      因为同色对象数量通常是2-4，而输出点数通常是5-7，两者明显不同。
#    - 规则不依赖颜色 id 排序，也不使用任何不可见属性；仍然按图例行次序逐行输出。
# f) 可视化整洁度:
#    - 对象继续保持单色、紧凑、小尺寸，并且留有8连通空隙，避免彼此粘连或和图例混淆。
#    - 图例框内除了键列和输出点列没有别的标记，输出区域依然整洁。
# g) 颜色变化可推断性:
#    - 没有新增颜色映射；关键颜色只在本行输出自身颜色的圆点。
#    - 可推断的新关系是“同色多精灵 -> 选面积最大的那个 -> 用该面积决定点数”。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure the largest outside sprite for each legend color and encode it as dots.

    Input structure:
    - A black legend rectangle contains one colored key pixel on every other row of a single key column.
    - Outside that rectangle, each key color appears in several disconnected 8-connected sprites of varying sizes.

    Rule:
    - For each key color, inspect only the same-colored sprites outside the legend and find the sprite with maximum area.
    - Repaint the legend box as black, then draw one dot per pixel of that largest sprite, starting at the key column and skipping one cell between dots.

    Key insight: The legend still indexes colors row by row, but the outside evidence now encodes a per-color maximum component area rather than a component count.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None  # (area, bbox)
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

    # STEP 2) Inside the legend bbox, find the key column:
    # leftmost column with any non-black pixels (the "key pixels").
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

    # STEP 3/4) Find the maximum outside CC area for each key color, then draw 1 dot per pixel of that maximum-area sprite.
    for rr, col in keys:
        largest_area = 0
        for coords in components_mask((g == int(col)) & outside, connectivity=8):
            largest_area = max(largest_area, int(coords.shape[0]))
        for i in range(largest_area):
            cc = int(key_col + 2 * i)
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
COMPONENT_COUNT_RANGE = (2, 4)
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
LARGEST_AREA_RANGE = (5, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

# Retry budget for rejection sampling.
GENERATION_ATTEMPTS = 600


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


def _make_sprite(color: int, bg: int, area: int) -> np.ndarray:
    """Sample one connected same-color sprite with a prescribed area."""
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


def _sample_area_lists(k: int) -> list[list[int]]:
    """Sample per-color sprite areas so each key color has a meaningful largest member."""
    all_areas: list[list[int]] = []
    for _ in range(k):
        n = _randint(COMPONENT_COUNT_RANGE)
        largest = _randint(LARGEST_AREA_RANGE)
        areas = [largest, _randint((SPRITE_AREA_RANGE[0], largest - 1))]
        while len(areas) < n:
            areas.append(_randint((SPRITE_AREA_RANGE[0], largest)))
        random.shuffle(areas)
        all_areas.append(areas)
    return all_areas


def _legend_shape(k: int, max_area: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest reported area."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_area + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], largest_areas: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and largest sprite areas."""
    bh, bw = _legend_shape(len(key_colors), max(largest_areas))
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, area) in enumerate(zip(key_colors, largest_areas)):
        rr = _key_row(i)
        for j in range(area):
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


def _build_sprite_queue(
    key_colors: list[int],
    area_lists: list[list[int]],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    """Create per-color area-varied sprites plus an optional distractor, then shuffle them."""
    sprites = []
    for color, areas in zip(key_colors, area_lists):
        for area in areas:
            sprites.append(_make_sprite(color, bg, area))
    if distractor_color is not None:
        sprites.append(_make_sprite(distractor_color, bg, _randint(SPRITE_AREA_RANGE)))
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
    area_lists = _sample_area_lists(len(key_colors))
    largest_areas = [max(areas) for areas in area_lists]
    expected = _build_expected(key_colors, largest_areas)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, area_lists, distractor_color, bg)
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
