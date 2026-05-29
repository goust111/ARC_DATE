# Variant of 58490d8a
# Dimension: D4 — 变换操作：计数到圆点序列 (transform)
# Change: 每种关键色的框外连通分量数量n转换为n个同色圆点，从关键列开始每隔一列放置一个... → 从右向左放置圆点（反方向）

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# Design analysis for the D4 inversion variant.
# variant_spec:
#   source_task: none
#   unchanged_structure:
#     - 输入仍然是一块黑色图例框，关键色像素仍放在框内固定的左侧关键列。
#     - 框外仍散布 2-4 种关键色连通分量，以及可选的 0-1 个干扰色精灵。
#     - 输出仍然只返回图例框大小的黑底画布，不复制整张输入网格。
#   core_change:
#     - 原题把某行的 n 个计数点放在 key_col, key_col+2, key_col+4, ...
#     - 变体改为从图例框右边起向左放：right_anchor, right_anchor-2, right_anchor-4, ...
#     - 这里取 right_anchor = bw-2，和左侧 key_col=1 形成镜像式的一格边距。
#   behavior_trace:
#     - 例：bw=8，某关键色在框外出现 3 个连通分量。
#     - 原题输出列为 1,3,5。
#     - 本变体输出列为 6,4,2。
#     - 静态图上可见差异是“点列贴右侧”，而不是“从关键像素往右生长”。
#   complexity_calibration:
#     - 只反转方向，不增加新对象、新颜色关系或排序规则，避免把 D4 以外的维度混进来。
#     - 但若仍允许最小宽度 bw=2*max_count+1，则当 count=max_count 时，左右两种规则会占用同一组列，
#       只剩“放置顺序”不同，而 ARC 输出是静态图，肉眼无法区分。
#     - 因此生成器强制 legend 右侧至少保留 1 列 slack，使右锚点确实右移，变体在样例中持续可见。
#   hidden_assumptions_review:
#     - main() 中寻找 legend 的逻辑依赖“最大黑色连通块的 bbox 内含非黑像素”，该前提仍成立。
#     - main() 中寻找 key_col 的逻辑依赖“关键像素都在最左非黑列”，该前提仍成立。
#     - 原题唯一直接依赖旧方向的语句是 cc = key_col + 2*i；此处必须改为从右锚点回退。
#     - 输出保持全黑底，不把输入的 key 像素原样保留；这是原题已有性质，在变体中仍成立。
#   placement_legality:
#     - legend 尺寸仍满足 max_count 个点可放下：当 bw=2*max_count+1+slack 且 slack>=1 时，
#       最左一个反向点列位置为 bw-2-2*(max_count-1)=slack+1 >= 2，不越界且不会碰到左边界。
#     - 精灵放置仍用 random_free_location + padding=1，保证彼此及图例框都不发生 8 邻接触碰。
#   visual_observability:
#     - 关键变化完全由几何位置表达，不依赖颜色 id、计数大小排序等不可见因素。
#     - 右侧固定留空后，即使所有计数都等于最大值，输出也会整体右对齐，不会退化回原题外观。
#   cleanliness:
#     - 同一区域仍只有黑底、关键像素和离散点列三类元素；对象层次与原题相同，阅读负担不增加。


def _right_anchor(width: int) -> int:
    """Return the rightmost tally column used by the reversed dot sequence."""
    return int(width) - 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and rewrite the legend as right-anchored tallies.

    Input structure:
    - A black legend rectangle contains one colored key pixel per active row in its left-side key column.
    - Outside that box, separate monochrome sprites of those key colors are scattered across the background.

    Rule:
    - For each key color, count how many connected components of that color lie outside the legend.
    - Return a black canvas the size of the legend box, placing one same-color dot per component every other column, but starting at the right side and moving left.

    Key insight:
    - The legend still tells which row belongs to which color, but the tally direction is inverted: counts become right-aligned dot runs instead of runs that grow from the key column.
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
    right_anchor = _right_anchor(bw)

    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            cc = int(right_anchor - 2 * i)
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
LEGEND_SLACK_RANGE = (1, 3)

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
    """Sample the target outside-component count for each key color."""
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the reversed right-to-left tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    right_anchor = _right_anchor(bw)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, right_anchor - TALLY_COL_STEP * j] = int(color)
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


def _gen_attempt() -> np.ndarray | None:
    """Run one output-first generation attempt and reject solver mismatches."""
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
    raise RuntimeError("failed to generate a valid 58490d8a D4 input")
