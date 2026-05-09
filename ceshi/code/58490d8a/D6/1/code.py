# Variant of 58490d8a
# Dimension: D6 — 图例关键列位置 (symbol_instructions)
# Change: 关键色像素固定放在图例框内的第2列（index=1），是框内最左侧的非黑列... → 关键色放在中间列
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - 输入图例框仍然是黑色矩形，外部仍然散布若干同色精灵。
#     - 唯一被系统性改动的图例说明方式，是“每一行的关键色像素”不再贴左，而是放在图例框的水平中心列。
#     - 输出 tally 也不再从该像素单向向右推进，而是以中心列为锚点，按间隔 2 的位置向左右扩展。
#   concrete_behavior:
#     - 对某个 key 行，如果外部该颜色有 5 个连通块，则输出列落点是 center, center-2, center+2, center-4, center+4。
#     - 若有 3 个连通块，则是 center, center-2, center+2。
#     - 若有 1 个连通块，则只有中心点。
#   behavior_trace:
#     - 原题同样在黑框中找 key 色并计数，但原逻辑把结果写在 key_col, key_col+2, key_col+4, ...
#     - 本变体仍然数“框外同色连通块数”，差异出现在写回阶段：第 2、3、4... 个点不再只往右，而是按半径递增地分配到左右两侧。
#     - 因而同一输入上，旧 solver 会生成右偏的序列，新 solver 会生成围绕中心列的双向序列。
#   complexity_calibration:
#     - 若允许偶数个计数，严格“双侧对称”会出现缺半边的歧义；为避免样例难以归纳，生成器只采样 1/3/5 三档计数。
#     - 每个样例至少包含一行 count > 1，并要求同一图例中至少出现两种不同计数，保证变体行为肉眼可见。
#     - 这样没有引入新的对象类型、排序规则或颜色规则，仍保持原题“按颜色分行计数”的主任务。
#   implicit_assumption_audit:
#     - 原 main() 第一步“找最大黑色连通块且其 bbox 含非黑像素”仍成立；关键像素改到中列不会破坏黑框整体连通。
#     - 原 main() 第二步“取最左非黑列作为 key 列”依赖旧题的左贴边假设；变体里改为寻找图例中的非黑列，并按检测到的那一列作为中心锚点。
#     - 原 main() 第三步“cc = key_col + 2 * i”依赖 tally 单向扩展；变体必须改成中心向外的双向列序列。
#     - 生成器新增的关键前提是：图例宽度必须是奇数，且足以容纳最大 count 对应的左右半径；代码用最小宽度公式和对称 slack 保证这一点。
#   placement_legality:
#     - 图例宽度按最大计数确定，最远点半径为 2 * ceil((count-1)/2)，因此最小宽度为 2*radius+1。
#     - 额外 slack 总是左右对称添加，所以关键色像素始终处于精确中心列。
#     - 外部精灵放置仍沿用原题的 random_free_location + padding=1，确保彼此以及与图例框都不接触。
#   visual_observability:
#     - 输入里唯一非黑列落在图例正中，输出中的同色圆点围绕该中线展开；与原题“左起向右记数”相比，视觉差异直接而稳定。
#     - 由于每个样例都含至少一行 3 或 5 个点，人类解题者能在 2-3 组样例中观察到中心轴和双向扩展。
#   cleanliness:
#     - 图例区仍只承担“索引 + tally”两种作用，外部区仍只放精灵，没有新增叠加元素。
#     - 输出仍是纯黑底 + 稀疏同色点，结构比原题更规整，不会引入额外视觉噪声。

from __future__ import annotations

import random
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


def _tally_positions(center_col: int, count: int) -> list[int]:
    """Return center-out tally columns: 0, -2, +2, -4, +4, ... around center_col."""
    cols = [int(center_col)]
    radius = 1
    while len(cols) < int(count):
        cols.append(int(center_col - 2 * radius))
        if len(cols) < int(count):
            cols.append(int(center_col + 2 * radius))
        radius += 1
    return cols


def _legend_min_width(max_count: int) -> int:
    """Smallest odd width that fits the center-out tally for max_count dots."""
    farthest_radius = 2 * ((int(max_count) - 1 + 1) // 2)
    return 2 * farthest_radius + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and encode the totals around a centered legend key.

    Input structure:
    - A black rectangular legend box contains one key-color pixel per key row, and these key pixels all sit in the box's horizontal middle column.
    - Outside the box, separate monochrome objects of the key colors are scattered around the background.

    Rule:
    - For each key color, count same-colored connected components outside the legend box.
    - Repaint the legend box on black: each key row gets one dot per counted object, starting at the middle key column and then stepping outward every other cell to the left and right.

    Key insight: The centered key pixel acts as the anchor of a bilateral tally, so counting objects also determines a symmetric center-out placement pattern.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None  # (area, bbox)
    for coords in components_mask(g == 0, connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != 0).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Find the legend key column: in this variant it is the unique non-black column
    # inside the box, located at the horizontal center in generated examples.
    nonblack_cols = [int(cc) for cc in range(bw) if (box[:, int(cc)] != 0).any()]
    if not nonblack_cols:
        return np.zeros_like(box)
    key_col = int(nonblack_cols[len(nonblack_cols) // 2])

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != 0]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3/4) Count CCs of each key color outside the legend, then draw a centered tally.
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for cc in _tally_positions(key_col, int(nc)):
            if 0 <= int(cc) < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_CHOICES = (1, 3, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_SIDE_SLACK_RANGE = (0, 2)

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
    """Sample visually informative odd counts so the bilateral expansion is exact and visible."""
    while True:
        counts = [random.choice(TARGET_COUNT_CHOICES) for _ in range(k)]
        if any(count > 1 for count in counts) and len(set(counts)) >= min(2, k):
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and odd width for a centered tally with symmetric slack."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = _legend_min_width(max_count) + 2 * _randint(LEGEND_SIDE_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _key_col(width: int) -> int:
    """Return the horizontal center column of an odd-width legend."""
    return int(width) // 2


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and centered counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    center_col = _key_col(bw)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for cc in _tally_positions(center_col, int(count)):
            expected[rr, int(cc)] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with centered key pixels only."""
    legend = np.full(shape, BLACK, dtype=int)
    center_col = _key_col(shape[1])
    for i, color in enumerate(key_colors):
        legend[_key_row(i), center_col] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a input")
