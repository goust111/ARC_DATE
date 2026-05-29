# Variant of 58490d8a
# Dimension: D7 — 干扰项策略 (meta)
# Change: 0-1个额外颜色的干扰精灵，颜色不在图例关键色中，不影响计数... → 图例框内也有干扰行
#
# Design analysis
# ===============
# spec:
#   variant_core:
#     original_legend: 每个有色行都默认是有效 key 行，solver 直接对该颜色做框外连通块计数。
#     variant_legend: 图例框内除了真实 key 行，还混入 1-2 条“假行”；它们也占据同一 key 列，
#       但对应的颜色在框外只出现 1 个干扰精灵，不应被输出为 tally。
#     necessary_adaptation:
#       - 若假行颜色在框外完全不存在，原 solver 也会自然得到 0 点，变体会退化成原题。
#       - 因此将真实 key 的计数范围改为 2-5，而假行固定只有 1 个框外精灵。
#       - 这样“是否为真实 key 行”会真实影响输出：旧逻辑会给假行画出 1 个点，新逻辑必须过滤。
#   concrete_layout:
#     - 仍然只有一个黑色 legend box，输出仍然是这个 box 的同尺寸黑底 tally 图。
#     - 所有候选行都用单个彩色像素放在 box 的第 2 列，行间保持 1 行黑色间隔。
#     - 候选行总数 = 真实 key 行数(2-4) + 干扰行数(1-2)，行顺序会打乱。
#     - 真实 key 颜色在框外各有 2-5 个同色连通精灵；干扰颜色在框外各有恰好 1 个精灵。
#   behavior_trace:
#     example:
#       legend_rows:
#         row1: red
#         row3: yellow
#         row5: blue
#       outside_components:
#         red: 3
#         yellow: 1
#         blue: 4
#       original_logic:
#         - 把三个图例行都当 key，输出 red 行 3 点、yellow 行 1 点、blue 行 4 点。
#       variant_logic:
#         - 先统计每个图例颜色的框外连通块数。
#         - 只有 count >= 2 的行视为真实 key；yellow 行因只有 1 个外部精灵而整行保持黑色。
#       visible_difference:
#         - 输入里 yellow 仍然像“图例项”，但输出不再给它 tally，必须先做真假行区分。
#   complexity_calibration:
#     - 干扰行固定为 1-2 条，避免规则太弱。
#     - 真实 key 固定至少有 2 个外部精灵，和干扰行的 1 个外部精灵形成稳定二分。
#     - 所有 legend 颜色互异，避免“重复颜色行”把问题变成去重题。
#   implicit_assumptions_review:
#     - 原 `key_col` 查找仍然成立：所有候选行仍在同一最左非黑列。
#     - 原 `keys = ... if box[row, key_col] != 0` 这一步不再等价于“真实 key 列表”，
#       因为其中混入干扰行；现在只能视为 candidate rows。
#     - 原 `for rr, col in keys:` 中无条件绘制 `nc` 个点会在 `nc == 1` 的干扰行上产生错误输出。
#       新 solver 需要显式过滤 `count < 2`。
#   placement_legality:
#     - legend box 仍与所有外部精灵保持 8 邻域间隔，避免 black bbox 与彩色对象粘连。
#     - 精灵仍使用同色单一连通块、2x2 到 3x3 bbox、面积 4-7，并通过 `random_free_location`
#       做不重叠放置。
#     - legend 宽度按真实 key 的最大 count 构造；干扰行不参与宽度需求，因为它们不会输出。
#   visual_observability:
#     - 每个样例都至少有 1 条假行且框外有对应单个精灵，因此人眼能看到“有些 legend 项被忽略”。
#     - 真行与假行的可见差别只依赖外部对象数量，不依赖颜色 id 排序等不可见因素。
#   cleanliness:
#     - legend 内部仍然是规则的奇数行阵列，不叠加额外标记。
#     - 框外对象继续保持间距，输入不会因为 D7 改动变得拥挤难读。

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count repeated colored objects while ignoring singleton legend distractors.

    Input structure:
    - A black rectangular legend box contains one shared key column with several colored candidate rows.
    - Outside the box, every candidate color may appear as scattered same-colored sprites, but only some colors repeat.

    Rule:
    - Measure the number of outside connected components for each candidate legend color.
    - Repaint the legend box as a black tally board: only rows whose color has at least two outside objects are valid, and each valid row gets equally spaced dots of that color.

    Key insight: The box no longer lists only true keys; a legend row is real only if its color is supported by repeated evidence outside the box, while singleton matches are decoys.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

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

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    candidates = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    candidates.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in candidates:
        mask = (g == int(col)) & outside
        _lab, count = label_mask(mask, connectivity=8)
        if int(count) < 2:
            continue
        for i in range(int(count)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0

KEY_COUNT_RANGE = (2, 4)
DISTRACTOR_ROW_COUNT_RANGE = (1, 2)
TARGET_COUNT_RANGE = (2, 5)
DISTRACTOR_OUTSIDE_COUNT = 1
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


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    d = _randint(DISTRACTOR_ROW_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    chosen = random.sample(remaining, k + d)
    key_colors = chosen[:k]
    distractor_colors = chosen[k:]
    return bg, key_colors, distractor_colors


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(total_rows: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * total_rows + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _row_specs(
    key_colors: list[int],
    counts: list[int],
    distractor_colors: list[int],
) -> list[tuple[int, int, int]]:
    slots = list(range(len(key_colors) + len(distractor_colors)))
    random.shuffle(slots)

    specs = []
    cursor = 0
    for color, count in zip(key_colors, counts):
        specs.append((_key_row(slots[cursor]), int(color), int(count)))
        cursor += 1
    for color in distractor_colors:
        specs.append((_key_row(slots[cursor]), int(color), int(DISTRACTOR_OUTSIDE_COUNT)))
        cursor += 1
    specs.sort(key=lambda item: int(item[0]))
    return specs


def _build_expected(shape: tuple[int, int], row_specs: list[tuple[int, int, int]]) -> np.ndarray:
    expected = np.full(shape, BLACK, dtype=int)
    for rr, color, count in row_specs:
        if int(count) < 2:
            continue
        for j in range(int(count)):
            expected[int(rr), KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], row_specs: list[tuple[int, int, int]]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for rr, color, _count in row_specs:
        legend[int(rr), KEY_COLUMN] = int(color)
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


def _build_sprite_queue(row_specs: list[tuple[int, int, int]], bg: int) -> list[np.ndarray]:
    sprites = []
    for _rr, color, count in row_specs:
        sprite = _make_sprite(int(color), bg)
        for _ in range(int(count)):
            sprites.append(sprite.copy())
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
    bg, key_colors, distractor_colors = _sample_palette()
    counts = _sample_counts(len(key_colors))
    max_count = max(counts)
    row_specs = _row_specs(key_colors, counts, distractor_colors)
    shape = _legend_shape(len(row_specs), max_count)
    expected = _build_expected(shape, row_specs)
    legend = _build_legend(shape, row_specs)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(row_specs, bg)
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
    raise RuntimeError("failed to generate a valid 58490d8a D7 input")
