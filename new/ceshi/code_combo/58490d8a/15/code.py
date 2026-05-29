# Combo variant of 58490d8a (index 15)
# combo_spec: D7/3+D11/0
# Name: 图例内干扰行+4连通计数

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis
# ===============
# spec:
#   fused_rule:
#     - 黑色 legend box 里不再只有真实 key 行，还会混入 1-2 条候选干扰行。
#     - 所有候选行都和原题一样放在同一 key 列；从外观上看它们完全像合法图例项。
#     - 真实 key 颜色在框外对应 2-5 个 4 连通分量；干扰行颜色在框外只对应 1 个 4 连通分量。
#     - 计数规则改为 4 连通，因此一个单独可见的同色 motif 也可能贡献 2-5 个 tally 点。
#   concrete_example:
#     legend_rows:
#       row1: red
#       row3: yellow
#       row5: blue
#     outside:
#       red: 一个对角链模板，8 连通看作 1 个整体，但 4 连通拆成 3 个分量
#       yellow: 一个 2x2 实心块，4 连通数量 = 1
#       blue: 两个相同对角模板，每个 4 连通数量 = 2，总数 = 4
#     original_solver_trace:
#       - 默认把三条 legend 行都当真 key。
#       - 用 8 连通数外部对象，得到 red=1, yellow=1, blue=2。
#       - 因而会错误地给 yellow 画 1 点，也会低估 red 和 blue。
#     fused_solver_trace:
#       - 先把 legend 中所有有色行当 candidate rows，而不是直接当 key。
#       - 对每个候选颜色只统计 box 外像素，并且用 4 连通计数。
#       - 只有 count >= 2 的行保留为真实 key；yellow 因只有 1 个 4 连通分量被清空。
#       - red 行画 3 点，blue 行画 4 点。
#     visible_difference:
#       - 输入里 yellow 看起来像正常图例行，但输出整行是黑的。
#       - red / blue 的输出点数又明显大于其“可见 sprite 个数”，迫使解题者推断对角不连通。
#   complexity_calibration:
#     - 干扰行固定至少 1 条，避免 D7 分支退化。
#     - 真实 key 的总计数固定在 2-5，和干扰行的 1 构成稳定二分。
#     - 外部 motif 限制为少量固定模板，4 连通分量数只有 1..5，避免自由度过大。
#     - 每个样本至少强制 1 个真实 key 使用 four_count > 1 的模板，保证 D11 分支可见。
#   implicit_assumptions_review:
#     - `largest black CC whose bbox contains non-black` 仍可定位 legend，因为背景非黑、外部 sprite 不用黑色。
#     - `key_col = leftmost non-black column in box` 仍成立，因为所有候选行仍共用同一列。
#     - `keys = colored rows in key_col` 不再等于“真实 key 列表”，只能看作候选行列表。
#     - `label_mask(..., connectivity=8)` 的旧前提已失效；融合后必须改为 4 连通，否则会低估拆分模板。
#     - `count >= 2 means real key` 是新的 solver 假设；generator 通过让干扰行恰好只有 1 个 4 连通分量、
#       真实 key 至少 2 个 4 连通分量来保证该阈值有效。
#     - 候选行颜色必须互异，否则相同颜色的真假行会共享同一外部计数。generator 会显式保证颜色唯一。
#   placement_legality:
#     - legend 仍整体粘贴为黑底矩形，且与所有外部 sprite 保持 8 邻域间隔。
#     - 外部 sprite 继续使用 `random_free_location(..., padding=1, padding_connectivity=8)`，
#       防止不同 sprite 彼此接触或与 legend 相碰，避免额外连通性歧义。
#     - 干扰行只放 1 个 four_count=1 的 sprite；真实 key 的总 tally 数由 `copies * unit` 精确控制。
#   visual_observability:
#     - 每个样本同时出现“图例里有行被忽略”和“单个外部 motif 贡献多个点”两种证据。
#     - 规则完全依赖可见的行位置、颜色匹配和局部连接方式，不依赖颜色编号大小等不可视因素。
#   cleanliness:
#     - legend 内仍是规则的隔行单像素清单，没有额外覆盖层。
#     - 外部对象尺寸小、间距固定，输入不会变成拥挤噪声图。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a noisy legend and tally only repeated 4-connected evidence.

    Input structure:
    - A black rectangular legend box contains several colored candidate rows in one shared key column, and some of those rows are decoys.
    - Outside the box, each candidate color appears as small same-colored motifs, with some motifs joined only by diagonal contact.

    Rule:
    - For every candidate legend color, count outside same-colored components using 4-connectivity, so corner-touching pixels split apart.
    - Repaint the legend box on black: only rows whose color has at least two outside 4-connected components are valid, and each valid row gets evenly spaced tally dots of that color.

    Key insight: Solving now needs two filters in sequence: first reject legend rows supported by only a singleton outside match, then count the surviving color with 4-connectivity rather than 8-connectivity.
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

    candidates = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    candidates.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in candidates:
        mask = (g == int(col)) & outside
        _lab, count = label_mask(mask, connectivity=4)
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

# Each base mask is visually one 8-connected motif, but its 4-connected component
# count equals the dict key. Reusing a small fixed library keeps the split rule legible.
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


def _legend_shape(total_rows: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * total_rows + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


@lru_cache(maxsize=None)
def _mask_variants(four_count: int) -> tuple[np.ndarray, ...]:
    base = BASE_MASKS[int(four_count)]
    seen = set()
    variants = []
    for flip in (False, True):
        arr = np.fliplr(base) if flip else base
        for turns in range(4):
            rot = np.rot90(arr, turns)
            key = (rot.shape, tuple(map(int, rot.flatten())))
            if key in seen:
                continue
            seen.add(key)
            variants.append(rot.copy())
    return tuple(variants)


def _make_sprite(color: int, bg: int, four_count: int) -> np.ndarray:
    mask = random.choice(_mask_variants(int(four_count)))
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)
    key_count = _randint(KEY_COUNT_RANGE)
    distractor_count = _randint(DISTRACTOR_ROW_COUNT_RANGE)
    usable = [c for c in colors if c != bg]
    chosen = random.sample(usable, key_count + distractor_count)
    return bg, chosen[:key_count], chosen[key_count:]


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _choose_units(counts: list[int]) -> list[int]:
    divisor_options = []
    for total in counts:
        divisor_options.append([unit for unit in sorted(BASE_MASKS) if int(total) % int(unit) == 0])

    forced = random.randrange(len(counts))
    forced_repeat = random.choice([i for i in range(len(counts)) if i != forced])
    units = []
    for i, divisors in enumerate(divisor_options):
        split_divisors = [unit for unit in divisors if int(unit) > 1]
        if i == forced:
            units.append(random.choice(split_divisors))
        elif i == forced_repeat:
            # Keep one real key as repeated whole motifs so the sample shows both
            # the D7-style "repeated evidence" cue and the D11 split-count cue.
            units.append(1)
        elif split_divisors and random.random() < 0.6:
            units.append(random.choice(split_divisors))
        else:
            units.append(random.choice(divisors))
    return units


def _row_specs(
    key_colors: list[int],
    counts: list[int],
    units: list[int],
    distractor_colors: list[int],
) -> list[dict[str, int | bool]]:
    slots = list(range(len(key_colors) + len(distractor_colors)))
    random.shuffle(slots)

    specs: list[dict[str, int | bool]] = []
    cursor = 0
    for color, count, unit in zip(key_colors, counts, units):
        specs.append(
            {
                "row": _key_row(slots[cursor]),
                "color": int(color),
                "count": int(count),
                "unit": int(unit),
                "is_real": True,
            }
        )
        cursor += 1

    for color in distractor_colors:
        specs.append(
            {
                "row": _key_row(slots[cursor]),
                "color": int(color),
                "count": 1,
                "unit": 1,
                "is_real": False,
            }
        )
        cursor += 1

    specs.sort(key=lambda item: int(item["row"]))
    return specs


def _build_expected(shape: tuple[int, int], row_specs: list[dict[str, int | bool]]) -> np.ndarray:
    expected = np.full(shape, BLACK, dtype=int)
    for spec in row_specs:
        rr = int(spec["row"])
        color = int(spec["color"])
        count = int(spec["count"])
        if count < 2:
            continue
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = color
    return expected


def _build_legend(shape: tuple[int, int], row_specs: list[dict[str, int | bool]]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for spec in row_specs:
        legend[int(spec["row"]), KEY_COLUMN] = int(spec["color"])
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


def _build_sprite_queue(row_specs: list[dict[str, int | bool]], bg: int) -> list[np.ndarray]:
    sprites = []
    for spec in row_specs:
        color = int(spec["color"])
        count = int(spec["count"])
        unit = int(spec["unit"])
        copies = count // unit
        sprite = _make_sprite(color, bg, unit)
        for _ in range(copies):
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
    units = _choose_units(counts)
    row_specs = _row_specs(key_colors, counts, units, distractor_colors)
    shape = _legend_shape(len(row_specs), max(counts))
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
