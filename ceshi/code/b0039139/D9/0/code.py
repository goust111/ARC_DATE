# Variant of b0039139
# Dimension: D9 — 色板区段的表示方式 (symbol_instructions)
# Change: 区段3和区段4各为纯色矩形块，直接用其填充颜色作为前景色和背景色... → 色板为渐变条纹：取最频繁颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis
# a) 具体化与行为追踪
#    - 四段布局、蓝色分隔线、模板裁剪、计数板计数和平铺方向全部保持原题。
#    - 唯一可见变化发生在区段 3/4：它们不再是单色矩形，而是沿 section 的短轴排布的多色条纹。
#      例如横向布局时，前景色板可能是若干整列条纹 [6, 8, 6, 5, 6]，此时目标前景色是 6，
#      因为 6 的列宽总和最大。背景色板同理。
#    - 原流程：读取区段 3/4 的纯色值 -> 重着色模板 -> 按区段 2 的连通块数平铺。
#      变体流程：统计区段 3/4 的颜色频率，取出现次数最多的颜色 -> 其余步骤不变。
#      差异只落在“取色”这一步，输入构造和输出结构仍与原题一致。
# b) 复杂度校准
#    - 色板固定为 2-3 种干扰色 + 目标色的条纹，目标色面积始终超过一半，避免多数色不明显。
#    - 同时目标色分散到 2-3 段条纹中，避免退化成“一个大纯色块 + 少量噪声”的近原题视觉。
# c) 隐式假设审查
#    - `_split_sections` 只依赖整行/整列蓝色分隔线；色板内部改成多色条纹不影响。
#    - 原 helper 名为 `_solid_color`，但实现其实已经是 argmax 频率；旧注释依赖“色板纯色”，
#      这个前提在变体中不成立，所以这里显式改名为 `_majority_color` 并把文档改成多数色解码。
#    - 模板 bbox、裁剪、计数板连通块数量、平铺方向均不依赖色板纯色性，因此保持不变。
# d) 放置合法性
#    - 条纹色板仍是完整矩形 section，只在 section 内部分条填色，不与其他 section 重叠。
#    - 条纹方向总是沿色板的可变轴展开，因此 section 的共享轴尺寸继续与其他区段对齐，
#      `_join_sections` 无需改动。
#    - 生成器强制每个色板至少 3 种颜色且 `unique_colors > 1`，并保证唯一多数色，solver 不会遇到歧义。
# e) 视觉可观察性
#    - 人类从少量样例可直接看到：输出只用了色板中的某一种颜色，而不是把所有条纹搬到输出；
#      对照多组样例可归纳出“选条纹里占比最大的颜色”。
#    - 不使用颜色 id 的排序或数值关系；只使用像素频率这一可见属性。
# f) 可视化整洁度
#    - 条纹是规则的整行/整列带，不与模板或计数板叠加，输入四段依旧清爽。
# g) 颜色变化可推断性
#    - 输出只涉及两个目标色，且两块色板各自只需做一次“找最多”的局部归纳，复杂度低于任意映射。
# h) `main()` 的新假设与生成器保证
#    - 新 solver 假设色板存在唯一 modal color；生成器通过目标色面积 > 50% 保证这一点。
#    - 新 solver 仍假设前四个 section 顺序不变、模板非空、计数板按 4 连通计数；生成器保持原约束。


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = 1

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _majority_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two striped palette bars, recolor a cropped template, and repeat it by a stencil count.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a black-background template; section 2 is a component-count stencil.
    - Sections 3 and 4 are multicolor striped palette bars whose most frequent colors encode foreground and background.

    Rule:
    - Crop section 1 to the bbox of its non-black cells.
    - Read sections 3 and 4 by majority color, then recolor the cropped template with those foreground/background colors.
    - Count 4-connected non-black components in section 2 and repeat the recolored tile that many times, inserting one background-colored spacer row/column between copies.

    Key insight: The repetition logic is unchanged; the only extra reasoning step is that each palette section must be summarized by its modal color instead of being read as a solid swatch.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _majority_color(foreground_block)
    background = _majority_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), background, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), background, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)             # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)                # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)            # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))         # fg/bg + swatch distractor colors
TEMPLATE_COLOR = 4                      # shape drawn in yellow inside template
STENCIL_COLOR = 3                       # blob color in stencil section
SEPARATOR_COLOR = 1                     # blue separator rows/cols
SWATCH_DIM_RANGE = (5, 9)               # larger varying axis so striped palettes are visible
SWATCH_DISTRACTOR_CHOICES = (2, 3)      # number of non-target stripe colors
STENCIL_VARYING_EXTRA = 4               # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200                # per-blob placement attempts in stencil
GEN_RETRIES = 300                       # outer generate_input retry limit


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _split_positive(total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(int(cut - prev))
        prev = cut
    return vals


def _make_majority_swatch(shared_dim: int, target_color: int, other_target: int, horizontal: bool) -> np.ndarray:
    """Create a multicolor striped swatch whose most frequent color is `target_color`."""
    varying = random.randint(*SWATCH_DIM_RANGE)
    n_distractors = 2 if varying <= 6 else random.choice(SWATCH_DISTRACTOR_CHOICES)
    max_target_total = varying - n_distractors
    min_target_total = max(3, varying // 2 + 1)
    target_total = random.randint(min_target_total, max_target_total)
    distractor_total = varying - target_total

    n_target_stripes = 2 if target_total <= 4 else random.choice((2, 3))
    target_widths = _split_positive(target_total, n_target_stripes)
    distractor_widths = _split_positive(distractor_total, n_distractors)

    distractor_pool = [c for c in COLOR_POOL if c not in {target_color, other_target}]
    distractor_colors = random.sample(distractor_pool, n_distractors)

    stripes: list[tuple[int, int]] = []
    for idx, width in enumerate(target_widths):
        if idx < len(distractor_widths):
            stripes.append((distractor_colors[idx], distractor_widths[idx]))
        stripes.append((target_color, width))
    for idx in range(len(target_widths), len(distractor_widths)):
        stripes.append((distractor_colors[idx], distractor_widths[idx]))

    blocks = []
    if horizontal:
        for color, width in stripes:
            blocks.append(np.full((shared_dim, width), color, dtype=int))
        swatch = np.hstack(blocks)
    else:
        for color, width in stripes:
            blocks.append(np.full((width, shared_dim), color, dtype=int))
        swatch = np.vstack(blocks)

    assert swatch.shape == ((shared_dim, varying) if horizontal else (varying, shared_dim))
    assert int(np.unique(swatch).size) >= 3
    assert int(_majority_color(swatch)) == int(target_color)
    return swatch


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    """Join sections with single-pixel separator lines of SEPARATOR_COLOR."""
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for s in sections[1:]:
            parts.extend([sep, s])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for s in sections[1:]:
        parts.extend([sep, s])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs (1x2 or 2x1) of STENCIL_COLOR on a black grid.
    Blobs are separated by >=1 cell of black in all directions."""
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= n_blobs:
            break
        br, bc = (1, 2) if random.choice([True, False]) else (2, 1)

        if h - 1 - br < 1 or w - 1 - bc < 1:
            continue
        r = random.randint(1, h - 1 - br)
        c = random.randint(1, w - 1 - bc)

        r0, c0 = max(0, r - 1), max(0, c - 1)
        r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
        if np.any(grid[r0:r1, c0:c1] != 0):
            continue

        grid[r:r + br, c:c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1:crop_h + 1, 1:crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_majority_swatch(shared_dim, fg_color, bg_color, horizontal)
    bg_swatch = _make_majority_swatch(shared_dim, bg_color, fg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
