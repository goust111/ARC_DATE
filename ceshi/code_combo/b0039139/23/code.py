# Combo variant of b0039139 (index 23)
# combo_spec: D9/0+D12/2
# Name: 条纹色板配亮度权重着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis
# a) 具体化与行为追踪
#    - 保留原题四段布局：模板 / 计数板 / 前景色板 / 背景色板，仍由整行或整列蓝色分隔线切开。
#    - D9/0 让区段 3 和 4 变成多色条纹，不再能直接读单一填充值；各自的目标色改为“该色板中出现最多的颜色”。
#    - D12/2 让模板内部不再是“所有非黑像素统一着前景色”。这里把“亮度权重”离散成可见的两档：
#      若某个模板像素在裁剪后形状中的 4 邻域里有 2 个或更多背景方向，则视为更暴露，输出成混合色；
#      否则输出成前景主色。bbox 外侧按背景处理，所以角点天然比内部像素更亮。
#    - 融合点在于：混合色并不是另起一套隐藏映射，而是直接由两块条纹色板中“共同出现的次要条纹色”提供。
#      例如前景色板多数色为 8、背景色板多数色为 2，两者都含少量 6，则输出里内部像素用 8，
#      暴露像素用 6，空白与分隔带用 2。计数板仍只决定平铺次数。
# b) 复杂度校准
#    - 输出固定只用三种角色色：前景主色、背景主色、共享混合色。避免颜色映射过散，仍能明显区别于原题。
#    - 每块色板至少含 3 种颜色：多数目标色、共享混合色、仅该色板出现的私有干扰色。这样“共同次要色”可见，
#      同时又不会退化成只有两色的简单二选一。
#    - 模板随机形状需同时含“暴露度 >= 2”的像素和“暴露度 <= 1”的像素，确保三色输出真正出现，
#      而不是退化成全混合色或全前景色。
# c) 隐式假设审查
#    - `_split_sections` 只要求整行/整列蓝色分隔线；条纹色板不破坏该前提。
#    - 原题默认色板是纯色矩形，因此 `_solid_color` 的语义不再准确；这里显式改为 `_majority_color`。
#    - 原题把模板裁剪后直接二值着色；融合后必须先计算裁剪形状的局部暴露度，再决定前景像素是主色还是混合色。
#      这要求生成器保证形状非空，且共享混合色能从两块色板中唯一恢复。
#    - 新 solver 假设两块色板至少共享一种非多数色；生成器进一步保证共享的只有一个显著候选，
#      因而 `_shared_accent_color` 不会歧义。
# d) 放置合法性
#    - 模板仍置于 1 像素黑边框内；计数板 blob 仍保持互相至少隔 1 格黑色，避免连通块数量串联。
#    - 条纹色板仍是完整矩形 section，只在 section 内沿可变轴分条，不越界、不重叠，也不影响 shared axis 对齐。
#    - 生成器先构造输出，再反推输入，且通过 `main(inp) == output` 做 oracle 检查。
# e) 视觉可观察性
#    - 人类能从少量样例看到：输出里出现第三种颜色，但它不是计数板颜色，也不是模板原色，而是两块条纹色板中共同出现的颜色。
#    - 同时又能看到第三色只出现在形状边角/细颈等更“暴露”的像素上，内部厚实区域仍保留前景主色。
#    - 规则只依赖可见结构：条纹出现频率、两色板共有颜色、像素周围背景多少；不依赖颜色编号大小规律。
# f) 可视化整洁度
#    - 模板区、计数板区、两块色板区保持原题的强分区结构，减少多规则叠加造成的混乱。
#    - 输出虽然从双色变三色，但三色的职责清晰：背景填充、主体主色、边缘混合色，读图负担仍低。
# g) 变化的可推断性
#    - 混合色不是任意颜色，而是两块色板共同暴露出的唯一公共次要色，这给颜色来源提供了直接线索。
#    - 暴露度阈值固定为“背景邻居数 >= 2”，在角点、细线、凸起处会重复出现，少量样例就能支撑归纳。
# h) `main()` 的新假设与生成器保证
#    - `main()` 假设形状裁剪后至少含一个低暴露像素和一个高暴露像素；生成器通过 `_valid_shape_mask` 保证。
#    - `main()` 假设两块色板的共同非多数色可视且唯一；生成器通过“共享混合色 + 各自私有干扰色”的构造保证。
#    - `main()` 仍假设前四个 section 顺序固定、模板非空、计数板用 4 连通计数；这些均沿用原题约束。


REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
OUTPUT_COLOR_POOL = [2, 5, 6, 7, 8, 9]
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (7, 10)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 400


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = SEPARATOR_COLOR

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


def _shared_accent_color(
    foreground_block: np.ndarray,
    background_block: np.ndarray,
    foreground: int,
    background: int,
) -> int:
    fg_vals, fg_counts = np.unique(foreground_block, return_counts=True)
    bg_vals, bg_counts = np.unique(background_block, return_counts=True)

    fg_freq = {int(v): int(c) for v, c in zip(fg_vals, fg_counts) if int(v) != int(foreground)}
    bg_freq = {int(v): int(c) for v, c in zip(bg_vals, bg_counts) if int(v) != int(background)}
    shared = sorted(set(fg_freq) & set(bg_freq))
    if not shared:
        return int(foreground)
    return int(max(shared, key=lambda color: fg_freq[color] + bg_freq[color]))


def _exposure_counts(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exposure = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            count = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not bool(mask[rr, cc]):
                    count += 1
            exposure[r, c] = count
    return exposure


def _recolor_tile(mask: np.ndarray, foreground: int, background: int, blend: int) -> np.ndarray:
    exposure = _exposure_counts(mask)
    tile = np.full(mask.shape, background, dtype=int)
    tile[mask & (exposure >= 2)] = blend
    tile[mask & (exposure <= 1)] = foreground
    return tile


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode striped palettes, brighten exposed template pixels with a shared accent, and repeat the tile by a stencil count.

    Input structure:
    - Blue (1) separator rows or columns divide the input into four ordered sections.
    - Section 1 is a black-background template; section 2 is a component-count stencil.
    - Sections 3 and 4 are multicolor striped palette bars whose majority colors encode the main foreground and background.
    - The same secondary accent color appears in both palette bars and serves as the blend color for exposed template pixels.

    Rule:
    - Crop section 1 to the bbox of its non-black cells.
    - Read sections 3 and 4 by majority color to get the main foreground/background, and recover the blend color as the strongest shared non-majority stripe color across both palette bars.
    - Recolor the cropped template so background stays the background color, template pixels with at least two background-facing 4-neighbors become the blend color, and less exposed template pixels become the foreground color.
    - Count 4-connected non-black components in section 2 and repeat that recolored tile that many times, inserting one background-colored spacer row/column between copies.

    Key insight: The palette stripes now carry two pieces of information at once: majority color picks the main role, while the only shared accent across both palettes explains the third output color used on high-exposure shape pixels.
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
    blend = _shared_accent_color(foreground_block, background_block, foreground, background)

    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = _recolor_tile(tile_mask, foreground, background, blend)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    return _tile_repeat(tile, n_repeats, background, orientation == "horizontal")


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


def _make_striped_palette(
    shared_dim: int,
    target_color: int,
    blend_color: int,
    private_color: int,
    horizontal: bool,
) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    min_target = max(4, varying // 2 + 1)
    target_total = random.randint(min_target, varying - 3)
    remaining = varying - target_total
    blend_total = random.randint(2, remaining - 1)
    private_total = remaining - blend_total

    target_parts = _split_positive(target_total, 2 if target_total <= 4 else random.choice((2, 3)))
    blend_parts = _split_positive(blend_total, 1 if blend_total <= 2 else random.choice((1, 2)))
    private_parts = _split_positive(private_total, 1)

    accents = [(blend_color, width) for width in blend_parts] + [(private_color, width) for width in private_parts]
    random.shuffle(accents)

    stripes: list[tuple[int, int]] = []
    for idx, width in enumerate(target_parts):
        if idx < len(accents):
            stripes.append(accents[idx])
        stripes.append((target_color, width))
    stripes.extend(accents[len(target_parts) :])

    blocks = []
    if horizontal:
        for color, width in stripes:
            blocks.append(np.full((shared_dim, width), color, dtype=int))
        swatch = np.hstack(blocks)
    else:
        for color, width in stripes:
            blocks.append(np.full((width, shared_dim), color, dtype=int))
        swatch = np.vstack(blocks)

    assert int(_majority_color(swatch)) == int(target_color)
    assert blend_color in set(map(int, np.unique(swatch)))
    assert private_color in set(map(int, np.unique(swatch)))
    return swatch


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
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

        grid[r : r + br, c : c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _valid_shape_mask(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    exposure = _exposure_counts(mask)[mask]
    return bool(np.any(exposure >= 2) and np.any(exposure <= 1))


def _generate_shape() -> np.ndarray:
    for _ in range(200):
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
        if _valid_shape_mask(shape != 0):
            return shape
    raise RuntimeError("Failed to sample a shape with both exposure bands")


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape = _generate_shape()
    tile_mask = shape != 0

    fg_color, bg_color, blend_color, fg_private, bg_private = random.sample(OUTPUT_COLOR_POOL, 5)
    tile = _recolor_tile(tile_mask, fg_color, bg_color, blend_color)
    if not (np.any(tile == fg_color) and np.any(tile == blend_color)):
        return None
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    crop_h, crop_w = shape.shape
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_striped_palette(shared_dim, fg_color, blend_color, fg_private, horizontal)
    bg_swatch = _make_striped_palette(shared_dim, bg_color, blend_color, bg_private, horizontal)
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
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
