# Variant of b0039139
# Dimension: D12 — 模板中特殊像素的处理 (meta)
# Change: 模板区段中只有两种像素值：色4(模板形状)和色0(背景)，不存在其他颜色... → 模板包含多种前景色
#
# Design analysis
# spec:
#   preserved_layout:
#     - 仍是四分区输入，分区顺序不变：template / stencil / fg_swatch / bg_swatch。
#     - 仍由单像素蓝色分隔线切开；分隔方向决定输出平铺方向。
#     - stencil 仍通过非黑连通分量个数编码重复次数。
#   D12_core_change:
#     - template 裁剪后不再是二值掩码，而是 0/4/5 三值图案。
#     - 裁剪区域内：4 -> 第三区段纯色；5 -> 第二分区中“最常见的非黑颜色”；0 -> 第四区段纯色。
#     - 为了让“最常见颜色”在视觉上可推断，stencil 允许两种 blob 颜色，但 dominant color 唯一。
#   behavior_trace:
#     - 原逻辑里 `template != 0` 会把 4 和 5 一起压成 True，随后整块前景都被刷成同一个 fg。
#     - 现在若 template crop 是
#         [[4, 5],
#          [0, 4]]
#       且 fg_swatch=8, bg_swatch=2, stencil 的 dominant 非黑色=6，
#       原逻辑会得到 [[8, 8], [2, 8]]，变体逻辑必须得到 [[8, 6], [2, 8]]。
#     - 差异只发生在“模板裁剪后如何着色”这一步；重复次数和平铺方式不变。
#   complexity_calibration:
#     - 不新增第五分区，避免把规则改成多源索引。
#     - stencil 至多使用两种非黑颜色，减少人类归纳负担，同时保证 dominant color 可见。
#     - template 强制同时含 4 和 5，避免样本退化回原始单色模板。
#   implicit_assumption_audit:
#     - `_split_sections(...)` 只依赖蓝色整行/整列分隔，不受 D12 影响，可保留。
#     - 旧版 `template_mask = template != 0` 依赖“所有前景同色”，在变体中失效；必须保留裁剪后的原像素值。
#     - 旧版 `tile = np.where(tile_mask, foreground, background)` 也依赖二值模板；必须改成显式 4/5 映射。
#     - `components_mask(repetition_pattern != 0, connectivity=4)` 只依赖 blob 是否非黑，不要求单一颜色，因此可保留。
#     - `_solid_color` 仍只用于两块 swatch；生成器保证 swatch 纯色，所以假设仍成立。
#   placement_legality:
#     - template 外围仍保留 1 像素黑框，bbox 裁剪稳定。
#     - stencil blob 仍是互不接触的 2-cell 组件，因此组件计数稳定且 dominant 非黑色可控。
#     - 输出平铺仍只在一条轴上复制，分隔条使用 bg_color，避免副本融合。
#   visual_observability:
#     - 每个样例都强制出现 4 和 5，两种模板像素在输出中会变成两种不同前景色。
#     - 第二种前景色直接出现在 stencil 中，且是唯一 dominant 非黑色，不依赖颜色编号排序。
#     - minority stencil blobs 只偶尔出现一种次级颜色，既能提示“看颜色”，又不会让区域显得杂乱。
#   generator_guarantees_for_new_main:
#     - template crop 内至少 1 个 4 和至少 1 个 5。
#     - stencil 至少 1 个非黑 blob，且 dominant 非黑色唯一；该颜色与 fg/bg 都不同。
#     - 所有 section 的 shared axis 仍对齐，所以 `_join_sections` / `_split_sections` 的结构假设成立。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


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


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _dominant_nonzero_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block[block != 0], return_counts=True)
    if vals.size == 0:
        return 0
    return int(vals[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped multicolor template and repeat it using a component-count board.

    Input structure:
    - Blue (1) separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a black-backed template that may contain both 4-pixels and 5-pixels.
    - Section 2 is the repetition stencil; sections 3 and 4 are solid swatches for the
      output color of template-4 pixels and for the output background.

    Rule:
    - Crop section 1 to the bbox of its non-black content.
    - Recolor the crop cell-by-cell: 4 takes the section-3 color, 5 takes the most frequent
      non-black color from section 2, and 0 takes the section-4 color.
    - Count 4-connected non-black components in section 2, then repeat the recolored crop
      that many times with one background-colored separator row/column between copies.

    Key insight: the stencil still serves two roles at once: its component count decides how
    many copies to place, while its dominant visible color supplies the second foreground color.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    accent = _dominant_nonzero_color(repetition_pattern)
    template_crop = crop_bbox(template, template_bbox)

    tile = np.full(template_crop.shape, background, dtype=int)
    tile[template_crop == 4] = foreground
    tile[template_crop == 5] = accent

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


# --------------- generator (D12 multicolor template) ---------------

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
OUTPUT_COLOR_POOL = [2, 3, 6, 7, 8, 9]
TEMPLATE_PRIMARY = 4
TEMPLATE_SECONDARY = 5
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 300


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


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


def _grow_connected_subset(mask: np.ndarray, target_size: int) -> np.ndarray:
    coords = [tuple(x) for x in np.argwhere(mask)]
    chosen = {random.choice(coords)}

    while len(chosen) < target_size:
        frontier = []
        for r, c in chosen:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in chosen:
                    frontier.append((rr, cc))
        if not frontier:
            frontier = [coord for coord in coords if coord not in chosen]
        chosen.add(random.choice(frontier))

    out = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        out[r, c] = True
    return out


def _build_multicolor_shape(crop_h: int, crop_w: int) -> np.ndarray:
    density = random.uniform(*DENSITY_RANGE)
    base = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_PRIMARY],
        connectivity=4,
        symmetry="not_symmetric",
    )
    mask = base != 0
    area = int(mask.sum())
    if area < 2:
        return _build_multicolor_shape(crop_h, crop_w)

    min_secondary = 1 if area <= 3 else 2
    max_secondary = area - 1
    secondary_size = random.randint(min_secondary, max_secondary)
    secondary_mask = _grow_connected_subset(mask, secondary_size)

    shape = np.zeros_like(base, dtype=int)
    shape[mask] = TEMPLATE_PRIMARY
    shape[secondary_mask] = TEMPLATE_SECONDARY

    if not np.any(shape == TEMPLATE_PRIMARY) or not np.any(shape == TEMPLATE_SECONDARY):
        return _build_multicolor_shape(crop_h, crop_w)
    return shape


def _build_stencil(blob_colors: list[int], shared_dim: int, horizontal: bool) -> np.ndarray | None:
    min_varying = max(4, len(blob_colors) * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= len(blob_colors):
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

        grid[r : r + br, c : c + bc] = blob_colors[placed]
        placed += 1

    return grid if placed == len(blob_colors) else None


def _choose_stencil_blob_colors(n_blobs: int, accent_color: int, alternate_color: int) -> list[int]:
    if n_blobs <= 2:
        dominant = n_blobs
    else:
        dominant = random.randint((n_blobs // 2) + 1, n_blobs)

    colors = [accent_color] * dominant + [alternate_color] * (n_blobs - dominant)
    random.shuffle(colors)
    return colors


def _recolor_template(shape: np.ndarray, fg_color: int, accent_color: int, bg_color: int) -> np.ndarray:
    tile = np.full(shape.shape, bg_color, dtype=int)
    tile[shape == TEMPLATE_PRIMARY] = fg_color
    tile[shape == TEMPLATE_SECONDARY] = accent_color
    return tile


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    shape = _build_multicolor_shape(crop_h, crop_w)

    fg_color, accent_color, bg_color, alt_stencil_color = random.sample(OUTPUT_COLOR_POOL, 4)
    blob_colors = _choose_stencil_blob_colors(n_repeats, accent_color, alt_stencil_color)

    tile = _recolor_template(shape, fg_color, accent_color, bg_color)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(blob_colors, shared_dim, horizontal)
    if stencil is None:
        return None

    if _dominant_nonzero_color(stencil) != accent_color:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
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
