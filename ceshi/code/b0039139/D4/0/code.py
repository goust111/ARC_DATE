# Variant of b0039139
# Dimension: D4 — 副本平铺方向 (grid_shape)
# Change: 平铺方向由分隔线方向决定——水平分隔线(行)则垂直平铺，垂直分隔线(列)则水平平铺... → 平铺方向与分隔线方向相同

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   preserved_structure:
#   - 输入仍然由 3 条蓝色分隔线拆成 4 个区段，顺序仍是 template / stencil / fg_swatch / bg_swatch。
#   - template 仍是黑底黄形状，solver 仍只取其非黑像素外接矩形。
#   - stencil 仍用 4-连通非黑块个数编码重复次数 N。
#   - 输出仍在副本间插入 1 像素背景色分隔带。
#   changed_dimension:
#   - 原任务里，“区段排布方向” 与 “输出平铺方向” 相同：
#       列分隔(区段横向排布) -> 输出横向平铺；
#       行分隔(区段纵向排布) -> 输出纵向平铺。
#   - 本变体改成“输出平铺方向 = 分隔线方向本身”：
#       行分隔 -> 输出横向平铺；
#       列分隔 -> 输出纵向平铺。
#   concrete_trace:
#   - 例：若输入是四块上下堆叠，中间隔着蓝色整行，那么 `_split_sections()` 会返回 `layout="vertical"`。
#   - 原逻辑在 `layout="vertical"` 时走 `vstack`，得到自上而下重复。
#   - 变体逻辑在同样输入上要走 `hstack`，得到从左到右重复。
#   - 差异只发生在“把已重着色的 tile 沿哪个轴复制”这一步。
#   complexity_calibration:
#   - 规则只有两档：行分隔 vs 列分隔；不引入额外子类或多步条件。
#   - N 仍取 2-6，足以让方向差异在样例中稳定可见。
#   - crop_h / crop_w 保持 2-5，因此即使模板接近方形，输出整体长宽比也会明显区分横铺与竖铺。
#   implicit_assumption_audit_main:
#   - `_split_sections()` 仍假设分隔线是完整的整行或整列蓝色；generator 继续严格构造这一点。
#   - `sections[:4]` 仍假设恰好有四个非空区段；generator 固定生成四段。
#   - `bbox(template != 0)` 仍假设 template 非空；generator 用 `random_sprite(..., connectivity=4)` 保证存在单个连通形状。
#   - `components_mask(stencil != 0)` 仍把所有非黑色 blob 当作计数对象；generator 继续只在 stencil 中使用颜色 3。
#   - 新增关键假设：solver 现在把 `layout="vertical"` 映射到横向平铺、`layout="horizontal"` 映射到纵向平铺；
#     因而 generator 必须显式让输出轴与区段排布轴相反，否则会退化回原任务。
#   placement_legality:
#   - 区段拼接时共用轴长度仍由 template 的外框尺寸决定，故 `_join_sections()` 不会错位。
#   - stencil 仍在黑底中放置互相至少隔 1 格的 2-cell blob，避免组件误合并。
#   - swatch 仍是纯色矩形，因此 `_solid_color()` 稳定。
#   visual_observability:
#   - 人类可从 2-4 个样例归纳：看到“上下分区”的样例输出反而横向展开，看到“左右分区”的样例输出反而纵向展开。
#   - 该规律完全来自可见的空间关系，不依赖颜色 id、排序或隐藏元数据。
#   cleanliness:
#   - 没有新增叠加层；变体只改变输出排布轴，因此输入区域依旧清爽、输出也只是规则条带。


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


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the chosen output axis with 1-pixel bg separators."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _output_horizontal_from_layout(layout: str) -> bool:
    """Map section layout to output axis for the D4 inversion variant."""
    return layout == "vertical"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a cropped template and repeat it using a component-count stencil.

    Input structure:
    - Full blue rows or full blue columns divide the input into four ordered sections.
    - The first section is a black field with a yellow shape template; the second is a black counting panel.
    - The last two sections are solid swatches that specify the output foreground and background colors.

    Rule:
    - Crop the template to the bounding box of its non-black cells and recolor that crop with the two swatch colors.
    - Count the 4-connected non-black components in the stencil to obtain the repeat count.
    - Repeat the recolored crop that many times with a one-cell background separator, but tile along the same direction as the blue separator lines themselves.

    Key insight:
    - The stencil says only how many copies to make, not where to place them.
    - The subtle part of this variant is the reversed mapping: separator rows create a left-to-right strip, while separator columns create a top-to-bottom strip.
    """
    g = np.asarray(grid, dtype=int)
    layout, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    return _tile_repeat(
        tile,
        n_repeats,
        background,
        horizontal=_output_horizontal_from_layout(layout),
    )


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
GEN_RETRIES = 300                    # outer generate_input retry limit


def _make_swatch(shared_dim: int, color: int, sections_horizontal: bool) -> np.ndarray:
    """Create a solid color swatch. `shared_dim` is the axis shared by all four sections."""
    varying = random.randint(*SWATCH_DIM_RANGE)
    if sections_horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, sections_horizontal: bool) -> np.ndarray:
    """Join sections with single-pixel separator lines of SEPARATOR_COLOR."""
    if sections_horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, sections_horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs on black, separated by at least one black cell."""
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if sections_horizontal else (varying_dim, shared_dim)
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


def _gen_attempt() -> np.ndarray | None:
    sections_horizontal = random.choice([True, False])
    output_horizontal = not sections_horizontal
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
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal=output_horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = template.shape[0] if sections_horizontal else template.shape[1]
    stencil = _build_stencil(n_repeats, shared_dim, sections_horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, sections_horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, sections_horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, sections_horizontal)

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
