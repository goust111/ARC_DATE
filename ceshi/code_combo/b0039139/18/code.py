# Combo variant of b0039139 (index 18)
# combo_spec: D1/2+D4/1+D7/0
# Name: 二维网格中按分量位置交替着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   concrete_behavior:
#     - 输入仍由 3 条蓝色整行/整列分隔线切成四段, 顺序保持 template / stencil / fg_swatch / bg_swatch。
#     - template 仍是黑底黄色(4)单连通形状, solver 仍先裁到其非黑像素的外接矩形。
#     - 但输出不再是一维条带。stencil 中的连通分量现在被布置成一个完整的二维矩阵, 这里具体化为 2x2
#       或 2x3 的位置网格；哪个分量在上/下/左/右, 就决定对应副本位于输出矩阵的哪个行列。
#     - 每个输出位置 `(r, c)` 的着色不再取决于“一维第几个副本”, 而取决于二维奇偶性：
#       若 `(r + c) % 2 == 0`, 则模板像素用区段3、背景像素用区段4；否则交换两者。
#   behavior_trace:
#     - 具体例子：若 stencil 有 6 个分量, 排成两行三列, 左上到右下分别位于
#       `(0,0) (0,1) (0,2) / (1,0) (1,1) (1,2)`。
#     - 原题会只读出“共有 6 个分量”, 然后沿单一轴做 6 次同色重复。
#     - D1 单变体会把这 6 个副本按一维顺序做 A/B/A/B/A/B 换色, 但仍是一条带。
#     - 本融合版则输出一个 2x3 网格: `(0,0)(0,2)(1,1)` 用 A 前景+B 背景, `(0,1)(1,0)(1,2)`
#       用 B 前景+A 背景。于是分量的二维空间位置同时控制了“副本放哪”和“副本用哪套颜色”。
#   complexity_calibration:
#     - 若允许任意因数分解, N=2 或 N=3 会退化成 1xN / Nx1, 视觉上难与原题区分；因此生成器只取
#       2x2 与 2x3 两档, 强制两维都大于 1, 让 D4/1 的二维化稳定可见。
#     - 若再叠加 D7 提示里“面积决定旋转”等额外编码, 人类需要同时推断位置、旋转、颜色三套关系,
#       复杂度过高且样例数不足。因此这里把 D7/0 收敛成最直接、最可观察的实现：位置只编码位置。
#     - 棋盘奇偶只产生两套配色, 且每题里会重复出现 2-3 次, 足以支撑人类从少量样例归纳。
#   implicit_assumption_audit:
#     - 原 `n_repeats = len(components_mask(...))` 只依赖数量；融合后失效, 必须保留每个分量的空间坐标。
#     - 原 `tile = np.where(...)` 只构造一个统一配色 tile；融合后每个网格位置都可能翻转前景/背景。
#     - 原 solver 把分隔线方向直接映射到输出平铺轴；融合后分隔线方向只负责切四段, 不再决定输出是一维横条还是竖条。
#     - 原输出中的分隔带总是同一种背景色；融合后不同格点背景不同, 不能再假设全局唯一背景。
#     - 新冲突点在于“双色棋盘 + 单像素间隔”可能让某些间隔色归属不清。这里将每个格子的右侧/下侧间隔
#       统一归该格子所有, 即每个宏块先铺本格背景, 再在其左上角写入模板图案, 从而避免冲突。
#   placement_legality:
#     - template 仍放在 7x7 黑底区段里, 保持 1 像素外边框, 不会越界。
#     - 为了在 stencil 里合法放下 2x3 的二维组件矩阵, 生成器固定使用 7x7 的 stencil 画布, 并把每个组件
#       画成 2x1 竖条；这样两行可放在起点 1/4, 三列可放在起点 1/3/5, 分量之间始终至少隔 1 格黑色。
#     - swatch 仍是纯色矩形, `_solid_color()` 的前提不变。
#   visual_observability:
#     - 人类可以直接看到 stencil 是“上两排/下两排”的二维点阵, 输出也对应成 2 行 2/3 列的矩阵。
#     - 输出中的颜色翻转遵循清晰的棋盘式交替, 且线索完全来自可见的位置关系, 不依赖颜色编号大小或扫描顺序。
#   cleanliness:
#     - 输入里只有四个大区段与规则排列的细小竖条分量, 没有叠加标记。
#     - 输出里只有规则矩阵、单像素分隔和双色互换, 视觉结构比“位置+旋转+额外标记”更干净。
#   color_inference:
#     - 只有区段3和区段4两种颜色参与重着色, 每题里两种配色都至少重复两次, 不会出现“一次性颜色变化无法归纳”的问题。
#   new_solver_assumptions:
#     - main() 现在假设 stencil 的分量行列是可分辨的, 即同一行的分量共享同一个最小行坐标, 同一列共享同一个最小列坐标；
#       generator 用固定锚点放置 2x1 竖条来保证这一点。
#     - main() 还假设输出位置集合形成一个完整的矩形网格；generator 只生成满 2x2 或满 2x3 的布局, 不制造空洞槽位。
#     - main() 依赖裁剪后的模板 bbox 内至少有一个背景像素, 才能让“前景/背景互换”在 tile 内可见；generator 显式拒绝满实心模板。


TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
COLOR_POOL = list(range(2, 10))
GRID_SHAPES = [(2, 2), (2, 3)]
TEMPLATE_CANVAS_SIZE = 7
SPRITE_SIZE = 5
DENSITY_RANGE = (0.45, 0.75)
SWATCH_DIM_RANGE = (3, 8)
GEN_RETRIES = 300


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = SEPARATOR_COLOR

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        parts = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", parts

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    parts = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", parts


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _palette_for_cell(row: int, col: int, color_a: int, color_b: int) -> tuple[int, int]:
    if (int(row) + int(col)) % 2 == 0:
        return int(color_a), int(color_b)
    return int(color_b), int(color_a)


def _component_grid(mask: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return 0, 0, []

    anchors = [(int(coords[:, 0].min()), int(coords[:, 1].min())) for coords in comps]
    row_keys = sorted({r for r, _ in anchors})
    col_keys = sorted({c for _, c in anchors})
    row_index = {value: idx for idx, value in enumerate(row_keys)}
    col_index = {value: idx for idx, value in enumerate(col_keys)}
    positions = [(row_index[r], col_index[c]) for r, c in anchors]
    return len(row_keys), len(col_keys), positions


def _build_output(
    tile_mask: np.ndarray,
    n_rows: int,
    n_cols: int,
    positions: list[tuple[int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    if n_rows <= 0 or n_cols <= 0 or not positions:
        fg, bg = _palette_for_cell(0, 0, color_a, color_b)
        return np.where(tile_mask, fg, bg).astype(int)

    tile_h, tile_w = tile_mask.shape
    out_h = n_rows * tile_h + (n_rows - 1)
    out_w = n_cols * tile_w + (n_cols - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    for row, col in positions:
        fg, bg = _palette_for_cell(row, col, color_a, color_b)
        r0 = row * (tile_h + 1)
        c0 = col * (tile_w + 1)
        block_h = tile_h + (1 if row < n_rows - 1 else 0)
        block_w = tile_w + (1 if col < n_cols - 1 else 0)
        out[r0 : r0 + block_h, c0 : c0 + block_w] = bg
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = np.where(tile_mask, fg, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a template, place one copy per stencil position, and alternate the palette by 2D parity.

    Input structure:
    - Blue separator rows or columns still divide the input into four ordered sections.
    - Section 1 is a black template containing a color-4 shape; section 2 is a black stencil with disjoint non-black components.
    - Sections 3 and 4 are solid swatches that provide the two colors reused throughout the output.

    Rule:
    - Crop section 1 to the bounding box of its non-black pixels.
    - Read every 4-connected component in section 2 and map it to a grid coordinate by its spatial row and column among the other components.
    - Create one output tile at each such coordinate. Cells with even `(row + col)` parity use section 3 for template pixels and section 4 for the cropped background; odd-parity cells swap those roles.
    - Arrange the tiles as a 2D matrix with one-pixel separators, where each tile owns the separator band immediately to its right and below, colored with that tile's background.

    Key insight: The stencil no longer says only “how many copies”; its 2D component layout determines both where each copy goes in the matrix and whether that location keeps or inverts the two swatch colors.
    """
    g = np.asarray(grid, dtype=int)
    _orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    color_a = _solid_color(foreground_block)
    color_b = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    n_rows, n_cols, positions = _component_grid(repetition_pattern != 0)
    return _build_output(tile_mask, n_rows, n_cols, positions, color_a, color_b)


# --------------- generator (combo) ---------------


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
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


def _build_stencil(n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.zeros((TEMPLATE_CANVAS_SIZE, TEMPLATE_CANVAS_SIZE), dtype=int)
    row_anchors = [1, 4]
    col_anchors = [1, 5] if n_cols == 2 else [1, 3, 5]

    for row in range(n_rows):
        for col in range(n_cols):
            r0 = row_anchors[row]
            c0 = col_anchors[col]
            grid[r0 : r0 + 2, c0] = STENCIL_COLOR
    return grid


def _make_template(shape: np.ndarray) -> np.ndarray:
    template = np.zeros((TEMPLATE_CANVAS_SIZE, TEMPLATE_CANVAS_SIZE), dtype=int)
    template[1 : 1 + SPRITE_SIZE, 1 : 1 + SPRITE_SIZE] = shape
    return template


def _sample_shape() -> np.ndarray | None:
    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        SPRITE_SIZE,
        SPRITE_SIZE,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )
    shape_mask = shape != 0
    shape_bbox = bbox(shape_mask)
    if shape_bbox is None:
        return None
    cropped = crop_bbox(shape_mask, shape_bbox)
    if bool(np.all(cropped)):
        return None
    return shape


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_rows, n_cols = random.choice(GRID_SHAPES)
    shape = _sample_shape()
    if shape is None:
        return None

    color_a, color_b = random.sample(COLOR_POOL, 2)
    template = _make_template(shape)
    stencil = _build_stencil(n_rows, n_cols)
    fg_swatch = _make_swatch(TEMPLATE_CANVAS_SIZE, color_a, horizontal)
    bg_swatch = _make_swatch(TEMPLATE_CANVAS_SIZE, color_b, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], TEMPLATE_CANVAS_SIZE, horizontal)

    try:
        out = main(inp)
    except Exception:
        return None

    tile_mask = crop_bbox(template != 0, bbox(template != 0))
    expected_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    expected = _build_output(tile_mask, n_rows, n_cols, expected_positions, color_a, color_b)
    if np.array_equal(out, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
