# Combo variant of b0039139 (index 1)
# combo_spec: D1/2+D4/1
# Name: 交替着色棋盘网格

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   concrete_behavior:
#     - 输入仍由四个区段构成：模板、stencil、前景色板、背景色板；蓝色分隔线仍只负责切分区段。
#     - 输出不再是一条线性重复带，而是把同一裁剪模板铺成 rows x cols 的二维矩阵。
#     - 每个副本仍只使用区段3和区段4两种颜色，但颜色分配不再全局固定，而是按副本顺序奇偶交替交换。
#   behavior_trace:
#     - 原题若 stencil 有 6 个连通分量，横向布局会输出 6 个完全同色副本排成一行。
#     - 本融合版仍从 6 个连通分量出发，但先把 6 分解成 2x3（横向输入）或 3x2（纵向输入），
#       再把副本放进二维网格。
#     - 为了让 D1/2 的奇偶交替在二维里可见，副本编号采用蛇形顺序：
#       第一行从左到右，第二行从右到左，第三行再从左到右。
#       这样连续编号的奇偶翻转会落成真正的棋盘格，而不会退化成按行或按列的条纹。
#     - 例：横向输入、6 个组件时，输出位置编号为
#         1 2 3
#         6 5 4
#       因此颜色使用 A/B/A 与 B/A/B，视觉上正好是棋盘式交替。
#   complexity_calibration:
#     - 若只做二维平铺而保留单一配色，变化不够明显；若只做一维交替，则 D4/1 没有真正参与推理。
#     - 蛇形编号把“二维因数分解”和“奇偶换色”绑成同一条推理链：先确定网格，再确定每格的颜色角色。
#     - 因数分解规则固定为“选择最接近平方的非平凡因子对”；横向区段布局取较宽形状，纵向区段布局取较高形状。
#       这样输出比例稳定，且生成器能把 stencil 也排成同样的二维槽位，帮助人类观察。
#   implicit_assumption_audit:
#     - 原 main() 假设 section 2 只编码数量；这里不够，因为奇偶换色依赖顺序，顺序又要和二维位置绑定。
#     - 原 main() 只需要一个 tile；这里必须为每个编号重新决定前景/背景。
#     - 原 main() 只在单一轴向插入分隔；这里必须同时处理行间和列间分隔。
#     - 原 main() 默认所有分隔都用同一个背景色；这里每个网格单元的右/下分隔都跟随该单元自己的背景色。
#     - 新 main() 额外假设 stencil 组件能按行优先切成 rows 个等长分组；生成器用规则化槽位布局保证这一点。
#   placement_legality:
#     - 模板仍带 1 像素黑边，bbox 裁剪合法。
#     - stencil 中每个 blob 仍恰好占 2 格，且与其它 blob 至少隔 1 格黑色。
#     - 横向区段布局时，stencil 用 1x2 blob 排成 rows x cols 槽位；纵向区段布局时，用 2x1 blob 排成同构槽位。
#       这样受 shared_dim 限制的一侧只承担较小因子，始终可放下。
#     - 输出画布大小固定为 rows * tile_h + (rows - 1) 行、cols * tile_w + (cols - 1) 列，不会越界重叠。
#   observability_and_cleanliness:
#     - 输出只使用两种颜色反复交换，且交换规律由二维位置决定，2-3 个样例即可看出棋盘式交替。
#     - stencil 也被排成同样的二维阵列，解题者可以直接看到“组件数不是线性重复，而是被组织成矩阵”。
#     - 分隔条只延伸当前单元的背景色，不引入第三种输出颜色，视觉上仍是整洁的双色网格。
#   color_inference:
#     - 颜色变化只在区段3和区段4之间交换，类别数严格为 2。
#     - 因为棋盘格会同时在横向与纵向相邻位置上重复出现，样例里会反复支撑“相邻格交换 fg/bg 角色”的规律。
#   new_solver_assumptions:
#     - 连通分量数最好有非平凡因子对；生成器只采样允许二维分解的计数。
#     - 排序规则依赖“每一行恰好 cols 个组件”；生成器用固定槽位与充足间距确保分组无歧义。
#     - 棋盘效果依赖蛇形编号与网格位置一致；生成器把 stencil 与输出都按同一蛇形顺序构造。


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


def _grid_shape(n: int, horizontal: bool) -> tuple[int, int]:
    best: tuple[int, int] | None = None
    for a in range(2, int(n**0.5) + 1):
        if n % a != 0:
            continue
        b = n // a
        pair = (a, b)
        if best is None or (pair[1] - pair[0], pair[0]) < (best[1] - best[0], best[0]):
            best = pair

    if best is None:
        return (1, n) if horizontal else (n, 1)

    small, large = best
    return (small, large) if horizontal else (large, small)


def _ordered_components(mask: np.ndarray, rows: int, cols: int) -> list[np.ndarray]:
    comps = components_mask(mask, connectivity=4)
    row_major = sorted(comps, key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    ordered: list[np.ndarray] = []
    for row_idx in range(rows):
        row = row_major[row_idx * cols : (row_idx + 1) * cols]
        row = sorted(row, key=lambda coords: int(coords[:, 1].min()))
        if row_idx % 2 == 1:
            row = list(reversed(row))
        ordered.extend(row)
    return ordered


def _serpentine_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r in range(rows):
        col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in col_range:
            positions.append((r, c))
    return positions


def _tile_with_palette(tile_mask: np.ndarray, color_a: int, color_b: int, index: int) -> tuple[np.ndarray, int]:
    odd = index % 2 == 1
    foreground = color_a if odd else color_b
    background = color_b if odd else color_a
    tile = np.where(tile_mask, foreground, background).astype(int)
    return tile, background


def _build_output(
    tile_mask: np.ndarray,
    ordered_components: list[np.ndarray | int],
    rows: int,
    cols: int,
    color_a: int,
    color_b: int,
) -> np.ndarray:
    tile_h, tile_w = tile_mask.shape
    out_h = rows * tile_h + (rows - 1)
    out_w = cols * tile_w + (cols - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    for index, (_component, (grid_r, grid_c)) in enumerate(zip(ordered_components, _serpentine_positions(rows, cols)), start=1):
        tile, background = _tile_with_palette(tile_mask, color_a, color_b, index)
        r0 = grid_r * (tile_h + 1)
        c0 = grid_c * (tile_w + 1)
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
        if c0 + tile_w < out_w:
            out[r0 : r0 + tile_h, c0 + tile_w] = background
        if r0 + tile_h < out_h:
            out[r0 + tile_h, c0 : c0 + tile_w] = background
        if r0 + tile_h < out_h and c0 + tile_w < out_w:
            out[r0 + tile_h, c0 + tile_w] = background

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a template, factor the stencil count into a 2D grid, and color the copies in a checkerboard-like alternation.

    Input structure:
    - Blue separator rows or columns split the input into four ordered sections.
    - Section 1 is a black template with a color-4 shape; section 2 is a stencil of disconnected non-black 2-cell blobs.
    - Sections 3 and 4 are solid color swatches that provide the two colors reused across all output copies.

    Rule:
    - Crop section 1 to the bounding box of its non-black template pixels.
    - Count the 4-connected components in section 2, then factor that count into a non-trivial rows x cols grid using the closest factor pair; column-separated layouts use the wider orientation, row-separated layouts use the taller orientation.
    - Order the stencil components by rows, reversing every other row to obtain a snake-like sequence.
    - Place one cropped copy per ordered component into the output grid, assigning odd indices the palette (section 3 foreground, section 4 background) and even indices the swapped palette.
    - Between neighboring copies, insert one-cell gutters that inherit the background of the copy above or to the left, so the full output remains a two-color tiled matrix.

    Key insight: Section 2 no longer means only “how many copies”; its component count must be factorized into a matrix, and the snake-ordered odd/even indices of that matrix determine the checkerboard-style palette swap.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    horizontal = orientation == "horizontal"
    color_a = _solid_color(foreground_block)
    color_b = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)

    comps = components_mask(repetition_pattern != 0, connectivity=4)
    count = len(comps)
    if count == 0:
        return np.empty((0, 0), dtype=int)

    rows, cols = _grid_shape(count, horizontal)
    ordered = _ordered_components(repetition_pattern != 0, rows, cols)
    return _build_output(tile_mask, ordered, rows, cols, color_a, color_b)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNTS = (4, 6, 8, 9, 10, 12)   # only counts with non-trivial factor pairs
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
GEN_RETRIES = 400


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


def _choose_crop_size(horizontal: bool, rows: int, cols: int) -> tuple[int, int]:
    min_crop_h = max(CROP_SIZE_RANGE[0], 2 * rows - 3 if horizontal else 2)
    min_crop_w = max(CROP_SIZE_RANGE[0], 2 * cols - 3 if not horizontal else 2)
    crop_h = random.randint(min_crop_h, CROP_SIZE_RANGE[1])
    crop_w = random.randint(min_crop_w, CROP_SIZE_RANGE[1])
    return crop_h, crop_w


def _build_stencil(rows: int, cols: int, shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        blob_h, blob_w = 1, 2
        min_h = 2 * rows - 1
        min_w = 3 * cols - 1
        assert shared_dim >= min_h
        h = shared_dim
        w = random.randint(min_w, min_w + STENCIL_VARYING_EXTRA)
        grid = np.zeros((h, w), dtype=int)

        row_pad = random.randint(0, h - min_h)
        col_pad = random.randint(0, w - min_w)
        row_positions = [row_pad + 2 * r for r in range(rows)]
        col_positions = [col_pad + 3 * c for c in range(cols)]
        for r in row_positions:
            for c in col_positions:
                grid[r : r + blob_h, c : c + blob_w] = STENCIL_COLOR
        return grid

    blob_h, blob_w = 2, 1
    min_h = 3 * rows - 1
    min_w = 2 * cols - 1
    assert shared_dim >= min_w
    h = random.randint(min_h, min_h + STENCIL_VARYING_EXTRA)
    w = shared_dim
    grid = np.zeros((h, w), dtype=int)

    row_pad = random.randint(0, h - min_h)
    col_pad = random.randint(0, w - min_w)
    row_positions = [row_pad + 3 * r for r in range(rows)]
    col_positions = [col_pad + 2 * c for c in range(cols)]
    for r in row_positions:
        for c in col_positions:
            grid[r : r + blob_h, c : c + blob_w] = STENCIL_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.choice(REPEAT_COUNTS)
    rows, cols = _grid_shape(n_repeats, horizontal)

    crop_h, crop_w = _choose_crop_size(horizontal, rows, cols)
    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )

    color_a, color_b = random.sample(COLOR_POOL, 2)
    tile_mask = shape != 0
    output = _build_output(tile_mask, list(range(n_repeats)), rows, cols, color_a, color_b)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(rows, cols, shared_dim, horizontal)
    fg_swatch = _make_swatch(shared_dim, color_a, horizontal)
    bg_swatch = _make_swatch(shared_dim, color_b, horizontal)
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
