# Variant of b0039139
# Dimension: D1 — 模板重着色规则 (transform)
# Change: 模板像素(色4)替换为区段3的颜色，模板内背景像素(色0)替换为区段4的颜色，形成双色裁剪图案... → 根据连通分量索引交替着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D1 variant
# ----------------------------------
# spec:
#   concrete_behavior:
#     - 输入仍然由四个区段组成：模板、计数/排序用 stencil、前景色板、背景色板。
#     - 输出仍然是把模板裁剪后沿同一轴向重复平铺，但每个副本不再共用同一套前景/背景颜色。
#     - stencil 的每个连通分量既贡献“有一个副本”，又通过其沿平铺轴的次序贡献“这是第几个副本”。
#   behavior_trace:
#     - 原逻辑：若 stencil 有 4 个连通分量，则生成 4 个完全同色的副本，全部使用区段3作前景、区段4作背景。
#     - 新逻辑：仍然生成 4 个副本，但第 1/3 个副本使用 (区段3前景, 区段4背景)，第 2/4 个副本改用
#       (区段4前景, 区段3背景)。因此输出呈现 A-B-A-B 的双色交替。
#     - 差异落点只在“单个副本如何着色”这一层；模板裁剪、重复次数、输出方向都保持原任务机制。
#   complexity_calibration:
#     - 如果只按组件数量而不定义组件顺序，奇偶规则对人类不可见，因此这里把顺序绑定到重复轴：
#       横向输出按 stencil 连通分量从左到右编号，纵向输出按从上到下编号。
#     - 生成器会把 blobs 放进沿该轴递增的离散槽位，避免两个 blob 拥有相同主坐标而产生顺序歧义。
#   implicit_assumption_audit:
#     - 原 main() 中 `n_repeats = len(components_mask(...))` 只依赖“数量”，在变体中不够，需要改成
#       “提取组件并排序，再按索引逐个建 tile”。
#     - 原 main() 中 separator 一直使用全局 background；变体里背景随副本奇偶变化，因此 separator 也必须
#       跟随当前副本的背景色，否则会把偶数副本误画成原题样式。
#     - 原 main() 里 `tile = np.where(tile_mask, foreground, background)` 只生成一个 tile；变体里必须在循环
#       中对每个索引重新决定 fg/bg，不能复用单一 tile。
#   placement_legality:
#     - 模板仍带 1 像素黑边，bbox 裁剪后得到合法 crop。
#     - stencil 使用固定槽位放置 2-cell blobs，相邻槽位至少留 1 列/行黑色空隙，保证 4 邻接下组件互不连通。
#     - 色板仍然是纯色矩形，共享轴尺寸与其它区段一致，因此 `_join_sections()` 逻辑无需改变。
#   observability_and_cleanliness:
#     - 交替效果直接出现在输出上，且只涉及两种颜色反复交换，人类可以从 2-3 个样例中归纳奇偶规律。
#     - stencil 的 blobs 沿重复轴排成有序序列，不需要依赖颜色 id、面积大小或隐藏坐标规则做排序。
#     - 输出仍是规则平铺，没有额外叠加元素，视觉上与原题同样整洁。
#   color_inference:
#     - 只有区段3和区段4两种颜色参与重着色，且每个样例中会重复出现多次 A/B 交换，足以支撑规律归纳。
#   new_solver_assumptions:
#     - main() 现在假设“组件顺序 = 沿平铺轴排序后的顺序”；生成器通过槽位布局保证这一点稳定成立。
#     - main() 还假设每个组件至少能被明确地区分为前一个或后一个；生成器保证主坐标严格递增，不出现并列。


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


def _ordered_components(mask: np.ndarray, horizontal: bool) -> list[np.ndarray]:
    comps = components_mask(mask, connectivity=4)
    if horizontal:
        return sorted(comps, key=lambda coords: (int(coords[:, 1].min()), int(coords[:, 0].min())))
    return sorted(comps, key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))


def _colored_tile(tile_mask: np.ndarray, color_a: int, color_b: int, index: int) -> tuple[np.ndarray, int]:
    odd_index = index % 2 == 1
    foreground = color_a if odd_index else color_b
    background = color_b if odd_index else color_a
    tile = np.where(tile_mask, foreground, background).astype(int)
    return tile, background


def _build_output(tile_mask: np.ndarray, n_repeats: int, color_a: int, color_b: int, horizontal: bool) -> np.ndarray:
    pieces: list[np.ndarray] = []
    repeats = max(1, int(n_repeats))

    for index in range(1, repeats + 1):
        tile, background = _colored_tile(tile_mask, color_a, color_b, index)
        pieces.append(tile)
        if index == repeats:
            continue
        if horizontal:
            pieces.append(np.full((tile.shape[0], 1), background, dtype=int))
        else:
            pieces.append(np.full((1, tile.shape[1]), background, dtype=int))

    if horizontal:
        return np.hstack(pieces)
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a template and tile it while flipping the palette on alternating stencil indices.

    Input structure:
    - Blue separator rows or columns divide the input into four ordered sections.
    - Section 1 is a black template containing a color-4 shape; section 2 is a stencil made of disjoint non-black blobs.
    - Sections 3 and 4 are solid swatches that provide the two colors used in the output copies.

    Rule:
    - Crop section 1 to the bounding box of its template pixels.
    - Order the 4-connected components in section 2 along the repetition axis: left-to-right for column-separated layouts, top-to-bottom for row-separated layouts.
    - Emit one cropped copy per ordered component: odd-numbered copies use section 3 for the template pixels and section 4 for the cropped background, while even-numbered copies swap those roles.
    - After each copy except the last, insert a one-cell separator colored with that copy's background color.

    Key insight: The stencil no longer encodes only how many copies to make; its component order determines which copies keep the swatch order and which copies invert it.
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
    ordered = _ordered_components(repetition_pattern != 0, horizontal)
    return _build_output(tile_mask, len(ordered), color_a, color_b, horizontal)


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
GEN_RETRIES = 300                    # outer generate_input retry limit


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


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray:
    min_varying = max(4, n_blobs * 3 + 1)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)
    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    for index in range(n_blobs):
        br, bc = random.choice([(1, 2), (2, 1)])
        if horizontal:
            slot_c = 1 + index * 3
            c = slot_c + random.randint(0, 2 - bc)
            r = random.randint(1, h - br - 1)
        else:
            slot_r = 1 + index * 3
            r = slot_r + random.randint(0, 2 - br)
            c = random.randint(1, w - bc - 1)
        grid[r : r + br, c : c + bc] = STENCIL_COLOR

    return grid


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

    color_a, color_b = random.sample(COLOR_POOL, 2)
    tile_mask = shape != 0
    output = _build_output(tile_mask, n_repeats, color_a, color_b, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
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
