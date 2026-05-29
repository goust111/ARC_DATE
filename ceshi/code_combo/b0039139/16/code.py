# Combo variant of b0039139 (index 16)
# combo_spec: D1/1+D12/1
# Name: 三色边缘着色配多前景色模板

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox


# Design analysis:
# concrete_behavior:
#   - 输入仍由 3 条蓝色分隔线切成 4 段，顺序保持 template / stencil / edge_swatch / interior_swatch。
#   - 但 template 不再是单色 4 掩码，而是 0/4/5 三值图案；4 和 5 都属于模板前景。
#   - 裁剪到非黑 bbox 后，输出不再看“原颜色映射”，而是看“同色 4 邻域是否完整包围”：
#       1. 若一个 4 或 5 像素的某个正交邻居是黑色/越界/另一种模板色，则它是边缘，着成区段 3 颜色。
#       2. 仅当四个正交邻居都与它同色时，它才是内部，着成区段 4 颜色。
#       3. bbox 内原本为 0 的孔洞保持黑色。
#   - 例：若裁剪模板为
#         44444
#         45554
#         45554
#         45554
#         44404
#     原题会把所有非零都刷成一种前景色；D1 单变体会只把靠黑边/孔洞的像素判成边缘；
#     本融合版还会把 4/5 交界两侧都判成边缘，因此围绕 5 区块会多出一圈边缘色。
# complexity_calibration:
#   - 若模板只是稀疏细线，则几乎所有像素都会落在边缘类，区段 4 颜色失效。
#   - 若 5 区域太小或贴着黑背景，4/5 的存在也不会改变边缘判定，样本会退化成 D1。
#   - 生成器因此强制：存在黑孔洞、存在 3x3 以上的 5 色岛、同时存在至少 1 个“只因 4/5 接壤而新增”的边缘像素，
#     并保留至少 1 个内部像素，使三类输出角色都可见。
# implicit_assumption_audit:
#   - 原 `template_mask = template != 0` 把所有前景压成二值，这会抹掉 4/5 交界信息；融合版必须保留裁剪后的原像素值。
#   - 原 `np.where(tile_mask, foreground, background)` 假设前景统一着色、bbox 背景统一着色；现在要拆成边缘/内部/孔洞三类。
#   - 计数板仍只承担“数连通分量”职责，因此 `components_mask(repetition_pattern != 0, connectivity=4)` 仍成立。
#   - 输出副本之间的分隔条不能用区段 4 颜色，否则会把“内部色”和“真实黑背景孔洞”混在一起；因此仍固定为黑色。
# placement_legality:
#   - template 仍放在 1 像素黑框内，bbox 裁剪稳定，越界按黑色解释也始终合法。
#   - stencil 仍由互不接触的 2 格小块组成，保证组件计数稳定且不会意外合并。
#   - 5 色岛放在模板内部，并与黑孔洞留出缓冲，避免颜色边界把所有内部像素都吃掉。
# visual_observability:
#   - 每个样本都能同时看到：黑孔洞、由外轮廓与孔洞产生的边缘、以及由 4/5 内部分界额外产生的边缘环。
#   - 解题者不需要读取颜色编号，只需观察：template 中两种前景色的接缝在输出里会统一转成边缘色。
#   - stencil 与两块 swatch 维持原题的简洁布局，复杂度集中在模板局部结构，视觉上仍整洁。
# new_main_assumption_audit:
#   - `main()` 新增假设：裁剪后的模板里至少有一些 4 或 5 像素拥有四个同色邻居，否则内部色不会出现。
#   - 还假设存在 4/5 交界，才能体现融合行为；生成器用量化指标 `extra_boundary_edges > 0` 明确保证这一点。
#   - 所有这些假设都由 `_build_multicolor_template()` 先构造、再经 `_measure_fusion_signal()` 验证后才放行。


TEMPLATE_PRIMARY = 4
TEMPLATE_SECONDARY = 5
SEPARATOR_COLOR = 1
STENCIL_COLOR = 3
OUTPUT_BACKGROUND = 0

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (6, 8)
OUTPUT_COLOR_POOL = [2, 3, 6, 7, 8, 9]
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
TEMPLATE_RETRIES = 200
GEN_RETRIES = 300


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == SEPARATOR_COLOR))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEPARATOR_COLOR))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _classify_multicolor_pixels(template_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crop = np.asarray(template_crop, dtype=int)
    padded = np.pad(crop, 1, constant_values=-1)
    center = crop
    same_up = padded[:-2, 1:-1] == center
    same_down = padded[2:, 1:-1] == center
    same_left = padded[1:-1, :-2] == center
    same_right = padded[1:-1, 2:] == center

    foreground = center != 0
    interior = foreground & same_up & same_down & same_left & same_right
    edge = foreground & ~interior
    return edge, interior


def _classify_binary_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    padded = np.pad(mask_bool, 1, constant_values=False)
    fully_surrounded = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    interior = mask_bool & fully_surrounded
    edge = mask_bool & ~interior
    return edge, interior


def _recolor_cropped_template(template_crop: np.ndarray, edge_color: int, interior_color: int) -> np.ndarray:
    edge_mask, interior_mask = _classify_multicolor_pixels(template_crop)
    tile = np.zeros(template_crop.shape, dtype=int)
    tile[edge_mask] = int(edge_color)
    tile[interior_mask] = int(interior_color)
    return tile


def _measure_fusion_signal(template_crop: np.ndarray) -> dict[str, int]:
    edge_mask, interior_mask = _classify_multicolor_pixels(template_crop)
    binary_edge, _ = _classify_binary_pixels(template_crop != 0)
    extra_boundary_edges = int(np.sum(edge_mask & ~binary_edge))
    return {
        "edge_pixels": int(np.sum(edge_mask)),
        "interior_pixels": int(np.sum(interior_mask)),
        "holes": int(np.sum(template_crop == 0)),
        "extra_boundary_edges": extra_boundary_edges,
        "has_primary": int(np.any(template_crop == TEMPLATE_PRIMARY)),
        "has_secondary": int(np.any(template_crop == TEMPLATE_SECONDARY)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a cropped two-foreground template into a three-tone tile and repeat it by stencil count.

    Input structure:
    - Blue separator rows or columns split the grid into four ordered sections.
    - Section 1 is a black-backed template that may contain both 4-pixels and 5-pixels; section 2 is a black stencil whose non-black components encode the copy count.
    - Sections 3 and 4 are solid swatches that provide the output edge color and the output interior color.

    Rule:
    - Crop section 1 to the bbox of its non-black content.
    - Within that crop, leave 0-cells black. For each 4/5 template pixel, test its 4-neighbors: if any neighbor is black, outside the crop, or the other template color, paint it with the section-3 color; otherwise paint it with the section-4 color.
    - Count 4-connected non-black components in section 2, then repeat the recolored crop that many times with one black separator row or column between copies.

    Key insight: the second template color does not map to its own output color; instead it changes which pixels qualify as edges, because 4-to-5 boundaries behave like miniature internal outlines that also take the edge swatch.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, edge_block, interior_block = sections[:4]

    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    edge_color = _solid_color(edge_block)
    interior_color = _solid_color(interior_block)
    template_crop = crop_bbox(template, template_bbox)
    tile = _recolor_cropped_template(template_crop, edge_color, interior_color)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), OUTPUT_BACKGROUND, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), OUTPUT_BACKGROUND, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


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
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
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


def _choose_island_bounds(crop_h: int, crop_w: int) -> tuple[int, int, int, int]:
    island_h = random.randint(3, crop_h - 2)
    island_w = random.randint(3, crop_w - 2)
    top = random.randint(1, crop_h - island_h - 1)
    left = random.randint(1, crop_w - island_w - 1)
    return top, left, island_h, island_w


def _build_multicolor_template() -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        crop_h = random.randint(*CROP_SIZE_RANGE)
        crop_w = random.randint(*CROP_SIZE_RANGE)
        shape = np.full((crop_h, crop_w), TEMPLATE_PRIMARY, dtype=int)

        top, left, island_h, island_w = _choose_island_bounds(crop_h, crop_w)
        shape[top : top + island_h, left : left + island_w] = TEMPLATE_SECONDARY

        forbidden = np.zeros_like(shape, dtype=bool)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(crop_h, top + island_h + 1)
        c1 = min(crop_w, left + island_w + 1)
        forbidden[r0:r1, c0:c1] = True

        hole_candidates = [
            (r, c)
            for r in range(1, crop_h - 1)
            for c in range(1, crop_w - 1)
            if shape[r, c] == TEMPLATE_PRIMARY and not forbidden[r, c]
        ]
        if not hole_candidates:
            continue

        hole_count = random.randint(1, min(2, len(hole_candidates)))
        for r, c in random.sample(hole_candidates, hole_count):
            shape[r, c] = 0

        metrics = _measure_fusion_signal(shape)
        if metrics["interior_pixels"] == 0:
            continue
        if metrics["holes"] == 0:
            continue
        if metrics["extra_boundary_edges"] == 0:
            continue
        return shape
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    template_crop = _build_multicolor_template()
    if template_crop is None:
        return None

    edge_color, interior_color = random.sample(OUTPUT_COLOR_POOL, 2)
    tile = _recolor_cropped_template(template_crop, edge_color, interior_color)
    output = _tile_repeat(tile, n_repeats, OUTPUT_BACKGROUND, horizontal)

    crop_h, crop_w = template_crop.shape
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = template_crop

    shared_dim = template.shape[0] if horizontal else template.shape[1]
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    edge_swatch = _make_swatch(shared_dim, edge_color, horizontal)
    interior_swatch = _make_swatch(shared_dim, interior_color, horizontal)
    inp = _join_sections([template, stencil, edge_swatch, interior_swatch], shared_dim, horizontal)

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
