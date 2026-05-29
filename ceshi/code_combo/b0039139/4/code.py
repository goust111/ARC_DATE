# Combo variant of b0039139 (index 4)
# combo_spec: D1/1+D2/2
# Name: 旋转模板的三色边缘着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox


# Design analysis:
# concrete_behavior:
#   - 输入仍然被 3 条整行/整列蓝线切成 4 段，顺序不变：模板、计数板、边缘色板、内部色板。
#   - 模板段仍是黑底黄形状，但 solver 先裁剪到黄像素 bbox，再顺时针旋转 90 度；后续所有着色都基于“旋转后的裁剪图形”。
#   - 三色输出 tile 的角色固定为：
#       1. 旋转后图形中与黑色 4 邻接的黄像素 -> 区段 3 颜色。
#       2. 旋转后图形中四个正交邻居也都是黄像素的内部像素 -> 区段 4 颜色。
#       3. 旋转后裁剪框中的黑色孔洞/空位 -> 保持黑色。
#   - 计数板仍只编码重复次数；输出沿分隔线方向平铺，副本间分隔条也保持黑色，因为黑色现在是 tile 的唯一背景。
# trace_example:
#   - 设原模板裁剪掩码为
#       1111
#       1111
#       1011
#       1110
#     原题会直接把 1 统一改成前景色、把 0 改成背景色。
#   - 本融合版先旋转成
#       1111
#       1101
#       1111
#       0111
#     再做边缘检测：靠外轮廓和靠黑孔的 1 变成区段 3 颜色，只有被 1 四向包围的少数像素才变成区段 4 颜色，0 继续是黑色。
#   - 因而变化不是“先三色后旋转的随意组合”，而是“输出中的边缘色分布来自旋转后的几何布局”。
# complexity_calibration:
#   - 若模板没有内部像素，区段 4 会失效；若 bbox 内没有黑孔，黑背景保留也不明显；若旋转后三色 tile 与未旋转三色 tile 相同，则 D2 失效。
#   - 生成器因此强制每个模板同时满足：存在边缘像素、存在内部像素、存在 bbox 内内部黑孔、且旋转前后的三色标签图不同。
#   - 颜色只使用两块色板并固定黑背景，避免引入额外映射自由度；重复次数和布局保持原题，让复杂度集中在“旋转 + 边界分类”这条链上。
# old_code_dependency_audit:
#   - 原 `tile = np.where(tile_mask, foreground, background)` 假设对象/背景只有两类颜色；融合后必须拆成 edge/interior/black 三类。
#   - 原 `tile_mask = crop_bbox(...)` 默认裁剪后的朝向就是最终朝向；现在必须先旋转，再进行边界分类与着色。
#   - 原分隔条颜色来自背景色板；这里背景不再由区段 4 提供，而是固定黑色，所以平铺分隔条也必须固定为 0。
#   - `_split_sections`、bbox 裁剪、计数板连通分量统计本身不依赖上述旧假设，可直接复用。
# placement_validity:
#   - 模板依旧嵌在 1 像素黑边框内，旋转只发生在 solver 内部，因此输入放置规则与四段拼接规则不变。
#   - 计数板仍放置互不接触的 2 格 blob；色板仍是与共享轴对齐的纯色矩形，所以不会越界或与分隔线混淆。
#   - 模板生成从满矩形出发挖内孔和少量边界缺口，再检查 4 连通与四边触达，保证裁剪结果稳定且 bbox 语义明确。
# visual_observability:
#   - 每个样本都会同时展示三种视觉角色：边缘色环、内部色块、黑孔/黑分隔条。这样 D1 的新规则可以直接从输出中观察到。
#   - 同时生成器拒绝“旋转前后三色 tile 完全一样”的模板，确保 D2 也能被 2-4 个样例稳定看出，而不是被对称图形掩盖。
# cleanliness_and_inference:
#   - 色板颜色池排除了 1/3/4：蓝线只负责分段，绿色只在计数板出现，黄色只在模板里出现，减少跨区域颜色复用造成的视觉噪声。
#   - 黑色只承担模板孔洞、tile 背景和副本分隔条这一个结构角色，人类可从少量样例归纳“黑色背景保留不染色”。
# assumption_audit_for_new_main:
#   - `main()` 新增假设 1：模板里确有黄色像素可裁剪；生成器始终构造非空单连通模板。
#   - 新增假设 2：旋转后的裁剪图形仍然需要区分边缘/内部；生成器强制两类像素都存在。
#   - 新增假设 3：bbox 外侧与孔洞都应视为黑背景；模板外围保留黑边，且 bbox 内至少有一个内部黑孔来支撑这个判定。
#   - 新增假设 4：组合题不能退化为 D1-only；生成器通过“旋转前后三色标签图不同”保证旋转真的影响最终可见图案。


TEMPLATE_COLOR = 4
OUTPUT_BACKGROUND = 0
SEPARATOR_COLOR = 1
STENCIL_COLOR = 3
COLOR_POOL = [2, 5, 6, 7, 8, 9]

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (4, 6)
HOLE_COUNT_RANGE = (1, 3)
NOTCH_COUNT_RANGE = (0, 2)
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
SHAPE_RETRIES = 240
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


def _rotate_clockwise(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=3)


def _classify_template_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _recolor_cropped_template(mask: np.ndarray, edge_color: int, interior_color: int) -> np.ndarray:
    edge_mask, interior_mask = _classify_template_pixels(mask)
    tile = np.zeros(mask.shape, dtype=int)
    tile[edge_mask] = int(edge_color)
    tile[interior_mask] = int(interior_color)
    return tile


def _three_tone_labels(mask: np.ndarray) -> np.ndarray:
    edge_mask, interior_mask = _classify_template_pixels(mask)
    labels = np.zeros(mask.shape, dtype=int)
    labels[edge_mask] = 1
    labels[interior_mask] = 2
    return labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a cropped template, split it into edge and interior pixels, and repeat the result by stencil count.

    Input structure:
    - Blue separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a black template containing yellow shape pixels; section 2 is a black stencil whose non-black components encode the copy count.
    - Sections 3 and 4 are solid-color swatches that provide the edge color and the interior color.

    Rule:
    - Crop section 1 to the bbox of its yellow pixels, rotate that crop 90 degrees clockwise, and keep black cells black.
    - In the rotated crop, recolor yellow pixels that touch black in the 4-neighborhood with the section-3 color, and recolor the remaining yellow pixels with the section-4 color.
    - Count the 4-connected components in section 2 and repeat the recolored tile that many times, inserting one black separator row or column between copies.

    Key insight: The stencil still contributes only the repeat count and the separator layout still sets the tiling axis, but the visible motif comes from a rotated template whose boundary pixels and fully surrounded pixels must be distinguished after that rotation.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, edge_block, interior_block = sections[:4]

    template_mask = template == TEMPLATE_COLOR
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    edge_color = _solid_color(edge_block)
    interior_color = _solid_color(interior_block)
    cropped_mask = crop_bbox(template_mask, template_bbox)
    rotated_mask = _rotate_clockwise(cropped_mask)
    tile = _recolor_cropped_template(rotated_mask, edge_color, interior_color)

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


def _rotation_changes_fused_tile(mask: np.ndarray) -> bool:
    return not np.array_equal(_three_tone_labels(mask), _three_tone_labels(_rotate_clockwise(mask)))


def _build_shape_mask() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        crop_h = random.randint(*CROP_SIZE_RANGE)
        crop_w = random.randint(*CROP_SIZE_RANGE)
        mask = np.ones((crop_h, crop_w), dtype=bool)

        interior_cells = [(r, c) for r in range(1, crop_h - 1) for c in range(1, crop_w - 1)]
        boundary_cells = (
            [(0, c) for c in range(1, crop_w - 1)]
            + [(crop_h - 1, c) for c in range(1, crop_w - 1)]
            + [(r, 0) for r in range(1, crop_h - 1)]
            + [(r, crop_w - 1) for r in range(1, crop_h - 1)]
        )

        max_holes = min(HOLE_COUNT_RANGE[1], len(interior_cells))
        min_holes = min(HOLE_COUNT_RANGE[0], max_holes)
        hole_count = random.randint(min_holes, max_holes)
        for r, c in random.sample(interior_cells, hole_count):
            mask[r, c] = False

        if boundary_cells:
            max_notches = min(NOTCH_COUNT_RANGE[1], len(boundary_cells))
            notch_count = random.randint(NOTCH_COUNT_RANGE[0], max_notches)
            for r, c in random.sample(boundary_cells, notch_count):
                mask[r, c] = False

        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not (~mask[1:-1, 1:-1]).any():
            continue

        edge_mask, interior_mask = _classify_template_pixels(mask)
        if not edge_mask.any() or not interior_mask.any():
            continue
        if not _rotation_changes_fused_tile(mask):
            continue
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape_mask = _build_shape_mask()
    if shape_mask is None:
        return None

    edge_color, interior_color = random.sample(COLOR_POOL, 2)
    rotated_mask = _rotate_clockwise(shape_mask)
    tile = _recolor_cropped_template(rotated_mask, edge_color, interior_color)
    output = _tile_repeat(tile, n_repeats, OUTPUT_BACKGROUND, horizontal)

    crop_h, crop_w = shape_mask.shape
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape_mask.astype(int) * TEMPLATE_COLOR

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
