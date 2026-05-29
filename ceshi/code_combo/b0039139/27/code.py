# Combo variant of b0039139 (index 27)
# combo_spec: D1/1+D2/3
# Name: 水平镜像模板配三色边缘着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox


# Design analysis:
# concrete_behavior:
#   - 输入仍然由 3 条整行/整列蓝色分隔线切成 4 段，顺序不变：模板、计数板、颜色板 A、颜色板 B。
#   - 融合后的关键顺序是：
#       1. 在区段1中取黄色模板的 bbox。
#       2. 对 bbox 内布尔模板先做左右镜像。
#       3. 再在镜像后的模板上区分边缘模板像素和内部模板像素。
#       4. 边缘像素 -> 区段3颜色，内部像素 -> 区段4颜色，bbox 内黑色空位保持黑色。
#       5. 区段2 的 4 连通分量个数仍然给出重复次数；输出方向仍由分隔线方向决定。
#   - 小例子：原 crop mask 若为
#       11110
#       10110
#       11111
#       01111
#     原题会直接着色成前景/背景双色；
#     D1-only 会在这个朝向上把边缘着成色板A、内部着成色板B；
#     本融合先镜像成
#       01111
#       01101
#       11111
#       11110
#     然后才做边缘检测，所以输出既换了左右朝向，又以镜像后的邻接关系决定哪一些格子属于边缘。
# complexity_calibration:
#   - 若模板水平对称，则“先镜像”不可观察，任务会退化为 D1-only。
#   - 若模板没有内部像素，则区段4颜色不起作用，任务会退化为“镜像 + 单色边缘”。
#   - 若 bbox 内没有黑色空位，则“背景保持黑色”不明显，容易被误解成普通双色替换。
#   - 生成器因此强制模板同时满足：
#       1. 至少一个内部模板像素；
#       2. 至少一个 bbox 内黑色空位；
#       3. 至少一个边缘模板像素；
#       4. 水平镜像后图案与未镜像图案不同；
#       5. 融合 tile 与 D1-only tile、D2-only tile 都不同。
# implicit_assumption_audit:
#   - `_split_sections` 只依赖完整蓝线分段；融合不改变这一点。
#   - 原题 `tile = where(mask, fg, bg)` 假设“裁剪后模板就是最终几何，且对象/背景各只有一种颜色”；
#     融合后该假设同时失效，必须改成“镜像后的 mask -> edge/interior/background 三分类”。
#   - 边缘检测把越界和 bbox 内黑格都视作背景，因此模板外仍需有黑边；生成器保留原题 1 格黑框。
#   - 输出拼接分隔条不能再使用区段4颜色。区段4现在表示内部颜色，而输出背景固定为黑色，故分隔条也固定黑色。
# placement_validity:
#   - 模板仍嵌在 `(crop_h + 2) x (crop_w + 2)` 的黑底中，不越界且裁剪唯一。
#   - 计数板仍使用互相至少隔 1 格黑色的 2 格 blob，连通分量数量稳定可数。
#   - 色板仍为纯色矩形，共享轴长度与其余区段一致，因此拼接布局合法。
# visual_observability:
#   - 每个样例都同时展示三类可见线索：左右翻转后的几何轮廓、边缘环颜色、内部块颜色，以及黑色孔洞/缺口。
#   - 人类从 2-4 组样例可以对照看到：模板并非直接复制，而是先镜像；镜像后的轮廓最外圈和孔洞边界使用同一色板。
# visual_cleanliness:
#   - 所有逻辑都发生在裁剪后的单个矩形 tile 内，不叠加额外标记，不引入多重编码。
#   - 输出仍是规则的矩形平铺，中间只用 1 格黑条分隔，视觉结构与原题一致，变化集中在 tile 内部。
# color_inference:
#   - 仍只使用两块色板，但它们现在分别绑定到“边缘模板像素”和“内部模板像素”两个结构角色。
#   - 黑色背景来自模板 bbox 内空位与复制间隔，可作为稳定锚点，帮助解题者推断不是任意三色映射。
# assumption_audit_for_new_main:
#   - `main()` 新增假设：镜像后的模板上存在可判定的 edge/interior 分层。生成器通过内部像素检查保证这一点。
#   - `main()` 还默认镜像确实会改变图案朝向。生成器显式过滤掉水平对称模板，避免退化样本。
#   - `main()` 按黄色像素提模板，默认模板段里没有其他非黑干扰色；生成器只在模板段放 0/4。


TEMPLATE_COLOR = 4
OUTPUT_BACKGROUND = 0
SEPARATOR_COLOR = 1
STENCIL_COLOR = 3
COLOR_POOL = list(range(2, 10))

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


def _recolor_three_tone(mask: np.ndarray, edge_color: int, interior_color: int) -> np.ndarray:
    edge_mask, interior_mask = _classify_template_pixels(mask)
    tile = np.zeros(mask.shape, dtype=int)
    tile[edge_mask] = int(edge_color)
    tile[interior_mask] = int(interior_color)
    return tile


def _mirror_then_recolor(mask: np.ndarray, edge_color: int, interior_color: int) -> np.ndarray:
    mirrored_mask = np.fliplr(np.asarray(mask, dtype=bool))
    return _recolor_three_tone(mirrored_mask, edge_color, interior_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the cropped template, split it into edge and interior zones, and repeat it by stencil count.

    Input structure:
    - Blue separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a black template with yellow shape pixels; section 2 is a black stencil whose non-black components encode the repeat count.
    - Sections 3 and 4 are solid swatches that provide the edge color and the interior color for the mirrored template.

    Rule:
    - Crop section 1 to the bbox of its yellow pixels, flip that crop left-right, then classify the mirrored yellow pixels by 4-neighborhood.
    - In the mirrored crop, yellow pixels touching black are recolored with the section-3 swatch, fully surrounded yellow pixels are recolored with the section-4 swatch, and black cells stay black.
    - Count the 4-connected components in section 2 and repeat that three-tone mirrored tile along the separator axis, inserting one black separator row or column between copies.

    Key insight: The stencil still contributes only the copy count, but the template logic is now a two-step composition: reflect first, then detect boundaries on the reflected shape, so both geometry and color placement come from the mirrored crop.
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
    tile = _mirror_then_recolor(cropped_mask, edge_color, interior_color)

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


def _d2_only_tile(mask: np.ndarray, edge_color: int) -> np.ndarray:
    return np.where(np.fliplr(mask), edge_color, OUTPUT_BACKGROUND).astype(int)


def _is_visible_combo_mask(mask: np.ndarray) -> bool:
    edge_mask, interior_mask = _classify_template_pixels(mask)
    if not edge_mask.any() or not interior_mask.any():
        return False
    if not (~mask).any():
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False

    reference_edge = 2
    reference_interior = 8
    original_three_tone = _recolor_three_tone(mask, reference_edge, reference_interior)
    combo_three_tone = _mirror_then_recolor(mask, reference_edge, reference_interior)
    if np.array_equal(combo_three_tone, original_three_tone):
        return False

    reflected_binary = _d2_only_tile(mask, reference_edge)
    if np.array_equal(combo_three_tone, reflected_binary):
        return False
    return True


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
        if not _is_visible_combo_mask(mask):
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
    tile = _mirror_then_recolor(shape_mask, edge_color, interior_color)
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
