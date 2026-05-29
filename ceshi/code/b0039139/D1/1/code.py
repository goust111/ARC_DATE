# Variant of b0039139
# Dimension: D1 — 模板重着色规则 (transform)
# Change: 模板像素(色4)替换为区段3的颜色，模板内背景像素(色0)替换为区段4的颜色，形成双色裁剪图案... → 三色重着色：边缘像素使用第三种颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox


# Design analysis:
# concrete_behavior:
#   - 输入仍然由 3 条蓝色分隔线切成 4 段：模板、计数板、颜色板 A、颜色板 B。
#   - 变体只改模板裁剪后的着色方式。模板段里黄色(4)表示对象，黑色(0)表示背景。
#   - 裁剪到对象 bbox 后，输出 tile 中出现 3 种角色：
#       1. 与黑色背景 4 邻接的对象像素 -> 颜色板 A。
#       2. 四个正交邻居都仍是对象的对象像素 -> 颜色板 B。
#       3. bbox 内原本就是黑色的孔洞/空位 -> 继续保持黑色。
#   - 计数板仍然只编码重复次数，输出方向仍由分隔线方向决定。
# trace_example:
#   - 若裁剪后的模板掩码为
#       1111
#       1011
#       1111
#       1111
#     原题会把 1 全部改成区段3颜色、把 0 改成区段4颜色。
#   - 本变体会把靠外轮廓和靠孔洞的 1 都改成区段3颜色，仅把完全被 1 包围的内部像素改成区段4颜色，中心孔保持 0。
#   - 因此 solver 新增的唯一关键推理是“区分边缘模板像素和内部模板像素”。
# complexity_calibration:
#   - 若模板没有内部像素，则区段4颜色完全失效；若模板 bbox 内没有黑色孔洞，则“背景保持黑色”不可见。
#   - 生成器因此强制模板同时满足：至少 1 个内部对象像素、至少 1 个 bbox 内黑色孔洞、至少 1 个边缘对象像素。
#   - 颜色只保留两块色板，避免额外颜色映射自由度；计数板和布局机制维持原题，复杂度集中在 D1。
# old-code dependency audit:
#   - 原 `tile = np.where(tile_mask, foreground, background)` 假设所有对象像素同色、所有 bbox 背景同色；在变体中该假设失效，必须拆成 edge/interior/background 三类。
#   - 原重复分隔条沿用“背景色板”颜色；新规则里 tile 背景固定为黑色，因此分隔条也必须固定为黑色，否则会引入未被模板定义的新背景色。
#   - `_split_sections`、bbox 裁剪、计数板连通分量统计都不依赖旧的双色着色假设，可以原样保留。
# placement_validity:
#   - 模板仍放在黑色边框中，计数板 blob 的放置规则不变，所以四段切分和连通分量统计仍合法。
#   - 生成器从完整矩形出发挖孔/挖缺口，再验证 4 连通、四边仍被触达、存在内部像素，保证 solver 的边界检测有明确语义。
# visual_observability:
#   - 每个样本都能同时看到：区段3颜色的边缘环、区段4颜色的内部块、以及黑色孔洞/分隔条。
#   - 这样人类可在少量样例中直接观察到“边缘 vs 内部”的颜色分工，而不是退化成原题的简单双色替换。
# assumption_audit_for_new_main:
#   - `main()` 新假设：模板对象是单个黄色连通体，且至少有一些像素的 4 邻域不全为对象。生成器通过连通性和内部像素检查来保证。
#   - `main()` 把越界视为黑色背景，这要求模板 bbox 的外侧确实是背景。原题模板外本就包着 1 像素黑边，生成器保持这一点。


TEMPLATE_COLOR = 4
OUTPUT_BACKGROUND = 0
SEPARATOR_COLOR = 1
STENCIL_COLOR = 3
COLOR_POOL = list(range(2, 10))


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


def _recolor_cropped_template(mask: np.ndarray, edge_color: int, interior_color: int) -> np.ndarray:
    edge_mask, interior_mask = _classify_template_pixels(mask)
    tile = np.zeros(mask.shape, dtype=int)
    tile[edge_mask] = int(edge_color)
    tile[interior_mask] = int(interior_color)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the cropped template into a three-tone motif and repeat it by stencil count.

    Input structure:
    - Blue separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a black template containing yellow shape pixels; section 2 is a black stencil whose non-black components encode the copy count.
    - Sections 3 and 4 are solid-color swatches that supply the edge color and the interior color.

    Rule:
    - Crop section 1 to the bbox of its yellow pixels.
    - Inside that crop, leave black cells black; recolor yellow cells that touch black in the 4-neighborhood as the section-3 color; recolor the remaining yellow cells as the section-4 color.
    - Count the 4-connected components in section 2 and repeat the recolored crop that many times, inserting one black separator row or column between copies.

    Key insight: The stencil still tells you only how many copies to make, but solving the template now requires boundary detection: outer-rim pixels and hole-adjacent pixels use one swatch, while fully surrounded pixels use the other.
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
    tile = _recolor_cropped_template(cropped_mask, edge_color, interior_color)

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


# --------------- generator (c) ---------------

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (4, 6)
HOLE_COUNT_RANGE = (1, 3)
NOTCH_COUNT_RANGE = (0, 2)
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
SHAPE_RETRIES = 200
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
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape_mask = _build_shape_mask()
    if shape_mask is None:
        return None

    edge_color, interior_color = random.sample(COLOR_POOL, 2)
    tile = _recolor_cropped_template(shape_mask, edge_color, interior_color)
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
