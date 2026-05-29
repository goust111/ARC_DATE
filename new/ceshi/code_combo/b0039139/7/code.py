# Combo variant of b0039139 (index 7)
# combo_spec: D1/2+D5/0
# Name: 无分隔交替着色

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   concrete_behavior:
#     - 输入仍由四个区段组成：模板、stencil、前景色板、背景色板；四段布局和蓝色分隔线规则不变。
#     - 输出仍然来自“裁剪模板 + 按 stencil 连通分量重复”，但两个地方同时变化：
#       1) 副本颜色不再固定，而是按副本序号奇偶在两块色板之间来回交换。
#       2) 副本之间不再插入背景色分隔带，而是直接首尾拼接。
#   behavior_trace:
#     - 例子：若裁剪后模板掩码是 3x2，stencil 沿重复轴上有 4 个分量，色板分别是 A 和 B。
#     - 原题输出是 [A/B tile] + B-separator + [A/B tile] + ...，所有 tile 同色，分隔带显式给出边界。
#     - 现在输出改为 [A/B tile][B/A tile][A/B tile][B/A tile]，没有任何空列/空行，边界只能从颜色突变中读出。
#     - 差异不只是“删掉分隔带”或“换色”之一，而是二者叠加后让副本既紧密相连又仍可被人眼分段。
#   complexity_calibration:
#     - 若只按组件数量而不给组件排序，奇偶换色对人类和 solver 都不可见，因此必须把副本序号绑定到重复轴：
#       横向输出按左到右编号，纵向输出按上到下编号。
#     - 若允许模板是实心矩形或边界型谱过弱，去掉分隔带后会显得过于平凡或边界几乎不可见；因此生成器要求
#       模板同时含有前景与背景像素，并要求每条拼接边至少有足够多的位置产生颜色跳变。
#   implicit_assumption_audit:
#     - 原 main() 的 `len(components_mask(...))` 只依赖数量；融合后不够，因为颜色取决于“第几个副本”。
#     - 原 main() 默认存在 1 像素分隔带，`np.hstack`/`np.vstack` 前会插入 separator；融合后这个前提失效，
#       输出尺寸公式也随之改变为 `crop_dim * N`，不能再额外加 `(N-1)`。
#     - 原 main() 只构建一个 tile 并重复复用；融合后必须逐副本决定前景/背景角色，否则 D1/2 会丢失。
#     - 新规则新增了“组件顺序要稳定”的假设；生成器通过离散槽位布局保证各 blob 在主轴上的最小坐标严格递增。
#   placement_legality:
#     - 模板仍放在 1 像素黑边内部，bbox 裁剪合法。
#     - stencil 中每个 blob 仍为 2 格且相邻槽位至少留 1 格黑色间隔，保证 4 连通下不会并成一团。
#     - 输出不再插入分隔带，所以只有 tile 本身参与拼接，不存在额外对象重叠风险。
#   observability_and_cleanliness:
#     - 颜色变化只涉及两块色板，且在同一样例中重复出现 A-B-A-B，规律足够少、足够重复，适合 ARC 归纳。
#     - 删除分隔带后输出更紧凑，但拼接方向上的周期性仍清晰，因为相邻副本在同一模板上做前景/背景互换。
#     - stencil 继续是简单的等色 blob 序列，不叠加额外标记，输入视觉结构保持干净。
#   color_inference:
#     - 人类只需识别两件事：模板裁剪后的形状，以及第 1/2/3/... 个副本在两块色板之间交替交换角色。
#     - 颜色 id 本身不参与隐藏排序；顺序完全由 blob 的空间次序给出，可从样例中直接观察。
#   anti_degeneracy_checks:
#     - 生成器拒绝“裁剪模板全满”样本，避免输出退化成整块纯色条纹。
#     - 生成器检查无分隔拼接边上的颜色跳变比例，避免样本看起来像一整块未分段图案。
#   new_solver_assumptions:
#     - main() 假设重复轴上的组件排序无歧义；生成器保证主坐标槽位严格递增。
#     - main() 假设两块色板颜色不同；生成器继续使用 `random.sample` 保证这一点。


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


def _tile_for_index(tile_mask: np.ndarray, color_a: int, color_b: int, index: int) -> np.ndarray:
    if index % 2 == 1:
        foreground, background = color_a, color_b
    else:
        foreground, background = color_b, color_a
    return np.where(tile_mask, foreground, background).astype(int)


def _build_output(tile_mask: np.ndarray, n_repeats: int, color_a: int, color_b: int, horizontal: bool) -> np.ndarray:
    repeats = max(1, int(n_repeats))
    pieces = [_tile_for_index(tile_mask, color_a, color_b, index) for index in range(1, repeats + 1)]
    if horizontal:
        return np.hstack(pieces)
    return np.vstack(pieces)


def _seam_change_ratio(tile_mask: np.ndarray, horizontal: bool) -> float:
    if horizontal:
        edge_a = tile_mask[:, -1]
        edge_b = tile_mask[:, 0]
    else:
        edge_a = tile_mask[-1, :]
        edge_b = tile_mask[0, :]
    return float(np.mean(edge_a == edge_b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a template and pack alternatingly recolored copies flush against each other.

    Input structure:
    - Blue separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a black template carrying a color-4 shape; section 2 is a stencil made of disjoint non-black blobs.
    - Sections 3 and 4 are solid swatches that provide the only two colors used in the output copies.

    Rule:
    - Crop section 1 to the bounding box of its non-black template pixels.
    - Order the 4-connected components in section 2 along the repetition axis: left-to-right for column-separated layouts, top-to-bottom for row-separated layouts.
    - Emit one cropped copy per ordered component: odd-numbered copies use section 3 on the template pixels and section 4 on the cropped background, while even-numbered copies swap those roles.
    - Concatenate the copies directly with no separator rows or columns.

    Key insight: The stencil now encodes both how many copies to make and which copy parity each placement has, while the missing separator band forces the boundary to be inferred from the alternating palette flips.
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
MIN_SEAM_CHANGE_RATIO = 0.4          # seam visibility between adjacent copies
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


def _valid_template(tile_mask: np.ndarray, horizontal: bool) -> bool:
    if not np.any(~tile_mask):
        return False
    seam_ratio = _seam_change_ratio(tile_mask, horizontal)
    return seam_ratio >= MIN_SEAM_CHANGE_RATIO and not math.isclose(seam_ratio, 1.0)


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

    tile_mask = shape != 0
    if not _valid_template(tile_mask, horizontal):
        return None

    color_a, color_b = random.sample(COLOR_POOL, 2)
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
