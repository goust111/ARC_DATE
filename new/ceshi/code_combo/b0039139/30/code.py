# Combo variant of b0039139 (index 30)
# combo_spec: D5/1+D8/1
# Name: 双倍分隔带配非对称裁剪

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox, components_mask, random_sprite


"""
Design analysis for combo variant D5/1 + D8/1
==============================================

spec_delta_summary:
  preserved_core:
    - 输入仍由 4 个 section 组成, section 顺序不变: template / stencil / fg_swatch / bg_swatch。
    - 蓝色分区线仍是 1 像素宽; `D5/1` 改的是输出副本之间的背景分隔带, 不是输入分区线。
    - section 2 仍只编码重复次数: 数 section 2 里 4 连通绿色块个数得到 N。
  fused_changes:
    - `D8/1`: template 不再裁成完整 bbox, 只在行方向裁到有形状的最小范围, 列方向保留 section 1 全宽。
    - `D5/1`: 输出平铺时, 相邻副本之间改为 2 像素宽背景带。

behavior_trace:
  concrete_example:
    - 设 template section 为 6x7, 黄色形状只占其中第 1..4 行、第 2..4 列。
    - 原题会裁成 4x3 的 bbox, 若 N=3, 横向输出宽度是 `3*3 + 2 = 11`。
    - 本融合变体只裁行, 因此 tile 变成 4x7; 再用 2 列背景带平铺, 输出宽度变成 `3*7 + 2*2 = 25`。
    - 可见变化不是单点微调: D8/1 让每个副本更宽, D5/1 让副本间空带更厚, 两者共同放大最终尺寸。
  observable_difference:
    - 解题者会看到: output 中每个副本左右都带着背景边距, 且副本之间永远隔着 2 条完整背景列/行。
    - 如果仍按原题“完整 bbox + 1 像素分隔”去做, 输出会系统性偏窄。

complexity_calibration:
  - 保留原题的四段输入与计数机制, 避免再引入新的对象类型, 让主要推理负担集中在两处几何变化。
  - 生成器固定使用 2-6 个 stencil blob, 并保持 swatch 为纯色矩形, 使解题者可稳定识别“颜色来自后两段”。
  - template 的列向保留至少比 bbox 宽 2, 因为两侧黑边一定会进入输出; 这保证 D8/1 每个样本都可见。
  - N 至少为 2, 这样 2 像素分隔带一定出现, 不会退化成看不出 D5/1 的单副本情况。

implicit_assumption_audit:
  - `_split_sections` 的前提是输入里只有真正的分区线会形成整行/整列蓝色。生成器继续禁用蓝色 swatch, 因此该前提保留。
  - 原 `template_bbox = bbox(template_mask)` 仍可用, 但原来的 `crop_bbox(template_mask, template_bbox)` 隐含“行列都裁”的旧假设, 被 D8/1 打破。
  - 原 `components_mask(repetition_pattern != 0, connectivity=4)` 仍成立, 因为 stencil 编码方式不变。
  - 原 `_tile_repeat(..., sep_width=1)` 的默认前提被 D5/1 打破; 现在必须无条件写入 2 像素背景带。
  - 新 solver 额外假设 template 至少有一个非黑像素, 且 section 2 的 blob 彼此不接触。generator 显式构造这些条件。

placement_legality:
  - template 仍是黑底, 黄色 shape 嵌在四周至少 1 格黑边的画布里, 因而行向裁剪永远合法。
  - stencil 继续放置 2 格小块, 并检查其 1 邻域全黑, 保证连通分量数量稳定。
  - swatch 与 template/stencil 共享固定轴尺寸, 用 1 像素蓝线拼接, 不重叠也不越界。
  - 输出采用“tile + 2 像素背景带 + tile ...”的直接拼接, 没有覆盖冲突。

visual_observability_and_tidiness:
  - 版面仍然整洁: 输入分区职责单一, 输出只有前景色/背景色两色, 没有叠层标记。
  - D8/1 的证据是“每个副本都保留完整列宽”; D5/1 的证据是“副本间总有两条纯背景带”。
  - 颜色映射仍由 section 3/4 的纯色板给出, 没有依赖颜色编号大小或其他人类不可见因素。

variant_metric_for_validation:
  - Metric A: `row_crop_width - full_bbox_width >= 2`。这量化了 D8/1 的列向保留量。
  - Metric B: 输出相邻副本之间的纯背景分隔宽度恒为 2。
  - Metric C: 同一输入上, 新规则输出尺寸与原规则输出尺寸必须不同; 否则样本退化。
"""


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = 1

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        sections = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", sections

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    sections = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", sections


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _row_crop_keep_width(template: np.ndarray) -> np.ndarray:
    mask = template != 0
    template_bbox = bbox(mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)
    r0, _c0, r1, _c1 = template_bbox
    return template[int(r0) : int(r1) + 1, :].copy()


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool, sep_width: int = 2) -> np.ndarray:
    if n <= 1:
        return tile.copy()
    if horizontal:
        sep = np.full((tile.shape[0], sep_width), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((sep_width, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a partially cropped template and repeat it according to a component-count stencil.

    Input structure:
    - Solid blue (1) separator rows or columns still divide the input into four sections.
    - Section 1 is a black-background template drawn in yellow; section 2 is the repeat-count stencil.
    - Sections 3 and 4 are solid color swatches that provide the output foreground and background.

    Rule:
    - Find the template's occupied-row span, crop only those rows, and keep the full original column width.
    - Recolor non-black template pixels with the foreground swatch color and all remaining cells with the background swatch color.
    - Count the 4-connected non-black components in section 2, then repeat the recolored tile that many times with a 2-pixel background separator between copies.

    Key insight:
    - The stencil still tells you only how many copies to make, but the tile itself is no longer the tight bbox:
      you must preserve the template's full width while trimming only vertically, then space copies farther apart.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    cropped_template = _row_crop_keep_width(template)
    if cropped_template.size == 0:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile = np.where(cropped_template != 0, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    return _tile_repeat(tile, n_repeats, background, orientation == "horizontal", sep_width=2)


# --------------- generator (combo D5/1 + D8/1) ---------------

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (3, 6)
DENSITY_RANGE = (0.45, 0.8)
COLOR_POOL = [2, 3, 4, 5, 6, 7, 8, 9]
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 5
BLOB_PLACE_RETRIES = 250
SHAPE_RETRIES = 80
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


def _generate_shape(crop_h: int, crop_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        shape_bbox = bbox(shape != 0)
        if shape_bbox is None:
            continue
        r0, c0, r1, c1 = shape_bbox
        bbox_w = int(c1 - c0 + 1)
        if bbox_w >= 1 and bbox_w <= crop_w:
            return np.asarray(shape, dtype=int)
    return None


def _make_template(shape: np.ndarray) -> np.ndarray:
    crop_h, crop_w = map(int, shape.shape)
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape
    return template


def _original_style_tile(shape: np.ndarray, fg_color: int, bg_color: int) -> np.ndarray:
    shape_bbox = bbox(shape != 0)
    assert shape_bbox is not None
    r0, c0, r1, c1 = shape_bbox
    cropped = shape[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return np.where(cropped != 0, fg_color, bg_color).astype(int)


def _variant_tile(template: np.ndarray, fg_color: int, bg_color: int) -> np.ndarray:
    cropped_template = _row_crop_keep_width(template)
    return np.where(cropped_template != 0, fg_color, bg_color).astype(int)


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    shape = _generate_shape(crop_h, crop_w)
    if shape is None:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    template = _make_template(shape)
    tile = _variant_tile(template, fg_color, bg_color)
    legacy_tile = _original_style_tile(shape, fg_color, bg_color)

    width_bonus = int(tile.shape[1] - legacy_tile.shape[1])
    if width_bonus < 2:
        return None

    output = _tile_repeat(tile, n_repeats, bg_color, horizontal, sep_width=2)
    legacy_output = _tile_repeat(legacy_tile, n_repeats, bg_color, horizontal, sep_width=1)
    if output.shape == legacy_output.shape:
        return None

    shared_dim = int(template.shape[0] if horizontal else template.shape[1])
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
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
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
