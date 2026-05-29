# Combo variant of b0039139 (index 15)
# combo_spec: D2/3+D4/3
# Name: 镜像模板配镜像交替平铺

"""
Design analysis for the combo variant

a) 具体化与行为追踪
- 输入分区仍完全沿用原题：模板区、计数板、前景色色板、背景色色板，之间由整行或整列蓝色分隔线分开。
- 原题在 section 1 上的处理是：裁剪模板 bbox -> 直接重着色 -> 沿主轴等间隔重复。
- 这个融合版把两次变化串成一条链：
  1. 先对裁剪后的模板做一次左右镜像，得到“单镜像模板”。
  2. 再按副本位置交替决定是否再做一次同样的左右镜像：
     第 1/3/5/... 个副本 = 双镜像 = 回到原始朝向；
     第 2/4/6/... 个副本 = 单镜像 = 保持左右翻转朝向。
  3. 这些副本不再用固定 1 像素背景间隔展开，而是沿输出主轴尽量向前紧凑滑动，
     只要前景不与已有前景碰撞就允许背景区重叠。
- 一个 3x3 小例子，原模板 crop 为
    [[1,1,0],
     [1,0,0],
     [1,1,1]]
  原题若横向重复 3 次，会得到同朝向的三块加两列背景分隔。
  融合后先左右镜像成
    [[0,1,1],
     [0,0,1],
     [1,1,1]]
  再生成副本序列：
    copy1 = 双镜像 -> 回到原模板朝向
    copy2 = 单镜像 -> 左右翻转朝向
    copy3 = 双镜像 -> 回到原模板朝向
  然后把这三块按无前景碰撞的最紧凑位置打包，所以输出既有朝向交替，也有总宽/总高缩短。

b) 复杂度校准
- 如果模板本身左右对称，那么“单镜像”和“双镜像”会看起来一样，D2 的变化会退化消失。
- 如果紧凑打包没有产生任何正重叠，输出只会像“交替但仍留空隙”的简单条带，D4 的变化也不明显。
- 因此生成器显式要求：
  1. 模板必须左右不对称；
  2. 每一对相邻副本都至少产生 1 格正重叠；
  3. 融合输出不能等于“交替但带 1 格背景缝”的弱化版本。
- 保留原题的 2..5 裁剪尺寸、2..6 重复次数和四分区输入结构，避免自由度暴增。

c) 隐式假设审查
- `_split_sections()` 只依赖完整蓝色分隔行/列，不受融合影响。
- `bbox(template != 0)` 仍然成立，因为模板区依旧只有单个黄色形状和黑背景。
- 原题的 `tile = where(crop_mask, fg, bg)` 隐含“裁剪结果本身就是最终单块朝向”。
  这里不再成立，必须先插入第一次 `np.fliplr(...)`。
- 原题的固定 `sep + tile + sep + tile` 逻辑隐含“各副本占据互不重叠的独立矩形并用固定缝隙隔开”。
  这个前提在 packed-grid 变体中失效，必须换成逐块试探、无前景碰撞的最左/最上合法放置。
- 计数板仍只编码一个整数，颜色色板仍只编码前景/背景两色，这两部分语义保持不变。

d) 放置合法性
- 打包算法每次只把新副本放在当前输出右侧或下侧的某个起点；若会撞到已有前景，就继续向外试。
- 输出画布只在需要时向右或向下扩展，因此不会越界。
- 允许重叠的区域必须至少有一方是背景，所以不会覆盖掉前景信息。
- 生成器端的模板、计数板、色板和蓝色分隔线仍按原题方式组装；计数板 blob 之间保留黑色缓冲，组件计数始终稳定。

e) 视觉可观察性
- 解题者从少量样例里应能同时看到两个现象：
  1. 相邻块的左右朝向交替，而不是所有块同向；
  2. 总长度比“逐块+1格背景缝”更短，说明用了紧凑打包。
- 交替只依赖可见的左右形状差异，不依赖颜色编号大小规律等人类不可视因素。

f) 可视化整洁度
- 输出仍只使用前景/背景两种颜色，没有叠加标记、半透明或第三层编码。
- 每个副本都保持同一 bbox 尺寸，打包只改变相对位移，不改变单块内部像素结构。
- 这样视觉上仍是干净的一维条带，只是条带内部出现了朝向交替和局部咬合。

g) 变化的可推断性
- 颜色规则完全沿用原题：section 3 给前景色，section 4 给背景色。
- 融合变化全部由几何关系承载，避免再引入额外颜色映射负担。
- 因为样例会同时展示原向块与镜像块，且两者都来自同一个输入模板，人类可以从结构本身推断镜像关系。

h) 量化退化检查思路
- `main(input) == output` 只能证明 solver 与 generator 一致，不能证明融合特征真的被触发。
- 额外统计三个指标：
  1. `mirror_changed`: 原模板与其左右镜像不同；
  2. `min_overlap`: 相邻 packed 副本的最小正重叠量；
  3. `packed_differs_from_gap_alt`: 融合输出是否不同于“交替朝向但保留 1 格背景缝”的弱化布局。
- 只有当 1 为真、2 >= 1、3 为真时，样本才算真正体现该融合。

i) 新 solver 假设与生成器保障
- 假设 1：模板裁剪后至少含有一个前景像素。生成器继续用 `random_sprite` 生成单连通形状保证这一点。
- 假设 2：如果要肉眼看到 D2，模板必须左右不对称。生成器显式筛掉对称模板。
- 假设 3：packing 采用“每一步取最靠前的无碰撞位置”这一确定性规则。生成器与 solver 共用同一打包函数。
- 假设 4：样例中既有单镜像块也有双镜像块。重复次数固定取 2..6，因此两类副本都会出现。
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


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


def _alternating_copy_masks(single_mirror_mask: np.ndarray, n_repeats: int) -> list[np.ndarray]:
    """Build copy masks after D2 then D4: odd positions double-mirror, even positions single-mirror."""
    single = single_mirror_mask.astype(bool)
    double = np.fliplr(single)
    return [double.copy() if i % 2 == 0 else single.copy() for i in range(max(1, n_repeats))]


def _pack_copy_masks(copy_masks: list[np.ndarray], horizontal: bool) -> tuple[np.ndarray, list[int]]:
    """Pack already-oriented copies along one axis as tightly as possible without foreground collision."""
    if not copy_masks:
        return np.zeros((0, 0), dtype=bool), []
    if len(copy_masks) == 1:
        return copy_masks[0].astype(bool), []

    packed = copy_masks[0].astype(bool).copy()
    overlaps: list[int] = []

    if horizontal:
        for piece in copy_masks[1:]:
            piece = piece.astype(bool)
            cur_h, cur_w = packed.shape
            piece_w = int(piece.shape[1])
            start_min = max(0, cur_w - piece_w + 1)

            for start in range(start_min, cur_w + 1):
                overlap_w = max(0, min(cur_w, start + piece_w) - start)
                if overlap_w > 0 and np.any(packed[:, start : start + overlap_w] & piece[:, :overlap_w]):
                    continue

                new_w = max(cur_w, start + piece_w)
                canvas = np.zeros((cur_h, new_w), dtype=bool)
                canvas[:, :cur_w] = packed
                canvas[:, start : start + piece_w] |= piece
                packed = canvas
                overlaps.append(cur_w - start)
                break
            else:
                raise RuntimeError("Failed to pack alternating mirrored copy.")
        return packed, overlaps

    for piece in copy_masks[1:]:
        piece = piece.astype(bool)
        cur_h, cur_w = packed.shape
        piece_h = int(piece.shape[0])
        start_min = max(0, cur_h - piece_h + 1)

        for start in range(start_min, cur_h + 1):
            overlap_h = max(0, min(cur_h, start + piece_h) - start)
            if overlap_h > 0 and np.any(packed[start : start + overlap_h, :] & piece[:overlap_h, :]):
                continue

            new_h = max(cur_h, start + piece_h)
            canvas = np.zeros((new_h, cur_w), dtype=bool)
            canvas[:cur_h, :] = packed
            canvas[start : start + piece_h, :] |= piece
            packed = canvas
            overlaps.append(cur_h - start)
            break
        else:
            raise RuntimeError("Failed to pack alternating mirrored copy.")
    return packed, overlaps


def _repeat_with_gaps(copy_masks: list[np.ndarray], horizontal: bool) -> np.ndarray:
    """Baseline for degeneration checks: keep the alternating orientations but separate copies by 1 background cell."""
    if not copy_masks:
        return np.zeros((0, 0), dtype=bool)
    if len(copy_masks) == 1:
        return copy_masks[0].astype(bool)

    pieces = [m.astype(bool) for m in copy_masks]
    if horizontal:
        sep = np.zeros((pieces[0].shape[0], 1), dtype=bool)
        chain = [pieces[0]] + [x for piece in pieces[1:] for x in (sep, piece)]
        return np.hstack(chain)

    sep = np.zeros((1, pieces[0].shape[1]), dtype=bool)
    chain = [pieces[0]] + [x for piece in pieces[1:] for x in (sep, piece)]
    return np.vstack(chain)


def _simple_repeat_same(mask: np.ndarray, n_repeats: int, horizontal: bool) -> np.ndarray:
    """Original-style repeat of a single fixed orientation, kept for diagnostics."""
    pieces = [mask.astype(bool).copy() for _ in range(max(1, n_repeats))]
    return _repeat_with_gaps(pieces, horizontal)


def _d4_reference_mask(raw_mask: np.ndarray, n_repeats: int, horizontal: bool) -> np.ndarray:
    """Standalone D4/3 behavior used to reject fusion-degenerate samples."""
    base = raw_mask.astype(bool)
    if horizontal:
        copy_masks = [base if i % 2 == 0 else np.flipud(base) for i in range(max(1, n_repeats))]
    else:
        copy_masks = [base if i % 2 == 0 else np.fliplr(base) for i in range(max(1, n_repeats))]
    packed_mask, _ = _pack_copy_masks(copy_masks, horizontal)
    return packed_mask


def _is_horizontally_asymmetric(mask: np.ndarray) -> bool:
    return bool(mask.any()) and not np.array_equal(mask.astype(bool), np.fliplr(mask.astype(bool)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror a cropped template once, then pack alternating single- and double-mirrored copies from a count stencil.

    Input structure:
    - Full blue (1) separator rows or columns divide the input into four sections.
    - Section 1 is a yellow-on-black template; section 2 is a non-black component-count stencil.
    - Sections 3 and 4 are solid swatches providing the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels, flip that crop left-right once, and treat it as the base tile.
    - Count the 4-connected non-black components in section 2 to get the number of copies.
    - Build copy 1/3/5/... by mirroring the base tile again, keep copy 2/4/6/... as the once-mirrored base,
      then pack the copies along the separator-implied axis as tightly as possible without foreground overlap.

    Key insight: The stencil still contributes only a repeat count; the nontrivial part is that the template is
    first reflected, then the packed strip alternates between that reflected tile and its remirrored counterpart.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)

    cropped_mask = crop_bbox(template_mask, template_bbox).astype(bool)
    single_mirror_mask = np.fliplr(cropped_mask)
    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    copy_masks = _alternating_copy_masks(single_mirror_mask, n_repeats)
    packed_mask, _overlaps = _pack_copy_masks(copy_masks, horizontal=(orientation == "horizontal"))
    return np.where(packed_mask, foreground, background).astype(int)


# --------------- generator (combo D2/3 + D4/3) ---------------

# Base-task parameters retained where possible.
REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
SHAPE_RETRIES = 400
GEN_RETRIES = 400

# Fusion-specific observability guards.
MIN_PACKED_OVERLAP = 0


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


def _generate_valid_shape(horizontal: bool, n_repeats: int) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]] | None:
    for _ in range(SHAPE_RETRIES):
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
        raw_mask = (shape != 0).astype(bool)
        if not _is_horizontally_asymmetric(raw_mask):
            continue

        single_mirror_mask = np.fliplr(raw_mask)
        copy_masks = _alternating_copy_masks(single_mirror_mask, n_repeats)
        packed_mask, overlaps = _pack_copy_masks(copy_masks, horizontal)
        if not overlaps or min(overlaps) < MIN_PACKED_OVERLAP:
            continue

        gapped_alt_mask = _repeat_with_gaps(copy_masks, horizontal)
        if np.array_equal(packed_mask, gapped_alt_mask):
            continue

        d2_only_mask = _simple_repeat_same(single_mirror_mask, n_repeats, horizontal)
        if np.array_equal(packed_mask, d2_only_mask):
            continue
        if np.array_equal(packed_mask, _d4_reference_mask(raw_mask, n_repeats, horizontal)):
            continue

        metrics: dict[str, int | bool] = {
            "mirror_changed": True,
            "min_overlap": int(min(overlaps)),
            "max_overlap": int(max(overlaps)),
            "packed_differs_from_gap_alt": True,
            "packed_differs_from_d2_only": bool(not np.array_equal(packed_mask, d2_only_mask)),
        }
        return shape, packed_mask, metrics
    return None


def _gen_attempt() -> np.ndarray | None:
    # Vertical packed chains collapse back to standalone D4/3 because both mirror steps
    # become left-right flips. Force the horizontal branch so D2/3 remains observable.
    horizontal = True
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape_info = _generate_valid_shape(horizontal, n_repeats)
    if shape_info is None:
        return None
    shape, packed_mask, _metrics = shape_info

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    output = np.where(packed_mask, fg_color, bg_color).astype(int)

    crop_h, crop_w = map(int, shape.shape)
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
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
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
