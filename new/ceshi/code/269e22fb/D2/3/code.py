# Variant of 269e22fb
# Dimension: D2 — 源对象：裁剪窗口的形状 (src)
# Change: 矩形裁剪窗口，行数8-16，列数8-12，面积80-156... → 带遮挡的裁剪窗口

from __future__ import annotations

import numpy as np

from arc_utils import colors_present

# Design analysis:
# spec:
#   variant_core:
#     - 输入仍然是同一个 20x20 二值母图案的某个二面体方向的矩形裁剪。
#     - 但裁剪图现在有 3 种颜色：前景色、背景色、遮挡色。
#     - 遮挡色组成单个轴对齐矩形块，盖住裁剪窗口的一部分像素；其下方真实图案可能同时包含前景/背景两类像素。
#   concrete_behavior:
#     - 原题在窗口级别做“完整二值子阵列精确匹配”。
#     - 变体改成“忽略遮挡块上的像素，只用剩余可见像素做带掩码匹配”。
#     - 例：10x10 裁剪里若右下角 3x4 被第三色覆盖，原 solver 会因 `len(colors)==2` 失效；
#       新 solver 需要先猜哪个颜色是遮挡色，再只比较非遮挡位置上的 0/1 图案。
#   complexity_calibration:
#     - 保留原题全部自由度：20x20 固定母图案、8 个二面体方向、矩形窗口尺寸与偏移、前景/背景极性。
#     - 只新增一个连续遮挡矩形，面积约占窗口的 12%-33%，既能肉眼看见，又不会抹掉过多识别信息。
#     - 生成时强制遮挡块同时覆盖两种底层像素，并且两种底层像素在可见区也都保留，使“忽略遮挡”成为必要且可学的规则。
#   assumptions_review:
#     - 原 `assert len(colors) == 2` 不再成立，必须扩展为 3 色并枚举遮挡色/前景色角色分配。
#     - 原 `np.array_equal(window, visible)` 依赖“所有像素都可见”，现在必须改成只在 `visible_mask` 上比较。
#     - 原题默认每个颜色都属于图案本身；现在第三色不参与输出重绘，因此输出只能使用剩余两色。
#     - 新 solver 隐含假设是“存在唯一的 (遮挡色, 前景色, 背景色, 方向, 偏移) 组合”，generator 用拒绝采样来保证。
#   placement_legality:
#     - 窗口尺寸仍严格来自原 spec: rows 8-16, cols 8-12, area 80-156。
#     - 遮挡矩形完全放在窗口内，非空，且不会覆盖整个窗口。
#     - generator 只接受能被新 solver 唯一匹配且输出与预期 full motif 完全一致的样本。
#   visual_observability:
#     - 第三色块是单个大矩形，不依赖颜色 id 排序，也不引入额外标记系统。
#     - 从少量样例中，人类可直接看到：输入比原题多了一块“盖住信息”的遮挡，但输出仍需恢复完整母图案。
#   visual_cleanliness:
#     - 输入只有一个原始裁剪和一个整洁的遮挡层，不存在多重叠画或碎片噪声。
#     - 输出保持原题的干净 20x20 双色图案。


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)

_VALID_SIZES = [
    (h, w)
    for h in range(8, 17)
    for w in range(8, 13)
    if 80 <= h * w <= 156
]
_ARC_COLORS = tuple(range(10))


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_BASE_MASK = _pattern_mask()
_VARIANTS = _dihedral_variants(_BASE_MASK)


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, int, np.ndarray, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 3

    matches: list[tuple[int, int, int, np.ndarray, int, int]] = []
    for occ in colors:
        visible = g != int(occ)
        if not visible.any():
            continue
        remaining = [int(col) for col in colors if int(col) != int(occ)]
        for fg in remaining:
            bg = int(remaining[0] if int(remaining[1]) == int(fg) else remaining[1])
            visible_fg = g == int(fg)
            if not visible_fg[visible].any():
                continue
            if not (g[visible] == int(bg)).any():
                continue
            for pattern in _VARIANTS:
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        window = pattern[r : r + H, c : c + W]
                        if np.array_equal(window[visible], visible_fg[visible]):
                            matches.append((int(fg), int(bg), int(occ), pattern, int(r), int(c)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared 20x20 binary motif from a partially occluded crop.

    Input structure:
    - The grid is a rectangular crop from one rotation/reflection of the same 20x20 motif.
    - Two colors belong to the motif itself (foreground/background), and a third color is an occluder painted over part of that crop.

    Rule:
    - Try each input color as the possible occlusion color and each remaining color as the possible motif foreground.
    - For every dihedral orientation and crop offset, compare only the non-occluded cells against the candidate motif window.
    - Repaint the uniquely matched full 20x20 motif using the recovered foreground/background colors; the occlusion color disappears.

    Key insight:
    - The third color is not new content; it is missing data. The visible subset of the crop still carries enough orientation and offset information to identify one full motif uniquely.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, _occ, pattern, _r, _c = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _sample_colors() -> tuple[int, int, int]:
    fg, bg, occ = np.random.choice(_ARC_COLORS, size=3, replace=False)
    return int(fg), int(bg), int(occ)


def _sample_occluder(H: int, W: int) -> np.ndarray | None:
    area = H * W
    min_occ = max(10, area // 8)
    max_occ = min(area // 3, area - 50)
    if min_occ > max_occ:
        return None

    for _ in range(80):
        oh = int(np.random.randint(2, H))
        ow = int(np.random.randint(2, W))
        occ_area = oh * ow
        if not (min_occ <= occ_area <= max_occ):
            continue
        r0 = int(np.random.randint(0, H - oh + 1))
        c0 = int(np.random.randint(0, W - ow + 1))
        mask = np.zeros((H, W), dtype=bool)
        mask[r0 : r0 + oh, c0 : c0 + ow] = True
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = _VALID_SIZES[int(np.random.randint(len(_VALID_SIZES)))]
    fg, bg, occ = _sample_colors()
    pattern = _VARIANTS[int(np.random.randint(len(_VARIANTS)))]
    out = np.where(pattern, int(fg), int(bg)).astype(int)

    r = int(np.random.randint(0, 20 - H + 1))
    c = int(np.random.randint(0, 20 - W + 1))
    crop_mask = pattern[r : r + H, c : c + W]

    occ_mask = _sample_occluder(H, W)
    if occ_mask is None:
        return None

    visible = ~occ_mask
    if not crop_mask[occ_mask].any():
        return None
    if not (~crop_mask[occ_mask]).any():
        return None
    if not crop_mask[visible].any():
        return None
    if not (~crop_mask[visible]).any():
        return None

    inp = np.where(occ_mask, int(occ), np.where(crop_mask, int(fg), int(bg))).astype(int)
    matches = _find_matches(inp)
    if len(matches) != 1:
        return None

    mfg, mbg, _mocc, matched_pattern, _mr, _mc = matches[0]
    candidate = np.where(matched_pattern, int(mfg), int(mbg)).astype(int)
    if not np.array_equal(candidate, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
