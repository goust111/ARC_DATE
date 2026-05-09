# Variant of 269e22fb
# Dimension: D14 — 符号编码：颜色与图案的映射方式 (symbol_instructions)
# Change: 二值图案直接映射为两种颜色，无额外编码层... → 裁剪窗口边框颜色编码旋转角度

from __future__ import annotations

import numpy as np

from arc_utils import colors_present


# Design analysis
# ---
# variant_spec:
#   core_change:
#     original_input: 只有矩形裁剪区本体，两种颜色直接表示 motif 前景/背景。
#     new_input: 在裁剪区外增加 1 像素统一边框，边框颜色来自固定四色码本，分别表示
#       0/90/180/270 度旋转。裁剪区内部仍然只保留两色 motif。
#   concrete_pixels:
#     interior_crop: HxW，H in [8,16]，W in [8,12]，内容仍是某个 20x20 motif 变体的局部窗口。
#     frame: (H+2)x(W+2) 外圈全同色，内部不出现该色；总输入颜色数因此稳定为 3。
#     output: 仍然是 20x20 全图，只用 interior 的两种颜色重绘，不保留边框。
#   behavior_trace:
#     original_flow:
#       - 把整个输入当作 crop。
#       - 在 8 个二面体方向和 2 个前景极性上做穷举匹配。
#     new_flow:
#       - 先剥掉边框，读出边框颜色并解码出 rotation_k。
#       - 只在该旋转下检查 2 个候选: 未翻转 / 左右翻转。
#       - 再对 interior 的两种颜色尝试前景极性，得到唯一匹配后输出完整 20x20。
#     difference:
#       - 新信息层不再改变输出图案本身，而是把“旋转角度”显式编码到边框颜色里。
#   complexity_calibration:
#     - 旋转只保留 4 档，颜色码本也固定为 4 色，避免自由度过大。
#     - interior crop 尺寸仍沿用原题区间，并通过 rejection 保证在“已知旋转、未知翻转”
#       的条件下仍然唯一，避免因为搜索空间缩小而退化成可多解输入。
#     - 边框颜色固定复用，训练样例里会出现重复边框色，解题者能从输入输出对应关系归纳
#       “边框色 -> 旋转角度”的规则，而不是把边框当成装饰。
#   implicit_assumption_audit:
#     - 原 `H, W = g.shape` 默认整个网格就是 crop；变体下必须先去掉 frame。
#     - 原 `colors_present(g)` 默认总共只有 2 色；变体下要改为 `colors_present(interior)`。
#     - 原 `_dihedral_variants(base)` 全量穷举依赖“输入不携带方向线索”；变体下该前提失效，
#       继续全量穷举会忽略 D14 变化，因此必须先由边框解码旋转。
#     - 原输出直接沿用匹配到的 fg/bg；这一点仍成立，但必须确保 border 色不会误入
#       interior 配色，因此 generator 固定从非码本颜色里选 fg/bg。
#   placement_legality:
#     - border 永远包住整块 crop，不与 interior 重叠，也不会越界，因为输入网格就是
#       生成后的 framed crop 本身。
#     - generator 先在 20x20 完整图上选合法窗口，再向外扩一圈边框，因此 interior 一定
#       可被 solver 还原。
#   visual_observability:
#     - 核心视觉差异是每个输入都多了统一外框，且外框颜色与输出最终朝向相关。
#     - 输出不再包含边框，解题者会看到“某种边框色对应某种完整朝向”的跨样例重复。
#   cleanliness:
#     - 边框只占最外层 1 像素，不覆盖 interior 图案，层次清晰。
#   color_inference:
#     - 码本固定为 4 种高辨识度颜色，interior 颜色从剩余调色板选取，避免边框色和
#       图案色重合而削弱可读性。
#   solver_assumptions_after_change:
#     - 输入必须有至少 3x3 尺寸，且外围是完整单色框。
#     - 边框颜色必须来自固定四色码本。
#     - interior 必须恰好两色，并且在“旋转已知、翻转未知、极性未知”的搜索下唯一。
#     - generator 通过构造和 `main(input) == expected_output` 验证来保证这些前提。


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

_ROTATION_TO_BORDER = {
    0: 1,  # blue -> 0 degrees
    1: 2,  # red -> 90 degrees
    2: 3,  # green -> 180 degrees
    3: 4,  # yellow -> 270 degrees
}
_BORDER_TO_ROTATION = {color: k for k, color in _ROTATION_TO_BORDER.items()}
_INTERIOR_PALETTE = (0, 5, 6, 7, 8, 9)


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _extract_framed_crop(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 3 and W >= 3

    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    border_color = int(border[0])
    assert np.all(border == border_color)
    assert border_color in _BORDER_TO_ROTATION

    interior = g[1:-1, 1:-1]
    assert interior.shape[0] <= 20 and interior.shape[1] <= 20
    return border_color, interior


def _rotation_candidates(mask: np.ndarray, rotation_k: int) -> list[np.ndarray]:
    rotated = np.rot90(mask, rotation_k)
    return [rotated, np.fliplr(rotated)]


def _collect_matches(visible: np.ndarray, rotation_k: int) -> list[np.ndarray]:
    H, W = map(int, visible.shape)
    matches: list[np.ndarray] = []
    for pattern in _rotation_candidates(_pattern_mask(), rotation_k):
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                if np.array_equal(pattern[r : r + H, c : c + W], visible):
                    matches.append(pattern)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the shared 20x20 motif from a framed crop whose border color announces the rotation.

    Input structure:
    - The grid is a two-color rectangular crop of the motif, wrapped in a 1-pixel single-color frame.
    - The frame color is a rotation code: one of four fixed colors meaning 0/90/180/270 degrees.
    - Inside the frame, one color is motif foreground and the other is motif background.

    Rule:
    - Remove the outer frame and decode its color into the rotation of the hidden 20x20 motif.
    - Try each interior color as the foreground mask, but only search the two candidates that share the decoded rotation: unflipped and horizontally flipped.
    - Repaint the uniquely matched full motif with the two interior colors and discard the frame.

    Key insight:
    - The border is not decorative; it eliminates three rotation classes up front, so the crop only needs to disambiguate flip state, offset, and foreground polarity.
    """
    border_color, interior = _extract_framed_crop(np.asarray(grid, dtype=int))

    colors = colors_present(interior)
    assert len(colors) == 2

    rotation_k = _BORDER_TO_ROTATION[border_color]
    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = interior == int(fg)
        for pattern in _collect_matches(visible, rotation_k):
            matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    base = _pattern_mask()

    for _ in range(1000):
        fg, bg = map(int, rng.choice(_INTERIOR_PALETTE, size=2, replace=False))
        rotation_k = int(rng.integers(0, 4))
        flipped = bool(rng.integers(0, 2))

        pattern = np.rot90(base, rotation_k)
        if flipped:
            pattern = np.fliplr(pattern)

        output = np.where(pattern, fg, bg).astype(int)

        crop_h = int(rng.integers(8, 17))
        crop_w = int(rng.integers(8, 13))
        r0 = int(rng.integers(0, 21 - crop_h))
        c0 = int(rng.integers(0, 21 - crop_w))

        interior = output[r0 : r0 + crop_h, c0 : c0 + crop_w]
        visible = interior == fg
        if len(_collect_matches(visible, rotation_k)) != 1:
            continue

        border_color = _ROTATION_TO_BORDER[rotation_k]
        framed = np.full((crop_h + 2, crop_w + 2), border_color, dtype=int)
        framed[1:-1, 1:-1] = interior

        assert border_color not in colors_present(interior)
        if np.array_equal(main(framed), output):
            return framed

    raise AssertionError("failed to generate a unique framed crop")
