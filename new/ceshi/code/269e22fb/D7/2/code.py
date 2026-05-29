# Variant of 269e22fb
# Dimension: D7 — 裁剪窗口尺寸约束 (meta)
# Change: 行数8-16，列数8-12，面积80-156，保证min(H,W)>=11或max(H,W)>=12时唯一性100%... → 裁剪窗口尺寸编码额外信息

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc_utils import colors_present


# Design analysis
# ===============
# spec:
#   source_task: None
#   preserved_core:
#     - 输入仍是同一个 20x20 二值母图某个二面体朝向的双色矩形裁剪。
#     - 解题仍先靠裁剪内容唯一定位: 前景/背景极性 + 朝向 + 偏移。
#   D7_change:
#     - 裁剪尺寸不再只是被动约束, 而是额外编码一个二面体变换。
#     - 行数奇偶编码翻转:
#         * H 为偶数 -> 不翻转
#         * H 为奇数 -> 在旋转之后做一次左右翻转
#     - 列数编码旋转次数:
#         * W=8 或 12 -> 旋转 0 次
#         * W=9       -> 逆时针旋转 1 次
#         * W=10      -> 逆时针旋转 2 次
#         * W=11      -> 逆时针旋转 3 次
#     - 因而 (H parity, W code) 一共得到 8 个分支, 正好覆盖一个完整的二面体动作集。
#   behavior_trace:
#     - 原任务: 识别裁剪来自哪个完整朝向 -> 直接输出该完整朝向。
#     - 新变体: 先识别裁剪来自哪个完整朝向 -> 再读取 H/W 的隐含编码 -> 对完整朝向追加一次尺寸指定的旋转/翻转 -> 输出结果。
#     - 具体例子:
#         * 若某个 13x10 输入窗口匹配到 pattern_A,
#           原任务会输出 pattern_A;
#           现在因为 H=13 为奇数、W=10 表示 rot180,
#           新任务输出 fliplr(rot180(pattern_A))。
#   complexity_calibration:
#     - 仍然只有一个固定 20x20 母图, 颜色仍然只保留 2 色, 没有引入新的对象层。
#     - 新自由度严格限制为 8 个尺寸编码分支, 不依赖颜色 id、随机顺序或隐藏元数据。
#     - generator 显式先抽取这 8 个分支之一, 避免长期退化到 identity 分支。
#   implicit_assumption_review:
#     - 原 main 中 H/W 只用于枚举窗口位置; 现在 H/W 还必须在唯一匹配之后参与输出变换。
#     - 原先 "匹配到的 pattern 就是最终输出 pattern" 这个前提已失效, 必须拆成:
#         1. 找到 observed full pattern
#         2. 再计算 transformed output pattern
#     - 原先唯一性断言仍然需要; generate_input 通过尺寸下界与重试机制保证它成立。
#   placement_legality:
#     - 仍然只从 20x20 完整图中截取一个 HxW 子窗口, 因此不会有重叠或越界新问题。
#     - 为复用原任务的唯一性保证:
#         * 当 W <= 10 时, 强制 H >= 12
#         * 当 W >= 11 时, 允许 H >= 11
#       这样始终满足 "max(H,W)>=12 或 min(H,W)>=11"。
#     - 同时保留原始面积约束 80-156。
#   visual_observability:
#     - 输入中 H/W 是直接可见的, 而输出的整体朝向变化也很显眼。
#     - 宽度只分成 4 个可见档位, 高度只编码奇偶, 规律足够小, 可从少量样例归纳。
#     - 母图本身非对称, 旋转/翻转在视觉上清楚可见, 不会被纯色或对称性掩盖。
#   cleanliness:
#     - 网格中仍只有一个裁剪窗口和一个完整图案, 不引入额外标记, 视觉上保持原题整洁。
#   color_reasoning:
#     - 颜色关系完全继承原题: 仅用裁剪中的两色重绘完整图案。
#     - 尺寸编码不依赖颜色值, 避免不可见的颜色 id 规则。
#   post_change_main_assumptions:
#     - solver 假设 W 一定属于 8..12 的五个合法宽度之一, generate_input 严格构造。
#     - solver 假设 H 的奇偶就是唯一翻转位, generate_input 不会混入其他尺寸语义。
#     - solver 假设裁剪内容本身仍可唯一定位完整图案, generator 对每个样本做显式验证。


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

_WIDTH_TO_ROTATIONS = {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 0,
}
_ROTATION_TO_WIDTHS = {
    0: (8, 12),
    1: (9,),
    2: (10,),
    3: (11,),
}
_ALL_COLORS = tuple(range(10))


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _decode_size_transform(height: int, width: int) -> tuple[int, bool]:
    assert 8 <= int(height) <= 16
    assert int(width) in _WIDTH_TO_ROTATIONS
    rotations = _WIDTH_TO_ROTATIONS[int(width)]
    flip = bool(int(height) % 2)
    return rotations, flip


def _apply_size_transform(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rotations, flip = _decode_size_transform(height, width)
    transformed = np.rot90(pattern, rotations)
    if flip:
        transformed = np.fliplr(transformed)
    return transformed


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height <= 20 and width <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20 - height + 1):
                for c in range(20 - width + 1):
                    if np.array_equal(pattern[r : r + height, c : c + width], visible):
                        matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden 20x20 motif, then re-orient it using a size-coded transform.

    Input structure:
    - The grid is still a two-color rectangular crop from one dihedral orientation of the shared 20x20 motif.
    - The crop dimensions now carry extra meaning: the height parity and the width bucket encode an additional transform.

    Rule:
    - First infer which full motif orientation and color polarity produced the visible crop.
    - Decode the crop size: odd heights request a left-right mirror, while widths 8/12, 9, 10, 11 request 0/90/180/270 counterclockwise turns.
    - Apply that decoded transform to the recovered full motif and repaint it with the matched foreground/background colors.

    Key insight:
    - The pixels inside the window tell you which full motif was sampled, but the window's shape tells you how to rotate/flip that motif before producing the answer.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    fg, bg, pattern = _find_unique_match(g)
    transformed = _apply_size_transform(pattern, height, width)
    return np.where(transformed, int(fg), int(bg)).astype(int)


def _height_options(width: int, flip: bool) -> list[int]:
    min_height = 11 if int(width) >= 11 else 12
    max_height = min(16, 156 // int(width))
    parity = 1 if flip else 0
    return [
        height
        for height in range(min_height, max_height + 1)
        if height % 2 == parity and 80 <= height * int(width) <= 156
    ]


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    for _ in range(512):
        rotations = int(rng.integers(0, 4))
        flip = bool(int(rng.integers(0, 2)))
        width = int(rng.choice(_ROTATION_TO_WIDTHS[rotations]))
        height_choices = _height_options(width, flip)
        assert height_choices
        height = int(rng.choice(height_choices))

        fg, bg = map(int, rng.choice(_ALL_COLORS, size=2, replace=False))
        pattern = variants[int(rng.integers(0, len(variants)))]

        r = int(rng.integers(0, 20 - height + 1))
        c = int(rng.integers(0, 20 - width + 1))
        crop = pattern[r : r + height, c : c + width]
        inp = np.where(crop, fg, bg).astype(int)

        matched_fg, matched_bg, matched_pattern = _find_unique_match(inp)
        out_pattern = _apply_size_transform(pattern, height, width)
        expected = np.where(out_pattern, fg, bg).astype(int)

        # Reject anything that would weaken the intended branch semantics.
        if matched_fg != fg or matched_bg != bg:
            continue
        if not np.array_equal(matched_pattern, pattern):
            continue
        if not np.array_equal(main(inp), expected):
            continue

        return inp, expected, {
            "height": height,
            "width": width,
            "rotations": rotations,
            "flip": int(flip),
            "offset_r": r,
            "offset_c": c,
        }

    raise RuntimeError("failed to generate a valid D7 variant example")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    inp, _expected, _meta = _build_case(rng)
    return inp
