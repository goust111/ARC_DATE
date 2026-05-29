# Variant of 269e22fb
# Dimension: D4 — 变换：颜色重绘规则 (transform)
# Change: 用裁剪窗口中的前景色和背景色直接重绘完整图案（前景->color_a, 背景->color_b）... → 使用第三色标记窗口位置

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# ---------------
# spec_like:
#   variant_goal:
#     keep_input: 输入仍然是 20x20 motif 某个 D4 变体上的一个双色矩形裁剪窗口
#     change_output: 输出仍然恢复完整 20x20 motif，但窗口覆盖到的 motif 前景像素改涂固定第三色
#   concrete_pixels:
#     input:
#       - 只有 2 色: `fg` 表示 motif=1, `bg` 表示 motif=0
#       - 看起来和原题一样，只是一个 HxW 的局部裁剪
#     output:
#       - motif 外观仍由同一个 20x20 布尔模板决定
#       - 窗口外的前景像素: `fg`
#       - 全部背景像素: `bg`
#       - 窗口内且属于 motif 前景的像素: `MARKER_COLOR`
#       - 因此输出恰好 3 色，三种语义分别是: 普通前景 / 普通背景 / 被窗口命中的前景
#   behavior_trace:
#     original:
#       - 若输入 crop 中某格是前景，输出完整图中所有 motif 前景都涂成同一个输入前景色
#     variant:
#       - 仍先根据 crop 唯一确定 orientation 和 offset
#       - 但在重绘完整图时，只把“该 crop 在原图位置里覆盖到的前景像素”单独改成第三色
#     difference:
#       - offset 不再只是隐含信息，而是直接体现在输出中的红色高亮区域
#   complexity_calibration:
#     - 不改变对象数量、变换空间和匹配机制，避免偏离原题
#     - 只在输出端增加一层可见语义，复杂度和参考任务里的“额外控制色”同量级
#     - 为避免红色高亮太稀疏，生成器要求 crop 内至少有一定数量的前景像素
#   implicit_assumptions_audit:
#     old_code_dependencies:
#       - `len(colors)==2`: 仍成立，因为输入不变
#       - 两色之一可当作前景并与某个窗口精确匹配: 仍成立
#       - 找到唯一匹配后只需记 orientation: 变体下不够，还必须保留窗口 offset `(r,c)`
#       - 输出只依赖 `pattern`: 变体下不够，还依赖窗口范围 `(r,c,H,W)`
#     new_dependencies:
#       - 第三色必须和输入两色不同，否则“标记窗口”会退化不可见
#       - crop 里至少要有一些 motif 前景，否则输出不会出现第三色
#     generator_guarantees:
#       - 输入颜色从不使用 `MARKER_COLOR`
#       - 只采样包含足够前景像素的窗口
#   placement_legality:
#     - motif 仍固定在 20x20 内，不存在额外对象，因此不会新增重叠/越界问题
#     - 唯一需要合法化的是窗口 `(r,c,H,W)`，必须满足 0 <= r <= 20-H, 0 <= c <= 20-W
#   visual_observability:
#     - 同一组训练样例里，输入总是双色 crop，输出总是三色完整图
#     - 新增第三色只出现在一个与输入尺寸一致的局部前景子集上，人类可从 2-3 例归纳“它标的是输入窗口位置”
#   visual_cleanliness:
#     - 只对已有前景换色，不叠加边框、不改背景，避免输出变乱
#   color_reasoning:
#     - 第三色固定为红色，语义在样例间稳定
#     - 输入两色随机但均不等于红色，因此红色总能被直接识别为“新角色”
#   main_assumption_review:
#     - 枚举 `fg` 候选时假设输入恰有两色，生成器保证
#     - 枚举 D4 变体和窗口时假设 crop 能唯一定位，沿用原题尺寸约束并在生成时自检
#     - 最终高亮时假设窗口内至少存在 motif 前景，生成器用 `fg_count` 下界保证


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

MARKER_COLOR = 2
_INPUT_COLOR_POOL = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _render_output(
    pattern: np.ndarray, fg: int, bg: int, top: int, left: int, height: int, width: int
) -> np.ndarray:
    out = np.where(pattern, int(fg), int(bg)).astype(int)
    window = pattern[top : top + height, left : left + width]
    out_slice = out[top : top + height, left : left + width]
    out_slice[window] = int(MARKER_COLOR)
    return out


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2
    assert int(MARKER_COLOR) not in colors

    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern, int(r), int(c)))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared 20x20 motif, then highlight where the observed crop came from.

    Input structure:
    - The grid is still a two-color rectangular crop from one rotation/reflection of the fixed 20x20 motif.
    - One input color encodes motif foreground and the other encodes motif background.

    Rule:
    - Test each input color as the possible foreground mask and align that mask against every window in every dihedral motif orientation.
    - Once the crop's orientation and offset are uniquely identified, repaint the full 20x20 motif with the input foreground/background colors.
    - Recolor only the matched window's foreground cells to a fixed third marker color.

    Key insight:
    - The crop still serves only to identify the hidden global orientation and placement, but the output now exposes that recovered placement by coloring the crop-covered motif pixels differently.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    fg, bg, pattern, top, left = _find_unique_match(g)
    assert pattern[top : top + H, left : left + W].any()
    return _render_output(pattern, fg, bg, top, left, H, W)


def _random_colors() -> tuple[int, int]:
    fg, bg = random.sample(_INPUT_COLOR_POOL, 2)
    return int(fg), int(bg)


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base = _pattern_mask()
    variants = _dihedral_variants(base)
    for _ in range(256):
        fg, bg = _random_colors()
        pattern = random.choice(variants)

        height = random.randint(11, 16)
        width = random.randint(8, 12)
        top = random.randint(0, 20 - height)
        left = random.randint(0, 20 - width)

        window = pattern[top : top + height, left : left + width]
        area = int(height * width)
        fg_count = int(window.sum())

        if fg_count < max(12, area // 5):
            continue
        if fg_count > area - max(10, area // 6):
            continue

        inp = np.where(window, int(fg), int(bg)).astype(int)
        expected = _render_output(pattern, fg, bg, top, left, height, width)
        actual = main(inp)
        if np.array_equal(actual, expected):
            stats = {
                "height": int(height),
                "width": int(width),
                "fg_count": int(fg_count),
                "area": int(area),
            }
            return inp, expected, stats

    raise RuntimeError("failed to generate a non-degenerate unique crop")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
