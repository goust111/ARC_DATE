# Variant of 269e22fb
# Dimension: D3 — 目标输出：完整图案的展示方式 (dst)
# Change: 输出完整的20x20图案，前景/背景用输入的两种颜色绘制... → 输出图案的特定象限

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# ===============
# a) 具体化与行为追踪
# - 输入仍然是一个 20x20 固定二值母图在某个二面体朝向下的矩形裁剪，网格里只有两种颜色。
# - 原题在唯一匹配出完整 20x20 朝向后，直接把整张 20x20 图案重绘出来。
# - 本变体只改“展示结果”这一步：先像原题一样恢复出完整朝向与裁剪偏移，再看裁剪窗口几何中心
#   落在完整 20x20 的哪个 10x10 象限中，输出该象限的对角象限。
# - 具体例子：若某个 12x10 裁剪位于完整图案的 rows 1:13, cols 2:12，则中心是 (7.0, 7.0)，属于左上象限；
#   原题输出完整 20x20，本题则输出右下 10x10，即 rows 10:20, cols 10:20 的重绘结果。
# - 差异点因此只发生在“匹配成功之后的取图范围”，输入构造和唯一匹配逻辑保持一致。
#
# b) 复杂度校准
# - 若仍输出 20x20，只是换个颜色或微调局部，会和原题过于接近；改成“只输出一个象限”能形成明确的
#   尺寸与内容变化，肉眼可见。
# - 若象限选择依赖太复杂的几何关系，人类难以从 2-4 个样例归纳；这里把控制变量压缩为 4 个象限分支。
# - 为避免生成器长期落在同一分支，generate_input() 会先均匀抽一个源象限，再在该象限里抽合法偏移。
#
# c) 隐式假设审查
# - `assert H <= 20 and W <= 20`：仍成立，裁剪仍来自 20x20 母图。
# - `colors_present(g)` / `assert len(colors) == 2`：仍成立，输入依旧严格双色。
# - 原题 `matches.append((fg, bg, pattern))` 只保存颜色和朝向；本变体还需要窗口偏移来判定象限，
#   所以必须额外保存 `(r, c)`。
# - 原题 `return np.where(pattern, fg, bg)` 默认输出整张图；在本变体中这个假设失效，必须先切出目标 10x10 象限。
# - 新增的象限判定假设“窗口中心不落在横/竖中线 x=10 或 y=10 上”，否则哪个象限包含中心会歧义；
#   generate_input() 会显式避开这些位置。
#
# d) 放置合法性
# - 裁剪仍通过 `r in [0, 20-H]`, `c in [0, 20-W]` 保证不越界。
# - 不存在多个对象，因此不存在重叠冲突；唯一需要保证的是中心象限可判定且匹配唯一。
# - 原题 spec 给出唯一性保障：`min(H, W) >= 11 or max(H, W) >= 12` 时唯一匹配有保证。
#   生成器因此只采样满足该条件的尺寸，避免高拒绝率。
#
# e) 视觉可观察性
# - 输出尺寸从 20x20 变为 10x10，且具体输出哪一个象限取决于裁剪位置，是直观可见的 D3 变化。
# - 规则依赖空间位置而非颜色 id；颜色仅沿用输入的前景/背景，不承担隐藏编码作用。
# - 四个分支都是“源象限 -> 对角象限”的同一映射规律，人类可从少量样例归纳。
#
# f) 可视化整洁度
# - 输出只保留一个完整 10x10 象限，不叠加额外标记，画面比在 20x20 上再做符号标注更干净。
# - 输入仍然只有一个矩形裁剪，不引入新元素，不会破坏原题的清晰度。
#
# g) 颜色变化可推断性
# - 没有新增颜色映射；目标象限仍用输入两色重绘，颜色语义与原题一致。
#
# h) 生成器验证指标
# - 关键控制变量是“窗口中心所在源象限”。若生成器退化，四个源象限会严重失衡。
# - 额外统计时检查四个源象限与四个目标象限的分布，确认新分支实际被触发。
#
# i) main() 的新假设与生成器担保
# - main() 现在假设唯一匹配结果里包含明确 `(r, c)` 偏移；_find_matches() 统一提供。
# - main() 依赖 `_quadrant_of_window()` 不触发边界断言；generate_input() 通过 `_is_clear_of_midlines()`
#   保证中心离中线至少 1 个单位。


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

_QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))


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


def _quadrant_slices(quadrant: tuple[int, int]) -> tuple[slice, slice]:
    qr, qc = quadrant
    r0 = 10 * int(qr)
    c0 = 10 * int(qc)
    return slice(r0, r0 + 10), slice(c0, c0 + 10)


def _quadrant_of_window(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    row_center = float(r) + float(h) / 2.0
    col_center = float(c) + float(w) / 2.0
    assert row_center != 10.0 and col_center != 10.0
    return int(row_center >= 10.0), int(col_center >= 10.0)


def _opposite_quadrant(quadrant: tuple[int, int]) -> tuple[int, int]:
    qr, qc = quadrant
    return 1 - int(qr), 1 - int(qc)


def _selected_quadrant_mask(pattern: np.ndarray, r: int, c: int, h: int, w: int) -> np.ndarray:
    src_quadrant = _quadrant_of_window(r, c, h, w)
    dst_quadrant = _opposite_quadrant(src_quadrant)
    rs, cs = _quadrant_slices(dst_quadrant)
    return pattern[rs, cs]


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - h + 1):
                for c in range(20 - w + 1):
                    if np.array_equal(pattern[r : r + h, c : c + w], visible):
                        matches.append((int(fg), int(bg), pattern, r, c))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared motif, then reveal only the rule-selected 10x10 quadrant.

    Input structure:
    - The input is still a two-color rectangular crop taken from one dihedral orientation of the same 20x20 binary motif.
    - The two observed colors encode motif foreground vs. motif background, but their roles are not labeled in advance.

    Rule:
    - Test both observed colors as the possible foreground and find the unique matching orientation and crop offset in the 20x20 motif.
    - Use the crop window's geometric center to determine which 10x10 quadrant of the full motif the crop belongs to.
    - Output the diagonally opposite 10x10 quadrant, repainted with the matched foreground/background colors.

    Key insight:
    - The crop still uniquely identifies the full motif orientation; the new work is to keep the recovered offset and convert that location into an opposite-quadrant selection.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 20 and w <= 20

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern, r, c = matches[0]
    quadrant_mask = _selected_quadrant_mask(pattern, r, c, h, w)
    return np.where(quadrant_mask, int(fg), int(bg)).astype(int)


def _sample_dimensions(rng: random.Random) -> tuple[int, int]:
    while True:
        h = rng.randint(8, 16)
        w = rng.randint(8, 12)
        if min(h, w) >= 11 or max(h, w) >= 12:
            return h, w


def _is_clear_of_midlines(r: int, c: int, h: int, w: int) -> bool:
    row_center = float(r) + float(h) / 2.0
    col_center = float(c) + float(w) / 2.0
    return abs(row_center - 10.0) >= 1.0 and abs(col_center - 10.0) >= 1.0


def _expected_output(pattern: np.ndarray, fg: int, bg: int, r: int, c: int, h: int, w: int) -> np.ndarray:
    quadrant_mask = _selected_quadrant_mask(pattern, r, c, h, w)
    return np.where(quadrant_mask, int(fg), int(bg)).astype(int)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    palette = list(range(10))

    for _ in range(1000):
        fg, bg = rng.sample(palette, 2)
        pattern = _VARIANTS[rng.randrange(len(_VARIANTS))]
        full = np.where(pattern, int(fg), int(bg)).astype(int)

        h, w = _sample_dimensions(rng)
        src_quadrant = _QUADRANTS[rng.randrange(len(_QUADRANTS))]

        valid_offsets = [
            (r, c)
            for r in range(20 - h + 1)
            for c in range(20 - w + 1)
            if _is_clear_of_midlines(r, c, h, w) and _quadrant_of_window(r, c, h, w) == src_quadrant
        ]
        if not valid_offsets:
            continue

        r, c = valid_offsets[rng.randrange(len(valid_offsets))]
        inp = full[r : r + h, c : c + w].copy()
        expected = _expected_output(pattern, fg, bg, r, c, h, w)
        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp, expected, {
                "src_quadrant": src_quadrant,
                "dst_quadrant": _opposite_quadrant(src_quadrant),
                "shape": (h, w),
                "offset": (r, c),
            }

    raise RuntimeError("failed to generate a valid variant input after 1000 attempts")


def generate_input() -> np.ndarray:
    rng = random.Random()
    inp, _, _ = _build_case(rng)
    return inp
