# Combo variant of 269e22fb (index 16)
# combo_spec: D3/3+D12/0
# Name: 角落裁剪对角象限输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题输入是 20x20 固定二值母图某个二面体朝向的双色矩形裁剪，裁剪可落在任意合法偏移；输出是整张 20x20 图。
# - 本融合版把两个环节同时改掉：
#   1. 裁剪位置不再自由，而是必须贴住四角之一。
#   2. 解出完整朝向后，不输出整张图，而是输出“裁剪所在角落”的对角 10x10 象限。
# - 具体例子：若 H=12, W=9，裁剪来自某朝向图案的左上角 `(r, c) = (0, 0)`，输入就是 rows `0:12`, cols `0:9`。
#   原题会输出完整 20x20；本题先识别这是左上角裁剪，再输出右下 10x10，即 rows `10:20`, cols `10:20`。
# - 若裁剪来自右下角 `(20-H, 20-W)`，则输出左上 10x10。差异因此同时体现在“搜索空间”与“目标图范围”。
#
# b) 复杂度校准
# - 只改成“角落裁剪”而仍输出整图，会让视觉差异不够强；只改成“输出象限”但允许任意偏移，又与单变体 D3/3
#   过于接近。二者叠加后，输入和输出都显著不同。
# - 四个角落对应四个对角象限，控制变量只有 4 档，样例中容易重复出现并被人类归纳。
# - 角落锚定还让匹配只需检查 4 个偏移，而不是整张 20x20 上的所有窗口，规则更聚焦。
#
# c) 隐式假设审查
# - `assert h <= 20 and w <= 20` 仍成立，裁剪仍来自同一张 20x20 母图。
# - `colors_present(g)` / `assert len(colors) == 2` 仍成立，输入依旧严格双色。
# - 原题默认“任意偏移都可能是候选”；融合后这个前提失效，所以 `_find_matches()` 必须只枚举四个角落偏移。
# - 原题只需保存 `(fg, bg, pattern)`；本题还要根据角落决定目标象限，所以必须保留 `(r, c, h, w)`。
# - 单独的 D3/3 参考实现依赖“窗口中心落在哪个象限”；融合后不再用中心判定，否则会把角落规则弱化成更宽泛的
#   几何规则。这里直接由角落锚定位置决定输出象限。
#
# d) 放置合法性
# - 四个合法偏移固定为 `(0, 0)`, `(0, 20-W)`, `(20-H, 0)`, `(20-H, 20-W)`，天然不越界。
# - 输入仍只有一个裁剪窗口，不存在对象重叠问题。
# - 生成器先选尺寸，再由尺寸导出四个角落偏移，因此不会产生非法位置。
# - 为了让四个角落分支在样例中稳定出现，生成器按轮转顺序覆盖四角，而不是完全依赖随机命中。
#
# e) 视觉可观察性
# - 输入窗口总是“吃住一个角”，这是非常强的空间线索；输出尺寸固定为 10x10，也与原题的 20x20 明显不同。
# - 四个分支共享同一条可见规律：哪个角被裁掉，就输出对角象限。这个规律只依赖位置，不依赖颜色编号。
# - 颜色仍沿用输入两色的前景/背景角色，解题者只需先确定哪种颜色是母图前景，再应用角落到对角象限的映射。
#
# f) 可视化整洁度
# - 输入没有附加标记或额外对象，只保留原题最干净的“双色裁剪”外观。
# - 输出仅保留一个完整 10x10 象限，不会在同一画布叠加多层提示，视觉上比在 20x20 中局部高亮更整洁。
#
# g) 变化的可推断性
# - 没有新增颜色体系，也没有依赖颜色大小、索引顺序等不可视因素。
# - 颜色变化只有“前景/背景极性待判定”这一层，和原题一致；新的可推断信息全部来自角落位置与对角象限的对应。
#
# h) 生成器验证指标
# - 关键控制变量是 `src_corner`。若生成器退化，四个角落分支会失衡，甚至长期只出现某一个角。
# - 这里用轮转采样保证连续 4 个样例恰好覆盖 4 个 `src_corner`，并同步轮转尺寸与朝向，避免分支被随机性淹没。
# - 统计时检查四个 `src_corner` 及其 `dst_corner` 的分布；同时新输出恒为 10x10，因此与原题整图输出不会混淆。
#
# i) main() 的新假设与生成器担保
# - `main()` 假设唯一匹配一定发生在四角之一；生成器只从 `_corner_offsets()` 产出输入，保证这一点。
# - `main()` 假设匹配结果唯一；在原始尺寸范围 `H in [8,16]`, `W in [8,12]` 下，四角搜索经枚举验证可唯一确定。
# - `_corner_of_offset()` 假设 `(r, c)` 精确等于某个角偏移；由于 `_find_matches()` 只扫描四角，这个断言不会误触发。


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

_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_DIMENSIONS = tuple((h, w) for h in range(8, 17) for w in range(8, 13))
_GENERATOR_CASE_INDEX = 0


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


def _quadrant_slices(corner: tuple[int, int]) -> tuple[slice, slice]:
    qr, qc = corner
    r0 = 10 * int(qr)
    c0 = 10 * int(qc)
    return slice(r0, r0 + 10), slice(c0, c0 + 10)


def _opposite_corner(corner: tuple[int, int]) -> tuple[int, int]:
    qr, qc = corner
    return 1 - int(qr), 1 - int(qc)


def _corner_offsets(h: int, w: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0), (0, 20 - w), (20 - h, 0), (20 - h, 20 - w))


def _corner_of_offset(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    mapping = {
        (0, 0): (0, 0),
        (0, 20 - w): (0, 1),
        (20 - h, 0): (1, 0),
        (20 - h, 20 - w): (1, 1),
    }
    assert (r, c) in mapping
    return mapping[(r, c)]


def _selected_quadrant_mask(pattern: np.ndarray, r: int, c: int, h: int, w: int) -> np.ndarray:
    src_corner = _corner_of_offset(r, c, h, w)
    dst_corner = _opposite_corner(src_corner)
    rs, cs = _quadrant_slices(dst_corner)
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
            for r, c in _corner_offsets(h, w):
                if np.array_equal(pattern[r : r + h, c : c + w], visible):
                    matches.append((int(fg), int(bg), pattern, r, c))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the motif orientation from a corner-anchored crop, then return the opposite 10x10 quadrant.

    Input structure:
    - The input is a two-color rectangular crop from one dihedral orientation of the fixed 20x20 binary motif.
    - Unlike the base task, the crop is always anchored to one of the four corners of the hidden 20x20 pattern.

    Rule:
    - Try both observed colors as candidate foreground, and compare the crop against every dihedral orientation of the motif.
    - Only consider the four corner offsets compatible with the crop size; exactly one orientation, polarity, and corner placement matches.
    - Determine which corner the crop occupies, take the diagonally opposite 10x10 quadrant from the recovered full motif, and repaint it with the matched colors.

    Key insight:
    - The crop still reveals enough local structure to identify the motif orientation, and the corner anchoring turns the crop location itself into the selector for which opposite quadrant to output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 20 and w <= 20

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern, r, c = matches[0]
    quadrant_mask = _selected_quadrant_mask(pattern, r, c, h, w)
    return np.where(quadrant_mask, int(fg), int(bg)).astype(int)


def _next_case_plan() -> tuple[np.ndarray, int, int, tuple[int, int]]:
    global _GENERATOR_CASE_INDEX

    case_index = _GENERATOR_CASE_INDEX
    _GENERATOR_CASE_INDEX += 1

    corner_index = case_index % len(_CORNERS)
    dim_index = (case_index // len(_CORNERS)) % len(_DIMENSIONS)
    variant_index = (case_index // (len(_CORNERS) * len(_DIMENSIONS))) % len(_VARIANTS)
    h, w = _DIMENSIONS[dim_index]
    return _VARIANTS[variant_index], h, w, _CORNERS[corner_index]


def _expected_output(pattern: np.ndarray, fg: int, bg: int, r: int, c: int, h: int, w: int) -> np.ndarray:
    quadrant_mask = _selected_quadrant_mask(pattern, r, c, h, w)
    return np.where(quadrant_mask, int(fg), int(bg)).astype(int)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    palette = list(range(10))

    for _ in range(1000):
        fg, bg = rng.sample(palette, 2)
        pattern, h, w, src_corner = _next_case_plan()
        full = np.where(pattern, int(fg), int(bg)).astype(int)

        corner_index = _CORNERS.index(src_corner)
        r, c = _corner_offsets(h, w)[corner_index]

        inp = full[r : r + h, c : c + w].copy()
        expected = _expected_output(pattern, fg, bg, r, c, h, w)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp, expected, {
                "src_corner": src_corner,
                "dst_corner": _opposite_corner(src_corner),
                "shape": (h, w),
                "offset": (r, c),
            }

    raise RuntimeError("failed to generate a valid combo input after 1000 attempts")


def generate_input() -> np.ndarray:
    rng = random.Random()
    inp, _, _ = _build_case(rng)
    return inp
