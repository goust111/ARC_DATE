# Combo variant of 269e22fb (index 24)
# combo_spec: D3/3+D4/3
# Name: 象限色彩系统与象限选择输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题输入是某个 20x20 固定二值母图在二面体朝向下的一个双色矩形裁剪；恢复成功后输出整张 20x20，
#   并统一用输入的前景/背景颜色重绘。
# - 本融合变体同时改两处：
#   1. 20x20 画面被固定分成四个 10x10 象限，每个象限都有自己的有序颜色对 (fg, bg)。
#   2. 输出不再是整张 20x20，而是“输入所在象限的对角象限”的 10x10 内容，并且要用目标象限自己的颜色对。
# - 具体跟踪：若输入是 top-right 象限内一个 8x9 裁剪，且该象限颜色对为 (2, 3)，那么求解器先据此知道
#   “本地前景色是 2，本地背景色是 3，搜索偏移只能在 top-right 的 10x10 区域里进行”。一旦匹配出完整朝向，
#   不输出整张四色 20x20，而是切出对角的 bottom-left 10x10，并改用 bottom-left 颜色对 (4, 1) 着色。
# - 因此与原题的核心差异有两层：输入颜色不再只是全局前景/背景，而是源象限编码；输出也不再是整图，而是
#   空间上按对角规则选出的局部象限。
#
# b) 复杂度校准
# - 如果四个象限颜色对按样例随机变化，人类无法从单个输入推断未出现的目标象限颜色，因此颜色系统必须在整题固定。
# - 采用四个固定象限颜色对，并保证四个“无序颜色集合”互不相同，这样输入的两色就能唯一标识源象限。
# - 裁剪尺寸限制在 8..10 之间，使其能够完整落在单个 10x10 象限中并保持双色，同时仍保留足够局部结构来唯一匹配朝向。
# - 输出固定为 10x10 象限，视觉上与原题的 20x20 整图和 D4 单变体的 20x20 四色整图都有明确区别。
#
# c) 隐式假设审查
# - 原题 `assert len(colors) == 2` 仍然成立，但不再来自“任意位置裁剪都双色”；现在必须由生成器保证裁剪完全位于单一象限。
# - 原题“尝试两个输入色分别作为全局前景”的逻辑失效，因为在融合后前景色由源象限的有序颜色对唯一决定；
#   若仍双向尝试，会把象限编码误当作极性自由度。
# - 原题匹配阶段遍历全部 20x20 偏移；融合后这一假设也失效，必须先通过颜色推断源象限，再把搜索限制到对应象限内部。
# - 原题 `return np.where(pattern, fg, bg)` 假设输出沿用输入两色且覆盖全图；融合后需要先选对角象限，再用目标象限颜色对重绘。
# - 新增假设是“输入颜色集合必须恰好等于某一个固定象限的颜色集合”；生成器只从已渲染的象限编码大图里裁剪，来担保这一点。
#
# d) 放置合法性
# - 每个输入裁剪必须满足 `H, W in [8, 10]`，并且 `(r, c)` 只能选自某一 10x10 象限内部的合法左上角集合，避免越界或跨象限。
# - 输入直接从已按象限着色的 20x20 完整图裁出，因此不会产生颜色冲突，也不会违背双色约束。
# - 生成器只保留 `_find_matches()` 返回唯一解的样本，从而保证 solver 的唯一匹配断言总是成立。
#
# e) 视觉可观察性
# - 输入中的两色会在不同样例里重复出现为某个固定象限的“身份标签”，输出则始终切换到对角象限，并显式换成另一组颜色。
# - 人类观察 2-4 组样例时，可以同时看到两个稳定结构：颜色对对应固定空间象限，输出位置总是对角映射。
# - 规则不依赖颜色数值大小或排序，只依赖肉眼可见的颜色组合和象限位置。
#
# f) 可视化整洁度
# - 输入仍然只是一个干净的矩形裁剪，没有额外标记。
# - 输出只展示一个完整 10x10 象限，避免把“四色系统”和“象限选择”叠在同一张 20x20 图里造成过度拥挤。
# - 四个象限边界固定为 10x10 分区，视觉结构简单明确。
#
# g) 变化的可推断性
# - 固定颜色循环采用：
#     TL = (1, 2)
#     TR = (2, 3)
#     BL = (4, 1)
#     BR = (3, 4)
#   四个无序颜色对分别是 {1,2}, {2,3}, {1,4}, {3,4}，彼此唯一。
# - 对角象限总会切换到另一组颜色对，因此融合行为能被样例直接观察到，而不会退化成“只切象限但颜色不变”。
#
# h) 生成器验证指标
# - 关键控制变量是源象限 `src_quadrant`。若生成器退化，四个源象限或四个对角迁移分支会明显失衡。
# - 另一个关键指标是 `set(input_colors) != set(output_colors)`。若这一项频繁失败，就说明 D4 分量被削弱成了 D3 单变体。
#
# i) main() 的新假设与生成器担保
# - `_quadrant_from_colors()` 假设输入颜色集合能唯一映射到某个固定象限；生成器只从四色编码整图的单一象限内取裁剪。
# - `_find_matches()` 假设真实偏移一定在 `_quadrant_offsets(src_quadrant, H, W)` 里；生成器对偏移做同样的边界限制。
# - `main()` 假设匹配结果唯一；`_enumerate_valid_cases()` 在离线枚举时已经过滤掉所有非唯一窗口。


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

_QUADRANT_BOUNDS = (
    (0, 10, 0, 10),    # TL
    (0, 10, 10, 20),   # TR
    (10, 20, 0, 10),   # BL
    (10, 20, 10, 20),  # BR
)

_QUADRANT_PAIRS = (
    (1, 2),  # TL
    (2, 3),  # TR
    (4, 1),  # BL
    (3, 4),  # BR
)


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


def _quadrant_offsets(quadrant: int, h: int, w: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _QUADRANT_BOUNDS[quadrant]
    return [(r, c) for r in range(r0, r1 - h + 1) for c in range(c0, c1 - w + 1)]


def _quadrant_slices(quadrant: int) -> tuple[slice, slice]:
    r0, r1, c0, c1 = _QUADRANT_BOUNDS[quadrant]
    return slice(r0, r1), slice(c0, c1)


def _opposite_quadrant(quadrant: int) -> int:
    return 3 - int(quadrant)


def _quadrant_from_colors(colors: list[int]) -> int:
    color_set = {int(c) for c in colors}
    candidates = [
        quadrant
        for quadrant, pair in enumerate(_QUADRANT_PAIRS)
        if {int(pair[0]), int(pair[1])} == color_set
    ]
    assert len(candidates) == 1
    return int(candidates[0])


def _render_full_quadrant_system(pattern: np.ndarray) -> np.ndarray:
    out = np.zeros((20, 20), dtype=int)
    for quadrant in range(4):
        rs, cs = _quadrant_slices(quadrant)
        fg, bg = _QUADRANT_PAIRS[quadrant]
        out[rs, cs] = np.where(pattern[rs, cs], int(fg), int(bg))
    return out


def _render_selected_output(pattern: np.ndarray, src_quadrant: int) -> np.ndarray:
    dst_quadrant = _opposite_quadrant(src_quadrant)
    rs, cs = _quadrant_slices(dst_quadrant)
    fg, bg = _QUADRANT_PAIRS[dst_quadrant]
    return np.where(pattern[rs, cs], int(fg), int(bg)).astype(int)


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2

    src_quadrant = _quadrant_from_colors(colors)
    fg, _bg = _QUADRANT_PAIRS[src_quadrant]
    visible = g == int(fg)

    matches: list[tuple[int, int, int, np.ndarray]] = []
    for pattern in _VARIANTS:
        for r, c in _quadrant_offsets(src_quadrant, h, w):
            if np.array_equal(pattern[r : r + h, c : c + w], visible):
                matches.append((int(src_quadrant), int(r), int(c), pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the source quadrant from the crop colors, recover the full motif, then output only the diagonally opposite quadrant.

    Input structure:
    - The grid is a two-color rectangular crop taken entirely from one 10x10 quadrant of a rotated/flipped 20x20 binary motif.
    - Each of the four 10x10 quadrants uses its own fixed ordered foreground/background color pair, so the crop colors identify the source quadrant.

    Rule:
    - Read the crop's two colors as a quadrant code, which also fixes which observed color is local foreground.
    - Match that local foreground mask against every legal offset inside the inferred source quadrant across all dihedral variants of the 20x20 motif.
    - After the unique match is found, select the diagonally opposite 10x10 quadrant and repaint that quadrant with its own fixed color pair.

    Key insight:
    - The crop no longer tells you a global palette; it tells you where in the four-quadrant color system the crop came from, and that location determines both the allowed search region and which opposite quadrant should be revealed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 10 and w <= 10

    matches = _find_matches(g)
    assert len(matches) == 1
    src_quadrant, _r, _c, pattern = matches[0]
    return _render_selected_output(pattern, src_quadrant)


def _enumerate_valid_cases() -> list[tuple[int, int, int, int, int, int]]:
    cases: list[tuple[int, int, int, int, int, int]] = []
    for variant_idx, pattern in enumerate(_VARIANTS):
        full = _render_full_quadrant_system(pattern)
        for src_quadrant in range(4):
            for h in range(8, 11):
                for w in range(8, 11):
                    for r, c in _quadrant_offsets(src_quadrant, h, w):
                        crop = full[r : r + h, c : c + w]
                        if len(colors_present(crop)) != 2:
                            continue
                        if len(_find_matches(crop)) != 1:
                            continue
                        if set(colors_present(crop)) == set(_QUADRANT_PAIRS[_opposite_quadrant(src_quadrant)]):
                            continue
                        cases.append((variant_idx, src_quadrant, h, w, r, c))
    assert cases
    return cases


_VALID_CASES = _enumerate_valid_cases()
_VALID_CASES_BY_QUADRANT = {
    quadrant: [spec for spec in _VALID_CASES if spec[1] == quadrant]
    for quadrant in range(4)
}
_GEN_QUADRANT_ORDER: list[int] = []


def _next_source_quadrant() -> int:
    global _GEN_QUADRANT_ORDER
    if not _GEN_QUADRANT_ORDER:
        _GEN_QUADRANT_ORDER = list(range(4))
        random.shuffle(_GEN_QUADRANT_ORDER)
    return int(_GEN_QUADRANT_ORDER.pop())


def _build_case(spec: tuple[int, int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    variant_idx, src_quadrant, h, w, r, c = spec
    pattern = _VARIANTS[variant_idx]
    full = _render_full_quadrant_system(pattern)
    input_grid = full[r : r + h, c : c + w].copy()
    output_grid = _render_selected_output(pattern, src_quadrant)
    return input_grid, output_grid, {
        "variant_idx": int(variant_idx),
        "src_quadrant": int(src_quadrant),
        "dst_quadrant": _opposite_quadrant(src_quadrant),
        "shape": (int(h), int(w)),
        "offset": (int(r), int(c)),
        "input_colors": tuple(colors_present(input_grid)),
        "output_colors": tuple(colors_present(output_grid)),
    }


def generate_input() -> np.ndarray:
    src_quadrant = _next_source_quadrant()
    spec = random.choice(_VALID_CASES_BY_QUADRANT[src_quadrant])
    input_grid, output_grid, _meta = _build_case(spec)
    assert np.array_equal(main(input_grid), output_grid)
    return input_grid
