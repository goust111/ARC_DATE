# Combo variant of 269e22fb (index 11)
# combo_spec: D6/1+D15/1
# Name: 16变体50%密度高歧义匹配

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from arc_utils import Color, colors_present


# Design analysis:
# a) 具体化与行为追踪
# - 原题是“二维裁剪窗 -> 在 8 个二面体方向里找唯一匹配 -> 用两种输入色重绘完整 20x20 图案”。
# - 这次融合把“几何方向”和“极性”合并成 16 个逻辑候选：8 个几何变体，每个都允许两种前景解释。
# - 同时把基底图案改成严格 200/400 前景像素，并满足 `rot180(mask) == ~mask`。于是某个裁剪若按
#   `variant=rot0, fg=color_a` 成立，那么它也会按 `variant=rot180, fg=color_b` 成立。
# - 小追踪：原题里某个输入窗通常只命中 1 个 `(方向, 前景色解释, 偏移)`；融合后，同一输入窗稳定命中
#   2 个逻辑候选，但这 2 个候选重绘出的完整 20x20 彩色输出完全相同。solver 因此不能再断言
#   “原始匹配唯一”，而要断言“去重后的完整彩色输出唯一”。
# b) 复杂度校准
# - 这里只引入一个受控的新增自由度：极性歧义。没有额外对象、没有外部标记、没有第三种颜色。
# - 16 个逻辑候选不会全部互相竞争；由于 180 度旋转补集关系，它们成对坍缩成 8 个可渲染结果。
# - 这种“16 选 2 再坍缩”的结构在每个样本里都会重复出现，足够明显，但不会把任务变成难以读图的
#   大规模离散编码题。
# c) 隐式假设审查（针对原 main）
# - 原代码的 `matches.append((fg, bg, pattern))` + `assert len(matches) == 1` 依赖“前景极性唯一”。
#   50% 密度下这个前提失效：同一输入窗会稳定出现 2 个原始匹配。
# - 原代码默认“颜色互换只是错误分支”；现在颜色互换可能对应另一个合法的几何候选。
# - 原代码用的是原始 42.2% 密度基底；融合版必须改用 50% 密度且与 180 度补集耦合的新基底，否则
#   `16 候选但要去重` 这件事不会稳定发生。
# d) 放置合法性
# - 生成器仍采用 output-first：先选完整 20x20 输出，再在其内裁出输入。
# - 图案始终占满固定的 20x20 画布，因此没有对象间重叠问题；裁剪只需保证不越界。
# - 尺寸仍保持原 spec 的 `H in [8,16]`、`W in [8,12]`，偏移取所有合法位置，再用 solver 自检。
# e) 视觉可观察性
# - 从样例上，人类仍会看到“所有输入都像是同一张大图的局部”，这保留了原题的视觉主线。
# - 与原题不同的是，完整图案前景/背景各占一半，并且 180 度对置区域呈互补关系，因此不同样例里会
#   反复出现“换一种颜色解释也说得通”的现象。
# - 这种规律完全靠位置和形状可见，不依赖颜色编号大小。
# f) 可视化整洁度
# - 输入输出仍只有一个双色图案和一个矩形裁剪窗，没有叠加标记层，画面很干净。
# - 50% 密度让图案更均衡，但仍是一个连续的结构化纹理，不会变成噪声棋盘。
# g) 变化的可推断性
# - 颜色规则仍然简单：输出只复用输入中的两种颜色。
# - 哪种颜色是“逻辑前景”不再由颜色 id 决定，而由全局匹配关系决定；solver 通过枚举两种极性并在
#   完整彩色输出层去重来实现，generator 也只依赖这一结构性规则。
# h) 新 solver 假设审查及生成保证
# - 假设 1：输入恰好有两种颜色。生成器始终从双色完整输出裁剪，保证成立。
# - 假设 2：在 16 个逻辑候选上搜索后，合法样本会得到有限个原始命中，且去重后的 20x20 彩色输出唯一。
#   生成器显式检查 `raw_match_count == 2` 且 `len(outputs) == 1`。
# - 假设 3：`rot180(mask) == ~mask`。`_pattern_mask()` 内部断言该性质，防止基底退化回原题。


_TOP_HALF = (
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
)


def _pattern_mask() -> np.ndarray:
    top = np.array([[ch == "1" for ch in row] for row in _TOP_HALF], dtype=bool)
    bottom = np.flip(~top, axis=(0, 1))
    pattern = np.vstack([top, bottom])
    assert pattern.shape == (20, 20)
    assert int(pattern.sum()) == 200
    assert np.array_equal(np.rot90(pattern, 2), ~pattern)
    return pattern


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    assert len(variants) == 8
    assert len({variant.tobytes() for variant in variants}) == 8
    return variants


def _other_color(colors: list[int], fg: int) -> int:
    return int(colors[0] if int(colors[1]) == int(fg) else colors[1])


def _render(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _sample_logical_candidate(
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int, int]:
    # Sample the 16-way logical search space explicitly:
    # 8 geometric variants x 2 choices for which observed color paints True cells.
    color0, color1 = sorted(int(x) for x in rng.choice(Color.ALL_COLORS, size=2, replace=False))
    branch = int(rng.integers(16))
    geom_idx = branch // 2
    true_color_is_high = bool(branch % 2)

    fg = color1 if true_color_is_high else color0
    bg = color0 if true_color_is_high else color1
    pattern = _dihedral_variants(_pattern_mask())[geom_idx]
    return pattern, fg, bg, branch


def _dedupe_grids(grids: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for grid in grids:
        key = np.ascontiguousarray(grid).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(grid)
    return unique


def _candidate_outputs(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2

    variants = _dihedral_variants(_pattern_mask())
    rendered: list[np.ndarray] = []
    raw_match_count = 0

    # Logical search space = 8 geometries x 2 polarity interpretations.
    for variant in variants:
        for fg in colors:
            bg = _other_color(colors, int(fg))
            visible = g == int(fg)
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(variant[r : r + H, c : c + W], visible):
                        raw_match_count += 1
                        rendered.append(_render(variant, int(fg), int(bg)))
    return _dedupe_grids(rendered), raw_match_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from a crop when geometry and polarity are jointly ambiguous.

    Input structure:
    - The grid is a two-color rectangular crop cut from one dihedral view of a fixed 20x20 binary motif.
    - The hidden motif has exactly 50% foreground density, so either observed color may be the logical foreground.

    Rule:
    - Enumerate all 16 logical candidates formed by 8 geometric variants and 2 foreground/background interpretations.
    - For every raw match, repaint the entire 20x20 candidate with the same two observed colors.
    - Deduplicate those full colored reconstructions and return the single distinct output that remains.

    Key insight:
    - Because the balanced motif is designed so that a 180-degree rotation equals its binary complement, two different logical candidates can fit the same crop even though they collapse to the same final colored 20x20 image.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    outputs, _ = _candidate_outputs(g)
    assert len(outputs) == 1
    return outputs[0]


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    pattern, fg, bg, _ = _sample_logical_candidate(rng)
    output = _render(pattern, fg, bg)

    H = int(rng.integers(8, 17))
    W = int(rng.integers(8, 13))
    r = int(rng.integers(0, 21 - H))
    c = int(rng.integers(0, 21 - W))
    grid = output[r : r + H, c : c + W].copy()

    outputs, raw_match_count = _candidate_outputs(grid)
    assert raw_match_count == 2
    assert len(outputs) == 1
    assert np.array_equal(outputs[0], output)
    return grid, output, raw_match_count


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(100):
        grid, output, _ = _gen_attempt(rng)
        if np.array_equal(main(grid), output):
            return grid
    raise RuntimeError("failed to generate a valid combo sample after 100 attempts")
