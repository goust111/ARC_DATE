# Variant of 269e22fb
# Dimension: D15 — 图案前景密度 (meta)
# Change: 前景密度固定为42.2%（169/400），接近但不等于50%... → 恰好50%前景密度

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
# - 仍然是“从两色裁剪窗恢复固定 20x20 图案”的任务；输入/输出尺寸、二面体变换、双色重绘都保留。
# - D15 的核心改动是把图案改成恰好 200/400 个前景像素，并额外强制 rot180(mask) == ~mask。
#   因此完整图案在 180° 旋转后会变成颜色互补版；这让“把哪种颜色当成前景”不再能靠全图计数区分。
# - 小例子：若某裁剪来自变体 V，原题逻辑会得到唯一匹配 (V, fg=color_a)。现在同一裁剪还会出现
#   第二个原始匹配 (rot180(V), fg=color_b)。但把两种匹配各自重绘成 20x20 后，彩色结果完全相同，
#   所以 solver 应该从“唯一原始匹配”改成“唯一彩色输出”。
# b) 复杂度校准
# - 只增加一个受控自由度：极性歧义。没有额外颜色规则，也没有额外对象。
# - rot180 互补关系在所有样本中重复出现，既能稳定触发 D15，又不会引入多种互相竞争的判定机制。
# c) 隐式假设审查（针对原 main）
# - `colors_present(g)` 依赖“输入恰好两色”；新生成器继续保证这一点。
# - 原代码把每个匹配都记为 `(fg, bg, pattern)`，再断言 `len(matches)==1`。这隐含依赖“极性唯一”。
#   D15 下该前提失效：合法样本稳定会出现 2 个原始匹配，所以必须改成对重绘后的 20x20 结果去重。
# - `np.where(pattern, fg, bg)` 仍然正确，因为即便逻辑前景解释不同，只要最终彩色 20x20 相同，输出就唯一。
# d) 放置合法性
# - 生成器继续 output-first：先选二面体变体、两种颜色、裁剪尺寸和偏移，再直接从完整输出裁剪输入。
# - 图案固定为 20x20，不存在对象碰撞问题；裁剪只需满足不越界。
# e) 视觉可观察性
# - 输出仍是统一大图案，人类能从多组样例看出“每个小窗都属于同一 20x20 模式”。
# - 与原题不同的是，新图案上下两半存在明显的 180° 互补关系，完整输出也呈现更均衡的前景/背景分布。
# f) 可视化整洁度
# - 仍然只有一个双色图案和一个矩形裁剪窗，没有额外标记或叠加层，画面干净。
# g) 颜色变化可推断性
# - 颜色规则没变：输出只复用输入中的两种颜色，图案色/背景色由匹配后的完整重绘决定。
#   没有使用颜色 id 做隐藏排序，也没有引入不可见颜色映射。
# i) 新 solver 假设审查
# - `_candidate_outputs()` 假设：所有合法输入在枚举两种极性和全部二面体窗口后，去重后的彩色输出唯一。
#   `generate_input()` 用同一函数自检并强制 `raw_match_count == 2`，从构造上保证这一点且避免退化回原题。


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
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _other_color(colors: list[int], fg: int) -> int:
    return int(colors[0] if int(colors[1]) == int(fg) else colors[1])


def _render(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


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
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    rendered: list[np.ndarray] = []
    raw_match_count = 0
    for fg in colors:
        bg = _other_color(colors, int(fg))
        visible = g == int(fg)
        for pattern in variants:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        raw_match_count += 1
                        rendered.append(_render(pattern, int(fg), int(bg)))
    return _dedupe_grids(rendered), raw_match_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the full 20x20 balanced motif from a two-color crop of one of its dihedral/complement views.

    Input structure:
    - The grid is a rectangular window cut from one rotation/reflection of a fixed 20x20 binary motif.
    - Exactly two colors are present, but because the motif has 50% foreground density, either color may be the logical foreground.

    Rule:
    - Test both observed colors as the candidate foreground mask and compare that mask against every window in every dihedral orientation.
    - Turn each raw match into a full 20x20 colored reconstruction, then deduplicate those reconstructions by rendered pixels.
    - Return the single distinct 20x20 reconstruction that remains after resolving the polarity ambiguity.

    Key insight:
    - The motif is designed so that a 180-degree rotation is exactly its binary complement, so swapping foreground/background may create a second raw match without changing the final colored output.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    outputs, _ = _candidate_outputs(g)
    assert len(outputs) == 1
    return outputs[0]


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    colors = [int(x) for x in rng.choice(Color.ALL_COLORS, size=2, replace=False)]
    fg, bg = colors

    base = _pattern_mask()
    variants = _dihedral_variants(base)
    pattern = variants[int(rng.integers(len(variants)))]
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
    raise RuntimeError("failed to generate a valid D15 sample after 100 attempts")
