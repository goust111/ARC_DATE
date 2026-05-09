from __future__ import annotations

# Variant of 269e22fb
# Dimension: D11 — 前景/背景极性消歧方式 (match_cond)
# Change: 匹配时同时搜索两种极性（color_a为前景或color_b为前景），由唯一匹配自动消歧... → 用角落像素指示极性
#
# design_analysis:
#   concretization_and_trace:
#     - 输入仍然是同一个 20x20 二值 motif 的一个矩形裁剪，且仍只有两种颜色。
#     - 变体新增约束：裁剪窗口左上角 `(0, 0)` 必为 motif 前景。于是左上角像素的颜色不再只是普通内容，
#       而是一个“控制像素”：它告诉 solver 哪个颜色应被解释为 True。
#     - 小例子：若输入左上角是 `8`，另一个颜色是 `2`，则 solver 只允许 `8` 作为前景去匹配布尔窗口。
#       原题会同时尝试 `8` 和 `2` 两种极性；本变体会直接锁定前景为 `8`。
#   complexity_calibration:
#     - 对原 motif 做枚举后可知：原题常用的 8x8 以上大窗口几乎总能在双极性下唯一匹配，D11 会失去作用。
#     - 因此生成器必须改采更小的窗口，并且只保留“固定左上角前景时唯一、反极性仍存在候选”的窗口。
#     - 为避免窗口过于退化，尺寸限制在一组紧凑的小矩形上，例如 `3x4, 3x5, 4x4, 5x3, 6x3` 等。
#   implicit_assumptions_in_original_main:
#     - `for fg in colors:` 依赖“极性未知，必须把两色都试一遍”；变体后该前提失效，必须删掉该分支搜索。
#     - `matches.append((fg, bg, pattern))` 原先允许极性成为匹配维度之一；现在极性由角落像素固定。
#     - `assert len(matches) == 1` 仍然保留，但它现在只针对“固定极性后的 orientation+offset 唯一性”。
#   placement_legality:
#     - 生成器仍采用 output-first：先选某个 dihedral variant，再选其合法裁剪窗口。
#     - 候选窗口全部来自 20x20 motif 内部，因此不会越界；窗口本身就是 output 的子矩形，也不存在重叠问题。
#     - 生成器还会再次调用 `main(input)` 做自验证，确保 solver 假设与样本构造一致。
#   visual_observability:
#     - 样例之间会反复出现同一规律：输入左上角颜色在输出中总是成为整幅 motif 的前景色。
#     - 旧规则对应的“靠唯一匹配自动判断极性”在这些样本上会失败或至少变得不充分，因此控制像素不是摆设。
#   visual_cleanliness:
#     - 输入仍只有两个颜色，没有新增噪声、标记或重叠图层；唯一新语义只附着在现有角落像素上。
#   color_inference:
#     - 不使用颜色 id 排序或数值关系。颜色只承担“前景/背景”两类角色，而这个角色由角落位置给出。

import random
from collections import Counter

import numpy as np

from arc_utils import colors_present


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

_ALLOWED_CROP_SIZES = (
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (3, 6),
    (6, 3),
    (3, 7),
    (7, 3),
    (4, 4),
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


def _window_key(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8), axis=None).tobytes()


def _candidate_catalog() -> list[tuple[int, int, int, int, int, int]]:
    """
    Precompute windows where:
    - the crop's top-left cell is motif foreground;
    - with that polarity fixed, the window matches exactly once;
    - the complemented window still appears somewhere, so the old two-polarity rule would not be enough.
    """
    variants = _dihedral_variants(_pattern_mask())
    candidates: list[tuple[int, int, int, int, int, int]] = []

    for H, W in _ALLOWED_CROP_SIZES:
        counts: Counter[bytes] = Counter()
        windows: list[tuple[int, int, int, np.ndarray, bytes]] = []
        for variant_idx, pattern in enumerate(variants):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    window = pattern[r : r + H, c : c + W]
                    key = _window_key(window)
                    counts[key] += 1
                    windows.append((variant_idx, r, c, window, key))

        for variant_idx, r, c, window, key in windows:
            opposite_count = int(counts[_window_key(~window)])
            has_bg = bool(np.any(~window))
            if bool(window[0, 0]) and has_bg and int(counts[key]) == 1 and opposite_count >= 1:
                candidates.append((H, W, variant_idx, r, c, opposite_count))

    assert candidates
    return candidates


_PATTERNS = _dihedral_variants(_pattern_mask())
_CANDIDATES = _candidate_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a visible crop into the shared 20x20 binary motif using the corner-indicated polarity.

    Input structure:
    - The grid is a two-color rectangular crop of one rotation/reflection of the same 20x20 motif.
    - The crop's top-left pixel is guaranteed to use the motif foreground color; the other color is the background.

    Rule:
    - Read the color at `(0, 0)` as the foreground color and convert the crop to a boolean mask with that polarity.
    - Match that mask against every possible window in every dihedral orientation of the motif.
    - Repaint the uniquely matched full 20x20 motif with the corner color as foreground and the other input color as background.

    Key insight:
    - The crop is still a local window of one fixed binary motif, but polarity is no longer inferred by trying both colors;
    - the upper-left cell is a control pixel that directly states which color should fill the motif in the output.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert 1 <= H <= 20 and 1 <= W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    fg = int(g[0, 0])
    bg = int(next(color for color in colors if int(color) != int(fg)))
    visible = g == int(fg)

    matches: list[np.ndarray] = []
    for pattern in _PATTERNS:
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                if np.array_equal(pattern[r : r + H, c : c + W], visible):
                    matches.append(pattern)

    assert len(matches) == 1
    return np.where(matches[0], int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    palette = list(range(10))

    for _ in range(100):
        H, W, variant_idx, r, c, opposite_count = random.choice(_CANDIDATES)
        fg, bg = random.sample(palette, 2)
        pattern = _PATTERNS[variant_idx]

        output = np.where(pattern, int(fg), int(bg)).astype(int)
        grid = output[r : r + H, c : c + W].copy()

        # This is the core D11 guarantee: the corner pixel encodes polarity,
        # while the opposite polarity would still have matched elsewhere.
        assert int(grid[0, 0]) == int(fg)
        assert opposite_count >= 1

        if np.array_equal(main(grid), output):
            return grid

    raise RuntimeError("failed to generate a valid D11 sample")
