# Combo variant of 269e22fb (index 2)
# combo_spec: D1/1+D11/2
# Name: 噪声容错匹配加角落极性提示
#
# Design analysis:
# - concretization_and_trace:
#   - 输入仍是同一个 20x20 固定二值 motif 的某个二面体变体的矩形裁剪，且只出现两种 ARC 颜色。
#   - 融合后有两条新约束同时生效：
#     1. 裁剪中的 1-3 个非角落像素会被翻转，必须做最小汉明距离匹配，不能再要求精确相等。
#     2. 左上角像素 `(0, 0)` 保证是真实前景，因此极性不再通过“两色都试一遍”来消歧，而是先由角落直接锁定。
#   - 一个具体追踪：设某个 `3x6` 干净窗口的左上角为前景色 `8`，其余位置里有背景 `2`；生成器再翻转两个非角落像素。
#     原题逻辑会在 `np.array_equal(...)` 上全部失败；D1-only 逻辑会同时尝试 `8` 和 `2` 两种极性，
#     并可能在反极性下找到同样距离为 2 的补窗口候选；融合逻辑则先读出 `8` 是前景，只在该极性下枚举
#     orientation+offset，并选出唯一最小汉明距离候选。
# - complexity_calibration:
#   - 若沿用原题常见的 `8x8` 以上大窗口，角落极性几乎不会真正起作用，因为双极性最近邻通常已经唯一。
#   - 因此生成器改用一组紧凑小窗口尺寸 `(3x4..3x8, 4x4 及转置)`；这些窗口仍有足够结构可见，同时更容易
#     让“正确窗口唯一，但补窗口也出现过”这一现象成立。
#   - 生成时不仅要求固定角落极性后的最近邻唯一，还要求若恢复旧版双极性最近邻搜索，则最优集合里必须包含
#     反极性候选。这样角落控制像素就不是摆设。
# - implicit_assumption_audit_for_main:
#   - `colors_present(g)` 仍假设恰有两色；生成器始终从两种颜色绘制并只在二者间翻转像素。
#   - `fg = g[0, 0]` 新增了“左上角一定未被噪声污染且确属前景”的假设；生成器明确禁止翻转 `(0, 0)`。
#   - 最近邻搜索现在只在固定极性下进行，因此 `for fg in colors:` 这一旧前提已失效，必须删除。
#   - `assert len(matches) == 1` 不再表示“双极性全局唯一”，而是表示“角落极性固定后，orientation+offset 的最小距离候选唯一”。
# - placement_legality:
#   - 所有窗口仍来自 20x20 motif 内的合法偏移；噪声只作用于窗口内部，且不会越界，也不会引入额外对象。
# - visual_observability:
#   - 人类可直接看到两个信号：左上角颜色在输出中始终成为整幅 motif 前景；窗口内部又有少数看起来不合局部规律的错色点。
#   - 多组样例里，若忽略角落像素，就会遇到反极性同距候选；若忽略噪声，又无法精确匹配。因此两条改动都能被观察到。
# - visual_cleanliness:
#   - 场景仍只有一个双色矩形，没有额外标记层；角落语义复用现有像素，噪声数量也限制在 1-3，避免画面过脏。
# - inference_of_change:
#   - 颜色值大小、排序、黑色特殊地位都不参与推理。颜色只承载“前景/背景”二元角色，而角色由角落位置给出。

from __future__ import annotations

from collections import Counter
from itertools import count

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

_ARC_COLORS = tuple(range(10))
_ALLOWED_CROP_SIZES = (
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (3, 6),
    (6, 3),
    (3, 7),
    (7, 3),
    (3, 8),
    (8, 3),
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


def _other_color(colors: list[int], color: int) -> int:
    for candidate in colors:
        if int(candidate) != int(color):
            return int(candidate)
    raise AssertionError("expected exactly two distinct colors")


def _paint_pattern(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _scored_candidates_for_fg(grid: np.ndarray, fg: int) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    visible = g == int(fg)

    scored: list[tuple[int, int, int, int]] = []
    for variant_idx, pattern in enumerate(_VARIANTS):
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                window = pattern[r : r + H, c : c + W]
                dist = int(np.count_nonzero(window != visible))
                scored.append((variant_idx, r, c, dist))
    return scored


def _best_candidates_for_fg(grid: np.ndarray, fg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    scored = _scored_candidates_for_fg(grid, fg)
    best_dist = min(item[3] for item in scored)
    best = [item for item in scored if item[3] == best_dist]
    return best_dist, best


def _best_dual_polarity_candidates(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int, int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g)
    assert len(colors) == 2

    scored: list[tuple[int, int, int, int, int, int]] = []
    for fg in colors:
        bg = _other_color(colors, int(fg))
        for variant_idx, r, c, dist in _scored_candidates_for_fg(g, int(fg)):
            scored.append((int(fg), int(bg), variant_idx, r, c, dist))

    best_dist = min(item[5] for item in scored)
    best = [item for item in scored if item[5] == best_dist]
    return best_dist, best


def _candidate_catalog() -> list[tuple[int, int, int, int, int, int]]:
    """
    Precompute clean windows where:
    - the crop's top-left cell is motif foreground;
    - with polarity fixed by that corner, the clean window is exact-unique;
    - the complemented window also exists somewhere, so old two-polarity matching would stay viable.
    """
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for H, W in _ALLOWED_CROP_SIZES:
        counts: Counter[bytes] = Counter()
        windows: list[tuple[int, int, int, np.ndarray, bytes]] = []
        for variant_idx, pattern in enumerate(_VARIANTS):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    window = pattern[r : r + H, c : c + W]
                    key = _window_key(window)
                    counts[key] += 1
                    windows.append((variant_idx, r, c, window, key))

        for variant_idx, r, c, window, key in windows:
            complement_count = int(counts[_window_key(~window)])
            has_bg = bool(np.any(~window))
            if bool(window[0, 0]) and has_bg and int(counts[key]) == 1 and complement_count >= 1:
                candidates.append((H, W, variant_idx, r, c, complement_count))

    assert candidates
    return candidates


_VARIANTS = _dihedral_variants(_pattern_mask())
_CANDIDATES = _candidate_catalog()
_NOISE_COUNT_CYCLE = count()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from a noisy crop using corner-fixed polarity and nearest matching.

    Input structure:
    - The grid is a two-color rectangular crop from one rotation/reflection of the shared 20x20 binary motif.
    - The crop may contain 1-3 flipped cells, but its top-left pixel is guaranteed to be a true motif-foreground cell.

    Rule:
    - Read the color at `(0, 0)` as the motif foreground and the other visible color as background.
    - Compare that fixed-polarity boolean crop against every valid window in every dihedral variant using Hamming distance.
    - Take the unique minimum-distance candidate and repaint its entire 20x20 motif with the corner color as foreground.

    Key insight:
    - The corner pixel resolves polarity before any search happens, and the remaining mismatch is explained as small local noise;
    - solving requires combining both cues, not merely exact matching and not merely trying both polarities.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert 1 <= H <= 20 and 1 <= W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    fg = int(g[0, 0])
    bg = _other_color(colors, fg)
    _, matches = _best_candidates_for_fg(g, fg)
    assert len(matches) == 1

    variant_idx, _, _, _ = matches[0]
    return _paint_pattern(_VARIANTS[variant_idx], fg, bg)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    target_noise_count = 1 + (next(_NOISE_COUNT_CYCLE) % 3)

    for _ in range(5000):
        H, W, variant_idx, r, c, _ = _CANDIDATES[int(rng.integers(len(_CANDIDATES)))]
        fg, bg = map(int, rng.choice(_ARC_COLORS, size=2, replace=False))
        pattern = _VARIANTS[variant_idx]
        output = _paint_pattern(pattern, fg, bg)
        clean_crop = output[r : r + H, c : c + W].copy()

        assert int(clean_crop[0, 0]) == int(fg)

        max_noise = min(3, H * W - 1)
        if target_noise_count > max_noise:
            continue
        noise_count = target_noise_count
        noisy_crop = clean_crop.copy()
        flip_ids = rng.choice(np.arange(1, H * W), size=noise_count, replace=False)
        for flat_idx in flip_ids:
            rr, cc = divmod(int(flat_idx), W)
            noisy_crop[rr, cc] = bg if int(noisy_crop[rr, cc]) == fg else fg

        best_dist, fixed_best = _best_candidates_for_fg(noisy_crop, fg)
        if best_dist != noise_count or len(fixed_best) != 1:
            continue

        matched_variant_idx, matched_r, matched_c, _ = fixed_best[0]
        if (matched_variant_idx, matched_r, matched_c) != (variant_idx, r, c):
            continue

        dual_best_dist, dual_best = _best_dual_polarity_candidates(noisy_crop)
        if dual_best_dist != noise_count:
            continue

        wrong_polarity_best = [item for item in dual_best if item[0] == bg]
        if not wrong_polarity_best:
            continue

        if not any(
            not np.array_equal(_paint_pattern(_VARIANTS[item[2]], item[0], item[1]), output)
            for item in wrong_polarity_best
        ):
            continue

        if not np.array_equal(main(noisy_crop), output):
            continue

        return noisy_crop

    raise RuntimeError("failed to generate a noisy corner-guided crop")
