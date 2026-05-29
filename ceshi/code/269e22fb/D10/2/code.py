# Variant of 269e22fb
# Dimension: D10 — 多步处理：匹配-重建流程 (multi_step)
# Change: 单步处理：匹配裁剪窗口到唯一的(变体, 偏移, 极性)，然后直接输出完整图案... → 匹配后二次变换
#
# Design analysis
# spec_like:
#   concrete_behavior:
#     - Input keeps the original task's physical form: a 2-color rectangular crop cut from one
#       dihedral variant of the fixed 20x20 binary motif.
#     - The crop is taken from the pre-transform motif, not from the final answer. This preserves
#       the original "identify variant + offset + polarity" subproblem.
#     - New second step: after reconstructing the full motif, inspect the visible crop width.
#       If the width is odd, rotate the full reconstruction 90 degrees counterclockwise.
#       If the width is even, keep the reconstruction unchanged.
#     - Small trace:
#       original: match a 9x10 crop -> recover variant V -> output V.
#       variant:  match the same 9x10 crop -> recover variant V -> width is even -> output V.
#       original: match a 9x11 crop -> recover variant V -> output V.
#       variant:  match the same 9x11 crop -> recover variant V -> width is odd -> output rot90(V).
#   complexity_calibration:
#     - Only one extra control variable is added: width parity. It has two repeated, visible states.
#     - The extra transform is geometric, not color-id-based, so humans can inspect it directly.
#     - The fixed motif is already asymmetric across 90-degree rotations, so the odd-width branch
#       produces a visibly different full image instead of a visually silent change.
#   implicit_assumptions_review:
#     - The original `colors_present` == 2 assumption still holds; generator preserves exactly two colors.
#     - The original exhaustive matching over both polarities and all dihedral variants still works
#       because the crop itself is unchanged by the second step.
#     - `assert len(matches) == 1` remains valid only if generation rejects ambiguous windows.
#       The generator therefore retries until the crop has a unique legacy match.
#     - The new branch condition reads `W % 2`, which is a direct visible property of the input.
#       No hidden dependence on color ids, sample index, or random state is introduced.
#   placement_validity:
#     - Crops are still sampled as axis-aligned windows fully inside a 20x20 motif variant.
#     - Because the post-match rotation is applied only after full reconstruction, no extra
#       placement constraints are needed beyond the original crop bounds and uniqueness check.
#   visual_observability:
#     - Width parity is easy to read from the input rectangle itself.
#     - The output difference is a whole-image reorientation, which is large and globally visible.
#     - A few even-width examples can still match the legacy output, but odd-width examples cannot;
#       this prevents full regression to the original one-step task while keeping the rule learnable.
#   visual_cleanliness:
#     - The task still uses one motif and two colors. No overlays, annotations, or extra markers are
#       introduced, so the grids remain as clean as the source task.
#   color_change_inference:
#     - Colors are preserved across both branches. The only new rule concerns geometry, which avoids
#       adding an extra color-mapping inference burden on top of the second processing step.
#   main_assumptions_after_change:
#     - `_find_unique_match` assumes the visible crop came from one unmodified dihedral variant.
#       `generate_input()` guarantees this by cropping before any post-match transform.
#     - `_post_transform` assumes the reconstructed canvas is square, which is fixed at 20x20 here.
#     - The odd-width branch assumes the motif is not 90-degree symmetric; the hard-coded pattern
#       satisfies this, so the branch is behaviorally distinct from the legacy solver.

from __future__ import annotations

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

_PALETTE = tuple(range(10))


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


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern.copy()))

    assert len(matches) == 1
    return matches[0]


def _post_transform(pattern: np.ndarray, crop_width: int) -> np.ndarray:
    if int(crop_width) % 2 == 1:
        return np.rot90(pattern, 1)
    return pattern


def _paint(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden 20x20 motif from the crop, then optionally rotate the reconstruction.

    Input structure:
    - The grid is still a two-color rectangular crop from one rotation/reflection of the shared 20x20 motif.
    - The crop itself is taken before any extra whole-image transform, so it only reveals a local window and its width.

    Rule:
    - Try each input color as foreground and find the unique motif orientation/offset whose window matches the crop.
    - Rebuild the full 20x20 motif in the matched foreground/background colors.
    - If the crop width is odd, rotate that full reconstruction 90 degrees counterclockwise; if the width is even, leave it unchanged.

    Key insight:
    - Solving now has two layers: first identify which hidden motif instance produced the crop, then use the crop width parity to decide whether the completed motif must be rotated.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = map(int, g.shape)
    fg, bg, pattern = _find_unique_match(g)
    transformed = _post_transform(pattern, w)
    return _paint(transformed, fg, bg)


def generate_input() -> np.ndarray:
    while True:
        fg, bg = map(int, np.random.choice(_PALETTE, size=2, replace=False))
        pattern = _VARIANTS[int(np.random.randint(len(_VARIANTS)))]

        H = int(np.random.randint(8, 17))
        W = int(np.random.randint(8, 13))
        r = int(np.random.randint(0, 20 - H + 1))
        c = int(np.random.randint(0, 20 - W + 1))

        full = _paint(pattern, fg, bg)
        inp = full[r : r + H, c : c + W].copy()

        try:
            m_fg, m_bg, matched = _find_unique_match(inp)
        except AssertionError:
            continue

        expected = _paint(_post_transform(pattern, W), fg, bg)
        if (m_fg, m_bg) != (fg, bg):
            continue
        if not np.array_equal(matched, pattern):
            continue
        if W % 2 == 1 and np.array_equal(expected, full):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
