# Combo variant of 269e22fb (index 25)
# combo_spec: D3/0+D4/0+D10/2
# Name: 条件反转补全输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# - Concrete fused behavior:
#   The crop still comes from one dihedral variant of the fixed 20x20 motif, but
#   the generator shows that crop in the opposite polarity from the hidden motif.
#   After matching, the solver does not repaint the whole board blindly. It keeps
#   the raw crop rectangle unchanged and fills only the outside area. The outside
#   fill then depends on the visible crop width:
#   * odd width  -> invert the completion polarity relative to the crop match
#   * even width -> keep the completion polarity aligned with the crop match
# - Small trace:
#   Suppose the hidden motif window is a 12x9 patch with True->red, False->blue.
#   The input crop is rendered inverted, so the visible 12x9 patch becomes
#   True->blue, False->red.
#   Original solver: match that crop as foreground=blue, then output the full
#   20x20 motif entirely in blue/red.
#   This combo:
#   1. match the same crop and recover (variant, offset, crop polarity)
#   2. notice width=9 is odd
#   3. fill only the exterior with the opposite polarity, i.e. True->red,
#      False->blue
#   4. paste the original crop back unchanged
#   Result: the outer area shows the completed motif, while the crop remains as a
#   visible negative rectangle. If the width were 10 instead, step 3 would keep
#   the exterior in blue/red and the final image would be a coherent full
#   inverted motif with no polarity break across the crop boundary.
# - Complexity calibration:
#   The task still has one fixed asymmetric motif, eight dihedral orientations,
#   two colors, and one explicit branch bit from width parity. This keeps the
#   identification problem rich without adding hidden state or too many cases.
#   Generator sampling is parity-balanced so both branches appear repeatedly.
# - Implicit assumption review:
#   `colors_present(g)` assumes exactly two colors; generator enforces exactly two.
#   The exhaustive match assumes the crop is a literal axis-aligned window from a
#   single motif orientation under one polarity; generator constructs exactly that.
#   `assert len(matches) == 1` assumes the crop uniquely identifies orientation,
#   offset, and crop polarity; generator uses H in [12,16], which already gives
#   max(H, W) >= 12, then still verifies the unique match.
#   The conditional completion assumes width parity is directly readable from the
#   input and independent of color ids; generator chooses width parity explicitly.
#   The overwrite step assumes the matched rectangle lies within the 20x20 board;
#   generation uses 0 <= r <= 20-H and 0 <= c <= 20-W.
# - Placement legality:
#   There is only one rectangular crop. It is always sampled fully inside the
#   20x20 motif, so there is no overlap or boundary conflict beyond preserving
#   that same rectangle during output assembly.
# - Visual observability:
#   Width parity is visible from the rectangle size itself. Odd-width examples
#   show a preserved negative patch against an opposite-polarity exterior, while
#   even-width examples show the same crop completed seamlessly. Because the odd
#   branch changes a large exterior region, humans can infer the conditional rule
#   from a few examples without relying on color ids.
# - Visual cleanliness:
#   The output still uses only the two crop colors and one rectangular preserved
#   region. No markers, borders, or extra objects are added, so the composition
#   remains easy to parse.
# - Color-rule inference:
#   The only color logic is a binary switch controlled by width parity, and both
#   colors participate symmetrically. There is no hidden mapping from specific
#   color numbers to behavior.
# - Main/generator contract after the change:
#   `_find_unique_match` returns the crop polarity and placement. `main()` uses
#   that result plus `W % 2` to choose how to paint the exterior, then restores
#   the raw crop. `generate_input()` builds crops in the inverted crop polarity,
#   checks uniqueness, verifies `main(inp)`, and rejects odd-width samples only if
#   they accidentally fail to create the intended visible polarity contrast.


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
_NEXT_ODD_WIDTH = False


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), r, c, pattern.copy()))

    assert len(matches) == 1
    return matches[0]


def _paint(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _completion_canvas(pattern: np.ndarray, crop_width: int, fg: int, bg: int) -> np.ndarray:
    if int(crop_width) % 2 == 1:
        return _paint(pattern, bg, fg)
    return _paint(pattern, fg, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the crop to the hidden motif, then complete only the missing exterior with a parity-controlled polarity rule.

    Input structure:
    - The grid is a two-color rectangular crop from one dihedral orientation of the fixed 20x20 binary motif.
    - The visible crop is shown in the opposite polarity from the hidden full motif, and its width is the branch signal.

    Rule:
    - Try each input color as the crop foreground and find the unique motif orientation and offset whose window matches the crop.
    - Use the crop width parity to choose the exterior completion polarity: odd widths flip foreground/background for the completion, even widths keep the matched polarity.
    - Paint the full 20x20 canvas with that chosen completion polarity, then restore the matched crop rectangle exactly as given so only the outside is synthesized.

    Key insight:
    - Solving has three linked stages: identify the underlying motif instance, inspect the visible rectangle width, and then decide whether the missing area should continue the crop polarity or invert around it.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    fg, bg, r, c, pattern = _find_unique_match(g)
    out = _completion_canvas(pattern, W, fg, bg)
    out[r : r + H, c : c + W] = g
    return out


def generate_input() -> np.ndarray:
    global _NEXT_ODD_WIDTH
    rng = random.Random()
    target_odd_width = bool(_NEXT_ODD_WIDTH)

    while True:
        fg, bg = rng.sample(_PALETTE, 2)
        pattern = _VARIANTS[rng.randrange(len(_VARIANTS))]

        H = rng.randint(12, 16)
        W = rng.choice((9, 11)) if target_odd_width else rng.choice((8, 10, 12))
        r = rng.randint(0, 20 - H)
        c = rng.randint(0, 20 - W)

        window = pattern[r : r + H, c : c + W]
        inp = np.where(window, int(bg), int(fg)).astype(int)

        try:
            m_fg, m_bg, m_r, m_c, matched = _find_unique_match(inp)
        except AssertionError:
            continue

        if (m_fg, m_bg) != (int(bg), int(fg)):
            continue
        if (m_r, m_c) != (r, c):
            continue
        if not np.array_equal(matched, pattern):
            continue

        expected = _completion_canvas(pattern, W, m_fg, m_bg)
        expected[r : r + H, c : c + W] = inp

        if W % 2 == 1:
            same_polarity = _paint(pattern, m_fg, m_bg)
            same_polarity[r : r + H, c : c + W] = inp
            if np.array_equal(expected, same_polarity):
                continue

        if not np.array_equal(main(inp), expected):
            continue
        _NEXT_ODD_WIDTH = not target_odd_width
        return inp
