# Variant of 269e22fb
# Dimension: D3 — 目标输出：完整图案的展示方式 (dst)
# Change: 输出完整的20x20图案，前景/背景用输入的两种颜色绘制... → 输出仅补全缺失部分

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# - source_task: None.
# - Concrete behavior:
#   The original task outputs a fully repainted 20x20 motif, so the observed crop
#   disappears into the final image. For D3 we need the crop to remain visible as
#   an already-present patch while only the missing area is completed.
# - Required adaptation for observability:
#   If the crop used the same rendering as the finished motif, "preserve the crop"
#   and "repaint the whole motif" would produce identical 20x20 outputs. To avoid
#   that degeneration, the generator renders the input crop as the color-inverted
#   view of the hidden motif window. The solver still recovers the same motif,
#   orientation, and offset, but it fills only the outside using the opposite
#   polarity and then pastes the raw crop back unchanged.
# - Small trace:
#   Original: match crop -> repaint all 400 cells with matched polarity.
#   Variant: match crop -> infer offset -> paint the 20x20 motif with the opposite
#   polarity -> overwrite the matched window with the original crop.
#   Result: the window becomes a visibly preserved negative patch, and the outside
#   is the completed normal motif.
# - Complexity calibration:
#   We keep the same fixed 20x20 asymmetric motif and the same 8 dihedral
#   orientations so the identification problem stays rich. Only the output display
#   and the crop rendering are changed.
# - Main() assumption audit:
#   `colors_present(g)` assumes exactly two colors; generator enforces that.
#   The window match loops assume the input is one exact rectangle from one motif
#   orientation under one polarity; generator constructs exactly that.
#   The new overwrite step assumes the matched offset is unique; generator uses
#   dimensions with max(H, W) >= 12, then still verifies by calling `main()`.
# - Placement legality:
#   The crop is chosen with 0 <= r <= 20-H and 0 <= c <= 20-W, so it never leaves
#   the 20x20 canvas. The overwrite uses the same rectangle, so no overlap issues
#   exist beyond that single window.
# - Visual clarity:
#   Output contains two layers only: a rectangular preserved patch and the
#   completion outside it. No extra markers, borders, or third colors are added.
# - Color inference:
#   Only two colors appear. The rule is structural and visible: the rectangle keeps
#   its local coloring while the missing exterior is completed with the opposite
#   polarity.


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


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(_pattern_mask()):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), r, c, pattern))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed a uniquely matching crop back into the 20x20 motif and fill only the missing exterior.

    Input structure:
    - The grid is a two-color rectangular crop from one dihedral orientation of the fixed 20x20 motif.
    - Inside the crop, the motif window is shown with the opposite polarity from the finished full image.

    Rule:
    - Test each input color as the crop foreground and locate the unique matching window across all motif orientations.
    - Build the full 20x20 motif using the opposite color assignment from that match.
    - Keep the matched crop rectangle exactly as given and only complete the cells outside that rectangle.

    Key insight:
    - The crop still identifies orientation and offset exactly as before, but the final image is not a full repaint: the observed rectangle remains a visible preserved patch, so the outside must be completed around it.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    fg, bg, r, c, pattern = _find_unique_match(g)

    # The matched polarity describes the crop itself; the completion outside the
    # crop uses the opposite mapping so the raw window remains visibly preserved.
    out = np.where(pattern, int(bg), int(fg)).astype(int)
    out[r : r + H, c : c + W] = g
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    while True:
        fg, bg = rng.sample(range(10), 2)
        pattern = variants[rng.randrange(len(variants))]

        H = rng.randint(12, 16)
        W = rng.randint(8, 12)
        r = rng.randint(0, 20 - H)
        c = rng.randint(0, 20 - W)

        finished = np.where(pattern, int(fg), int(bg)).astype(int)
        window = pattern[r : r + H, c : c + W]
        inp = np.where(window, int(bg), int(fg)).astype(int)

        expected = finished.copy()
        expected[r : r + H, c : c + W] = inp

        if np.array_equal(main(inp), expected):
            return inp
