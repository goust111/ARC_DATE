# Combo variant of 269e22fb (index 3)
# combo_spec: D3/0+D4/1
# Name: 补全缺失区域并三色标记窗口

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# - Fusion summary:
#   The input stays a normal two-color crop from one dihedral orientation of the
#   fixed 20x20 motif. The fused output no longer redraws that crop with motif
#   colors. Instead, it completes only the missing exterior with the input
#   foreground/background colors and replaces the entire matched crop rectangle
#   with one fixed third color. This combines D3's "outside-only completion"
#   semantics with D4's "window becomes visibly marked" semantics.
# - Concrete behavior trace:
#   Suppose a small analog motif is 5x5 and the input is a 2x3 crop from rows
#   1:3 and cols 2:5.
#   Original task:
#     match crop -> repaint all 25 cells with the crop's fg/bg colors.
#   D3 alone:
#     match crop -> keep the crop appearance -> fill only the outside cells.
#   D4 alone:
#     match crop -> repaint all 25 cells -> recolor only the crop-covered motif
#     foreground cells to a third color.
#   This combo:
#     match crop -> repaint the full 20x20 motif normally outside the crop ->
#     overwrite the entire matched HxW rectangle with the marker color.
#   The difference is visible in one glance: the output exposes both the
#   completed hidden area and the exact rectangular known region.
# - Complexity calibration:
#   We keep the same hard-coded asymmetric motif, the same 8 dihedral
#   orientations, and the same rectangular-window search space. Only one new
#   visual state is added: a single solid marker rectangle. To prevent weak or
#   ambiguous cases, the generator requires the input crop itself to contain both
#   motif states and also requires the exterior completion to show both motif
#   states, so all three output color roles are observable.
# - Implicit assumption audit:
#   `colors_present(g)` requires exactly two input colors; the generator samples
#   two distinct non-marker colors.
#   The matching loops assume the input is an exact crop from one motif variant
#   under one polarity; the generator constructs precisely that.
#   The uniqueness assert assumes the crop dimensions are informative enough; the
#   generator uses heights 12-16 so max(H, W) is always at least 12, then still
#   self-checks with `main()`.
#   The fused output now depends on offset as well as orientation, so
#   `_find_unique_match()` must return `(top, left)` instead of only `pattern`.
#   The rectangle-overwrite step assumes 0 <= top <= 20-H and 0 <= left <= 20-W;
#   the generator samples only legal placements.
# - Placement legality:
#   There is still only one object family, the 20x20 motif. The rectangular crop
#   is always chosen wholly inside that canvas, so the later marker overwrite is
#   guaranteed in-bounds and introduces no overlaps beyond the intended one
#   rectangle.
# - Visual observability:
#   Across examples, the input is always a two-color crop while the output is
#   always a 20x20 grid with three semantic color roles:
#     input-foreground color = normal motif foreground outside the window
#     input-background color = normal motif background outside the window
#     fixed third color = the known crop region itself
#   Because the marker covers a full rectangle matching the crop size, humans can
#   infer the rule from a few examples without relying on numeric color ids.
# - Visual cleanliness:
#   The marked region is a single uniform block, not a mixture of marker and
#   motif cells. That avoids a noisy overlay and makes the "known vs completed"
#   split much easier to read.
# - Color reasoning:
#   The marker color never appears in the input, so it is unambiguously a new
#   semantic role rather than another possible crop polarity.
# - Post-change assumption review:
#   Every search loop in `_find_unique_match()` still reasons only over the
#   binary crop structure. The new output logic additionally assumes the matched
#   rectangle should be marked wholesale, so the generator enforces that the
#   window contains both foreground and background cells; otherwise this combo
#   would visually collapse toward the D4-only "mark foreground" behavior.


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

MARKER_COLOR = 2
_INPUT_COLOR_POOL = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2
    assert int(MARKER_COLOR) not in colors

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(_pattern_mask()):
            for top in range(20 - H + 1):
                for left in range(20 - W + 1):
                    if np.array_equal(pattern[top : top + H, left : left + W], visible):
                        matches.append((int(fg), int(bg), pattern, int(top), int(left)))

    assert len(matches) == 1
    return matches[0]


def _render_output(
    pattern: np.ndarray, fg: int, bg: int, top: int, left: int, height: int, width: int
) -> np.ndarray:
    out = np.where(pattern, int(fg), int(bg)).astype(int)
    out[top : top + height, left : left + width] = int(MARKER_COLOR)
    return out


def _random_colors() -> tuple[int, int]:
    fg, bg = random.sample(_INPUT_COLOR_POOL, 2)
    return int(fg), int(bg)


def _window_counts(pattern: np.ndarray, top: int, left: int, height: int, width: int) -> dict[str, int]:
    window = pattern[top : top + height, left : left + width]
    area = int(height * width)
    window_fg = int(window.sum())
    window_bg = int(area - window_fg)

    outside = np.ones(pattern.shape, dtype=bool)
    outside[top : top + height, left : left + width] = False
    outside_fg = int(pattern[outside].sum())
    outside_bg = int(outside.sum() - outside_fg)

    return {
        "area": area,
        "window_fg": window_fg,
        "window_bg": window_bg,
        "outside_fg": outside_fg,
        "outside_bg": outside_bg,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden 20x20 motif, complete only the unseen exterior, and mark the known crop area with a third color.

    Input structure:
    - The grid is a two-color rectangular crop from one rotation/reflection of the fixed 20x20 binary motif.
    - One input color represents motif foreground and the other represents motif background; the marker color never appears in the input.

    Rule:
    - Try each input color as the possible motif foreground and align that binary crop against every legal window in every dihedral orientation of the motif.
    - Once the crop's orientation and offset are uniquely identified, render the full 20x20 motif with the matched input foreground/background colors.
    - Replace the entire matched crop rectangle with a fixed third color, so only the cells outside the rectangle show the completed motif colors.

    Key insight:
    - The crop still serves as a locator for the hidden global orientation and placement, but the output no longer redraws that known patch structurally. Instead, the patch becomes a solid marker block, and the reasoning burden shifts to completing everything around it.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    fg, bg, pattern, top, left = _find_unique_match(g)
    return _render_output(pattern, fg, bg, top, left, H, W)


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    variants = _dihedral_variants(_pattern_mask())

    for _ in range(512):
        fg, bg = _random_colors()
        pattern = random.choice(variants)

        height = random.randint(12, 16)
        width = random.randint(8, 12)
        top = random.randint(0, 20 - height)
        left = random.randint(0, 20 - width)

        counts = _window_counts(pattern, top, left, height, width)
        area = counts["area"]

        if counts["window_fg"] < max(12, area // 5):
            continue
        if counts["window_bg"] < max(12, area // 5):
            continue
        if counts["outside_fg"] < 20:
            continue
        if counts["outside_bg"] < 20:
            continue

        window = pattern[top : top + height, left : left + width]
        inp = np.where(window, int(fg), int(bg)).astype(int)
        expected = _render_output(pattern, fg, bg, top, left, height, width)

        assert len(colors_present(inp)) == 2
        assert len(colors_present(expected)) == 3

        if not np.array_equal(main(inp), expected):
            continue

        full = np.where(pattern, int(fg), int(bg)).astype(int)
        changed_vs_original = int(np.count_nonzero(expected != full))
        changed_bg_inside_window = int(np.count_nonzero(window == 0))
        assert changed_vs_original == area
        assert changed_bg_inside_window == counts["window_bg"]
        assert changed_bg_inside_window > 0

        stats = {
            "height": int(height),
            "width": int(width),
            "area": int(area),
            "window_fg": int(counts["window_fg"]),
            "window_bg": int(counts["window_bg"]),
            "outside_fg": int(counts["outside_fg"]),
            "outside_bg": int(counts["outside_bg"]),
            "changed_vs_original": int(changed_vs_original),
            "changed_bg_inside_window": int(changed_bg_inside_window),
        }
        return inp, expected, stats

    raise RuntimeError("failed to generate a unique and visibly non-degenerate fused crop")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
