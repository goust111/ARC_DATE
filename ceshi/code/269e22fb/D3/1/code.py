from __future__ import annotations

# Variant of 269e22fb
# Dimension: D3 — 目标输出：完整图案的展示方式 (dst)
# Change: 输出完整的20x20图案，前景/背景用输入的两种颜色绘制... → 输出裁剪窗口的互补区域
#
# Design analysis:
# spec:
#   invariant:
#     - The canonical 20x20 binary motif and its 8 dihedral orientations stay unchanged.
#     - The input is still a two-color rectangular crop from one oriented motif, so the
#       recognition problem is the same as the original task.
#   d3_change:
#     - Original output: repaint the full matched 20x20 motif.
#     - Variant output: repaint only the cells outside the matched crop rectangle, and
#       clear the matched rectangle back to the motif background color.
#     - A crop strictly inside the motif yields a frame-with-hole; a crop touching exactly
#       one border yields a U-shaped remainder. The generator samples both cases.
#   flow_trace:
#     - If an input crop matches rows [r:r+H) and cols [c:c+W) of one oriented motif,
#       the old solver returned the whole 20x20 orientation.
#     - The new solver returns that same orientation with the matched window erased,
#       so the output represents the unknown complement rather than the known crop.
#   complexity_calibration:
#     - No extra object types or new transformations are introduced; only the display
#       target changes after the same unique match is found.
#     - The generator keeps border-contact count in {0,1} so the visual result remains a
#       readable frame/U-shape rather than arbitrary corner-trimmed cases.
#   implicit_assumptions_review:
#     - The input still has exactly 2 colors, and one polarity/orientation/offset match.
#     - Unlike the original task, the matched offset must be retained because it determines
#       which rectangle is erased in the output.
#     - Erasing to the matched background color keeps the task two-color; the generator
#       ensures the crop contains both motif values, so some foreground is always removed.
#   placement_legality:
#     - Crop sizes remain within the original bounds: 8<=H<=16 and 8<=W<=12.
#     - Interior mode chooses 1<=r<=20-H-1 and 1<=c<=20-W-1.
#     - Edge mode touches exactly one border while staying interior on the other axis.
#   observability:
#     - Across examples, the inputs look like the original task, but the outputs are no
#       longer completed motifs: they are the remaining frame/U region after deleting the
#       observed rectangle. This is visible without relying on color ids or hidden state.

import random

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


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_ALL_PATTERNS = _dihedral_variants(_pattern_mask())


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _ALL_PATTERNS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern, int(r), int(c)))

    assert len(matches) == 1
    return matches[0]


def _render_complement(
    pattern: np.ndarray, fg: int, bg: int, r: int, c: int, H: int, W: int
) -> np.ndarray:
    keep = pattern.copy()
    keep[int(r) : int(r + H), int(c) : int(c + W)] = False
    return np.where(keep, int(fg), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the visible crop inside the shared motif, then output only the unseen remainder.

    Input structure:
    - The grid is a two-color rectangular crop from one rotation/reflection of the same 20x20 binary motif.
    - One input color marks motif foreground and the other marks motif background, but the polarity is not pre-labeled.

    Rule:
    - Test each input color as the candidate foreground and compare the crop against every offset in every dihedral motif variant.
    - Keep the unique match, including its orientation and crop position.
    - Repaint the matched 20x20 motif with the input colors, but erase the matched crop rectangle so only the complementary region remains.

    Key insight:
    - The local crop still uniquely identifies the full motif placement; the variant changes only the final rendering step, from completion to subtraction of the observed window.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    fg, bg, pattern, r, c = _find_unique_match(g)
    return _render_complement(pattern, fg, bg, r, c, H, W)


def _choose_offsets(H: int, W: int) -> tuple[int, int, str]:
    mode = random.choice(("frame", "u_shape"))
    if mode == "frame":
        r = random.randint(1, 20 - H - 1)
        c = random.randint(1, 20 - W - 1)
        return int(r), int(c), mode

    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return 0, random.randint(1, 20 - W - 1), mode
    if side == "bottom":
        return 20 - H, random.randint(1, 20 - W - 1), mode
    if side == "left":
        return random.randint(1, 20 - H - 1), 0, mode
    return random.randint(1, 20 - H - 1), 20 - W, mode


def _sample_case() -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    fg, bg = random.sample(range(10), 2)
    pattern = random.choice(_ALL_PATTERNS)
    H = random.randint(8, 16)
    W = random.randint(8, 12)
    r, c, mode = _choose_offsets(H, W)

    inp = np.where(pattern[r : r + H, c : c + W], int(fg), int(bg)).astype(int)
    if len(colors_present(inp)) != 2:
        raise AssertionError("crop lost one of the two motif values")

    expected = _render_complement(pattern, fg, bg, r, c, H, W)
    if np.array_equal(expected, np.where(pattern, int(fg), int(bg)).astype(int)):
        raise AssertionError("variant degenerated to the original full-motif output")

    meta = {
        "mode": mode,
        "removed_fg": int(pattern[r : r + H, c : c + W].sum()),
        "border_contacts": int(r == 0) + int(r + H == 20) + int(c == 0) + int(c + W == 20),
        "height": int(H),
        "width": int(W),
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(1000):
        try:
            inp, expected, _meta = _sample_case()
            if not np.array_equal(main(inp), expected):
                continue
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a uniquely solvable D3 variant input")
