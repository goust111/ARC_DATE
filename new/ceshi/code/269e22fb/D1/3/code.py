# Variant of 269e22fb
# Dimension: D1 — 匹配条件：裁剪窗口唯一性匹配方式 (match_cond)
# Change: 通过穷举8个二面体变体x2种前景/背景极性x所有偏移位置，找到唯一匹配裁剪窗口的(变体, 偏移, 极性)组合... → 边缘轮廓匹配代替全像素匹配

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

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


# Design analysis for the D1 variant
# a) Concretization and behavior trace:
#    - The 20x20 binary motif and its 8 dihedral orientations stay unchanged.
#    - The input is still one rectangular crop, but its solid regions are hidden:
#      only cells touching an opposite-valued 4-neighbor inside the crop keep the
#      original fg/bg color; every non-boundary cell is overwritten by one filler color.
#    - Example flow:
#      original crop window -> a full two-color patch with filled regions;
#      variant crop window -> the same patch converted to a three-value signature
#      {background-edge, foreground-edge, filler}. The solver can no longer use
#      area equality on the raw crop and must instead compare contour signatures.
# b) Complexity calibration:
#    - The contour still depends on orientation, polarity, and offset, so the task
#      keeps the same latent search space but exposes less information per crop.
#    - The original crop-size ranges remain viable because this motif's contour
#      windows are already >91% unique even at 8x8 and become 100% unique for the
#      larger original sizes.
# c) Old-code assumptions that break here:
#    - `assert len(colors) == 2` no longer holds; the filler introduces a third color.
#    - `visible = g == fg` assumed every cell directly encoded fg/bg membership;
#      that fails once most cells are replaced by filler.
#    - `np.array_equal(pattern_window, visible)` assumed full-pixel equality; the
#      new match must instead compare a derived contour signature from each window.
# d) Placement legality:
#    - Crops are still axis-aligned windows fully inside a 20x20 oriented motif.
#    - The generator rejects windows lacking fg-edge, bg-edge, or filler cells, so
#      every input really shows a two-sided contour embedded in a hidden interior.
#    - The generator also rejects any crop whose contour signature is not uniquely
#      solved by `main()`.
# e) Visual observability:
#    - Across examples, humans see the same style change every time: solid regions
#      are stripped out and replaced by one uniform third color, leaving only a
#      colored outline. That difference is immediate without relying on color ids.
# f) Visual neatness:
#    - The contour is one-cell thick and uses the original two motif colors, while
#      the filler is uniform, so the three roles remain visually separable.
# g) New solver assumptions and how generation enforces them:
#    - `main()` assumes exactly three colors, one being filler.
#    - It assumes both sides of the contour are visible somewhere in the crop.
#    - It assumes the contour signature matches exactly one
#      (orientation, offset, polarity, filler-choice) interpretation.
#    - `generate_input()` guarantees all three conditions before returning.


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


def _edge_mask(window: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(window, dtype=bool)
    vertical = window[:-1, :] != window[1:, :]
    horizontal = window[:, :-1] != window[:, 1:]
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    return edge


def _contour_signature(window: np.ndarray) -> np.ndarray:
    edge = _edge_mask(window)
    sig = np.full(window.shape, 2, dtype=np.int8)
    sig[edge & window] = 1
    sig[edge & ~window] = 0
    return sig


def _render_contour_input(window: np.ndarray, fg: int, bg: int, filler: int) -> np.ndarray:
    sig = _contour_signature(window)
    out = np.full(window.shape, int(filler), dtype=int)
    out[sig == 1] = int(fg)
    out[sig == 0] = int(bg)
    return out


def _candidate_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    matches: list[tuple[int, int, np.ndarray]] = []
    for filler in colors:
        edge_colors = [int(color) for color in colors if int(color) != int(filler)]
        if len(edge_colors) != 2:
            continue
        for fg in edge_colors:
            bg = int(edge_colors[0] if edge_colors[1] == fg else edge_colors[1])
            observed = np.full((H, W), 2, dtype=np.int8)
            observed[g == bg] = 0
            observed[g == fg] = 1
            for pattern in _VARIANTS:
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        if np.array_equal(_contour_signature(pattern[r : r + H, c : c + W]), observed):
                            matches.append((int(fg), int(bg), pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a contour-only crop of one dihedral orientation.

    Input structure:
    - The grid is a rectangular crop of the shared motif, but only pixels on a local fg/bg boundary keep motif colors.
    - Every non-boundary cell inside that crop is replaced by one uniform filler color, so the input uses exactly three colors.

    Rule:
    - Try each input color as the filler and each ordering of the other two colors as foreground/background edge labels.
    - For every dihedral orientation and crop offset of the canonical 20x20 mask, convert the candidate window into the same contour signature.
    - Use the unique contour match to repaint the entire 20x20 motif with the recovered foreground and background colors.

    Key insight:
    - Even after hiding region interiors, the two-sided outline geometry inside the crop still uniquely identifies the motif's orientation, offset, and polarity.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 3

    matches = _candidate_matches(g)
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(2000):
        fg, bg, filler = map(int, rng.choice(_PALETTE, size=3, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        H = int(rng.integers(8, 17))
        W = int(rng.integers(8, 13))
        r = int(rng.integers(0, 20 - H + 1))
        c = int(rng.integers(0, 20 - W + 1))

        window = pattern[r : r + H, c : c + W]
        contour_grid = _render_contour_input(window, fg, bg, filler)
        counts = {int(color): int(np.count_nonzero(contour_grid == color)) for color in (fg, bg, filler)}
        if min(counts.values()) == 0:
            continue

        expected = np.where(pattern, fg, bg).astype(int)
        try:
            solved = main(contour_grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return contour_grid

    raise RuntimeError("failed to generate a unique contour crop")
