# Variant of 269e22fb
# Dimension: D10 — 多步处理：匹配-重建流程 (multi_step)
# Change: 单步处理：匹配裁剪窗口到唯一的(变体, 偏移, 极性)，然后直接输出完整图案... → 链式图案补全

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

def _find_solvers_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "solvers" / "arc_utils.py"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError("Could not locate solvers/arc_utils.py from this variant file")


_SOLVERS_DIR = _find_solvers_dir()
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import colors_present


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    - The hidden object is still the same fixed 20x20 binary motif under one of its 8 dihedral variants.
#    - Input: a two-color HxW crop A cut from variant V1 at offset (r, c).
#    - Output: not the whole 20x20 motif anymore. Instead, reconstruct V1, rotate the whole 20x20
#      motif 90 degrees counterclockwise to get V2, then return crop B = V2[r:r+H, c:c+W].
#    - Example trace:
#      original task: identify (V1, r, c, polarity) -> repaint full 20x20 V1.
#      new task:      identify (V1, r, c, polarity) -> derive V2 = rot90(V1) -> repaint only B.
#      The changed step is the final reconstruction target, not the matching phase.
# b) Complexity calibration:
#    - The core freedom stays small and visible: one fixed chain relation, namely whole-motif rot90.
#    - Crop sizes remain the original 8..16 by 8..12 ranges, so uniqueness behavior stays close to the
#      source task and the puzzle does not collapse into a trivial local rewrite.
# c) Hidden-assumption audit against the original main():
#    - `colors_present(g)` assuming exactly 2 colors still holds.
#    - Trying both colors as foreground still holds because polarity is still unknown.
#    - Exhaustive search over all dihedral variants and offsets still holds for recovering V1.
#    - The old code only kept `pattern`; this variant also needs `(r, c)` because the answer is a crop
#      from a linked variant at the same offset.
#    - The old final `np.where(pattern, fg, bg)` is no longer valid because output is not 20x20.
# d) Placement legality:
#    - Generator picks a dihedral variant, crop size, and offset fully inside the 20x20 canvas.
#    - It rejects any crop that is not uniquely solvable across all 8 variants and both polarities.
#    - The same offset is always legal in the rotated target because every target remains 20x20.
# e) Visual observability:
#    - Humans can still see one rectangular two-color window, but input and output now have the same
#      size while the internal bitmap changes in a consistent way across examples.
#    - The rule uses no color-id arithmetic; colors only encode foreground/background and are preserved.
# f) Visual cleanliness:
#    - No extra markers or overlaid decorations are introduced; the scene remains a single clean crop.
# g) Color inferability:
#    - Exactly two colors are used, unchanged from input to output. The only change is the binary mask.


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


_VARIANTS = _dihedral_variants(_pattern_mask())


def _render(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern, r, c))
    return matches


def _match_unique_crop(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, int]:
    matches = _find_matches(grid)
    assert len(matches) == 1
    return matches[0]


def _linked_crop(
    fg: int,
    bg: int,
    pattern: np.ndarray,
    r: int,
    c: int,
    H: int,
    W: int,
) -> np.ndarray:
    linked_pattern = np.rot90(pattern, 1)
    return _render(linked_pattern[r : r + H, c : c + W], fg, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a uniquely identifiable crop into the corresponding crop from a linked motif orientation.

    Input structure:
    - The grid is a two-color rectangular crop taken from one dihedral orientation of the fixed 20x20 motif.
    - The crop keeps the original offset of that hidden 20x20 canvas; only the surrounding cells are omitted.

    Rule:
    - Try each input color as the foreground mask and locate the unique matching (orientation, offset, polarity).
    - Reconstruct the hidden full motif in that orientation, then rotate the entire 20x20 motif 90 degrees counterclockwise.
    - Return the crop from the rotated motif at the same offset, repainted with the matched foreground/background colors.

    Key insight:
    - The visible patch is only the first step: it identifies where the crop came from, and the answer is the sibling crop cut from a linked global variant rather than the full motif itself.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    fg, bg, pattern, r, c = _match_unique_crop(g)
    return _linked_crop(fg, bg, pattern, r, c, H, W)


def generate_example(rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(1000):
        fg, bg = map(int, rng.choice(np.arange(10), size=2, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]

        H = int(rng.integers(8, 17))
        W = int(rng.integers(8, 13))
        r = int(rng.integers(0, 20 - H + 1))
        c = int(rng.integers(0, 20 - W + 1))

        input_grid = _render(pattern[r : r + H, c : c + W], fg, bg)
        matches = _find_matches(input_grid)
        if len(matches) != 1:
            continue

        matched_fg, matched_bg, matched_pattern, matched_r, matched_c = matches[0]
        output_grid = _linked_crop(matched_fg, matched_bg, matched_pattern, matched_r, matched_c, H, W)
        pixel_delta = int(np.count_nonzero(input_grid != output_grid))
        if pixel_delta == 0:
            continue

        solved = main(input_grid)
        assert np.array_equal(solved, output_grid)
        return input_grid, output_grid, {
            "height": H,
            "width": W,
            "row": r,
            "col": c,
            "pixel_delta": pixel_delta,
        }

    raise AssertionError("Failed to generate a unique linked-crop example")


def generate_input() -> np.ndarray:
    input_grid, _, _ = generate_example()
    return input_grid
