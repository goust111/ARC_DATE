# Variant of 269e22fb
# Dimension: D5 — 图案结构：基础图案类型 (meta)
# Change: 硬编码的20x20固定二值图案，前景密度42.2%，8个二面体变体均不同且无互补关系... → 图案尺寸可变（非固定20x20）

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "solvers" / "arc_utils.py"
    if _candidate.exists():
        sys.path.append(str(_candidate.parent))
        break

from arc_utils import Color, colors_present


# Design analysis
# - Concretization:
#   The input stays a two-color rectangular crop, but the hidden full motif is no longer
#   always 20x20. Instead it is one member of a small size family: 15x15, 20x20, or 25x25.
#   Each full motif is a nearest-neighbor resize of the original 20x20 binary drawing, then
#   optionally rotated/reflected. The output is the whole resized motif in the matched
#   orientation, repainted using the two colors from the crop.
#   Example trace: in the original task a 9x10 crop only had to be matched inside one 20x20
#   motif family. In this variant the same crop must first be tested against 15x15, 20x20,
#   and 25x25 motif families, then against all dihedral variants and all offsets. The new
#   hidden variable is the full motif size.
# - Complexity calibration:
#   Sizes are limited to three reusable levels so the task is visibly different but not
#   combinatorially unbounded. Reusing the same motif identity across sizes keeps the rule
#   legible across a few examples, while cross-size uniqueness is enforced by rejection
#   sampling in the generator.
# - Implicit assumption audit:
#   The original `assert H <= 20 and W <= 20` relied on a fixed output size; now `main()`
#   only requires the crop to fit at least one candidate size. The original `range(20 - H + 1)`
#   / `range(20 - W + 1)` loops assumed a single canvas size; they are replaced with
#   size-specific ranges. The original hard-coded `_pattern_mask()` assumed one fixed motif;
#   now it becomes `_pattern_mask(size)`.
# - Placement legality:
#   The generator still uses output-first construction. It chooses a target size, orientation,
#   and crop offset that are always in bounds for that chosen full motif. No objects overlap
#   because there is still only one motif and one rectangular window.
# - Visual observability:
#   Training pairs would now show outputs of different square sizes while the local crop rule
#   remains otherwise stable. A human can notice that the crop always expands into the complete
#   motif, but the finished motif may be 15x15, 20x20, or 25x25 depending on the example.
# - Visual cleanliness:
#   The scene stays a single two-color rectangle with no overlays or extra markers, so the
#   variation is attributable specifically to motif scale.
# - Color inference:
#   Colors are unchanged from the original rule: exactly two observed colors, one mapped to
#   motif foreground and the other to motif background. No color-id-dependent logic is used.
# - New solver assumptions checked against the generator:
#   `main()` assumes exactly two colors, square candidate motif sizes from `_CANDIDATE_SIZES`,
#   and that one unique (size, orientation, offset, polarity) match exists. `generate_input()`
#   guarantees those conditions by rejection sampling and by verifying `main(inp) == output`.


_BASE_PATTERN_20 = (
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

_CANDIDATE_SIZES = (15, 20, 25)


def _canonical_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN_20], dtype=bool)


def _resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    row_idx = (np.arange(size) * mask.shape[0]) // size
    col_idx = (np.arange(size) * mask.shape[1]) // size
    return mask[row_idx][:, col_idx]


@lru_cache(maxsize=None)
def _pattern_mask(size: int) -> np.ndarray:
    assert size in _CANDIDATE_SIZES
    return _resize_binary_mask(_canonical_mask(), int(size))


@lru_cache(maxsize=None)
def _dihedral_variants(size: int) -> tuple[np.ndarray, ...]:
    base = _pattern_mask(size)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return tuple(variants)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full resized motif that produced the visible crop.

    Input structure:
    - The grid is a two-color rectangular window cut from one rotation/reflection of a shared binary motif family.
    - The hidden full motif is square, but its size is not fixed: it may be 15x15, 20x20, or 25x25.

    Rule:
    - Try each observed input color as the motif foreground.
    - For every candidate full size, compare the resulting boolean crop against every dihedral orientation and every legal crop offset.
    - Repaint the uniquely matched full motif using the matched foreground/background colors and return it at that inferred size.

    Key insight:
    - The crop is still a literal window from the final answer, but now the solver must infer which motif size generated it before completing the whole pattern.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)

    colors = colors_present(g)
    assert len(colors) == 2
    assert any(H <= size and W <= size for size in _CANDIDATE_SIZES)

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for size in _CANDIDATE_SIZES:
            if H > size or W > size:
                continue
            for pattern in _dihedral_variants(size):
                for r in range(size - H + 1):
                    for c in range(size - W + 1):
                        if np.array_equal(pattern[r : r + H, c : c + W], visible):
                            matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _random_colors() -> tuple[int, int]:
    fg, bg = random.sample(Color.ALL_COLORS, 2)
    return int(fg), int(bg)


def _random_crop_shape(size: int) -> tuple[int, int]:
    h_max = min(size - 3, 14)
    w_max = min(size - 3, 12)
    H = random.randint(8, h_max)
    W = random.randint(8, w_max)
    return H, W


def _sample_instance() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    for _ in range(400):
        size = random.choice(_CANDIDATE_SIZES)
        pattern = random.choice(_dihedral_variants(size))
        fg, bg = _random_colors()
        output = np.where(pattern, fg, bg).astype(int)

        H, W = _random_crop_shape(size)
        r = random.randint(0, size - H)
        c = random.randint(0, size - W)
        input_grid = output[r : r + H, c : c + W].copy()
        meta = {"size": size, "crop_h": H, "crop_w": W, "offset_r": r, "offset_c": c}
        try:
            solved = main(input_grid)
        except AssertionError:
            continue
        if np.array_equal(solved, output):
            return input_grid, output, meta
    raise RuntimeError("failed to sample a uniquely solvable variable-size instance")


def generate_input() -> np.ndarray:
    inp, _expected, _meta = _sample_instance()
    return inp
