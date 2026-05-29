# Combo variant of 269e22fb (index 13)
# combo_spec: D5/2+D7/2
# Name: 尺寸双重编码推理

from __future__ import annotations

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
# - Concretization and behavior trace:
#   The visible input is still a two-color rectangular crop, but now two different size ideas
#   must be read from it at once.
#   1. Crop content identifies which full square motif family member was sampled: 15x15,
#      20x20, or 25x25, plus its dihedral orientation and offset.
#   2. Crop dimensions then encode a second instruction applied after recovery:
#      * width 8/12 -> rotate 0 times
#      * width 9    -> rotate 1 time
#      * width 10   -> rotate 2 times
#      * width 11   -> rotate 3 times
#      * odd height -> after rotation, do a left-right flip
#      * even height -> no flip
#   Example trace:
#   - Original task: a 13x11 crop is matched inside one fixed 20x20 motif, and that matched
#     full motif is returned directly.
#   - This combo: the same style of crop is matched across 15x15/20x20/25x25 resized motif
#     families. Suppose it uniquely matches a 25x25 source motif. Because width 11 encodes
#     rot270 and height 13 is odd, the final output is fliplr(rot90(source, 3)) at 25x25.
#   The key difference is that crop pixels reveal the hidden source size/orientation, while
#   crop shape reveals an extra transform on top of that recovered source.
# - Complexity calibration:
#   Candidate output sizes are limited to three values so the added latent variable is visible
#   but bounded. The width code has only four buckets and the height contributes one parity
#   bit, so the second layer is also compact. Generator-side rejection enforces uniqueness
#   across all sizes, offsets, orientations, and polarities.
# - Implicit assumption audit against the original solver:
#   * Original `_pattern_mask()` assumed a single 20x20 motif. Here it becomes
#     `_pattern_mask(size)` with cached resized variants.
#   * Original `assert H <= 20 and W <= 20` assumed a fixed canvas. Here the crop only needs
#     to fit at least one candidate size.
#   * Original `for r in range(20 - H + 1)` / `for c in range(20 - W + 1)` assumed one search
#     canvas. Here the search loops are size-specific.
#   * Original "matched pattern is the answer" is no longer true. Matching now recovers the
#     hidden source motif, and only then the size-coded transform is applied to get the output.
#   * New assumption: width must belong to the coding set {8,9,10,11,12}. Generator enforces
#     that directly.
# - Placement legality:
#   There is still exactly one motif and one crop window, so no overlap logic is introduced.
#   Generator first chooses a source motif size, then a crop shape that fits that square, then
#   an offset within `[0, size-H] x [0, size-W]`; every sample is in-bounds by construction.
# - Visual observability:
#   Across examples, humans can see outputs sometimes end at 15x15, sometimes 20x20 or 25x25,
#   so the motif family size is clearly changing. They can also see that two inputs with very
#   similar local content but different crop dimensions may yield different global orientations.
#   The width buckets are few, height parity is coarse, and the motif is asymmetric, so the
#   rotation/flip effects remain visually legible.
# - Visual cleanliness and color inference:
#   The scene remains a single two-color crop expanding into one two-color motif. No extra
#   markers or color-id-dependent rules are used; the two observed colors are simply foreground
#   and background for the reconstructed full pattern.
# - Branch coverage and anti-degeneracy policy for the generator:
#   The solver supports the full coding table, including the identity branch `(rot0, no flip)`.
#   Generator-side scheduling therefore cycles through all 8 legal `(rotation, flip)` branches
#   instead of silently dropping one. To keep the fusion legible from the first few examples,
#   the cycle places the identity branch last, so early calls preferentially expose visible
#   size-coded reorientation while long-run sampling still covers the complete D7 codebook.
# - Post-change `main()` assumptions checked against `generate_input()`:
#   * exactly two colors are present -> generator samples two distinct ARC colors
#   * the hidden full size is one of 15/20/25 -> generator samples from `_CANDIDATE_SIZES`
#   * crop width is one of the coded widths -> generator samples from `_ROTATION_TO_WIDTHS`
#   * one unique `(fg/bg, size, orientation, offset)` match exists -> generator retries until
#     `_find_unique_match()` succeeds and recovers the intended source motif
#   * repeated `generate_input()` calls cover all 8 size-code branches and all 3 motif sizes
#   * non-identity branches remain prominent in early samples, so the combo does not visually
#     collapse into a pure D5 task during normal dataset construction


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
_WIDTH_TO_ROTATIONS = {8: 0, 9: 1, 10: 2, 11: 3, 12: 0}
_ROTATION_TO_WIDTHS = {
    0: (8, 12),
    1: (9,),
    2: (10,),
    3: (11,),
}
_ALL_BRANCHES = tuple(
    (rotations, flip) for rotations in range(4) for flip in (False, True)
)
_BRANCH_CYCLE = tuple(branch for branch in _ALL_BRANCHES if branch != (0, False)) + (
    (0, False),
)
_SIZE_CYCLE = (15, 20, 25)
_branch_cycle_index = 0
_size_cycle_index = 0


def _canonical_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN_20], dtype=bool)


def _resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    row_idx = (np.arange(int(size)) * mask.shape[0]) // int(size)
    col_idx = (np.arange(int(size)) * mask.shape[1]) // int(size)
    return mask[row_idx][:, col_idx]


@lru_cache(maxsize=None)
def _pattern_mask(size: int) -> np.ndarray:
    assert int(size) in _CANDIDATE_SIZES
    return _resize_binary_mask(_canonical_mask(), int(size))


@lru_cache(maxsize=None)
def _dihedral_variants(size: int) -> tuple[np.ndarray, ...]:
    base = _pattern_mask(int(size))
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return tuple(variants)


def _decode_size_transform(height: int, width: int) -> tuple[int, bool]:
    assert int(width) in _WIDTH_TO_ROTATIONS
    rotations = _WIDTH_TO_ROTATIONS[int(width)]
    flip = bool(int(height) % 2)
    return rotations, flip


def _apply_size_transform(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rotations, flip = _decode_size_transform(int(height), int(width))
    transformed = np.rot90(pattern, rotations)
    if flip:
        transformed = np.fliplr(transformed)
    return transformed


def _other_color(colors: list[int], fg: int) -> int:
    return int(colors[0] if int(colors[1]) == int(fg) else colors[1])


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    colors = colors_present(g)
    assert len(colors) == 2
    assert any(height <= size and width <= size for size in _CANDIDATE_SIZES)

    matches: list[tuple[int, int, int, np.ndarray]] = []
    for fg in colors:
        bg = _other_color(colors, int(fg))
        visible = g == int(fg)
        for size in _CANDIDATE_SIZES:
            if height > size or width > size:
                continue
            for pattern in _dihedral_variants(size):
                for r in range(size - height + 1):
                    for c in range(size - width + 1):
                        if np.array_equal(pattern[r : r + height, c : c + width], visible):
                            matches.append((int(fg), int(bg), int(size), pattern))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a variable-size hidden motif, then re-orient it using crop-size codes.

    Input structure:
    - The grid is a two-color rectangular crop taken from one dihedral orientation of a shared binary motif family.
    - The hidden full motif is square but not fixed-size: it may be 15x15, 20x20, or 25x25.
    - The crop dimensions carry extra meaning: width selects a rotation bucket and height parity selects whether to mirror.

    Rule:
    - Try each observed color as the motif foreground and match the resulting boolean crop against every legal offset in every dihedral variant of every candidate size.
    - Once the unique source motif is found, decode the crop shape: widths 8/12, 9, 10, 11 mean 0/90/180/270 counterclockwise turns, and odd heights request an additional left-right flip.
    - Apply that size-coded transform to the recovered full motif and repaint it with the matched foreground/background colors.

    Key insight:
    - Pixel content tells you which full motif instance generated the crop, but the crop's own row/column counts tell you how that recovered motif must be rotated and flipped before returning the answer.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    fg, bg, _size, source_pattern = _find_unique_match(g)
    output_pattern = _apply_size_transform(source_pattern, height, width)
    return np.where(output_pattern, int(fg), int(bg)).astype(int)


@lru_cache(maxsize=None)
def _height_options(size: int, width: int, flip: bool) -> tuple[int, ...]:
    min_height = 11 if int(width) >= 11 else 12
    max_height = min(int(size), 16, 156 // int(width))
    parity = 1 if flip else 0
    heights = [
        height
        for height in range(min_height, max_height + 1)
        if height % 2 == parity and 80 <= height * int(width) <= 156
    ]
    return tuple(heights)


@lru_cache(maxsize=None)
def _shape_options(size: int, rotations: int, flip: bool) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for width in _ROTATION_TO_WIDTHS[int(rotations)]:
        for height in _height_options(int(size), int(width), bool(flip)):
            if height <= int(size) and int(width) <= int(size):
                shapes.append((int(height), int(width)))
    return tuple(shapes)


def _random_colors(rng: np.random.Generator) -> tuple[int, int]:
    fg, bg = rng.choice(Color.ALL_COLORS, size=2, replace=False)
    return int(fg), int(bg)


def _next_branch() -> tuple[int, bool]:
    global _branch_cycle_index
    branch = _BRANCH_CYCLE[_branch_cycle_index % len(_BRANCH_CYCLE)]
    _branch_cycle_index += 1
    return int(branch[0]), bool(branch[1])


def _next_size() -> int:
    global _size_cycle_index
    size = _SIZE_CYCLE[_size_cycle_index % len(_SIZE_CYCLE)]
    _size_cycle_index += 1
    return int(size)


def _sample_instance(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    rotations, flip = _next_branch()
    size = _next_size()
    shapes = _shape_options(size, int(rotations), bool(flip))
    assert shapes

    for _ in range(2048):
        height, width = shapes[int(rng.integers(0, len(shapes)))]
        source_pattern = _dihedral_variants(size)[int(rng.integers(0, 8))]
        fg, bg = _random_colors(rng)

        offset_r = int(rng.integers(0, size - height + 1))
        offset_c = int(rng.integers(0, size - width + 1))
        crop = source_pattern[offset_r : offset_r + height, offset_c : offset_c + width]
        input_grid = np.where(crop, fg, bg).astype(int)

        output_pattern = _apply_size_transform(source_pattern, height, width)
        delta_cells = int(np.count_nonzero(output_pattern != source_pattern))
        if (int(rotations), bool(flip)) != (0, False) and delta_cells == 0:
            continue
        expected = np.where(output_pattern, fg, bg).astype(int)

        try:
            matched_fg, matched_bg, matched_size, matched_pattern = _find_unique_match(input_grid)
        except AssertionError:
            continue

        if matched_fg != fg or matched_bg != bg:
            continue
        if matched_size != size or not np.array_equal(matched_pattern, source_pattern):
            continue
        if not np.array_equal(main(input_grid), expected):
            continue

        return input_grid, expected, {
            "size": size,
            "height": height,
            "width": width,
            "rotations": int(rotations),
            "flip": bool(flip),
            "offset_r": offset_r,
            "offset_c": offset_c,
            "delta_cells": delta_cells,
        }

    raise RuntimeError("failed to generate a valid combo instance")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    inp, _expected, _meta = _sample_instance(rng)
    return inp
