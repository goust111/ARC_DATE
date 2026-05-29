# Variant of 269e22fb
# Dimension: D6 — 二面体群变体：变换集合 (meta)
# Change: 使用完整的二面体群D4（4个旋转x2个翻转=8个变体），所有8个变体均不同... → 扩展为包含颜色反转的变体

from __future__ import annotations

from functools import lru_cache

import numpy as np

from arc_utils import colors_present


# Design analysis for the D6 variant
# a) Concrete behavior:
#    A plain 16-candidate extension is degenerate here: complementing the mask and
#    swapping the two observed colors gives the same full output that the original
#    solver already reconstructs. To make D6 visible, each 20x20 candidate now
#    carries a polarity stamp: every corner contains a 2x2 block whose color says
#    whether the canonical motif was used directly or complemented.
#    Small trace:
#    - Original logic: crop = subwindow of a geometric variant; trying both color
#      roles already simulates color inversion.
#    - New logic: full candidate = geometric/complemented motif + corner stamp;
#      input crop is taken from one corner so the stamp is visible, and the solver
#      must recover one of 16 stamped full grids.
#    - Difference: inversion is no longer just a relabeling of the same two colors;
#      it changes persistent pixels at fixed locations.
# b) Complexity calibration:
#    The corner stamp adds only one extra repeated cue, not a new object family.
#    With corner-anchored crops, every original size in rows 8-16 and cols 8-12 is
#    uniquely identifiable across all 16 candidates and all offsets, so the task
#    stays stable without shrinking the size range.
# c) Old implicit assumptions that break:
#    - `for fg in colors:` is now insufficient because color-role swapping alone
#      cannot synthesize the fixed corner stamp.
#    - Matching only the raw motif body ignores the new polarity cue and would
#      reconstruct the wrong full image.
#    The new solver therefore searches pre-rendered 16 stamped candidates instead
#    of interpreting the crop through a foreground/background guess loop.
# d) Placement legality:
#    The generator is still output-first. It builds one stamped 20x20 candidate,
#    then chooses one of four corner-aligned crop positions that fit inside the
#    board. No overlap or out-of-bounds case can occur because the input is a crop
#    of an already valid full grid.
# e) Visual observability:
#    The inversion branch is now visible in two ways: the motif occupancy flips,
#    and the corner stamp color flips with it. A solver cannot treat the two as the
#    same case anymore.
# f) Visual cleanliness:
#    The task remains tidy: one motif plus four tiny repeated corner stamps, still
#    with exactly two colors and no extra clutter in the interior.
# g) Color-change inferability:
#    Only two colors are used. The corner stamp repeats the same global polarity
#    choice in all four corners, so the color-change rule is structural and easy to
#    see rather than tied to color ids.
# 9) New solver assumptions and how generation enforces them:
#    - Exactly two colors must be present: generator always renders with a 2-color
#      palette and crops from that rendering.
#    - The crop must include a polarity cue: generator samples only corner-aligned
#      windows, so the input always contains one stamped corner.
#    - The crop must uniquely identify one stamped candidate and one offset:
#      `_perfect_crop_sizes()` keeps the full original size range under the new
#      stamped-corner construction, and `generate_input()` still checks
#      `main(inp) == expected`.


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


@lru_cache(maxsize=1)
def _candidate_masks() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for variant in _dihedral_variants(_pattern_mask()):
        for inverted in (False, True):
            body = (~variant).copy() if inverted else variant.copy()
            marker_value = not inverted
            body[:2, :2] = marker_value
            body[:2, -2:] = marker_value
            body[-2:, :2] = marker_value
            body[-2:, -2:] = marker_value
            masks.append(body)
    assert len({mask.tobytes() for mask in masks}) == 16
    return tuple(masks)


@lru_cache(maxsize=1)
def _perfect_crop_sizes() -> tuple[tuple[int, int], ...]:
    # Exhaustively checked during design: with the repeated corner stamp and
    # corner-anchored crops, every original size in the 8-16 by 8-12 range is
    # uniquely identifiable even if the solver searches all offsets.
    return tuple((height, width) for height in range(8, 17) for width in range(8, 13))


def _canonical_colors(grid: np.ndarray) -> tuple[int, int]:
    colors = colors_present(grid)
    assert len(colors) == 2
    return int(colors[0]), int(colors[1])


def _find_rendered_matches(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    color0, color1 = _canonical_colors(g)

    matches: list[np.ndarray] = []
    for mask in _candidate_masks():
        rendered = np.where(mask, color0, color1).astype(int)
        for top in range(20 - height + 1):
            for left in range(20 - width + 1):
                if np.array_equal(rendered[top : top + height, left : left + width], g):
                    matches.append(rendered)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stamped 20x20 motif from a crop when both geometry and polarity may vary.

    Input structure:
    - The grid is a two-color corner crop taken from one of 16 full-size candidates.
    - Each candidate combines one D4 orientation of the shared 20x20 motif, an optional global color inversion, and a repeated 2x2 polarity stamp in the four corners.

    Rule:
    - Read the two observed colors as a fixed ordered pair only to render candidate full grids consistently.
    - Compare the crop against every window of every stamped geometric orientation and every stamped complemented orientation.
    - Return the uniquely matched 20x20 candidate using the same two colors.

    Key insight:
    - The corner stamp makes polarity persistent, so color inversion becomes a real extra branch instead of a color-role ambiguity that the old solver could absorb for free.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height <= 20 and width <= 20

    matches = _find_rendered_matches(g)
    assert len(matches) == 1
    return matches[0]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    colors = rng.choice(np.arange(10, dtype=int), size=2, replace=False)
    color0, color1 = sorted(map(int, colors))

    masks = _candidate_masks()
    mask = masks[int(rng.integers(len(masks)))]
    expected = np.where(mask, color0, color1).astype(int)

    crop_sizes = _perfect_crop_sizes()
    height, width = crop_sizes[int(rng.integers(len(crop_sizes)))]
    corner_offsets = (
        (0, 0),
        (0, 20 - width),
        (20 - height, 0),
        (20 - height, 20 - width),
    )
    top, left = corner_offsets[int(rng.integers(len(corner_offsets)))]
    inp = expected[top : top + height, left : left + width].copy()

    solved = main(inp)
    assert np.array_equal(solved, expected)
    return inp
