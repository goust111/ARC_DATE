# Combo variant of 269e22fb (index 14)
# combo_spec: D10/0+D14/2
# Name: 色块指示变体加边框指示旋转的两阶段

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present


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

_HINT_H = 2
_HINT_W = 2
_GAP_W = 1
_SEP = Color.BLACK

_ROTATION_TO_BORDER = {
    0: Color.BLUE,
    1: Color.RED,
    2: Color.GREEN,
    3: Color.YELLOW,
}
_BORDER_TO_ROTATION = {color: rotation for rotation, color in _ROTATION_TO_BORDER.items()}

_FLIP_TO_HINT = {
    False: Color.GREY,
    True: Color.MAROON,
}
_HINT_TO_FLIP = {color: flipped for flipped, color in _FLIP_TO_HINT.items()}

_INTERIOR_PALETTE = (Color.PINK, Color.ORANGE, Color.TEAL)


# Design analysis
# ---
# fusion_spec:
#   concrete_grid:
#     - Input is split into two visible regions: a 2x2 solid hint block on the
#       left and a framed crop on the right, separated by one black column.
#     - The right region is exactly the D14-style crop-with-border:
#       a (h+2)x(w+2) rectangle whose outer ring is one uniform color and whose
#       interior is the original two-color crop. To keep the new flip hint
#       active for this specific motif, the generator uses a rotation-dependent
#       shortlist of anisotropic windows:
#         rotation 0/180 -> (9x5) or (9x6)
#         rotation 90/270 -> (5x9) or (6x9)
#       These are the smallest legal windows that still keep the crop readable
#       while preserving mirror ambiguity after one cue is removed.
#     - The left hint block uses a fixed 2-color codebook, not the crop colors.
#       Grey means "not flipped" and maroon means "flipped".
#   behavior_trace:
#     - Original task:
#       search over 8 dihedral variants x 2 foreground polarities x all offsets
#       in one joint brute-force loop.
#     - Fused task:
#       1. Read the 2x2 hint block first. Example: a maroon square says the
#          hidden motif belongs to the mirrored branch of the dihedral family.
#       2. Read the border color next. Example: a red frame means rotation=90 deg.
#       3. Only then match the interior crop inside that one exact orientation to
#          recover the offset and repaint the full 20x20 motif.
#     - Difference:
#       the full orientation is no longer inferred from the crop alone; it is
#       assembled from two disjoint visual instructions before the offset search.
#   complexity_calibration:
#     - The border uses only 4 fixed colors, matching the 4 rotation classes.
#     - The hint block uses only 2 fixed colors, matching the mirror bit.
#     - Generator does not sample arbitrary crop sizes. It uses the shortlist
#       above because wider windows make the mirror bit too easy to recover from
#       the crop alone, which would collapse the D10 half of the fusion.
#     - Within that shortlist, generator accepts only windows where the raw crop
#       is still ambiguous if only one instruction source is used. Concretely,
#       every returned sample satisfies:
#         total_matches_across_all_variants > 1
#         matches_with_known_rotation_only > 1
#         matches_with_known_flip_only > 1
#         matches_with_both_hints == 1
#     - This makes the combo measurable: both hint channels are functionally
#       necessary, not merely decorative.
#   implicit_assumption_audit:
#     - Original `colors_present(g)` assumed the full input was just the crop.
#       That is false now because black separator, border, and hint colors are
#       extra syntax, so the solver must read crop colors only from the framed
#       interior.
#     - Original `H, W = g.shape` implicitly meant "crop size"; now it means the
#       composed layout, so the solver must first split hint / separator / frame.
#     - Original search over `_dihedral_variants(base)` in `main()` assumed the
#       crop carried all orientation information. In this fusion, skipping the
#       hint or border would ignore the rule, so `main()` must decode both before
#       any offset matching.
#     - The new hint is palette-coded rather than relative to fg/bg, so generator
#       must keep hint colors, border colors, and interior colors disjoint.
#   placement_legality:
#     - The left hint occupies rows 0:2, cols 0:2. Rows below it on the left stay
#       black, so it never overlaps the framed crop.
#     - The framed crop occupies all rows and the columns to the right of the
#       separator. Since the crop is sampled from a legal 20x20 window first and
#       only then wrapped with a border, no object can go out of bounds.
#   visual_observability:
#     - Across a few examples, a human can notice two independent repeated cues:
#       border color tracks rotation, and the isolated 2x2 color block tracks the
#       mirrored/non-mirrored choice. Neither cue depends on hidden numeric color
#       ordering or any invisible metadata.
#   cleanliness:
#     - Syntax layers are separated spatially: hint block left, separator center,
#       framed crop right. The crop itself remains untouched except for the
#       1-pixel frame, so the grid stays legible.
#   inferability_of_color_rules:
#     - Only 3 semantic color groups exist:
#         hint code colors: 2 fixed choices for flip
#         border code colors: 4 fixed choices for rotation
#         interior repaint colors: 2 chosen from a disjoint 3-color palette
#       This keeps color variety bounded and structurally anchored to location.
#   post_change_solver_assumptions:
#     - The hint block must be a uniform 2x2 square in the fixed hint palette.
#     - The separator column and the left padding under the hint must be black.
#     - The right region must have a uniform 1-pixel frame in the fixed rotation
#       palette, and its interior must contain exactly 2 colors.
#     - The chosen interior window must be unique after both cues are applied.
#       `generate_input()` enforces this with exact match counts and by checking
#       `main(inp) == expected_output` before returning.


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _variant_from_parts(mask: np.ndarray, rotation: int, flipped: bool) -> np.ndarray:
    rotated = np.rot90(mask, rotation)
    return np.fliplr(rotated) if flipped else rotated


@lru_cache(maxsize=None)
def _pattern_catalog() -> dict[tuple[int, bool], np.ndarray]:
    base = _pattern_mask()
    return {
        (rotation, flipped): _variant_from_parts(base, rotation, flipped)
        for rotation in range(4)
        for flipped in (False, True)
    }


def _build_output(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _crop_matches(pattern: np.ndarray, visible: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, visible.shape)
    matches: list[tuple[int, int]] = []
    for r in range(20 - h + 1):
        for c in range(20 - w + 1):
            if np.array_equal(pattern[r : r + h, c : c + w], visible):
                matches.append((r, c))
    return matches


def _decode_hint_block(block: np.ndarray) -> bool:
    assert block.shape == (_HINT_H, _HINT_W)
    color = int(block[0, 0])
    assert np.all(block == color)
    assert color in _HINT_TO_FLIP
    return _HINT_TO_FLIP[color]


def _extract_framed_crop(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3

    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    border_color = int(border[0])
    assert np.all(border == border_color)
    assert border_color in _BORDER_TO_ROTATION

    interior = g[1:-1, 1:-1]
    assert interior.shape[0] <= 20 and interior.shape[1] <= 20
    return border_color, interior


@lru_cache(maxsize=None)
def _window_frequencies(h: int, w: int) -> tuple[dict[bytes, int], dict[tuple[int, bytes], int], dict[tuple[bool, bytes], int], dict[tuple[int, bool, bytes], int]]:
    total_freq: dict[bytes, int] = {}
    rotation_freq: dict[tuple[int, bytes], int] = {}
    flip_freq: dict[tuple[bool, bytes], int] = {}
    exact_freq: dict[tuple[int, bool, bytes], int] = {}

    for (rotation, flipped), pattern in _pattern_catalog().items():
        for r in range(20 - h + 1):
            for c in range(20 - w + 1):
                key = pattern[r : r + h, c : c + w].tobytes()
                total_freq[key] = total_freq.get(key, 0) + 1
                rotation_key = (rotation, key)
                rotation_freq[rotation_key] = rotation_freq.get(rotation_key, 0) + 1
                flip_key = (flipped, key)
                flip_freq[flip_key] = flip_freq.get(flip_key, 0) + 1
                exact_key = (rotation, flipped, key)
                exact_freq[exact_key] = exact_freq.get(exact_key, 0) + 1

    return total_freq, rotation_freq, flip_freq, exact_freq


def _family_count(window: np.ndarray, *, rotation: int | None = None, flipped: bool | None = None) -> int:
    visible = np.asarray(window, dtype=bool)
    total_freq, rotation_freq, flip_freq, exact_freq = _window_frequencies(*visible.shape)
    keys = {visible.tobytes(), np.logical_not(visible).tobytes()}

    if rotation is not None and flipped is not None:
        return sum(exact_freq.get((rotation, flipped, key), 0) for key in keys)
    if rotation is not None:
        return sum(rotation_freq.get((rotation, key), 0) for key in keys)
    if flipped is not None:
        return sum(flip_freq.get((flipped, key), 0) for key in keys)
    return sum(total_freq.get(key, 0) for key in keys)


def _candidate_metrics(window: np.ndarray, rotation: int, flipped: bool) -> tuple[int, int, int, int]:
    visible = np.asarray(window, dtype=bool)
    exact_count = _family_count(visible, rotation=rotation, flipped=flipped)
    rotation_count = _family_count(visible, rotation=rotation)
    flip_count = _family_count(visible, flipped=flipped)
    total_count = _family_count(visible)
    return exact_count, rotation_count, flip_count, total_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the shared 20x20 motif from a crop that splits orientation clues across a color block and a border.

    Input structure:
    - The left side begins with a 2x2 solid-color hint block, followed by a 1-column black separator.
    - The right side is a two-color crop wrapped in a 1-pixel uniform border.
    - The hint block color encodes the mirror choice, while the border color encodes the rotation.

    Rule:
    - Read the hint block first to determine whether the hidden motif is mirrored.
    - Strip the border next and decode its color into one of the four rotation classes.
    - With that exact orientation fixed, match the interior crop at one offset and repaint the full 20x20 motif with the interior's two colors.

    Key insight:
    - The crop no longer has to reveal the whole dihedral state by itself; the solver must assemble the orientation from two explicit visual instructions before doing the usual offset match.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h >= 3 and w > _HINT_W + _GAP_W + 2
    assert np.all(g[:, _HINT_W : _HINT_W + _GAP_W] == _SEP)
    assert np.all(g[_HINT_H:, :_HINT_W] == _SEP)

    hint = g[:_HINT_H, :_HINT_W]
    flipped = _decode_hint_block(hint)

    border_color, interior = _extract_framed_crop(g[:, _HINT_W + _GAP_W :])
    rotation = _BORDER_TO_ROTATION[border_color]

    colors = colors_present(interior)
    assert len(colors) == 2

    pattern = _pattern_catalog()[(rotation, flipped)]
    matches: list[tuple[int, int]] = []
    chosen_fg = chosen_bg = None
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = interior == int(fg)
        crop_matches = _crop_matches(pattern, visible)
        if len(crop_matches) == 1:
            matches.append(crop_matches[0])
            chosen_fg, chosen_bg = int(fg), int(bg)

    assert len(matches) == 1
    assert chosen_fg is not None and chosen_bg is not None
    return _build_output(pattern, chosen_fg, chosen_bg)


def generate_input() -> np.ndarray:
    rng = random.Random()
    preferred_sizes = {
        0: ((9, 5), (9, 6)),
        1: ((5, 9), (6, 9)),
        2: ((9, 5), (9, 6)),
        3: ((5, 9), (6, 9)),
    }

    for _ in range(400):
        fg, bg = rng.sample(_INTERIOR_PALETTE, 2)
        rotation = rng.randrange(4)
        flipped = bool(rng.randrange(2))
        crop_h, crop_w = rng.choice(preferred_sizes[rotation])

        pattern = _pattern_catalog()[(rotation, flipped)]
        strong: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []

        for r in range(20 - crop_h + 1):
            for c in range(20 - crop_w + 1):
                window = pattern[r : r + crop_h, c : c + crop_w]
                metrics = _candidate_metrics(window, rotation, flipped)
                exact_count, rotation_count, flip_count, total_count = metrics
                if exact_count == 1 and rotation_count > 1 and flip_count > 1 and total_count > 1:
                    strong.append((metrics, (r, c)))

        if not strong:
            continue

        _, (r0, c0) = rng.choice(strong)
        interior = _build_output(pattern[r0 : r0 + crop_h, c0 : c0 + crop_w], fg, bg)
        border_color = _ROTATION_TO_BORDER[rotation]
        hint_color = _FLIP_TO_HINT[flipped]

        inp = np.full((crop_h + 2, _HINT_W + _GAP_W + crop_w + 2), _SEP, dtype=int)
        inp[:_HINT_H, :_HINT_W] = hint_color
        inp[:, _HINT_W + _GAP_W :] = border_color
        inp[1:-1, _HINT_W + _GAP_W + 1 : _HINT_W + _GAP_W + 1 + crop_w] = interior

        expected = _build_output(pattern, fg, bg)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid combo input")
