# Variant of 269e22fb
# Dimension: D10 — 多步处理：匹配-重建流程 (multi_step)
# Change: 单步处理：匹配裁剪窗口到唯一的(变体, 偏移, 极性)，然后直接输出完整图案... → 两阶段处理：先匹配变体再匹配偏移

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parents[5] / "solvers"
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
_HINT_W = 4
_GAP_W = 1
_SEP = Color.BLACK


# Design analysis
# spec:
#   variant_goal: Split the original joint search into two visible stages.
#   input_layout:
#     - Whole grid size is H x (W + 5), with the crop sampled from an
#       ambiguity-friendly subset of the original bounds: H in [8, 11],
#       W in [8, 10].
#     - Left top corner is a 2x4 hint board, column 4 is a black separator, and
#       columns 5.. are the original two-color crop.
#     - Rows below the hint board on the left stay black, so the two input parts
#       are visually isolated.
#   hint_encoding:
#     - Top hint row encodes rotation only: a left-justified foreground-colored run
#       of length rotation + 1, followed by background-colored cells.
#     - Bottom hint row encodes mirror only: all foreground means flipped,
#       all background means not flipped.
#     - The hint reuses the crop's two colors, while black is reserved for
#       separation, keeping color semantics local and visible.
#   concrete_trace:
#     - Original solver logic searched over every (foreground choice, variant,
#       offset) triple at once.
#     - New logic reads the 2x4 hint first. Example:
#         [fg fg bg bg]
#         [fg fg fg fg]
#       means rotation=1 and flipped=True, so only that one dihedral orientation
#       remains. The crop is then matched only inside that fixed 20x20 pattern to
#       recover the offset.
#   complexity_calibration:
#     - The number of orientation states stays at 8, matching the original task.
#     - To make the stage split matter, the generator prefers smaller crops whose
#       raw boolean windows are sometimes ambiguous across orientations, while they
#       remain unique once the hint locks the orientation.
#   hidden_assumptions_audit:
#     - Original `colors_present(g)` assumed the whole input was the crop; now we
#       must ignore the black separator and require the repaint colors to be
#       non-black.
#     - Original `_dihedral_variants(base)` search is no longer the first step;
#       the variant must come from the hint before any offset matching.
#     - Original polarity brute force happened to be harmless because the crop was
#       the only structure. Here the hint uses a prefix pattern, so wrong polarity
#       becomes invalid immediately instead of being silently tolerated.
#   placement_legality:
#     - Hint and crop never overlap because they occupy disjoint rectangles with a
#       fixed separator column between them.
#     - Choosing foreground/background from non-black colors ensures the separator
#       is never mistaken for crop content.
#   visual_observability:
#     - The new behavior is visible in just a few examples: cue top-row run length
#       varies with rotation, cue bottom-row fill varies with reflection, and the
#       crop alone no longer explains the full input.
#   cleanliness_and_colors:
#     - Only three colors appear in the input: black separator plus the crop's two
#       repaint colors. No arbitrary color-id lookup table is introduced.


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _variant_from_parts(mask: np.ndarray, rotation: int, flipped: bool) -> np.ndarray:
    rotated = np.rot90(mask, rotation)
    return np.fliplr(rotated) if flipped else rotated


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [_variant_from_parts(mask, rotation, bool(flipped)) for rotation in range(4) for flipped in range(2)]


def _encode_hint(rotation: int, flipped: bool, fg: int, bg: int) -> np.ndarray:
    hint = np.full((_HINT_H, _HINT_W), int(bg), dtype=int)
    hint[0, : rotation + 1] = int(fg)
    if flipped:
        hint[1, :] = int(fg)
    return hint


def _decode_hint(hint: np.ndarray, fg: int, bg: int) -> tuple[int, bool] | None:
    if hint.shape != (_HINT_H, _HINT_W):
        return None
    if not np.all((hint == int(fg)) | (hint == int(bg))):
        return None

    row0 = hint[0]
    run = int(np.sum(row0 == int(fg)))
    if not (1 <= run <= _HINT_W):
        return None
    if not np.all(row0[:run] == int(fg)):
        return None
    if not np.all(row0[run:] == int(bg)):
        return None

    row1 = hint[1]
    if np.all(row1 == int(fg)):
        flipped = True
    elif np.all(row1 == int(bg)):
        flipped = False
    else:
        return None
    return run - 1, flipped


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


@lru_cache(maxsize=None)
def _window_frequency(h: int, w: int) -> dict[bytes, int]:
    freq: dict[bytes, int] = {}
    for pattern in _dihedral_variants(_pattern_mask()):
        for r in range(20 - h + 1):
            for c in range(20 - w + 1):
                key = pattern[r : r + h, c : c + w].tobytes()
                freq[key] = freq.get(key, 0) + 1
    return freq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a hint-guided crop in two ordered stages.

    Input structure:
    - The left side starts with a 2x4 color hint board, followed by a 1-column black separator.
    - The right side is the original two-color rectangular crop of the motif.

    Rule:
    - Read the hint board first: the top-row foreground run gives the rotation, and the bottom row gives the mirror bit.
    - Use only that decoded dihedral orientation when matching the crop, rather than searching all orientations at once.
    - Once the crop offset is found, repaint the full 20x20 motif with the crop's foreground/background colors.

    Key insight:
    - The hint removes the orientation uncertainty explicitly, so the remaining job is only to place the crop inside one known variant of the shared motif.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= _HINT_H and W > _HINT_W + _GAP_W
    assert np.all(g[:, _HINT_W : _HINT_W + _GAP_W] == _SEP)
    assert np.all(g[_HINT_H:, :_HINT_W] == _SEP)

    colors = colors_present(g, exclude={_SEP})
    assert len(colors) == 2

    hint = g[:_HINT_H, :_HINT_W]
    crop = g[:, _HINT_W + _GAP_W :]
    assert crop.shape[0] <= 20 and crop.shape[1] <= 20

    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        decoded = _decode_hint(hint, int(fg), int(bg))
        if decoded is None:
            continue
        rotation, flipped = decoded
        pattern = _variant_from_parts(base, rotation, flipped)
        visible = crop == int(fg)
        crop_matches = _crop_matches(pattern, visible)
        if len(crop_matches) == 1:
            matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return _build_output(pattern, fg, bg)


def generate_input() -> np.ndarray:
    base = _pattern_mask()

    for _ in range(200):
        fg, bg = random.sample(Color.NOT_BLACK, 2)
        H = random.randint(8, 11)
        W = random.randint(8, 10)
        freq = _window_frequency(H, W)

        rotation = random.randrange(4)
        flipped = bool(random.randrange(2))
        pattern = _variant_from_parts(base, rotation, flipped)

        ambiguous_offsets: list[tuple[int, int]] = []
        fallback_offsets: list[tuple[int, int]] = []
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                window = pattern[r : r + H, c : c + W]
                if len(_crop_matches(pattern, window)) != 1:
                    continue
                offset = (r, c)
                if freq[window.tobytes()] > 1:
                    ambiguous_offsets.append(offset)
                else:
                    fallback_offsets.append(offset)

        candidates = ambiguous_offsets if ambiguous_offsets and random.random() < 0.75 else ambiguous_offsets + fallback_offsets
        if not candidates:
            continue

        r, c = random.choice(candidates)
        crop = _build_output(pattern[r : r + H, c : c + W], fg, bg)
        hint = _encode_hint(rotation, flipped, fg, bg)

        inp = np.full((H, _HINT_W + _GAP_W + W), _SEP, dtype=int)
        inp[:_HINT_H, :_HINT_W] = hint
        inp[:, _HINT_W + _GAP_W :] = crop

        expected = _build_output(pattern, fg, bg)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid D10 variant input")
