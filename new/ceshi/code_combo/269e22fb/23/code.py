# Combo variant of 269e22fb (index 23)
# combo_spec: D8/2+D11/0
# Name: 颜色值双重语义解码

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis
# ===============
# a) Concrete behavior trace:
# - Original task:
#   pick any two colors -> try both as foreground -> search all 8 dihedral variants and
#   all legal offsets -> the unique spatial match determines orientation, offset, and
#   polarity.
# - Fused task:
#   the two visible colors are chosen from a fixed library of 8 reserved color pairs.
#   After sorting the pair, the larger value is always the motif foreground and the
#   smaller value is always the motif background. The sorted pair itself picks the
#   dihedral variant index, so spatial matching only has to recover the crop offset
#   inside that preselected variant.
# - Example trace:
#   if the crop uses colors {3, 8}, then `8` must be the foreground because it is larger,
#   and `(3, 8)` maps to one specific variant. The 4x9 or 4x4 crop mask is then matched
#   only against that one variant to find its offset. The output is the full 20x20 motif
#   in that decoded orientation, repainted with fg=8 and bg=3.
# - Difference from the original:
#   orientation and polarity are no longer inferred from local geometry. They are both
#   encoded by the color values, while geometry only resolves position.
#
# b) Complexity calibration:
# - If we kept the original 8..16 by 8..12 crop sizes, exhaustive enumeration shows that
#   no crop has a complementary-polarity twin anywhere in the motif family, so the
#   "larger color is foreground" rule would never matter in practice.
# - To make both fused dimensions active, the generator instead uses a small library of
#   ambiguous crops, preferring 9x4 or 4x9 windows and falling back to 4x4 / 5x5 cases.
# - These windows are intentionally chosen so that:
#   1. the true foreground mask appears in at least two different dihedral variants, so
#      the color-pair-to-variant mapping is necessary;
#   2. the complemented mask also appears somewhere in the motif family, so the
#      larger-color foreground rule removes extra candidates that the original solver
#      would still consider.
#
# c) Implicit-assumption review against the original main():
# - `len(colors) == 2` still holds; the input is still a pure two-color crop.
# - "either input color could be foreground" is no longer valid. Polarity is fixed by
#   numeric ordering, so trying both colors would contradict the fused rule.
# - "all 8 dihedral variants are candidate orientations" is no longer valid. The color
#   pair itself chooses exactly one variant index.
# - The remaining spatial-search assumption still holds: once polarity and orientation are
#   decoded, the visible crop must land at exactly one offset inside that chosen variant.
# - `generate_input()` therefore guarantees that every emitted crop has a unique offset in
#   its mapped variant, even though the same bit pattern may also appear in other variants
#   or under the opposite polarity elsewhere.
#
# d) Placement legality:
# - Inputs remain ordinary rectangular crops from one 20x20 dihedral variant of the fixed
#   motif; there are no extra objects, overlaps, or out-of-bounds placements.
# - Every stored case is prevalidated from legal `(h, w, r, c)` windows, so crops always
#   lie inside the 20x20 pattern.
#
# e) Visual observability:
# - The fused behavior is subtle because ARC colors are visually categorical while the new
#   semantics depend on fixed pair membership and numeric ordering. To keep the rule at
#   least learnable from examples, the generator cycles through a small, repeated library
#   of 8 color pairs instead of arbitrary colors.
# - The crop sizes are also chosen so that the local geometry alone is ambiguous across
#   multiple variants, making the color semantics operational rather than decorative.
#
# f) Visual cleanliness:
# - The grid stays as clean as the original task: one small two-color crop in, one full
#   two-color motif out. No side markers, frames, or overlay codes are added.
# - Using only two colors avoids clutter even though those colors now carry two layers of
#   meaning.
#
# g) Change inferability:
# - The critical control variables are the sorted color pair and its max/min split.
# - Generator-side validation rejects any case where the crop would already be uniquely
#   solvable by the original "all variants, both polarities" search, because that would
#   collapse the fusion back into the base task.
# - The behavior metric used after implementation is the original-search ambiguity count:
#   number of `(variant, offset, polarity)` matches seen by the old solver on fused inputs.
#   We require this to stay above 1, and we also track how many distinct variants survive
#   under the correct polarity versus the swapped polarity.


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

_COLOR_PAIR_BY_VARIANT = (
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
    (1, 7),
    (2, 8),
    (3, 9),
    (4, 6),
)
_VARIANT_BY_COLOR_PAIR = {pair: idx for idx, pair in enumerate(_COLOR_PAIR_BY_VARIANT)}
_SHAPES = ((9, 4), (4, 9), (4, 4), (5, 5))
_GENERATOR_CASE_INDEX = 0


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


def _paint(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _decode_colors(grid: np.ndarray) -> tuple[int, int, int]:
    colors = colors_present(grid)
    assert len(colors) == 2
    bg, fg = map(int, sorted(colors))
    pair = (bg, fg)
    assert pair in _VARIANT_BY_COLOR_PAIR
    return bg, fg, int(_VARIANT_BY_COLOR_PAIR[pair])


def _window_indexes() -> tuple[
    dict[tuple[int, int, int], dict[bytes, list[tuple[int, int]]]],
    dict[tuple[int, int], dict[bytes, frozenset[int]]],
]:
    offsets_by_variant: dict[tuple[int, int, int], dict[bytes, list[tuple[int, int]]]] = {}
    variants_by_window: dict[tuple[int, int], dict[bytes, frozenset[int]]] = {}

    for h, w in _SHAPES:
        aggregate: dict[bytes, set[int]] = {}
        for variant_idx, pattern in enumerate(_VARIANTS):
            bucket: dict[bytes, list[tuple[int, int]]] = {}
            for r in range(20 - h + 1):
                for c in range(20 - w + 1):
                    key = pattern[r : r + h, c : c + w].tobytes()
                    bucket.setdefault(key, []).append((r, c))
                    aggregate.setdefault(key, set()).add(int(variant_idx))
            offsets_by_variant[variant_idx, h, w] = bucket
        variants_by_window[h, w] = {key: frozenset(value) for key, value in aggregate.items()}
    return offsets_by_variant, variants_by_window


_WINDOW_OFFSETS, _WINDOW_VARIANTS = _window_indexes()


def _original_match_count(visible: np.ndarray) -> int:
    h, w = map(int, visible.shape)
    total = 0
    key = visible.tobytes()
    comp_key = np.logical_not(visible).tobytes()
    for variant_idx in range(len(_VARIANTS)):
        bucket = _WINDOW_OFFSETS[variant_idx, h, w]
        total += len(bucket.get(key, ()))
        total += len(bucket.get(comp_key, ()))
    return int(total)


def _candidate_metrics(variant_idx: int, visible: np.ndarray) -> dict[str, int]:
    h, w = map(int, visible.shape)
    key = visible.tobytes()
    comp_key = np.logical_not(visible).tobytes()
    bucket = _WINDOW_OFFSETS[variant_idx, h, w]
    correct_offsets = bucket.get(key, ())
    same_variants = _WINDOW_VARIANTS[h, w].get(key, frozenset())
    swapped_variants = _WINDOW_VARIANTS[h, w].get(comp_key, frozenset())
    return {
        "correct_offset_count": int(len(correct_offsets)),
        "same_polarity_variant_count": int(len(same_variants)),
        "swapped_polarity_variant_count": int(len(swapped_variants)),
        "original_match_count": _original_match_count(visible),
    }


def _build_valid_cases() -> dict[int, list[dict[str, object]]]:
    cases: dict[int, list[dict[str, object]]] = {idx: [] for idx in range(len(_VARIANTS))}
    for variant_idx, pattern in enumerate(_VARIANTS):
        preferred_shapes = ((9, 4), (4, 4), (5, 5)) if variant_idx in (0, 1, 4, 5) else ((4, 9), (4, 4), (5, 5))
        for h, w in preferred_shapes:
            bucket = _WINDOW_OFFSETS[variant_idx, h, w]
            for r in range(20 - h + 1):
                for c in range(20 - w + 1):
                    visible = pattern[r : r + h, c : c + w]
                    metrics = _candidate_metrics(variant_idx, visible)
                    if metrics["correct_offset_count"] != 1:
                        continue
                    if metrics["same_polarity_variant_count"] < 2:
                        continue
                    if metrics["swapped_polarity_variant_count"] < 1:
                        continue
                    if metrics["original_match_count"] <= 1:
                        continue
                    cases[variant_idx].append(
                        {
                            "shape": (h, w),
                            "offset": (r, c),
                            "metrics": metrics,
                            "window_key": visible.tobytes(),
                        }
                    )

            if cases[variant_idx]:
                break

    for variant_idx, entries in cases.items():
        assert entries, f"no valid fused cases for variant {variant_idx}"
    return cases


_VALID_CASES = _build_valid_cases()


def _find_offsets(pattern: np.ndarray, visible: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, visible.shape)
    matches: list[tuple[int, int]] = []
    for r in range(20 - h + 1):
        for c in range(20 - w + 1):
            if np.array_equal(pattern[r : r + h, c : c + w], visible):
                matches.append((r, c))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full motif by decoding orientation from the color pair and polarity from the larger color value.

    Input structure:
    - The input is still a two-color rectangular crop cut from one dihedral orientation of the fixed 20x20 binary motif.
    - The visible color pair is not arbitrary: each allowed sorted pair is reserved for one specific dihedral variant.

    Rule:
    - Read the two colors, sort them, and treat the larger value as the motif foreground while the smaller value is the motif background.
    - Use that sorted color pair to select the only candidate dihedral variant, then match the resulting foreground mask against offsets inside that variant only.
    - Once the offset is identified, repaint the full 20x20 motif with the same foreground/background colors.

    Key insight:
    - The crop no longer has to disambiguate orientation or polarity geometrically; those two decisions are encoded in the color values themselves, and the local shape only determines where the crop came from.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 20 and w <= 20

    bg, fg, variant_idx = _decode_colors(g)
    pattern = _VARIANTS[variant_idx]
    visible = g == int(fg)
    matches = _find_offsets(pattern, visible)

    assert len(matches) == 1
    return _paint(pattern, fg, bg)


def _next_case_index() -> int:
    global _GENERATOR_CASE_INDEX
    case_index = _GENERATOR_CASE_INDEX
    _GENERATOR_CASE_INDEX += 1
    return int(case_index)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    case_index = _next_case_index()
    variant_idx = case_index % len(_VARIANTS)
    pair = _COLOR_PAIR_BY_VARIANT[variant_idx]
    bg, fg = map(int, pair)
    pattern = _VARIANTS[variant_idx]

    entries = _VALID_CASES[variant_idx]
    entry = entries[(case_index // len(_VARIANTS)) % len(entries)]
    if len(entries) > 1:
        entry = rng.choice(entries)

    h, w = map(int, entry["shape"])
    r, c = map(int, entry["offset"])
    full = _paint(pattern, fg, bg)
    inp = full[r : r + h, c : c + w].copy()
    expected = full.copy()

    actual = main(inp)
    assert np.array_equal(actual, expected)
    return inp, expected, {
        "variant_idx": int(variant_idx),
        "pair": pair,
        "fg": int(fg),
        "bg": int(bg),
        "shape": (h, w),
        "offset": (r, c),
        "metrics": dict(entry["metrics"]),
    }


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    if rng is None:
        rng = random.Random()
    inp, _, _ = _build_case(rng)
    return inp
