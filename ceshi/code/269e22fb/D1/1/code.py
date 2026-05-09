# Variant of 269e22fb
# Dimension: D1 — 匹配条件：裁剪窗口唯一性匹配方式 (match_cond)
# Change: 通过穷举8个二面体变体x2种前景/背景极性x所有偏移位置，找到唯一匹配裁剪窗口的(变体, 偏移, 极性)组合... → 带噪声的容错匹配
#
# Design analysis:
# spec:
#   preserved_structure:
#     motif: same fixed 20x20 binary pattern and same 8 dihedral orientations
#     colors: exactly 2 visible ARC colors, still interpreted as motif-foreground vs motif-background
#     output: same full 20x20 reconstruction in the matched orientation
#   core_change:
#     old_match_cond: require one exact crop match across all orientation/offset/polarity candidates
#     new_match_cond: input crop contains 1-3 flipped cells, so the solver must choose the unique candidate with minimum Hamming distance
#   concrete_pixel_behavior:
#     input: a rectangular crop from one oriented motif, then 1-3 cells in that crop are inverted between the two input colors
#     output: the clean full motif in the orientation/color polarity whose crop is closest to the noisy input
#   traced_example:
#     original_flow:
#       - true crop exactly equals one candidate window
#       - solver appends that candidate because np.array_equal(...) is True
#       - unique exact match determines the output
#     variant_flow:
#       - true crop is first extracted cleanly
#       - generator flips 1-3 crop cells, so no exact match should remain
#       - solver scores every candidate by count_nonzero(candidate_window != visible_mask)
#       - the true candidate now has distance k (the number of flips), and generation accepts only samples where every other candidate scores strictly worse
#     difference:
#       - exact equality is no longer the decision rule
#       - nearest-neighbor matching under small binary noise is now essential
#   complexity_calibration:
#     search_space: 8 orientations x 2 polarities x up to 9x11 offsets, which is small enough for exhaustive scoring
#     ambiguity_control:
#       - generator keeps crop sizes in the original visual regime but biases them toward safer unique-identification ranges
#       - generator rejects samples unless the noisy crop has exactly one minimum-distance candidate
#       - generator also rejects samples if an exact match survives, preventing regression to the original task
#   implicit_assumption_audit_for_main:
#     - colors_present(g) must return exactly 2 colors: generator always paints with two distinct ARC colors and flips only between them
#     - H,W must be <= 20: generator samples only valid crop sizes
#     - the minimum-distance candidate must be unique: generator checks this explicitly before returning an input
#     - the best candidate should correspond to the intended output orientation/polarity: generator verifies main(input) == output
#   placement_legality:
#     - crops are extracted from valid offsets inside the 20x20 pattern, so no overlap/out-of-bounds issue exists
#     - noise flips occur only inside the crop rectangle, preserving the task's single-object layout
#   visual_observability:
#     - every generated input contains actual wrong-color pixels, so the new behavior is visible in the crop itself
#     - the rest of the scene is unchanged, keeping the innovation focused on tolerant matching rather than introducing extra objects
#   cleanliness:
#     - no extra markers or overlay objects are added; the scene remains a single two-color rectangle, so the signal is not visually cluttered
#   color_reasoning:
#     - no hidden color-id ordering is used
#     - colors only serve as the two states of the binary motif, exactly as in the source task

from __future__ import annotations

from typing import Iterable

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

_ARC_COLORS = tuple(range(10))


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


def _other_color(colors: Iterable[int], fg: int) -> int:
    for color in colors:
        if int(color) != int(fg):
            return int(color)
    raise AssertionError("expected two distinct colors")


def _scored_candidates(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2

    scored: list[tuple[int, int, np.ndarray, int, int, int]] = []
    for fg in colors:
        bg = _other_color(colors, int(fg))
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    window = pattern[r : r + H, c : c + W]
                    dist = int(np.count_nonzero(window != visible))
                    scored.append((int(fg), int(bg), pattern, r, c, dist))
    return scored


def _best_candidates(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray, int, int, int]]]:
    scored = _scored_candidates(grid)
    best_dist = min(item[5] for item in scored)
    best = [item for item in scored if item[5] == best_dist]
    return best_dist, best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a noisy two-color crop by nearest dihedral matching.

    Input structure:
    - The grid is still a rectangular crop from one rotation/reflection of the shared 20x20 binary motif.
    - Exactly two colors are present, but 1-3 cells inside the crop may be flipped between motif and background.

    Rule:
    - Interpret each of the two colors as the possible motif foreground in turn.
    - Score every crop position in every dihedral variant by Hamming distance to the observed crop.
    - Take the unique minimum-distance candidate and repaint its full 20x20 motif with that candidate's foreground/background colors.

    Key insight:
    - The crop is no longer trusted cell-for-cell; the solution comes from finding which global motif placement best explains a small amount of local corruption.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    best_dist, matches = _best_candidates(g)
    assert len(matches) == 1

    fg, bg, pattern, _, _, _ = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(2000):
        fg, bg = map(int, rng.choice(_ARC_COLORS, size=2, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        output = np.where(pattern, fg, bg).astype(int)

        H = int(rng.integers(11, 17))
        W = int(rng.integers(9, 13))
        r = int(rng.integers(0, 21 - H))
        c = int(rng.integers(0, 21 - W))

        clean_crop = output[r : r + H, c : c + W].copy()
        clean_best_dist, clean_matches = _best_candidates(clean_crop)
        if clean_best_dist != 0 or len(clean_matches) != 1:
            continue

        noise_count = int(rng.integers(1, 4))
        noisy_crop = clean_crop.copy()
        flip_ids = rng.choice(H * W, size=noise_count, replace=False)
        for flat_idx in flip_ids:
            rr, cc = divmod(int(flat_idx), W)
            noisy_crop[rr, cc] = bg if int(noisy_crop[rr, cc]) == fg else fg

        best_dist, matches = _best_candidates(noisy_crop)
        if best_dist != noise_count or len(matches) != 1:
            continue

        matched_fg, matched_bg, matched_pattern, _, _, _ = matches[0]
        matched_output = np.where(matched_pattern, matched_fg, matched_bg).astype(int)
        if not np.array_equal(matched_output, output):
            continue

        if np.array_equal(noisy_crop, clean_crop):
            continue

        assert np.array_equal(main(noisy_crop), output)
        return noisy_crop

    raise RuntimeError("failed to generate a uniquely denoisable noisy crop")
