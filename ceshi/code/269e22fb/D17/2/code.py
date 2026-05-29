# Variant of 269e22fb
# Dimension: D17 — 图案的拓扑结构 (meta)
# Change: 平面拓扑，图案有明确边界，裁剪窗口不跨越边界... → 图案边界用特殊颜色标记

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D17 variant
# a) Concretization and trace:
#    - The input is still a rectangular crop of one 20x20 dihedral variant of the
#      same binary motif, but the crop may now extend beyond the motif's real
#      boundary. Cells outside the 20x20 support are filled with a third "out of
#      bounds" color.
#    - Example trace: if a 10x9 crop starts at r0=-2, c0=5, the top two rows of
#      the crop are fully the border color, while rows 2..9 show motif data from
#      rows 0..7. In the original task this placement was illegal because r0 had
#      to satisfy 0 <= r0 <= 10. In the variant the solver must match against a
#      padded plane, not only against in-bounds windows.
# b) Complexity calibration:
#    - The crop still uses one fixed 20x20 motif and the same 8 dihedral
#      orientations. This keeps the original orientation-identification challenge.
#    - The new signal is restricted to exactly one extra color and to windows that
#      cross one edge or one corner. That is visually obvious without introducing
#      many new cases.
#    - Crop sizes stay near the original ranges (8-16 by 8-12), and generation
#      requires all three colors to appear so polarity and border cues are both
#      meaningful.
# c) Hidden assumptions in the original main():
#    - `len(colors) == 2` is no longer valid; the variant requires exactly 3
#      colors in the input.
#    - Offsets `r in [0, 20-H]`, `c in [0, 20-W]` assumed the crop never crossed
#      the boundary. The variant needs offsets from `-(H-1)` / `-(W-1)` up to 19.
#    - `visible = g == fg` treated every non-foreground cell as background. That
#      breaks once a third border color exists, so matching must compare a full
#      three-color rendering.
#    - The final `np.where(pattern, fg, bg)` remains valid because the output is
#      still the full 20x20 motif with only foreground/background recoloring.
# d) Placement legality:
#    - Generation only uses windows with positive overlap with the 20x20 motif and
#      with at least one crossed edge. The crop is never fully outside the motif.
#    - The crop must contain border, foreground, and background colors, and the
#      solver must find exactly one match before the sample is accepted.
# e) Visual observability:
#    - The third color appears as a clean strip or corner block at the crop's
#      margin, so a human can see that the window has gone past the object's real
#      edge.
#    - Samples vary between single-edge and corner crossings so the boundary cue is
#      not a one-off accident.
# f) Visual cleanliness:
#    - There is still only one rectangular crop and one reconstructed output.
#      Nothing overlaps besides the natural juxtaposition of motif data and the
#      border fill.
# g) Color inference:
#    - The border color is structurally identifiable by occupying the rectangular
#      outside region. No rule depends on numeric color ids or hidden ordering.


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


_BASE_MASK = _pattern_mask()
_VARIANTS = _dihedral_variants(_BASE_MASK)
_ARC_COLORS = tuple(range(10))


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, np.asarray(grid).flat)))


def _render_window(
    pattern: np.ndarray,
    H: int,
    W: int,
    r0: int,
    c0: int,
    fg: int,
    bg: int,
    border: int,
) -> np.ndarray:
    out = np.full((H, W), int(border), dtype=int)
    src_r0 = max(0, r0)
    src_r1 = min(20, r0 + H)
    src_c0 = max(0, c0)
    src_c1 = min(20, c0 + W)
    if src_r0 < src_r1 and src_c0 < src_c1:
        patch = np.where(pattern[src_r0:src_r1, src_c0:src_c1], int(fg), int(bg))
        out[src_r0 - r0 : src_r1 - r0, src_c0 - c0 : src_c1 - c0] = patch
    return out


def _rectangular_inside(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    rect = np.zeros_like(mask, dtype=bool)
    rect[r0:r1, c0:c1] = True
    if not np.array_equal(mask, rect):
        return None
    return int(r0), int(r1), int(c0), int(c1)


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = _colors_present(g)
    assert len(colors) == 3

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for border in colors:
        inside = g != border
        inside_rect = _rectangular_inside(inside)
        if inside_rect is None:
            continue
        dst_r0, dst_r1, dst_c0, dst_c1 = inside_rect
        if dst_r0 > 0 and dst_r1 < H:
            continue
        if dst_c0 > 0 and dst_c1 < W:
            continue

        overlap_h = dst_r1 - dst_r0
        overlap_w = dst_c1 - dst_c0
        row_starts = [0] if dst_r0 > 0 else ([20 - overlap_h] if dst_r1 < H else list(range(0, 21 - overlap_h)))
        col_starts = [0] if dst_c0 > 0 else ([20 - overlap_w] if dst_c1 < W else list(range(0, 21 - overlap_w)))
        motif_colors = [c for c in colors if c != border]
        for fg, bg in ((motif_colors[0], motif_colors[1]), (motif_colors[1], motif_colors[0])):
            visible = g[dst_r0:dst_r1, dst_c0:dst_c1] == fg
            for pattern in _VARIANTS:
                for src_r0 in row_starts:
                    src_r1 = src_r0 + overlap_h
                    for src_c0 in col_starts:
                        src_c1 = src_c0 + overlap_w
                        if np.array_equal(pattern[src_r0:src_r1, src_c0:src_c1], visible):
                            crop_r0 = src_r0 - dst_r0
                            crop_c0 = src_c0 - dst_c0
                            matches.append((int(fg), int(bg), pattern, int(crop_r0), int(crop_c0)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a crop that may spill beyond the motif boundary.

    Input structure:
    - The grid is a three-color rectangular window cut from one dihedral orientation of the shared 20x20 motif.
    - Two colors encode motif foreground/background inside the true 20x20 support, and a third color marks cells where the crop extends outside that support.

    Rule:
    - Try each input color as the possible out-of-bounds marker, and each ordering of the remaining two colors as foreground/background.
    - Use the border-colored cells to recover the rectangular in-bounds part of the crop, which reveals which sides of the 20x20 support were crossed and sharply narrows the allowed source offsets.
    - Compare that recovered in-bounds rectangle against every dihedral motif orientation, then repaint the uniquely matched full 20x20 motif using only the recovered foreground/background colors.

    Key insight:
    - The border color reveals when the window hits a real edge of the motif, so the crop is identified by matching against a padded version of the motif rather than only against internal sub-rectangles.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern, _, _ = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _sample_crossing_sides(rng: random.Random) -> tuple[str, ...]:
    if rng.random() < 0.5:
        return (rng.choice(("top", "bottom", "left", "right")),)
    return rng.choice(
        (
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
        )
    )


def _sample_axis_start(size: int, touches_low: bool, touches_high: bool, rng: random.Random) -> int:
    min_overlap = 5
    if touches_low:
        return rng.randint(-(size - min_overlap), -1)
    if touches_high:
        return rng.randint(20 - size + 1, 20 - min_overlap)
    return rng.randint(0, 20 - size)


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D17 variant.

    New solver assumptions introduced above:
    - exactly three colors appear in the input;
    - one contiguous rectangular margin region lies outside the 20x20 motif;
    - the visible in-bounds part still contains both motif states.

    This generator enforces those assumptions by construction and validates every
    accepted sample with `main`.
    """
    rng = random.Random()

    for _ in range(2000):
        fg, bg, border = rng.sample(_ARC_COLORS, 3)
        pattern = rng.choice(_VARIANTS)
        output = np.where(pattern, int(fg), int(bg)).astype(int)

        H = rng.randint(8, 16)
        W = rng.randint(8, 12)
        sides = _sample_crossing_sides(rng)
        r0 = _sample_axis_start(H, "top" in sides, "bottom" in sides, rng)
        c0 = _sample_axis_start(W, "left" in sides, "right" in sides, rng)

        grid = _render_window(pattern, H, W, r0, c0, fg, bg, border)
        if _colors_present(grid) != sorted((fg, bg, border)):
            continue

        matches = _find_matches(grid)
        if len(matches) != 1:
            continue
        if not np.array_equal(main(grid), output):
            continue
        return grid

    raise AssertionError("failed to generate a unique border-aware crop")
