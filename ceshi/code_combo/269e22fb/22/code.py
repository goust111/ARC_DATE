# Combo variant of 269e22fb (index 22)
# combo_spec: D2/1+D17/2
# Name: 条形裁剪含边界标记

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# yaml
# variant:
#   task: Complete a boundary-aware extreme strip crop into the full 20x20 motif
#   preserved_core:
#     - The hidden object is still one dihedral orientation of the same fixed 20x20 binary motif.
#     - The output is still the full 20x20 motif recolored only by the crop's foreground/background colors.
#     - Orientation is still recovered by exhaustive matching against the canonical motif family.
#   fusion_rule:
#     - D2 contributes the crop geometry: the input grid itself is a long thin strip, not a chunky rectangle.
#     - D17 contributes the topology cue: the strip window is allowed to cross the motif boundary, and every out-of-bounds cell is painted with a third border color.
#     - The fused input is therefore one 3-color strip such as 4x19 or 19x5 whose edge segment may be fully border-colored.
#   concretization_and_trace:
#     - Example: choose a horizontal strip of shape 4x19 and place its source window at r0=-2, c0=1 on one motif orientation.
#     - Original task: r0 had to be in [0, 16], so the input would have been a 2-color in-bounds crop only.
#     - D2-only task: the crop would still be 4x19, but every cell would come from inside the motif support.
#     - D17-only task: the crop could cross the boundary, but it would usually still be a medium rectangle.
#     - Fused task: the first two rows of the 4x19 strip become a solid border-color band, while the lower two rows show motif fg/bg cells.
#     - Difference: the solver must use both the thin-strip geometry and the out-of-bounds marker simultaneously; matching only the visible motif sub-rectangle would throw away the fusion signal.
#   complexity_calibration:
#     - Extremely thin strips are visually obvious, but many left/right shifts are ambiguous if the strip only crosses along its long axis.
#     - To keep the boundary cue informative, generation only samples shapes from a vetted long-strip set and always forces a crossed side on the strip's short axis.
#     - Optional corner crossings are still allowed, but long-axis overlap is kept large so the strip remains recognizably long and still contains enough motif structure.
#   hidden_assumption_audit_main:
#     - `len(colors) == 2` from the original solver is invalid; fused inputs intentionally use exactly 3 colors.
#     - `for r in range(20 - H + 1)` and `for c in range(20 - W + 1)` assumed the crop stayed fully in-bounds; fused inputs need origins from `-(H-1)` / `-(W-1)` up to 19.
#     - `visible = g == fg` treated every non-foreground cell as background; that breaks once a third border color exists, so matching must compare the full rendered 3-color strip.
#     - The final repaint step `np.where(pattern, fg, bg)` still holds because the output remains the full motif with no border cells.
#     - The new `_find_matches` loop assumes the input is a single rectangular strip produced by exactly one boundary-crossing window of one motif orientation; `_generate_case` enforces this and rejects ambiguous samples.
#   placement_legality:
#     - The strip window is always axis-aligned and rectangular.
#     - Window origins are sampled so there is positive overlap with the 20x20 motif and at least one crossed boundary side.
#     - The border region is not a separate object; it is simply the out-of-bounds portion of the same strip window, so there is no object overlap ambiguity.
#   visual_observability:
#     - Humans can immediately see the D2 change because the input is a long bar rather than a block.
#     - Humans can immediately see the D17 change because one end or corner of the bar is a clean third-color band/block attached to the strip edge.
#     - Across a few examples, the repeated pattern is stable: same output type, but the input strip sometimes includes a border-colored cap when the window hits the motif boundary.
#   visual_cleanliness:
#     - Only one rectangular strip is present in the input, with at most three flat colors and no overdraw.
#     - Border-colored cells always appear as whole rows/columns at the strip margin, so the cue is easy to parse.
#   color_inference:
#     - No rule depends on numeric color ids.
#     - The border color is structurally identified by occupying the out-of-bounds band, while the remaining two colors still play the motif fg/bg roles.
#   generator_guarantees_for_main:
#     - Every sample uses one of the allowed extreme strip shapes.
#     - Every sample crosses at least one real motif boundary and therefore contains the third border color.
#     - Every accepted sample contains all 3 colors, has exactly 1 match under `_find_matches`, and satisfies `main(input) == output`.


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


_STRIP_SHAPES: tuple[tuple[int, int], ...] = (
    (4, 18),
    (18, 4),
    (4, 19),
    (19, 4),
    (5, 17),
    (17, 5),
    (5, 18),
    (18, 5),
    (5, 19),
    (19, 5),
)

_HORIZONTAL_SINGLE_SIDE_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("top",),
    ("bottom",),
)

_HORIZONTAL_CORNER_SIDE_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)

_VERTICAL_SINGLE_SIDE_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("left",),
    ("right",),
)

_VERTICAL_CORNER_SIDE_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("top", "left"),
    ("bottom", "left"),
    ("top", "right"),
    ("bottom", "right"),
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


def _is_extreme_strip(h: int, w: int) -> bool:
    return (int(h), int(w)) in _STRIP_SHAPES


def _overlap_bounds(H: int, W: int, r0: int, c0: int) -> tuple[int, int, int, int]:
    src_r0 = max(0, int(r0))
    src_r1 = min(20, int(r0) + int(H))
    src_c0 = max(0, int(c0))
    src_c1 = min(20, int(c0) + int(W))
    return src_r0, src_r1, src_c0, src_c1


def _crossed_sides(H: int, W: int, r0: int, c0: int) -> tuple[str, ...]:
    sides: list[str] = []
    if int(r0) < 0:
        sides.append("top")
    if int(r0) + int(H) > 20:
        sides.append("bottom")
    if int(c0) < 0:
        sides.append("left")
    if int(c0) + int(W) > 20:
        sides.append("right")
    return tuple(sides)


def _crosses_boundary(H: int, W: int, r0: int, c0: int) -> bool:
    return len(_crossed_sides(H, W, r0, c0)) > 0


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
    src_r0, src_r1, src_c0, src_c1 = _overlap_bounds(H, W, r0, c0)
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


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 3
    assert _is_extreme_strip(H, W)

    matches: list[tuple[int, int, np.ndarray, int, int, int]] = []
    for border in colors:
        inside = g != int(border)
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

        motif_colors = [c for c in colors if int(c) != int(border)]
        for fg in motif_colors:
            bg = int(motif_colors[0] if int(motif_colors[1]) == int(fg) else motif_colors[1])
            visible = g[dst_r0:dst_r1, dst_c0:dst_c1] == int(fg)
            for pattern in _VARIANTS:
                for src_r0 in row_starts:
                    src_r1 = src_r0 + overlap_h
                    for src_c0 in col_starts:
                        src_c1 = src_c0 + overlap_w
                        if np.array_equal(pattern[src_r0:src_r1, src_c0:src_c1], visible):
                            crop_r0 = src_r0 - dst_r0
                            crop_c0 = src_c0 - dst_c0
                            if not _crosses_boundary(H, W, crop_r0, crop_c0):
                                continue
                            candidate = _render_window(pattern, H, W, crop_r0, crop_c0, int(fg), int(bg), int(border))
                            if np.array_equal(candidate, g):
                                matches.append((int(fg), int(bg), pattern, int(border), int(crop_r0), int(crop_c0)))
    return matches


def _sample_crossing_sides(H: int, W: int, rng: np.random.Generator) -> tuple[str, ...]:
    if int(H) < int(W):
        single_options = _HORIZONTAL_SINGLE_SIDE_OPTIONS
        corner_options = _HORIZONTAL_CORNER_SIDE_OPTIONS
    else:
        single_options = _VERTICAL_SINGLE_SIDE_OPTIONS
        corner_options = _VERTICAL_CORNER_SIDE_OPTIONS

    options = single_options if int(rng.integers(2)) == 0 else corner_options
    return tuple(options[int(rng.integers(len(options)))])


def _sample_axis_start(
    size: int,
    touches_low: bool,
    touches_high: bool,
    rng: np.random.Generator,
    min_overlap: int,
) -> int:
    size = int(size)
    min_overlap = int(min_overlap)
    assert 1 <= min_overlap <= size

    if touches_low:
        return int(rng.integers(-(size - min_overlap), 0))
    if touches_high:
        return int(rng.integers(20 - size + 1, 21 - min_overlap))
    return int(rng.integers(0, 21 - size))


def _generate_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | tuple[int, int] | tuple[str, ...] | str]]:
    for _ in range(4000):
        fg, bg, border = map(int, rng.choice(Color.ALL_COLORS, size=3, replace=False))
        pattern_idx = int(rng.integers(len(_VARIANTS)))
        pattern = _VARIANTS[pattern_idx]
        output = np.where(pattern, int(fg), int(bg)).astype(int)

        H, W = _STRIP_SHAPES[int(rng.integers(len(_STRIP_SHAPES)))]
        sides = _sample_crossing_sides(H, W, rng)
        min_overlap_h = 2 if H <= 5 else 12
        min_overlap_w = 2 if W <= 5 else 12
        r0 = _sample_axis_start(H, "top" in sides, "bottom" in sides, rng, min_overlap_h)
        c0 = _sample_axis_start(W, "left" in sides, "right" in sides, rng, min_overlap_w)

        grid = _render_window(pattern, H, W, r0, c0, fg, bg, border)
        if colors_present(grid) != sorted((fg, bg, border)):
            continue

        border_cells = int(np.count_nonzero(grid == border))
        if border_cells == 0:
            continue

        matches = _find_matches(grid)
        if len(matches) != 1:
            continue
        match_fg, match_bg, match_pattern, _, _, _ = matches[0]
        solved = np.where(match_pattern, int(match_fg), int(match_bg)).astype(int)
        if not np.array_equal(solved, output):
            continue

        src_r0, src_r1, src_c0, src_c1 = _overlap_bounds(H, W, r0, c0)
        meta = {
            "shape": (H, W),
            "pattern_idx": pattern_idx,
            "fg": fg,
            "bg": bg,
            "border": border,
            "window_origin": (r0, c0),
            "crossed_sides": _crossed_sides(H, W, r0, c0),
            "overlap_shape": (src_r1 - src_r0, src_c1 - src_c0),
            "border_cells": border_cells,
            "orientation": "horizontal" if H < W else "vertical",
        }
        return grid, output, meta

    raise AssertionError("failed to generate a unique boundary-aware strip")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared 20x20 binary motif from a long strip crop that may extend beyond the motif boundary.

    Input structure:
    - The grid itself is an extreme-aspect-ratio rectangular strip such as 4x19 or 19x5.
    - Two colors encode motif foreground/background where the strip overlaps one dihedral orientation of the 20x20 motif.
    - A third color marks the out-of-bounds part of the same strip window whenever it crosses a real motif edge.

    Rule:
    - Treat each input color as the possible border marker and each ordering of the remaining two colors as foreground/background.
    - For every dihedral motif orientation and every boundary-crossing strip origin, render the full 3-color strip exactly as it would appear with out-of-bounds cells filled by the border color.
    - Select the unique exact match and repaint the full 20x20 motif using only the recovered foreground/background colors.

    Key insight:
    - The strip is too thin to rely only on chunky 2D context, but the third-color cap/block reveals which true motif edge the strip hit, so the correct orientation and offset can be found by matching the entire rendered boundary-aware strip.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20
    assert _is_extreme_strip(H, W)

    colors = colors_present(g)
    assert len(colors) == 3

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern, _, _, _ = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    """
    Output-first generator for the fused D2+D17 variant.

    New solver assumptions introduced above:
    - the input is always one of the allowed extreme strip sizes;
    - the strip always crosses at least one true motif boundary and therefore contains a border color;
    - the full 3-color strip must uniquely determine motif orientation, polarity, and source origin.

    This generator enforces those assumptions by construction and validates every
    accepted sample with `_find_matches` and `main`.
    """
    rng = np.random.default_rng()
    input_grid, output_grid, _ = _generate_case(rng)
    assert np.array_equal(main(input_grid), output_grid)
    return input_grid
