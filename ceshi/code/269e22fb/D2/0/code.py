# Variant of 269e22fb
# Dimension: D2 — 源对象：裁剪窗口的形状 (src)
# Change: 矩形裁剪窗口，行数8-16，列数8-12，面积80-156... → L形裁剪窗口
#
# Design analysis
# spec:
#   task: Recover the full 20x20 binary motif from a masked L-shaped viewport.
#   motif:
#     shape: Same fixed 20x20 binary pattern as the original task.
#     transforms: Same 8 dihedral variants.
#   input:
#     grid: Rectangular bounding box of one motif crop, but one corner sub-rectangle
#       is replaced by a third "mask" color, so the remaining visible cells form an L.
#     bbox_rows: 10-16
#     bbox_cols: 9-12
#     visible_area: 80-156
#     colors: exactly 3 distinct ARC colors in input
#       - 2 motif colors kept for visible cells
#       - 1 mask color used only on the removed corner rectangle
#     mask_shape:
#       - solid rectangle
#       - touches exactly one corner of the bounding box
#       - does not span full height or full width
#       - leaves both L arms at least 3 cells thick for visual clarity
#   output:
#     grid: 20x20
#     colors: only the 2 motif colors, painted over the matched full motif
#
# concretization_and_flow_trace:
#   original:
#     - input was a plain HxW rectangle cut from the motif
#     - solver compared every HxW window by full equality
#   variant:
#     - input is still stored in an HxW array, but a corner rectangle is occluded
#       by a third color; the visible support is therefore the rectangle minus one
#       corner block, i.e. an L-shaped viewport
#     - solver must first detect which color is the occlusion mask, then compare
#       only the visible cells against candidate motif windows
#   example:
#     - if a 12x10 crop loses its top-left 4x3 corner, original logic would fail
#       because those 12 masked cells no longer equal motif foreground/background
#     - the new logic ignores exactly those masked cells and matches the other 108
#       visible cells against each dihedral variant and offset
#
# complexity_calibration:
#   - A third color is necessary here. Without it, a non-rectangular viewport would
#     be invisible in an ARC rectangular grid because the hidden corner would be
#     indistinguishable from genuine background pixels.
#   - The cut-out area is constrained to be non-trivial (at least 10 cells) so the
#     L shape is visually obvious, but not so large that uniqueness collapses.
#   - The generator still brute-force verifies uniqueness with the updated solver,
#     preventing ambiguous crops.
#
# implicit_assumption_review_of_original_main:
#   - `len(colors) == 2`: no longer valid; the variant needs 3 input colors because
#     one color now denotes "not visible".
#   - `visible = g == fg`: still valid for visible cells, but must be masked so the
#     occluded corner is excluded from comparisons.
#   - `pattern[r:r+H, c:c+W] == visible`: no longer valid as a full-window equality;
#     only the L-shaped support may be compared.
#   - `bg` inferred as the other motif color: still valid after separating the mask.
#
# placement_validity:
#   - The missing rectangle always stays inside the crop bounding box.
#   - Because the full motif is sampled first, visible motif/background cells never
#     overlap illegally; only the occluded corner is overwritten by the mask color.
#   - The generator rejects any sample whose visible cells do not contain both motif
#     colors or whose reconstruction is not unique.
#
# observability_and_cleanliness:
#   - The human-visible change is explicit: every training input contains a clean
#     corner bite taken out of an otherwise rectangular crop.
#   - Only one extra region is added, with one uniform color, so the scene remains
#     easy to parse and does not create clutter.
#
# color_inference:
#   - The mask color is inferred structurally from shape, never from numeric color id.
#   - The remaining two colors are tried as foreground/background polarity exactly as
#     in the original task.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Assumption: one input color forms a solid corner rectangle. Guaranteed by
#     `generate_input()` construction.
#   - Assumption: the other two colors both appear on visible cells. Guaranteed by
#     rejection sampling.
#   - Assumption: exactly one (mask color, polarity, orientation, offset) match
#     exists. Guaranteed by calling `main()` inside the generator.

from __future__ import annotations

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


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _corner_cut_info(mask: np.ndarray) -> tuple[str, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    H, W = map(int, mask.shape)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    rect = np.zeros_like(mask, dtype=bool)
    rect[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    if not np.array_equal(mask, rect):
        return None

    cut_h = int(r1 - r0 + 1)
    cut_w = int(c1 - c0 + 1)
    if r0 == 0 and c0 == 0 and r1 < H - 1 and c1 < W - 1:
        return "tl", cut_h, cut_w
    if r0 == 0 and c1 == W - 1 and r1 < H - 1 and c0 > 0:
        return "tr", cut_h, cut_w
    if r1 == H - 1 and c0 == 0 and r0 > 0 and c1 < W - 1:
        return "bl", cut_h, cut_w
    if r1 == H - 1 and c1 == W - 1 and r0 > 0 and c0 > 0:
        return "br", cut_h, cut_w
    return None


def _make_corner_mask(H: int, W: int, corner: str, cut_h: int, cut_w: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    if corner == "tl":
        mask[:cut_h, :cut_w] = True
    elif corner == "tr":
        mask[:cut_h, W - cut_w :] = True
    elif corner == "bl":
        mask[H - cut_h :, :cut_w] = True
    else:
        assert corner == "br"
        mask[H - cut_h :, W - cut_w :] = True
    return mask


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray]] = []

    for mask_color in colors:
        occluded = g == int(mask_color)
        info = _corner_cut_info(occluded)
        if info is None:
            continue
        visible_mask = ~occluded
        motif_colors = [int(c) for c in colors if int(c) != int(mask_color)]
        if len(motif_colors) != 2:
            continue
        for fg in motif_colors:
            bg = motif_colors[0] if motif_colors[1] == fg else motif_colors[1]
            visible_fg = g == int(fg)
            for pattern in _dihedral_variants(base):
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        window = pattern[r : r + H, c : c + W]
                        if np.array_equal(window[visible_mask], visible_fg[visible_mask]):
                            matches.append((int(fg), int(bg), pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from a partial L-shaped viewport inside a bounding box.

    Input structure:
    - The grid is a three-color bounding rectangle cut from one rotated/reflected copy of the shared motif.
    - Two colors belong to the motif itself; the third color fills one corner sub-rectangle, so only an L-shaped set of cells is actually visible.

    Rule:
    - Detect which color acts as the corner mask by looking for a solid rectangle attached to one corner of the input.
    - Try each remaining color as motif foreground, and match only the unmasked cells against every offset of every dihedral motif variant.
    - Repaint the uniquely matched full 20x20 motif with the two motif colors, ignoring the mask color.

    Key insight:
    - The occluded corner tells you exactly which cells should be ignored, so the problem becomes matching a sparse but still distinctive L-shaped sample of the canonical pattern.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 3

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    base = _pattern_mask()
    variants = _dihedral_variants(base)
    palette = np.arange(10, dtype=int)
    corners = ("tl", "tr", "bl", "br")

    for _ in range(5000):
        fg, bg, mask_color = map(int, rng.choice(palette, size=3, replace=False))
        pattern = variants[int(rng.integers(len(variants)))]
        output = np.where(pattern, fg, bg).astype(int)

        H = int(rng.integers(10, 17))
        W = int(rng.integers(9, 13))
        full_area = H * W
        corner = corners[int(rng.integers(len(corners)))]

        cut_h = int(rng.integers(2, H - 2))
        cut_w = int(rng.integers(2, W - 2))
        if H - cut_h < 3 or W - cut_w < 3:
            continue

        cut_area = cut_h * cut_w
        visible_area = full_area - cut_area
        if not (10 <= cut_area <= full_area // 3):
            continue
        if not (80 <= visible_area <= 156):
            continue

        r = int(rng.integers(0, 21 - H))
        c = int(rng.integers(0, 21 - W))
        crop = output[r : r + H, c : c + W].copy()

        occlusion = _make_corner_mask(H, W, corner, cut_h, cut_w)
        crop[occlusion] = mask_color

        visible_colors = colors_present(crop, exclude=[mask_color])
        if len(visible_colors) != 2:
            continue
        if _corner_cut_info(crop == mask_color) is None:
            continue

        try:
            solved = main(crop)
        except Exception:
            continue
        if np.array_equal(solved, output):
            return crop

    raise RuntimeError("failed to generate a unique L-shaped crop")
