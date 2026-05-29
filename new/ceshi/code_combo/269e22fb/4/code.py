# Combo variant of 269e22fb (index 4)
# combo_spec: D2/0+D9/1
# Name: L形裁剪输入对称区域输出
#
# Design analysis
# spec:
#   task: Identify a dihedral 20x20 motif placement from a masked L-shaped crop,
#     then output the point-reflected L-shaped partner region instead of the full motif.
#   inherited_core:
#     motif: same fixed 20x20 binary pattern as the source task
#     transforms: same 8 dihedral variants
#     polarity: the two motif colors still have unknown fg/bg roles
#   fused_changes:
#     D2:
#       input_support: the crop is stored in an HxW bounding box, but one corner
#         sub-rectangle is replaced by a third mask color, leaving an L-shaped
#         visible support
#     D9:
#       output_shape: not 20x20 anymore; return an HxW box matching the input size
#       output_content: the point-reflected counterpart of the input L-shaped region
#         inside the same matched motif orientation
#
# concretization_and_flow_trace:
#   original:
#     - choose a rectangular crop pattern[r:r+H, c:c+W]
#     - match that full rectangle directly
#     - repaint the whole 20x20 motif
#   fused variant:
#     - choose the same HxW crop, then overwrite one corner rectangle with a third
#       color; the remaining visible cells form an L
#     - example: if HxW = 12x10 and the input hides a 4x3 top-left corner, the
#       solver ignores exactly those 12 cells and matches the other 108 cells
#     - if the matched crop sits at (r, c), the opposite bounding box is
#       (r2, c2) = (20-H-r, 20-W-c)
#     - the visible support itself is also point-reflected, so a top-left hidden
#       corner becomes a bottom-right hidden corner in the output
#   difference:
#     - D2 changes what part of the crop is trustworthy during matching
#     - D9 changes where the final pixels are read from
#     - the fusion requires both at once: infer placement from an L-shaped sample,
#       then project the reflected L, not a rectangle and not the full motif
#
# complexity_calibration:
#   - The mask color is necessary; otherwise the removed corner would be confused
#     with ordinary background and the L shape would be invisible.
#   - Cut sizes stay moderate and both L arms must remain at least 3 cells thick,
#     keeping the shape obvious without making matches too ambiguous.
#   - The generator rejects degenerate cases where the reflected L output would be
#     identical to the input, so the center-symmetry behavior is visible in examples.
#
# implicit_assumption_review_of_original_main:
#   - `len(colors) == 2` no longer holds; the fused input has 3 colors because one
#     color marks hidden cells.
#   - `visible = g == fg` is still useful, but comparisons must be restricted to the
#     unmasked cells only.
#   - Full-window equality against `pattern[r:r+H, c:c+W]` is invalid because the
#     corner mask deliberately destroys part of the rectangle.
#   - The old solver only needed `(fg, bg, pattern)` because it always returned the
#     whole motif; this fusion also needs `(r, c)` and the hidden-corner geometry to
#     build the reflected partner crop.
#
# placement_validity:
#   - The input cut-out rectangle is always fully inside the HxW crop and never spans
#     the full height or width.
#   - The reflected bounding box is always legal because `(20-H-r, 20-W-c)` is the
#     point-reflected offset of another in-bounds HxW window.
#   - Output masking is applied after reading the reflected motif window, so no
#     illegal overlap occurs; the hidden cells are simply overwritten by the mask.
#
# observability_and_cleanliness:
#   - The input difference from the source task is immediately visible: every sample
#     has a clean corner bite and therefore a non-rectangular viewport.
#   - The output difference is also visible: it has the same overall size as the
#     input, shows a different part of the motif, and the hidden corner flips to the
#     opposite corner under point reflection.
#   - Only one extra uniform region is introduced, so the scene stays sparse and easy
#     to parse.
#
# color_inference_and_rule_inferability:
#   - Colors are not remapped numerically. The two motif colors are preserved, and
#     the same mask color is reused in the output.
#   - The mask is inferred by shape alone: one color must form a solid corner
#     rectangle. This gives a structural clue that humans can read from examples.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Assumption: exactly one input color forms a proper corner rectangle. Guaranteed
#     by generation and rechecked by `_corner_cut_info`.
#   - Assumption: the other two colors both appear on visible cells. Guaranteed by
#     rejection sampling.
#   - Assumption: exactly one `(mask color, polarity, orientation, offset)` match
#     exists. Guaranteed by brute-force validation in `generate_input()`.
#   - Assumption: the output should hide the reflected corner, not the original one.
#     Guaranteed by explicitly reflecting the corner label via `_opposite_corner()`.

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

_SIZE = 20
_ROW_RANGE = (10, 16)
_COL_RANGE = (9, 12)
_VISIBLE_AREA_RANGE = (80, 156)
_PALETTE = np.arange(10, dtype=int)
_CORNERS = ("tl", "tr", "bl", "br")
_GEN_RETRIES = 5000


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


def _opposite_corner(corner: str) -> str:
    return {
        "tl": "br",
        "tr": "bl",
        "bl": "tr",
        "br": "tl",
    }[corner]


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, int, np.ndarray, int, int, str, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    base = _pattern_mask()
    matches: list[tuple[int, int, int, np.ndarray, int, int, str, int, int]] = []

    for mask_color in colors:
        occluded = g == int(mask_color)
        info = _corner_cut_info(occluded)
        if info is None:
            continue
        corner, cut_h, cut_w = info
        visible_mask = ~occluded
        motif_colors = [int(c) for c in colors if int(c) != int(mask_color)]
        if len(motif_colors) != 2:
            continue
        for fg in motif_colors:
            bg = motif_colors[0] if motif_colors[1] == fg else motif_colors[1]
            visible_fg = g == int(fg)
            for pattern in _dihedral_variants(base):
                for r in range(_SIZE - H + 1):
                    for c in range(_SIZE - W + 1):
                        window = pattern[r : r + H, c : c + W]
                        if np.array_equal(window[visible_mask], visible_fg[visible_mask]):
                            matches.append((int(fg), int(bg), int(mask_color), pattern, int(r), int(c), corner, int(cut_h), int(cut_w)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match a masked L-shaped motif crop and return the point-reflected L-shaped partner window.

    Input structure:
    - The grid is a three-color HxW bounding box cut from one rotation/reflection of the shared 20x20 binary motif.
    - Two colors belong to the motif itself, while the third color fills one corner sub-rectangle so that only an L-shaped subset of the crop is visible.

    Rule:
    - Detect which color is the corner mask by finding a solid masked rectangle attached to exactly one corner, then try the other two colors as motif foreground/background.
    - Match only the unmasked cells against every offset of every dihedral motif variant to recover the unique `(orientation, offset)` of the hidden crop.
    - Point-reflect that crop location through the 20x20 frame center, read the opposite HxW motif window, and reapply the reflected corner mask with the same mask color.

    Key insight:
    - The masked corner tells you both which cells to ignore during recognition and how the non-rectangular viewport itself must move under center reflection, so solving the task means reflecting the entire L-shaped viewing region, not merely the visible pixels.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= _SIZE and W <= _SIZE

    colors = colors_present(g)
    assert len(colors) == 3

    matches = _find_matches(g)
    assert len(matches) == 1

    fg, bg, mask_color, pattern, r, c, corner, cut_h, cut_w = matches[0]
    r2 = _SIZE - H - int(r)
    c2 = _SIZE - W - int(c)
    opposite = pattern[r2 : r2 + H, c2 : c2 + W]
    out = np.where(opposite, int(fg), int(bg)).astype(int)

    reflected_corner = _opposite_corner(corner)
    occlusion = _make_corner_mask(H, W, reflected_corner, cut_h, cut_w)
    out[occlusion] = int(mask_color)
    return out


def generate_input() -> np.ndarray:
    """
    Generate a uniquely identifiable masked L crop whose target is the center-opposite L crop.

    The generator samples a full motif orientation first, chooses an HxW crop and a corner cut,
    then validates that the fused solver reconstructs the intended reflected L-shaped output.
    """
    rng = np.random.default_rng()
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    for _ in range(_GEN_RETRIES):
        fg, bg, mask_color = map(int, rng.choice(_PALETTE, size=3, replace=False))
        pattern = variants[int(rng.integers(len(variants)))]

        H = int(rng.integers(_ROW_RANGE[0], _ROW_RANGE[1] + 1))
        W = int(rng.integers(_COL_RANGE[0], _COL_RANGE[1] + 1))
        full_area = H * W
        corner = _CORNERS[int(rng.integers(len(_CORNERS)))]

        cut_h = int(rng.integers(2, H - 2))
        cut_w = int(rng.integers(2, W - 2))
        if H - cut_h < 3 or W - cut_w < 3:
            continue

        cut_area = cut_h * cut_w
        visible_area = full_area - cut_area
        if not (_VISIBLE_AREA_RANGE[0] <= visible_area <= _VISIBLE_AREA_RANGE[1]):
            continue
        if not (10 <= cut_area <= full_area // 3):
            continue

        r = int(rng.integers(0, _SIZE - H + 1))
        c = int(rng.integers(0, _SIZE - W + 1))
        r2 = _SIZE - H - int(r)
        c2 = _SIZE - W - int(c)
        if r2 == r and c2 == c:
            continue

        crop = np.where(pattern[r : r + H, c : c + W], fg, bg).astype(int)
        opposite = np.where(pattern[r2 : r2 + H, c2 : c2 + W], fg, bg).astype(int)

        input_mask = _make_corner_mask(H, W, corner, cut_h, cut_w)
        output_mask = _make_corner_mask(H, W, _opposite_corner(corner), cut_h, cut_w)
        crop[input_mask] = mask_color
        expected = opposite.copy()
        expected[output_mask] = mask_color

        if _corner_cut_info(crop == mask_color) is None:
            continue
        if len(colors_present(crop, exclude=[mask_color])) != 2:
            continue
        if len(colors_present(expected, exclude=[mask_color])) != 2:
            continue

        # Reject examples where the fused behavior collapses visually.
        if np.array_equal(crop, expected):
            continue
        if np.array_equal(crop[~input_mask], opposite[~output_mask]):
            continue

        try:
            solved = main(crop)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return crop

    raise RuntimeError("failed to generate a unique non-degenerate fused L-reflection sample")
