# Variant of 269e22fb
# Dimension: D14 — 符号编码：颜色与图案的映射方式 (symbol_instructions)
# Change: 二值图案直接映射为两种颜色，无额外编码层... → 输入包含变体索引的色块编码

from __future__ import annotations

import numpy as np

from arc_utils import colors_present


# Design analysis
# ---------------
# a) Concrete layout and behavior trace
#    - Original input: only a two-color crop, so the solver had to search all 8 dihedral variants.
#    - Variant input: prepend a 4-row header. Inside its top-left 3x3 corner, a third color fills N cells
#      in row-major order; N-1 is the dihedral variant index in the same order returned by
#      _dihedral_variants(). Rows 4..end remain the original rectangular two-color crop.
#    - Example trace:
#      original: crop -> try both polarities -> try 8 variants -> slide every offset -> keep unique match.
#      variant: count marker cells in header -> variant index fixed -> try both polarities only inside that
#      one variant -> slide offsets -> keep unique match.
#      difference: the new symbol layer carries orientation, so the crop only has to determine offset/polarity.
# b) Complexity calibration
#    - The marker has only 8 possible counts, matching the 8 dihedral states exactly.
#    - The crop itself is unchanged, so the main visual object is still the motif window; the header is a small,
#      clean extra clue rather than a second large puzzle.
#    - The generator prefers small windows when possible and records whether the encoded variant actually
#      reduces ambiguity across multiple dihedral candidates.
# c) Original main() assumptions that no longer hold
#    - "The whole grid is the crop": false now; rows 0..3 are metadata.
#    - "Exactly two colors in the input": false now; there are exactly three total colors, but still only two
#      inside the crop.
#    - "Search every dihedral orientation": no longer desired; the count code chooses one orientation.
#    - "H and W are the crop dimensions": false now; crop height is total height minus the 4-row header.
# d) Placement legality
#    - Crop size stays within the original 8..16 by 8..12 range, so the total input size is at most 20x12.
#    - The 3x3 marker always fits because width >= 8 and header height is fixed at 4.
#    - Header cells use only background + code color, so they never interfere with crop matching once removed.
# e) Visual observability
#    - A human sees a small counted color block in the corner plus the same style of crop as before.
#    - Across examples, different counts pair with different completed orientations, making the extra rule
#      visible without relying on hidden color IDs or non-spatial metadata.
# f) Visual cleanliness
#    - Metadata is isolated in a header; the crop is still a plain rectangle.
#    - No object overlaps: the marker is above the crop and uses a dedicated third color.
# g) Color inference
#    - Only three input colors are used: motif fg, motif bg, and marker color.
#    - Output ignores the marker color and repaints the full motif with the crop's two colors.
#
# New solver assumptions checked against generator
# -----------------------------------------------
# - The crop starts at row HEADER_H and spans the full width. generate_input() always builds that layout.
# - The crop contains exactly two colors and both appear at least once. generator filters out monochrome windows.
# - The marker color appears only in the header. generator chooses three distinct colors and never uses the
#   marker color in the crop.
# - The 3x3 marker contains between 1 and 8 coded cells. generator encodes index+1, so count is always valid.
# - The encoded variant has exactly one valid offset/polarity match. generator validates this before returning.


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

HEADER_H = 4
CODE_BOX = 3
MAX_ATTEMPTS = 400


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _other_color(colors: list[int], chosen: int) -> int:
    return int(colors[0] if int(colors[1]) == int(chosen) else colors[1])


def _find_matches(crop: np.ndarray, patterns: list[np.ndarray]) -> list[tuple[int, int, int, int, int]]:
    crop = np.asarray(crop, dtype=int)
    h, w = map(int, crop.shape)
    colors = colors_present(crop)
    assert len(colors) == 2

    matches: list[tuple[int, int, int, int, int]] = []
    for variant_idx, pattern in enumerate(patterns):
        for fg in colors:
            bg = _other_color(colors, int(fg))
            visible = crop == int(fg)
            for r in range(20 - h + 1):
                for c in range(20 - w + 1):
                    if np.array_equal(pattern[r : r + h, c : c + w], visible):
                        matches.append((variant_idx, int(fg), int(bg), r, c))
    return matches


def _encode_header(width: int, bg: int, code_color: int, variant_idx: int) -> np.ndarray:
    header = np.full((HEADER_H, width), int(bg), dtype=int)
    count = int(variant_idx) + 1
    coords = [(r, c) for r in range(CODE_BOX) for c in range(CODE_BOX)]
    for r, c in coords[:count]:
        header[r, c] = int(code_color)
    return header


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif after reading a counted corner marker for orientation.

    Input structure:
    - Rows 4..end form the original rectangular two-color crop of the shared motif.
    - The top-left 3x3 corner in the 4-row header uses a third color; its cell count encodes which dihedral variant was used.

    Rule:
    - Remove the header, count the marker cells, and convert that count into the single allowed dihedral orientation.
    - On that one orientation only, try both crop colors as foreground polarity and slide the crop over every offset.
    - Repaint the full 20x20 motif using the recovered foreground/background colors from the crop.

    Key insight:
    - The extra counted symbol carries the orientation choice, so the crop no longer needs to distinguish among all 8 rotations/reflections; it only identifies offset and polarity inside the encoded variant.
    """
    g = np.asarray(grid, dtype=int)
    total_h, total_w = map(int, g.shape)
    assert HEADER_H + 8 <= total_h <= HEADER_H + 16
    assert 8 <= total_w <= 12

    crop = g[HEADER_H:, :]
    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2

    all_colors = colors_present(g)
    assert len(all_colors) == 3
    code_colors = [c for c in all_colors if c not in crop_colors]
    assert len(code_colors) == 1
    code_color = int(code_colors[0])

    count = int(np.sum(g[:CODE_BOX, :CODE_BOX] == code_color))
    assert 1 <= count <= 8
    variant_idx = count - 1

    base = _pattern_mask()
    pattern = _dihedral_variants(base)[variant_idx]
    matches = _find_matches(crop, [pattern])
    assert len(matches) == 1

    _variant_zero, fg, bg, _r, _c = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _sample_crop_shape(rng: np.random.Generator) -> tuple[int, int]:
    height_choices = np.array([8, 8, 9, 9, 10, 10, 11, 12, 13, 14, 15, 16], dtype=int)
    width_choices = np.array([8, 8, 9, 9, 10, 10, 11, 12], dtype=int)
    h = int(height_choices[int(rng.integers(0, len(height_choices)))])
    w = int(width_choices[int(rng.integers(0, len(width_choices)))])
    return h, w


def _candidate_windows(
    pattern: np.ndarray,
    fg: int,
    bg: int,
    variants: list[np.ndarray],
    crop_h: int,
    crop_w: int,
) -> list[tuple[int, int, np.ndarray, int]]:
    chosen_only = [pattern]
    candidates: list[tuple[int, int, np.ndarray, int]] = []
    for r in range(20 - crop_h + 1):
        for c in range(20 - crop_w + 1):
            window = pattern[r : r + crop_h, c : c + crop_w]
            if window.all() or (~window).all():
                continue
            crop = np.where(window, int(fg), int(bg)).astype(int)
            local_matches = _find_matches(crop, chosen_only)
            if len(local_matches) != 1:
                continue
            global_matches = _find_matches(crop, variants)
            candidates.append((r, c, crop, len(global_matches)))
    return candidates


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    palette = rng.choice(np.arange(10), size=3, replace=False)
    fg, bg, code_color = map(int, palette)
    variant_idx = int(rng.integers(0, len(variants)))
    pattern = variants[variant_idx]

    crop_h, crop_w = _sample_crop_shape(rng)
    candidates = _candidate_windows(pattern, fg, bg, variants, crop_h, crop_w)
    if not candidates:
        return None

    ambiguous = [cand for cand in candidates if cand[3] > 1]
    pool = ambiguous if ambiguous else candidates
    r, c, crop, _global_count = pool[int(rng.integers(0, len(pool)))]

    header = _encode_header(crop_w, bg, code_color, variant_idx)
    inp = np.vstack([header, crop])
    out = np.where(pattern, int(fg), int(bg)).astype(int)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
