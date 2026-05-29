# Combo variant of 269e22fb (index 12)
# combo_spec: D2/1+D3/1
# Name: 条形输入补集输出

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# yaml
# combo:
#   task: Recover a dihedral motif from a padded extreme strip, then output the motif
#     with that matched strip removed, using the strip's two colors.
#   fusion_summary:
#     - D2 changes the source evidence from a chunky crop to a very thin horizontal or
#       vertical strip shown on a separate 20x20 padding canvas.
#     - D3 changes the destination from "reconstruct the whole motif" to "show only the
#       complement of the matched strip window", with an explicit tight-crop step.
#     - The two changes interact: the solver must first ignore the strip's canvas
#       position, then infer the hidden motif position of that strip, and only then can
#       it erase the correct band from the recovered motif.
#   concretization_and_trace:
#     - Concrete example: pick a 4x18 strip from motif rows [0:4), cols [1:19). Because
#       it touches exactly one outer border of the 20x20 motif, removing it leaves a
#       large U-shaped geometric remainder.
#     - Original task flow: visible rectangle directly encodes one local window ->
#       unique orientation/offset match -> repaint full 20x20 motif.
#     - Fused flow: the same motif window is first relocated onto an unrelated padding
#       canvas -> solver extracts the non-padding strip -> unique orientation/offset
#       match is found from the thin evidence -> that matched strip window is erased from
#       the recovered motif -> the remaining region is tight-cropped.
#     - Difference from the original: the input support is much thinner and the output is
#       subtractive rather than completive, so both the extraction step and the matched
#       offset are semantically important.
#   complexity_calibration:
#     - Very thin strips can become ambiguous, so the generator reuses a curated set of
#       extreme strip sizes that are still uniquely identifiable on this motif.
#     - To make the complement visibly U-shaped rather than an arbitrary hole, motif
#       offsets are restricted to placements with exactly one border contact.
#     - The tight-crop operation is still implemented explicitly; for these one-border
#       U-shapes the remainder already reaches all four outer extremes, so the crop is
#       usually vacuous and the output stays 20x20. This is acceptable because the core
#       D3 behavior is the erased strip, not forced canvas shrinkage.
#   hidden_assumption_audit_main:
#     - The original `len(colors_present(grid)) == 2` assumption is false for the full
#       input canvas, because the fused input has one padding color plus the strip's two
#       motif colors.
#     - The original code could treat the full input as the matched window; the fused
#       solver must instead isolate one rectangular non-padding strip and validate that
#       its bbox is one of the allowed extreme aspect ratios.
#     - The original output no longer suffices once a unique match is found: `r` and `c`
#       must be preserved because they locate the strip window to erase.
#     - Tight-cropping must be driven by the geometric complement mask, not by motif
#       foreground pixels; otherwise the crop would depend on the particular color layout
#       of the motif rather than on the intended removed region.
#   placement_legality:
#     - The hidden strip always comes from an axis-aligned subwindow of a 20x20 motif, so
#       motif offsets obey `0 <= r <= 20-H` and `0 <= c <= 20-W`.
#     - Exactly one border contact is enforced in motif coordinates:
#       `int(r==0)+int(r+H==20)+int(c==0)+int(c+W==20) == 1`.
#     - The displayed strip is pasted onto a separate 20x20 padding canvas at any legal
#       top-left `(top, left)` with `0 <= top <= 20-H`, `0 <= left <= 20-W`. There is
#       only one object, so no overlap bookkeeping is needed.
#   visual_observability:
#     - Across examples, humans see an obvious long strip in the input and a much larger
#       motif-like output with a conspicuous missing band of the same shape.
#     - Because the strip can be horizontal or vertical and the missing band can occur in
#       different motif offsets, the complement rule is visible from a few examples
#       without depending on absolute color ids.
#   visual_cleanliness:
#     - Input: exactly one padded strip, with no extra markers or clutter.
#     - Output: exactly two colors, preserving the motif texture but blanking the matched
#       strip rectangle, so the removed region reads cleanly as a hole/U-shape.
#   inferability_of_change:
#     - Color semantics remain structural only: the two strip colors become motif
#       foreground/background, and the deleted band is reset to the recovered background.
#     - No hidden ordering by numeric color id is used; all reasoning is based on visible
#       shape, orientation, and spatial subtraction.
#   generator_guarantees_for_main:
#     - Every chosen strip shape comes from a prevalidated list whose one-border placements
#       have unique matches across all 8 dihedral motif variants.
#     - Every sampled strip contains both motif values, so polarity is meaningful and the
#       complement really removes at least one foreground pixel from the full motif.
#     - Each generated example is checked by `main(input) == expected_output`.


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
    (3, 19),
    (19, 3),
    (4, 15),
    (15, 4),
    (4, 16),
    (16, 4),
    (4, 17),
    (17, 4),
    (4, 18),
    (18, 4),
    (4, 19),
    (19, 4),
    (5, 15),
    (15, 5),
    (5, 16),
    (16, 5),
    (5, 17),
    (17, 5),
    (5, 18),
    (18, 5),
    (5, 19),
    (19, 5),
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


_ALL_PATTERNS = _dihedral_variants(_pattern_mask())


def _is_extreme_strip(h: int, w: int) -> bool:
    return (int(h), int(w)) in _STRIP_SHAPES


def _border_offsets(h: int, w: int) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for r in range(20 - int(h) + 1):
        for c in range(20 - int(w) + 1):
            contacts = int(r == 0) + int(r + h == 20) + int(c == 0) + int(c + w == 20)
            if contacts == 1:
                offsets.append((int(r), int(c)))
    return tuple(offsets)


_BORDER_OFFSETS = {shape: _border_offsets(*shape) for shape in _STRIP_SHAPES}


def _border_side(h: int, w: int, r: int, c: int) -> str:
    if r == 0:
        return "top"
    if r + h == 20:
        return "bottom"
    if c == 0:
        return "left"
    return "right"


def _extract_strip(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g)
    assert len(colors) == 3

    for pad in colors:
        mask = g != int(pad)
        if not np.any(mask):
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0 = int(np.argmax(rows))
        r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        c0 = int(np.argmax(cols))
        c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        strip = g[r0 : r1 + 1, c0 : c1 + 1]
        if not _is_extreme_strip(*strip.shape):
            continue
        if np.any(strip == int(pad)):
            continue
        if np.any(g[:r0, :] != int(pad)) or np.any(g[r1 + 1 :, :] != int(pad)):
            continue
        if np.any(g[:, :c0] != int(pad)) or np.any(g[:, c1 + 1 :] != int(pad)):
            continue
        if len(colors_present(strip)) != 2:
            continue
        return strip

    raise AssertionError("expected one padded extreme strip with two motif colors")


def _find_unique_match(strip: np.ndarray) -> tuple[int, int, np.ndarray, int, int]:
    g = np.asarray(strip, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 20 and w <= 20
    assert _is_extreme_strip(h, w)

    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _ALL_PATTERNS:
            for r in range(20 - h + 1):
                for c in range(20 - w + 1):
                    if np.array_equal(pattern[r : r + h, c : c + w], visible):
                        matches.append((int(fg), int(bg), pattern, int(r), int(c)))

    assert len(matches) == 1
    return matches[0]


def _render_complement_tight(
    pattern: np.ndarray, fg: int, bg: int, r: int, c: int, h: int, w: int
) -> np.ndarray:
    keep = np.ones((20, 20), dtype=bool)
    keep[int(r) : int(r + h), int(c) : int(c + w)] = False

    full = np.where(pattern, int(fg), int(bg)).astype(int)
    full[~keep] = int(bg)

    coords = np.argwhere(keep)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return full[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared motif from a padded thin strip, then return only the strip's complement.

    Input structure:
    - The grid is a 20x20 padding field containing one two-color strip cut from some rotation/reflection of the fixed 20x20 motif.
    - The informative object is an extreme-aspect-ratio band, either narrow-horizontal or narrow-vertical; the third color is padding only.

    Rule:
    - Detect the non-padding strip, crop it to its bbox, and treat each of its two colors as the possible motif foreground in turn.
    - Match that boolean strip against every offset in every dihedral motif orientation until one orientation, polarity, and motif offset remains.
    - Repaint the matched motif with the recovered foreground/background colors, erase the matched strip rectangle back to background, and tight-crop the remaining geometric complement.

    Key insight:
    - The strip's position on the padding canvas is irrelevant; what matters is where that thin local band fits inside the hidden motif, because that recovered motif offset is exactly the band that must be removed in the output.
    """
    strip = _extract_strip(np.asarray(grid, dtype=int))
    h, w = map(int, strip.shape)
    fg, bg, pattern, r, c = _find_unique_match(strip)
    return _render_complement_tight(pattern, fg, bg, r, c, h, w)


def _generate_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | tuple[int, int]]]:
    fg, bg, pad = map(int, rng.choice(Color.ALL_COLORS, size=3, replace=False))
    pattern_idx = int(rng.integers(len(_ALL_PATTERNS)))
    pattern = _ALL_PATTERNS[pattern_idx]

    h, w = _STRIP_SHAPES[int(rng.integers(len(_STRIP_SHAPES)))]
    offsets_by_side: dict[str, list[tuple[int, int]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for r, c in _BORDER_OFFSETS[(h, w)]:
        offsets_by_side[_border_side(h, w, r, c)].append((r, c))
    available_sides = [side for side, side_offsets in offsets_by_side.items() if side_offsets]
    border_side = available_sides[int(rng.integers(len(available_sides)))]
    offsets = offsets_by_side[border_side]
    r, c = offsets[int(rng.integers(len(offsets)))]

    strip_mask = pattern[r : r + h, c : c + w]
    assert strip_mask.any() and (~strip_mask).any()
    strip = np.where(strip_mask, int(fg), int(bg)).astype(int)

    canvas = np.full((20, 20), int(pad), dtype=int)
    top = int(rng.integers(0, 20 - h + 1))
    left = int(rng.integers(0, 20 - w + 1))
    canvas[top : top + h, left : left + w] = strip

    expected = _render_complement_tight(pattern, fg, bg, r, c, h, w)
    full = np.where(pattern, int(fg), int(bg)).astype(int)
    erased_fg = int(strip_mask.sum())

    assert len(colors_present(strip)) == 2
    assert expected.shape == (20, 20)
    assert erased_fg > 0
    assert not np.array_equal(expected, full)

    meta = {
        "shape": (int(h), int(w)),
        "motif_offset": (int(r), int(c)),
        "canvas_offset": (int(top), int(left)),
        "pattern_idx": int(pattern_idx),
        "border_contacts": int(r == 0) + int(r + h == 20) + int(c == 0) + int(c + w == 20),
        "erased_fg": int(erased_fg),
    }
    return canvas, expected, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(1000):
        input_grid, output_grid, _meta = _generate_case(rng)
        if np.array_equal(main(input_grid), output_grid):
            return input_grid
    raise RuntimeError("failed to generate a valid combo variant input")
