# Variant of 269e22fb
# Dimension: D2 — 源对象：裁剪窗口的形状 (src)
# Change: 矩形裁剪窗口，行数8-16，列数8-12，面积80-156... → 条形裁剪窗口（极端纵横比）

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# yaml
# variant:
#   task: Complete a two-color strip crop into the full 20x20 binary motif
#   preserved_core:
#     - The hidden object is still one dihedral variant of the same fixed 20x20 binary motif.
#     - The output is still the full 20x20 motif repainted with the strip's two colors.
#     - The solver still identifies orientation/offset by exact boolean-window matching once the strip is extracted.
#   d2_change:
#     from:
#       crop_shape: medium rectangle, roughly 8-16 by 8-12
#     to:
#       crop_shape: extreme strip, either horizontal or vertical
#       allowed_sizes:
#         - [3, 19]
#         - [19, 3]
#         - [4, 15]
#         - [15, 4]
#         - [4, 16]
#         - [16, 4]
#         - [4, 17]
#         - [17, 4]
#         - [4, 18]
#         - [18, 4]
#         - [4, 19]
#         - [19, 4]
#         - [5, 15]
#         - [15, 5]
#         - [5, 16]
#         - [16, 5]
#         - [5, 17]
#         - [17, 5]
#         - [5, 18]
#         - [18, 5]
#         - [5, 19]
#         - [19, 5]
#   concretization_and_trace:
#     - Pixel-wise, the input shows only a narrow band cut from the motif, embedded inside a uniform padding color on a 20x20 canvas: for example a 4x18 row strip or an 18x4 column strip.
#     - Original flow: pick a chunky rectangle -> match it against all pattern placements -> one match remains.
#     - New flow: locate the non-padding strip -> match that strip against all pattern placements -> the strip still resolves to one match, but because its 2D support is much thinner, the visible information is distributed mostly along one axis.
#     - Difference: the solver must first recover the true strip support, and the generator must control crop dimensions carefully to avoid ambiguous windows.
#   complexity_calibration:
#     - Very thin strips such as 2x18 are sometimes ambiguous.
#     - The allowed size list above was chosen because those shapes are still visually long-and-thin, yet every possible placement across all 8 dihedral variants is uniquely identifiable under either color polarity.
#     - This keeps the D2 change observable without introducing extra objects or branching rules.
#   hidden_assumption_audit_main:
#     - `H <= 20 and W <= 20`: still valid because strips are subwindows of the 20x20 motif.
#     - `len(colors) == 2` is no longer valid on the full canvas because a third padding color is present; the solver must identify the padding color and crop to the strip's bbox first.
#     - Trying each strip color as foreground remains necessary because polarity is not externally marked.
#     - Enumerating every pattern, offset, and comparing the boolean crop remains valid because the strip is still an axis-aligned crop, not a sparse mask with holes.
#     - `assert len(matches) == 1` becomes more fragile under D2, so generator construction must guarantee uniqueness rather than relying on the old rectangle ranges.
#   placement_legality:
#     - Every strip is sampled as an axis-aligned subwindow of the 20x20 output, so it never crosses boundaries and never overlaps unrelated objects because there are no additional objects.
#     - Offsets use `r in [0, 20-H]`, `c in [0, 20-W]`.
#   visual_observability:
#     - Humans can see the D2 change immediately because the input is no longer a block-like patch; it is a long narrow slice.
#     - The output remains the same type of full-motif reconstruction, so the changed role of crop shape is the only new inference burden.
#   visual_cleanliness:
#     - The task stays visually clean: exactly one strip in the input, surrounded by one flat padding color, and one full motif in the output.
#   color_inference:
#     - No color-id-dependent rule is introduced. Colors only repaint foreground/background exactly as before.
#   generator_guarantees_for_main:
#     - Inputs always use one of the prevalidated strip sizes.
#     - Inputs always contain one padding color plus exactly two strip colors sampled from the ARC palette.
#     - Each generated strip-on-canvas input is output-first verified by `main(input) == output`.


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


def _is_extreme_strip(h: int, w: int) -> bool:
    return (h, w) in _STRIP_SHAPES


def _extract_strip(grid: np.ndarray) -> np.ndarray:
    colors = colors_present(grid)
    assert len(colors) == 3
    g = np.asarray(grid, dtype=int)

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
        if np.any(mask[:r0, :]) or np.any(mask[r1 + 1 :, :]) or np.any(mask[:, :c0]) or np.any(mask[:, c1 + 1 :]):
            continue
        strip = g[r0 : r1 + 1, c0 : c1 + 1]
        if _is_extreme_strip(*strip.shape) and len(colors_present(strip)) == 2:
            return strip

    raise AssertionError("expected a unique padded extreme strip")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared 20x20 binary motif from a padded long, thin crop in the correct orientation.

    Input structure:
    - The grid is a 20x20 canvas with one flat padding color plus a two-color strip cut from one rotation/reflection of the motif.
    - The non-padding strip is extremely elongated: a narrow horizontal band or a narrow vertical band.

    Rule:
    - Find the bbox of the non-padding strip and discard the padding field.
    - Treat each of the two strip colors as the possible motif foreground in turn.
    - Match that boolean strip against every offset in every dihedral variant of the motif.
    - Repaint the uniquely matched full 20x20 motif using the recovered foreground/background colors.

    Key insight:
    - The visible evidence is only a narrow band, so the meaningful support is the extracted strip rather than the whole padded canvas.
    """
    g = np.asarray(grid, dtype=int)
    strip = _extract_strip(g)
    H, W = map(int, strip.shape)
    assert H <= 20 and W <= 20
    assert _is_extreme_strip(H, W)

    colors = colors_present(strip)
    assert len(colors) == 2

    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = strip == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, int | tuple[int, int]]]:
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    fg, bg, pad = map(int, rng.choice(Color.ALL_COLORS, size=3, replace=False))
    pattern_idx = int(rng.integers(len(variants)))
    pattern = variants[pattern_idx]
    output = np.where(pattern, fg, bg).astype(int)

    H, W = _STRIP_SHAPES[int(rng.integers(len(_STRIP_SHAPES)))]
    r = int(rng.integers(0, 20 - H + 1))
    c = int(rng.integers(0, 20 - W + 1))
    strip = output[r : r + H, c : c + W].copy()
    canvas = np.full((20, 20), pad, dtype=int)
    top = int(rng.integers(0, 20 - H + 1))
    left = int(rng.integers(0, 20 - W + 1))
    canvas[top : top + H, left : left + W] = strip

    meta = {
        "shape": (H, W),
        "motif_offset": (r, c),
        "canvas_offset": (top, left),
        "pattern_idx": pattern_idx,
        "fg": fg,
        "bg": bg,
        "pad": pad,
    }
    return canvas, output, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    input_grid, output_grid, _ = _generate_case(rng)
    assert np.array_equal(main(input_grid), output_grid)
    return input_grid
