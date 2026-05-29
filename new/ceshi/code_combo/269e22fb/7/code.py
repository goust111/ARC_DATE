# Combo variant of 269e22fb (index 7)
# combo_spec: D8/0+D14/0
# Name: 带边框和图例的三色输入

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The base task showed only an HxW two-color crop from one dihedral variant of the
#    fixed 20x20 motif, and main() brute-forced both visible colors as possible motif
#    foreground. In this combo, the visible crop is no longer the whole input. A
#    concrete 10x8 crop becomes a 14x13 canvas:
#      - the actual crop sits at rows 2:12, cols 3:11,
#      - that crop is surrounded by a one-cell border of a third color at row 1/12 and
#        col 2/11,
#      - outside that ring there is one more border-colored padding row on top/bottom
#        and one padding column on the right,
#      - the far-left edge contains a 2x1 foreground legend swatch stacked above a 2x1
#        background legend swatch, separated from the crop border by a border-colored
#        column.
#    So the solver flow changes from "match the whole grid and infer polarity" to
#    "strip metadata regions, read the legend, then match only the interior crop with a
#    fixed polarity".
# b) Complexity calibration:
#    The extra structure is visible but tightly controlled. Only three colors appear in
#    the input, the legend location is fixed, the crop border is exactly one cell thick,
#    and the crop size stays inside the original 8-16 by 8-12 window. This preserves the
#    original orientation/offset identification challenge instead of replacing it with a
#    loose layout puzzle.
# c) Implicit assumptions broken by the fusion:
#    - Base `len(colors) == 2` no longer holds; the full input now has exactly 3 colors.
#    - Base `g.shape` is not the crop size; the true crop is `g[2:-2, 3:-2]`.
#    - Base polarity brute force is no longer the intended semantics; the legend fixes
#      which crop color is motif-1 and which is motif-0.
#    - Base "every cell belongs to the crop" is false because most border-colored cells
#      are metadata and must be ignored during motif matching.
#    The new solver therefore assumes a fixed metadata layout and a directed fg/bg legend.
# d) Placement legality:
#    Generator still uses output-first sampling: choose a motif orientation, paint the
#    full 20x20 output, extract a legal crop, and only then wrap it in the border/legend
#    canvas. Because the crop occupies `rows 2:-2, cols 3:-2`, the ring, legend, and
#    padding never overlap or go out of bounds. The minimum crop height 8 leaves ample
#    room for the 4-cell tall legend on the left edge.
# e) Visual observability:
#    Across 2-4 examples, a human can see three stable regions: a large interior crop, a
#    uniform third-color border around that crop, and a small left-edge legend whose top
#    swatch matches the output foreground color while the lower swatch matches the output
#    background color. The border color never appears in the output, so its metadata role
#    is directly inferable.
# f) Visual cleanliness:
#    The legend never touches the crop because column 1 is always border-colored and the
#    crop border itself starts at column 2. The crop is the only large non-metadata
#    region, so there is no ambiguous overlap between annotation and content.
# g) Change inferability:
#    The fusion keeps the color vocabulary small enough for ARC: two semantic colors are
#    repeated both inside the crop and again in the legend, while the third color repeats
#    only in the border/padding. That repeated support makes the mapping rule observable
#    rather than arbitrary.
# h) Anti-degeneration target:
#    Samples must always satisfy all of the following:
#    - exactly 3 input colors,
#    - a non-empty interior crop using exactly the 2 legend colors,
#    - a full one-cell crop border in the third color,
#    - the wrong legend polarity yields zero directed motif matches.
#    These constraints ensure the produced inputs really exercise both the border-stripping
#    step and the legend-reading step.
# i) New solver assumptions audited against the generator:
#    - `_extract_layout()` assumes fixed margins and a 2x1+2x1 legend layout.
#      `_build_input()` constructs exactly that geometry.
#    - `_extract_layout()` assumes all non-crop, non-legend cells are border-colored.
#      `_build_input()` fills the full canvas with the border color before inserting the
#      crop and legend swatches.
#    - `main()` assumes the crop contains both legend colors and that the directed crop
#      has a unique motif match. `generate_input()` rejects any crop failing those tests.


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

_CROP_H_RANGE = (8, 16)
_CROP_W_RANGE = (8, 12)
_PALETTE = tuple(range(10))
_MAX_ATTEMPTS = 1000

_LEGEND_FG_ROWS = slice(1, 3)
_LEGEND_BG_ROWS = slice(3, 5)
_LEGEND_COL = 0
_CROP_R0 = 2
_CROP_C0 = 3
_BOTTOM_MARGIN = 2
_RIGHT_MARGIN = 2


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


def _find_directed_matches(crop: np.ndarray, fg: int) -> list[np.ndarray]:
    crop = np.asarray(crop, dtype=int)
    H, W = map(int, crop.shape)
    assert H <= 20 and W <= 20
    visible = crop == int(fg)

    matches: list[np.ndarray] = []
    for pattern in _VARIANTS:
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                if np.array_equal(pattern[r : r + H, c : c + W], visible):
                    matches.append(pattern)
    return matches


def _find_any_polarity_matches(crop: np.ndarray) -> list[np.ndarray]:
    crop = np.asarray(crop, dtype=int)
    colors = colors_present(crop)
    assert len(colors) == 2

    matches: list[np.ndarray] = []
    for fg in colors:
        matches.extend(_find_directed_matches(crop, int(fg)))
    return matches


def _extract_layout(grid: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert _CROP_H_RANGE[0] + 4 <= H <= _CROP_H_RANGE[1] + 4
    assert _CROP_W_RANGE[0] + 5 <= W <= _CROP_W_RANGE[1] + 5

    colors = colors_present(g)
    assert len(colors) == 3

    crop = g[_CROP_R0:-_BOTTOM_MARGIN, _CROP_C0:-_RIGHT_MARGIN]
    ch, cw = map(int, crop.shape)
    assert _CROP_H_RANGE[0] <= ch <= _CROP_H_RANGE[1]
    assert _CROP_W_RANGE[0] <= cw <= _CROP_W_RANGE[1]

    fg = int(g[_LEGEND_FG_ROWS.start, _LEGEND_COL])
    bg = int(g[_LEGEND_BG_ROWS.start, _LEGEND_COL])
    assert fg != bg
    assert np.all(g[_LEGEND_FG_ROWS, _LEGEND_COL] == fg)
    assert np.all(g[_LEGEND_BG_ROWS, _LEGEND_COL] == bg)

    border_mask = np.ones_like(g, dtype=bool)
    border_mask[_CROP_R0:-_BOTTOM_MARGIN, _CROP_C0:-_RIGHT_MARGIN] = False
    border_mask[_LEGEND_FG_ROWS, _LEGEND_COL] = False
    border_mask[_LEGEND_BG_ROWS, _LEGEND_COL] = False
    border_colors = colors_present(g[border_mask])
    assert len(border_colors) == 1
    border = int(border_colors[0])
    assert border not in {fg, bg}

    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2
    assert set(map(int, crop_colors)) == {fg, bg}
    return crop, fg, bg, border


def _render_output(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _build_input(crop: np.ndarray, fg: int, bg: int, border: int) -> np.ndarray:
    H, W = map(int, crop.shape)
    inp = np.full((H + 4, W + 5), int(border), dtype=int)
    inp[_CROP_R0:-_BOTTOM_MARGIN, _CROP_C0:-_RIGHT_MARGIN] = crop
    inp[_LEGEND_FG_ROWS, _LEGEND_COL] = int(fg)
    inp[_LEGEND_BG_ROWS, _LEGEND_COL] = int(bg)
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from a bordered crop whose color meaning is declared by a side legend.

    Input structure:
    - The input uses exactly 3 colors.
    - A rectangular crop from one dihedral variant of the shared 20x20 binary motif appears inside the canvas.
    - That crop is enclosed by a one-cell border in the third color, and the far-left edge contains a 2x1 foreground swatch stacked above a 2x1 background swatch.

    Rule:
    - Ignore every border-colored metadata cell and isolate the true crop.
    - Read the left legend to decide which crop color corresponds to motif foreground versus background.
    - Match that directed boolean crop against every offset of every dihedral motif variant, then repaint the uniquely matched full 20x20 motif with the legend-declared foreground/background colors.

    Key insight:
    - The crop still comes from the same fixed 20x20 pattern as the base puzzle, but solving now requires two preprocessing steps before the old completion logic applies: strip the third-color border and use the edge legend instead of polarity brute force.
    """
    crop, fg, bg, _border = _extract_layout(np.asarray(grid, dtype=int))
    matches = _find_directed_matches(crop, fg)
    assert len(matches) == 1
    pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        fg, bg, border = random.sample(_PALETTE, 3)
        pattern = random.choice(_VARIANTS)
        full = _render_output(pattern, fg, bg)

        crop_h = random.randint(*_CROP_H_RANGE)
        crop_w = random.randint(*_CROP_W_RANGE)
        r0 = random.randint(0, 20 - crop_h)
        c0 = random.randint(0, 20 - crop_w)
        crop = full[r0 : r0 + crop_h, c0 : c0 + crop_w].copy()

        if set(colors_present(crop)) != {int(fg), int(bg)}:
            continue
        if len(_find_directed_matches(crop, fg)) != 1:
            continue
        if len(_find_directed_matches(crop, bg)) != 0:
            continue

        inp = _build_input(crop, fg, bg, border)
        if np.array_equal(main(inp), full):
            return inp

    raise RuntimeError("failed to generate a valid combo sample")


def variant_metrics(grid: np.ndarray) -> dict[str, int | bool | float]:
    g = np.asarray(grid, dtype=int)
    crop, fg, bg, border = _extract_layout(g)

    border_mask = np.ones_like(g, dtype=bool)
    border_mask[_CROP_R0:-_BOTTOM_MARGIN, _CROP_C0:-_RIGHT_MARGIN] = False
    border_mask[_LEGEND_FG_ROWS, _LEGEND_COL] = False
    border_mask[_LEGEND_BG_ROWS, _LEGEND_COL] = False

    directed = _find_directed_matches(crop, fg)
    wrong_directed = _find_directed_matches(crop, bg)
    undirected = _find_any_polarity_matches(crop)
    return {
        "input_color_count": len(colors_present(g)),
        "crop_height": int(crop.shape[0]),
        "crop_width": int(crop.shape[1]),
        "border_distinct": bool(border not in {fg, bg}),
        "legend_top_is_fg": bool(np.all(g[_LEGEND_FG_ROWS, _LEGEND_COL] == fg)),
        "legend_bottom_is_bg": bool(np.all(g[_LEGEND_BG_ROWS, _LEGEND_COL] == bg)),
        "border_cell_count": int(np.sum(border_mask)),
        "metadata_fraction": float(1.0 - crop.size / g.size),
        "directed_match_count": int(len(directed)),
        "wrong_legend_match_count": int(len(wrong_directed)),
        "undirected_match_count": int(len(undirected)),
    }
