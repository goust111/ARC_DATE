# Combo variant of b0039139 (index 11)
# combo_spec: D6/0+D9/1
# Name: 多色分隔线加2x2色板编码

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the fused D6/0 + D9/1 variant:
# a) Concrete behavior and trace:
#    - The preserved backbone is still "crop the template, recolor it, then repeat it by the number
#      of 4-connected blobs in the stencil".
#    - Original trace:
#      [template | blue | stencil | blue | fg block | blue | bg block]
#      -> read sections by fixed order -> take solid fg/bg -> tile cropped template N times.
#    - Fused trace:
#      [template | red | stencil | blue | palette] or [palette | blue | stencil | red | template]
#      -> first decode section roles from separator colors (red means template/stencil boundary,
#         blue means stencil/palette boundary)
#      -> then crop the lone 2x2 code inside the merged palette and read main-diagonal = fg,
#         anti-diagonal = bg
#      -> tile the cropped template N times exactly as before.
#    - The key difference is that neither the section order nor the colors can be read by the old
#      fixed-position assumptions anymore: the solver must use both boundary-color decoding and the
#      2x2 palette rule.
# b) Complexity calibration:
#    - Only two separator colors are used, with a fixed semantic meaning across all examples. This
#      keeps the inference load human-solvable while still making the layout visibly different.
#    - The layout is allowed to appear in forward or reversed order, which makes the separator colors
#      necessary rather than decorative.
#    - The palette still uses exactly two colors in a fixed 2x2 arrangement, so the symbolic rule is
#      observable without introducing too many free degrees of freedom.
# c) Hidden assumptions from the original main():
#    - `_split_sections(...)` can no longer search only for blue separators; it must accept multiple
#      full-line colors and preserve their order.
#    - `sections[:4]` is invalid because D9/1 merges the two swatches into one palette section.
#    - `_solid_color(...)` on the palette is invalid because the merged palette is mostly black with
#      a small informative 2x2 token.
#    - Fixed semantic order is invalid once the generator can reverse the three sections.
#    - The repeat axis is still tied to whether separators are columns or rows, so that part remains.
# d) Placement legality:
#    - Every separator is a single full row/column and sections never contain their own full nonzero
#      row/column, so splitting is unambiguous.
#    - The palette section is at least 4 cells along both axes, guaranteeing a 2x2 code with a
#      1-cell black margin.
#    - Stencil blobs keep the original spacing rule, so component counting is stable.
# e) Visual observability:
#    - Across examples, humans can see that one boundary is red and another is blue, and that the
#      middle section touching both colors is the counting stencil.
#    - The output uses exactly the two colors appearing in the diagonal palette token, supporting the
#      positional mapping from palette to recolored template.
# f) Visual cleanliness:
#    - Template, stencil, and palette occupy separate regions with only one symbolic element inside
#      the palette. This avoids layered marks competing in the same area.
#    - Foreground/background output colors are drawn from colors not used by separators or scaffold
#      symbols, which keeps the code panel visually distinct.
# g) Change inferability:
#    - The separator-color meanings are categorical and repeated across all samples: red always marks
#      the template/stencil boundary, blue always marks the stencil/palette boundary.
#    - The palette token always uses the same checker rule, so a few examples are enough to infer
#      which diagonal is foreground and which is background.
# h) Quantifiable anti-degeneracy target:
#    - A non-degenerate sample must have both separator colors present and exactly one merged palette
#      section containing a 2x2 non-black bbox.
#    - The generator also randomizes forward vs reversed semantic order; reversed samples cannot be
#      solved by a fixed left-to-right or top-to-bottom section assumption.
# i) New solver assumptions guaranteed by the generator:
#    - There are exactly three semantic sections and exactly two separators.
#    - Separator colors are exactly {red=2, blue=1} in one of the two legal orders: (2,1) or (1,2).
#    - The palette's non-black bbox is exactly 2x2 with matching diagonals and distinct decoded
#      colors. The template contains at least one non-black cell, and the stencil contains 2-6 blobs.


TEMPLATE_STENCIL_SEPARATOR = 2
STENCIL_PALETTE_SEPARATOR = 1

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = [5, 6, 7, 8, 9]
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
PALETTE_DIM_RANGE = (4, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 300


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray], list[int]]:
    """Split on full nonzero separator rows/cols and return orientation, sections, separator colors."""
    h, w = grid.shape

    sep_rows = [(r, int(grid[r, 0])) for r in range(h) if int(grid[r, 0]) != 0 and bool(np.all(grid[r, :] == grid[r, 0]))]
    if sep_rows:
        cuts = [-1] + [r for r, _ in sep_rows] + [h]
        sections = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", sections, [color for _, color in sep_rows]

    sep_cols = [(c, int(grid[0, c])) for c in range(w) if int(grid[0, c]) != 0 and bool(np.all(grid[:, c] == grid[0, c]))]
    cuts = [-1] + [c for c, _ in sep_cols] + [w]
    sections = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", sections, [color for _, color in sep_cols]


def _assign_sections(sections: list[np.ndarray], sep_colors: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use separator colors to identify template, stencil, and palette sections."""
    assert len(sections) == 3, "grid must contain exactly three sections"
    assert len(sep_colors) == 2, "grid must contain exactly two separators"

    sig = tuple(int(c) for c in sep_colors)
    if sig == (TEMPLATE_STENCIL_SEPARATOR, STENCIL_PALETTE_SEPARATOR):
        template, stencil, palette = sections
    elif sig == (STENCIL_PALETTE_SEPARATOR, TEMPLATE_STENCIL_SEPARATOR):
        palette, stencil, template = sections
    else:
        raise AssertionError(f"unexpected separator-color order: {sig}")
    return template, stencil, palette


def _decode_palette_symbol(block: np.ndarray) -> tuple[int, int]:
    """Read fg/bg from a 2x2 code block: main diagonal = fg, anti-diagonal = bg."""
    code_bbox = bbox(block != 0)
    assert code_bbox is not None, "palette section must contain a non-black code block"
    code = crop_bbox(block, code_bbox)
    assert code.shape == (2, 2), "palette code must crop to a 2x2 block"

    foreground = int(code[0, 0])
    background = int(code[0, 1])
    assert int(code[1, 1]) == foreground, "palette main diagonal must encode the foreground color"
    assert int(code[1, 0]) == background, "palette anti-diagonal must encode the background color"
    assert foreground != background, "foreground and background colors must differ"
    return foreground, background


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with a 1-pixel background separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a color-tagged three-section layout, recolor the cropped template, and repeat it by a stencil count.

    Input structure:
    - Single-pixel separator rows or columns split the grid into three sections rather than four.
    - One separator is red (2), marking the boundary between template and stencil; the other is blue (1),
      marking the boundary between stencil and the merged palette section.
    - The three semantic sections may appear in forward or reversed order, and the palette section is a
      black panel containing one 2x2 code whose main diagonal is the foreground color and anti-diagonal
      is the background color.

    Rule:
    - Use the separator colors to identify which cropped block is the template, which is the component-count
      stencil, and which is the merged palette.
    - Crop the template to the bbox of its non-black pixels, decode the 2x2 palette symbol into foreground
      and background colors, and recolor the cropped mask with those two colors.
    - Count 4-connected non-black components in the stencil and repeat the recolored crop that many times,
      inserting a one-cell background-colored row or column between copies.

    Key insight: The task now has two linked decoding stages. The boundary colors tell you how to assign
    section roles before any content is interpreted, and only then can the 2x2 palette token reveal the
    two colors needed to build the repeated output.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections, sep_colors = _split_sections(g)
    template, repetition_pattern, palette_block = _assign_sections(sections, sep_colors)

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground, background = _decode_palette_symbol(palette_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    return _tile_repeat(tile, n_repeats, background, horizontal=(orientation == "horizontal"))


def _make_palette_symbol(shared_dim: int, fg_color: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Create a black section containing a single 2x2 diagonal color code."""
    varying = random.randint(*PALETTE_DIM_RANGE)
    h, w = (shared_dim, varying) if horizontal else (varying, shared_dim)
    panel = np.zeros((h, w), dtype=int)

    r0 = random.randint(1, h - 3)
    c0 = random.randint(1, w - 3)
    panel[r0 : r0 + 2, c0 : c0 + 2] = np.array(
        [[fg_color, bg_color], [bg_color, fg_color]],
        dtype=int,
    )
    return panel


def _join_sections(
    sections: list[np.ndarray],
    sep_colors: list[int],
    shared_dim: int,
    horizontal: bool,
) -> np.ndarray:
    """Join sections with explicit separator colors between adjacent semantic blocks."""
    assert len(sections) == len(sep_colors) + 1, "separator list must match section count"
    if horizontal:
        parts = [sections[0]]
        for color, section in zip(sep_colors, sections[1:]):
            parts.extend([np.full((shared_dim, 1), color, dtype=int), section])
        return np.hstack(parts)

    parts = [sections[0]]
    for color, section in zip(sep_colors, sections[1:]):
        parts.extend([np.full((1, shared_dim), color, dtype=int), section])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs of STENCIL_COLOR on black, with a 1-cell moat between blobs."""
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= n_blobs:
            break
        br, bc = (1, 2) if random.choice([True, False]) else (2, 1)

        if h - 1 - br < 1 or w - 1 - bc < 1:
            continue
        r = random.randint(1, h - 1 - br)
        c = random.randint(1, w - 1 - bc)

        r0, c0 = max(0, r - 1), max(0, c - 1)
        r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
        if np.any(grid[r0:r1, c0:c1] != 0):
            continue

        grid[r : r + br, c : c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    reverse_order = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    palette = _make_palette_symbol(shared_dim, fg_color, bg_color, horizontal)

    if reverse_order:
        sections = [palette, stencil, template]
        sep_colors = [STENCIL_PALETTE_SEPARATOR, TEMPLATE_STENCIL_SEPARATOR]
    else:
        sections = [template, stencil, palette]
        sep_colors = [TEMPLATE_STENCIL_SEPARATOR, STENCIL_PALETTE_SEPARATOR]

    inp = _join_sections(sections, sep_colors, shared_dim, horizontal)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
