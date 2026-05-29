# Combo variant of b0039139 (index 20)
# combo_spec: D6/0+D9/1+D12/1
# Name: 多色分隔线加2x2色板加多前景色模板

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant:
# a) Concrete behavior and trace:
#    - The original task can be read as
#        [template | stencil | fg_swatch | bg_swatch]
#      with one separator color and a binary template.
#    - This combo collapses the two swatches into one black palette section that contains a
#      2x2 code, upgrades the template to use both 4- and 5-pixels, and makes separator colors
#      meaningful instead of merely decorative.
#    - Small trace:
#        physical order: [palette | blue sep | stencil | red sep | template]
#        palette code : [[8, 7],
#                        [7, 8]]
#        template crop:
#            [[4, 5],
#             [0, 4]]
#        stencil colors: three disconnected blobs in 6, one blob in 9
#      Original logic would fail in three ways at once:
#        1. it would misread the first section as the template because it trusts position;
#        2. it would try to treat the palette as a solid swatch;
#        3. it would collapse template colors 4 and 5 into one foreground mask.
#      Combo logic must instead:
#        1. use separator colors to infer semantic order template/stencil/palette;
#        2. decode 8 as the primary foreground and 7 as the background from the 2x2 symbol;
#        3. recolor 4 -> 8, 5 -> dominant stencil color 6, 0 -> 7;
#        4. count four stencil components and tile the recolored crop four times.
# b) Complexity calibration:
#    - Only two separator colors are used, with fixed meanings:
#        red (2) = template/stencil boundary
#        blue (1) = stencil/palette boundary
#      This keeps the cue visible and learnable.
#    - The physical order is limited to forward or reversed, so separator colors matter for
#      parsing but the layout does not become combinatorially noisy.
#    - The palette uses one fixed 2x2 diagonal mapping, avoiding ambiguous symbol grammars.
#    - The template always contains both 4 and 5 so the multicolor recoloring rule is not optional.
# c) Hidden-assumption audit against the original main():
#    - `_split_sections(...)` assumed all separators share one color. That breaks here, so the
#      solver must detect full-line separators by membership in a small separator-color set and
#      preserve their colors in order.
#    - `sections[:4]` and positional unpacking are no longer valid because the palette swatches are
#      merged and the three semantic sections may appear reversed.
#    - `_solid_color(...)` on the palette is invalid because the palette section is mostly black.
#      The solver must crop the non-black bbox and decode the 2x2 diagonal rule.
#    - `template != 0` is no longer sufficient for recoloring because 4 and 5 map to different
#      output colors. The crop must preserve raw template values.
#    - Component counting on `stencil != 0` still holds: the combo changes stencil colors, not the
#      fact that disconnected non-black blobs encode the repeat count.
# d) Placement legality:
#    - All sections still share one axis, so single-pixel separator rows or columns remain legal.
#    - The palette panel is large enough to place the 2x2 code with a one-cell black margin.
#    - Stencil blobs stay 2-cell, 4-connected, and separated by at least one black cell in every
#      direction, so both component count and dominant-color reading remain stable.
#    - The template keeps a one-cell black border around the colored crop, preserving bbox cropping.
# e) Visual observability:
#    - Reversing the section order in some samples forces attention onto separator colors; position
#      alone is no longer a reliable parser.
#    - The palette visibly changes from large uniform blocks to a compact diagonal 2x2 token.
#    - The output visibly contains two foreground colors, one tied to palette-diagonal color and
#      one tied to the most common visible stencil color.
# f) Visual cleanliness:
#    - The palette contains only one symbol on a black field.
#    - The stencil uses at most two non-black colors, and only one is dominant.
#    - Output copies are still separated only by the decoded background color, so repeated tiles do
#      not visually merge.
# g) Change inferability:
#    - The color transfer is supported by repeated structural cues rather than numeric ids:
#      diagonal placement inside the 2x2 code identifies primary foreground/background, and
#      repeated majority presence inside the stencil identifies the secondary foreground.
#    - Because the generator always includes both template colors and usually includes a minority
#      stencil color when the repeat count allows it, a solver cannot safely collapse back to the
#      original single-foreground or position-only interpretation.
# h) Generator guarantees for the new solver assumptions:
#    - Exactly three semantic sections exist, with separator color signatures [2,1] or [1,2].
#    - The template crop contains at least one 4 and at least one 5, and no other non-black color.
#    - The palette section contains exactly one non-black 2x2 bbox with main diagonal = fg and
#      anti-diagonal = bg, with fg != bg.
#    - The stencil has at least one non-black component and a unique dominant non-black color that
#      differs from the palette foreground/background.


REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
OUTPUT_COLOR_POOL = [3, 6, 7, 8, 9]
TEMPLATE_PRIMARY = 4
TEMPLATE_SECONDARY = 5
SEP_STENCIL_PALETTE = 1
SEP_TEMPLATE_STENCIL = 2
SEPARATOR_COLORS = {SEP_STENCIL_PALETTE, SEP_TEMPLATE_STENCIL}
PALETTE_DIM_RANGE = (4, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 400


def _split_colored_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray], list[int]]:
    h, w = grid.shape

    sep_rows = [
        r
        for r in range(h)
        if int(grid[r, 0]) in SEPARATOR_COLORS and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        sections = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", sections, [int(grid[r, 0]) for r in sep_rows]

    sep_cols = [
        c
        for c in range(w)
        if int(grid[0, c]) in SEPARATOR_COLORS and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    assert sep_cols, "grid must contain colored separator rows or columns"
    cuts = [-1] + sep_cols + [w]
    sections = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", sections, [int(grid[0, c]) for c in sep_cols]


def _semantic_sections(sections: list[np.ndarray], sep_colors: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(sections) == 3, "combo variant expects exactly three sections"
    assert len(sep_colors) == 2, "combo variant expects exactly two separator lines"

    if sep_colors == [SEP_TEMPLATE_STENCIL, SEP_STENCIL_PALETTE]:
        template, stencil, palette = sections
        return template, stencil, palette

    if sep_colors == [SEP_STENCIL_PALETTE, SEP_TEMPLATE_STENCIL]:
        palette, stencil, template = sections
        return template, stencil, palette

    raise AssertionError(f"invalid separator signature: {sep_colors}")


def _decode_palette_symbol(block: np.ndarray) -> tuple[int, int]:
    """Decode the lone 2x2 symbol: main diagonal = primary fg, anti-diagonal = bg."""
    code_bbox = bbox(block != 0)
    assert code_bbox is not None, "palette section must contain a non-black code"
    code = crop_bbox(block, code_bbox)
    assert code.shape == (2, 2), "palette code must crop to exactly 2x2"

    foreground = int(code[0, 0])
    background = int(code[0, 1])
    assert int(code[1, 1]) == foreground, "palette main diagonal must encode foreground"
    assert int(code[1, 0]) == background, "palette anti-diagonal must encode background"
    assert foreground != background, "palette foreground and background must differ"
    return foreground, background


def _dominant_nonzero_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block[block != 0], return_counts=True)
    assert vals.size > 0, "stencil must contain non-black cells"
    max_count = int(counts.max())
    winners = vals[counts == max_count]
    assert winners.size == 1, "stencil dominant non-black color must be unique"
    return int(winners[0])


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _grow_connected_subset(mask: np.ndarray, target_size: int) -> np.ndarray:
    coords = [tuple(x) for x in np.argwhere(mask)]
    chosen = {random.choice(coords)}

    while len(chosen) < target_size:
        frontier = []
        for r, c in chosen:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in chosen:
                    frontier.append((rr, cc))
        if not frontier:
            frontier = [coord for coord in coords if coord not in chosen]
        chosen.add(random.choice(frontier))

    out = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        out[r, c] = True
    return out


def _build_multicolor_shape(crop_h: int, crop_w: int) -> np.ndarray:
    density = random.uniform(*DENSITY_RANGE)
    base = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_PRIMARY],
        connectivity=4,
        symmetry="not_symmetric",
    )
    mask = base != 0
    area = int(mask.sum())
    if area < 2:
        return _build_multicolor_shape(crop_h, crop_w)

    min_secondary = 1 if area <= 3 else 2
    secondary_size = random.randint(min_secondary, area - 1)
    secondary_mask = _grow_connected_subset(mask, secondary_size)

    shape = np.zeros_like(base, dtype=int)
    shape[mask] = TEMPLATE_PRIMARY
    shape[secondary_mask] = TEMPLATE_SECONDARY
    if not np.any(shape == TEMPLATE_PRIMARY) or not np.any(shape == TEMPLATE_SECONDARY):
        return _build_multicolor_shape(crop_h, crop_w)
    return shape


def _make_palette_symbol(shared_dim: int, fg_color: int, bg_color: int, horizontal: bool) -> np.ndarray:
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


def _build_stencil(blob_colors: list[int], shared_dim: int, horizontal: bool) -> np.ndarray | None:
    min_varying = max(4, len(blob_colors) * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= len(blob_colors):
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

        grid[r : r + br, c : c + bc] = blob_colors[placed]
        placed += 1

    return grid if placed == len(blob_colors) else None


def _choose_stencil_blob_colors(n_blobs: int, accent_color: int, alternate_color: int) -> list[int]:
    if n_blobs <= 2:
        return [accent_color] * n_blobs

    dominant = random.randint((n_blobs // 2) + 1, n_blobs - 1)
    colors = [accent_color] * dominant + [alternate_color] * (n_blobs - dominant)
    random.shuffle(colors)
    return colors


def _recolor_template(shape: np.ndarray, fg_color: int, accent_color: int, bg_color: int) -> np.ndarray:
    tile = np.full(shape.shape, bg_color, dtype=int)
    tile[shape == TEMPLATE_PRIMARY] = fg_color
    tile[shape == TEMPLATE_SECONDARY] = accent_color
    return tile


def _join_sections_with_colors(
    sections: list[np.ndarray],
    separator_colors: list[int],
    shared_dim: int,
    horizontal: bool,
) -> np.ndarray:
    assert len(sections) == len(separator_colors) + 1

    if horizontal:
        parts = [sections[0]]
        for sep_color, section in zip(separator_colors, sections[1:]):
            parts.append(np.full((shared_dim, 1), sep_color, dtype=int))
            parts.append(section)
        return np.hstack(parts)

    parts = [sections[0]]
    for sep_color, section in zip(separator_colors, sections[1:]):
        parts.append(np.full((1, shared_dim), sep_color, dtype=int))
        parts.append(section)
    return np.vstack(parts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped multicolor template and repeat it using a separator-coded stencil layout.

    Input structure:
    - The grid is split by full red and blue separator rows or columns, and the separator colors
      encode which neighboring sections are being separated rather than serving as interchangeable cuts.
    - There are three semantic sections: a black-backed template with 4/5 pixels, a stencil board
      of disconnected non-black blobs, and one black palette panel containing a single 2x2 code.
    - Red (2) marks the template/stencil boundary and blue (1) marks the stencil/palette boundary,
      so the physical order may be forward or reversed.

    Rule:
    - First use the separator-color signature to recover which block is the template, which is the
      stencil, and which is the palette.
    - Crop the template to its non-black bbox. Decode the palette's 2x2 code so the main diagonal
      gives the output color for template-4 pixels and the anti-diagonal gives the output background.
    - Read the stencil's unique dominant non-black color as the output color for template-5 pixels,
      count 4-connected stencil components, and tile the recolored crop that many times with one
      background-colored separator row or column between copies.

    Key insight: the stencil now carries two kinds of information at once while the separator colors
    are also semantic; solving requires parsing section identity from the red/blue boundaries before
    combining palette decoding with the stencil's majority-color cue.
    """
    g = np.asarray(grid, dtype=int)
    orientation, raw_sections, sep_colors = _split_colored_sections(g)
    template, repetition_pattern, palette_block = _semantic_sections(raw_sections, sep_colors)

    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground, background = _decode_palette_symbol(palette_block)
    accent = _dominant_nonzero_color(repetition_pattern)
    template_crop = crop_bbox(template, template_bbox)
    assert bool(np.all(np.isin(template_crop, [0, TEMPLATE_PRIMARY, TEMPLATE_SECONDARY])))

    tile = np.full(template_crop.shape, background, dtype=int)
    tile[template_crop == TEMPLATE_PRIMARY] = foreground
    tile[template_crop == TEMPLATE_SECONDARY] = accent

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile
    return _tile_repeat(tile, n_repeats, background, orientation == "horizontal")


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    reverse_order = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    shape = _build_multicolor_shape(crop_h, crop_w)

    fg_color, accent_color, bg_color, alt_stencil_color = random.sample(OUTPUT_COLOR_POOL, 4)
    blob_colors = _choose_stencil_blob_colors(n_repeats, accent_color, alt_stencil_color)

    tile = _recolor_template(shape, fg_color, accent_color, bg_color)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(blob_colors, shared_dim, horizontal)
    if stencil is None:
        return None
    if _dominant_nonzero_color(stencil) != accent_color:
        return None

    palette = _make_palette_symbol(shared_dim, fg_color, bg_color, horizontal)
    if reverse_order:
        sections = [palette, stencil, template]
        separator_colors = [SEP_STENCIL_PALETTE, SEP_TEMPLATE_STENCIL]
    else:
        sections = [template, stencil, palette]
        separator_colors = [SEP_TEMPLATE_STENCIL, SEP_STENCIL_PALETTE]

    inp = _join_sections_with_colors(sections, separator_colors, shared_dim, horizontal)

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
