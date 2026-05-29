# Variant of b0039139
# Dimension: D9 — 色板区段的表示方式 (symbol_instructions)
# Change: 区段3和区段4各为纯色矩形块，直接用其填充颜色作为前景色和背景色
#   -> 色板合并为一个区段；该区段内只有一个 2x2 编码块，左上/右下是前景色，右上/左下是背景色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D9 variant:
# a) Concrete behavior:
#    - The input still has separator-defined sections and the output is still a recolored cropped
#      template repeated by the component count in the stencil.
#    - The only structural change is the color-key area: instead of two solid swatches, there is
#      one black section containing a lone 2x2 checkerboard-like code. The main diagonal gives the
#      foreground color and the anti-diagonal gives the background color.
#    - Small trace:
#      original: [template | stencil | solid fg=8 | solid bg=3] -> crop template -> recolor with 8/3.
#      variant : [template | stencil | code [[8,3],[3,8]]]      -> decode diagonal rule -> recolor with 8/3.
#      The repetition count and tiling axis are unchanged.
# b) Complexity calibration:
#    - The code uses exactly two colors and a fixed positional rule; this is a visible symbolic cue,
#      but not so free-form that examples become ambiguous.
#    - Embedding the 2x2 patch inside a larger black section preserves the original section layout
#      scale, so the D9 change is obvious to a human instead of collapsing to a tiny borderless token.
# c) Hidden assumptions from the original main():
#    - `sections[:4]` no longer holds; the solver must expect exactly three semantic sections.
#    - `_solid_color(...)` on swatches is invalid because the palette section is mostly black with
#      two informative colors. The new solver must crop the non-black bbox and decode positions.
#    - The recoloring/tile-repeat code still assumes a single cropped template mask and a background
#      color available from the palette code; the generator guarantees both.
# d) Placement legality:
#    - The palette section is sized on the same shared axis as the template/stencil so joining with
#      single-pixel separators still works.
#    - The 2x2 code is placed fully inside the section with at least a one-cell black margin.
#    - Stencil blobs keep their non-touching constraint, so component counting remains unambiguous.
# e) Visual observability:
#    - Across samples the palette section visibly changes from two large solid areas to one compact
#      diagonal checkerboard symbol. Humans can match the repeated output colors to the two diagonals.
#    - The positional mapping is fixed across all examples, avoiding underdetermined branching.
# f) Visual cleanliness:
#    - Only one symbol is placed in the palette section; the surrounding black field prevents overlap
#      with separators or extra marks, so the code remains easy to isolate by eye.
# g) Color inference:
#    - Exactly two non-black colors appear in the palette code, and both reappear in the output.
#      This gives enough structure to infer which diagonal means foreground vs background.
# h) New solver assumptions after modification:
#    - The palette section contains exactly one non-black 2x2 bbox.
#    - Its main diagonal colors match each other, its anti-diagonal colors match each other, and the
#      two decoded colors are distinct.
#    - The generator enforces these conditions directly.


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = 1

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and repeat it according to a component-count stencil.

    Input structure:
    - Solid blue (1) separator rows or columns split the grid into three sections.
    - Section 1 is a black-background shape template; section 2 is a component-count stencil.
    - Section 3 is a black palette panel containing one 2x2 code: top-left/bottom-right = foreground,
      top-right/bottom-left = background.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels and recolor it using the decoded 2x2 symbol.
    - Count 4-connected non-black components in section 2.
    - Repeat the recolored crop that many times, inserting one background-colored row/column between copies.

    Key insight: The stencil still tells only how many copies to output; the new step is that the two
    output colors are not given as solid swatches anymore, but must be read from a fixed-position symbol.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    assert len(sections) >= 3, "grid must contain template, stencil, and symbol palette sections"
    template, repetition_pattern, palette_block = sections[:3]

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

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), background, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), background, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
PALETTE_DIM_RANGE = (4, 8)           # palette section varying-axis size
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
GEN_RETRIES = 300                    # outer generate_input retry limit


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


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


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    """Join sections with single-pixel separator lines of SEPARATOR_COLOR."""
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for s in sections[1:]:
            parts.extend([sep, s])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for s in sections[1:]:
        parts.extend([sep, s])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs (1x2 or 2x1) of STENCIL_COLOR on a black grid.
    Blobs are separated by >=1 cell of black in all directions."""
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
    inp = _join_sections([template, stencil, palette], shared_dim, horizontal)

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
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
