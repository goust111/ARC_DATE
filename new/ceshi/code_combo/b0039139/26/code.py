# Combo variant of b0039139 (index 26)
# combo_spec: D6/2+D11/0
# Name: 区段顺序编码操作顺序配先铺后着色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = str(Path("/home/lixinyu/snap/new/solvers"))
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)
from arc_utils import bbox, components_mask, crop_bbox, random_sprite


SEPARATOR_COLOR = 1
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
RAW_BACKGROUND = 0
COLOR_POOL = (2, 5, 6, 7, 8, 9)

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 400

_NEXT_MODE = 0

# Design analysis for combo D6/2 + D11/0
# a) Concrete behavior and trace:
#    - Original b0039139 always reads the four sections in fixed order
#      [template, stencil, fg, bg], crops the yellow template to its tight bbox,
#      recolors that cropped mask, then repeats it N times.
#    - This combo makes the section order informative. We only use two clean
#      orders so the cue is visible:
#         1. `[template, stencil, fg, bg]` -> template is before both swatches.
#            This means "recolor first, then tile", so the output is the original
#            tight copy with 1-cell bg gaps.
#         2. `[fg, bg, template, stencil]` -> template is after both swatches.
#            This means "tile first, then recolor". Here the repeated unit is the
#            entire raw template section, including its 1-cell black border, and
#            only afterwards are 4->fg and 0->bg applied globally.
#    - Small trace with a cropped shape of size 2x3 inside a 4x5 template box and
#      N=3:
#         original / recolor-first output width = 3*3 + 2 = 11
#         fused tile-first output width         = 3*5 + 2 = 17
#      The tile-first branch visibly keeps a bg frame around every copy, so the
#      order cue changes both output size and spacing instead of being cosmetic.
# b) Complexity calibration:
#    - If D11/0 were implemented as "crop first, then tile raw 0/4, then recolor",
#      it would render the same picture as recolor-first and D6/2 would become
#      invisible. To avoid that degeneration, this combo interprets "raw template"
#      literally as the whole template section before bbox cropping.
#    - Section-order freedom is limited to two repeated patterns rather than full
#      permutation of four parts. That keeps the cue learnable from 2-4 examples.
# c) Implicit-assumption audit against the original solver:
#    - `template, stencil, foreground_block, background_block = sections[:4]`
#      is no longer valid because section order is no longer fixed.
#    - `tile_mask = crop_bbox(template_mask, template_bbox)` is still valid only
#      for the recolor-first branch. The tile-first branch must bypass this crop.
#    - The original code treated the two swatches by absolute positions 3 and 4.
#      The fused solver instead identifies the two solid-color blocks and assigns
#      their roles by local order: earlier swatch = fg, later swatch = bg.
#    - The original output path assumed a single pipeline. The fused solver now
#      has two branches selected by whether the template block sits before or after
#      the swatch pair in the input ordering.
# d) Placement legality:
#    - The generator keeps the original one-axis layout: all four sections share
#      one common height or width and are separated by single blue lines.
#    - Template and stencil never overlap because they occupy disjoint sections.
#      Inside the stencil, each 2-cell blob keeps a 1-cell black moat, so the
#      component count is stable.
#    - Tile-first output sizes are computed exactly from the raw template section,
#      so repeated frames and inserted 1-cell separators never overlap.
# e) Visual observability:
#    - The two operation orders create an obvious difference: tight repeated glyphs
#      versus repeated framed tiles. Humans can see both the order cue in the input
#      and the changed spacing/border behavior in the output.
#    - Generator calls alternate between the two orders, so a multi-example task
#      built from successive calls naturally contains repeated support for both.
# f) Visual cleanliness:
#    - Swatch colors never use 1/3/4, so blue separators, green stencil blobs, and
#      yellow template pixels remain visually distinct from the palette blocks.
#    - Only four region types appear and they stay separated by solid blue bars,
#      so no area mixes multiple annotation systems.
# g) Change inferability / anti-degeneracy:
#    - We quantify the fusion by comparing the fused output to the old fixed-order
#      solver. For template-after-swatches samples, the outputs must differ because
#      the raw border is preserved; for template-before-swatches samples, they match
#      the original branch by design. The generator enforces both branch values.
# 9) New solver assumptions checked against the generator:
#    - Exactly one section uses colors {0,4} with at least one 4: the template.
#      Generator enforces TEMPLATE_COLOR only there.
#    - Exactly one section uses colors {0,3} with at least one 3: the stencil.
#      Generator enforces STENCIL_COLOR only there.
#    - The remaining two sections are solid swatches, and their order is fg then bg.
#      Generator always constructs the swatch pair in that local order.
#    - The template is either before both swatches or after both swatches; it is
#      never placed between them, so the branch test is unambiguous.


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = map(int, grid.shape)
    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == SEPARATOR_COLOR))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        sections = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", sections

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEPARATOR_COLOR))]
    cuts = [-1] + sep_cols + [w]
    sections = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", sections


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _section_role(block: np.ndarray) -> str:
    vals = set(map(int, np.unique(block)))
    if vals == {RAW_BACKGROUND, TEMPLATE_COLOR}:
        return "template"
    if vals == {RAW_BACKGROUND, STENCIL_COLOR}:
        return "stencil"
    if len(vals) == 1:
        return "swatch"
    raise ValueError(f"Unrecognized section colors: {sorted(vals)}")


def _extract_roles(sections: list[np.ndarray]) -> tuple[int, int, int, int]:
    template_idx = -1
    stencil_idx = -1
    swatches: list[int] = []
    for idx, section in enumerate(sections):
        role = _section_role(section)
        if role == "template":
            template_idx = idx
        elif role == "stencil":
            stencil_idx = idx
        else:
            swatches.append(idx)

    if template_idx < 0 or stencil_idx < 0 or len(swatches) != 2:
        raise ValueError("Expected one template, one stencil, and two swatches")
    swatches.sort()
    fg_idx, bg_idx = swatches
    return template_idx, stencil_idx, fg_idx, bg_idx


def _repeat_with_separator(
    tile: np.ndarray,
    n_repeats: int,
    sep_color: int,
    orientation: str,
) -> np.ndarray:
    if n_repeats <= 1:
        return tile.copy()

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), sep_color, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), sep_color, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _build_recolor_first(
    template: np.ndarray,
    n_repeats: int,
    foreground: int,
    background: int,
    orientation: str,
) -> np.ndarray:
    template_mask = template == TEMPLATE_COLOR
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)
    return _repeat_with_separator(tile, n_repeats, background, orientation)


def _build_tile_first(
    template: np.ndarray,
    n_repeats: int,
    foreground: int,
    background: int,
    orientation: str,
) -> np.ndarray:
    raw_tiled = _repeat_with_separator(template, n_repeats, RAW_BACKGROUND, orientation)
    return np.where(raw_tiled == TEMPLATE_COLOR, foreground, background).astype(int)


def _old_reference_output(
    template: np.ndarray,
    n_repeats: int,
    foreground: int,
    background: int,
    orientation: str,
) -> np.ndarray:
    return _build_recolor_first(template, n_repeats, foreground, background, orientation)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read four blue-separated sections where the template's position relative to the two swatches decides whether the task uses the old tight recolor-first pipeline or the new raw-template tile-first pipeline.

    Input structure:
    - Solid blue rows or columns split the grid into four sections laid out on one axis.
    - One section is a yellow-on-black template, one is a green-on-black component-count stencil, and two are solid color swatches.
    - The two swatches always keep their local order: earlier swatch = foreground color, later swatch = background color.
    - The template is placed either before both swatches or after both swatches, and that relative order is the operation-order code.

    Rule:
    - Count the 4-connected green components in the stencil to get the repeat count.
    - If the template comes before the swatch pair, crop the yellow template to its bbox, recolor that tight crop, and then repeat it with 1-cell background separators.
    - If the template comes after the swatch pair, first repeat the entire raw template section, including its black border, using raw black separators, and then recolor the whole repeated block so yellow becomes foreground and black becomes background.

    Key insight:
    - The four sections no longer have a fixed semantic order. You must first identify section roles by their internal structure and then use the template-vs-swatch ordering to decide which processing pipeline applies.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    if len(sections) != 4:
        raise ValueError("Expected exactly four sections")

    template_idx, stencil_idx, fg_idx, bg_idx = _extract_roles(sections)
    template = sections[template_idx]
    stencil = sections[stencil_idx]
    foreground = _solid_color(sections[fg_idx])
    background = _solid_color(sections[bg_idx])
    n_repeats = len(components_mask(stencil == STENCIL_COLOR, connectivity=4))

    if template_idx < fg_idx:
        return _build_recolor_first(template, n_repeats, foreground, background, orientation)
    return _build_tile_first(template, n_repeats, foreground, background, orientation)


def _make_swatch(shared_dim: int, color: int, orientation: str) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if orientation == "horizontal":
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)

    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, orientation: str) -> np.ndarray | None:
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)
    h, w = (shared_dim, varying_dim) if orientation == "horizontal" else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= n_blobs:
            break
        br, bc = random.choice(((1, 2), (2, 1)))
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


def _make_template(shape: np.ndarray) -> np.ndarray:
    crop_h, crop_w = map(int, shape.shape)
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape
    return template


def _next_mode() -> int:
    global _NEXT_MODE
    mode = _NEXT_MODE
    _NEXT_MODE = 1 - _NEXT_MODE
    return mode


def _gen_attempt(mode: int) -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
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
    if not np.any(shape == TEMPLATE_COLOR):
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    template = _make_template(shape)
    shared_dim = template.shape[0] if orientation == "horizontal" else template.shape[1]
    stencil = _build_stencil(n_repeats, shared_dim, orientation)
    if stencil is None:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, orientation)
    bg_swatch = _make_swatch(shared_dim, bg_color, orientation)

    if mode == 0:
        section_order = [template, stencil, fg_swatch, bg_swatch]
        expected = _build_recolor_first(template, n_repeats, fg_color, bg_color, orientation)
    else:
        section_order = [fg_swatch, bg_swatch, template, stencil]
        expected = _build_tile_first(template, n_repeats, fg_color, bg_color, orientation)
        original = _old_reference_output(template, n_repeats, fg_color, bg_color, orientation)
        if np.array_equal(expected, original):
            return None

    inp = _join_sections(section_order, shared_dim, orientation)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    mode = _next_mode()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(mode)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
