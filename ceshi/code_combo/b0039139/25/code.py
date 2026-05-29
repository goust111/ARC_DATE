# Combo variant of b0039139 (index 25)
# combo_spec: D1/3+D3/2
# Name: 渐变着色配像素总数决定重复次数

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVERS_DIR = ROOT / "solvers"
if str(SOLVERS_DIR) not in sys.path:
    sys.path.append(str(SOLVERS_DIR))

from arc_utils import bbox, crop_bbox, random_sprite


"""
Design analysis for combo D1/3 + D3/2
=====================================

a) Concrete behavior:
   - The original task crops the yellow template, recolors it with one fixed foreground color and one fixed background
     color, then repeats that same tile once per green component in section 2.
   - In this fusion, section 1 is still the same template section and sections 3-4 still give the starting foreground
     color and constant background color. The change is that copy i uses foreground `start + i`, clamped at color 9,
     while the background stays fixed.
   - Section 2 no longer means "how many blobs exist". It means "how many green pixels exist". A concrete trace:
       template crop = a 3x2 mask
       section 3 color = 3, section 4 color = 0
       stencil pixels = 7 green cells split as [2,1,2,1,1]
     Original output would have 5 copies, all foreground 3.
     Fused output has 7 copies with foregrounds [3,4,5,6,7,8,9].

b) Complexity calibration:
   - If every stencil component still had area 2, a learner could shortcut section 2 as "double the blob count".
     The generator therefore mixes singleton and domino components, so pixel total is not recoverable from component
     count by a fixed multiplier.
   - If the starting foreground were too high, the gradient would immediately saturate at 9 and D1/3 would be hard to
     see. The generator keeps the start in {3,4}, giving a visible run before saturation.
   - Repeat counts stay in 6-20, so the long-strip effect is obvious without making grids unwieldy.

c) Implicit assumptions audit:
   - The old solver assumed every non-black stencil component contributed exactly one repeat. That assumption is broken;
     the solver must count only color-3 pixels.
   - The old `_tile_repeat` helper assumed every copy was identical. That is broken; each copy now needs its own
     foreground color, although all copies still share one background separator color.
   - The old generator assumed the foreground swatch alone determined the final paint color. That is now only the
     gradient seed, so examples must expose multiple copies.

d) Placement legality:
   - Section layout remains four blocks separated by full blue rows or columns.
   - Template, stencil, and swatches are built independently and then joined with exact shared dimensions, so no
     section overlaps or spills out of bounds.
   - Stencil components are placed sequentially with at least one blank row or column between bounding boxes, which
     keeps them disconnected and fully inside the stencil section.

e) Visual observability:
   - Humans can directly count green pixels in the stencil because the components are tiny isolated motifs on black.
   - The first few output copies visibly step through successive ARC colors, so the gradient direction and unit step
     are observable even when later copies saturate at 9.
   - No hidden ordering uses numeric sorting of objects, creation order, or background color ids.

f) Visual cleanliness:
   - The fusion keeps the original four-section reading order, so only two concepts change: how many repeats to make
     and how each successive copy is colored.
   - The stencil uses only singletons and dominoes, which are easy to parse and do not clutter the board.
   - Output separators still use the background color, so the repeated template copies remain neatly segmented.

g) New solver assumptions checked against generation:
   - The solver assumes section 1 contains at least one template-colored pixel; the generator crops the random sprite to
     its nonzero bbox before embedding it, so the template is never empty.
   - The solver assumes the gradient seed from section 3 is below or equal to 9 and that clamping at 9 suffices. The
     generator restricts the seed to {3,4}.
   - The solver assumes section 2 uses color 3 as the count source. The generator paints all stencil motifs in color 3
     and keeps section backgrounds black.
"""


TOTAL_REPEAT_RANGE = (6, 20)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
START_FG_POOL = [3, 4]
BACKGROUND_POOL = [0, 2]
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
SECTION_EXTRA_PADDING = 4
GEN_RETRIES = 400


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
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


def _gradient_colors(start_color: int, n: int) -> list[int]:
    return [min(9, int(start_color) + i) for i in range(int(n))]


def _render_tile(tile_mask: np.ndarray, fg_color: int, bg_color: int) -> np.ndarray:
    return np.where(tile_mask, int(fg_color), int(bg_color)).astype(int)


def _stack_gradient_tiles(
    tile_mask: np.ndarray, foregrounds: list[int], bg_color: int, horizontal: bool
) -> np.ndarray:
    tiles = [_render_tile(tile_mask, fg, bg_color) for fg in foregrounds]
    if len(tiles) == 1:
        return tiles[0]

    if horizontal:
        sep = np.full((tile_mask.shape[0], 1), bg_color, dtype=int)
        pieces: list[np.ndarray] = [tiles[0]]
        for tile in tiles[1:]:
            pieces.extend([sep, tile])
        return np.hstack(pieces)

    sep = np.full((1, tile_mask.shape[1]), bg_color, dtype=int)
    pieces = [tiles[0]]
    for tile in tiles[1:]:
        pieces.extend([sep, tile])
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the template once, then emit a long strip of copies whose colors climb copy by copy.

    Input structure:
    - Blue separator rows or columns split the input into four ordered sections.
    - Section 1 is a black template board with the shape drawn in color 4; section 2 is a black stencil board with
      green counting marks.
    - Sections 3 and 4 are solid swatches giving the starting foreground color and the constant background color.

    Rule:
    - Crop section 1 to the bbox of its color-4 pixels to obtain a binary tile mask.
    - Count all color-3 pixels in section 2; that total is the number of tile copies to produce.
    - Recolor copy i with foreground `start + i`, clamped at 9, recolor the tile background with the section-4 color,
      and place one background-colored separator row or column between adjacent copies.

    Key insight:
    - Section 2 encodes an area total rather than a component count, and section 3 encodes only the first foreground
      color. The full output requires combining those two facts with the original crop-and-tile structure.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_bbox = bbox(template == TEMPLATE_COLOR)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    tile_mask = crop_bbox(template == TEMPLATE_COLOR, template_bbox)
    start_foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    n_repeats = int(np.count_nonzero(repetition_pattern == STENCIL_COLOR))
    if n_repeats <= 0:
        return np.empty((0, 0), dtype=int)

    foregrounds = _gradient_colors(start_foreground, n_repeats)
    return _stack_gradient_tiles(tile_mask, foregrounds, background, orientation == "horizontal")


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
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


def _domino(horizontal: bool) -> np.ndarray:
    if horizontal:
        return np.full((1, 2), STENCIL_COLOR, dtype=int)
    return np.full((2, 1), STENCIL_COLOR, dtype=int)


def _stencil_components(total_pixels: int) -> list[np.ndarray]:
    max_dominoes = (int(total_pixels) - 2) // 2
    n_dominoes = random.randint(2, max_dominoes)
    n_singletons = int(total_pixels) - 2 * n_dominoes
    assert n_singletons >= 2

    components = [np.full((1, 1), STENCIL_COLOR, dtype=int) for _ in range(n_singletons)]
    for _ in range(n_dominoes):
        components.append(_domino(horizontal=random.choice([True, False])))
    random.shuffle(components)
    return components


def _build_stencil(total_pixels: int, shared_dim: int, horizontal: bool) -> np.ndarray:
    components = _stencil_components(total_pixels)
    gaps = [random.randint(1, 2) for _ in range(len(components) - 1)] if len(components) > 1 else []

    if horizontal:
        varying_dim = (
            2
            + sum(component.shape[1] for component in components)
            + sum(gaps)
            + random.randint(0, SECTION_EXTRA_PADDING)
        )
        grid = np.zeros((shared_dim, varying_dim), dtype=int)
        c = 1
        for idx, component in enumerate(components):
            ch, cw = component.shape
            r = random.randint(1, shared_dim - ch - 1)
            grid[r : r + ch, c : c + cw] = component
            c += cw
            if idx < len(gaps):
                c += gaps[idx]
        return grid

    varying_dim = (
        2
        + sum(component.shape[0] for component in components)
        + sum(gaps)
        + random.randint(0, SECTION_EXTRA_PADDING)
    )
    grid = np.zeros((varying_dim, shared_dim), dtype=int)
    r = 1
    for idx, component in enumerate(components):
        ch, cw = component.shape
        c = random.randint(1, shared_dim - cw - 1)
        grid[r : r + ch, c : c + cw] = component
        r += ch
        if idx < len(gaps):
            r += gaps[idx]
    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    total_repeats = random.randint(*TOTAL_REPEAT_RANGE)

    raw_h = random.randint(*CROP_SIZE_RANGE)
    raw_w = random.randint(*CROP_SIZE_RANGE)
    density = random.uniform(*DENSITY_RANGE)
    raw_shape = random_sprite(
        raw_h,
        raw_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )

    shape_bbox = bbox(raw_shape != 0)
    if shape_bbox is None:
        return None
    shape = crop_bbox(raw_shape, shape_bbox)
    tile_mask = shape != 0
    tile_h, tile_w = tile_mask.shape

    start_fg = random.choice(START_FG_POOL)
    background_choices = [color for color in BACKGROUND_POOL if color != start_fg]
    background = random.choice(background_choices)

    foregrounds = _gradient_colors(start_fg, total_repeats)
    output = _stack_gradient_tiles(tile_mask, foregrounds, background, horizontal)

    template = np.zeros((tile_h + 2, tile_w + 2), dtype=int)
    template[1 : tile_h + 1, 1 : tile_w + 1] = shape

    shared_dim = template.shape[0] if horizontal else template.shape[1]
    stencil = _build_stencil(total_repeats, shared_dim, horizontal)
    fg_swatch = _make_swatch(shared_dim, start_fg, horizontal)
    bg_swatch = _make_swatch(shared_dim, background, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
