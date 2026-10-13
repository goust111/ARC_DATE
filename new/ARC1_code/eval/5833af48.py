from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import Color, crop_bbox, find_components, paste
import random


def _border_bg(g: np.ndarray) -> int:
    border = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _only_component_bbox(components: list[tuple[set[int], tuple[int, int, int, int]]], colors: set[int]) -> tuple[int, int, int, int]:
    matches = [bbox for comp_colors, bbox in components if comp_colors == colors]
    assert len(matches) == 1, f"expected exactly one component with colors {colors}, got {len(matches)}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each teal macro-cell in the upper-right pattern with the teal mask from the left control sprite.

    Input structure:
    - The black background contains three 4-connected foreground components.
    - A left control sprite is a full rectangle of red and teal pixels.
    - An upper-right pattern is a full rectangle of the main color and teal pixels, arranged as uniform macro-cells.
    - A lower solid rectangle of the main color gives the final output size.

    Rule:
    - Compress the upper-right pattern into macro-cells, using the lower rectangle size divided by the control
      sprite size to determine the compressed pattern dimensions.
    - Make an output filled with the main color.
    - For each teal macro-cell, paste the control sprite's teal mask into the corresponding output block, turning
      the control sprite's red pixels into the main color.

    Key insight:
    - The upper-right object is not copied directly; it is a coarse stencil whose teal cells are substituted by
      the entire teal mask from the separate red/teal control sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_bg(g)
    colors = set(map(int, np.unique(g)))
    main_colors = colors - {bg, Color.RED, Color.TEAL}
    assert len(main_colors) == 1, f"expected one varying main color, got {main_colors}"
    main = next(iter(main_colors))

    raw_components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    components = []
    for comp in raw_components:
        coords = comp["coords"]
        comp_colors = set(map(int, np.unique(g[coords[:, 0], coords[:, 1]])))
        components.append((comp_colors, comp["bbox"]))

    control_bbox = _only_component_bbox(components, {Color.RED, Color.TEAL})
    pattern_bbox = _only_component_bbox(components, {main, Color.TEAL})
    canvas_bbox = _only_component_bbox(components, {main})

    control = crop_bbox(g, control_bbox)
    pattern = crop_bbox(g, pattern_bbox)
    canvas = crop_bbox(g, canvas_bbox)
    assert np.all((control == Color.RED) | (control == Color.TEAL)), "control bbox must be filled by red/teal"
    assert np.any(control == Color.TEAL), "control sprite must contain teal mask pixels"
    assert np.all(canvas == main), "lower canvas must be a solid main-color rectangle"

    stamp_h, stamp_w = control.shape
    out_h, out_w = canvas.shape
    assert out_h % stamp_h == 0 and out_w % stamp_w == 0, "canvas must be tiled by control-sized blocks"
    macro_h, macro_w = out_h // stamp_h, out_w // stamp_w

    pattern_h, pattern_w = pattern.shape
    assert pattern_h % macro_h == 0 and pattern_w % macro_w == 0, "pattern must divide into macro-cells"
    cell_h, cell_w = pattern_h // macro_h, pattern_w // macro_w

    selector = np.zeros((macro_h, macro_w), dtype=bool)
    for r in range(macro_h):
        for c in range(macro_w):
            cell = pattern[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cell_colors = set(map(int, np.unique(cell)))
            assert len(cell_colors) == 1, f"macro-cell {(r, c)} is not uniform"
            cell_color = next(iter(cell_colors))
            assert cell_color in {main, Color.TEAL}, f"unexpected macro-cell color {cell_color}"
            selector[r, c] = cell_color == Color.TEAL

    stamp = np.where(control == Color.TEAL, Color.TEAL, main)
    out = np.full((out_h, out_w), main, dtype=int)
    for r in range(macro_h):
        for c in range(macro_w):
            if selector[r, c]:
                out[r * stamp_h : (r + 1) * stamp_h, c * stamp_w : (c + 1) * stamp_w] = stamp
    return out

# --------------- generator (o5x) ---------------

# Palette and object size choices.
MAIN_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)
STAMP_SIZES = (3, 4)
PATTERN_CELL_SIZES = (1, 2)

# Macro stencil size and density ranges.
MACRO_H_RANGE = (2, 3)
MACRO_W_RANGE = (3, 6)
STAMP_TEAL_FRACTION = (0.40, 0.60)
SELECTOR_TEAL_FRACTION = (0.25, 0.50)

# Input layout padding and inter-object gaps.
TOP_PAD_RANGE = (1, 2)
LEFT_PAD_RANGE = (1, 2)
RIGHT_PAD_RANGE = (1, 3)
BOTTOM_PAD_RANGE = (1, 3)
OBJECT_GAP = 1

# Validity retry budget.
GEN_RETRY_LIMIT = 300


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _gen_count_for_fraction(total: int, lo: float, hi: float) -> int:
    low = int(np.ceil(total * lo))
    high = int(np.floor(total * hi))
    return random.randint(low, high)


def _gen_stamp(size: int) -> np.ndarray:
    stamp = np.full((size, size), Color.RED, dtype=int)
    teal_count = _gen_count_for_fraction(size * size, *STAMP_TEAL_FRACTION)
    stamp.ravel()[random.sample(range(size * size), teal_count)] = Color.TEAL
    return stamp


def _gen_selector(macro_h: int, macro_w: int) -> np.ndarray:
    selector = np.zeros((macro_h, macro_w), dtype=bool)
    total = macro_h * macro_w
    teal_count = _gen_count_for_fraction(total, *SELECTOR_TEAL_FRACTION)
    selector.ravel()[random.sample(range(total), teal_count)] = True
    return selector


def _gen_pattern(selector: np.ndarray, main_color: int, cell_size: int) -> np.ndarray:
    macro_cells = np.where(selector, Color.TEAL, main_color).astype(int)
    return np.repeat(np.repeat(macro_cells, cell_size, axis=0), cell_size, axis=1)


def _gen_output(stamp: np.ndarray, selector: np.ndarray, main_color: int) -> np.ndarray:
    stamp_h, stamp_w = stamp.shape
    macro_h, macro_w = selector.shape
    out = np.full((macro_h * stamp_h, macro_w * stamp_w), main_color, dtype=int)
    rendered_stamp = np.where(stamp == Color.TEAL, Color.TEAL, main_color)
    for r, c in np.argwhere(selector):
        paste(out, rendered_stamp, int(r) * stamp_h, int(c) * stamp_w)
    return out


def _gen_assemble_input(
    stamp: np.ndarray,
    pattern: np.ndarray,
    canvas: np.ndarray,
) -> np.ndarray:
    """Arrange the three components with the fixed top-row and lower-canvas layout."""
    top_pad = _randint(TOP_PAD_RANGE)
    left_pad = _randint(LEFT_PAD_RANGE)
    right_pad = _randint(RIGHT_PAD_RANGE)
    bottom_pad = _randint(BOTTOM_PAD_RANGE)

    top_h = max(stamp.shape[0], pattern.shape[0])
    top_w = stamp.shape[1] + OBJECT_GAP + pattern.shape[1]
    used_h = top_h + OBJECT_GAP + canvas.shape[0]
    used_w = max(top_w, canvas.shape[1])
    grid = np.full((top_pad + used_h + bottom_pad, left_pad + used_w + right_pad), Color.BLACK, dtype=int)

    paste(grid, stamp, top_pad, left_pad)
    paste(grid, pattern, top_pad, left_pad + stamp.shape[1] + OBJECT_GAP)
    paste(grid, canvas, top_pad + top_h + OBJECT_GAP, left_pad)
    return grid


def _gen_attempt() -> np.ndarray | None:
    main_color = random.choice(MAIN_COLORS)
    stamp_size = random.choice(STAMP_SIZES)
    stamp = _gen_stamp(stamp_size)

    macro_h = _randint(MACRO_H_RANGE)
    macro_w = _randint(MACRO_W_RANGE)
    selector = _gen_selector(macro_h, macro_w)

    expected = _gen_output(stamp, selector, main_color)
    cell_size = random.choice(PATTERN_CELL_SIZES)
    pattern = _gen_pattern(selector, main_color, cell_size)
    canvas = np.full(expected.shape, main_color, dtype=int)
    inp = _gen_assemble_input(stamp, pattern, canvas)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 5833af48 input")
