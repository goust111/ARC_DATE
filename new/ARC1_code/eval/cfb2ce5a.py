from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _recolor_from_controls(
    block: np.ndarray,
    template: np.ndarray,
    bg: int,
    source_colors: set[int],
) -> np.ndarray:
    mapping: dict[int, int] = {}
    controls = np.argwhere(block != bg)
    assert controls.size, "each block must contain at least one control/template pixel"

    for r, c in controls:
        symbol = int(template[int(r), int(c)])
        color = int(block[int(r), int(c)])
        assert symbol in source_colors, "control pixel must sit over a template symbol"
        if symbol in mapping:
            assert mapping[symbol] == color, "same template symbol must map to one color"
        else:
            mapping[symbol] = color

    out = np.full(template.shape, bg, dtype=int)
    for symbol in source_colors:
        out[template == symbol] = mapping.get(symbol, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the four inner quadrants as flipped, recolored copies of the top-left template.

    Input structure:
    - A one-cell background border surrounds a square interior split into a 2x2 block grid.
    - The upper-left block is a complete two-color template.
    - The other blocks contain sparse control pixels placed where their corresponding
      flipped template symbols should take those colors.

    Rule:
    - Use the upper-left block unchanged for the first quadrant.
    - Horizontally flip the template for the upper-right quadrant, vertically flip it
      for the lower-left quadrant, and flip both axes for the lower-right quadrant.
    - In each quadrant, every non-background input pixel maps the transformed
      template symbol under it to that pixel's color; symbols without a control
      pixel become background.

    Key insight: The control pixels are not objects to move; they are a local
    recoloring legend for the transformed template underneath each quadrant.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid must have an interior"
    assert (h - 2) % 2 == 0 and (w - 2) % 2 == 0, "interior must split into 2x2 blocks"
    block_h = (h - 2) // 2
    block_w = (w - 2) // 2
    assert block_h == block_w, "template blocks must be square"

    assert np.all(g[0, :] == bg) and np.all(g[-1, :] == bg), "top/bottom border must be background"
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "left/right border must be background"

    n = block_h
    base = g[1 : 1 + n, 1 : 1 + n]
    assert not np.any(base == bg), "upper-left template must be complete"
    source_colors = set(map(int, np.unique(base)))
    assert len(source_colors) == 2, "template must use exactly two symbols"

    out = np.full_like(g, bg)
    for br in range(2):
        for bc in range(2):
            r0 = 1 + br * n
            c0 = 1 + bc * n
            template = base
            if br:
                template = np.flipud(template)
            if bc:
                template = np.fliplr(template)

            block = g[r0 : r0 + n, c0 : c0 + n]
            out[r0 : r0 + n, c0 : c0 + n] = _recolor_from_controls(
                block, template, bg, source_colors
            )

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and palette.
BG = 0
GRID_SHAPE = (10, 10)
BLOCK_SIZE = 4
BORDER = 1
NON_BG_COLORS = range(1, 10)

# Template shape sampling.
TEMPLATE_PRIMARY_COUNT_RANGE = (6, 10)
TEMPLATE_FORCED_PRIMARY = ((0, 0), (0, 1))
TEMPLATE_FORCED_SECONDARY = frozenset({(0, 3), (3, 1)})
TEMPLATE_TRANSFORM_COUNT = 8

# Control mapping and retry limits.
CONTROL_QUADRANTS = ((0, 1), (1, 0), (1, 1))
FULL_QUADRANT_COUNT_CHOICES = (2, 3)
GENERATION_RETRIES = 300


def _gen_template() -> np.ndarray:
    colors = random.sample(NON_BG_COLORS, 2)
    primary_count = random.randint(*TEMPLATE_PRIMARY_COUNT_RANGE)

    bits = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for r, c in TEMPLATE_FORCED_PRIMARY:
        bits[r, c] = True

    fixed_cells = set(TEMPLATE_FORCED_PRIMARY) | TEMPLATE_FORCED_SECONDARY
    free = [
        (r, c) for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE)
        if (r, c) not in fixed_cells
    ]
    for r, c in random.sample(free, primary_count - len(TEMPLATE_FORCED_PRIMARY)):
        bits[r, c] = True

    template = np.where(bits, colors[0], colors[1]).astype(int)
    transform = random.randrange(TEMPLATE_TRANSFORM_COUNT)
    if transform >= TEMPLATE_TRANSFORM_COUNT // 2:
        template = np.fliplr(template)
    template = np.rot90(template, transform % (TEMPLATE_TRANSFORM_COUNT // 2))
    return template.copy()


def _place_controls(inp, out, template, source_colors, control_palette, full_quadrants) -> None:
    """Add sparse control pixels to input and matching completed blocks to output."""
    for br, bc in CONTROL_QUADRANTS:
        r0 = BORDER + br * BLOCK_SIZE
        c0 = BORDER + bc * BLOCK_SIZE
        flipped = template
        if br:
            flipped = np.flipud(flipped)
        if bc:
            flipped = np.fliplr(flipped)

        if (br, bc) in full_quadrants:
            mapped_symbols = source_colors[:]
            random.shuffle(mapped_symbols)
        else:
            mapped_symbols = [random.choice(source_colors)]

        mapped_colors = random.sample(control_palette, len(mapped_symbols))
        out_block = out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
        for symbol, color in zip(mapped_symbols, mapped_colors):
            coords = np.argwhere(flipped == symbol)
            rr, cc = coords[random.randrange(len(coords))]
            inp[r0 + int(rr), c0 + int(cc)] = int(color)
            out_block[flipped == symbol] = int(color)


def _gen_attempt() -> np.ndarray | None:
    inp = np.full(GRID_SHAPE, BG, dtype=int)
    out = np.full(GRID_SHAPE, BG, dtype=int)

    template = _gen_template()
    source_colors = list(map(int, np.unique(template)))
    control_palette = [color for color in NON_BG_COLORS if color not in source_colors]

    inp[BORDER : BORDER + BLOCK_SIZE, BORDER : BORDER + BLOCK_SIZE] = template
    out[BORDER : BORDER + BLOCK_SIZE, BORDER : BORDER + BLOCK_SIZE] = template

    full_count = random.choice(FULL_QUADRANT_COUNT_CHOICES)
    full_quadrants = set(random.sample(CONTROL_QUADRANTS, full_count))
    _place_controls(inp, out, template, source_colors, control_palette, full_quadrants)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cfb2ce5a input")
