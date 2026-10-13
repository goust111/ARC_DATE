import numpy as np


def _ring_colors(g: np.ndarray) -> list[int]:
    height, width = map(int, g.shape)
    colors: list[int] = []

    for offset in range((min(height, width) + 1) // 2):
        r0, c0 = offset, offset
        r1, c1 = height - 1 - offset, width - 1 - offset
        coords = [(r0, c) for c in range(c0, c1 + 1)]
        coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
        if r1 > r0:
            coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
        if c1 > c0:
            coords.extend((r, c0) for r in range(r1 - 1, r0, -1))

        ring_values = {int(g[r, c]) for r, c in coords}
        assert len(ring_values) == 1, "each rectangular ring must be monochromatic"
        colors.append(ring_values.pop())

    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the concentric rectangular rings by cycling the outside-to-inside color order one step outward.

    Input structure:
    - The whole grid is partitioned into one-cell-thick concentric rectangular rings.
    - Each ring is monochromatic.
    - Reading rings from outside to inside gives at least two distinct colors; a color may reappear on a later ring.

    Rule:
    - Record the distinct colors in their first outside-to-inside ring order.
    - Replace every pixel of each recorded color with the previous color in that cyclic order.

    Key insight:
    - The transformation is a color-cycle over nesting order, not over numeric color ids or ring sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) > 0, "expected a nonempty grid"

    order: list[int] = []
    for color in _ring_colors(g):
        if color not in order:
            order.append(color)

    assert len(order) >= 2, "expected at least two distinct ring colors"
    assert set(map(int, np.unique(g))) == set(order), "all colors must appear in the ring order"

    mapping = {color: order[i - 1] for i, color in enumerate(order)}
    out = np.empty_like(g)
    for old_color, new_color in mapping.items():
        out[g == old_color] = new_color
    return out

# --------------- generator (o5x) ---------------

# Canvas, color, and retry sampling.
GEN_CANVAS_SIZES = (6, 8, 10)
GEN_NUM_COLORS = 3
GEN_PALETTE_SIZE = 10
GEN_MAX_ATTEMPTS = 300


def _gen_paint_ring(grid: np.ndarray, offset: int, color: int) -> None:
    n = int(grid.shape[0])
    r0 = c0 = offset
    r1 = c1 = n - 1 - offset
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(GEN_CANVAS_SIZES))
    colors = [int(c) for c in np.random.choice(GEN_PALETTE_SIZE, size=GEN_NUM_COLORS, replace=False)]

    inp = np.empty((n, n), dtype=int)
    out = np.empty_like(inp)
    mapping = {color: colors[(i - 1) % GEN_NUM_COLORS] for i, color in enumerate(colors)}

    for offset in range(n // 2):
        color = colors[offset % GEN_NUM_COLORS]
        _gen_paint_ring(inp, offset, color)
        _gen_paint_ring(out, offset, mapping[color])

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bda2d7a6 input")
