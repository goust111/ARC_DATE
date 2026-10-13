from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the colors of the nested rectangular rings while preserving their positions.

    Input structure
    The grid is made of concentric one-cell-thick rectangular layers.  Every cell
    at the same distance from the nearest edge has the same color.

    Rule
    Find the ordered color sequence from the outermost layer to the innermost
    layer, reverse that sequence, and recolor each layer at its original depth
    with the corresponding reversed color.

    Key insight
    The geometry is already correct; only the depth-indexed color sequence of the
    nested rings is mirrored from outside to inside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    depth = np.minimum(
        np.minimum(rows, height - 1 - rows),
        np.minimum(cols, width - 1 - cols),
    )
    layer_count = int(depth.max()) + 1

    layer_colors = []
    for layer in range(layer_count):
        colors = np.unique(g[depth == layer])
        assert colors.size == 1, "expected each rectangular layer to be monochrome"
        layer_colors.append(int(colors[0]))

    out = np.empty_like(g)
    for layer, color in enumerate(reversed(layer_colors)):
        out[depth == layer] = color
    return out

# --------------- generator (o5x) ---------------

# Generation knobs: even grid sizes, non-black color pool, and oracle retry budget.
GRID_SIZE_OPTIONS = [6, 8, 10, 12, 14]
LAYER_COLOR_POOL = np.arange(1, 10)
MAX_GENERATION_ATTEMPTS = 300


def _gen_ring_grid(size: int, colors: list[int]) -> np.ndarray:
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    depth = np.minimum(
        np.minimum(rows, size - 1 - rows),
        np.minimum(cols, size - 1 - cols),
    )

    grid = np.empty((size, size), dtype=int)
    for layer, color in enumerate(colors):
        grid[depth == layer] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    size = int(np.random.choice(GRID_SIZE_OPTIONS))
    layer_count = size // 2
    colors = [int(c) for c in np.random.choice(LAYER_COLOR_POOL, layer_count, replace=False)]

    inp = _gen_ring_grid(size, colors)
    expected = _gen_ring_grid(size, list(reversed(colors)))
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 85c4e7cd input")
