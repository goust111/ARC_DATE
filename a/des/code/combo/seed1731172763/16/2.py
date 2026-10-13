from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the colors of concentric square rings one layer inward.

    Input structure
    - The background is black.
    - The only foreground object is a filled odd-sized square made of concentric
      one-cell-thick square rings with a single center pixel.
    - Each ring layer has a uniform color, and adjacent layers have no gaps.

    Rule
    - Number the layers from outermost to innermost.
    - Recolor each layer in place so layer 0 gets the old innermost color and
      every other layer gets the old color from the layer just outside it.

    Key insight
    - The geometry stays fixed; only the mapping from ring index to color
      rotates by one step.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0 if np.any(g == 0) else int(np.bincount(g.ravel()).argmax())

    mask = g != bg
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a concentric-ring object"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    assert side_h == side_w, "foreground bbox must be square"
    assert side_h % 2 == 1, "foreground square must have a single center pixel"

    square = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert np.all(square != bg), "rings must fill the square without gaps"

    layer_count = (side_h + 1) // 2
    layer_masks = [_layer_mask(side_h, layer) for layer in range(layer_count)]
    layer_colors = []
    for layer_mask in layer_masks:
        colors = np.unique(square[layer_mask])
        assert len(colors) == 1, "each ring layer must have a single color"
        layer_colors.append(int(colors[0]))

    rotated = [layer_colors[-1], *layer_colors[:-1]]
    out = g.copy()
    out_square = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    for layer_mask, color in zip(layer_masks, rotated):
        out_square[layer_mask] = int(color)
    return out


def _layer_mask(side: int, layer: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    lo = int(layer)
    hi = int(side - layer - 1)
    mask[lo, lo : hi + 1] = True
    mask[hi, lo : hi + 1] = True
    mask[lo : hi + 1, lo] = True
    mask[lo : hi + 1, hi] = True
    return mask


# --------------- generator ---------------

BG = 0
PALETTE = tuple(range(1, 10))
MIN_LAYERS = 2
MAX_LAYERS = len(PALETTE)
MARGIN_MIN = 1
MARGIN_MAX = 6
GENERATION_ATTEMPTS = 300


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring input")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    layer_count = int(rng.integers(MIN_LAYERS, MAX_LAYERS + 1))
    side = 2 * layer_count - 1
    pad_y = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    pad_x = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))

    height = side + 2 * pad_y
    width = side + 2 * pad_x
    grid = np.full((height, width), BG, dtype=int)

    colors = list(map(int, rng.permutation(PALETTE)[:layer_count]))
    if len(set(colors)) != layer_count:
        return None

    r0, c0 = pad_y, pad_x
    for layer, color in enumerate(colors):
        _paint_layer(grid, r0, c0, side, layer, color)

    # Keep the puzzle visually clear and the transformation non-trivial.
    if np.array_equal(main(grid), grid):
        return None
    return grid


def _paint_layer(
    grid: np.ndarray, top: int, left: int, side: int, layer: int, color: int
) -> None:
    lo = int(layer)
    hi = int(side - layer - 1)
    grid[top + lo, left + lo : left + hi + 1] = int(color)
    grid[top + hi, left + lo : left + hi + 1] = int(color)
    grid[top + lo : top + hi + 1, left + lo] = int(color)
    grid[top + lo : top + hi + 1, left + hi] = int(color)
