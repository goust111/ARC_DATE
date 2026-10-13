import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GRID_SIDE = 30
MAX_LAYERS = 5
MAX_RETRIES = 300


def _non_bg_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0, "input must contain a non-background structure"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_color(subgrid: np.ndarray) -> int:
    h, w = subgrid.shape
    values = set(int(v) for v in subgrid[0, :])
    values.update(int(v) for v in subgrid[h - 1, :])
    values.update(int(v) for v in subgrid[1 : h - 1, 0])
    values.update(int(v) for v in subgrid[1 : h - 1, w - 1])
    assert len(values) == 1, "each visible ring must have a single border color"
    color = values.pop()
    assert color != BG
    return color


def _extract_layers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = _non_bg_bbox(grid)
    layers: list[tuple[int, int, int]] = []

    while True:
        subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = subgrid.shape
        assert h > 0 and w > 0

        if np.all(subgrid == subgrid[0, 0]):
            color = int(subgrid[0, 0])
            assert color != BG
            layers.append((h, w, color))
            return layers

        layers.append((h, w, _border_color(subgrid)))
        assert h > 2 and w > 2, "non-final frame must leave room for an inner layer"
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1


def _render_output(layers: list[tuple[int, int, int]]) -> np.ndarray:
    outer_h = layers[0][0]
    out_w = sum(w for _, w, _ in layers) + len(layers) - 1
    assert outer_h <= MAX_GRID_SIDE and out_w <= MAX_GRID_SIDE

    out = np.full((outer_h, out_w), BG, dtype=int)
    col = 0
    for h, w, color in reversed(layers):
        top = (outer_h - h) // 2
        out[top : top + h, col : col + w] = color
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unwrap concentric touching rectangles into side-by-side solid rectangles.

    Input structure:
    - The background is black.
    - One nested stack of axis-aligned concentric rectangles occupies the foreground.
    - Each outer layer is visible as a one-cell ring because the next inner rectangle covers its interior.
    - The innermost layer is either a solid rectangle or a single pixel.

    Rule:
    - Find the bounding box of the nested structure.
    - Repeatedly read the current outer boundary color and shrink one cell inward on every side.
    - Stop when the remaining rectangle is monochrome; that remaining block is the innermost layer.
    - Build a black output whose height matches the outermost rectangle, then place solid rectangles from left to right in inner-to-outer order with one black separator column between neighbors.

    Key insight:
    - The visible colored rings encode hidden full rectangles of decreasing size, so solving the task means peeling the nest apart and repainting each recovered layer as its own solid block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _render_output(_extract_layers(g))


def _max_inner_width(layer_count: int) -> int:
    width_from_output = (MAX_GRID_SIDE - (layer_count * layer_count - 1)) // layer_count
    width_from_object = MAX_GRID_SIDE - 2 * (layer_count - 1)
    return max(0, min(width_from_output, width_from_object))


def _sample_inner_size(rng: random.Random, layer_count: int, innermost_pixel: bool) -> tuple[int, int]:
    if innermost_pixel:
        return 1, 1

    max_h = MAX_GRID_SIDE - 2 * (layer_count - 1)
    max_w = _max_inner_width(layer_count)
    assert max_h >= 1 and max_w >= 1

    for _ in range(100):
        inner_h = rng.randint(1, max_h)
        inner_w = rng.randint(1, max_w)
        if inner_h > 1 or inner_w > 1:
            return inner_h, inner_w
    raise RuntimeError("failed to sample a non-pixel innermost rectangle")


def _build_input(
    layer_count: int,
    inner_h: int,
    inner_w: int,
    colors: list[int],
    top: int,
    left: int,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    outer_h = inner_h + 2 * (layer_count - 1)
    outer_w = inner_w + 2 * (layer_count - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    layers: list[tuple[int, int, int]] = []
    for depth, color in enumerate(colors):
        h = outer_h - 2 * depth
        w = outer_w - 2 * depth
        r0 = top + depth
        c0 = left + depth
        grid[r0 : r0 + h, c0 : c0 + w] = int(color)
        layers.append((h, w, int(color)))

    return grid, layers


def _generate_case(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    layer_count = rng.randint(1, MAX_LAYERS)
    innermost_pixel = rng.choice([False, True])
    inner_h, inner_w = _sample_inner_size(rng, layer_count, innermost_pixel)

    outer_h = inner_h + 2 * (layer_count - 1)
    outer_w = inner_w + 2 * (layer_count - 1)
    assert outer_h <= MAX_GRID_SIDE and outer_w <= MAX_GRID_SIDE
    assert _render_output(
        [(outer_h - 2 * i, outer_w - 2 * i, i + 1) for i in range(layer_count)]
    ).shape[1] <= MAX_GRID_SIDE

    grid_h = outer_h + rng.randint(0, min(6, MAX_GRID_SIDE - outer_h))
    grid_w = outer_w + rng.randint(0, min(6, MAX_GRID_SIDE - outer_w))
    top = rng.randint(0, grid_h - outer_h)
    left = rng.randint(0, grid_w - outer_w)
    colors = rng.sample(PALETTE, layer_count)

    return _build_input(layer_count, inner_h, inner_w, colors, top, left, grid_h, grid_w)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid, _layers = _generate_case(rng)
        return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
