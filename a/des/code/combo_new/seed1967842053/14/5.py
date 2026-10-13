import numpy as np


def _nested_layers(grid: np.ndarray):
    rows, cols = np.where(grid != 0)
    if rows.size == 0:
        return []

    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())
    block = grid[top : bottom + 1, left : right + 1]

    layers = []
    r0, c0 = 0, 0
    r1, c1 = block.shape[0] - 1, block.shape[1] - 1
    while r0 <= r1 and c0 <= c1:
        color = block[r0, c0]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        layers.append((h, w, color))

        region = block[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(region == color):
            break
        r0 += 1
        r1 -= 1
        c0 += 1
        c1 -= 1

    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """Unwrap nested rectangular frames into separate left-to-right frames.

    Input structure:
    The grid has a black background and one non-black object made from
    concentric, one-pixel-wide rectangular frames with distinct colors. The
    outer frame is the bounding box of the object, and the deepest layer is a
    solid rectangle.

    Rule:
    Read the layers from the outside inward. The output height is the outer
    frame height, and the output width is the sum of every layer's original
    width plus one black separator column between neighboring layers. Draw each
    layer top-aligned from left to right: all non-final layers become isolated
    hollow rectangular frames in their own color, and the final layer becomes a
    solid rectangle in its own color.

    Key insight:
    The visible nesting order supplies the ordering of the separated output
    objects; colors identify layers but do not determine their positions.
    """
    layers = _nested_layers(grid)
    if not layers:
        return np.zeros((1, 1), dtype=grid.dtype)

    out_h = layers[0][0]
    out_w = sum(w for _, w, _ in layers) + len(layers) - 1
    out = np.zeros((out_h, out_w), dtype=grid.dtype)

    x = 0
    last = len(layers) - 1
    for i, (h, w, color) in enumerate(layers):
        if i == last:
            out[:h, x : x + w] = color
        else:
            out[0, x : x + w] = color
            out[h - 1, x : x + w] = color
            out[:h, x] = color
            out[:h, x + w - 1] = color
        x += w + 1

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    layer_count = int(rng.integers(3, 6))

    min_outer = 2 * (layer_count - 1) + 1
    max_outer_w = (30 + (layer_count - 1) ** 2) // layer_count
    outer_w = int(rng.integers(min_outer, max_outer_w + 1))

    max_outer_h = 30
    outer_h = int(rng.integers(min_outer, max_outer_h + 1))

    grid_h = int(rng.integers(outer_h, 31))
    grid_w = int(rng.integers(outer_w, 31))
    top = int(rng.integers(0, grid_h - outer_h + 1))
    left = int(rng.integers(0, grid_w - outer_w + 1))

    colors = rng.choice(np.arange(1, 10), size=layer_count, replace=False)
    grid = np.zeros((grid_h, grid_w), dtype=np.int64)

    for depth, color in enumerate(colors):
        r0 = top + depth
        r1 = top + outer_h - 1 - depth
        c0 = left + depth
        c1 = left + outer_w - 1 - depth
        if depth == layer_count - 1:
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color

    return grid
