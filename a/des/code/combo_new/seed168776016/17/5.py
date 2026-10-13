import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ARC_MAX_SIDE = 30
INNER_SIZE_CHOICES = tuple((h, w) for h in range(1, 4) for w in range(1, 4))
MIN_LAYERS = 3
MAX_LAYERS = 5
LEFT_MARGIN_MAX = 3
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_ring(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top, left : left + width] = color
    canvas[top + height - 1, left : left + width] = color
    canvas[top : top + height, left] = color
    canvas[top : top + height, left + width - 1] = color


def _extract_layers(grid: np.ndarray) -> tuple[list[tuple[int, int, int, bool]], tuple[int, int, int, int]]:
    mask = grid != BG
    r0, c0, r1, c1 = _bbox(mask)
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(block != BG), "nested rectangle stack should fill its outer bbox"

    layers_outer_to_inner: list[tuple[int, int, int, bool]] = []
    current = block
    while True:
        height, width = map(int, current.shape)
        if np.all(current == current[0, 0]):
            color = int(current[0, 0])
            assert color != BG, "innermost layer must be colored"
            layers_outer_to_inner.append((height, width, color, True))
            break

        border = np.zeros_like(current, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        colors = np.unique(current[border])
        assert colors.size == 1, "each layer border must be monochrome"
        color = int(colors[0])
        assert color != BG, "ring color must be non-background"
        layers_outer_to_inner.append((height, width, color, False))
        assert height > 2 and width > 2, "rings must leave room for an inner layer"
        current = current[1:-1, 1:-1]

    layers = list(reversed(layers_outer_to_inner))
    assert len(layers) >= MIN_LAYERS, "puzzle requires at least three nested layers"
    return layers, (r0, c0, r1, c1)


def _packed_width(inner_width: int, layers: int) -> int:
    return sum(inner_width + 2 * depth for depth in range(layers)) + (layers - 1)


def _build_nested_block(inner_height: int, inner_width: int, colors_inner_to_outer: list[int]) -> np.ndarray:
    layers = len(colors_inner_to_outer)
    outer_height = inner_height + 2 * (layers - 1)
    outer_width = inner_width + 2 * (layers - 1)
    block = np.zeros((outer_height, outer_width), dtype=int)

    for depth, color in enumerate(colors_inner_to_outer):
        height = inner_height + 2 * depth
        width = inner_width + 2 * depth
        top = layers - depth - 1
        left = layers - depth - 1
        if depth == 0:
            block[top : top + height, left : left + width] = color
        else:
            _draw_ring(block, top, left, height, width, int(color))
    return block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unpack a nested stack of concentric rectangular layers into separate horizontal rectangles.

    Input structure:
    - The background is black.
    - One multicolor structure occupies a solid outer rectangle on the left side of the grid.
    - Its colors form tightly nested concentric rectangular layers: the innermost layer is a
      small solid rectangle, and every outer layer is a one-pixel-thick rectangular ring.

    Rule:
    - Read the layers from the center outward by repeatedly peeling the monochrome border of the
      solid outer rectangle.
    - Erase the original nested structure to black.
    - Repaint the recovered layers from left to right in inner-to-outer order with one black
      separator column between neighbors; keep layer 1 solid and draw every later layer as an
      independent hollow rectangular ring of its original size.
    - Align all placed layers so their vertical centers match the outermost rectangle's center row.

    Key insight:
    - The nested rectangle is a compressed encoding of several separate objects; solving the task
      requires decomposing the stack layer-by-layer and then laying those layers out independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    layers, (r0, _c0, r1, _c1) = _extract_layers(g)
    packed_width = sum(layer_width for _h, layer_width, _color, _solid in layers) + (len(layers) - 1)
    assert packed_width <= width, "output packing must fit within the original grid width"

    center_row = (r0 + r1) // 2
    out = np.full_like(g, BG)

    left = 0
    for layer_height, layer_width, color, solid in layers:
        top = center_row - (layer_height - 1) // 2
        assert 0 <= top and top + layer_height <= height
        assert left + layer_width <= width
        if solid:
            out[top : top + layer_height, left : left + layer_width] = color
        else:
            _draw_ring(out, top, left, layer_height, layer_width, color)
        left += layer_width + 1

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_height, inner_width = rng.choice(INNER_SIZE_CHOICES)
    valid_layer_counts = [
        layers
        for layers in range(MIN_LAYERS, MAX_LAYERS + 1)
        if _packed_width(inner_width, layers) <= ARC_MAX_SIDE
    ]
    if not valid_layer_counts:
        return None

    layer_count = rng.choice(valid_layer_counts)
    outer_height = inner_height + 2 * (layer_count - 1)
    outer_width = inner_width + 2 * (layer_count - 1)
    packed_width = _packed_width(inner_width, layer_count)

    grid_height = rng.randint(outer_height, ARC_MAX_SIDE)
    grid_width = rng.randint(packed_width, ARC_MAX_SIDE)
    top = rng.randint(0, grid_height - outer_height)
    left = rng.randint(0, min(LEFT_MARGIN_MAX, grid_width - outer_width))

    colors = rng.sample(PALETTE, layer_count)
    block = _build_nested_block(inner_height, inner_width, colors)

    grid = np.zeros((grid_height, grid_width), dtype=int)
    grid[top : top + outer_height, left : left + outer_width] = block

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
