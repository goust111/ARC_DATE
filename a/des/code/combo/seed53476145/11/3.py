from __future__ import annotations

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
RNG = np.random.default_rng()

FRAME_COUNT_RANGE = (1, 5)
SOLID_CENTER_HEIGHT_RANGE = (2, 6)
SOLID_CENTER_WIDTH_RANGE = (2, 6)
MARGIN_RANGE = (1, 6)
PIXEL_CENTER_PROBABILITY = 0.5
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_pixel_count(height: int, width: int) -> int:
    inner_h = max(int(height) - 2, 0)
    inner_w = max(int(width) - 2, 0)
    return int(height * width - inner_h * inner_w)


def _extract_layers(crop: np.ndarray) -> list[tuple[int, int]]:
    layers: list[tuple[int, int]] = []
    cur = np.asarray(crop, dtype=int)

    while True:
        h, w = map(int, cur.shape)
        assert h > 0 and w > 0
        assert np.all(cur != BG), "concentric layers must be tightly packed"

        unique = np.unique(cur)
        if len(unique) == 1:
            # The final solid center is treated as the innermost ring of its bbox.
            layers.append((int(unique[0]), _ring_pixel_count(h, w)))
            break

        assert h >= 3 and w >= 3, "a non-uniform layer must have an inner region"
        border = np.concatenate((cur[0, :], cur[-1, :], cur[1:-1, 0], cur[1:-1, -1]))
        border_colors = np.unique(border)
        assert len(border_colors) == 1 and int(border_colors[0]) != BG

        color = int(border_colors[0])
        layers.append((color, _ring_pixel_count(h, w)))

        inner = cur[1:-1, 1:-1]
        assert inner.size > 0
        assert np.all(inner != color), "adjacent layers must use different colors"
        cur = inner

    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll concentric rectangular rings into horizontal bars.

    Input structure
    - Background is black.
    - A single stack of tightly nested 1px rectangular rings sits in the grid.
    - Colors are distinct from outer to inner.
    - The innermost layer may be a solid rectangle or a single pixel.

    Rule
    - Read the layers from outer to inner.
    - For each layer, measure the number of pixels on that layer's rectangular border.
    - Output one row per layer, left-aligned, using the layer color and measured length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    box = _bbox(g != BG)
    assert box is not None, "expected at least one non-background layer"
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]

    layers = _extract_layers(crop)
    out_w = int(layers[0][1])
    out = np.full((len(layers), out_w), BG, dtype=int)
    for r, (color, length) in enumerate(layers):
        out[r, : int(length)] = int(color)
    return out


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(int(lo), int(hi) + 1))


def _choose_center_shape() -> tuple[int, int]:
    if float(RNG.random()) < PIXEL_CENTER_PROBABILITY:
        return 1, 1
    return (
        _randint(*SOLID_CENTER_HEIGHT_RANGE),
        _randint(*SOLID_CENTER_WIDTH_RANGE),
    )


def _sample_colors(count: int) -> list[int]:
    picked = RNG.choice(np.array(NON_BG_COLORS, dtype=int), size=int(count), replace=False)
    return [int(v) for v in picked]


def _build_object(layer_colors: list[int], center_h: int, center_w: int) -> np.ndarray:
    obj = np.full((int(center_h), int(center_w)), int(layer_colors[-1]), dtype=int)
    for color in reversed(layer_colors[:-1]):
        grown = np.full((obj.shape[0] + 2, obj.shape[1] + 2), int(color), dtype=int)
        grown[1:-1, 1:-1] = obj
        obj = grown
    return obj


def _expected_output(layer_colors: list[int], center_h: int, center_w: int, frame_count: int) -> np.ndarray:
    dims = [
        (int(center_h + 2 * depth), int(center_w + 2 * depth))
        for depth in range(int(frame_count), -1, -1)
    ]
    lengths = [_ring_pixel_count(h, w) for h, w in dims]

    out = np.full((len(layer_colors), int(lengths[0])), BG, dtype=int)
    for r, (color, length) in enumerate(zip(layer_colors, lengths)):
        out[r, : int(length)] = int(color)
    return out


def _generate_attempt() -> np.ndarray | None:
    center_h, center_w = _choose_center_shape()
    frame_count = _randint(*FRAME_COUNT_RANGE)
    layer_colors = _sample_colors(frame_count + 1)

    obj = _build_object(layer_colors, center_h, center_w)
    expected = _expected_output(layer_colors, center_h, center_w, frame_count)

    top = _randint(*MARGIN_RANGE)
    bottom = _randint(*MARGIN_RANGE)
    left = _randint(*MARGIN_RANGE)
    right = _randint(*MARGIN_RANGE)

    grid = np.full((obj.shape[0] + top + bottom, obj.shape[1] + left + right), BG, dtype=int)
    grid[top : top + obj.shape[0], left : left + obj.shape[1]] = obj

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring input")
