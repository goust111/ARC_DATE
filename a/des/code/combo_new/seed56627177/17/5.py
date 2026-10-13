import random

import numpy as np


BACKGROUND = 0
GENERATION_RETRIES = 200


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _layer_counts(outer_h: int, outer_w: int, layers: int, solid_inner: bool):
    counts = []
    for depth in range(layers):
        height = outer_h - 2 * depth
        width = outer_w - 2 * depth
        if height < 1 or width < 1:
            return None
        is_last = depth == layers - 1
        if is_last and solid_inner:
            counts.append(int(height * width))
        else:
            if height < 3 or width < 3:
                return None
            counts.append(int(2 * height + 2 * width - 4))
    return counts


def _valid_configs():
    configs = []
    for layers in range(2, 6):
        for solid_inner in (False, True):
            for outer_h in range(3, 18):
                for outer_w in range(3, 18):
                    counts = _layer_counts(outer_h, outer_w, layers, solid_inner)
                    if counts is None or max(counts) > 30:
                        continue
                    configs.append((layers, solid_inner, outer_h, outer_w))
    return configs


VALID_CONFIGS = _valid_configs()


def main(grid: np.ndarray) -> np.ndarray:
    """Peel concentric rectangular layers into a horizontal count bar chart.

    Input structure:
    - The background is black.
    - One nested object made of 2 to 5 differently colored concentric rectangular layers sits on the grid.
    - Each outer layer is a one-pixel-thick rectangular ring; the innermost layer may be either another ring or a solid rectangle.

    Rule:
    - Repeatedly find the current outer bounding box of the non-background structure and identify the single color covering that box's perimeter.
    - Record the number of pixels belonging to that outer layer, remove it, and continue from outside to inside.
    - Build an output grid with one row per layer and width equal to the largest recorded pixel count.
    - Row k is filled from the left with exactly that layer's pixel count in the layer's color; remaining cells stay black.

    Key insight:
    - The spatial nesting order is recovered by peeling the unique perimeter color of the remaining rectangle, then each peeled layer is converted into a one-dimensional length equal to its pixel total.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    remaining = g.copy()
    layers = []

    while True:
        bbox = _bbox(remaining != BACKGROUND)
        if bbox is None:
            break

        r0, c0, r1, c1 = bbox
        box = remaining[r0 : r1 + 1, c0 : c1 + 1]
        perimeter = _ring_mask(*box.shape)
        perimeter_colors = {int(v) for v in box[perimeter] if int(v) != BACKGROUND}
        assert len(perimeter_colors) == 1, "outer perimeter must have a unique color"
        color = next(iter(perimeter_colors))

        color_mask = remaining == color
        color_bbox = _bbox(color_mask)
        assert color_bbox == bbox, "each layer color must occupy the current outer bbox"
        local_mask = color_mask[r0 : r1 + 1, c0 : c1 + 1]
        pixel_count = int(color_mask.sum())

        remaining[color_mask] = BACKGROUND
        next_bbox = _bbox(remaining != BACKGROUND)
        if next_bbox is None:
            assert np.array_equal(local_mask, perimeter) or local_mask.all(), (
                "final layer must be a ring or a solid rectangle"
            )
        else:
            assert np.array_equal(local_mask, perimeter), "non-final layers must be one-pixel rings"
            assert next_bbox == (r0 + 1, c0 + 1, r1 - 1, c1 - 1), (
                "layers must be perfectly nested with one-cell spacing"
            )

        layers.append((color, pixel_count))

    assert 2 <= len(layers) <= 5

    width = max(count for _, count in layers)
    out = np.zeros((len(layers), width), dtype=int)
    for row, (color, count) in enumerate(layers):
        out[row, :count] = color
    return out


def _gen_attempt(rng: random.Random):
    layers, solid_inner, outer_h, outer_w = rng.choice(VALID_CONFIGS)
    colors = rng.sample(range(1, 10), layers)

    pad_h = rng.randint(2, min(12, 30 - outer_h))
    pad_w = rng.randint(2, min(12, 30 - outer_w))
    top = rng.randint(1, pad_h - 1)
    left = rng.randint(1, pad_w - 1)
    height = outer_h + pad_h
    width = outer_w + pad_w
    grid = np.zeros((height, width), dtype=int)

    for depth, color in enumerate(colors):
        r0 = top + depth
        c0 = left + depth
        r1 = top + outer_h - depth - 1
        c1 = left + outer_w - depth - 1
        if depth == layers - 1 and solid_inner:
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        try:
            out = main(grid)
        except Exception:
            continue
        if max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
