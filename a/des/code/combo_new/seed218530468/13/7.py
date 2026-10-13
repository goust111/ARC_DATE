from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _pack_layers(layers: list[np.ndarray]) -> np.ndarray:
    height = layers[0].shape[0]
    width = sum(layer.shape[1] for layer in layers) + len(layers) - 1
    out = np.zeros((height, width), dtype=int)

    col = 0
    for i, layer in enumerate(layers):
        h, w = layer.shape
        out[height - h :, col : col + w] = layer
        col += w
        if i != len(layers) - 1:
            col += 1
    return out


def _extract_layers(grid: np.ndarray) -> list[np.ndarray]:
    bbox = _bbox(grid != BG)
    assert bbox is not None, "input must contain a foreground object"

    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert sub.shape[0] == sub.shape[1], "nested object must occupy a square bbox"

    layers: list[np.ndarray] = []
    side0 = sub.shape[0]
    depth = 0

    while True:
        side = side0 - 2 * depth
        assert side > 0
        window = sub[depth : depth + side, depth : depth + side]

        if side <= 2:
            colors = [int(v) for v in np.unique(window) if int(v) != BG]
            assert len(colors) == 1
            color = colors[0]
            assert np.all(window == color)
            layers.append(np.full((side, side), color, dtype=int))
            break

        border = _border_mask(side)
        border_vals = window[border]
        colors = {int(v) for v in border_vals if int(v) != BG}
        assert len(colors) == 1
        color = next(iter(colors))
        assert np.count_nonzero(border_vals == BG) == 1
        assert np.all((border_vals == BG) | (border_vals == color))

        sprite = np.zeros((side, side), dtype=int)
        sprite[border & (window == color)] = color
        layers.append(sprite)
        depth += 1

    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """Separate the nested square layers and line them up from outer to inner.

    Input structure
    The background is black. A single compact object is made of tightly nested
    concentric square layers with distinct non-black colors. Every non-innermost
    layer is a one-pixel square frame with one black gap on exactly one side,
    while the innermost layer is a solid 1x1 or 2x2 block.

    Rule
    Find the square bounding box of the nested object and peel it one border at
    a time from the outside inward. Reconstruct each peeled layer as its own
    sprite, preserving its color and the side where the one-pixel gap appears.
    Place these sprites left to right in outer-to-inner order with one black
    column between neighbors and with their bottoms aligned.

    Key insight
    The object is a stack of visible square shells: once each shell is isolated
    by successive insets, the task becomes a bottom-aligned horizontal packing
    of those preserved layer shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    layers = _extract_layers(g)
    return _pack_layers(layers)


def _gap_index(side: int, rng: random.Random) -> int:
    if side % 2 == 1:
        return side // 2
    return rng.choice((side // 2 - 1, side // 2))


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    side: int,
    color: int,
    gap_side: int,
    gap_idx: int,
) -> np.ndarray:
    sprite = np.zeros((side, side), dtype=int)
    sprite[0, :] = color
    sprite[-1, :] = color
    sprite[:, 0] = color
    sprite[:, -1] = color

    if gap_side == 0:
        sprite[0, gap_idx] = BG
    elif gap_side == 1:
        sprite[gap_idx, side - 1] = BG
    elif gap_side == 2:
        sprite[side - 1, gap_idx] = BG
    else:
        sprite[gap_idx, 0] = BG

    grid[top : top + side, left : left + side] = np.where(
        sprite != BG, sprite, grid[top : top + side, left : left + side]
    )
    return sprite


def _near_center_offset(span: int, used: int, rng: random.Random) -> int:
    extra = span - used
    if extra <= 0:
        return 0
    center = extra // 2
    lo = max(1, center - 2) if extra >= 2 else 0
    hi = min(extra - 1, center + 2) if extra >= 2 else extra
    if lo > hi:
        return center
    return rng.randint(lo, hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_side = rng.choice((1, 2))
    layer_count = rng.randint(2, 5 if inner_side == 1 else 4)
    outer_side = inner_side + 2 * (layer_count - 1)

    colors = rng.sample(COLORS, layer_count)
    layers: list[np.ndarray] = []

    for depth in range(layer_count):
        side = outer_side - 2 * depth
        color = colors[depth]
        if side <= 2:
            layers.append(np.full((side, side), color, dtype=int))
        else:
            gap_side = rng.randrange(4)
            gap_idx = _gap_index(side, rng)
            sprite = np.full((side, side), BG, dtype=int)
            sprite[0, :] = color
            sprite[-1, :] = color
            sprite[:, 0] = color
            sprite[:, -1] = color
            if gap_side == 0:
                sprite[0, gap_idx] = BG
            elif gap_side == 1:
                sprite[gap_idx, -1] = BG
            elif gap_side == 2:
                sprite[-1, gap_idx] = BG
            else:
                sprite[gap_idx, 0] = BG
            layers.append(sprite)

    out = _pack_layers(layers)
    if max(out.shape) > 30:
        return None

    grid_h = rng.randint(outer_side + 2, min(30, outer_side + 14))
    grid_w = rng.randint(outer_side + 2, min(30, outer_side + 14))
    grid = np.zeros((grid_h, grid_w), dtype=int)

    top = _near_center_offset(grid_h, outer_side, rng)
    left = _near_center_offset(grid_w, outer_side, rng)

    for depth, color in enumerate(colors):
        side = outer_side - 2 * depth
        r = top + depth
        c = left + depth
        if side <= 2:
            grid[r : r + side, c : c + side] = color
            continue
        gap_side = None
        gap_idx = None
        sprite = layers[depth]
        gap_pos = np.argwhere((_border_mask(side)) & (sprite == BG))
        assert gap_pos.shape == (1, 2)
        gr, gc = map(int, gap_pos[0])
        if gr == 0:
            gap_side = 0
            gap_idx = gc
        elif gc == side - 1:
            gap_side = 1
            gap_idx = gr
        elif gr == side - 1:
            gap_side = 2
            gap_idx = gc
        else:
            gap_side = 3
            gap_idx = gr
        _draw_frame(grid, r, c, side, color, gap_side, gap_idx)

    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(
        f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts"
    )
