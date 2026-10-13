from __future__ import annotations

import numpy as np


GENERATION_ATTEMPTS = 100
RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_layers(grid: np.ndarray) -> list[dict[str, int | bool]]:
    bbox = _bbox(grid != 0)
    if bbox is None:
        raise ValueError("expected at least one non-black pixel")

    r0, c0, r1, c1 = bbox
    if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == 0):
        raise ValueError("the nested structure must fill its bounding box")

    layers: list[dict[str, int | bool]] = []
    top, left, bottom, right = r0, c0, r1, c1
    outer_top = r0

    while top <= bottom and left <= right:
        sub = grid[top : bottom + 1, left : right + 1]
        color = int(sub[0, 0])
        if color == 0:
            raise ValueError("layer border cannot be black")

        border_ok = (
            np.all(sub[0, :] == color)
            and np.all(sub[-1, :] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        )
        if not border_ok:
            raise ValueError("each layer must have a uniform rectangular boundary")

        filled = bool(np.all(sub == color))
        layers.append(
            {
                "color": color,
                "height": int(sub.shape[0]),
                "width": int(sub.shape[1]),
                "depth": int(top - outer_top),
                "filled": filled,
            }
        )

        if filled:
            break

        top += 1
        left += 1
        bottom -= 1
        right -= 1

    colors = [int(layer["color"]) for layer in layers]
    if len(colors) != len(set(colors)):
        raise ValueError("layer colors must be distinct")
    return layers


def _make_patch(height: int, width: int, color: int, filled: bool) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    if filled or height <= 2 or width <= 2:
        patch[:, :] = color
        return patch

    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel a nested colored rectangle stack into separate left-to-right layers.

    Input structure:
    - The background is black.
    - A single non-black object is a fully tiled nest of colored rectangles:
      each outer layer is a one-cell-thick rectangular border around the next
      layer, and the innermost layer is a solid rectangle or a single pixel.
    - Every layer has a different color.

    Rule:
    - Find the non-black bounding box and peel the structure from the outside in.
    - For each layer, keep only that layer's own pixels: border layers become
      hollow rectangles with black interiors, while the final solid core stays solid.
    - Arrange the extracted layers from left to right with one black separator
      column, using the outer layer's height as the output height and keeping
      each smaller layer vertically inset by its original nesting depth.

    Key insight:
    - The colored nest completely fills one bounding box, so the task is an
      onion-peel over successive shrinking rectangles rather than any kind of scaling.
    """
    g = np.asarray(grid, dtype=int)
    layers = _extract_layers(g)

    outer_height = int(layers[0]["height"])
    total_width = sum(int(layer["width"]) for layer in layers) + max(0, len(layers) - 1)
    out = np.zeros((outer_height, total_width), dtype=int)

    col = 0
    for idx, layer in enumerate(layers):
        if idx > 0:
            col += 1

        height = int(layer["height"])
        width = int(layer["width"])
        depth = int(layer["depth"])
        color = int(layer["color"])
        filled = bool(layer["filled"])

        patch = _make_patch(height, width, color, filled)
        out[depth : depth + height, col : col + width] = patch
        col += width

    return out


def _split_margin(extra: int) -> tuple[int, int]:
    low = extra // 2
    high = extra - low
    if high > low and int(RNG.integers(0, 2)) == 1:
        return high, low
    return low, high


def _gen_attempt() -> np.ndarray | None:
    layer_count = int(RNG.integers(3, 6))
    pixel_core = bool(RNG.integers(0, 2))

    max_inner_w = (31 - layer_count * layer_count) // layer_count
    if max_inner_w < 1:
        return None

    max_inner_h = 30 - 2 * (layer_count - 1)
    if max_inner_h < 1:
        return None

    if pixel_core:
        inner_h = 1
        inner_w = 1
    else:
        inner_w = int(RNG.integers(1, max_inner_w + 1))
        inner_h = int(RNG.integers(1, max_inner_h + 1))
        if inner_h == 1 and inner_w == 1:
            if max_inner_h > 1:
                inner_h = 2
            elif max_inner_w > 1:
                inner_w = 2
            else:
                return None

    outer_h = inner_h + 2 * (layer_count - 1)
    outer_w = inner_w + 2 * (layer_count - 1)
    if outer_h > 30 or outer_w > 30:
        return None

    extra_h = int(RNG.integers(0, 30 - outer_h + 1))
    extra_w = int(RNG.integers(0, 30 - outer_w + 1))
    top_margin, bottom_margin = _split_margin(extra_h)
    left_margin, right_margin = _split_margin(extra_w)
    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin

    colors = RNG.choice(np.arange(1, 10), size=layer_count, replace=False).astype(int)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    top = top_margin
    left = left_margin
    height = outer_h
    width = outer_w

    for color in colors[:-1]:
        grid[top, left : left + width] = color
        grid[top + height - 1, left : left + width] = color
        grid[top : top + height, left] = color
        grid[top : top + height, left + width - 1] = color
        top += 1
        left += 1
        height -= 2
        width -= 2

    grid[top : top + height, left : left + width] = int(colors[-1])

    layers = _extract_layers(grid)
    if len(layers) != layer_count:
        return None
    if pixel_core != (int(layers[-1]["height"]) == 1 and int(layers[-1]["width"]) == 1):
        return None

    out = main(grid)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
