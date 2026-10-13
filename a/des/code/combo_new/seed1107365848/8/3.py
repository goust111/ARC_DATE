from __future__ import annotations

import random
import numpy as np


SIDE_CHOICES = ("top", "bottom", "left", "right")
ODD_SIZES = (7, 9, 11, 13, 15, 17, 19)
MAX_GEN_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _edge_values(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return np.asarray(grid[0, :], dtype=int)
    if side == "bottom":
        return np.asarray(grid[-1, :], dtype=int)
    if side == "left":
        return np.asarray(grid[:, 0], dtype=int)
    if side == "right":
        return np.asarray(grid[:, -1], dtype=int)
    raise ValueError(f"unknown side: {side}")


def _shared_corner_indices(side: str, other: str, size: int) -> set[int]:
    if side == "top":
        if other == "left":
            return {0}
        if other == "right":
            return {0}
        return set()
    if side == "bottom":
        if other == "left":
            return {size - 1}
        if other == "right":
            return {size - 1}
        return set()
    if side == "left":
        if other == "top":
            return {0}
        if other == "bottom":
            return {0}
        return set()
    if side == "right":
        if other == "top":
            return {size - 1}
        if other == "bottom":
            return {size - 1}
        return set()
    raise ValueError(f"unknown side: {side}")


def _extract_seed(grid: np.ndarray) -> tuple[str, list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"

    bg = _most_common_color(g)
    layer_count = min(h, w) // 2

    counts = {}
    for side in SIDE_CHOICES:
        edge = _edge_values(g, side)
        counts[side] = int(np.count_nonzero(edge != bg))

    side = max(SIDE_CHOICES, key=lambda name: (counts[name], -SIDE_CHOICES.index(name)))
    assert counts[side] == layer_count, "seed edge must contain one color per layer"

    for other in SIDE_CHOICES:
        if other == side:
            continue
        edge = _edge_values(g, other)
        nz = set(map(int, np.flatnonzero(edge != bg).tolist()))
        allowed = _shared_corner_indices(side, other, h)
        assert nz.issubset(allowed), "non-seed edges may only contain shared-corner spillover"

    seed_edge = _edge_values(g, side)
    colors = [int(v) for v in seed_edge.tolist() if int(v) != bg]
    assert len(colors) == layer_count
    return side, colors, bg


def _paint_layer(out: np.ndarray, depth: int, color: int) -> None:
    h, w = out.shape
    r0 = depth
    c0 = depth
    r1 = h - depth - 1
    c1 = w - depth - 1
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _render_output(shape: tuple[int, int], bg: int, colors: list[int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    for depth, color in enumerate(colors):
        _paint_layer(out, depth, int(color))
    if colors and h - 2 * len(colors) == 1 and w - 2 * len(colors) == 1:
        out[len(colors), len(colors)] = int(colors[-1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a border-side color legend into nested shrinking rectangular frames.

    Input structure:
    - The grid is an odd-sized square with one dominant background color.
    - Exactly one outer edge contains a sparse ordered seed sequence of non-background pixels.
    - The number of seed pixels equals the number of full border layers that fit from outside to inside.

    Rule:
    - Detect the unique seed edge and read its non-background colors in edge order.
    - Use the first seed color for the outermost 1-pixel rectangular frame, the second for the next inner frame, and so on.
    - After all frames are drawn, fill the remaining center cell, if any, with the last seed color.

    Key insight: The colored edge is not part of the final picture; it is an ordered legend that specifies the frame colors from outermost to innermost.
    """
    g = np.asarray(grid, dtype=int)
    _side, colors, bg = _extract_seed(g)
    return _render_output(g.shape, bg, colors)


def _place_seed_pixels(grid: np.ndarray, side: str, positions: list[int], colors: list[int]) -> None:
    if side == "top":
        grid[0, positions] = colors
    elif side == "bottom":
        grid[-1, positions] = colors
    elif side == "left":
        grid[positions, 0] = colors
    elif side == "right":
        grid[positions, -1] = colors
    else:
        raise ValueError(f"unknown side: {side}")


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(ODD_SIZES)
    layer_count = size // 2
    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    if len(palette) < layer_count:
        return None

    side = random.choice(SIDE_CHOICES)
    colors = random.sample(palette, layer_count)
    positions = sorted(random.sample(list(range(1, size - 1)), layer_count))

    grid = np.full((size, size), bg, dtype=int)
    _place_seed_pixels(grid, side, positions, colors)

    try:
        detected_side, detected_colors, detected_bg = _extract_seed(grid)
    except AssertionError:
        return None

    if detected_side != side:
        return None
    if detected_colors != colors:
        return None
    if detected_bg != bg:
        return None

    expected = _render_output(grid.shape, bg, colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
