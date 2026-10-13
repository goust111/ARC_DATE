from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MIN_LEGEND_LEN = 3
MAX_LEGEND_LEN = 5
GENERATION_RETRIES = 300
EXTRA_MARGIN_MAX = 4
CLIPPED_DIM_MIN = 4
CORNERS = ("tl", "tr", "bl", "br")
LEGEND_KINDS = ("row", "col")


def _corner_seed_slices(shape: tuple[int, int], corner: str) -> tuple[slice, slice]:
    h, w = shape
    if corner == "tl":
        return slice(0, 2), slice(0, 2)
    if corner == "tr":
        return slice(0, 2), slice(w - 2, w)
    if corner == "bl":
        return slice(h - 2, h), slice(0, 2)
    if corner == "br":
        return slice(h - 2, h), slice(w - 2, w)
    raise ValueError(f"unknown corner: {corner}")


def _legend_spec(shape: tuple[int, int], corner: str, kind: str) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    if corner == "tl" and kind == "row":
        return (0, 2), (0, 1)
    if corner == "tl" and kind == "col":
        return (2, 0), (1, 0)
    if corner == "tr" and kind == "row":
        return (0, w - 3), (0, -1)
    if corner == "tr" and kind == "col":
        return (2, w - 1), (1, 0)
    if corner == "bl" and kind == "row":
        return (h - 1, 2), (0, 1)
    if corner == "bl" and kind == "col":
        return (h - 3, 0), (-1, 0)
    if corner == "br" and kind == "row":
        return (h - 1, w - 3), (0, -1)
    if corner == "br" and kind == "col":
        return (h - 3, w - 1), (-1, 0)
    raise ValueError(f"invalid corner/kind pair: {corner}/{kind}")


def _read_legend_run(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> list[int]:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    colors: list[int] = []
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == BACKGROUND:
            break
        colors.append(value)
        r += dr
        c += dc
    return colors


def _anchored_rect_mask(shape: tuple[int, int], corner: str, size: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)

    if corner[0] == "t":
        r0, r1 = 0, min(h, size)
    else:
        r0, r1 = max(0, h - size), h

    if corner[1] == "l":
        c0, c1 = 0, min(w, size)
    else:
        c0, c1 = max(0, w - size), w

    if r0 < r1 and c0 < c1:
        mask[r0:r1, c0:c1] = True
    return mask


def _render_output(
    shape: tuple[int, int], corner: str, seed_color: int, legend_colors: list[int]
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    previous = _anchored_rect_mask(shape, corner, 2)
    out[previous] = seed_color

    for layer_index, color in enumerate(legend_colors, start=1):
        current = _anchored_rect_mask(shape, corner, 2 + layer_index)
        out[current & ~previous] = color
        previous = current
    return out


def _find_seed_corner(grid: np.ndarray) -> tuple[str, int]:
    matches: list[tuple[str, int]] = []
    for corner in CORNERS:
        rs, cs = _corner_seed_slices(grid.shape, corner)
        block = grid[rs, cs]
        if block.shape != (2, 2):
            continue
        values = np.unique(block)
        if values.size == 1 and int(values[0]) != BACKGROUND:
            matches.append((corner, int(values[0])))

    assert len(matches) == 1, "expected exactly one monochrome 2x2 corner seed"
    return matches[0]


def _read_legend(grid: np.ndarray, corner: str, seed_color: int) -> list[int]:
    candidates: list[list[int]] = []
    for kind in LEGEND_KINDS:
        start, step = _legend_spec(grid.shape, corner, kind)
        sequence = _read_legend_run(grid, start, step)
        if sequence:
            candidates.append(sequence)

    assert len(candidates) == 1, "expected exactly one adjacent edge legend"
    legend = candidates[0]
    assert MIN_LEGEND_LEN <= len(legend) <= MAX_LEGEND_LEN, "legend length must be 3-5"
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    assert seed_color not in legend, "seed and legend colors must be distinct"
    return legend


def _render_input(
    shape: tuple[int, int],
    corner: str,
    legend_kind: str,
    seed_color: int,
    legend_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)

    rs, cs = _corner_seed_slices(shape, corner)
    grid[rs, cs] = seed_color

    start, step = _legend_spec(shape, corner, legend_kind)
    r, c = start
    dr, dc = step
    for color in legend_colors:
        grid[r, c] = color
        r += dr
        c += dc

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one-way concentric rectangular rings from a corner seed using the adjacent edge legend.

    Input structure:
    - Background is black.
    - Exactly one corner contains a monochrome 2x2 seed block.
    - One adjacent edge contains a contiguous 3-5 color legend starting just beyond the seed.

    Rule:
    - Keep the 2x2 seed unchanged.
    - Read the legend away from the seed corner.
    - Layer k expands the corner-anchored rectangle by one cell in each growing direction and colors only the newly
      visible clipped ring cells with the k-th legend color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert g.shape[0] >= CLIPPED_DIM_MIN and g.shape[1] >= CLIPPED_DIM_MIN, "grid is too small"

    corner, seed_color = _find_seed_corner(g)
    legend_colors = _read_legend(g, corner, seed_color)
    return _render_output(g.shape, corner, seed_color, legend_colors)


def _sample_shape(legend_len: int, legend_kind: str, clipped: bool) -> tuple[int, int]:
    needed = 2 + legend_len
    if legend_kind == "row":
        width = needed + random.randint(0, EXTRA_MARGIN_MAX)
        if clipped:
            height = random.randint(CLIPPED_DIM_MIN, needed - 1)
        else:
            height = needed + random.randint(0, EXTRA_MARGIN_MAX)
    else:
        height = needed + random.randint(0, EXTRA_MARGIN_MAX)
        if clipped:
            width = random.randint(CLIPPED_DIM_MIN, needed - 1)
        else:
            width = needed + random.randint(0, EXTRA_MARGIN_MAX)
    return height, width


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(MIN_LEGEND_LEN, MAX_LEGEND_LEN)
    corner = random.choice(CORNERS)
    legend_kind = random.choice(LEGEND_KINDS)
    clipped = random.random() < 0.5
    shape = _sample_shape(legend_len, legend_kind, clipped)

    colors = random.sample(PALETTE, legend_len + 1)
    seed_color = colors[0]
    legend_colors = colors[1:]

    grid = _render_input(shape, corner, legend_kind, seed_color, legend_colors)
    expected = _render_output(shape, corner, seed_color, legend_colors)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
