from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))

GRID_H_RANGE = (8, 18)
GRID_W_RANGE = (8, 18)
VISIBLE_SIDE_MIN = 3
MIN_BG_CELLS = 8
GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one non-background cell"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _checker_colors(window: np.ndarray) -> tuple[int, int]:
    parity = np.indices(window.shape).sum(axis=0) % 2 == 0

    even_colors = np.unique(window[parity])
    odd_colors = np.unique(window[~parity])
    assert even_colors.size == 1 and odd_colors.size == 1, "checkerboard layers must be uniform"

    even_color = int(even_colors[0])
    odd_color = int(odd_colors[0])
    assert even_color != odd_color, "checkerboard must use exactly two alternating colors"
    return even_color, odd_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the majority color layer of a two-color checkerboard rectangle.

    Input structure:
    - Background is black.
    - The non-black cells form one solid axis-aligned rectangle.
    - Inside that rectangle, two colors alternate in a checkerboard pattern.

    Rule:
    - Count how many pixels belong to each of the two checkerboard colors.
    - Keep the color with the larger count.
    - Recolor every pixel of the smaller color to black.
    - Preserve the existing black background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g != BLACK
    r0, c0, r1, c1 = _bbox(mask)
    window = g[r0 : r1 + 1, c0 : c1 + 1]

    assert bool(np.all(window != BLACK)), "checkerboard must fill its bounding box"
    even_color, odd_color = _checker_colors(window)

    even_count = int(np.sum(window == even_color))
    odd_count = int(np.sum(window == odd_color))
    assert even_count != odd_count, "visible checkerboard must have a unique majority color"

    keep = even_color if even_count > odd_count else odd_color
    out = g.copy()
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[patch != keep] = BLACK
    return out


def _odd_sizes(limit: int) -> list[int]:
    return [size for size in range(VISIBLE_SIDE_MIN, limit + 1) if size % 2 == 1]


def _valid_shapes(height: int, width: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for vis_h in _odd_sizes(height):
        for vis_w in _odd_sizes(width):
            if height * width - vis_h * vis_w < MIN_BG_CELLS:
                continue
            shapes.append((vis_h, vis_w))
    return shapes


def _valid_positions(height: int, width: int, vis_h: int, vis_w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(height - vis_h + 1):
        for c0 in range(width - vis_w + 1):
            touches_edge = r0 == 0 or c0 == 0 or r0 + vis_h == height or c0 + vis_w == width
            if touches_edge:
                positions.append((r0, c0))
    return positions


def _render_checkerboard(
    height: int,
    width: int,
    r0: int,
    c0: int,
    vis_h: int,
    vis_w: int,
    major_color: int,
    minor_color: int,
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    parity = np.indices((vis_h, vis_w)).sum(axis=0) % 2 == 0
    patch = np.where(parity, major_color, minor_color)
    grid[r0 : r0 + vis_h, c0 : c0 + vis_w] = patch
    return grid


def _expected_output(
    grid: np.ndarray,
    r0: int,
    c0: int,
    vis_h: int,
    vis_w: int,
    keep_color: int,
) -> np.ndarray:
    out = grid.copy()
    patch = out[r0 : r0 + vis_h, c0 : c0 + vis_w]
    patch[patch != keep_color] = BLACK
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    shapes = _valid_shapes(height, width)
    if not shapes:
        return None
    vis_h, vis_w = random.choice(shapes)

    positions = _valid_positions(height, width, vis_h, vis_w)
    if not positions:
        return None
    r0, c0 = random.choice(positions)

    color_a, color_b = random.sample(COLOR_POOL, 2)
    major_color, minor_color = (color_a, color_b) if random.random() < 0.5 else (color_b, color_a)

    grid = _render_checkerboard(height, width, r0, c0, vis_h, vis_w, major_color, minor_color)
    expected = _expected_output(grid, r0, c0, vis_h, vis_w, major_color)

    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid checkerboard sample after {GENERATION_ATTEMPTS} attempts")
