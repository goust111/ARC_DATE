from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
START_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BLOCK = 3
GEN_RETRIES = 600


def _block_origin(r: int, c: int) -> tuple[int, int]:
    return (r // BLOCK) * BLOCK, (c // BLOCK) * BLOCK


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    start_coords = np.argwhere((grid != BLACK) & (grid != GREY))
    grey_coords = np.argwhere(grid == GREY)

    assert start_coords.shape[0] == 1, "expected exactly one colored start pixel"
    assert grey_coords.shape[0] == 1, "expected exactly one grey direction marker"

    sr, sc = map(int, start_coords[0])
    gr, gc = map(int, grey_coords[0])
    dr = gr - sr
    dc = gc - sc
    assert abs(dr) == 1 and abs(dc) == 1, "grey marker must be diagonally adjacent to the start"

    return (sr, sc), (dr, dc), int(grid[sr, sc])


def _simulate_bounces(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    bounces: list[tuple[int, int]] = []

    while True:
        nr = r + dr
        if nr < 0 or nr >= height:
            break

        nc = c + dc
        if nc < 0 or nc >= width:
            dc = -dc
            nc = c + dc
            assert 0 <= nc < width
            bounces.append((int(nr), int(nc)))

        r, c = nr, nc

    return bounces


def _render_blocks(shape: tuple[int, int], bounce_points: list[tuple[int, int]], color: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0 in {_block_origin(r, c) for r, c in bounce_points}:
        out[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light the 3x3 macro-blocks touched by the invisible diagonal ray's wall bounces.

    Input structure:
    - The grid has a black background and both dimensions are multiples of 3.
    - There is exactly one colored start pixel and exactly one grey pixel diagonally adjacent to it.
    - The grey pixel specifies the ray's first diagonal step; there are no other foreground pixels.

    Rule:
    - Start from the colored pixel and move one row and one column each step in the grey-indicated direction.
    - If the next step would leave through the left or right edge, reverse only the horizontal direction and use that reflected landing cell as a bounce point.
    - Stop once the next step would leave through the top or bottom edge; do not draw the path itself, and instead fill each bounce point's entire 3x3 macro-block with the start color.

    Key insight:
    - The visible output records only the ray's reflection residues, so solving the puzzle requires simulating the hidden bouncing path rather than extending a visible line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % BLOCK == 0 and width % BLOCK == 0, "grid dimensions must be multiples of 3"
    assert np.all((g == BLACK) | (g == GREY) | np.isin(g, START_COLORS)), "unexpected colors in the input"

    start, direction, color = _parse_scene(g)
    bounce_points = _simulate_bounces(g.shape, start, direction)
    return _render_blocks(g.shape, bounce_points, color)


def _scene_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    height, width = map(int, grid.shape)
    if height > 30 or width > 30 or height % BLOCK != 0 or width % BLOCK != 0:
        return False

    try:
        start, direction, color = _parse_scene(grid)
    except AssertionError:
        return False

    if color == GREY:
        return False

    sr, sc = start
    if sr in (0, height - 1) or sc in (0, width - 1):
        return False

    bounce_points = _simulate_bounces(grid.shape, start, direction)
    block_origins = {_block_origin(r, c) for r, c in bounce_points}
    if len(bounce_points) < 2 or not (2 <= len(block_origins) <= 6):
        return False

    start_block = _block_origin(sr, sc)
    grey_block = _block_origin(sr + direction[0], sc + direction[1])
    if start_block in block_origins or grey_block in block_origins:
        return False

    side_blocks = {c0 for _r0, c0 in block_origins}
    if side_blocks != {0, width - BLOCK}:
        return False

    out = _render_blocks(grid.shape, bounce_points, color)
    if int((out != BLACK).sum()) != 9 * len(block_origins):
        return False

    if out[sr, sc] != BLACK or out[sr + direction[0], sc + direction[1]] != BLACK:
        return False

    alternatives = []
    for alt_dr in (-1, 1):
        for alt_dc in (-1, 1):
            if (alt_dr, alt_dc) == direction:
                continue
            alt_out = _render_blocks(
                grid.shape,
                _simulate_bounces(grid.shape, start, (alt_dr, alt_dc)),
                color,
            )
            alternatives.append(alt_out)
    if any(np.array_equal(out, alt_out) for alt_out in alternatives):
        return False

    return np.array_equal(out, main(grid.copy()))


def _build_attempt() -> np.ndarray | None:
    macro_h = random.randint(5, 10)
    macro_w = random.randint(3, 8)
    height = macro_h * BLOCK
    width = macro_w * BLOCK

    dr = random.choice((-1, 1))
    dc = random.choice((-1, 1))

    if dr == -1:
        sr = random.randint(max(1, height // 2), height - 2)
    else:
        sr = random.randint(1, min(height - 2, (height - 1) // 2))

    if dc == 1:
        sc = random.randint(max(1, width // 2), width - 2)
    else:
        sc = random.randint(1, min(width - 2, (width - 1) // 2))

    color = random.choice(START_COLORS)
    grid = np.zeros((height, width), dtype=int)
    grid[sr, sc] = color
    grid[sr + dr, sc + dc] = GREY

    return grid if _scene_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-ray macro-block scene")
