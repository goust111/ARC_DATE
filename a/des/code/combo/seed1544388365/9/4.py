from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
SEPARATOR_COLOR = 5
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

TILE_SIZE_RANGE = (4, 6)
GRID_SIDE_LIMIT = 30
GENERATION_ATTEMPTS = 400
PATCH_ATTEMPTS = 120
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _parse_separator_positions(positions: np.ndarray, length: int) -> tuple[int, list[int]] | None:
    pos = [int(x) for x in positions.tolist()]
    if not pos:
        return None
    if pos[0] <= 0:
        return None
    if any(pos[i] >= pos[i + 1] for i in range(len(pos) - 1)):
        return None

    if len(pos) == 1:
        tile_size = pos[0]
    else:
        step = pos[1] - pos[0]
        if step <= 1 or any(pos[i + 1] - pos[i] != step for i in range(len(pos) - 1)):
            return None
        tile_size = step - 1

    if tile_size <= 0:
        return None
    if pos[0] != tile_size:
        return None
    if any(p != (index + 1) * tile_size + index for index, p in enumerate(pos)):
        return None
    if length - pos[-1] - 1 != tile_size:
        return None

    starts = [index * (tile_size + 1) for index in range(len(pos) + 1)]
    if starts[-1] + tile_size != length:
        return None
    return tile_size, starts


def _find_layout(grid: np.ndarray, bg: int) -> tuple[int, int, list[int], int, list[int]]:
    candidates: list[tuple[int, int, list[int], int, list[int]]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue

        row_positions = np.flatnonzero(np.all(grid == color, axis=1))
        col_positions = np.flatnonzero(np.all(grid == color, axis=0))
        row_layout = _parse_separator_positions(row_positions, grid.shape[0])
        col_layout = _parse_separator_positions(col_positions, grid.shape[1])
        if row_layout is None or col_layout is None:
            continue

        tile_h, row_starts = row_layout
        tile_w, col_starts = col_layout
        candidates.append((color, tile_h, row_starts, tile_w, col_starts))

    assert len(candidates) == 1, "expected one regular separator lattice"
    return candidates[0]


def _rotate_tile(tile: np.ndarray, bg: int, turns: int) -> np.ndarray:
    k = turns % 4
    if k == 0:
        return tile.copy()

    if tile.shape[0] == tile.shape[1]:
        return np.rot90(tile, -k).copy()

    cy = (tile.shape[0] - 1) / 2.0
    cx = (tile.shape[1] - 1) / 2.0
    out = np.full_like(tile, bg)
    for r_raw, c_raw in np.argwhere(tile != bg):
        y = float(int(r_raw) - cy)
        x = float(int(c_raw) - cx)
        for _ in range(k):
            y, x = -x, y
        nr = int(round(y + cy))
        nc = int(round(x + cx))
        assert 0 <= nr < tile.shape[0] and 0 <= nc < tile.shape[1], "rotated cell left its tile"
        assert int(out[nr, nc]) == bg, "rotation must be collision free"
        out[nr, nc] = int(tile[int(r_raw), int(c_raw)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each tile's foreground pattern clockwise by (row + col) mod 4.

    The grid is partitioned by a one-cell separator lattice into equal tiles.
    Every input tile contains the same foreground pattern in the same orientation.
    In the output, the tile at row r and column c is rotated clockwise by
    ((r + c) mod 4) quarter-turns; separators and background stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    _separator_color, tile_h, row_starts, tile_w, col_starts = _find_layout(g, bg)

    out = g.copy()
    for tile_r, r0 in enumerate(row_starts):
        for tile_c, c0 in enumerate(col_starts):
            tile = g[r0 : r0 + tile_h, c0 : c0 + tile_w]
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = _rotate_tile(tile, bg, tile_r + tile_c)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_rotationally_asymmetric(mask: np.ndarray) -> bool:
    cropped = _crop_mask(mask)
    return all(
        cropped.shape != np.rot90(cropped, -turns).shape
        or not np.array_equal(cropped, np.rot90(cropped, -turns))
        for turns in (1, 2, 3)
    )


def _grow_patch(height: int, width: int, area: int) -> np.ndarray | None:
    if area <= 0 or area > height * width:
        return None

    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    cells = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in DIRS4
        if 0 <= seed[0] + dr < height and 0 <= seed[1] + dc < width
    }

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        cells.add(cell)
        for dr, dc in DIRS4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    return mask if len(cells) == area else None


def _sample_pattern(tile_size: int, fg_color: int) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        patch_h = random.randint(2, min(4, tile_size))
        patch_w = random.randint(2, min(4, tile_size))
        max_area = min(7, patch_h * patch_w - 1)
        if max_area < 4:
            continue

        patch = _grow_patch(patch_h, patch_w, random.randint(4, max_area))
        if patch is None or not _is_rotationally_asymmetric(patch):
            continue

        patch = _crop_mask(patch)
        pr, pc = patch.shape
        tile = np.full((tile_size, tile_size), BG_COLOR, dtype=int)
        r0 = random.randint(0, tile_size - pr)
        c0 = random.randint(0, tile_size - pc)
        tile[r0 : r0 + pr, c0 : c0 + pc][patch] = fg_color

        if _is_rotationally_asymmetric(tile):
            return tile
    return None


def _render_grid(rows: int, cols: int, tile: np.ndarray) -> np.ndarray:
    tile_size = tile.shape[0]
    height = rows * tile_size + (rows - 1)
    width = cols * tile_size + (cols - 1)
    grid = np.full((height, width), BG_COLOR, dtype=int)

    for r in range(1, rows):
        grid[r * tile_size + (r - 1), :] = SEPARATOR_COLOR
    for c in range(1, cols):
        grid[:, c * tile_size + (c - 1)] = SEPARATOR_COLOR

    for tile_r in range(rows):
        for tile_c in range(cols):
            r0 = tile_r * (tile_size + 1)
            c0 = tile_c * (tile_size + 1)
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = tile

    return grid


def _expected_output(rows: int, cols: int, tile: np.ndarray) -> np.ndarray:
    tile_size = tile.shape[0]
    out = _render_grid(rows, cols, tile)
    for tile_r in range(rows):
        for tile_c in range(cols):
            r0 = tile_r * (tile_size + 1)
            c0 = tile_c * (tile_size + 1)
            out[r0 : r0 + tile_size, c0 : c0 + tile_size] = np.rot90(tile, -((tile_r + tile_c) % 4))
    return out


def _sample_grid_shape(tile_size: int) -> tuple[int, int]:
    max_tiles = min(5, (GRID_SIDE_LIMIT + 1) // (tile_size + 1))
    assert max_tiles >= 3
    choices = [
        (rows, cols)
        for rows in range(2, max_tiles + 1)
        for cols in range(2, max_tiles + 1)
        if rows + cols >= 5
    ]
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.randint(*TILE_SIZE_RANGE)
    rows, cols = _sample_grid_shape(tile_size)
    fg_color = random.choice(FG_COLORS)
    tile = _sample_pattern(tile_size, fg_color)
    if tile is None:
        return None

    inp = _render_grid(rows, cols, tile)
    expected = _expected_output(rows, cols, tile)
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating-tile input")
