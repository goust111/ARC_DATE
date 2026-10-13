from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 7)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

TILE_SIZE_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _split_intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, length]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = {end - start for start, end in intervals}
    assert len(intervals) == 3 and len(sizes) == 1 and min(sizes) > 0
    return intervals


def _grid_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(row_separators) == 2 and len(col_separators) == 2
    return _split_intervals(grid.shape[0], row_separators), _split_intervals(grid.shape[1], col_separators)


def _tile_color(tile: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(tile) if int(c) != BACKGROUND]
    assert len(colors) == 1
    return colors[0]


def _is_diag_symmetric(tile: np.ndarray) -> bool:
    return bool(np.array_equal(tile, tile.T))


def _enclosed_background(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    is_bg = tile == BACKGROUND
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if is_bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if is_bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and is_bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return is_bg & ~seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the non-symmetric tiles' enclosed interiors, using the contour color borrowed from the symmetric tile in the same row.

    Input structure:
    - A black background grid is partitioned by full grey separator rows and columns into a 3x3 array of equal square tiles.
    - Each tile contains a single-color 4-connected closed contour on black, and its interior includes at least one black cell.
    - In each tile row, exactly one tile is symmetric across the main diagonal and the other two are not.

    Rule:
    - Split the grid at the grey separator lines into 9 square tiles.
    - For each tile, test main-diagonal symmetry by comparing the tile with its transpose.
    - Leave the symmetric tile unchanged; for each non-symmetric tile in that row, flood from the tile border through black cells and recolor only the black cells that remain enclosed.
    - The fill color is the contour color of that row's unique symmetric tile, while contour pixels and grey separators stay unchanged.

    Key insight:
    - Symmetry is both the selection gate and the color lookup key: the only symmetric tile in a row is excluded from filling but supplies the color used to fill the other two tiles' enclosed interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_intervals, col_intervals = _grid_structure(g)

    out = g.copy()
    for row_index, (r0, r1) in enumerate(row_intervals):
        tiles = [g[r0:r1, c0:c1] for c0, c1 in col_intervals]
        assert all(tile.shape[0] == tile.shape[1] for tile in tiles)

        symmetric_cols = [j for j, tile in enumerate(tiles) if _is_diag_symmetric(tile)]
        assert len(symmetric_cols) == 1
        source_col = symmetric_cols[0]
        fill_color = _tile_color(tiles[source_col])

        for col_index, (c0, c1) in enumerate(col_intervals):
            if col_index == source_col:
                continue
            tile = g[r0:r1, c0:c1]
            assert not _is_diag_symmetric(tile)
            enclosed = _enclosed_background(tile)
            assert enclosed.any()
            out[r0:r1, c0:c1][enclosed] = fill_color

    return out


def _boundary_from_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    boundary = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        for dr, dc in DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if not _in_bounds((h, w), nr, nc) or not region[nr, nc]:
                boundary[int(r), int(c)] = True
                break
    return boundary


def _tile_from_region(region: np.ndarray, color: int) -> np.ndarray:
    boundary = _boundary_from_region(region)
    tile = np.zeros(region.shape, dtype=int)
    tile[boundary] = int(color)
    return tile


def _core_region_ok(region: np.ndarray) -> bool:
    boundary = _boundary_from_region(region)
    return bool(np.any(region & ~boundary))


def _add_one_sided_strip(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])

    candidates: list[np.ndarray] = []
    span_h = bottom - top + 1
    span_w = right - left + 1

    if right + 1 < w:
        start = random.randint(top, bottom - 1 if span_h >= 2 else bottom)
        end = random.randint(start + 1, bottom + 1)
        grown = region.copy()
        grown[start:end, right + 1] = True
        candidates.append(grown)
    if bottom + 1 < h:
        start = random.randint(left, right - 1 if span_w >= 2 else right)
        end = random.randint(start + 1, right + 1)
        grown = region.copy()
        grown[bottom + 1, start:end] = True
        candidates.append(grown)
    if left - 1 >= 0:
        start = random.randint(top, bottom - 1 if span_h >= 2 else bottom)
        end = random.randint(start + 1, bottom + 1)
        grown = region.copy()
        grown[start:end, left - 1] = True
        candidates.append(grown)
    if top - 1 >= 0:
        start = random.randint(left, right - 1 if span_w >= 2 else right)
        end = random.randint(start + 1, right + 1)
        grown = region.copy()
        grown[top - 1, start:end] = True
        candidates.append(grown)

    return random.choice(candidates) if candidates else region


def _make_symmetric_region(tile_size: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        side = random.randint(3, tile_size)
        top = random.randint(0, tile_size - side)
        left = top
        region = np.zeros((tile_size, tile_size), dtype=bool)
        region[top : top + side, left : left + side] = True

        if side >= 3 and top + side < tile_size and random.random() < 0.65:
            ext = random.randint(1, tile_size - (top + side))
            start = random.randint(0, side - 2)
            span = random.randint(2, side - start)
            region[top + start : top + start + span, left + side : left + side + ext] = True
            region[top + side : top + side + ext, left + start : left + start + span] = True

        if side >= 3 and top > 0 and random.random() < 0.35:
            ext = random.randint(1, top)
            start = random.randint(0, side - 2)
            span = random.randint(2, side - start)
            region[top + start : top + start + span, left - ext : left] = True
            region[top - ext : top, left + start : left + start + span] = True

        tile = _tile_from_region(region, 1)
        if _core_region_ok(region) and _is_diag_symmetric(tile):
            return region
    raise RuntimeError("failed to build a symmetric contour region")


def _make_asymmetric_region(tile_size: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(3, tile_size)
        width = random.randint(3, tile_size)
        top = random.randint(0, tile_size - height)
        left = random.randint(0, tile_size - width)
        region = np.zeros((tile_size, tile_size), dtype=bool)
        region[top : top + height, left : left + width] = True

        if random.random() < 0.7:
            region = _add_one_sided_strip(region)

        tile = _tile_from_region(region, 1)
        if _core_region_ok(region) and not _is_diag_symmetric(tile):
            return region
    raise RuntimeError("failed to build an asymmetric contour region")


def _make_tile(tile_size: int, color: int, symmetric: bool) -> np.ndarray:
    region = _make_symmetric_region(tile_size) if symmetric else _make_asymmetric_region(tile_size)
    tile = _tile_from_region(region, color)
    enclosed = _enclosed_background(tile)
    assert enclosed.any()
    assert _is_diag_symmetric(tile) == symmetric
    return tile


def _assemble_grid(tile_rows: list[list[np.ndarray]]) -> np.ndarray:
    tile_size = tile_rows[0][0].shape[0]
    side = 3 * tile_size + 2
    grid = np.zeros((side, side), dtype=int)
    for sep in (tile_size, 2 * tile_size + 1):
        grid[sep, :] = SEPARATOR
        grid[:, sep] = SEPARATOR

    for tr in range(3):
        for tc in range(3):
            r0 = tr * tile_size + tr
            c0 = tc * tile_size + tc
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = tile_rows[tr][tc]
    return grid


def _make_row(tile_size: int) -> list[np.ndarray]:
    symmetric_pos = random.randrange(3)
    symmetric_color = random.choice(PALETTE)
    nonsym_palette = [color for color in PALETTE if color != symmetric_color]

    if random.random() < 0.35:
        borrowed = random.choice(nonsym_palette)
        nonsym_colors = [borrowed, borrowed]
    else:
        nonsym_colors = random.sample(nonsym_palette, 2)

    row: list[np.ndarray] = []
    nonsym_index = 0
    for col in range(3):
        if col == symmetric_pos:
            row.append(_make_tile(tile_size, symmetric_color, symmetric=True))
        else:
            row.append(_make_tile(tile_size, nonsym_colors[nonsym_index], symmetric=False))
            nonsym_index += 1
    return row


def _generation_attempt() -> np.ndarray | None:
    tile_size = random.randint(*TILE_SIZE_RANGE)
    rows = [_make_row(tile_size) for _ in range(3)]
    grid = _assemble_grid(rows)
    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
