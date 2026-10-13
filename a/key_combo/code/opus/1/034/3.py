import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4

_GEN_TILE_BAG: list[np.ndarray] = []


def _plus_centers(grid: np.ndarray) -> np.ndarray:
    centers = np.zeros(grid.shape, dtype=bool)
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return centers

    inner = (
        (grid[1:-1, 1:-1] == YELLOW)
        & (grid[:-2, 1:-1] == YELLOW)
        & (grid[2:, 1:-1] == YELLOW)
        & (grid[1:-1, :-2] == YELLOW)
        & (grid[1:-1, 2:] == YELLOW)
    )
    centers[1:-1, 1:-1] = inner
    return centers


def _plus_union_mask(grid: np.ndarray) -> np.ndarray:
    centers = _plus_centers(grid)
    union = centers.copy()
    union[:-1, :] |= centers[1:, :]
    union[1:, :] |= centers[:-1, :]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]
    return union


def _recolor_pluses(tiled: np.ndarray) -> np.ndarray:
    out = tiled.copy()
    plus_pixels = _plus_union_mask(out)
    out[plus_pixels] = GREEN
    return out


def _fill_rows_with_remaining_yellow(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    yellow_rows = np.any(out == YELLOW, axis=1)
    if np.any(yellow_rows):
        fill_mask = yellow_rows[:, None] & (out == BLACK)
        out[fill_mask] = BLUE
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    tiled = np.tile(grid, (3, 3))
    recolored = _recolor_pluses(tiled)
    return _fill_rows_with_remaining_yellow(recolored)


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the small yellow pattern, recolor boundary-emergent pluses green, then blue-fill only rows that still retain yellow.

    Input structure:
    - The input is a 4x4 or 5x5 black grid with one foreground color: yellow single-cell pixels.
    - Blue and green do not appear in the input; they are reserved for the output.
    - The yellow placement is chosen so that no full orthogonal yellow plus exists inside the original tile.

    Rule:
    - Repeat the tile three times horizontally and three times vertically.
    - In the tiled canvas, every yellow cell whose up, down, left, and right neighbors are all yellow defines a plus; recolor the center and those four orthogonal neighbors green.
    - After all plus recoloring is merged, scan each row: if that row still contains at least one yellow cell, fill every remaining black cell in that row blue.
    - Green and yellow are never overwritten by blue.

    Key insight:
    - The decisive pluses do not exist in one tile; they emerge only across tile boundaries, and that green recoloring can remove the last yellow from a row before row filling is decided.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _base_cases() -> list[tuple[int, int, int]]:
    cases: list[tuple[int, int, int]] = []
    for size in (4, 5):
        for row in (0, size - 1):
            for col in range(1, size - 1):
                cases.append((size, row, col))
    return cases


def _candidate_tiles() -> list[np.ndarray]:
    tiles: list[np.ndarray] = []
    for size, center_row, center_col in _base_cases():
        base = _build_case_grid(size, center_row, center_col)
        available = [
            (row, col)
            for row in range(size)
            for col in range(size)
            if base[row, col] == BLACK
        ]
        for extra_count in (0, 1, 2):
            for extras in itertools.combinations(available, extra_count):
                tile = base.copy()
                for row, col in extras:
                    tile[row, col] = YELLOW
                out = _solve(tile)
                if _semantic_checks(tile, out):
                    tiles.append(tile)
    return tiles


def _refill_tile_bag() -> None:
    global _GEN_TILE_BAG
    _GEN_TILE_BAG = _candidate_tiles()
    random.shuffle(_GEN_TILE_BAG)


def _take_tile() -> np.ndarray:
    if not _GEN_TILE_BAG:
        _refill_tile_bag()
    return _GEN_TILE_BAG.pop().copy()


def _case_positions(size: int, center_row: int, center_col: int) -> set[tuple[int, int]]:
    positions = {
        (center_row, center_col),
        ((center_row - 1) % size, center_col),
        ((center_row + 1) % size, center_col),
        (center_row, (center_col - 1) % size),
        (center_row, (center_col + 1) % size),
    }
    assert len(positions) == 5
    return positions


def _build_case_grid(size: int, center_row: int, center_col: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for row, col in _case_positions(size, center_row, center_col):
        grid[row, col] = YELLOW
    return grid


def _rows_with_color(grid: np.ndarray, color: int) -> np.ndarray:
    return np.any(grid == color, axis=1)


def _has_internal_plus(tile: np.ndarray) -> bool:
    return bool(np.any(_plus_centers(tile)))


def _semantic_checks(tile: np.ndarray, out: np.ndarray) -> bool:
    if tile.shape not in ((4, 4), (5, 5)):
        return False
    if not set(map(int, np.unique(tile))).issubset({BLACK, YELLOW}):
        return False

    # A boundary-emergent orthogonal plus on a 4x4/5x5 torus needs five distinct
    # yellow residues, so the textual "2~4 pixels" bound is infeasible here.
    # The generator therefore uses the minimum five, plus at most two distractors.
    yellow_count = int(np.count_nonzero(tile == YELLOW))
    if not (5 <= yellow_count <= 7):
        return False
    if _has_internal_plus(tile):
        return False

    tiled = np.tile(tile, (3, 3))
    centers = _plus_centers(tiled)
    if not np.any(centers):
        return False

    recolored = _recolor_pluses(tiled)
    if int(np.count_nonzero(recolored == GREEN)) == 0:
        return False

    before_yellow_rows = _rows_with_color(tiled, YELLOW)
    after_yellow_rows = _rows_with_color(recolored, YELLOW)
    blue_rows = _rows_with_color(out, BLUE)

    lost_rows = before_yellow_rows & (~after_yellow_rows)
    if not np.any(lost_rows & (~blue_rows)):
        return False
    if int(np.count_nonzero(out == BLUE)) == 0:
        return False
    if int(np.count_nonzero(out == YELLOW)) == 0:
        return False
    if int(np.count_nonzero(out == GREEN)) == 0:
        return False

    expected = _fill_rows_with_remaining_yellow(recolored)
    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(200):
        tile = _take_tile()
        out = _solve(tile)
        if _semantic_checks(tile, out):
            return tile
    raise RuntimeError("failed to generate a valid boundary-emergent plus instance")
