from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _row_major_color(row: np.ndarray) -> int:
    counts = np.bincount(np.asarray(row, dtype=int), minlength=10)
    best = int(counts.max())
    winners = np.flatnonzero(counts == best)
    assert len(winners) == 1, "each upper row must have a unique majority color"
    return int(winners[0])


def _compress_runs(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(int(value))
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(rows) == 1, "expected exactly one full grey separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split upper and lower zones"
    return sep


def _extract_tile(lower: np.ndarray, allowed_colors: set[int]) -> np.ndarray:
    mask = lower != BLACK
    box = _bbox(mask)
    assert box is not None, "expected one square tile below the separator"
    tile = _crop_bbox(lower, box)
    h, w = tile.shape
    assert h == w, "tile must be square"
    assert 2 <= h <= 4, "tile size must stay within the puzzle bounds"
    colors = set(map(int, np.unique(tile)))
    assert colors <= allowed_colors | {BLACK}, "tile colors must come from upper bands plus black"
    assert np.any(tile != BLACK), "tile must contain at least one non-black pixel"
    return tile


def _is_asymmetric(tile: np.ndarray) -> bool:
    for k in range(1, 4):
        if np.array_equal(tile, np.rot90(tile, k)):
            return False
    return not np.array_equal(tile, np.fliplr(tile))


def _describe(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    upper = g[:sep]
    lower = g[sep + 1 :]
    assert upper.shape[0] > 0 and lower.shape[0] > 0
    assert not np.any(upper == BLACK), "upper zone must be fully covered by colored bands"
    assert not np.any(upper == GREY), "grey may only appear on the separator row"

    row_majorities = [_row_major_color(row) for row in upper]
    bands = _compress_runs(row_majorities)
    assert 3 <= len(bands) <= 5, "expected 3 to 5 compressed bands"
    assert len(set(bands)) == len(bands), "each band color must appear only once top-to-bottom"

    tile = _extract_tile(lower, set(bands))
    return upper, row_majorities, bands, tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the noisy upper color bands and use their count as the square tiling multiplier for the lower tile.

    Input structure:
    - One full grey row splits the grid into an upper band zone and a lower tile zone.
    - The upper zone is completely filled by 3 to 5 horizontal color bands with no black background, but some rows
      contain a small minority intrusion from an adjacent band while keeping a unique row-majority color.
    - The lower zone contains one square tile made from a subset of the band colors and black.

    Rule:
    - For each upper-zone row, take the row-majority color and compress consecutive equal rows into a band sequence.
    - Let N be the length of that compressed sequence.
    - Extract the lower square tile.
    - Repeat the tile N times horizontally and N times vertically without any rotation or mirroring.

    Key insight:
    - The upper region is not copied; its only job is to encode the band count through row-majority compression, and
      that count alone determines the tiling factor for the lower tile.
    """
    _upper, _row_majorities, bands, tile = _describe(grid)
    return np.tile(tile, (len(bands), len(bands)))


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _make_upper(width: int, band_colors: list[int], band_lengths: list[int]) -> np.ndarray:
    row_colors = [color for color, length in zip(band_colors, band_lengths) for _ in range(length)]
    upper = np.array([np.full(width, color, dtype=int) for color in row_colors], dtype=int)

    boundary_rows = [
        r
        for r, color in enumerate(row_colors)
        if (r > 0 and row_colors[r - 1] != color) or (r + 1 < len(row_colors) and row_colors[r + 1] != color)
    ]
    max_intrusion = max(1, width // 3)
    noisy_rows = 0
    for r in boundary_rows:
        if random.random() < 0.45:
            continue
        color = row_colors[r]
        neighbors = []
        if r > 0 and row_colors[r - 1] != color:
            neighbors.append(row_colors[r - 1])
        if r + 1 < len(row_colors) and row_colors[r + 1] != color:
            neighbors.append(row_colors[r + 1])
        neighbor = random.choice(neighbors)
        intrusion = random.randint(1, max_intrusion)
        if random.random() < 0.5:
            upper[r, :intrusion] = neighbor
        else:
            upper[r, width - intrusion :] = neighbor
        noisy_rows += 1

    if noisy_rows == 0:
        r = random.choice(boundary_rows)
        color = row_colors[r]
        neighbors = []
        if r > 0 and row_colors[r - 1] != color:
            neighbors.append(row_colors[r - 1])
        if r + 1 < len(row_colors) and row_colors[r + 1] != color:
            neighbors.append(row_colors[r + 1])
        neighbor = random.choice(neighbors)
        intrusion = random.randint(1, max_intrusion)
        upper[r, :intrusion] = neighbor

    return upper


def _make_tile(size: int, band_colors: list[int]) -> np.ndarray | None:
    color_count = random.randint(1, min(3, len(band_colors)))
    tile_colors = random.sample(band_colors, color_count)

    for _ in range(120):
        tile = np.zeros((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                tile[r, c] = BLACK if random.random() < 0.35 else random.choice(tile_colors)

        for r, c in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)):
            tile[r, c] = random.choice(tile_colors)

        for r in range(size):
            if not np.any(tile[r] != BLACK):
                tile[r, random.randrange(size)] = random.choice(tile_colors)
        for c in range(size):
            if not np.any(tile[:, c] != BLACK):
                tile[random.randrange(size), c] = random.choice(tile_colors)

        if not np.any(tile == BLACK):
            rr = random.randrange(size)
            cc = random.randrange(size)
            if (rr, cc) in {(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)}:
                rr = min(1, size - 1)
                cc = min(1, size - 1)
            tile[rr, cc] = BLACK
            if not np.any(tile[rr] != BLACK):
                tile[rr, (cc + 1) % size] = random.choice(tile_colors)
            if not np.any(tile[:, cc] != BLACK):
                tile[(rr + 1) % size, cc] = random.choice(tile_colors)

        if len(set(map(int, np.unique(tile))) - {BLACK}) == 0:
            continue
        if len(set(map(int, np.unique(tile)))) < 2:
            continue
        if not _is_asymmetric(tile):
            continue
        return tile
    return None


def _assemble_input(upper: np.ndarray, tile: np.ndarray) -> np.ndarray:
    upper_h, width = upper.shape
    size = tile.shape[0]
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    left = random.randint(0, width - size)
    grid = np.zeros((upper_h + 1 + top_pad + size + bottom_pad, width), dtype=int)
    grid[:upper_h] = upper
    grid[upper_h, :] = GREY
    grid[upper_h + 1 + top_pad : upper_h + 1 + top_pad + size, left : left + size] = tile
    return grid


def _semantic_ok(inp: np.ndarray) -> bool:
    upper, row_majorities, bands, tile = _describe(inp)
    band_runs = _compress_runs(row_majorities)
    if band_runs != bands:
        return False
    if upper.shape[0] <= len(bands):
        return False
    if not any(np.any(upper[r] != row_majorities[r]) for r in range(upper.shape[0])):
        return False
    if not _is_asymmetric(tile):
        return False

    expected = np.tile(tile, (len(bands), len(bands)))
    if not np.array_equal(main(inp), expected):
        return False

    if expected.shape == np.tile(tile, (upper.shape[0], upper.shape[0])).shape:
        return False
    if np.array_equal(expected, np.tile(np.fliplr(tile), (len(bands), len(bands)))):
        return False
    for k in range(1, 4):
        if np.array_equal(expected, np.tile(np.rot90(tile, k), (len(bands), len(bands)))):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    band_count = random.randint(3, 5)
    tile_size = random.randint(2, 4)
    if band_count * tile_size > 30:
        return None

    width = random.randint(max(8, tile_size + 2), 12)
    total_rows = random.randint(band_count + 1, min(9, band_count * 3))
    band_lengths = _random_composition(total_rows, band_count)
    if max(band_lengths) == 1:
        return None

    band_colors = random.sample(PALETTE, band_count)
    upper = _make_upper(width, band_colors, band_lengths)
    tile = _make_tile(tile_size, band_colors)
    if tile is None:
        return None

    inp = _assemble_input(upper, tile)
    if max(inp.shape) > 30:
        return None
    if not _semantic_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
