from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
ORANGE = 7
TEAL = 8
TILE_SIZE = 3
STRIDE = 4
MAX_GENERATION_ATTEMPTS = 300


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def _tile_patterns() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for bits in range(1 << (TILE_SIZE * TILE_SIZE)):
        mask = np.array([(bits >> idx) & 1 for idx in range(TILE_SIZE * TILE_SIZE)], dtype=int)
        mask = mask.reshape((TILE_SIZE, TILE_SIZE)).astype(bool)
        teal_count = int(mask.sum())
        if teal_count < 2 or teal_count > 7:
            continue
        if any(int(mask[r].sum()) in (0, TILE_SIZE) for r in range(TILE_SIZE)):
            continue
        if any(int(mask[:, c].sum()) in (0, TILE_SIZE) for c in range(TILE_SIZE)):
            continue
        patterns.append(np.where(mask, TEAL, BG).astype(int))
    return patterns


PATTERNS = _tile_patterns()


def _infer_tile_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    teal_rows = [r for r in range(grid.shape[0]) if np.any(grid[r, :] == TEAL)]
    teal_cols = [c for c in range(grid.shape[1]) if np.any(grid[:, c] == TEAL)]
    row_bands = _runs(teal_rows)
    col_bands = _runs(teal_cols)
    assert row_bands and col_bands, "expected at least one tile band"
    assert all(r1 - r0 + 1 == TILE_SIZE for r0, r1 in row_bands), "expected 3-row tile bands"
    assert all(c1 - c0 + 1 == TILE_SIZE for c0, c1 in col_bands), "expected 3-col tile bands"
    assert all(row_bands[i + 1][0] - row_bands[i][1] == 2 for i in range(len(row_bands) - 1))
    assert all(col_bands[i + 1][0] - col_bands[i][1] == 2 for i in range(len(col_bands) - 1))
    return row_bands, col_bands


def _separator_positions(bands: list[tuple[int, int]], limit: int) -> list[int]:
    separators = [bands[0][0] - 1] + [end + 1 for _, end in bands]
    assert separators[0] >= 0 and separators[-1] < limit
    assert all(separators[i + 1] - separators[i] == STRIDE for i in range(len(separators) - 1))
    return separators


def _marker_lines(grid: np.ndarray) -> tuple[set[int], set[int]]:
    h, w = grid.shape
    horizontal_rows: set[int] = set()
    vertical_cols: set[int] = set()
    for r, c in np.argwhere(grid == ORANGE):
        r = int(r)
        c = int(c)
        on_top_bottom = r in (0, h - 1)
        on_left_right = c in (0, w - 1)
        assert on_top_bottom or on_left_right, "markers must lie on the boundary"
        if on_top_bottom:
            vertical_cols.add(c)
        if on_left_right:
            horizontal_rows.add(r)
    return horizontal_rows, vertical_cols


def _draw_full_span_lines(grid: np.ndarray, horizontal_rows: set[int], vertical_cols: set[int]) -> np.ndarray:
    out = grid.copy()
    for r in horizontal_rows:
        out[int(r), :] = ORANGE
    for c in vertical_cols:
        out[:, int(c)] = ORANGE
    return out


def _blocked_separators(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    horizontal_rows: set[int],
    vertical_cols: set[int],
    shape: tuple[int, int],
) -> tuple[list[bool], list[bool]]:
    row_separators = _separator_positions(row_bands, shape[0])
    col_separators = _separator_positions(col_bands, shape[1])
    assert horizontal_rows <= set(row_separators[1:-1]), "horizontal markers must target interior separator rows"
    assert vertical_cols <= set(col_separators[1:-1]), "vertical markers must target interior separator cols"
    blocked_h = [row_separators[i + 1] in horizontal_rows for i in range(len(row_bands) - 1)]
    blocked_v = [col_separators[i + 1] in vertical_cols for i in range(len(col_bands) - 1)]
    return blocked_h, blocked_v


def _open_tile_mask(rows: int, cols: int, blocked_h: list[bool], blocked_v: list[bool]) -> np.ndarray:
    open_mask = np.zeros((rows, cols), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(rows):
        for c in (0, cols - 1):
            if not open_mask[r, c]:
                open_mask[r, c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if not open_mask[r, c]:
                open_mask[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        neighbors = []
        if r > 0 and not blocked_h[r - 1]:
            neighbors.append((r - 1, c))
        if r + 1 < rows and not blocked_h[r]:
            neighbors.append((r + 1, c))
        if c > 0 and not blocked_v[c - 1]:
            neighbors.append((r, c - 1))
        if c + 1 < cols and not blocked_v[c]:
            neighbors.append((r, c + 1))
        for nr, nc in neighbors:
            if not open_mask[nr, nc]:
                open_mask[nr, nc] = True
                queue.append((nr, nc))

    return open_mask


def _invert_tile(tile: np.ndarray) -> np.ndarray:
    return np.where(tile == TEAL, BG, RED).astype(int)


def _render_tiles(tile_patterns: list[np.ndarray], tile_rows: int, tile_cols: int) -> np.ndarray:
    grid = np.zeros((tile_rows * STRIDE + 1, tile_cols * STRIDE + 1), dtype=int)
    idx = 0
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            r0 = 1 + tr * STRIDE
            c0 = 1 + tc * STRIDE
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile_patterns[idx]
            idx += 1
    return grid


def _place_markers(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    h_sep_indices: list[int],
    v_sep_indices: list[int],
) -> None:
    row_separators = _separator_positions(row_bands, grid.shape[0])
    col_separators = _separator_positions(col_bands, grid.shape[1])
    for sep_idx in h_sep_indices:
        r = row_separators[int(sep_idx)]
        c = 0 if random.random() < 0.5 else grid.shape[1] - 1
        grid[r, c] = ORANGE
    for sep_idx in v_sep_indices:
        c = col_separators[int(sep_idx)]
        r = 0 if random.random() < 0.5 else grid.shape[0] - 1
        grid[r, c] = ORANGE


def _tile_slices(row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], tr: int, tc: int) -> tuple[slice, slice]:
    r0, r1 = row_bands[int(tr)]
    c0, c1 = col_bands[int(tc)]
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        row_bands, col_bands = _infer_tile_bands(grid)
        if len(row_bands) < 4 or len(col_bands) < 4:
            return False

        for tr in range(len(row_bands)):
            for tc in range(len(col_bands)):
                rs, cs = _tile_slices(row_bands, col_bands, tr, tc)
                tile = grid[rs, cs]
                if tile.shape != (TILE_SIZE, TILE_SIZE):
                    return False
                if not np.all(np.isin(tile, [BG, TEAL])):
                    return False
                if int(np.sum(tile == TEAL)) < 2 or int(np.sum(tile == BG)) < 2:
                    return False

        tile_keys = {
            tuple(map(tuple, grid[_tile_slices(row_bands, col_bands, tr, tc)].tolist()))
            for tr in range(len(row_bands))
            for tc in range(len(col_bands))
        }
        if len(tile_keys) != len(row_bands) * len(col_bands):
            return False

        horizontal_rows, vertical_cols = _marker_lines(grid)
        if len(horizontal_rows) != 2 or len(vertical_cols) != 2:
            return False

        blocked_h, blocked_v = _blocked_separators(
            row_bands, col_bands, horizontal_rows, vertical_cols, grid.shape
        )
        open_mask = _open_tile_mask(len(row_bands), len(col_bands), blocked_h, blocked_v)
        closed_mask = ~open_mask
        if not closed_mask.any() or not open_mask.any():
            return False

        out = main(grid)
        expected_orange = np.zeros_like(out, dtype=bool)
        for r in horizontal_rows:
            expected_orange[int(r), :] = True
        for c in vertical_cols:
            expected_orange[:, int(c)] = True
        if not np.array_equal(out == ORANGE, expected_orange):
            return False

        for tr in range(len(row_bands)):
            for tc in range(len(col_bands)):
                rs, cs = _tile_slices(row_bands, col_bands, tr, tc)
                src_tile = grid[rs, cs]
                dst_tile = out[rs, cs]
                expected = _invert_tile(src_tile) if closed_mask[tr, tc] else src_tile
                if not np.array_equal(dst_tile, expected):
                    return False

        no_marker_grid = grid.copy()
        no_marker_grid[no_marker_grid == ORANGE] = BG
        if np.array_equal(main(no_marker_grid), out):
            return False
    except Exception:
        return False

    return True


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Draw full-span separator lines from edge markers, then invert only the tile regions cut off from the outside.

    Input structure:
    - The canvas is a black background with a regular lattice of 3x3 teal-pattern tiles.
    - Tiles are separated by one-cell black separator rows and columns; the outer border is part of the same separator lattice.
    - Orange single-pixel markers sit on the canvas boundary, and each marker aligns with one separator row or separator column.

    Rule:
    - Read every orange boundary marker and extend it across the whole matching separator row or separator column as an orange span line.
    - Treat unpainted separator gaps as open passages between neighboring tiles and from boundary-adjacent tiles to the outside.
    - The span lines split the tile lattice into connected tile regions; only regions whose tiles can no longer reach the outer boundary through those open passages are enclosed.
    - For each enclosed 3x3 tile, replace teal cells by black and replace black cells by red; open regions keep their original tile patterns.
    - Separator rows and columns stay black unless covered by a span line, in which case they become orange.

    Key insight:
    - The orange markers do not label tiles directly: they turn specific separator rows and columns into walls, and only tile groups fully cut off from the outer boundary undergo the local occupancy inversion.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    row_bands, col_bands = _infer_tile_bands(grid)
    horizontal_rows, vertical_cols = _marker_lines(grid)
    blocked_h, blocked_v = _blocked_separators(row_bands, col_bands, horizontal_rows, vertical_cols, grid.shape)
    open_mask = _open_tile_mask(len(row_bands), len(col_bands), blocked_h, blocked_v)
    closed_mask = ~open_mask

    out = _draw_full_span_lines(grid, horizontal_rows, vertical_cols)
    for tr in range(len(row_bands)):
        for tc in range(len(col_bands)):
            if not closed_mask[tr, tc]:
                continue
            rs, cs = _tile_slices(row_bands, col_bands, tr, tc)
            out[rs, cs] = _invert_tile(grid[rs, cs])
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile_rows = random.randint(4, 6)
        tile_cols = random.randint(4, 6)
        patterns = random.sample(PATTERNS, tile_rows * tile_cols)
        grid = _render_tiles(patterns, tile_rows, tile_cols)
        row_bands, col_bands = _infer_tile_bands(grid)

        h_sep_indices = sorted(random.sample(range(1, tile_rows), 2))
        v_sep_indices = sorted(random.sample(range(1, tile_cols), 2))
        _place_markers(grid, row_bands, col_bands, h_sep_indices, v_sep_indices)

        if _validate_generated_grid(grid):
            return grid

    raise RuntimeError("failed to generate a valid selective-tile-inversion input")
