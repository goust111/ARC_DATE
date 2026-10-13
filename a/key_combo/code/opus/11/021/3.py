from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
TILE_COLORS = (BLUE, RED, YELLOW)

TILE_SIZE_CHOICES = (3, 4)
REPETITION_RANGE = (3, 6)
MARKER_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _first_full_color_row(grid: np.ndarray, color: int) -> int:
    for r in range(1, grid.shape[0]):
        if np.all(grid[r, :] == color):
            return int(r)
    raise AssertionError("expected a full placeholder row below the visible tile")


def _first_full_color_col(grid: np.ndarray, color: int) -> int:
    for c in range(1, grid.shape[1]):
        if np.all(grid[:, c] == color):
            return int(c)
    raise AssertionError("expected a full placeholder column right of the visible tile")


def _matches_period(tile: np.ndarray, period_h: int, period_w: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[r % period_h, c % period_w]):
                return False
    return True


def _is_minimal_tile(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for period_h in range(1, h + 1):
        if h % period_h != 0:
            continue
        for period_w in range(1, w + 1):
            if w % period_w != 0:
                continue
            if period_h == h and period_w == w:
                continue
            if _matches_period(tile, period_h, period_w):
                return False
    return True


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    tile_h = _first_full_color_row(grid, GREY)
    tile_w = _first_full_color_col(grid, GREY)
    tile = np.asarray(grid[:tile_h, :tile_w], dtype=int)

    assert tile_h in TILE_SIZE_CHOICES and tile_w in TILE_SIZE_CHOICES
    assert tile.size > 0
    assert not np.any(tile == GREY)
    assert not np.any(tile == ORANGE)
    assert set(np.unique(tile)).issubset(set(TILE_COLORS))
    assert len(set(map(int, tile.flat))) >= 2
    assert _is_minimal_tile(tile)
    return tile


def _periodic_completion(grid: np.ndarray, tile: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    tile_h, tile_w = tile.shape
    rr = np.arange(h)[:, None] % tile_h
    cc = np.arange(w)[None, :] % tile_w
    completed = tile[rr, cc].astype(int)
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = completed[out == GREY]
    return out


def _block_bounds(
    r: int,
    c: int,
    tile_h: int,
    tile_w: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = grid_shape
    r0 = (int(r) // tile_h) * tile_h
    c0 = (int(c) // tile_w) * tile_w
    r1 = min(r0 + tile_h, h)
    c1 = min(c0 + tile_w, w)
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden periodic tiling, then use each orange marker's local tile position to choose which color to erase inside that marker's tile copy.

    Input structure:
    - A grid contains a fully visible 3x3 or 4x4 multicolor tile in the top-left corner and grey placeholder cells elsewhere.
    - The tile uses only blue, red, and yellow, with at least two distinct colors and no orange.
    - Two to four isolated orange marker pixels lie in placeholder territory, with each marker in a different tile copy.

    Rule:
    - Infer the tile size from the first all-grey row and first all-grey column, then treat the top-left rectangle before those boundaries as the tile.
    - Periodically repeat that tile across the whole grid, replacing grey placeholders but keeping orange markers visible for now.
    - For each orange marker, find the tile copy containing it, compute the marker's local coordinate within that copy, read the tile color at the same local coordinate, and erase every pixel of that color inside only that tile copy.
    - Finally erase the orange markers themselves to black.

    Key insight:
    - The orange pixel does not mark a single cell to delete; its position encodes a color choice via the repeated tile, and the deletion scope is only the marker's own tile instance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    tile = _infer_tile(g)
    tile_h, tile_w = tile.shape
    out = _periodic_completion(g, tile)

    for r, c in np.argwhere(g == ORANGE):
        r = int(r)
        c = int(c)
        r0, r1, c0, c1 = _block_bounds(r, c, tile_h, tile_w, g.shape)
        local_r = r - r0
        local_c = c - c0
        chosen_color = int(tile[local_r, local_c])
        block = out[r0:r1, c0:c1]
        block[block == chosen_color] = BLACK

    out[out == ORANGE] = BLACK
    return out


def _tile_color_counts(tile: np.ndarray) -> dict[int, int]:
    colors, counts = np.unique(tile, return_counts=True)
    return {int(color): int(count) for color, count in zip(colors, counts)}


def _random_tile() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile_h = random.choice(TILE_SIZE_CHOICES)
        tile_w = random.choice(TILE_SIZE_CHOICES)
        color_count = random.choice((2, 3))
        palette = random.sample(TILE_COLORS, color_count)
        base_cells = [color for color in palette for _ in range(2)]
        remaining = tile_h * tile_w - len(base_cells)
        cells = base_cells + [random.choice(palette) for _ in range(remaining)]
        random.shuffle(cells)
        tile = np.array(cells, dtype=int).reshape(tile_h, tile_w)

        if not _is_minimal_tile(tile):
            continue
        if len({tuple(map(int, row)) for row in tile}) < 2:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(tile_w)}) < 2:
            continue
        return tile
    raise RuntimeError("failed to construct a non-degenerate minimal tile")


def _marker_candidates(
    tile: np.ndarray,
    reps_h: int,
    reps_w: int,
) -> dict[tuple[int, int], list[tuple[int, int, int, int, int]]]:
    tile_h, tile_w = tile.shape
    counts = _tile_color_counts(tile)
    candidates: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}

    for block_r in range(reps_h):
        for block_c in range(reps_w):
            if (block_r, block_c) == (0, 0):
                continue
            block_candidates = []
            for local_r in range(tile_h):
                for local_c in range(tile_w):
                    chosen_color = int(tile[local_r, local_c])
                    if counts[chosen_color] < 2:
                        continue
                    grid_r = block_r * tile_h + local_r
                    grid_c = block_c * tile_w + local_c
                    if grid_r == tile_h or grid_c == tile_w:
                        continue
                    block_candidates.append((grid_r, grid_c, chosen_color, local_r, local_c))
            if block_candidates:
                candidates[(block_r, block_c)] = block_candidates

    return candidates


def _choose_markers(
    tile: np.ndarray,
    reps_h: int,
    reps_w: int,
) -> list[tuple[int, int, int, int, int]]:
    candidates_by_block = _marker_candidates(tile, reps_h, reps_w)
    blocks = list(candidates_by_block)
    max_markers = min(MARKER_COUNT_RANGE[1], len(blocks))
    min_markers = min(MARKER_COUNT_RANGE[0], max_markers)
    if max_markers < 2:
        raise RuntimeError("not enough blocks to place markers")

    for _ in range(MAX_GENERATION_ATTEMPTS):
        marker_count = random.randint(min_markers, max_markers)
        chosen_blocks = random.sample(blocks, marker_count)
        markers = [random.choice(candidates_by_block[block]) for block in chosen_blocks]

        chosen_colors = {marker[2] for marker in markers}
        local_positions = {(marker[3], marker[4]) for marker in markers}
        global_positions = {(marker[0], marker[1]) for marker in markers}
        if len(chosen_colors) < 2:
            continue
        if len(local_positions) < 2:
            continue
        if len(global_positions) != len(markers):
            continue
        if any(
            max(abs(r1 - r2), abs(c1 - c2)) <= 1
            for i, (r1, c1, _col1, _lr1, _lc1) in enumerate(markers)
            for r2, c2, _col2, _lr2, _lc2 in markers[i + 1 :]
        ):
            continue
        return markers

    raise RuntimeError("failed to place diverse markers")


def _build_input(tile: np.ndarray, markers: list[tuple[int, int, int, int, int]], reps_h: int, reps_w: int) -> np.ndarray:
    tile_h, tile_w = tile.shape
    grid = np.full((tile_h * reps_h, tile_w * reps_w), GREY, dtype=int)
    grid[:tile_h, :tile_w] = tile
    for r, c, _color, _local_r, _local_c in markers:
        grid[r, c] = ORANGE
    return grid


def _validate_instance(
    grid: np.ndarray,
    tile: np.ndarray,
    markers: list[tuple[int, int, int, int, int]],
    reps_h: int,
    reps_w: int,
) -> None:
    inferred = _infer_tile(grid)
    assert np.array_equal(inferred, tile)
    out = main(grid)
    tile_h, tile_w = tile.shape

    assert grid.shape == (tile_h * reps_h, tile_w * reps_w)
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    assert np.all(grid[tile_h, :] == GREY)
    assert np.all(grid[:, tile_w] == GREY)

    marker_cells = np.argwhere(grid == ORANGE)
    assert 2 <= len(marker_cells) <= 4
    marker_blocks = {(int(r) // tile_h, int(c) // tile_w) for r, c in marker_cells}
    assert len(marker_blocks) == len(marker_cells)
    assert (0, 0) not in marker_blocks

    selected_colors = []
    for r, c, chosen_color, local_r, local_c in markers:
        assert int(tile[local_r, local_c]) == chosen_color
        assert _tile_color_counts(tile)[chosen_color] >= 2
        r0, r1, c0, c1 = _block_bounds(r, c, tile_h, tile_w, grid.shape)
        block = out[r0:r1, c0:c1]
        assert np.count_nonzero(block == chosen_color) == 0
        selected_colors.append(chosen_color)

    assert len(set(selected_colors)) >= 2
    assert np.array_equal(out[:tile_h, :tile_w], tile)
    assert np.count_nonzero(out == BLACK) > len(markers)

    completed = _periodic_completion(grid, tile)
    erase_only_markers = completed.copy()
    erase_only_markers[erase_only_markers == ORANGE] = BLACK
    assert not np.array_equal(out, erase_only_markers)

    global_erase = completed.copy()
    global_erase[np.isin(global_erase, list(set(selected_colors)))] = BLACK
    global_erase[completed == ORANGE] = BLACK
    assert not np.array_equal(out, global_erase)

    untouched_block_found = False
    for block_r in range(reps_h):
        for block_c in range(reps_w):
            r0 = block_r * tile_h
            c0 = block_c * tile_w
            if (block_r, block_c) in marker_blocks:
                continue
            if np.array_equal(out[r0 : r0 + tile_h, c0 : c0 + tile_w], tile):
                untouched_block_found = True
                break
        if untouched_block_found:
            break
    assert untouched_block_found


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile = _random_tile()
        reps_h = random.randint(*REPETITION_RANGE)
        reps_w = random.randint(*REPETITION_RANGE)
        if tile.shape[0] * reps_h > 30 or tile.shape[1] * reps_w > 30:
            continue

        try:
            markers = _choose_markers(tile, reps_h, reps_w)
        except RuntimeError:
            continue

        grid = _build_input(tile, markers, reps_h, reps_w)
        try:
            _validate_instance(grid, tile, markers, reps_h, reps_w)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle input")
