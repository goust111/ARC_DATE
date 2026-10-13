from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREY = 5
SPRITE_COLORS = (1, 3, 4, 6, 7, 8, 9)
TILE_SIZE = 3
MACRO_SIZE = 3
STRIDE = TILE_SIZE + 1
GRID_SIZE = MACRO_SIZE * TILE_SIZE + (MACRO_SIZE - 1)
SEPARATORS = (3, 7)
GEN_RETRIES = 300
TILE_CELLS = [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)]


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    starts = [0] + [s + 1 for s in separators]
    ends = separators + [size]
    bands = list(zip(starts, ends))
    assert len(bands) == MACRO_SIZE
    assert all(lo < hi for lo, hi in bands)
    return bands


def _tile_slice(br: int, bc: int) -> tuple[slice, slice]:
    r0 = br * STRIDE
    c0 = bc * STRIDE
    return slice(r0, r0 + TILE_SIZE), slice(c0, c0 + TILE_SIZE)


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mirror_mask(mask: np.ndarray, local_row: int) -> np.ndarray:
    if local_row == 0:
        return np.flipud(mask)
    if local_row == 1:
        return np.fliplr(mask)
    if local_row == 2:
        return np.flipud(np.fliplr(mask))
    raise AssertionError("local row must be 0, 1, or 2")


def _build_empty_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    grid[list(SEPARATORS), :] = GREY
    grid[:, list(SEPARATORS)] = GREY
    return grid


def _source_tiles(grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for br in range(MACRO_SIZE):
        for bc in range(MACRO_SIZE):
            r0, r1 = row_bands[br]
            c0, c1 = col_bands[bc]
            tile = grid[r0:r1, c0:c1]
            if np.any(tile != BACKGROUND):
                hits.append((br, bc))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the marked sprite tile, mirror it from the marker's local row, and stamp it into the slot named by the marker's local column.

    Input structure:
    - The grid is an 11x11 canvas partitioned by full grey separator rows and columns into a 3x3 macro grid of 3x3 tiles.
    - Exactly one tile contains a single-color foreground sprite on black background, with one red pixel embedded in the sprite as a marker.
    - The other eight tiles are empty black, and separator cells stay grey.

    Rule:
    - Find the unique non-empty source tile and locate its red marker pixel at local coordinate (lr, lc) inside the 3x3 tile.
    - Restore the marker cell to the sprite color, then mirror the whole sprite using lr: 0 means vertical flip across a horizontal axis, 1 means horizontal flip across a vertical axis, 2 means both flips (180-degree rotation).
    - Stamp the mirrored sprite into the tile on the same macro row whose macro column equals lc, clear the source tile to black first, and preserve the grey separators.

    Key insight:
    - The single red marker encodes two independent instructions at once: its local row chooses the mirror transform, and its local column chooses the destination slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)

    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert row_seps == list(SEPARATORS)
    assert col_seps == list(SEPARATORS)

    row_bands = _bands(row_seps, g.shape[0])
    col_bands = _bands(col_seps, g.shape[1])
    assert all(hi - lo == TILE_SIZE for lo, hi in row_bands + col_bands)

    non_separator = np.ones_like(g, dtype=bool)
    non_separator[row_seps, :] = False
    non_separator[:, col_seps] = False
    assert not np.any(g[non_separator] == GREY)
    assert set(np.unique(g[non_separator])).issubset({BACKGROUND, RED, *SPRITE_COLORS})

    source_slots = _source_tiles(g, row_bands, col_bands)
    assert len(source_slots) == 1
    src_br, src_bc = source_slots[0]

    src_rs, src_cs = _tile_slice(src_br, src_bc)
    source_tile = g[src_rs, src_cs]
    markers = np.argwhere(source_tile == RED)
    assert markers.shape == (1, 2)
    local_row, local_col = map(int, markers[0])
    sprite_values = sorted(int(v) for v in np.unique(source_tile) if int(v) not in (BACKGROUND, RED))
    assert len(sprite_values) == 1
    sprite_color = int(sprite_values[0])

    sprite_mask = source_tile != BACKGROUND
    assert 3 <= int(sprite_mask.sum()) <= 7
    assert _connected4(sprite_mask)

    mirrored = _mirror_mask(sprite_mask, local_row)
    dst_br = src_br
    dst_bc = local_col

    out = _build_empty_grid()
    dst_rs, dst_cs = _tile_slice(dst_br, dst_bc)
    out[dst_rs, dst_cs][mirrored] = sprite_color
    return out


def _valid_source_masks() -> dict[tuple[int, int], list[np.ndarray]]:
    by_marker = {(r, c): [] for r, c in TILE_CELLS}
    for area in range(3, 8):
        for cells in itertools.combinations(range(TILE_SIZE * TILE_SIZE), area):
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
            rows, cols = np.divmod(cells, TILE_SIZE)
            mask[rows, cols] = True
            if not _connected4(mask):
                continue
            if np.array_equal(mask, np.flipud(mask)):
                continue
            if np.array_equal(mask, np.fliplr(mask)):
                continue
            if np.array_equal(mask, np.flipud(np.fliplr(mask))):
                continue
            if len(np.unique(np.argwhere(mask)[:, 0])) == 1:
                continue
            if len(np.unique(np.argwhere(mask)[:, 1])) == 1:
                continue
            for marker in np.argwhere(mask):
                by_marker[tuple(map(int, marker))].append(mask.copy())
    for marker, masks in by_marker.items():
        assert masks, f"no valid masks for marker {marker}"
    return by_marker


VALID_MASKS_BY_MARKER = _valid_source_masks()


def _expected_output(src_br: int, local_row: int, local_col: int, mask: np.ndarray, sprite_color: int) -> np.ndarray:
    out = _build_empty_grid()
    dst_rs, dst_cs = _tile_slice(src_br, local_col)
    out[dst_rs, dst_cs][_mirror_mask(mask, local_row)] = int(sprite_color)
    return out


def _attempt_input() -> np.ndarray | None:
    src_br = random.randrange(MACRO_SIZE)
    local_row = random.randrange(TILE_SIZE)
    local_col = random.randrange(TILE_SIZE)
    sprite_color = random.choice(SPRITE_COLORS)

    if random.random() < 0.5:
        src_bc = local_col
    else:
        src_bc = random.choice([bc for bc in range(MACRO_SIZE) if bc != local_col])

    mask = random.choice(VALID_MASKS_BY_MARKER[(local_row, local_col)]).copy()
    expected = _expected_output(src_br, local_row, local_col, mask, sprite_color)

    grid = _build_empty_grid()
    rs, cs = _tile_slice(src_br, src_bc)
    grid[rs, cs][mask] = sprite_color
    grid[rs.start + local_row, cs.start + local_col] = RED

    # Semantic validation: verify the intended decoding is unique and non-degenerate.
    tile = grid[rs, cs]
    assert np.count_nonzero(tile == RED) == 1
    assert np.array_equal(tile != BACKGROUND, mask)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
