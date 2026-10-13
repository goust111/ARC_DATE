from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 6
REF_BG = 8
CELL = 3
TILE = 2
GLYPH_COLORS = (1, 2, 3, 4, 5, 7, 9)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cell_signature(cell: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in cell.reshape(-1))


def _render_glyph(mask_bits: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), REF_BG, dtype=int)
    mask = np.array([(mask_bits >> i) & 1 for i in range(CELL * CELL)], dtype=bool).reshape(CELL, CELL)
    cell[mask] = int(color)
    return cell


def _parse_reference_structure(grid: np.ndarray, sep_row: int) -> tuple[tuple[int, ...], list[list[tuple[int, ...]]]]:
    top = grid[:sep_row, :]
    box = _bbox(top != BG)
    assert box is not None, "expected a non-empty reference structure above the separator"
    r0, c0, r1, c1 = box
    block = top[r0 : r1 + 1, c0 : c1 + 1]
    h, w = block.shape
    assert h % CELL == 0 and w % CELL == 0, "reference structure must align to 3x3 glyph cells"
    assert h >= 2 * CELL, "reference structure must contain a query row plus table rows"
    nrows = h // CELL - 1
    ncols = w // CELL

    query = tuple(
        _cell_signature(block[:CELL, c * CELL : (c + 1) * CELL])
        for c in range(ncols)
    )
    table = [
        [
            _cell_signature(block[(r + 1) * CELL : (r + 2) * CELL, c * CELL : (c + 1) * CELL])
            for c in range(ncols)
        ]
        for r in range(nrows)
    ]
    return query, table


def _matched_rows(grid: np.ndarray, sep_row: int) -> set[int]:
    query, table = _parse_reference_structure(grid, sep_row)
    query_set = set(query)
    hits: set[int] = set()
    for r, row in enumerate(table):
        for sig in row:
            if sig in query_set:
                hits.add(int(r))
    return hits


def _tile_output(bottom: np.ndarray, mirrored_rows: set[int]) -> np.ndarray:
    assert bottom.shape[0] >= TILE and bottom.shape[1] >= TILE, "bottom region must fit the 2x2 tile"
    tile = bottom[:TILE, :TILE].copy()
    assert not np.array_equal(tile, np.fliplr(tile)), "tile must not be left-right symmetric"

    out = np.zeros_like(bottom)
    tile_rows = bottom.shape[0] // TILE
    tile_cols = bottom.shape[1] // TILE
    for tr in range(tile_rows):
        use = np.fliplr(tile) if tr in mirrored_rows else tile
        for tc in range(tile_cols):
            r0 = tr * TILE
            c0 = tc * TILE
            out[r0 : r0 + TILE, c0 : c0 + TILE] = use
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror selected tile rows according to which reference-table rows match the query glyphs.

    Input structure:
    - A black background grid is split by one full-width pink separator row.
    - Above the separator, a cyan-backed glyph structure contains one query row of 3x3 glyph cells
      and, directly below it, a reference table of the same 3x3 glyph-cell format.
    - Below the separator, only the upper-left 2x2 tile is drawn; the rest of the lower region is blank.

    Rule:
    - Match every query glyph against every reference-table glyph using exact equality of both color
      and 3x3 shape.
    - Collect the internal table row indices of all matched cells.
    - Tile the lower region with the 2x2 seed tile, starting from the region's upper-left corner and
      ignoring any incomplete one-cell edge strip.
    - For each tile row, use the tile's horizontal mirror iff that tile-row index is in the matched-row set;
      otherwise use the original tile. Keep the upper region and separator unchanged.

    Key insight:
    - The lookup does not color the table itself; it produces row indices that dynamically decide which
      rows of a separate tiled array switch to the mirrored tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    assert len(sep_rows) == 1, "expected exactly one full-width separator row"
    sep_row = int(sep_rows[0])

    mirrored_rows = _matched_rows(g, sep_row)
    out = g.copy()
    out[sep_row + 1 :, :] = _tile_output(g[sep_row + 1 :, :], mirrored_rows)
    return out


PATTERN_POOL = tuple(
    mask
    for mask in range(1, 1 << (CELL * CELL))
    if 2 <= mask.bit_count() <= 6
    and np.count_nonzero(np.any(np.array([(mask >> i) & 1 for i in range(CELL * CELL)], dtype=bool).reshape(CELL, CELL), axis=1)) >= 2
    and np.count_nonzero(np.any(np.array([(mask >> i) & 1 for i in range(CELL * CELL)], dtype=bool).reshape(CELL, CELL), axis=0)) >= 2
)


def _sample_tile() -> np.ndarray:
    colors = random.sample(GLYPH_COLORS, 2)
    while True:
        tile = np.array(random.choices(colors, k=TILE * TILE), dtype=int).reshape(TILE, TILE)
        if len(set(map(int, tile.reshape(-1)))) != 2:
            continue
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        return tile


def _sample_reference_pairs(count: int) -> list[tuple[int, int]]:
    universe = [(mask, color) for mask in PATTERN_POOL for color in GLYPH_COLORS]
    return random.sample(universe, count)


def _choose_query_cells(table_rows: int, table_cols: int, mirrored_rows: set[int]) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    for r in sorted(mirrored_rows):
        c = random.randrange(table_cols)
        cell = (r, c)
        chosen.append(cell)
        used.add(cell)

    pool = [
        (r, c)
        for r in sorted(mirrored_rows)
        for c in range(table_cols)
        if (r, c) not in used
    ]
    random.shuffle(pool)
    while len(chosen) < table_cols:
        cell = pool.pop()
        chosen.append(cell)
        used.add(cell)

    random.shuffle(chosen)
    return chosen


def _build_reference_block(
    table_rows: int,
    table_cols: int,
    mirrored_rows: set[int],
) -> tuple[np.ndarray, list[list[tuple[int, ...]]], list[tuple[int, int]]]:
    pairs = _sample_reference_pairs(table_rows * table_cols)
    table_sigs: list[list[tuple[int, ...]]] = []
    block = np.full((CELL * (table_rows + 1), CELL * table_cols), REF_BG, dtype=int)

    idx = 0
    for r in range(table_rows):
        row_sigs: list[tuple[int, ...]] = []
        for c in range(table_cols):
            mask_bits, color = pairs[idx]
            idx += 1
            glyph = _render_glyph(mask_bits, color)
            row_sigs.append(_cell_signature(glyph))
            r0 = CELL * (r + 1)
            c0 = CELL * c
            block[r0 : r0 + CELL, c0 : c0 + CELL] = glyph
        table_sigs.append(row_sigs)

    query_cells = _choose_query_cells(table_rows, table_cols, mirrored_rows)
    for qc, (r, c) in enumerate(query_cells):
        glyph = np.array(table_sigs[r][c], dtype=int).reshape(CELL, CELL)
        c0 = CELL * qc
        block[:CELL, c0 : c0 + CELL] = glyph

    return block, table_sigs, query_cells


def _expected_from_parts(
    top_block: np.ndarray,
    bottom: np.ndarray,
    mirrored_rows: set[int],
    width: int,
) -> np.ndarray:
    top_h = top_block.shape[0]
    grid = np.zeros((top_h + 1 + bottom.shape[0], width), dtype=int)
    grid[:top_h, : top_block.shape[1]] = top_block
    grid[top_h, :] = SEP
    grid[top_h + 1 :, : bottom.shape[1]] = _tile_output(bottom, mirrored_rows)
    return grid


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    mirrored_rows: set[int],
    table_rows: int,
    bottom_shape: tuple[int, int],
    tile: np.ndarray,
) -> bool:
    top_h = CELL * (table_rows + 1)
    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    parsed_hits = _matched_rows(grid, top_h)
    if parsed_hits != mirrored_rows:
        return False

    if not mirrored_rows or len(mirrored_rows) == table_rows:
        return False
    if table_rows == 3 and mirrored_rows == {1}:
        return False

    tile_rows = bottom_shape[0] // TILE
    tile_cols = bottom_shape[1] // TILE
    bottom_out = out[top_h + 1 :, :]
    if tile_rows != table_rows:
        return False
    if not any(r in mirrored_rows for r in range(tile_rows)):
        return False
    if not any(r not in mirrored_rows for r in range(tile_rows)):
        return False

    mirror = np.fliplr(tile)
    for tr in range(tile_rows):
        want = mirror if tr in mirrored_rows else tile
        for tc in range(tile_cols):
            r0 = tr * TILE
            c0 = tc * TILE
            if not np.array_equal(bottom_out[r0 : r0 + TILE, c0 : c0 + TILE], want):
                return False

    if bottom_shape[0] % TILE:
        if np.any(bottom_out[-1, :] != BG):
            return False
    if bottom_shape[1] % TILE:
        if np.any(bottom_out[:, -1] != BG):
            return False
    return not np.array_equal(grid, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        table_rows = random.randint(3, 4)
        table_cols = random.randint(3, 4)
        subset_size = random.randint(1, min(table_rows - 1, table_cols))
        mirrored_rows = set(random.sample(range(table_rows), subset_size))
        if table_rows == 3 and mirrored_rows == {1}:
            continue

        top_block, _table_sigs, _query_cells = _build_reference_block(table_rows, table_cols, mirrored_rows)

        extra_h = random.choice((0, 0, 1))
        extra_w = random.choice((0, 0, 1))
        tile_cols = random.randint(3, 5)
        tile = _sample_tile()
        bottom = np.zeros((TILE * table_rows + extra_h, TILE * tile_cols + extra_w), dtype=int)
        bottom[:TILE, :TILE] = tile

        width = max(top_block.shape[1], bottom.shape[1])
        top_h = top_block.shape[0]
        grid = np.zeros((top_h + 1 + bottom.shape[0], width), dtype=int)
        grid[:top_h, : top_block.shape[1]] = top_block
        grid[top_h, :] = SEP
        grid[top_h + 1 :, : bottom.shape[1]] = bottom

        expected = _expected_from_parts(top_block, bottom, mirrored_rows, width)
        if _validate_case(grid, expected, mirrored_rows, table_rows, bottom.shape, tile):
            return grid

    raise RuntimeError("failed to generate a valid selective-mirror ARC input")
