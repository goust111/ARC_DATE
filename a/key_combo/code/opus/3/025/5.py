import numpy as np


BLACK = 0
MAROON = 9
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
ROW_MULTIPLES = {2: (8, 9, 10), 3: (5, 6, 7)}
COL_MULTIPLES = {2: (8, 9, 10), 3: (5, 6, 7)}
GEN_RETRIES = 300


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive (r0, c0, r1, c1) bbox for an (N, 2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g != BLACK)
    assert coords.size > 0, "expected a visible motif block"

    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    assert (r0, c0) == (0, 0), "the motif block must start at the upper-left corner"

    block = crop_bbox(g, bbox)
    h, w = block.shape
    assert h in (5, 7) and w in (5, 7), "motif block must be 5x5, 5x7, 7x5, or 7x7"
    assert np.all(block[::2, :] == MAROON), "every horizontal lattice line must be maroon"
    assert np.all(block[:, ::2] == MAROON), "every vertical lattice line must be maroon"

    outside = np.ones(g.shape, dtype=bool)
    outside[: h, : w] = False
    assert np.all(g[outside] == BLACK), "everything outside the motif block must be background"

    return block[1::2, 1::2].copy()


def _tiled_output(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    rows, cols = tile.shape
    H, W = map(int, shape)
    assert H % rows == 0 and W % cols == 0, "grid shape must be an integer number of decoded tiles"

    tile_rows = H // rows
    tile_cols = W // cols
    out = np.empty((H, W), dtype=int)
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            idx = tr * tile_cols + tc
            patch = tile if idx % 2 == 0 else tile[:, ::-1]
            r0 = tr * rows
            c0 = tc * cols
            out[r0 : r0 + rows, c0 : c0 + cols] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored slot map in the upper-left lattice and tile it with odd tiles mirrored.

    Input structure:
    - A black background contains exactly one maroon (9) lattice block in the upper-left corner.
    - The lattice block is (2R+1) by (2C+1), with 1-pixel maroon grid lines and RxC single-pixel slots.
    - Each odd/odd slot cell stores one visible color, possibly including maroon itself, while everything outside the block is black.

    Rule:
    - Read the slot colors inside the maroon lattice to obtain an RxC base tile.
    - Cover the full output canvas, including the original lattice area, with contiguous RxC tiles starting at the upper-left corner.
    - Number tiles in row-major order from 0; keep even-indexed tiles unchanged and mirror every odd-indexed tile left-to-right.

    Key insight:
    - The small maroon lattice is only a legend: its slot colors define the tile content, and the global alternation comes from each tile's row-major parity rather than from multiple different input tiles.
    """
    g = np.asarray(grid, dtype=int)
    tile = _extract_tile(g)
    return _tiled_output(tile, g.shape)


def _pick_alt_color(color: int) -> int:
    for alt in PALETTE:
        if int(alt) != int(color):
            return int(alt)
    raise AssertionError("palette must contain at least two colors")


def _generate_tile(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    while True:
        unique_count = int(rng.integers(2, min(rows * cols, 5) + 1))
        palette = rng.choice(PALETTE, size=unique_count, replace=False)
        tile = rng.choice(palette, size=(rows, cols), replace=True).astype(int)

        if rng.random() < 0.55:
            rr = int(rng.integers(rows))
            cc = int(rng.integers(cols))
            tile[rr, cc] = MAROON

        if len(np.unique(tile)) < 2:
            continue
        if np.array_equal(tile, tile[:, ::-1]):
            continue
        if np.count_nonzero(tile != tile[:, ::-1]) < 2:
            continue
        return tile


def _build_input(tile: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    rows, cols = tile.shape
    H = rows * tile_rows
    W = cols * tile_cols
    grid = np.zeros((H, W), dtype=int)

    block = np.full((2 * rows + 1, 2 * cols + 1), MAROON, dtype=int)
    block[1::2, 1::2] = tile
    grid[: block.shape[0], : block.shape[1]] = block
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        tile = _extract_tile(grid)
    except AssertionError:
        return False

    rows, cols = tile.shape
    H, W = grid.shape
    if H % rows != 0 or W % cols != 0:
        return False

    tile_rows = H // rows
    tile_cols = W // cols
    if tile_rows < 2 or tile_cols < 3:
        return False
    if len(np.unique(tile)) < 2:
        return False
    if np.array_equal(tile, tile[:, ::-1]):
        return False

    out = _tiled_output(tile, grid.shape)
    if not np.array_equal(out, main(grid)):
        return False

    no_mirror = np.tile(tile, (tile_rows, tile_cols))
    if np.array_equal(out, no_mirror):
        return False
    if not np.array_equal(out[:rows, :cols], tile):
        return False
    if not np.array_equal(out[:rows, cols : 2 * cols], tile[:, ::-1]):
        return False

    for rr in range(rows):
        for cc in range(cols):
            alt_tile = tile.copy()
            alt_tile[rr, cc] = _pick_alt_color(int(tile[rr, cc]))
            if np.array_equal(_tiled_output(alt_tile, grid.shape), out):
                return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        rows = int(rng.choice([2, 3]))
        cols = int(rng.choice([2, 3]))
        tile_rows = int(rng.choice(ROW_MULTIPLES[rows]))
        tile_cols = int(rng.choice(COL_MULTIPLES[cols]))

        tile = _generate_tile(rng, rows, cols)
        grid = _build_input(tile, tile_rows, tile_cols)
        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid alternating-mirror tiling puzzle")
