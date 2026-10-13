import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique stacked tile that breaks the diagonal symmetry shared by the other tiles.

    Input structure
    - A 9x3 grid made of three vertically stacked 3x3 two-color tiles.
    - Exactly two tiles are symmetric across the main diagonal, and exactly one is not.

    Rule
    - Split the input into its three 3x3 tiles.
    - Return the only tile whose cells are not equal to their transposed positions.

    Key insight
    - The desired tile is identified by structural asymmetry, not by its colors or by its position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert tuple(g.shape) == (9, 3), "expected three stacked 3x3 tiles"

    tiles = [g[r : r + 3, :] for r in range(0, 9, 3)]
    assert all(tile.shape == (3, 3) for tile in tiles)
    assert all(len(np.unique(tile)) == 2 for tile in tiles), "each tile should use two colors"

    asymmetric_tiles = [tile for tile in tiles if not np.array_equal(tile, tile.T)]
    assert len(asymmetric_tiles) == 1, "expected exactly one asymmetric tile"
    assert sum(np.array_equal(tile, tile.T) for tile in tiles) == 2

    return asymmetric_tiles[0].copy()

# --------------- generator (o5x) ---------------

# CFG: fixed stack geometry and non-black color pool.
TILE_SIZE, TILE_COUNT, COLORS_PER_TILE = 3, 3, 2
COLOR_VALUES = np.arange(1, 10)
# CFG: binary mask constraints and oracle retry budget.
MIN_MINOR_CELLS, MAX_MINOR_CELLS = 2, 4
MAX_GENERATION_ATTEMPTS = 300


def _build_binary_masks(require_symmetric: bool) -> list[np.ndarray]:
    masks = []
    tile_area = TILE_SIZE * TILE_SIZE
    for bits in range(1, 1 << tile_area):
        values = [(bits >> i) & 1 for i in range(tile_area)]
        mask = np.array(values, dtype=bool).reshape(TILE_SIZE, TILE_SIZE)
        if np.array_equal(mask, mask.T) != require_symmetric:
            continue
        minor_count = min(int(mask.sum()), mask.size - int(mask.sum()))
        if MIN_MINOR_CELLS <= minor_count <= MAX_MINOR_CELLS:
            masks.append(mask)
    return masks


MASKS_BY_SYMMETRY = {True: _build_binary_masks(True), False: _build_binary_masks(False)}


def _gen_tile(require_symmetric: bool, colors: np.ndarray) -> np.ndarray:
    masks = MASKS_BY_SYMMETRY[require_symmetric]
    mask = masks[np.random.randint(len(masks))]
    if np.random.randint(2):
        mask = ~mask
    return np.where(mask, int(colors[0]), int(colors[1]))


def _gen_attempt() -> np.ndarray | None:
    colors = np.random.choice(COLOR_VALUES, size=TILE_COUNT * COLORS_PER_TILE, replace=False)
    color_pairs = colors.reshape(TILE_COUNT, COLORS_PER_TILE)
    asymmetric_slot = int(np.random.randint(TILE_COUNT))

    tiles = [_gen_tile(slot != asymmetric_slot, color_pairs[slot]) for slot in range(TILE_COUNT)]

    inp = np.vstack(tiles).astype(int)
    expected = tiles[asymmetric_slot]
    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 662c240a input")
