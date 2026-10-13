import numpy as np

from arc_utils import Color, colors_present


TILE_SIZE = 3


def _strip_tiles(g: np.ndarray) -> list[np.ndarray]:
    h, w = g.shape
    horizontal = h == TILE_SIZE and w % TILE_SIZE == 0
    vertical = w == TILE_SIZE and h % TILE_SIZE == 0
    assert horizontal != vertical, "expected a horizontal or vertical strip of 3x3 tiles"

    if horizontal:
        tiles = [g[:, c : c + TILE_SIZE] for c in range(0, w, TILE_SIZE)]
    else:
        tiles = [g[r : r + TILE_SIZE, :] for r in range(0, h, TILE_SIZE)]

    assert len(tiles) >= 2, "expected multiple candidate tiles"
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the 3x3 sprite with the largest foreground-pixel count and return that sprite.

    Input structure
    The input is a one-dimensional strip of 3x3 tiles, either horizontal or vertical.
    Black is the background color. Each tile contains exactly one non-black color.

    Rule
    Partition the strip into its 3x3 tiles. Count non-black pixels in each tile,
    require a unique maximum count, and output that winning tile unchanged.

    Key insight
    The selected sprite is determined by filled area, not by color or by its position
    in the strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = Color.BLACK
    tiles = _strip_tiles(g)

    scores = []
    for tile in tiles:
        assert tile.shape == (TILE_SIZE, TILE_SIZE), "each candidate must be a 3x3 tile"
        foreground_colors = colors_present(tile, exclude={bg})
        assert len(foreground_colors) == 1, "each tile must have one foreground color"
        assert np.count_nonzero(tile != bg) > 0, "each tile must contain foreground pixels"
        scores.append(int(np.count_nonzero(tile != bg)))

    best_score = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == best_score]
    assert len(winners) == 1, "expected a unique largest foreground-pixel count"
    return tiles[winners[0]].copy()

# --------------- generator (o5x) ---------------

# Sampling bounds and retry budget.
_GEN_RNG = np.random.default_rng()
_GEN_BG, _GEN_TILE_SIZE = 0, 3
_GEN_TILE_CELLS = _GEN_TILE_SIZE * _GEN_TILE_SIZE
_GEN_COLORS = np.arange(1, 10)
_GEN_TILE_COUNT_BOUNDS = (3, 6)
_GEN_ORIENTATION_CHOICES = 2
_GEN_MAX_ATTEMPTS = 300

# Foreground-pixel count bounds; upper bounds are exclusive.
_GEN_LOSER_COUNT_BOUNDS = (4, 7)
_GEN_WINNER_MIN_COUNT, _GEN_WINNER_MAX_COUNT = 5, 9


def _gen_mask_with_count(count: int) -> np.ndarray:
    mask = np.zeros((_GEN_TILE_SIZE, _GEN_TILE_SIZE), dtype=bool)
    chosen = _GEN_RNG.choice(_GEN_TILE_CELLS, size=count, replace=False)
    mask.flat[chosen] = True
    return mask


def _gen_colored_tile(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.full(mask.shape, _GEN_BG, dtype=int)
    tile[mask] = color
    return tile


def _gen_attempt() -> np.ndarray | None:
    tile_count = int(_GEN_RNG.integers(*_GEN_TILE_COUNT_BOUNDS))
    horizontal = bool(_GEN_RNG.integers(0, _GEN_ORIENTATION_CHOICES))
    loser_count = int(_GEN_RNG.integers(*_GEN_LOSER_COUNT_BOUNDS))
    winner_low = max(_GEN_WINNER_MIN_COUNT, loser_count + 1)
    winner_count = int(_GEN_RNG.integers(winner_low, _GEN_WINNER_MAX_COUNT))
    loser_mask = _gen_mask_with_count(loser_count)
    winner_mask = _gen_mask_with_count(winner_count)

    colors = _GEN_RNG.choice(_GEN_COLORS, size=tile_count, replace=False)
    winner_pos = int(_GEN_RNG.integers(0, tile_count))

    tiles = []
    winner_tile = None
    for idx, color in enumerate(colors):
        if idx == winner_pos:
            winner_tile = _gen_colored_tile(winner_mask, int(color))
            tiles.append(winner_tile)
        else:
            tiles.append(_gen_colored_tile(loser_mask, int(color)))

    assert winner_tile is not None
    grid = np.concatenate(tiles, axis=1 if horizontal else 0)

    if np.array_equal(main(grid), winner_tile):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
