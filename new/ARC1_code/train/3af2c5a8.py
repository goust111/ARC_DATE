import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the entire input across both axes to form a two-by-two reflected pattern.

    Input structure:
    - A 3x4 grid containing a background color and one foreground pattern color.

    Rule:
    - Place the original grid in the upper-left quadrant of a 6x8 output.
    - Place its horizontal reflection in the upper-right quadrant.
    - Place its vertical reflection in the lower-left quadrant.
    - Place its 180-degree reflection in the lower-right quadrant.

    Key insight:
    - The output is not object-based; the whole grid is the tile, and the larger grid is formed by reflecting that tile across the central row and column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 4), "expected the observed 3x4 source tile"
    assert np.unique(g).size == 2, "expected one background color and one pattern color"

    top = np.concatenate((g, np.fliplr(g)), axis=1)
    bottom = np.concatenate((np.flipud(g), np.flipud(np.fliplr(g))), axis=1)
    return np.concatenate((top, bottom), axis=0)

# --------------- generator (o5x) ---------------
# Canvas and palette bounds.
_GEN_SHAPE_3AF2C5A8 = (3, 4)
_GEN_COLOR_COUNT_3AF2C5A8 = 10

# Foreground density and LR-asymmetry controls.
_GEN_FG_COUNTS_3AF2C5A8 = (4, 5, 6)
_GEN_LR_PAIRS_3AF2C5A8 = (
    ((0, 0), (0, 3)),
    ((0, 1), (0, 2)),
    ((1, 0), (1, 3)),
    ((1, 1), (1, 2)),
    ((2, 0), (2, 3)),
    ((2, 1), (2, 2)),
)

# Retry budget for the solver-oracle sanity check.
_GEN_RETRY_LIMIT_3AF2C5A8 = 300


def _gen_lr_asymmetric_mask_3af2c5a8(fg_count: int) -> np.ndarray:
    pair_count = len(_GEN_LR_PAIRS_3AF2C5A8)
    single_options = [
        s
        for s in range(1, min(pair_count, fg_count) + 1)
        if (fg_count - s) % 2 == 0 and (fg_count - s) // 2 <= pair_count - s
    ]
    single_count = int(np.random.choice(single_options))
    double_count = (fg_count - single_count) // 2
    pair_order = np.random.permutation(pair_count)
    mask = np.zeros(_GEN_SHAPE_3AF2C5A8, dtype=bool)

    for pair_idx in pair_order[:single_count]:
        pair = _GEN_LR_PAIRS_3AF2C5A8[int(pair_idx)]
        r, c = pair[int(np.random.randint(2))]
        mask[r, c] = True
    for pair_idx in pair_order[single_count : single_count + double_count]:
        for r, c in _GEN_LR_PAIRS_3AF2C5A8[int(pair_idx)]:
            mask[r, c] = True
    return mask


def _gen_tile_3af2c5a8() -> np.ndarray:
    bg, fg = np.random.choice(_GEN_COLOR_COUNT_3AF2C5A8, size=2, replace=False)
    tile = np.full(_GEN_SHAPE_3AF2C5A8, bg, dtype=int)
    fg_count = int(np.random.choice(_GEN_FG_COUNTS_3AF2C5A8))
    tile[_gen_lr_asymmetric_mask_3af2c5a8(fg_count)] = fg
    return tile


def _gen_reflected_output_3af2c5a8(tile: np.ndarray) -> np.ndarray:
    top = np.concatenate((tile, np.fliplr(tile)), axis=1)
    bottom = np.concatenate((np.flipud(tile), np.flipud(np.fliplr(tile))), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _gen_attempt() -> np.ndarray | None:
    tile = _gen_tile_3af2c5a8()
    if not np.array_equal(main(tile), _gen_reflected_output_3af2c5a8(tile)):
        return None
    return tile


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT_3AF2C5A8):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 3af2c5a8 input")
