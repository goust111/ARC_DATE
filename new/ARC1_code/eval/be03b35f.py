from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(map(int, mask.astype(int).ravel()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the corner tile mask that is missing from the rotation family shown in the input.

    Input structure
    - The grid is an odd-sized square split by a one-pixel background cross into four square corner tiles.
    - Three corner tiles contain masks made from one source color on the background.
    - The remaining corner tile is a solid non-background placeholder color.

    Rule
    - Read the three source masks from the corner tiles.
    - Generate the four rotations of that mask family.
    - Output the unique rotated mask that is not one of the three source masks, using the source color on the background.

    Key insight
    The placeholder marks the missing fourth member of a rotation set; the answer is that absent tile, not the placeholder itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1 and h > 1, "expected an odd square grid"

    n = h // 2
    bg = get_bg(g)
    assert np.all(g[n, :] == bg) and np.all(g[:, n] == bg), "center cross must be background"

    blocks = [
        g[:n, :n],
        g[:n, n + 1 :],
        g[n + 1 :, :n],
        g[n + 1 :, n + 1 :],
    ]
    assert all(block.shape == (n, n) for block in blocks), "expected four equal square corner tiles"

    color_block_counts: dict[int, int] = {}
    for block in blocks:
        for color in colors_present(block, exclude={bg}):
            color_block_counts[color] = color_block_counts.get(color, 0) + 1
    source_colors = [color for color, count in color_block_counts.items() if count == 3]
    assert len(source_colors) == 1, "expected exactly one color shared by the three source tiles"
    source_color = source_colors[0]

    source_masks = []
    placeholder_count = 0
    for block in blocks:
        colors = set(map(int, np.unique(block)))
        if colors <= {bg, source_color} and source_color in colors:
            source_masks.append(block == source_color)
        else:
            assert len(colors) == 1 and bg not in colors and source_color not in colors, (
                "placeholder tile must be solid and separate from background/source"
            )
            placeholder_count += 1

    assert len(source_masks) == 3 and placeholder_count == 1, "expected three source masks and one placeholder"
    assert len({int(mask.sum()) for mask in source_masks}) == 1, "source masks must have equal area"

    rotation_keys = {
        _mask_key(np.rot90(mask, turns))
        for mask in source_masks
        for turns in range(4)
    }
    seen_keys = {_mask_key(mask) for mask in source_masks}
    missing_keys = rotation_keys - seen_keys
    assert len(rotation_keys) == 4 and len(seen_keys) == 3, "expected a four-member rotation family"
    assert len(missing_keys) == 1, "expected exactly one missing rotated mask"

    missing_mask = np.array(next(iter(missing_keys)), dtype=bool).reshape(n, n)
    out = np.full((n, n), bg, dtype=int)
    out[missing_mask] = source_color
    return out

# --------------- generator (o5x) ---------------

# CFG: tile sizes/areas, fixed colors, rotation checks, retries, quadrant order.
TILE_SIZE_RANGE = (2, 5)
MAX_SOURCE_PIXELS_BY_SIZE = {2: 3, 3: 6, 4: 10}
BG_COLOR, SOURCE_COLOR, PLACEHOLDER_COLOR = 0, 1, 2
NUM_ROTATIONS = 4
RETRY_LIMIT = 300
QUADRANT_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _gen_render_source(mask: np.ndarray) -> np.ndarray:
    tile = np.full(mask.shape, BG_COLOR, dtype=int)
    tile[mask] = SOURCE_COLOR
    return tile


def _gen_place_tile(grid: np.ndarray, tile: np.ndarray, quad: int, n: int) -> None:
    row_band, col_band = QUADRANT_OFFSETS[quad]
    row_start = row_band * (n + 1)
    col_start = col_band * (n + 1)
    grid[row_start : row_start + n, col_start : col_start + n] = tile


def _gen_random_source_mask(n: int) -> np.ndarray:
    max_pixels = MAX_SOURCE_PIXELS_BY_SIZE[n]
    pixel_count = int(np.random.randint(1, max_pixels + 1))
    mask = np.zeros((n, n), dtype=bool)
    chosen = np.random.choice(n * n, size=pixel_count, replace=False)
    mask.ravel()[chosen] = True
    return mask


def _gen_rotation_family(n: int) -> list[np.ndarray] | None:
    mask = _gen_random_source_mask(n)
    rotations = [np.rot90(mask, k) for k in range(NUM_ROTATIONS)]
    if len({_mask_key(rot) for rot in rotations}) != NUM_ROTATIONS:
        return None
    return rotations


def _gen_build_grid(n: int, quad_masks: list[np.ndarray], placeholder_quad: int) -> np.ndarray:
    size = 2 * n + 1
    grid = np.full((size, size), BG_COLOR, dtype=int)
    placeholder_tile = np.full((n, n), PLACEHOLDER_COLOR, dtype=int)
    for quad, quad_mask in enumerate(quad_masks):
        tile = placeholder_tile if quad == placeholder_quad else _gen_render_source(quad_mask)
        _gen_place_tile(grid, tile, quad, n)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*TILE_SIZE_RANGE))
    rotations = _gen_rotation_family(n)
    if rotations is None:
        return None

    perm = np.random.permutation(NUM_ROTATIONS)
    quad_masks = [rotations[int(i)] for i in perm]
    placeholder_quad = int(np.random.randint(0, NUM_ROTATIONS))

    expected = _gen_render_source(quad_masks[placeholder_quad])
    grid = _gen_build_grid(n, quad_masks, placeholder_quad)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid be03b35f input")
