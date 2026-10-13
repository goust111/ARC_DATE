import numpy as np

from arc_utils import Color
import random


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    starts = [0] + [s + 1 for s in separators]
    ends = separators + [size]
    bands = list(zip(starts, ends))
    assert all(lo < hi for lo, hi in bands)
    return bands


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    hits = [i for i, (lo, hi) in enumerate(bands) if lo <= value < hi]
    assert len(hits) == 1
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the yellow-marked tile to the larger-grid slot named by the yellow pixel's local position.

    Input structure
    The input is a square canvas partitioned by full grey separator rows and
    columns into a 3x3 array of equal 3x3 tiles.  Non-separator cells are black
    background or colored foreground, with exactly one yellow marker pixel.

    Rule
    Find the tile containing the yellow marker.  Copy that entire 3x3 tile to
    the tile whose macro row and column are the marker's local row and column
    inside the source tile.  Clear all other non-separator cells to black and
    preserve the grey separator grid.

    Key insight
    The yellow pixel has two roles: it identifies the source sprite and, by its
    local coordinate within that sprite, encodes the destination slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_color = Color.GREY
    bg = Color.BLACK
    h, w = g.shape
    assert h == w
    row_seps = [r for r in range(h) if np.all(g[r, :] == sep_color)]
    col_seps = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(row_seps) == len(col_seps) == 2

    row_bands = _bands(row_seps, h)
    col_bands = _bands(col_seps, w)
    assert len(row_bands) == len(col_bands) == 3

    tile_heights = [hi - lo for lo, hi in row_bands]
    tile_widths = [hi - lo for lo, hi in col_bands]
    assert len(set(tile_heights)) == len(set(tile_widths)) == 1
    tile_h = tile_heights[0]
    tile_w = tile_widths[0]
    assert tile_h == len(row_bands)
    assert tile_w == len(col_bands)

    non_separator_mask = np.ones_like(g, dtype=bool)
    non_separator_mask[row_seps, :] = False
    non_separator_mask[:, col_seps] = False
    assert not np.any(g[non_separator_mask] == sep_color)

    markers = np.argwhere(g == Color.YELLOW)
    assert markers.shape == (1, 2)
    marker_r, marker_c = map(int, markers[0])

    src_br = _band_index(marker_r, row_bands)
    src_bc = _band_index(marker_c, col_bands)
    src_r0, src_r1 = row_bands[src_br]
    src_c0, src_c1 = col_bands[src_bc]

    dst_br = marker_r - src_r0
    dst_bc = marker_c - src_c0
    assert 0 <= dst_br < len(row_bands)
    assert 0 <= dst_bc < len(col_bands)

    dst_r0, dst_r1 = row_bands[dst_br]
    dst_c0, dst_c1 = col_bands[dst_bc]

    out = np.full_like(g, bg)
    out[row_seps, :] = sep_color
    out[:, col_seps] = sep_color
    out[dst_r0:dst_r1, dst_c0:dst_c1] = g[src_r0:src_r1, src_c0:src_c1]
    return out

# --------------- generator (o5x) ---------------
# Canvas layout and colors.
GRID_SIZE = 11
TILE_SIZE = 3
MACRO_SIZE = 3
TILE_STRIDE = 4
SEPARATOR_LINES = (3, 7)
BACKGROUND_COLOR, SEPARATOR_COLOR, MARKER_COLOR = 0, 5, 4
# Tile sampling ranges and palettes.
TILE_COLORS = [1, 2, 3, 6, 7, 8, 9]
TILE_POSITIONS = [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)]
MACRO_SLOTS = [(br, bc) for br in range(MACRO_SIZE) for bc in range(MACRO_SIZE)]
FILLER_COUNT_RANGE = (1, 4)
SOURCE_COUNT_RANGE = (3, 4)
# Retry budget for oracle-checked generation.
GEN_RETRIES = 300


def _gen_empty_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    grid[list(SEPARATOR_LINES), :] = SEPARATOR_COLOR
    grid[:, list(SEPARATOR_LINES)] = SEPARATOR_COLOR
    return grid


def _gen_tile_slice(br: int, bc: int) -> tuple[slice, slice]:
    r0 = br * TILE_STRIDE
    c0 = bc * TILE_STRIDE
    return slice(r0, r0 + TILE_SIZE), slice(c0, c0 + TILE_SIZE)


def _gen_random_tile(*, marker_pos: tuple[int, int] | None = None) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), BACKGROUND_COLOR, dtype=int)

    if marker_pos is None:
        paint_count = random.randint(*FILLER_COUNT_RANGE)
        available_positions = TILE_POSITIONS
    else:
        source_count = random.randint(*SOURCE_COUNT_RANGE)
        tile[marker_pos] = MARKER_COLOR
        paint_count = source_count - 1
        available_positions = [p for p in TILE_POSITIONS if p != marker_pos]

    chosen_positions = random.sample(available_positions, paint_count)
    chosen_colors = random.sample(TILE_COLORS, paint_count)
    for (r, c), color in zip(chosen_positions, chosen_colors):
        tile[r, c] = color
    return tile


def _gen_attempt() -> np.ndarray | None:
    src_br = random.randrange(MACRO_SIZE)
    src_bc = random.randrange(MACRO_SIZE)
    destination_slots = [slot for slot in MACRO_SLOTS if slot != (src_br, src_bc)]
    dst_br, dst_bc = random.choice(destination_slots)

    output = _gen_empty_grid()
    copied_tile = _gen_random_tile(marker_pos=(dst_br, dst_bc))
    rs, cs = _gen_tile_slice(dst_br, dst_bc)
    output[rs, cs] = copied_tile

    grid = _gen_empty_grid()
    for br in range(MACRO_SIZE):
        for bc in range(MACRO_SIZE):
            rs, cs = _gen_tile_slice(br, bc)
            if (br, bc) == (src_br, src_bc):
                grid[rs, cs] = copied_tile
            else:
                grid[rs, cs] = _gen_random_tile()

    if np.array_equal(main(grid), output):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6d0160f0 input")
