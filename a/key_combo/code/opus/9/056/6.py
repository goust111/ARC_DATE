from __future__ import annotations

import itertools

import numpy as np


GRID_SIZE = 11
TILE_SIZE = 3
MACRO_SIZE = 3
TILE_STRIDE = 4
SEPARATOR_COLOR = 5
BACKGROUND_COLOR = 0
MARKER_COLOR = 4
SEPARATOR_LINES = (3, 7)
PALETTE_COLORS = (1, 2, 3, 6, 7, 8, 9)
SLOTS = tuple((r, c) for r in range(MACRO_SIZE) for c in range(MACRO_SIZE))
LOCAL_POSITIONS = tuple((r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE) if (r, c) != (1, 1))
GEN_MAX_ATTEMPTS = 300

_RNG = np.random.default_rng()
_UNUSED_OUTPUT_SPECS: list[tuple[tuple[int, int], int, int]] = []


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    starts = [0] + [sep + 1 for sep in separators]
    ends = separators + [size]
    return list(zip(starts, ends))


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    hits = [index for index, (lo, hi) in enumerate(bands) if lo <= value < hi]
    assert len(hits) == 1
    return hits[0]


def _tile_slice(br: int, bc: int) -> tuple[slice, slice]:
    r0 = br * TILE_STRIDE
    c0 = bc * TILE_STRIDE
    return slice(r0, r0 + TILE_SIZE), slice(c0, c0 + TILE_SIZE)


def _tile_distances(center: tuple[int, int]) -> np.ndarray:
    rr = np.arange(TILE_SIZE)[:, None]
    cc = np.arange(TILE_SIZE)[None, :]
    return np.maximum(np.abs(rr - center[0]), np.abs(cc - center[1]))


def _distance_cells(center: tuple[int, int], dist: int) -> list[tuple[int, int]]:
    distances = _tile_distances(center)
    cells = np.argwhere(distances == dist)
    return sorted((int(r), int(c)) for r, c in cells)


def _canonical_ring_sample(center: tuple[int, int], dist: int) -> tuple[int, int]:
    cells = _distance_cells(center, dist)
    assert cells
    return cells[0]


def _empty_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    grid[list(SEPARATOR_LINES), :] = SEPARATOR_COLOR
    grid[:, list(SEPARATOR_LINES)] = SEPARATOR_COLOR
    return grid


def _build_source_tile(local_pos: tuple[int, int], ring1_color: int, ring2_color: int) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), BACKGROUND_COLOR, dtype=int)
    tile[local_pos] = MARKER_COLOR
    tile[_canonical_ring_sample(local_pos, 1)] = ring1_color
    tile[_canonical_ring_sample(local_pos, 2)] = ring2_color
    return tile


def _build_ring_tile(local_pos: tuple[int, int], ring1_color: int, ring2_color: int) -> np.ndarray:
    palette = {0: MARKER_COLOR, 1: ring1_color, 2: ring2_color}
    distances = _tile_distances(local_pos)
    out = np.empty((TILE_SIZE, TILE_SIZE), dtype=int)
    for dist, color in palette.items():
        out[distances == dist] = color
    return out


def _build_expected_output(local_pos: tuple[int, int], ring1_color: int, ring2_color: int) -> np.ndarray:
    out = _empty_grid()
    rs, cs = _tile_slice(*local_pos)
    out[rs, cs] = _build_ring_tile(local_pos, ring1_color, ring2_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move the yellow-marked source tile into the slot named by the marker and expand its distance palette into concentric square rings.

    Input structure
    The input is an 11x11 grid split by full grey separator rows and columns into a
    3x3 macro grid of 3x3 tiles. Exactly one tile contains all non-black content,
    including a single yellow marker pixel; every other tile is black.

    Rule
    Read the yellow marker's local coordinates (r, c) inside the source tile. Those
    same coordinates choose the destination macro slot and also serve as the center
    of a 3x3 Chebyshev-ring fill. In the source tile, the unique non-black pixel at
    distance 1 from the marker gives the color for ring 1, and the unique non-black
    pixel at distance 2 gives the color for ring 2. Clear the source tile, preserve
    the grey separators, and fill the destination tile so every cell gets the color
    for its Chebyshev distance from (r, c).

    Key insight
    The yellow marker simultaneously identifies the source tile, names the target
    slot, and fixes the geometric center of the output ring pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)

    row_separators = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR_COLOR)]
    col_separators = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR_COLOR)]
    assert row_separators == list(SEPARATOR_LINES)
    assert col_separators == list(SEPARATOR_LINES)

    row_bands = _bands(row_separators, g.shape[0])
    col_bands = _bands(col_separators, g.shape[1])
    assert row_bands == col_bands == [(0, 3), (4, 7), (8, 11)]

    markers = np.argwhere(g == MARKER_COLOR)
    assert markers.shape == (1, 2)
    marker_r, marker_c = map(int, markers[0])

    src_br = _band_index(marker_r, row_bands)
    src_bc = _band_index(marker_c, col_bands)
    src_rs, src_cs = _tile_slice(src_br, src_bc)
    source_tile = g[src_rs, src_cs].copy()

    non_separator = g.copy()
    non_separator[list(SEPARATOR_LINES), :] = BACKGROUND_COLOR
    non_separator[:, list(SEPARATOR_LINES)] = BACKGROUND_COLOR
    source_mask = np.zeros_like(non_separator, dtype=bool)
    source_mask[src_rs, src_cs] = True
    assert np.all(non_separator[~source_mask] == BACKGROUND_COLOR)

    local_pos = (marker_r - src_rs.start, marker_c - src_cs.start)
    assert local_pos in LOCAL_POSITIONS

    distances = _tile_distances(local_pos)
    nonblack_positions = np.argwhere(source_tile != BACKGROUND_COLOR)
    assert nonblack_positions.shape[0] == 3

    palette: dict[int, int] = {}
    for dist in (1, 2):
        cells = np.argwhere((distances == dist) & (source_tile != BACKGROUND_COLOR))
        assert cells.shape == (1, 2)
        rr, cc = map(int, cells[0])
        color = int(source_tile[rr, cc])
        assert color not in (BACKGROUND_COLOR, MARKER_COLOR, SEPARATOR_COLOR)
        palette[dist] = color

    assert palette[1] != palette[2]
    assert int(source_tile[local_pos]) == MARKER_COLOR

    out = _empty_grid()
    dst_rs, dst_cs = _tile_slice(*local_pos)
    out[dst_rs, dst_cs] = _build_ring_tile(local_pos, palette[1], palette[2])
    return out


def _init_output_specs() -> None:
    if _UNUSED_OUTPUT_SPECS:
        return
    specs = [
        (local_pos, ring1_color, ring2_color)
        for local_pos in LOCAL_POSITIONS
        for ring1_color, ring2_color in itertools.permutations(PALETTE_COLORS, 2)
    ]
    order = _RNG.permutation(len(specs))
    _UNUSED_OUTPUT_SPECS.extend(specs[index] for index in order)


def _next_output_spec() -> tuple[tuple[int, int], int, int]:
    _init_output_specs()
    assert _UNUSED_OUTPUT_SPECS, "output-spec pool exhausted"
    return _UNUSED_OUTPUT_SPECS.pop()


def _validate_generated_case(
    grid: np.ndarray,
    expected: np.ndarray,
    source_slot: tuple[int, int],
    local_pos: tuple[int, int],
    ring1_color: int,
    ring2_color: int,
) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not np.all(grid[list(SEPARATOR_LINES), :] == SEPARATOR_COLOR):
        return False
    if not np.all(grid[:, list(SEPARATOR_LINES)] == SEPARATOR_COLOR):
        return False

    src_rs, src_cs = _tile_slice(*source_slot)
    expected_source = _build_source_tile(local_pos, ring1_color, ring2_color)
    if not np.array_equal(grid[src_rs, src_cs], expected_source):
        return False

    for slot in SLOTS:
        if slot == source_slot:
            continue
        rs, cs = _tile_slice(*slot)
        if not np.all(grid[rs, cs] == BACKGROUND_COLOR):
            return False

    source_tile = grid[src_rs, src_cs]
    distances = _tile_distances(local_pos)
    if int(np.count_nonzero(source_tile == MARKER_COLOR)) != 1:
        return False
    if int(np.count_nonzero(source_tile != BACKGROUND_COLOR)) != 3:
        return False
    if int(np.count_nonzero((distances == 1) & (source_tile != BACKGROUND_COLOR))) != 1:
        return False
    if int(np.count_nonzero((distances == 2) & (source_tile != BACKGROUND_COLOR))) != 1:
        return False
    if source_slot == local_pos:
        return False
    if not np.array_equal(expected, _build_expected_output(local_pos, ring1_color, ring2_color)):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    local_pos, ring1_color, ring2_color = _next_output_spec()
    target_slot = local_pos
    source_choices = [slot for slot in SLOTS if slot != target_slot]
    source_slot = source_choices[int(_RNG.integers(0, len(source_choices)))]

    grid = _empty_grid()
    src_rs, src_cs = _tile_slice(*source_slot)
    grid[src_rs, src_cs] = _build_source_tile(local_pos, ring1_color, ring2_color)
    expected = _build_expected_output(local_pos, ring1_color, ring2_color)

    if _validate_generated_case(grid, expected, source_slot, local_pos, ring1_color, ring2_color):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
