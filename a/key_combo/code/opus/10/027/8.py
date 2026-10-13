from __future__ import annotations

import itertools
import numpy as np


MARKER = 4
SEPARATOR = 5
BACKGROUND = 0
PATTERN_COLORS = (1, 2)
SIZE_CHOICES = (4, 5, 6)
AREA_RANGE = (4, 9)  # [lo, hi)
MAX_GENERATION_ATTEMPTS = 500


def _rotate_cw(tile: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(tile, -int(turns))


def _separator_index(g: np.ndarray, color: int, axis: int) -> int:
    full = np.all(g == color, axis=axis)
    idx = np.flatnonzero(full)
    assert idx.size == 1, "expected exactly one full separator line on each axis"
    return int(idx[0])


def _split_tiles(g: np.ndarray) -> tuple[int, list[np.ndarray]]:
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"
    sep_r = _separator_index(g, SEPARATOR, axis=1)
    sep_c = _separator_index(g, SEPARATOR, axis=0)
    assert sep_r == sep_c, "separator cross must be centered"
    n = sep_r
    assert h == 2 * n + 1 and w == 2 * n + 1, "grid must be (2N+1)x(2N+1)"
    assert 4 <= n <= 14, "tile size must be at least 4"
    masks = np.ones_like(g, dtype=bool)
    masks[sep_r, :] = False
    masks[:, sep_c] = False
    assert not np.any(g[masks] == SEPARATOR), "separator color must only appear on the cross"
    tiles = [
        g[:n, :n],
        g[:n, n + 1 :],
        g[n + 1 :, :n],
        g[n + 1 :, n + 1 :],
    ]
    assert all(tile.shape == (n, n) for tile in tiles)
    return n, tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use marker counts inside grey-separated tiles to decide each tile's rotation before assembling the final mosaic.

    Input structure:
    - A square (2N+1)x(2N+1) grid with one full grey row and one full grey column crossing at the center, splitting the canvas into four NxN tiles.
    - Each tile contains a blue/red foreground pattern on black plus 0-3 yellow marker pixels placed on otherwise black cells.

    Rule:
    - Identify the four tiles using the grey separators.
    - For each tile, count its yellow pixels K, erase those yellow pixels to black, and rotate the cleaned tile K quarter-turns clockwise.
    - Reassemble the four rotated tiles in their original 2x2 positions without the grey separators.

    Key insight:
    - The yellow markers are metadata rather than picture content: only their count matters, and it encodes the tile's rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _n, tiles = _split_tiles(g)
    counts = [int(np.count_nonzero(tile == MARKER)) for tile in tiles]
    assert sorted(counts) == [0, 1, 2, 3], "expected marker counts to be a permutation of {0,1,2,3}"

    rotated = []
    for tile, turns in zip(tiles, counts):
        cleaned = tile.copy()
        cleaned[cleaned == MARKER] = BACKGROUND
        rotated.append(_rotate_cw(cleaned, turns))

    return np.block([[rotated[0], rotated[1]], [rotated[2], rotated[3]]]).astype(int)


def _grow_mask(n: int, area: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    seed = (int(rng.integers(0, n)), int(rng.integers(0, n)))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < n and 0 <= nc < n and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier
        nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _paint_pattern(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(coords) >= 4
    red_count = int(rng.integers(1, len(coords)))
    red_indices = set(int(x) for x in rng.choice(len(coords), size=red_count, replace=False))
    tile = np.zeros(mask.shape, dtype=int)
    for idx, (r, c) in enumerate(coords):
        tile[r, c] = PATTERN_COLORS[1] if idx in red_indices else PATTERN_COLORS[0]
    assert np.any(tile == PATTERN_COLORS[0]) and np.any(tile == PATTERN_COLORS[1])
    return tile


def _all_rotations_distinct(tile: np.ndarray) -> bool:
    variants = [_rotate_cw(tile, k) for k in range(4)]
    return len({variant.tobytes() for variant in variants}) == 4


def _generate_base_tile(n: int, rng: np.random.Generator) -> np.ndarray | None:
    area = int(rng.integers(*AREA_RANGE))
    mask = _grow_mask(n, area, rng)
    tile = _paint_pattern(mask, rng)
    if not _all_rotations_distinct(tile):
        return None
    if np.count_nonzero(tile == BACKGROUND) < 3:
        return None
    return tile


def _valid_marker_cells(base_tile: np.ndarray) -> list[tuple[int, int]]:
    n = int(base_tile.shape[0])
    cells = []
    center = (n // 2, n // 2) if n % 2 == 1 else None
    for r, c in itertools.product(range(n), repeat=2):
        if base_tile[r, c] != BACKGROUND:
            continue
        if center is not None and (r, c) == center:
            continue
        cells.append((r, c))
    return cells


def _build_input_from_base(
    base_tile: np.ndarray,
    counts: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    n = int(base_tile.shape[0])
    valid_cells = _valid_marker_cells(base_tile)
    if len(valid_cells) < 3:
        return None

    tiles: list[np.ndarray] = []
    for count in counts:
        tile = base_tile.copy()
        if count:
            chosen = rng.choice(len(valid_cells), size=count, replace=False)
            for idx in np.atleast_1d(chosen):
                r, c = valid_cells[int(idx)]
                tile[r, c] = MARKER
        tiles.append(tile)

    sep = np.full((n, 1), SEPARATOR, dtype=int)
    row_sep = np.full((1, 2 * n + 1), SEPARATOR, dtype=int)
    top = np.hstack([tiles[0], sep, tiles[1]])
    bottom = np.hstack([tiles[2], sep.copy(), tiles[3]])
    return np.vstack([top, row_sep, bottom]).astype(int)


def _same_clean_pattern(inp: np.ndarray) -> bool:
    _n, tiles = _split_tiles(inp)
    cleaned = []
    for tile in tiles:
        cur = tile.copy()
        cur[cur == MARKER] = BACKGROUND
        cleaned.append(cur)
    first = cleaned[0]
    return all(np.array_equal(first, other) for other in cleaned[1:])


def _hardcoded_position_output(base_tile: np.ndarray, counts_by_position: tuple[int, int, int, int]) -> np.ndarray:
    rotated = [_rotate_cw(base_tile, k) for k in counts_by_position]
    return np.block([[rotated[0], rotated[1]], [rotated[2], rotated[3]]]).astype(int)


def _semantic_ok(inp: np.ndarray) -> bool:
    n, tiles = _split_tiles(inp)
    if n not in SIZE_CHOICES:
        return False
    counts = tuple(int(np.count_nonzero(tile == MARKER)) for tile in tiles)
    if sorted(counts) != [0, 1, 2, 3]:
        return False
    if not _same_clean_pattern(inp):
        return False

    base_tile = tiles[0].copy()
    base_tile[base_tile == MARKER] = BACKGROUND
    if not _all_rotations_distinct(base_tile):
        return False

    out = main(inp)
    if out.shape != (2 * n, 2 * n):
        return False

    ccw_like = np.block(
        [
            [np.rot90(base_tile, counts[0]), np.rot90(base_tile, counts[1])],
            [np.rot90(base_tile, counts[2]), np.rot90(base_tile, counts[3])],
        ]
    ).astype(int)
    if np.array_equal(out, ccw_like):
        return False

    for fixed_counts in ((0, 1, 3, 2), (0, 3, 1, 2)):
        if np.array_equal(out, _hardcoded_position_output(base_tile, fixed_counts)):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice(np.array(SIZE_CHOICES)))
    base_tile = _generate_base_tile(n, rng)
    if base_tile is None:
        return None

    counts = tuple(int(x) for x in rng.permutation(np.array([0, 1, 2, 3])))
    if counts in ((0, 1, 3, 2), (0, 3, 1, 2)):
        return None

    inp = _build_input_from_base(base_tile, counts, rng)
    if inp is None:
        return None
    if not _semantic_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-count rotation puzzle")
