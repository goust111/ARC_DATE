import random

import numpy as np


AZURE = 8
MAROON = 9
NORMAL_COLORS = tuple(color for color in range(10) if color not in {AZURE, MAROON})
GENERATION_ATTEMPTS = 400
SIZE_RANGE = (4, 10)


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if 0 < r < h - 1 and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if 0 < c < w - 1 and bool(np.all(grid[:, c] == grid[0, c]))
    ]

    matches = []
    for r, row_color in row_candidates:
        for c, col_color in col_candidates:
            if row_color == col_color and int(grid[r, c]) == row_color:
                matches.append((r, c, row_color))

    assert len(matches) == 1, "expected exactly one full-row/full-column separator cross"
    return matches[0]


def _quadrant_slices(shape: tuple[int, int], sep_r: int, sep_c: int) -> list[tuple[slice, slice]]:
    h, w = shape
    return [
        (slice(0, sep_r), slice(0, sep_c)),
        (slice(0, sep_r), slice(sep_c + 1, w)),
        (slice(sep_r + 1, h), slice(0, sep_c)),
        (slice(sep_r + 1, h), slice(sep_c + 1, w)),
    ]


def _infer_tile(block: np.ndarray) -> np.ndarray:
    tile = np.full((2, 2), -1, dtype=int)
    for pr in range(2):
        for pc in range(2):
            values = {
                int(block[r, c])
                for r in range(pr, block.shape[0], 2)
                for c in range(pc, block.shape[1], 2)
                if int(block[r, c]) != MAROON
            }
            assert values, "each parity class needs at least one surviving pixel"
            assert len(values) == 1, "surviving pixels in a parity class must agree"
            tile[pr, pc] = values.pop()

    assert len(set(map(int, tile.flat))) == 2, "the periodic tile must use exactly two colors"
    return tile


def _repair_block(block: np.ndarray) -> np.ndarray:
    tile = _infer_tile(block)
    repaired = block.copy()
    maroon_cells = np.argwhere(repaired == MAROON)
    for r0, c0 in maroon_cells:
        r = int(r0)
        c = int(c0)
        repaired[r, c] = int(tile[r % 2, c % 2])
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the maroon-corrupted quadrant by inferring its underlying 2x2 periodic tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_r, sep_c, _cross_color = _find_cross(g)
    quadrants = _quadrant_slices(g.shape, sep_r, sep_c)
    anomaly_ids = [
        idx
        for idx, (rs, cs) in enumerate(quadrants)
        if bool(np.any(g[rs, cs] == MAROON))
    ]
    assert len(anomaly_ids) == 1, "expected exactly one corrupted quadrant"

    out = g.copy()
    anomaly = anomaly_ids[0]
    rs, cs = quadrants[anomaly]
    out[rs, cs] = _repair_block(g[rs, cs])
    return out


def _make_tile(color_a: int, color_b: int, kind: int) -> np.ndarray:
    if kind == 0:
        return np.array([[color_a, color_b], [color_b, color_a]], dtype=int)
    if kind == 1:
        return np.array([[color_a, color_b], [color_a, color_b]], dtype=int)
    if kind == 2:
        return np.array([[color_a, color_a], [color_b, color_b]], dtype=int)
    raise AssertionError(f"unsupported tile kind: {kind}")


def _tile_fill(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % 2
    cols = np.arange(width)[None, :] % 2
    return tile[rows, cols]


def _choose_tiles() -> list[np.ndarray]:
    colors = list(NORMAL_COLORS)
    random.shuffle(colors)
    tiles = []
    for idx in range(4):
        a = colors[2 * idx]
        b = colors[2 * idx + 1]
        kind = random.randrange(3)
        if random.random() < 0.5:
            a, b = b, a
        tiles.append(_make_tile(a, b, kind))
    return tiles


def _protected_cells(height: int, width: int) -> set[tuple[int, int]]:
    protected = set()
    for pr in range(2):
        for pc in range(2):
            cells = [
                (r, c)
                for r in range(pr, height, 2)
                for c in range(pc, width, 2)
            ]
            assert cells, "quadrant must expose all 2x2 parity classes"
            protected.add(random.choice(cells))
    return protected


def _corrupt_quadrant(block: np.ndarray) -> np.ndarray | None:
    height, width = block.shape
    protected = _protected_cells(height, width)
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in protected
    ]
    if not candidates:
        return None

    max_corrupt = min(len(candidates), max(1, block.size // 2))
    count = random.randint(1, max_corrupt)
    chosen = random.sample(candidates, count)

    corrupted = block.copy()
    for r, c in chosen:
        corrupted[r, c] = MAROON
    return corrupted


def _gen_attempt() -> np.ndarray | None:
    top = random.randint(*SIZE_RANGE)
    bottom = random.randint(*SIZE_RANGE)
    left = random.randint(*SIZE_RANGE)
    right = random.randint(*SIZE_RANGE)

    h = top + bottom + 1
    w = left + right + 1
    clean = np.full((h, w), AZURE, dtype=int)

    sep_r = top
    sep_c = left
    quadrants = _quadrant_slices(clean.shape, sep_r, sep_c)
    quadrant_sizes = [(top, left), (top, right), (bottom, left), (bottom, right)]
    tiles = _choose_tiles()

    for idx, ((rs, cs), (qh, qw)) in enumerate(zip(quadrants, quadrant_sizes)):
        clean[rs, cs] = _tile_fill(tiles[idx], qh, qw)

    anomaly = random.randrange(4)
    inp = clean.copy()
    rs, cs = quadrants[anomaly]
    corrupted = _corrupt_quadrant(inp[rs, cs])
    if corrupted is None:
        return None
    inp[rs, cs] = corrupted

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
