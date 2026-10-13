import numpy as np

from arc_utils import colors_present, get_bg


def _grid_order(side: int) -> int:
    candidates = [n for n in range(1, side + 1) if n * n + n - 1 == side]
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the separated array of one-hole tiles.

    Input structure
    A square grid has one background color and one foreground color. Its side
    length is n*n+n-1, partitioning it into an n by n array of n by n tiles
    separated by single background rows and columns. Some tiles are already
    present; each present tile is filled with foreground except for one
    background pixel at the local coordinate matching the tile's array position.

    Rule
    Keep the separator rows and columns background. For every tile position
    (tile row, tile column), fill that n by n tile with the foreground color and
    leave only the local pixel at (tile row, tile column) as background.

    Key insight
    The hole inside each tile is a coordinate label for that tile's position in
    the larger tile array, so the sparse input samples define the complete
    position-coded pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w

    n = _grid_order(h)
    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 1
    fg = foreground[0]

    out = np.full_like(g, bg)

    for i in range(1, n):
        sep = i * (n + 1) - 1
        assert np.all(g[sep, :] == bg)
        assert np.all(g[:, sep] == bg)

    for tile_r in range(n):
        for tile_c in range(n):
            r0 = tile_r * (n + 1)
            c0 = tile_c * (n + 1)
            expected = np.full((n, n), fg, dtype=int)
            expected[tile_r, tile_c] = bg

            observed = g[r0 : r0 + n, c0 : c0 + n]
            assert np.all(observed == bg) or np.array_equal(observed, expected)
            out[r0 : r0 + n, c0 : c0 + n] = expected

    return out

# --------------- generator (o5x) ---------------

# Grid geometry and palette.
GEN_N_MIN, GEN_N_MAX_EXCLUSIVE = 2, 6
GEN_SEPARATOR_WIDTH = 1
GEN_BG, GEN_COLOR_COUNT = 0, 10
GEN_FG_COLORS = [c for c in range(GEN_COLOR_COUNT) if c != GEN_BG]
# Tile subset sampling.
GEN_PRESENT_RATIO_MIN, GEN_PRESENT_RATIO_MAX = 0.4, 0.6
# Oracle validation.
GEN_MAX_ATTEMPTS = 300

_gen_rng = np.random.default_rng()


def _gen_completed_grid(n: int, bg: int, fg: int) -> np.ndarray:
    side = n * n + GEN_SEPARATOR_WIDTH * (n - 1)
    out = np.full((side, side), bg, dtype=int)
    step = n + GEN_SEPARATOR_WIDTH
    for tile_r in range(n):
        for tile_c in range(n):
            r0, c0 = tile_r * step, tile_c * step
            out[r0 : r0 + n, c0 : c0 + n] = fg
            out[r0 + tile_r, c0 + tile_c] = bg
    return out


def _gen_present_tile_indices(total_tiles: int) -> set[int]:
    """Sample the set of tile indices that remain present in the input."""
    lo = max(1, int(np.ceil(GEN_PRESENT_RATIO_MIN * total_tiles)))
    hi = min(total_tiles - 1, int(np.floor(GEN_PRESENT_RATIO_MAX * total_tiles)))
    present_count = int(_gen_rng.integers(lo, hi + 1))
    return set(
        int(x) for x in _gen_rng.choice(total_tiles, size=present_count, replace=False)
    )


def _gen_attempt() -> np.ndarray | None:
    n = int(_gen_rng.integers(GEN_N_MIN, GEN_N_MAX_EXCLUSIVE))
    bg = GEN_BG
    fg = int(_gen_rng.choice(GEN_FG_COLORS))
    out = _gen_completed_grid(n, bg, fg)
    inp = out.copy()
    total_tiles = n * n
    present = _gen_present_tile_indices(total_tiles)
    step = n + GEN_SEPARATOR_WIDTH
    for idx in range(total_tiles):
        if idx in present:
            continue
        tile_r, tile_c = divmod(idx, n)
        r0, c0 = tile_r * step, tile_c * step
        inp[r0 : r0 + n, c0 : c0 + n] = bg

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9ddd00f0 input")
