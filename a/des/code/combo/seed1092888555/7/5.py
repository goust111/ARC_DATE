import numpy as np


BG = 0
SEP = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
HEIGHT_OPTIONS = (4, 6, 8, 9, 10, 12)
PANEL_COUNT_RANGE = (1, 4)
SEPARATOR_WIDTH_OPTIONS = (1, 2)
TILE_WIDTH_RANGE = (2, 5)
REPEAT_X_RANGE = (2, 4)
REPEAT_Y_RANGE = (2, 4)
GEN_ATTEMPTS = 300


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _bands(length: int, sep_indices: list[int]) -> list[tuple[int, int]]:
    if not sep_indices:
        return [(0, length - 1)]
    bands = []
    start = 0
    i = 0
    seps = sorted(set(int(x) for x in sep_indices))
    while i < len(seps):
        j = i
        while j + 1 < len(seps) and seps[j + 1] == seps[j] + 1:
            j += 1
        if start <= seps[i] - 1:
            bands.append((start, seps[i] - 1))
        start = seps[j] + 1
        i = j + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    return bands


def _axis_period(arr: np.ndarray, axis: int) -> int:
    n = int(arr.shape[axis])
    for period in _divisors(n):
        idx = np.arange(n) % period
        if np.array_equal(arr, np.take(arr, idx, axis=axis)):
            return period
    raise AssertionError("unreachable")


def _minimal_tile(panel: np.ndarray) -> np.ndarray:
    ph = _axis_period(panel, axis=0)
    pw = _axis_period(panel, axis=1)
    tile = panel[:ph, :pw]
    rebuilt = np.tile(tile, (panel.shape[0] // ph, panel.shape[1] // pw))
    assert np.array_equal(rebuilt, panel)
    return tile.copy()


def _pack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    if len(tiles) == 1:
        return tiles[0].copy()
    out_h = max(int(tile.shape[0]) for tile in tiles)
    out_w = sum(int(tile.shape[1]) for tile in tiles) + len(tiles) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for tile in tiles:
        h, w = map(int, tile.shape)
        out[:h, c : c + w] = tile
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the smallest repeating unit from each yellow-separated panel."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = map(int, g.shape)
    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)

    panels = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            panel = g[r0 : r1 + 1, c0 : c1 + 1]
            if panel.size == 0:
                continue
            assert not np.any(panel == SEP)
            panels.append((c0, r0, panel))
    assert panels

    panels.sort(key=lambda item: (item[0], item[1]))
    tiles = [_minimal_tile(panel) for _, _, panel in panels]
    return _pack_tiles(tiles)


def _choose_repeat_y(rng: np.random.Generator, panel_h: int) -> int:
    options = [d for d in _divisors(panel_h) if REPEAT_Y_RANGE[0] <= d <= REPEAT_Y_RANGE[1]]
    assert options
    return int(rng.choice(options))


def _primitive_tile(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    colors = rng.choice(PALETTE, size=int(rng.integers(2, min(5, len(PALETTE) + 1))), replace=False)
    for _ in range(80):
        tile = rng.choice(colors, size=(h, w))
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _axis_period(tile, 0) != h:
            continue
        if _axis_period(tile, 1) != w:
            continue
        return tile.astype(int)
    return None


def _make_panel(rng: np.random.Generator, panel_h: int, branch: str) -> tuple[np.ndarray, np.ndarray] | None:
    if branch == "h":
        repeat_y = 1
        repeat_x = int(rng.integers(REPEAT_X_RANGE[0], REPEAT_X_RANGE[1] + 1))
    elif branch == "v":
        repeat_y = _choose_repeat_y(rng, panel_h)
        repeat_x = 1
    else:
        repeat_y = _choose_repeat_y(rng, panel_h)
        repeat_x = int(rng.integers(REPEAT_X_RANGE[0], REPEAT_X_RANGE[1] + 1))

    tile_h = panel_h // repeat_y
    tile_w = int(rng.integers(TILE_WIDTH_RANGE[0], TILE_WIDTH_RANGE[1] + 1))
    tile = _primitive_tile(rng, tile_h, tile_w)
    if tile is None:
        return None
    panel = np.tile(tile, (repeat_y, repeat_x))
    return panel.astype(int), tile


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_h = int(rng.choice(HEIGHT_OPTIONS))
    n_panels = int(rng.integers(PANEL_COUNT_RANGE[0], PANEL_COUNT_RANGE[1] + 1))
    branches = rng.choice(["h", "v", "hv"], size=n_panels, replace=True)

    panels = []
    expected_tiles = []
    for branch in branches:
        made = _make_panel(rng, panel_h, str(branch))
        if made is None:
            return None
        panel, tile = made
        panels.append(panel)
        expected_tiles.append(tile)

    sep_widths = [int(rng.choice(SEPARATOR_WIDTH_OPTIONS)) for _ in range(max(0, n_panels - 1))]
    total_w = sum(int(panel.shape[1]) for panel in panels) + sum(sep_widths)
    grid = np.full((panel_h, total_w), BG, dtype=int)

    c = 0
    for i, panel in enumerate(panels):
        pw = int(panel.shape[1])
        grid[:, c : c + pw] = panel
        c += pw
        if i < len(sep_widths):
            grid[:, c : c + sep_widths[i]] = SEP
            c += sep_widths[i]

    expected = _pack_tiles(expected_tiles)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if len(set(map(int, grid.ravel())) - {BG, SEP}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
