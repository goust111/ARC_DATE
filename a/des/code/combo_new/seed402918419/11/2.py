import random

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 5, 6, 7]
PANEL_DIM_OPTIONS = [6, 8, 9, 10, 12]
PANEL_COUNT_OPTIONS = [2, 3]
CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in separators:
        if start < idx:
            spans.append((start, idx))
        start = idx + 1
    if start < length:
        spans.append((start, length))
    return spans


def _factor_options(dim: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for seed_size in range(2, 5):
        for tile_count in range(2, 6):
            if seed_size * tile_count == dim:
                options.append((seed_size, tile_count))
    return options


def _extract_seed(panel: np.ndarray, corner: str, seed_h: int, seed_w: int) -> np.ndarray:
    if corner == "tl":
        return panel[:seed_h, :seed_w].copy()
    if corner == "tr":
        return panel[:seed_h, -seed_w:].copy()
    if corner == "bl":
        return panel[-seed_h:, :seed_w].copy()
    if corner == "br":
        return panel[-seed_h:, -seed_w:].copy()
    raise AssertionError(f"unknown corner {corner}")


def _corner_and_counts(panel: np.ndarray) -> tuple[str, int, int]:
    top = int(np.count_nonzero(panel[0] == AZURE))
    bottom = int(np.count_nonzero(panel[-1] == AZURE))
    left = int(np.count_nonzero(panel[:, 0] == AZURE))
    right = int(np.count_nonzero(panel[:, -1] == AZURE))

    matches: list[tuple[str, int, int]] = []
    if top > 0 and left > 0 and bottom == 0 and right == 0:
        matches.append(("tl", top, left))
    if top > 0 and right > 0 and bottom == 0 and left == 0:
        matches.append(("tr", top, right))
    if bottom > 0 and left > 0 and top == 0 and right == 0:
        matches.append(("bl", bottom, left))
    if bottom > 0 and right > 0 and top == 0 and left == 0:
        matches.append(("br", bottom, right))
    assert len(matches) == 1, "expected exactly one marked seed corner per panel"
    return matches[0]


def _tile_seed(seed: np.ndarray, corner: str, tile_rows: int, tile_cols: int) -> np.ndarray:
    seed_h, seed_w = seed.shape
    out = np.empty((seed_h * tile_rows, seed_w * tile_cols), dtype=int)
    seed_on_top = corner[0] == "t"
    seed_on_left = corner[1] == "l"
    for pr in range(tile_rows):
        vdist = pr if seed_on_top else tile_rows - 1 - pr
        for pc in range(tile_cols):
            hdist = pc if seed_on_left else tile_cols - 1 - pc
            tile = seed
            if vdist % 2 == 1:
                tile = np.flipud(tile)
            if hdist % 2 == 1:
                tile = np.fliplr(tile)
            r0 = pr * seed_h
            c0 = pc * seed_w
            out[r0 : r0 + seed_h, c0 : c0 + seed_w] = tile
    return out


def _solve_panel(panel: np.ndarray) -> np.ndarray:
    panel_h, panel_w = panel.shape
    corner, h_markers, v_markers = _corner_and_counts(panel)
    tile_rows = v_markers + 1
    tile_cols = h_markers + 1
    assert panel_h % tile_rows == 0, "panel height must be divisible by vertical tile count"
    assert panel_w % tile_cols == 0, "panel width must be divisible by horizontal tile count"
    seed_h = panel_h // tile_rows
    seed_w = panel_w // tile_cols
    seed = _extract_seed(panel, corner, seed_h, seed_w)

    forbidden = {BLACK, AZURE, MAROON}
    assert set(map(int, np.unique(seed))).isdisjoint(forbidden), "seed must use only motif colors"
    return _tile_seed(seed, corner, tile_rows, tile_cols)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-tile each corner seed across its marked panel until the whole panel is filled.

    Input structure:
    - Background is Black.
    - Single-pixel Maroon divider rows and columns partition the grid into equal rectangular panels.
    - Each panel contains one small rectangular multicolor seed placed flush in one corner.
    - Azure marker pixels lie on the two panel edges adjacent to that seed corner, outside the seed itself.

    Rule:
    - Detect the seed corner from which two panel edges contain Azure markers.
    - Count Azure markers on the seed-adjacent horizontal edge and vertical edge; these counts are H and V.
    - Infer the seed size from the panel size, extract the corner seed, and build a (V+1) x (H+1) array of copies.
    - Each step away from the seed alternates a mirror: horizontal neighbors flip left-right, vertical neighbors flip up-down; overwrite markers and background but keep Maroon dividers unchanged.

    Key insight:
    - The Azure edge counts encode how many mirrored copies extend away from the seed corner, so the panel dimensions reveal the seed rectangle and the full mirrored tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == MAROON)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == MAROON)]
    assert sep_rows and sep_cols, "expected maroon separator rows and columns"

    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one panel"

    panel_heights = {r1 - r0 for r0, r1 in row_spans}
    panel_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(panel_heights) == 1 and len(panel_widths) == 1, "panels must be equal-sized"

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0:r1, c0:c1] = _solve_panel(g[r0:r1, c0:c1])
    return out


def _make_seed(rng: random.Random, seed_h: int, seed_w: int) -> np.ndarray | None:
    cell_count = seed_h * seed_w
    palette_size = rng.randint(2, min(4, cell_count, len(SEED_COLORS)))
    palette = rng.sample(SEED_COLORS, palette_size)
    seed = np.array([rng.choice(palette) for _ in range(cell_count)], dtype=int).reshape(seed_h, seed_w)

    for color in palette:
        if color not in seed:
            idx = rng.randrange(cell_count)
            seed.flat[idx] = color

    if np.array_equal(seed, np.fliplr(seed)):
        return None
    if np.array_equal(seed, np.flipud(seed)):
        return None
    return seed


def _marker_positions(
    rng: random.Random,
    block_size: int,
    tile_count: int,
    seed_on_near_side: bool,
    total_length: int,
) -> list[int]:
    positions: list[int] = []
    if seed_on_near_side:
        block_indices = range(1, tile_count)
    else:
        block_indices = range(0, tile_count - 1)
    for block_idx in block_indices:
        block_start = block_idx * block_size
        candidates = list(range(block_start, block_start + block_size))
        if candidates[0] == 0:
            candidates = candidates[1:]
        if candidates[-1] == total_length - 1:
            candidates = candidates[:-1]
        assert candidates, "marker block must have an interior edge position"
        positions.append(rng.choice(candidates))
    return positions


def _build_panel(
    rng: random.Random,
    panel_h: int,
    panel_w: int,
    row_options: list[tuple[int, int]],
    col_options: list[tuple[int, int]],
) -> np.ndarray | None:
    seed_h, tile_rows = rng.choice(row_options)
    seed_w, tile_cols = rng.choice(col_options)
    corner = rng.choice(CORNERS)
    seed = _make_seed(rng, seed_h, seed_w)
    if seed is None:
        return None

    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    if corner == "tl":
        panel[:seed_h, :seed_w] = seed
    elif corner == "tr":
        panel[:seed_h, -seed_w:] = seed
    elif corner == "bl":
        panel[-seed_h:, :seed_w] = seed
    else:
        panel[-seed_h:, -seed_w:] = seed

    seed_on_top = corner[0] == "t"
    seed_on_left = corner[1] == "l"
    edge_row = 0 if seed_on_top else panel_h - 1
    edge_col = 0 if seed_on_left else panel_w - 1

    for c in _marker_positions(rng, seed_w, tile_cols, seed_on_left, panel_w):
        panel[edge_row, c] = AZURE
    for r in _marker_positions(rng, seed_h, tile_rows, seed_on_top, panel_h):
        panel[r, edge_col] = AZURE

    assert _corner_and_counts(panel)[0] == corner
    return panel


def _draw_dividers(grid: np.ndarray, panel_h: int, panel_w: int, panel_rows: int, panel_cols: int) -> None:
    for pr in range(panel_rows + 1):
        grid[pr * (panel_h + 1), :] = MAROON
    for pc in range(panel_cols + 1):
        grid[:, pc * (panel_w + 1)] = MAROON


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_h = rng.choice(PANEL_DIM_OPTIONS)
    panel_w = rng.choice(PANEL_DIM_OPTIONS)
    row_options = _factor_options(panel_h)
    col_options = _factor_options(panel_w)
    if not row_options or not col_options:
        return None

    panel_row_choices = [n for n in PANEL_COUNT_OPTIONS if n * (panel_h + 1) <= 29]
    panel_col_choices = [n for n in PANEL_COUNT_OPTIONS if n * (panel_w + 1) <= 29]
    if not panel_row_choices or not panel_col_choices:
        return None

    panel_rows = rng.choice(panel_row_choices)
    panel_cols = rng.choice(panel_col_choices)
    grid_h = panel_rows * panel_h + panel_rows + 1
    grid_w = panel_cols * panel_w + panel_cols + 1
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    _draw_dividers(grid, panel_h, panel_w, panel_rows, panel_cols)

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            panel = _build_panel(rng, panel_h, panel_w, row_options, col_options)
            if panel is None:
                return None
            r0 = 1 + pr * (panel_h + 1)
            c0 = 1 + pc * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-panel input")
