from __future__ import annotations

import random

import numpy as np


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        assert 0 < cut < size - 1, "separator lines must be internal"
        assert start < cut, "separator lines cannot touch"
        bands.append((start, cut))
        start = cut + 1
    assert start < size, "separator lines cannot end at the border"
    bands.append((start, size))
    return bands


def _find_separator_color(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = g.shape
    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one separator color spanning full rows and columns"

    sep = next(iter(candidates))
    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separator lines"
    return sep, sep_rows, sep_cols


def _equal_band_size(bands: list[tuple[int, int]]) -> int:
    sizes = {end - start for start, end in bands}
    assert len(sizes) == 1, "all panels must have the same size along each axis"
    return next(iter(sizes))


def _infer_background(g: np.ndarray, sep: int) -> int:
    values, counts = np.unique(g[g != sep], return_counts=True)
    assert len(values) >= 2, "expected a background and foreground colors inside the panels"
    order = np.argsort(counts)[::-1]
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "background must be uniquely dominant"
    return int(values[order[0]])


def _extract_panels(
    g: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    panels: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    panel_h = _equal_band_size(row_bands)
    panel_w = _equal_band_size(col_bands)

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            panel = g[r0:r1, c0:c1]
            assert panel.shape == (panel_h, panel_w)
            panels.append(panel.copy())
            positions.append((ri, ci))
    return panels, positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the separator-defined panels by foreground population, then mirror every even-ranked panel horizontally.

    Input structure:
    - The background is black.
    - One gray color forms complete one-cell-thick horizontal and vertical lines
      that partition the grid into equal rectangular panels.
    - Each panel contains a small pattern made of non-background, non-separator
      foreground pixels, and the panel foreground counts are all distinct.

    Rule:
    - Split the grid into panels using the full separator rows and columns.
    - Count the number of non-background, non-separator pixels in each panel and
      sort the panels from fewest to most foreground pixels.
    - Place the sorted panels back into the original panel lattice in reading
      order while keeping the gray separator lines unchanged.
    - Panels with odd ranks in the sorted order stay unchanged, while panels with
      even ranks are mirrored left-to-right inside their panel.

    Key insight:
    - The separator lattice defines the movable units, and two independent cues
      must be combined: foreground-pixel count determines panel order, while
      rank parity determines whether the panel is horizontally flipped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep, sep_rows, sep_cols = _find_separator_color(g)
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-panel lattice"

    bg = _infer_background(g, sep)
    panels, _positions = _extract_panels(g, row_bands, col_bands)

    counts = [int(np.count_nonzero((panel != bg) & (panel != sep))) for panel in panels]
    assert min(counts) > 0, "every panel must contain foreground pixels"
    assert len(set(counts)) == len(counts), "panel foreground counts must be distinct"

    sorted_indices = sorted(range(len(panels)), key=lambda idx: counts[idx])
    out = g.copy()

    for rank, src_idx in enumerate(sorted_indices, start=1):
        dst_idx = rank - 1
        dr = dst_idx // len(col_bands)
        dc = dst_idx % len(col_bands)
        r0, r1 = row_bands[dr]
        c0, c1 = col_bands[dc]

        panel = panels[src_idx]
        if rank % 2 == 0:
            panel = np.fliplr(panel)
        out[r0:r1, c0:c1] = panel

    return out


# --------------- generator ---------------

BG = 0
SEP = 5
FG_PALETTE = tuple(color for color in range(10) if color not in {BG, SEP})
MAX_GRID_SIDE = 30
PANEL_ROW_RANGE = range(2, 5)
PANEL_COL_RANGE = range(2, 5)
PANEL_SIDE_RANGE = range(3, 8)
MAX_PANEL_COLORS = 4
MAX_GENERATION_ATTEMPTS = 500


def _geometry_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for panel_rows in PANEL_ROW_RANGE:
        for panel_cols in PANEL_COL_RANGE:
            panel_count = panel_rows * panel_cols
            for panel_h in PANEL_SIDE_RANGE:
                for panel_w in PANEL_SIDE_RANGE:
                    height = panel_rows * panel_h + (panel_rows - 1)
                    width = panel_cols * panel_w + (panel_cols - 1)
                    if max(height, width) > MAX_GRID_SIDE:
                        continue
                    if panel_h * panel_w < panel_count + 2:
                        continue
                    options.append((panel_rows, panel_cols, panel_h, panel_w))
    return options


GEOMETRY_OPTIONS = _geometry_options()


def _grow_positions(panel_h: int, panel_w: int, count: int) -> list[tuple[int, int]]:
    seed = (random.randrange(panel_h), random.randrange(panel_w))
    selected = {seed}

    while len(selected) < count:
        frontier = set()
        for r, c in selected:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < panel_h and 0 <= cc < panel_w and (rr, cc) not in selected:
                    frontier.add((rr, cc))

        if frontier and random.random() < 0.8:
            selected.add(random.choice(tuple(frontier)))
        else:
            remaining = [
                (r, c)
                for r in range(panel_h)
                for c in range(panel_w)
                if (r, c) not in selected
            ]
            selected.add(random.choice(remaining))

    coords = list(selected)
    random.shuffle(coords)
    return coords


def _make_panel(panel_h: int, panel_w: int, count: int) -> np.ndarray | None:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    coords = _grow_positions(panel_h, panel_w, count)

    color_count = random.randint(1, min(MAX_PANEL_COLORS, len(FG_PALETTE), count))
    colors = random.sample(FG_PALETTE, color_count)
    color_weights = [random.randint(1, 4) for _ in colors]

    for r, c in coords:
        panel[r, c] = random.choices(colors, weights=color_weights, k=1)[0]

    if np.array_equal(panel, np.fliplr(panel)):
        return None
    return panel


def _render_grid(
    panels: list[np.ndarray],
    panel_rows: int,
    panel_cols: int,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    height = panel_rows * panel_h + (panel_rows - 1)
    width = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((height, width), SEP, dtype=int)

    for idx, panel in enumerate(panels):
        pr = idx // panel_cols
        pc = idx % panel_cols
        r0 = pr * (panel_h + 1)
        c0 = pc * (panel_w + 1)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    return grid


def _expected_output(
    panels: list[np.ndarray],
    counts: list[int],
    panel_rows: int,
    panel_cols: int,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    sorted_indices = sorted(range(len(panels)), key=lambda idx: counts[idx])
    out_panels: list[np.ndarray] = []
    for rank, idx in enumerate(sorted_indices, start=1):
        panel = panels[idx].copy()
        if rank % 2 == 0:
            panel = np.fliplr(panel)
        out_panels.append(panel)
    return _render_grid(out_panels, panel_rows, panel_cols, panel_h, panel_w)


def _background_is_unique(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid[grid != SEP], return_counts=True)
    if len(values) < 2:
        return False
    order = np.argsort(counts)[::-1]
    if len(order) > 1 and int(counts[order[0]]) == int(counts[order[1]]):
        return False
    return int(values[order[0]]) == BG


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols, panel_h, panel_w = random.choice(GEOMETRY_OPTIONS)
    panel_count = panel_rows * panel_cols
    max_count = panel_h * panel_w - 2
    counts_sorted = sorted(random.sample(range(1, max_count + 1), panel_count))

    order = list(range(panel_count))
    random.shuffle(order)
    if order == sorted(order):
        return None
    counts = [counts_sorted[idx] for idx in order]

    panels: list[np.ndarray] = []
    for count in counts:
        panel = _make_panel(panel_h, panel_w, count)
        if panel is None:
            return None
        panels.append(panel)

    inp = _render_grid(panels, panel_rows, panel_cols, panel_h, panel_w)
    expected = _expected_output(panels, counts, panel_rows, panel_cols, panel_h, panel_w)
    if np.array_equal(inp, expected):
        return None
    if not _background_is_unique(inp):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-sorting input")
