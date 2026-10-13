from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _separator_structure(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color with full rows and columns"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(separators)
    spans = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty panels"
        spans.append((start, sep))
        start = sep + 1
    assert start < length, "separator lines must leave a final panel"
    spans.append((start, length))
    return spans


def _validate_equal_sizes(spans: Iterable[tuple[int, int]]) -> int:
    sizes = {end - start for start, end in spans}
    assert len(sizes) == 1, "all panels must have the same size"
    return sizes.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-partitioned grid, keep the unique odd-count panel from each panel row.

    Input structure:
    - Black is the background.
    - Full gray rows and full gray columns partition the grid into equal rectangular panels.
    - All panels within the same horizontal band use the same foreground color.

    Rule:
    - For each row of panels, count the foreground pixels inside each panel.
    - Exactly one panel in that row has an odd count; all others are even.
    - Build an output containing one panel column, copying the odd panel from each row.
    - Preserve the horizontal separator rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, sep_rows, sep_cols = _separator_structure(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    panel_h = _validate_equal_sizes(row_spans)
    panel_w = _validate_equal_sizes(col_spans)
    assert separator != BLACK, "separator must differ from background"

    out = np.full((g.shape[0], panel_w), separator, dtype=int)
    for band_index, (r0, r1) in enumerate(row_spans):
        odd_tiles = []
        band_color = None
        for c0, c1 in col_spans:
            tile = g[r0:r1, c0:c1]
            colors = {int(x) for x in np.unique(tile) if int(x) not in {BLACK, separator}}
            assert len(colors) <= 1, "each panel must use at most one foreground color"
            color = next(iter(colors)) if colors else None
            if color is not None:
                if band_color is None:
                    band_color = color
                else:
                    assert color == band_color, "panels in one row band must share the same foreground color"

            fg_count = int(np.count_nonzero(tile == band_color)) if band_color is not None else 0
            if fg_count % 2 == 1:
                odd_tiles.append(tile)

        assert band_color is not None, "each row band must contain foreground pixels"
        assert len(odd_tiles) == 1, "each row band must have exactly one odd-count panel"
        out[r0:r1, :] = odd_tiles[0]
        if band_index < len(row_spans) - 1:
            sep_r = row_spans[band_index][1]
            out[sep_r, :] = separator

    return out


def _random_connected_shape(height: int, width: int, count: int) -> set[tuple[int, int]]:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _sample_counts(panel_area: int, n_cols: int) -> tuple[list[int], int] | None:
    odd_index = random.randrange(n_cols)
    counts = []
    for c in range(n_cols):
        parity = 1 if c == odd_index else 0
        options = [k for k in range(1, panel_area + 1) if k % 2 == parity]
        if not options:
            return None
        if parity == 0:
            options = [k for k in options if k >= 2]
        counts.append(random.choice(options))
    return counts, odd_index


def _render_tile(panel_h: int, panel_w: int, fg: int, count: int) -> np.ndarray:
    tile = np.zeros((panel_h, panel_w), dtype=int)
    mode = random.choice(["connected", "scatter", "rectangle"])
    if mode == "connected":
        coords = list(_random_connected_shape(panel_h, panel_w, count))
    elif mode == "rectangle":
        candidates = [
            (h, w)
            for h in range(1, panel_h + 1)
            for w in range(1, panel_w + 1)
            if h * w <= count
        ]
        h, w = random.choice(candidates)
        r0 = random.randint(0, panel_h - h)
        c0 = random.randint(0, panel_w - w)
        coords = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
        remaining = count - len(coords)
        if remaining:
            leftovers = [(r, c) for r in range(panel_h) for c in range(panel_w) if (r, c) not in set(coords)]
            coords.extend(random.sample(leftovers, remaining))
    else:
        coords = random.sample([(r, c) for r in range(panel_h) for c in range(panel_w)], count)

    for r, c in coords:
        tile[r, c] = fg
    return tile


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.randint(2, 4)
    panel_cols = random.randint(2, 5)
    panel_h = random.randint(2, 6)
    panel_w = random.randint(2, 6)

    height = panel_rows * panel_h + (panel_rows - 1)
    width = panel_cols * panel_w + (panel_cols - 1)
    grid = np.zeros((height, width), dtype=int)

    for r in range(1, panel_rows):
        grid[r * panel_h + (r - 1), :] = GRAY
    for c in range(1, panel_cols):
        grid[:, c * panel_w + (c - 1)] = GRAY

    row_colors = random.sample(COLOR_POOL, panel_rows)
    panel_area = panel_h * panel_w
    expected = np.full((height, panel_w), GRAY, dtype=int)

    for band in range(panel_rows):
        counts_info = _sample_counts(panel_area, panel_cols)
        if counts_info is None:
            return None
        counts, odd_index = counts_info
        fg = row_colors[band]
        r0 = band * panel_h + band
        r1 = r0 + panel_h
        chosen_tile = None

        for col in range(panel_cols):
            c0 = col * panel_w + col
            c1 = c0 + panel_w
            tile = _render_tile(panel_h, panel_w, fg, counts[col])
            grid[r0:r1, c0:c1] = tile
            if col == odd_index:
                chosen_tile = tile

        assert chosen_tile is not None
        expected[r0:r1, :] = chosen_tile
        if band < panel_rows - 1:
            expected[r1, :] = GRAY

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
