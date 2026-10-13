from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
SEPARATOR = 9
CONTENT_COLORS = tuple(range(1, 9))

MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400

PANEL_ROW_RANGE = (3, 5)
PANEL_COL_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (3, 6)
PANEL_WIDTH_RANGE = (3, 7)


def _uniform_color_groups(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    if axis == 0:
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] == grid[r, 0]):
                groups[int(grid[r, 0])].append(r)
    else:
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] == grid[0, c]):
                groups[int(grid[0, c])].append(c)
    return groups


def _bands(size: int, separators: list[int]) -> list[slice]:
    points = [-1] + sorted(separators) + [size]
    bands: list[slice] = []
    for left, right in zip(points, points[1:]):
        start = left + 1
        stop = right
        if stop > start:
            bands.append(slice(start, stop))
    return bands


def _panel_matrix(grid: np.ndarray) -> list[list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_groups = _uniform_color_groups(g, axis=0)
    col_groups = _uniform_color_groups(g, axis=1)
    candidates = set(row_groups) & set(col_groups)
    assert candidates, "expected a separator color shared by full rows and full columns"

    scores = {
        color: len(row_groups[color]) + len(col_groups[color])
        for color in candidates
    }
    best_score = max(scores.values())
    best = [color for color, score in scores.items() if score == best_score]
    assert len(best) == 1, "separator color must be unambiguous"
    separator = best[0]

    row_bands = _bands(g.shape[0], row_groups[separator])
    col_bands = _bands(g.shape[1], col_groups[separator])
    assert len(row_bands) >= 3, "need at least three panel rows to identify a unique panel"
    assert len(col_bands) >= 2, "need at least two panel columns"

    row_heights = {band.stop - band.start for band in row_bands}
    col_widths = {band.stop - band.start for band in col_bands}
    assert len(row_heights) == 1, "all panel rows must have the same height"
    assert len(col_widths) == 1, "all panel columns must have the same width"

    panels: list[list[np.ndarray]] = []
    for row_band in row_bands:
        row_panels: list[np.ndarray] = []
        for col_band in col_bands:
            row_panels.append(g[row_band, col_band].copy())
        panels.append(row_panels)
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique panel in each separator-defined column and concatenate them.
    """
    panels = _panel_matrix(np.asarray(grid, dtype=int))
    panel_rows = len(panels)
    panel_cols = len(panels[0])
    panel_h, panel_w = panels[0][0].shape

    uniques: list[np.ndarray] = []
    for col in range(panel_cols):
        column_panels = [panels[row][col] for row in range(panel_rows)]
        signatures = [panel.tobytes() for panel in column_panels]
        counts = Counter(signatures)
        assert len(counts) == 2, "each panel column must contain one odd panel and one repeated panel"
        frequencies = sorted(counts.values())
        assert frequencies == [1, panel_rows - 1], "each column must have exactly one unique panel"

        unique_signature = next(sig for sig, count in counts.items() if count == 1)
        unique_index = signatures.index(unique_signature)
        uniques.append(column_panels[unique_index])

    out_w = panel_cols * panel_w + (panel_cols - 1)
    out = np.full((panel_h, out_w), BACKGROUND, dtype=int)
    cursor = 0
    for panel in uniques:
        out[:, cursor : cursor + panel_w] = panel
        cursor += panel_w + 1
    return out


def _randint(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return int(rng.integers(low, high + 1))


def _random_bbox(rng: np.random.Generator, height: int, width: int) -> tuple[int, int, int, int]:
    r0 = int(rng.integers(0, height))
    r1 = int(rng.integers(r0, height))
    c0 = int(rng.integers(0, width))
    c1 = int(rng.integers(c0, width))
    return r0, r1, c0, c1


def _paint_diagonal(
    panel: np.ndarray, rng: np.random.Generator, color: int, reverse: bool
) -> None:
    h, w = panel.shape
    span = min(h, w)
    row0 = int(rng.integers(0, h - span + 1)) if h > span else 0
    col0 = int(rng.integers(0, w - span + 1)) if w > span else 0
    for k in range(span):
        rr = row0 + k
        cc = col0 + (span - 1 - k if reverse else k)
        panel[rr, cc] = color


def _draw_primitive(panel: np.ndarray, rng: np.random.Generator, color: int) -> None:
    h, w = panel.shape
    kind = int(rng.integers(0, 7))

    if kind == 0:
        r0, r1, c0, c1 = _random_bbox(rng, h, w)
        panel[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    if kind == 1:
        r0, r1, c0, c1 = _random_bbox(rng, h, w)
        panel[r0, c0 : c1 + 1] = color
        panel[r1, c0 : c1 + 1] = color
        panel[r0 : r1 + 1, c0] = color
        panel[r0 : r1 + 1, c1] = color
        return

    if kind == 2:
        row = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w))
        c1 = int(rng.integers(c0, w))
        panel[row, c0 : c1 + 1] = color
        return

    if kind == 3:
        col = int(rng.integers(0, w))
        r0 = int(rng.integers(0, h))
        r1 = int(rng.integers(r0, h))
        panel[r0 : r1 + 1, col] = color
        return

    if kind == 4:
        center_r = int(rng.integers(0, h))
        center_c = int(rng.integers(0, w))
        arm_h = int(rng.integers(0, h))
        arm_w = int(rng.integers(0, w))
        panel[max(0, center_r - arm_h) : min(h, center_r + arm_h + 1), center_c] = color
        panel[center_r, max(0, center_c - arm_w) : min(w, center_c + arm_w + 1)] = color
        return

    if kind == 5:
        _paint_diagonal(panel, rng, color, reverse=bool(rng.integers(0, 2)))
        return

    points = int(rng.integers(2, min(h * w, 5) + 1))
    flat = rng.choice(h * w, size=points, replace=False)
    panel.reshape(-1)[flat] = color


def _ensure_visible_structure(panel: np.ndarray, rng: np.random.Generator, palette: np.ndarray) -> np.ndarray:
    result = panel.copy()
    if not np.any(result != BACKGROUND):
        color = int(rng.choice(palette))
        result[int(rng.integers(0, result.shape[0])), int(rng.integers(0, result.shape[1]))] = color
    if np.all(result == result[0, 0]):
        color = int(rng.choice([c for c in palette if c != int(result[0, 0])]))
        result[-1, -1] = color
    return result


def _random_panel(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    panel = np.full((height, width), BACKGROUND, dtype=int)
    color_count = int(rng.integers(2, 5))
    palette = np.array(rng.choice(CONTENT_COLORS, size=color_count, replace=False), dtype=int)
    primitive_count = int(rng.integers(2, 6))

    for _ in range(primitive_count):
        color = int(rng.choice(palette))
        _draw_primitive(panel, rng, color)

    return _ensure_visible_structure(panel, rng, palette)


def _mutate_panel(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    choices = np.array((BACKGROUND,) + CONTENT_COLORS, dtype=int)
    for _ in range(60):
        panel = base.copy()
        mode = int(rng.integers(0, 4))

        if mode == 0:
            r = int(rng.integers(0, panel.shape[0]))
            c = int(rng.integers(0, panel.shape[1]))
            current = int(panel[r, c])
            replacement = int(rng.choice(choices[choices != current]))
            panel[r, c] = replacement
        elif mode == 1:
            r0, r1, c0, c1 = _random_bbox(rng, *panel.shape)
            current = panel[r0 : r1 + 1, c0 : c1 + 1]
            current_values = np.unique(current)
            available = np.array([c for c in choices if c not in current_values], dtype=int)
            if len(available) == 0:
                available = np.array([c for c in choices if c != int(current.flat[0])], dtype=int)
            panel[r0 : r1 + 1, c0 : c1 + 1] = int(rng.choice(available))
        elif mode == 2:
            color = int(rng.choice(CONTENT_COLORS))
            _draw_primitive(panel, rng, color)
        else:
            if bool(rng.integers(0, 2)):
                panel = np.flip(panel, axis=0)
            else:
                panel = np.flip(panel, axis=1)

        if not np.array_equal(panel, base):
            return panel
    raise RuntimeError("failed to mutate panel")


def _assemble_grid(panels: list[list[np.ndarray]], outer_border: bool) -> np.ndarray:
    panel_rows = len(panels)
    panel_cols = len(panels[0])
    panel_h, panel_w = panels[0][0].shape
    offset = 1 if outer_border else 0

    grid_h = panel_rows * panel_h + (panel_rows - 1) + 2 * offset
    grid_w = panel_cols * panel_w + (panel_cols - 1) + 2 * offset
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    for r in range(panel_rows - 1):
        sep_r = offset + (r + 1) * panel_h + r
        grid[sep_r, :] = SEPARATOR
    for c in range(panel_cols - 1):
        sep_c = offset + (c + 1) * panel_w + c
        grid[:, sep_c] = SEPARATOR
    if outer_border:
        grid[0, :] = SEPARATOR
        grid[-1, :] = SEPARATOR
        grid[:, 0] = SEPARATOR
        grid[:, -1] = SEPARATOR

    for r in range(panel_rows):
        row0 = offset + r * (panel_h + 1)
        for c in range(panel_cols):
            col0 = offset + c * (panel_w + 1)
            grid[row0 : row0 + panel_h, col0 : col0 + panel_w] = panels[r][c]
    return grid


def _sample_layout(rng: np.random.Generator) -> tuple[int, int, int, int, bool] | None:
    panel_rows = _randint(rng, PANEL_ROW_RANGE)
    panel_cols = _randint(rng, PANEL_COL_RANGE)
    panel_h = _randint(rng, PANEL_HEIGHT_RANGE)
    panel_w = _randint(rng, PANEL_WIDTH_RANGE)
    outer_border = bool(rng.integers(0, 2))

    offset = 1 if outer_border else 0
    grid_h = panel_rows * panel_h + (panel_rows - 1) + 2 * offset
    grid_w = panel_cols * panel_w + (panel_cols - 1) + 2 * offset
    if grid_h > MAX_GRID_SIZE or grid_w > MAX_GRID_SIZE:
        return None
    return panel_rows, panel_cols, panel_h, panel_w, outer_border


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    panel_rows, panel_cols, panel_h, panel_w, outer_border = layout

    panels: list[list[np.ndarray]] = [
        [np.full((panel_h, panel_w), BACKGROUND, dtype=int) for _ in range(panel_cols)]
        for _ in range(panel_rows)
    ]

    for col in range(panel_cols):
        common_panel = _random_panel(rng, panel_h, panel_w)
        if bool(rng.integers(0, 2)):
            unique_panel = _mutate_panel(common_panel, rng)
        else:
            for _ in range(80):
                unique_panel = _random_panel(rng, panel_h, panel_w)
                if not np.array_equal(unique_panel, common_panel):
                    break
            else:
                return None

        unique_row = int(rng.integers(0, panel_rows))
        for row in range(panel_rows):
            panels[row][col] = unique_panel.copy() if row == unique_row else common_panel.copy()

    grid = _assemble_grid(panels, outer_border=outer_border)

    try:
        solved = main(grid)
    except Exception:
        return None

    expected_w = panel_cols * panel_w + (panel_cols - 1)
    if solved.shape != (panel_h, expected_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
