from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)

MIN_TEMPLATE_HEIGHT = 6
MAX_TEMPLATE_HEIGHT = 14
MIN_TEMPLATE_WIDTH = 6
MAX_TEMPLATE_WIDTH = 18
MIN_ROW_MARKERS = 2
MIN_COL_MARKERS = 2
MAX_EXTRA_ROW_MARKERS = 3
MAX_EXTRA_COL_MARKERS = 4
MIN_TEMPLATE_SPAN = 3
MAX_GENERATION_ATTEMPTS = 400


def _neighbors(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _separator_row(grid: np.ndarray, bg: int) -> int:
    assert grid.ndim == 2 and grid.shape[0] % 2 == 1, "expected equal top/bottom sections"
    sep = grid.shape[0] // 2
    row = grid[sep]
    assert np.all(row == row[0]), "separator must be a solid row"
    assert int(row[0]) != bg, "separator color must differ from the background"
    return int(sep)


def _instruction_markers(top: np.ndarray, bg: int) -> tuple[dict[int, int], list[int]]:
    assert int(top[0, 0]) == bg, "top-left corner must be background"
    assert np.all(top[1:, 1:] == bg), "instruction interior must be empty"

    row_markers = {int(r): int(top[r, 0]) for r in range(1, top.shape[0]) if int(top[r, 0]) != bg}
    col_positions = [int(c) for c in range(1, top.shape[1]) if int(top[0, c]) != bg]

    assert row_markers, "expected at least one row marker"
    assert col_positions, "expected at least one column marker"
    assert len(set(row_markers.values())) == len(row_markers), "row marker colors must be unique"
    assert len({int(top[0, c]) for c in col_positions}) == len(col_positions), "column marker colors must be unique"
    return row_markers, col_positions


def _render_output(template: np.ndarray, row_markers: dict[int, int], col_positions: list[int], bg: int) -> np.ndarray:
    out = np.full(template.shape, bg, dtype=int)
    for r, color in row_markers.items():
        for c in col_positions:
            if int(template[r, c]) != bg:
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only template pixels whose coordinates are selected by top row/left column markers, coloring them by row marker.

    Input structure:
    - A solid non-background horizontal separator splits the grid into an upper instruction zone and a lower template zone of equal size.
    - The instruction zone has background everywhere except unique row markers on the leftmost column and unique column markers on the top row, with the top-left corner left as background.
    - The template zone uses the same background plus one foreground pattern color.

    Rule:
    - Read each row marker's row index and color from the instruction zone's left edge.
    - Read each column marker's column index from the instruction zone's top edge.
    - In the output, keep only template pixels at row/column marker intersections; if a selected template cell is foreground, color it with that row marker's color, otherwise leave it as background.

    Key insight:
    - The upper zone is a coordinate-selection matrix: row marker positions choose output rows, column marker positions choose output columns, and the row marker color is applied only where those coordinates hit the lower template.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(g[0, 0])
    sep = _separator_row(g, bg)

    top = g[:sep]
    template = g[sep + 1 :]
    assert top.shape == template.shape and top.size > 0, "instruction and template zones must match"

    row_markers, col_positions = _instruction_markers(top, bg)
    return _render_output(template, row_markers, col_positions, bg)


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbors(start[0], start[1], height, width))

    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nxt in _neighbors(cell[0], cell[1], height, width):
            if nxt not in cells:
                frontier.add(nxt)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _template_area_bounds(height: int, width: int) -> tuple[int, int]:
    total = height * width
    lo = (3 * total + 9) // 10
    hi = (5 * total) // 10
    return int(lo), int(hi)


def _template_is_readable(mask: np.ndarray) -> bool:
    grey_rows = [r for r in range(mask.shape[0]) if bool(mask[r].any())]
    grey_cols = [c for c in range(mask.shape[1]) if bool(mask[:, c].any())]

    if len(grey_rows) < MIN_TEMPLATE_SPAN or len(grey_cols) < MIN_TEMPLATE_SPAN:
        return False
    if any(bool(mask[r].all()) for r in range(mask.shape[0])):
        return False
    if any(bool(mask[:, c].all()) for c in range(mask.shape[1])):
        return False
    return True


def _choose_marker_positions(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    height, width = mask.shape
    row_candidates = [r for r in range(1, height) if bool(mask[r].any())]
    col_candidates = [c for c in range(1, width) if bool(mask[:, c].any())]
    if len(row_candidates) < MIN_ROW_MARKERS or len(col_candidates) < MIN_COL_MARKERS:
        return None

    bases: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rows in combinations(row_candidates, 2):
        for cols in combinations(col_candidates, 2):
            sub = mask[np.ix_(rows, cols)]
            active = int(sub.sum())
            if active < 2 or active == 4:
                continue
            if not bool(sub[0].any()) or not bool(sub[1].any()):
                continue
            if not bool(sub[:, 0].any()) or not bool(sub[:, 1].any()):
                continue
            bases.append((rows, cols))

    if not bases:
        return None

    base_rows, base_cols = random.choice(bases)
    extra_rows = [r for r in range(1, height) if r not in base_rows]
    extra_cols = [c for c in range(1, width) if c not in base_cols]
    random.shuffle(extra_rows)
    random.shuffle(extra_cols)

    row_total = len(base_rows) + random.randint(0, min(MAX_EXTRA_ROW_MARKERS, len(extra_rows)))
    col_total = len(base_cols) + random.randint(0, min(MAX_EXTRA_COL_MARKERS, len(extra_cols)))
    rows = sorted(list(base_rows) + extra_rows[: row_total - len(base_rows)])
    cols = sorted(list(base_cols) + extra_cols[: col_total - len(base_cols)])

    selected = mask[np.ix_(rows, cols)]
    active_selected = int(selected.sum())
    if active_selected < 2:
        return None
    if active_selected == len(rows) * len(cols):
        return None
    if int(mask.sum()) <= active_selected:
        return None
    if sum(bool(mask[r, cols].any()) for r in rows) < 2:
        return None
    if sum(bool(mask[rows, c].any()) for c in cols) < 2:
        return None
    return rows, cols


def _sample_template(height: int, width: int) -> tuple[np.ndarray, list[int], list[int]] | None:
    lo, hi = _template_area_bounds(height, width)
    for _ in range(60):
        area = random.randint(lo, hi)
        mask = _grow_connected_mask(height, width, area)
        if mask is None or not _template_is_readable(mask):
            continue
        markers = _choose_marker_positions(mask)
        if markers is None:
            continue
        rows, cols = markers
        return mask, rows, cols
    return None


def _build_input(
    template_mask: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    height, width = template_mask.shape
    top = np.full((height, width), BLACK, dtype=int)
    template = np.full((height, width), BLACK, dtype=int)
    template[template_mask] = GREY

    for r, color in zip(row_positions, row_colors):
        top[r, 0] = int(color)
    for c, color in zip(col_positions, col_colors):
        top[0, c] = int(color)

    separator = np.full((1, width), MAROON, dtype=int)
    return np.vstack([top, separator, template])


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_TEMPLATE_HEIGHT, MAX_TEMPLATE_HEIGHT)
    width = random.randint(MIN_TEMPLATE_WIDTH, MAX_TEMPLATE_WIDTH)

    sample = _sample_template(height, width)
    if sample is None:
        return None
    template_mask, row_positions, col_positions = sample

    row_colors = random.sample(MARKER_COLORS, len(row_positions))
    col_colors = random.sample(MARKER_COLORS, len(col_positions))
    inp = _build_input(template_mask, row_positions, col_positions, row_colors, col_colors)

    expected = np.full((height, width), BLACK, dtype=int)
    for r, color in zip(row_positions, row_colors):
        for c in col_positions:
            if template_mask[r, c]:
                expected[r, c] = int(color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction-template ARC input")
