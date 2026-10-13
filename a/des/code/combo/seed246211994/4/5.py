import random
from typing import Iterable

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    banned = set(int(x) for x in exclude)
    return [int(c) for c in np.unique(grid) if int(c) not in banned]


def _line_positions(indices: np.ndarray) -> tuple[int, int] | None:
    if len(indices) < 3:
        return None
    if indices[0] != 0:
        return None
    steps = np.diff(indices)
    if len(steps) == 0 or not np.all(steps == steps[0]) or int(steps[0]) <= 1:
        return None
    return int(steps[0]), int(indices[-1])


def _find_lattice(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []

    for color in _colors_present(grid, exclude={bg}):
        row_lines = np.flatnonzero(np.all(grid == color, axis=1))
        col_lines = np.flatnonzero(np.all(grid == color, axis=0))
        row_info = _line_positions(row_lines)
        col_info = _line_positions(col_lines)
        if row_info is None or col_info is None:
            continue
        row_step, row_last = row_info
        col_step, col_last = col_info
        if row_last != h - 1 or col_last != w - 1:
            continue

        ok = True
        for r in range(h):
            cols = np.flatnonzero(grid[r] == color)
            if r in row_lines:
                ok = len(cols) == w
            else:
                ok = np.array_equal(cols, col_lines)
            if not ok:
                break
        if not ok:
            continue
        for c in range(w):
            rows = np.flatnonzero(grid[:, c] == color)
            if c in col_lines:
                ok = len(rows) == h
            else:
                ok = np.array_equal(rows, row_lines)
            if not ok:
                break
        if ok and row_step > 1 and col_step > 1:
            candidates.append((color, row_lines, col_lines))

    assert len(candidates) == 1, "expected exactly one periodic lattice color"
    return candidates[0]


def _extract_cells(
    grid: np.ndarray,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
) -> list[list[np.ndarray]]:
    cells: list[list[np.ndarray]] = []
    for rr in range(len(row_lines) - 1):
        row_cells: list[np.ndarray] = []
        r0 = int(row_lines[rr]) + 1
        r1 = int(row_lines[rr + 1])
        for cc in range(len(col_lines) - 1):
            c0 = int(col_lines[cc]) + 1
            c1 = int(col_lines[cc + 1])
            row_cells.append(grid[r0:r1, c0:c1])
        cells.append(row_cells)
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique marker-bearing cell as a miniature blueprint for recoloring
    the repeated pattern inside every lattice cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    line_color, row_lines, col_lines = _find_lattice(g, bg)
    cell_rows = len(row_lines) - 1
    cell_cols = len(col_lines) - 1
    assert cell_rows >= 2 and cell_cols >= 2

    cells = _extract_cells(g, row_lines, col_lines)
    presence: dict[int, set[tuple[int, int]]] = {}
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            cell = cells[rr][cc]
            colors = _colors_present(cell, exclude={bg, line_color})
            assert colors, "every cell must contain the shared pattern"
            for color in colors:
                presence.setdefault(color, set()).add((rr, cc))

    pattern_candidates = [color for color, spots in presence.items() if len(spots) > 1]
    marker_candidates = [color for color, spots in presence.items() if len(spots) == 1]
    assert len(pattern_candidates) == 1
    assert len(marker_candidates) == 1
    pattern_color = int(pattern_candidates[0])
    marker_color = int(marker_candidates[0])

    special_cells = list(presence[marker_color])
    assert len(special_cells) == 1
    special_r, special_c = special_cells[0]
    special = cells[special_r][special_c]

    special_mask = np.isin(special, (pattern_color, marker_color))
    dot_rows = np.flatnonzero(np.any(special_mask, axis=1))
    dot_cols = np.flatnonzero(np.any(special_mask, axis=0))
    assert len(dot_rows) == cell_rows
    assert len(dot_cols) == cell_cols

    expected_mask = np.zeros_like(special_mask, dtype=bool)
    expected_mask[np.ix_(dot_rows, dot_cols)] = True
    assert np.array_equal(special_mask, expected_mask)

    for rr in range(cell_rows):
        for cc in range(cell_cols):
            cell = cells[rr][cc]
            cell_mask = np.isin(cell, (pattern_color, marker_color))
            assert np.array_equal(cell_mask, expected_mask)
            colors = set(_colors_present(cell, exclude={bg, line_color}))
            assert colors <= {pattern_color, marker_color}
            if (rr, cc) != (special_r, special_c):
                assert marker_color not in colors

    blueprint = special[np.ix_(dot_rows, dot_cols)] == marker_color
    assert np.any(blueprint)
    assert not np.all(blueprint)

    out = g.copy()
    for rr in range(cell_rows):
        r0 = int(row_lines[rr]) + 1
        r1 = int(row_lines[rr + 1])
        for cc in range(cell_cols):
            c0 = int(col_lines[cc]) + 1
            c1 = int(col_lines[cc + 1])
            target = marker_color if blueprint[rr, cc] else pattern_color
            cell = out[r0:r1, c0:c1]
            cell[expected_mask] = target
    return out


# --------------- generator ---------------

COLOR_IDS = tuple(range(10))
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (2, 4)
MIN_EXTRA_AXIS = 2
MAX_EXTRA_AXIS = 6
MAX_GENERATION_ATTEMPTS = 400


def _sample_axis_positions(length: int, count: int) -> np.ndarray:
    assert length >= 2 * count - 1
    choices = sorted(random.sample(range(length - count + 1), count))
    return np.array([value + idx for idx, value in enumerate(choices)], dtype=int)


def _sample_blueprint(rows: int, cols: int, force_value: bool | None = None) -> np.ndarray:
    total = rows * cols
    assert total >= 2
    for _ in range(100):
        mask = np.array([random.random() < 0.5 for _ in range(total)], dtype=bool).reshape(rows, cols)
        if not np.any(mask) or np.all(mask):
            continue
        if force_value is None or bool(mask[0, 0]) == force_value:
            return mask
    raise RuntimeError("failed to sample a non-degenerate blueprint")


def _paint_pattern(
    grid: np.ndarray,
    top: int,
    left: int,
    dot_rows: np.ndarray,
    dot_cols: np.ndarray,
    color_map: np.ndarray,
) -> None:
    for rr, dr in enumerate(dot_rows):
        for cc, dc in enumerate(dot_cols):
            grid[top + int(dr), left + int(dc)] = int(color_map[rr, cc])


def _build_grids(
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    dot_rows: np.ndarray,
    dot_cols: np.ndarray,
    bg: int,
    line_color: int,
    pattern_color: int,
    marker_color: int,
    special_cell: tuple[int, int],
    blueprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = cell_rows * cell_h + cell_rows + 1
    w = cell_cols * cell_w + cell_cols + 1
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    row_lines = np.arange(0, h, cell_h + 1)
    col_lines = np.arange(0, w, cell_w + 1)
    inp[row_lines, :] = line_color
    inp[:, col_lines] = line_color
    out[row_lines, :] = line_color
    out[:, col_lines] = line_color

    base_map = np.full((cell_rows, cell_cols), pattern_color, dtype=int)
    special_map = np.where(blueprint, marker_color, pattern_color)

    for rr in range(cell_rows):
        top = int(row_lines[rr]) + 1
        for cc in range(cell_cols):
            left = int(col_lines[cc]) + 1
            source_map = special_map if (rr, cc) == special_cell else base_map
            target_color = marker_color if blueprint[rr, cc] else pattern_color
            target_map = np.full((cell_rows, cell_cols), target_color, dtype=int)
            _paint_pattern(inp, top, left, dot_rows, dot_cols, source_map)
            _paint_pattern(out, top, left, dot_rows, dot_cols, target_map)

    return inp, out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[bg]) > max(int(counts[c]) for c in range(10) if c != bg)


def _gen_attempt() -> np.ndarray | None:
    cell_rows = random.randint(*GRID_ROWS_RANGE)
    cell_cols = random.randint(*GRID_COLS_RANGE)
    cell_h = random.randint(2 * cell_rows + MIN_EXTRA_AXIS, 2 * cell_rows + MAX_EXTRA_AXIS)
    cell_w = random.randint(2 * cell_cols + MIN_EXTRA_AXIS, 2 * cell_cols + MAX_EXTRA_AXIS)

    dot_rows = _sample_axis_positions(cell_h, cell_rows)
    dot_cols = _sample_axis_positions(cell_w, cell_cols)

    bg, line_color, pattern_color, marker_color = random.sample(COLOR_IDS, 4)
    special_cell = (random.randrange(cell_rows), random.randrange(cell_cols))

    blueprint = _sample_blueprint(cell_rows, cell_cols)
    if random.random() < 0.5:
        blueprint[special_cell] = True
    else:
        blueprint[special_cell] = False
    if not np.any(blueprint) or np.all(blueprint):
        return None

    inp, expected = _build_grids(
        cell_rows,
        cell_cols,
        cell_h,
        cell_w,
        dot_rows,
        dot_cols,
        bg,
        line_color,
        pattern_color,
        marker_color,
        special_cell,
        blueprint,
    )

    if not _bg_is_dominant(inp, bg):
        return None

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
