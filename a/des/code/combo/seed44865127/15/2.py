import random

import numpy as np


DIAGONAL_STEPS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _axis_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [length]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((left + 1, right - 1))
    return spans


def _find_separator_lattice(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in map(int, np.unique(grid)):
        sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
        sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
        if not sep_rows or not sep_cols:
            continue

        row_spans = _axis_spans(h, sep_rows)
        col_spans = _axis_spans(w, sep_cols)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if next(iter(row_sizes)) != next(iter(col_sizes)):
            continue

        score = len(sep_rows) + len(sep_cols)
        candidates.append((score, color, row_spans, col_spans))

    assert candidates, "expected a regular separator lattice"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, f"ambiguous separator lattice: {best}"
    _, sep_color, row_spans, col_spans = best[0]
    return sep_color, row_spans, col_spans


def _is_singleton_color(grid: np.ndarray, color: int) -> bool:
    points = np.argwhere(grid == color)
    if points.size == 0:
        return False

    h, w = grid.shape
    for r, c in points:
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
                return False
    return True


def _corner_step(local_r: int, local_c: int, size: int) -> tuple[int, int] | None:
    corners = [
        ((0, 0), (-1, -1)),
        ((0, size - 1), (-1, 1)),
        ((size - 1, 0), (1, -1)),
        ((size - 1, size - 1), (1, 1)),
    ]
    distances = []
    for (cr, cc), step in corners:
        dist = (local_r - cr) ** 2 + (local_c - cc) ** 2
        distances.append((dist, step))
    distances.sort(key=lambda item: item[0])
    if distances[0][0] == distances[1][0]:
        return None
    return distances[0][1]


def _paint_ray(
    out: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    sep_color: int,
    mark_color: int,
) -> None:
    h, w = out.shape
    r, c = start
    dr, dc = step
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if int(out[r, c]) != sep_color:
            out[r, c] = mark_color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    On a separator-defined checkerboard, cast a diagonal ray from each marker toward
    the nearest corner of its own cell.

    Input structure:
    - One color forms complete horizontal and vertical separator lines.
    - The separator lattice partitions the grid into equal square cells.
    - Each cell contains exactly one singleton marker pixel of the same color.

    Rule:
    - For each cell, find which of the four cell corners is nearest to the marker.
    - Starting from the marker, extend a diagonal ray toward that corner all the way
      to the grid edge.
    - Rays keep going across separator lines and other cells, but separator pixels
      themselves stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, row_spans, col_spans = _find_separator_lattice(g)
    cell_count = len(row_spans) * len(col_spans)

    marker_candidates = [
        color
        for color in map(int, np.unique(g))
        if color != sep_color
        and _is_singleton_color(g, color)
        and int(np.count_nonzero(g == color)) == cell_count
    ]
    assert len(marker_candidates) == 1, f"expected one marker color, got {marker_candidates}"
    mark_color = marker_candidates[0]

    out = g.copy()
    cell_size = row_spans[0][1] - row_spans[0][0] + 1
    for r0, r1 in row_spans:
        assert r1 - r0 + 1 == cell_size, "row spans must share one size"
        for c0, c1 in col_spans:
            assert c1 - c0 + 1 == cell_size, "column spans must match cell size"
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            markers = np.argwhere(patch == mark_color)
            assert len(markers) == 1, f"cell {(r0, c0, r1, c1)} must contain one marker"
            local_r, local_c = map(int, markers[0])
            step = _corner_step(local_r, local_c, cell_size)
            assert step is not None, f"marker at {(r0 + local_r, c0 + local_c)} is corner-ambiguous"
            _paint_ray(out, (r0 + local_r, c0 + local_c), step, sep_color, mark_color)

    return out


# --------------- generator ---------------

ROW_COUNT_RANGE = (2, 5)
COL_COUNT_RANGE = (2, 5)
CELL_SIZE_RANGE = (4, 7)
MAX_ATTEMPTS = 400


def _separator_indices(count: int, cell_size: int, border: bool) -> list[int]:
    if border:
        return [i * (cell_size + 1) for i in range(count + 1)]
    return [cell_size + i * (cell_size + 1) for i in range(count - 1)]


def _cell_starts(count: int, cell_size: int, border: bool) -> list[int]:
    offset = 1 if border else 0
    return [offset + i * (cell_size + 1) for i in range(count)]


def _corner_choices(size: int, step: tuple[int, int]) -> list[tuple[int, int]]:
    band = min(2, size // 2)
    if step[0] < 0:
        row_choices = range(0, band)
    else:
        row_choices = range(size - band, size)
    if step[1] < 0:
        col_choices = range(0, band)
    else:
        col_choices = range(size - band, size)

    choices = []
    for r in row_choices:
        for c in col_choices:
            if _corner_step(r, c, size) == step:
                choices.append((r, c))
    assert choices, f"expected non-empty choices for size={size}, step={step}"
    return choices


def _balanced_steps(count: int) -> list[tuple[int, int]]:
    steps: list[tuple[int, int]] = []
    while len(steps) < count:
        cycle = DIAGONAL_STEPS[:]
        random.shuffle(cycle)
        steps.extend(cycle)
    steps = steps[:count]
    random.shuffle(steps)
    return steps


def _render_expected(
    inp: np.ndarray,
    sep_color: int,
    mark_color: int,
    marker_specs: list[tuple[int, int, tuple[int, int]]],
) -> np.ndarray:
    out = inp.copy()
    for r, c, step in marker_specs:
        _paint_ray(out, (r, c), step, sep_color, mark_color)
    return out


def _build_input(
    rows: int,
    cols: int,
    cell_size: int,
    border: bool,
    bg: int,
    sep_color: int,
    mark_color: int,
) -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int]]]]:
    height = rows * cell_size + (rows + 1 if border else rows - 1)
    width = cols * cell_size + (cols + 1 if border else cols - 1)
    grid = np.full((height, width), bg, dtype=int)

    for r in _separator_indices(rows, cell_size, border):
        grid[r, :] = sep_color
    for c in _separator_indices(cols, cell_size, border):
        grid[:, c] = sep_color

    row_starts = _cell_starts(rows, cell_size, border)
    col_starts = _cell_starts(cols, cell_size, border)
    steps = _balanced_steps(rows * cols)

    marker_specs: list[tuple[int, int, tuple[int, int]]] = []
    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            step = steps[idx]
            idx += 1
            local_r, local_c = random.choice(_corner_choices(cell_size, step))
            r = r0 + local_r
            c = c0 + local_c
            grid[r, c] = mark_color
            marker_specs.append((r, c, step))

    return grid, marker_specs


def _is_valid_input(
    grid: np.ndarray,
    rows: int,
    cols: int,
    bg: int,
    sep_color: int,
    mark_color: int,
) -> bool:
    if _most_common_color(grid) != bg:
        return False
    if int(np.count_nonzero(grid == mark_color)) != rows * cols:
        return False
    if int(np.count_nonzero(grid == sep_color)) <= int(np.count_nonzero(grid == mark_color)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    cell_size = random.randint(*CELL_SIZE_RANGE)
    border = bool(random.getrandbits(1))
    bg, sep_color, mark_color = random.sample(range(10), 3)

    inp, marker_specs = _build_input(rows, cols, cell_size, border, bg, sep_color, mark_color)
    if not _is_valid_input(inp, rows, cols, bg, sep_color, mark_color):
        return None

    expected = _render_expected(inp, sep_color, mark_color, marker_specs)
    if np.array_equal(expected, inp):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
