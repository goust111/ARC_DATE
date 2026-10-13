from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
AZURE = 8
PATTERN_COLORS = (1, 3, 4, 5, 6, 7, 9)
PATTERN_KINDS = (
    "solid",
    "checker",
    "diag_down",
    "diag_up",
    "border",
    "cross",
    "vstripe",
    "hstripe",
    "xshape",
)
MAX_GENERATION_ATTEMPTS = 300


def _line_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    idxs = [int(v) for v in np.asarray(indices, dtype=int).tolist()]
    assert idxs, "expected at least one separator line"
    runs: list[tuple[int, int]] = []
    start = idxs[0]
    prev = idxs[0]
    for value in idxs[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def _cell_slices(full_lines: np.ndarray, limit: int) -> list[slice]:
    runs = _line_runs(full_lines)
    assert runs[0][0] == 0 and runs[-1][1] == limit - 1, "separator border must frame the whole grid"
    slices: list[slice] = []
    for left, right in zip(runs, runs[1:]):
        start = left[1] + 1
        stop = right[0]
        assert stop > start, "adjacent separator bands cannot touch"
        slices.append(slice(start, stop))
    lengths = {int(s.stop - s.start) for s in slices}
    assert len(lengths) == 1, "all cells must share the same size"
    return slices


def _split_cells(grid: np.ndarray) -> tuple[list[slice], list[slice]]:
    row_lines = np.where(np.all(grid == AZURE, axis=1))[0]
    col_lines = np.where(np.all(grid == AZURE, axis=0))[0]
    assert row_lines.size >= 2 and col_lines.size >= 2, "expected azure separator rows and columns"
    row_slices = _cell_slices(row_lines, int(grid.shape[0]))
    col_slices = _cell_slices(col_lines, int(grid.shape[1]))
    return row_slices, col_slices


def _legend_order(grid: np.ndarray, row_slices: list[slice], col_slices: list[slice]) -> list[tuple[int, int]]:
    marker = np.zeros((len(row_slices), len(col_slices)), dtype=bool)
    for r, rs in enumerate(row_slices):
        for c, cs in enumerate(col_slices):
            cell = grid[rs, cs]
            marker[r, c] = bool(np.any((cell != BLACK) & (cell != RED)))

    coords = np.argwhere(marker)
    assert coords.shape[0] == 4, "legend must occupy exactly four cells"
    rows = sorted({int(v) for v in coords[:, 0]})
    cols = sorted({int(v) for v in coords[:, 1]})

    if len(rows) == 1 and len(cols) == 4 and rows[0] == 0 and cols == list(range(cols[0], cols[0] + 4)):
        return [(rows[0], c) for c in cols]
    if len(cols) == 1 and len(rows) == 4 and cols[0] == 0 and rows == list(range(rows[0], rows[0] + 4)):
        return [(r, cols[0]) for r in rows]
    raise AssertionError("legend must be a 1x4 strip on the top or a 4x1 strip on the left")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_slices, col_slices = _split_cells(g)
    legend_cells = _legend_order(g, row_slices, col_slices)
    legend_set = set(legend_cells)
    patterns = [g[row_slices[r], col_slices[c]].copy() for r, c in legend_cells]

    out = g.copy()
    for r, rs in enumerate(row_slices):
        for c, cs in enumerate(col_slices):
            if (r, c) in legend_set:
                continue
            cell = g[rs, cs]
            assert not np.any((cell != BLACK) & (cell != RED)), "non-legend cells may only contain black or red"
            seed_count = int(np.count_nonzero(cell == RED))
            assert 0 <= seed_count <= 4, "seed count must stay within the legend range"
            if seed_count == 0:
                out[rs, cs] = BLACK
            else:
                out[rs, cs] = patterns[seed_count - 1]
    return out


def _render_pattern(kind: str, height: int, width: int, primary: int, secondary: int) -> np.ndarray:
    cell = np.full((height, width), BLACK, dtype=int)
    rr, cc = np.indices((height, width))

    if kind == "solid":
        cell[:, :] = primary
    elif kind == "checker":
        cell[(rr + cc) % 2 == 0] = primary
        cell[(rr + cc) % 2 == 1] = secondary
    elif kind == "diag_down":
        cell[(rr - cc) % 3 == 0] = primary
    elif kind == "diag_up":
        cell[(rr + cc) % 3 == 0] = primary
    elif kind == "border":
        cell[0, :] = primary
        cell[-1, :] = primary
        cell[:, 0] = primary
        cell[:, -1] = primary
    elif kind == "cross":
        rows = {height // 2}
        cols = {width // 2}
        if height % 2 == 0:
            rows.add(height // 2 - 1)
        if width % 2 == 0:
            cols.add(width // 2 - 1)
        for row in rows:
            cell[row, :] = primary
        for col in cols:
            cell[:, col] = primary
    elif kind == "vstripe":
        cell[:, ::2] = primary
    elif kind == "hstripe":
        cell[::2, :] = primary
    elif kind == "xshape":
        diag = np.arange(min(height, width))
        cell[diag, diag] = primary
        cell[diag, width - 1 - diag] = primary
    else:
        raise ValueError(f"unknown pattern kind: {kind}")

    assert not np.any(cell == RED)
    assert not np.any(cell == AZURE)
    assert np.any(cell != BLACK), "legend pattern must be visible"
    return cell


def _sample_patterns(height: int, width: int, rng: random.Random) -> list[np.ndarray]:
    for _ in range(200):
        kinds = rng.sample(PATTERN_KINDS, 4)
        primaries = rng.sample(PATTERN_COLORS, 4)
        patterns: list[np.ndarray] = []
        ok = True
        for kind, primary in zip(kinds, primaries):
            secondary_pool = [BLACK] + [color for color in PATTERN_COLORS if color != primary]
            secondary = rng.choice(secondary_pool)
            pattern = _render_pattern(kind, height, width, primary, secondary)
            if any(np.array_equal(pattern, other) for other in patterns):
                ok = False
                break
            patterns.append(pattern)
        if ok:
            return patterns
    raise RuntimeError("failed to sample four distinct patterns")


def _empty_lattice(cell_rows: int, cell_cols: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[slice], list[slice]]:
    height = cell_rows * cell_h + cell_rows + 1
    width = cell_cols * cell_w + cell_cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = [idx * (cell_h + 1) for idx in range(cell_rows + 1)]
    col_lines = [idx * (cell_w + 1) for idx in range(cell_cols + 1)]
    grid[row_lines, :] = AZURE
    grid[:, col_lines] = AZURE

    row_slices = [slice(line + 1, line + 1 + cell_h) for line in row_lines[:-1]]
    col_slices = [slice(line + 1, line + 1 + cell_w) for line in col_lines[:-1]]
    return grid, row_slices, col_slices


def _seed_positions(cell_h: int, cell_w: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    spots = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    assert count <= len(spots)
    return rng.sample(spots, count)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_top = rng.choice([True, False])
    cell_h = rng.randint(4, 7)
    cell_w = rng.randint(4, 7)
    patterns = _sample_patterns(cell_h, cell_w, rng)

    if legend_top:
        data_rows = rng.randint(2, 5)
        cell_rows = data_rows + 1
        cell_cols = 4
        legend_cells = [(0, c) for c in range(4)]
    else:
        data_cols = rng.randint(2, 5)
        cell_rows = 4
        cell_cols = data_cols + 1
        legend_cells = [(r, 0) for r in range(4)]

    grid, row_slices, col_slices = _empty_lattice(cell_rows, cell_cols, cell_h, cell_w)

    for index, (r, c) in enumerate(legend_cells):
        grid[row_slices[r], col_slices[c]] = patterns[index]

    data_cells = [
        (r, c)
        for r in range(cell_rows)
        for c in range(cell_cols)
        if (r, c) not in set(legend_cells)
    ]
    assert len(data_cells) >= 5

    counts = [0, 1, 2, 3, 4]
    counts.extend(rng.randint(0, 4) for _ in range(len(data_cells) - len(counts)))
    rng.shuffle(counts)

    for (r, c), seed_count in zip(data_cells, counts):
        if seed_count == 0:
            continue
        cell = grid[row_slices[r], col_slices[c]]
        for local_r, local_c in _seed_positions(cell_h, cell_w, seed_count, rng):
            cell[local_r, local_c] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
