from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

GRID_CELL_COUNT = 3
CELL_SIZE = 5
MAX_GENERATION_ATTEMPTS = 300


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    separator_set = set(map(int, separators))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in separator_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _cell_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = np.flatnonzero(np.all(grid == GREY, axis=1))
    col_lines = np.flatnonzero(np.all(grid == GREY, axis=0))
    row_runs = _open_runs(grid.shape[0], row_lines)
    col_runs = _open_runs(grid.shape[1], col_lines)

    assert len(row_runs) == GRID_CELL_COUNT, "expected a 3-row cell lattice"
    assert len(col_runs) == GRID_CELL_COUNT, "expected a 3-column cell lattice"
    row_sizes = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert row_sizes == {CELL_SIZE}, "expected 5-cell-tall lattice cells"
    assert col_sizes == {CELL_SIZE}, "expected 5-cell-wide lattice cells"
    return row_runs, col_runs


def _extract_cell(
    grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]], cell_r: int, cell_c: int
) -> np.ndarray:
    r0, r1 = row_runs[cell_r]
    c0, c1 = col_runs[cell_c]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _seed_cell(
    grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]
) -> tuple[int, int, np.ndarray]:
    hits: list[tuple[int, int, np.ndarray]] = []
    for cell_r in range(GRID_CELL_COUNT):
        for cell_c in range(GRID_CELL_COUNT):
            cell = _extract_cell(grid, row_runs, col_runs, cell_r, cell_c)
            if np.any((cell != BLACK) & (cell != GREY)):
                hits.append((cell_r, cell_c, cell))
    assert len(hits) == 1, "expected exactly one non-empty seed cell"
    return hits[0]


def _build_template(seed_cell: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    assert seed_cell.shape == (CELL_SIZE, CELL_SIZE)
    allowed = {BLACK, BLUE, RED}
    assert set(map(int, np.unique(seed_cell))) <= allowed, "seed cell may only contain black, blue, and red"

    blue_cols = np.flatnonzero(np.any(seed_cell == BLUE, axis=0)).tolist()
    red_rows = np.flatnonzero(np.any(seed_cell == RED, axis=1)).tolist()
    assert 1 <= len(blue_cols) <= 3, "blue markers must select 1-3 columns"
    assert 1 <= len(red_rows) <= 2, "red markers must select 1-2 rows"
    assert any(seed_cell[r, c] == BLACK for r in red_rows for c in blue_cols), "need at least one black row/col intersection"

    template = np.zeros_like(seed_cell)
    template[:, blue_cols] = BLUE
    template[red_rows, :] = RED
    for r in red_rows:
        template[r, blue_cols] = YELLOW
    return template, blue_cols, red_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a row-column stamp from the seed cell's blue columns and red rows, then broadcast it to same-parity cells.

    Input structure:
    - Grey full-line separators divide the canvas into a 3x3 lattice of equal 5x5 cells, optionally with a grey outer border.
    - Exactly one seed cell contains sparse blue marker pixels and sparse red marker pixels on an otherwise black cell background.
    - Blue markers occupy 1-3 local columns, red markers occupy 1-2 local rows, and their intersections are not directly drawn.

    Rule:
    - Detect the 3x3 cell lattice from the full grey separator rows and columns.
    - Find the unique non-empty seed cell, collect every local column containing blue and every local row containing red, and build a 5x5 template:
      intersections of selected rows and columns become yellow, blue-only selected-column cells stay blue, red-only selected-row cells stay red, and the rest stay black.
    - Replace the seed cell with that template and copy the same template into every other completely empty cell whose cell-index parity matches the seed cell parity.

    Key insight:
    - The blue and red pixels are instructions in cell coordinates rather than output pixels; they independently activate local columns and rows whose logical AND becomes yellow, and the finished template is then broadcast by lattice parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_runs, col_runs = _cell_runs(g)
    seed_r, seed_c, seed_cell = _seed_cell(g, row_runs, col_runs)
    template, _blue_cols, _red_rows = _build_template(seed_cell)

    parity = (seed_r + seed_c) % 2
    out = g.copy()
    for cell_r in range(GRID_CELL_COUNT):
        for cell_c in range(GRID_CELL_COUNT):
            cell = _extract_cell(g, row_runs, col_runs, cell_r, cell_c)
            if (cell_r, cell_c) == (seed_r, seed_c):
                should_stamp = True
            else:
                should_stamp = (cell_r + cell_c) % 2 == parity and np.all(cell == BLACK)
            if not should_stamp:
                continue
            r0, r1 = row_runs[cell_r]
            c0, c1 = col_runs[cell_c]
            out[r0 : r1 + 1, c0 : c1 + 1] = template

    return out


def _layout(border: bool) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    if border:
        size = GRID_CELL_COUNT * CELL_SIZE + GRID_CELL_COUNT + 1
        separators = [0, 6, 12, 18]
    else:
        size = GRID_CELL_COUNT * CELL_SIZE + GRID_CELL_COUNT - 1
        separators = [5, 11]
    grid = np.zeros((size, size), dtype=int)
    grid[separators, :] = GREY
    grid[:, separators] = GREY
    row_runs = _open_runs(size, np.array(separators, dtype=int))
    col_runs = _open_runs(size, np.array(separators, dtype=int))
    return grid, row_runs, col_runs


def _sample_markers(rng: np.random.Generator) -> tuple[list[int], list[int], set[tuple[int, int]], set[tuple[int, int]]]:
    blue_count = int(rng.integers(1, 4))
    red_count = int(rng.integers(1, 3))
    blue_cols = sorted(map(int, rng.choice(CELL_SIZE, size=blue_count, replace=False)))
    red_rows = sorted(map(int, rng.choice(CELL_SIZE, size=red_count, replace=False)))

    blue_only_rows = [r for r in range(CELL_SIZE) if r not in red_rows]
    red_only_cols = [c for c in range(CELL_SIZE) if c not in blue_cols]
    assert blue_only_rows and red_only_cols

    blue_pixels: set[tuple[int, int]] = set()
    red_pixels: set[tuple[int, int]] = set()

    for idx, col in enumerate(blue_cols):
        min_count = 2 if len(blue_cols) == 1 and idx == 0 else 1
        count = int(rng.integers(min_count, len(blue_only_rows) + 1))
        rows = rng.choice(blue_only_rows, size=count, replace=False)
        for row in rows:
            blue_pixels.add((int(row), col))

    for idx, row in enumerate(red_rows):
        min_count = 2 if len(red_rows) == 1 and idx == 0 else 1
        count = int(rng.integers(min_count, len(red_only_cols) + 1))
        cols = rng.choice(red_only_cols, size=count, replace=False)
        for col in cols:
            red_pixels.add((row, int(col)))

    return blue_cols, red_rows, blue_pixels, red_pixels


def _template_from_sets(blue_cols: list[int], red_rows: list[int]) -> np.ndarray:
    template = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    template[:, blue_cols] = BLUE
    template[red_rows, :] = RED
    for row in red_rows:
        template[row, blue_cols] = YELLOW
    return template


def _expected_output(
    inp: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    seed_r: int,
    seed_c: int,
    template: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    parity = (seed_r + seed_c) % 2
    for cell_r in range(GRID_CELL_COUNT):
        for cell_c in range(GRID_CELL_COUNT):
            r0, r1 = row_runs[cell_r]
            c0, c1 = col_runs[cell_c]
            cell = inp[r0 : r1 + 1, c0 : c1 + 1]
            if (cell_r, cell_c) == (seed_r, seed_c):
                out[r0 : r1 + 1, c0 : c1 + 1] = template
            elif (cell_r + cell_c) % 2 == parity:
                assert np.all(cell == BLACK), "non-seed parity targets must be empty in generated inputs"
                out[r0 : r1 + 1, c0 : c1 + 1] = template
            else:
                assert np.all(cell == BLACK), "non-target cells must stay empty in generated inputs"
    return out


def _validate_generated_instance(
    inp: np.ndarray,
    *,
    seed_r: int,
    seed_c: int,
    blue_cols: list[int],
    red_rows: list[int],
    border: bool,
) -> bool:
    row_runs, col_runs = _cell_runs(inp)
    parsed_seed_r, parsed_seed_c, seed_cell = _seed_cell(inp, row_runs, col_runs)
    if (parsed_seed_r, parsed_seed_c) != (seed_r, seed_c):
        return False

    expected_shape = (19, 19) if border else (17, 17)
    if inp.shape != expected_shape:
        return False

    parsed_template, parsed_blue_cols, parsed_red_rows = _build_template(seed_cell)
    if parsed_blue_cols != blue_cols or parsed_red_rows != red_rows:
        return False

    if any(seed_cell[r, c] != BLACK for r in red_rows for c in blue_cols):
        return False

    expected = _expected_output(inp, row_runs, col_runs, seed_r, seed_c, parsed_template)
    got = main(inp)
    if not np.array_equal(got, expected):
        return False

    target_cells = 0
    parity = (seed_r + seed_c) % 2
    for cell_r in range(GRID_CELL_COUNT):
        for cell_c in range(GRID_CELL_COUNT):
            r0, r1 = row_runs[cell_r]
            c0, c1 = col_runs[cell_c]
            out_cell = got[r0 : r1 + 1, c0 : c1 + 1]
            if (cell_r + cell_c) % 2 == parity:
                target_cells += 1
                if not np.array_equal(out_cell, parsed_template):
                    return False
            elif np.any(out_cell != BLACK):
                return False
    if target_cells < 4:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    border = bool(rng.integers(0, 2))
    inp, row_runs, col_runs = _layout(border)
    seed_r = int(rng.integers(0, GRID_CELL_COUNT))
    seed_c = int(rng.integers(0, GRID_CELL_COUNT))

    blue_cols, red_rows, blue_pixels, red_pixels = _sample_markers(rng)
    template = _template_from_sets(blue_cols, red_rows)

    r0, r1 = row_runs[seed_r]
    c0, c1 = col_runs[seed_c]
    assert (r1 - r0 + 1, c1 - c0 + 1) == (CELL_SIZE, CELL_SIZE)

    for local_r, local_c in blue_pixels:
        inp[r0 + local_r, c0 + local_c] = BLUE
    for local_r, local_c in red_pixels:
        inp[r0 + local_r, c0 + local_c] = RED

    if not _validate_generated_instance(
        inp,
        seed_r=seed_r,
        seed_c=seed_c,
        blue_cols=blue_cols,
        red_rows=red_rows,
        border=border,
    ):
        return None

    out = main(inp)
    if np.count_nonzero(out == YELLOW) != len(blue_cols) * len(red_rows) * (5 if (seed_r + seed_c) % 2 == 0 else 4):
        return None
    if np.array_equal(inp, out):
        return None
    if len(np.unique(template)) < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
