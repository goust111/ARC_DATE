import random

import numpy as np


BG = 0
LINE = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask.tolist():
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _grid_lines(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_runs = np.array([_longest_true_run(row != BG) for row in grid], dtype=int)
    col_runs = np.array([_longest_true_run(col != BG) for col in grid.T], dtype=int)

    line_rows = np.flatnonzero(row_runs == row_runs.max())
    line_cols = np.flatnonzero(col_runs == col_runs.max())

    assert len(line_rows) >= 2 and len(line_cols) >= 2, "expected a rectangular lattice"
    if len(line_rows) > 2:
        row_gaps = np.diff(line_rows)
        assert np.all(row_gaps == row_gaps[0]), "horizontal lines must be equally spaced"
    if len(line_cols) > 2:
        col_gaps = np.diff(line_cols)
        assert np.all(col_gaps == col_gaps[0]), "vertical lines must be equally spaced"
    return line_rows, line_cols


def _render_patch(seed_color: int, hole_mask: np.ndarray) -> np.ndarray:
    mask = np.array(hole_mask, dtype=bool, copy=True)
    if int(mask.sum()) % 2 == 1:
        mask = mask[::-1, :]
    patch = np.full(mask.shape, seed_color, dtype=int)
    patch[mask] = BG
    return patch


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    line_rows, line_cols = _grid_lines(g)
    out = g.copy()

    line_row_set = set(map(int, line_rows))
    line_col_set = set(map(int, line_cols))
    next_row = {int(line_rows[i]): int(line_rows[i + 1]) for i in range(len(line_rows) - 1)}
    next_col = {int(line_cols[i]): int(line_cols[i + 1]) for i in range(len(line_cols) - 1)}

    for r, c in np.argwhere((g != BG) & (g != LINE)):
        rr = int(r)
        cc = int(c)
        if rr not in line_row_set or cc not in line_col_set:
            continue
        if rr not in next_row or cc not in next_col:
            continue

        r1 = next_row[rr]
        c1 = next_col[cc]
        inner = g[rr + 1 : r1, cc + 1 : c1]
        hole_mask = inner == LINE
        out[rr + 1 : r1, cc + 1 : c1] = _render_patch(int(g[rr, cc]), hole_mask)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the grid cell to the lower-right of each colored lattice intersection, mirroring the hole pattern when the marker count is odd.

    Input structure:
    - Black background with a one-pixel-thick grey rectangular lattice made from equally spaced horizontal and vertical lines.
    - Some lattice intersections are replaced by unique colored seed pixels.
    - Cell interiors are otherwise black, except for scattered grey pixels that mark hole positions inside some cells.

    Rule:
    - For each colored seed on a lattice intersection, select the cell whose top-left corner is that intersection.
    - Treat every grey interior pixel in that cell as a hole marker; the rest of the cell interior is fillable.
    - Fill the cell interior with the seed color, leaving the hole-marker locations black instead of colored.
    - If the number of hole markers in that cell is odd, vertically mirror the hole pattern within the cell before applying it.
    - Leave all lattice lines, seed pixels, and cells without a seed unchanged.

    Key insight:
    - The colored intersection tells you which cell to paint, while the parity of the grey markers inside that cell decides whether the hole mask is used directly or flipped top-to-bottom.
    """
    return _transform(grid)


def _sample_holes(
    rng: random.Random,
    height: int,
    width: int,
    count: int,
    require_flip_visible: bool,
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    if count == 0:
        return []

    for _ in range(200):
        holes = rng.sample(cells, count)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in holes:
            mask[r, c] = True
        if require_flip_visible and np.array_equal(mask, mask[::-1, :]):
            continue
        return holes
    return holes


def _place_markers(
    grid: np.ndarray,
    top: int,
    left: int,
    holes: list[tuple[int, int]],
) -> None:
    for r, c in holes:
        grid[top + r, left + c] = LINE


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(2, 5)
    cell_w = rng.randint(2, 5)
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 4)
    margin_top = rng.randint(0, 2)
    margin_bottom = rng.randint(0, 2)
    margin_left = rng.randint(0, 2)
    margin_right = rng.randint(0, 2)

    grid_h = margin_top + (n_rows + 1) + n_rows * cell_h + margin_bottom
    grid_w = margin_left + (n_cols + 1) + n_cols * cell_w + margin_right
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    line_rows = [margin_top + i * (cell_h + 1) for i in range(n_rows + 1)]
    line_cols = [margin_left + i * (cell_w + 1) for i in range(n_cols + 1)]
    r0, r1 = line_rows[0], line_rows[-1]
    c0, c1 = line_cols[0], line_cols[-1]
    grid[np.ix_(line_rows, range(c0, c1 + 1))] = LINE
    grid[np.ix_(range(r0, r1 + 1), line_cols)] = LINE

    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    max_seed_count = min(len(all_cells), len(SEED_COLORS), 6)
    seed_count = rng.randint(1, max_seed_count)
    seed_cells = rng.sample(all_cells, seed_count)
    seed_palette = rng.sample(list(SEED_COLORS), seed_count)

    parities = [rng.randint(0, 1) for _ in range(seed_count)]
    if seed_count >= 2 and len(set(parities)) == 1:
        parities[0] ^= 1
    rng.shuffle(parities)

    for idx, (cell_r, cell_c) in enumerate(seed_cells):
        parity = parities[idx]
        area = cell_h * cell_w
        if parity == 0:
            counts = [k for k in (0, 2, 4) if k <= area]
            if seed_count == 1:
                counts = [k for k in counts if k != 0] or counts
        else:
            counts = [k for k in (1, 3, 5) if k <= area]
        if not counts:
            return None
        hole_count = rng.choice(counts)
        holes = _sample_holes(rng, cell_h, cell_w, hole_count, require_flip_visible=parity == 1)
        top = line_rows[cell_r] + 1
        left = line_cols[cell_c] + 1
        _place_markers(grid, top, left, holes)
        grid[line_rows[cell_r], line_cols[cell_c]] = seed_palette[idx]

    distractor_cells = [cell for cell in all_cells if cell not in seed_cells]
    rng.shuffle(distractor_cells)
    distractor_budget = min(len(distractor_cells), rng.randint(0, max(1, len(distractor_cells) // 2 + 1)))
    for cell_r, cell_c in distractor_cells[:distractor_budget]:
        hole_count = rng.choice((1, 2, 3))
        if hole_count > cell_h * cell_w:
            continue
        holes = _sample_holes(rng, cell_h, cell_w, hole_count, require_flip_visible=False)
        top = line_rows[cell_r] + 1
        left = line_cols[cell_c] + 1
        _place_markers(grid, top, left, holes)

    try:
        out = _transform(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
