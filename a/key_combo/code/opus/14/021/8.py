from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TABLE_SIZE = 4
GENERATION_RETRIES = 400
CELL_SIZE_OPTIONS = (3, 4)


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1, *lines, limit]
    regions = [(a + 1, b - 1) for a, b in zip(bounds, bounds[1:])]
    assert all(r0 <= r1 for r0, r1 in regions)
    return regions


def _parse_table(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BLACK, GREEN, GREY})

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREEN)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREEN)]
    assert len(full_rows) == TABLE_SIZE - 1
    assert len(full_cols) == TABLE_SIZE - 1

    expected_green = np.zeros(g.shape, dtype=bool)
    expected_green[full_rows, :] = True
    expected_green[:, full_cols] = True
    assert np.array_equal(g == GREEN, expected_green)

    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) == TABLE_SIZE
    assert len(col_regions) == TABLE_SIZE

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_regions}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_regions}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    assert row_sizes == col_sizes
    cell_size = next(iter(row_sizes))
    assert cell_size >= 2
    return row_regions, col_regions, int(cell_size)


def _extract_patterns(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    row_regions, col_regions, _cell_size = _parse_table(grid)
    patterns: list[list[np.ndarray]] = []
    for r0, r1 in row_regions:
        row_patterns: list[np.ndarray] = []
        for c0, c1 in col_regions:
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert set(map(int, np.unique(cell))).issubset({BLACK, GREY})
            grey_count = int(np.count_nonzero(cell == GREY))
            assert 1 <= grey_count <= 4
            row_patterns.append((cell == GREY).copy())
        patterns.append(row_patterns)
    return row_regions, col_regions, patterns


def _symmetry_matrix(patterns: list[list[np.ndarray]]) -> np.ndarray:
    size = len(patterns)
    sym = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            sym[i, j] = np.array_equal(patterns[i][j].T, patterns[j][i])
    return sym


def _cell_fill_color(is_symmetric: bool, row_i: int, col_i: int) -> int:
    if not is_symmetric:
        return BLUE
    if row_i in (0, TABLE_SIZE - 1) and col_i in (0, TABLE_SIZE - 1):
        return RED
    if 0 < row_i < TABLE_SIZE - 1 and 0 < col_i < TABLE_SIZE - 1:
        return ORANGE
    return YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each table cell according to whether its sprite matches its transposed partner.

    Input structure:
    - The input is a square black grid partitioned by three full green separator rows and three full green separator columns into a 4x4 table of equal square cells.
    - Each cell interior contains a small grey-on-black sprite with 1 to 4 grey pixels.
    - Cell (i,j) is paired with cell (j,i); diagonal cells pair with themselves.

    Rule:
    - Mark a cell symmetric when its sprite, transposed along the main diagonal, exactly matches the sprite in its transposed partner cell.
    - For diagonal cells, this means the cell's own sprite must equal its own transpose.
    - Preserve the green separators, and fill every non-green pixel in each cell with red for symmetric corners, orange for symmetric strict interiors, yellow for symmetric non-corner edges, and blue for every non-symmetric cell.

    Key insight:
    - The output color is controlled by a two-way interaction: first compare each cell with its transposed partner, then combine that boolean result with the cell's table position.
    """
    g = np.asarray(grid, dtype=int)
    row_regions, col_regions, patterns = _extract_patterns(g)
    sym = _symmetry_matrix(patterns)

    out = g.copy()
    for row_i, (r0, r1) in enumerate(row_regions):
        for col_i, (c0, c1) in enumerate(col_regions):
            out[r0 : r1 + 1, c0 : c1 + 1] = _cell_fill_color(bool(sym[row_i, col_i]), row_i, col_i)
    return out


def _random_mask_with_count(cell_size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    picks = rng.choice(cell_size * cell_size, size=count, replace=False)
    mask[picks // cell_size, picks % cell_size] = True
    return mask


def _random_nonsymmetric_mask(
    cell_size: int,
    rng: np.random.Generator,
    count: int | None = None,
) -> np.ndarray:
    for _ in range(200):
        target = int(rng.integers(1, 5)) if count is None else int(count)
        mask = _random_mask_with_count(cell_size, target, rng)
        if not np.array_equal(mask, mask.T):
            return mask
    raise RuntimeError("failed to sample a non-symmetric mask")


def _random_symmetric_mask(
    cell_size: int,
    rng: np.random.Generator,
    count: int | None = None,
) -> np.ndarray:
    diag_orbits = [[(i, i)] for i in range(cell_size)]
    pair_orbits = [[(r, c), (c, r)] for r in range(cell_size) for c in range(r + 1, cell_size)]

    for _ in range(200):
        target = int(rng.integers(1, 5)) if count is None else int(count)
        remaining = target
        orbit_sizes: list[int] = []
        while remaining > 0:
            if remaining == 1:
                orbit_sizes.append(1)
                remaining -= 1
            elif remaining == 2:
                orbit_sizes.append(int(rng.choice((1, 2))))
                remaining -= orbit_sizes[-1]
            else:
                orbit_sizes.append(int(rng.choice((1, 2))))
                remaining -= orbit_sizes[-1]
        if orbit_sizes.count(1) > len(diag_orbits) or orbit_sizes.count(2) > len(pair_orbits):
            continue

        mask = np.zeros((cell_size, cell_size), dtype=bool)
        diag_idx = rng.choice(len(diag_orbits), size=orbit_sizes.count(1), replace=False) if orbit_sizes.count(1) else []
        pair_idx = rng.choice(len(pair_orbits), size=orbit_sizes.count(2), replace=False) if orbit_sizes.count(2) else []
        for idx in np.atleast_1d(diag_idx):
            if orbit_sizes.count(1):
                for r, c in diag_orbits[int(idx)]:
                    mask[r, c] = True
        for idx in np.atleast_1d(pair_idx):
            if orbit_sizes.count(2):
                for r, c in pair_orbits[int(idx)]:
                    mask[r, c] = True
        if int(mask.sum()) == target and np.array_equal(mask, mask.T):
            return mask
    raise RuntimeError("failed to sample a symmetric mask")


def _random_status_matrix(rng: np.random.Generator) -> np.ndarray:
    corner_pos = ((0, 0), (0, 3), (3, 0), (3, 3))
    edge_pos = ((0, 1), (0, 2), (1, 0), (2, 0), (1, 3), (2, 3), (3, 1), (3, 2))
    inner_pos = ((1, 1), (1, 2), (2, 1), (2, 2))

    for _ in range(1000):
        mat = np.zeros((TABLE_SIZE, TABLE_SIZE), dtype=bool)
        for i in range(TABLE_SIZE):
            mat[i, i] = bool(rng.integers(0, 2))
            for j in range(i + 1, TABLE_SIZE):
                value = bool(rng.integers(0, 2))
                mat[i, j] = value
                mat[j, i] = value

        corners = [bool(mat[r, c]) for r, c in corner_pos]
        edges = [bool(mat[r, c]) for r, c in edge_pos]
        interiors = [bool(mat[r, c]) for r, c in inner_pos]
        diagonal = [bool(mat[i, i]) for i in range(TABLE_SIZE)]
        offdiag = [bool(mat[i, j]) for i in range(TABLE_SIZE) for j in range(i + 1, TABLE_SIZE)]

        if not any(corners) or all(corners):
            continue
        if not any(edges) or all(edges):
            continue
        if not any(interiors) or all(interiors):
            continue
        if not any(diagonal) or all(diagonal):
            continue
        if not any(offdiag) or all(offdiag):
            continue
        return mat
    raise RuntimeError("failed to sample a well-covered symmetry matrix")


def _group_positions(mat: np.ndarray, want: bool) -> tuple[list[int], list[tuple[int, int]]]:
    diag = [i for i in range(TABLE_SIZE) if bool(mat[i, i]) == want]
    offdiag = [(i, j) for i in range(TABLE_SIZE) for j in range(i + 1, TABLE_SIZE) if bool(mat[i, j]) == want]
    return diag, offdiag


def _build_patterns(sym: np.ndarray, cell_size: int, rng: np.random.Generator) -> list[list[np.ndarray]]:
    patterns: list[list[np.ndarray]] = [[np.zeros((cell_size, cell_size), dtype=bool) for _ in range(TABLE_SIZE)] for _ in range(TABLE_SIZE)]

    sym_diag, sym_offdiag = _group_positions(sym, True)
    nonsym_diag, nonsym_offdiag = _group_positions(sym, False)
    forced_sym_diag = int(rng.choice(sym_diag))
    forced_nonsym_diag = int(rng.choice(nonsym_diag))
    forced_transpose_pair = tuple(sym_offdiag[int(rng.integers(0, len(sym_offdiag)))])
    forced_samecount_pair = tuple(nonsym_offdiag[int(rng.integers(0, len(nonsym_offdiag)))])
    shared_diag_count = int(rng.choice((2, 3, 4)))
    samecount_pair_count = int(rng.choice((2, 3, 4)))

    for i in range(TABLE_SIZE):
        if bool(sym[i, i]):
            count = shared_diag_count if i == forced_sym_diag else None
            patterns[i][i] = _random_symmetric_mask(cell_size, rng, count)
        else:
            count = shared_diag_count if i == forced_nonsym_diag else None
            patterns[i][i] = _random_nonsymmetric_mask(cell_size, rng, count)

    for i in range(TABLE_SIZE):
        for j in range(i + 1, TABLE_SIZE):
            if bool(sym[i, j]):
                if (i, j) == forced_transpose_pair:
                    base = _random_nonsymmetric_mask(cell_size, rng)
                else:
                    base = _random_mask_with_count(cell_size, int(rng.integers(1, 5)), rng)
                patterns[i][j] = base
                patterns[j][i] = base.T.copy()
            else:
                if (i, j) == forced_samecount_pair:
                    first = _random_mask_with_count(cell_size, samecount_pair_count, rng)
                    for _ in range(200):
                        second = _random_mask_with_count(cell_size, samecount_pair_count, rng)
                        if not np.array_equal(first.T, second):
                            patterns[i][j] = first
                            patterns[j][i] = second
                            break
                    else:
                        raise RuntimeError("failed to build same-count non-symmetric pair")
                else:
                    for _ in range(200):
                        first = _random_mask_with_count(cell_size, int(rng.integers(1, 5)), rng)
                        second = _random_mask_with_count(cell_size, int(rng.integers(1, 5)), rng)
                        if not np.array_equal(first.T, second):
                            patterns[i][j] = first
                            patterns[j][i] = second
                            break
                    else:
                        raise RuntimeError("failed to build non-symmetric pair")

    return patterns


def _render_grid(patterns: list[list[np.ndarray]]) -> np.ndarray:
    cell_size = int(patterns[0][0].shape[0])
    side = TABLE_SIZE * cell_size + (TABLE_SIZE - 1)
    grid = np.full((side, side), BLACK, dtype=int)

    line_positions = []
    pos = cell_size
    for _ in range(TABLE_SIZE - 1):
        line_positions.append(pos)
        pos += cell_size + 1

    grid[line_positions, :] = GREEN
    grid[:, line_positions] = GREEN

    row_regions, col_regions, _ = _parse_table(grid)
    for i, (r0, r1) in enumerate(row_regions):
        for j, (c0, c1) in enumerate(col_regions):
            grid[r0 : r1 + 1, c0 : c1 + 1][patterns[i][j]] = GREY
    return grid


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _validate_example(grid: np.ndarray) -> bool:
    try:
        row_regions, col_regions, patterns = _extract_patterns(grid)
        sym = _symmetry_matrix(patterns)
        out = main(grid)
    except Exception:
        return False

    if not np.array_equal(sym, sym.T):
        return False

    corners = [bool(sym[r, c]) for r, c in ((0, 0), (0, 3), (3, 0), (3, 3))]
    edges = [bool(sym[r, c]) for r, c in ((0, 1), (0, 2), (1, 0), (2, 0), (1, 3), (2, 3), (3, 1), (3, 2))]
    interiors = [bool(sym[r, c]) for r, c in ((1, 1), (1, 2), (2, 1), (2, 2))]
    diagonal = [bool(sym[i, i]) for i in range(TABLE_SIZE)]
    offdiag_pairs = [(i, j) for i in range(TABLE_SIZE) for j in range(i + 1, TABLE_SIZE)]
    offdiag_status = [bool(sym[i, j]) for i, j in offdiag_pairs]

    if not (any(corners) and not all(corners)):
        return False
    if not (any(edges) and not all(edges)):
        return False
    if not (any(interiors) and not all(interiors)):
        return False
    if not (any(diagonal) and not all(diagonal)):
        return False
    if not (any(offdiag_status) and not all(offdiag_status)):
        return False

    has_transpose_only_pair = any(
        bool(sym[i, j]) and not np.array_equal(patterns[i][j], patterns[j][i])
        for i, j in offdiag_pairs
    )
    has_samecount_nonsym_pair = any(
        not bool(sym[i, j])
        and int(patterns[i][j].sum()) == int(patterns[j][i].sum())
        for i, j in offdiag_pairs
    )
    diagonal_sym_counts = {int(patterns[i][i].sum()) for i in range(TABLE_SIZE) if bool(sym[i, i])}
    diagonal_nonsym_counts = {int(patterns[i][i].sum()) for i in range(TABLE_SIZE) if not bool(sym[i, i])}
    if not has_transpose_only_pair or not has_samecount_nonsym_pair:
        return False
    if not (diagonal_sym_counts & diagonal_nonsym_counts):
        return False

    unique_patterns = {
        _mask_signature(patterns[i][j])
        for i in range(TABLE_SIZE)
        for j in range(TABLE_SIZE)
    }
    if len(unique_patterns) < 6:
        return False

    if set(map(int, np.unique(out))) != {BLUE, RED, GREEN, YELLOW, ORANGE}:
        return False
    for i, (r0, r1) in enumerate(row_regions):
        for j, (c0, c1) in enumerate(col_regions):
            cell = out[r0 : r1 + 1, c0 : c1 + 1]
            if len(set(map(int, np.unique(cell)))) != 1:
                return False
            if int(cell[0, 0]) != _cell_fill_color(bool(sym[i, j]), i, j):
                return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray:
    cell_size = int(rng.choice(CELL_SIZE_OPTIONS))
    sym = _random_status_matrix(rng)
    patterns = _build_patterns(sym, cell_size, rng)
    return _render_grid(patterns)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid transpose-table example")
