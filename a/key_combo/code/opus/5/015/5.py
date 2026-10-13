import random
from collections import Counter

import numpy as np


CELL_SIZE = 3
BG = 0
SEP = 3
PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _full_line_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]


def _infer_separator_color(grid: np.ndarray) -> int:
    colors = []
    for r in _full_line_indices(grid, axis=0):
        colors.append(int(grid[r, 0]))
    for c in _full_line_indices(grid, axis=1):
        colors.append(int(grid[0, c]))
    assert colors, "no uniform separator lines found"
    return int(Counter(colors).most_common(1)[0][0])


def _bands_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(set(int(x) for x in separators))
    bands = []
    prev = -1
    for cur in separators + [length]:
        if cur - prev > 1:
            bands.append((prev + 1, cur))
        prev = cur
    return bands


def _parse_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    sep = _infer_separator_color(g)
    row_seps = [r for r in _full_line_indices(g, axis=0) if int(g[r, 0]) == sep]
    col_seps = [c for c in _full_line_indices(g, axis=1) if int(g[0, c]) == sep]
    row_bands = _bands_from_separators(g.shape[0], row_seps)
    col_bands = _bands_from_separators(g.shape[1], col_seps)
    assert len(row_bands) >= 2 and len(col_bands) >= 3
    row_heights = {r1 - r0 for r0, r1 in row_bands}
    col_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1
    cell_h = next(iter(row_heights))
    cell_w = next(iter(col_widths))
    assert cell_h == cell_w == CELL_SIZE
    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_bands:
        row_cells = []
        for c0, c1 in col_bands:
            row_cells.append(g[r0:r1, c0:c1].copy())
        cells.append(row_cells)
    return sep, row_bands, col_bands, cells


def _pattern_key(cell: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in cell.flatten())


def _find_row_anomaly(row_cells: list[np.ndarray]) -> tuple[int, np.ndarray]:
    keys = [_pattern_key(cell) for cell in row_cells]
    counts = Counter(keys)
    unique_keys = [key for key, count in counts.items() if count == 1]
    repeated_keys = [key for key, count in counts.items() if count == len(row_cells) - 1]
    assert len(unique_keys) == 1 and len(repeated_keys) == 1
    idx = keys.index(unique_keys[0])
    return idx, row_cells[idx]


def _render_column(cells: list[np.ndarray], sep: int) -> np.ndarray:
    nrows = len(cells)
    out = np.full((nrows * CELL_SIZE + nrows + 1, CELL_SIZE + 2), int(sep), dtype=int)
    for r, cell in enumerate(cells):
        r0 = 1 + r * (CELL_SIZE + 1)
        out[r0 : r0 + CELL_SIZE, 1 : 1 + CELL_SIZE] = cell
    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _occupied_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _all_transforms(arr: np.ndarray) -> list[np.ndarray]:
    rots = [np.rot90(arr, k) for k in range(4)]
    flips = [np.fliplr(x) for x in rots]
    return rots + flips


def _is_asymmetric(arr: np.ndarray) -> bool:
    return all(not np.array_equal(arr, other) for other in _all_transforms(arr)[1:])


def _random_connected_pattern(rng: random.Random, colors: list[int]) -> np.ndarray:
    while True:
        area = rng.randint(3, 6)
        start = (rng.randrange(CELL_SIZE), rng.randrange(CELL_SIZE))
        shape = {start}
        while len(shape) < area:
            frontier = []
            for r, c in list(shape):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < CELL_SIZE and 0 <= nc < CELL_SIZE and (nr, nc) not in shape:
                        frontier.append((nr, nc))
            if not frontier:
                break
            shape.add(rng.choice(frontier))
        if len(shape) != area:
            continue
        mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        for r, c in shape:
            mask[r, c] = True
        if not _occupied_connected(mask):
            continue

        ncols = min(len(colors), rng.choice([2, 3]))
        chosen = rng.sample(colors, ncols)
        painted = chosen[:]
        while len(painted) < area:
            painted.append(rng.choice(chosen))
        rng.shuffle(painted)

        arr = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
        for (r, c), color in zip(sorted(shape), painted):
            arr[r, c] = int(color)
        if len(set(painted)) < 2:
            continue
        if not _is_asymmetric(arr):
            continue
        return arr


def _render_input(row_patterns: list[list[np.ndarray]]) -> np.ndarray:
    nrows = len(row_patterns)
    ncols = len(row_patterns[0])
    grid = np.full((nrows * CELL_SIZE + nrows + 1, ncols * CELL_SIZE + ncols + 1), SEP, dtype=int)
    for r, row in enumerate(row_patterns):
        for c, cell in enumerate(row):
            r0 = 1 + r * (CELL_SIZE + 1)
            c0 = 1 + c * (CELL_SIZE + 1)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cell
    return grid


def _expected_output(row_patterns: list[list[np.ndarray]], anomaly_cols: list[int]) -> np.ndarray:
    extracted = []
    ncols = len(row_patterns[0])
    for row_cells, anomaly_col in zip(row_patterns, anomaly_cols):
        shifted = row_cells[-anomaly_col:] + row_cells[:-anomaly_col] if anomaly_col else list(row_cells)
        new_idx = (2 * anomaly_col) % ncols
        extracted.append(shifted[new_idx])
    return _render_column(extracted, SEP)


def _validate_instance(grid: np.ndarray, expected: np.ndarray, anomaly_cols: list[int]) -> bool:
    try:
        sep, _row_bands, col_bands, cells = _parse_grid(grid)
        assert sep == SEP
        assert len(cells) == len(anomaly_cols)
        assert len(col_bands) == len(cells[0])
        assert len(set(anomaly_cols)) == len(anomaly_cols)
        assert len(set((2 * c) % len(col_bands) for c in anomaly_cols)) == len(anomaly_cols)
        assert any(c != 0 for c in anomaly_cols)
        anomaly_keys = []
        for row_cells, expected_col in zip(cells, anomaly_cols):
            got_col, anomaly = _find_row_anomaly(row_cells)
            assert got_col == expected_col
            shifted = row_cells[-got_col:] + row_cells[:-got_col] if got_col else list(row_cells)
            shifted_col, shifted_anomaly = _find_row_anomaly(shifted)
            assert shifted_col == (2 * got_col) % len(row_cells)
            assert np.array_equal(anomaly, shifted_anomaly)
            anomaly_keys.append(_pattern_key(anomaly))
        assert len(set(anomaly_keys)) == len(anomaly_keys)
        actual = main(grid)
        assert np.array_equal(actual, expected)
    except AssertionError:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the unique cell pattern in each lattice row, use its column index as a cyclic shift amount, then output the shifted-row outlier cells as a single column.

    Input structure:
    - The grid is a green-separated lattice of equal 3x3 cells, with 1-pixel separator lines.
    - In each lattice row, all cell interiors are identical except one anomalous 3x3 pattern.
    - The anomalous cell column is unique per lattice row.

    Rule:
    - For each lattice row, compare the 3x3 cell interiors and find the pattern that appears exactly once.
    - Let its original column index be C (0-indexed from the left).
    - Cyclically shift that entire lattice row of cell contents C steps to the right, keeping separator lines fixed.
    - After the shift, locate the still-unique cell and extract it; stack the extracted cells in the original row order into a single-column lattice with the same green separators.

    Key insight:
    - The anomalous cell has a dual role: it is the content to be extracted, and its original column index implicitly encodes the row-wise cyclic shift that moves it to a new column before extraction.
    """
    sep, _row_bands, _col_bands, cells = _parse_grid(np.asarray(grid, dtype=int))
    extracted = []
    for row_cells in cells:
        anomaly_col, _anomaly = _find_row_anomaly(row_cells)
        shifted = row_cells[-anomaly_col:] + row_cells[:-anomaly_col] if anomaly_col else list(row_cells)
        shifted_col, shifted_anomaly = _find_row_anomaly(shifted)
        assert shifted_col == (2 * anomaly_col) % len(row_cells)
        extracted.append(shifted_anomaly)
    return _render_column(extracted, sep)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        nrows = rng.randint(2, 4)
        ncols = rng.choice([n for n in (3, 5) if n >= nrows])
        anomaly_cols = rng.sample(range(ncols), nrows)

        row_patterns: list[list[np.ndarray]] = []
        used_patterns: set[tuple[int, ...]] = set()
        ok = True
        for anomaly_col in anomaly_cols:
            row_colors = rng.sample(PALETTE, rng.choice([3, 4]))
            base = _random_connected_pattern(rng, row_colors)
            for _inner in range(80):
                anomaly = _random_connected_pattern(rng, row_colors)
                if _pattern_key(anomaly) == _pattern_key(base):
                    continue
                if np.count_nonzero(base != anomaly) < 2:
                    continue
                if _pattern_key(base) in used_patterns or _pattern_key(anomaly) in used_patterns:
                    continue
                break
            else:
                ok = False
                break
            used_patterns.add(_pattern_key(base))
            used_patterns.add(_pattern_key(anomaly))
            row = [base.copy() for _ in range(ncols)]
            row[anomaly_col] = anomaly.copy()
            row_patterns.append(row)
        if not ok:
            continue

        grid = _render_input(row_patterns)
        expected = _expected_output(row_patterns, anomaly_cols)
        if _validate_instance(grid, expected, anomaly_cols):
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
