from __future__ import annotations

import numpy as np


BLACK = 0
GRAY = 5
ORDERS = (4, 2, 1)
ORDER_TO_ROW = {4: 0, 2: 1, 1: 2}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
VALID_LAYOUTS = [(5, 3), (5, 4), (5, 5), (7, 3), (9, 3)]
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start < sep:
            bands.append((start, sep))
        start = sep + 1
    if start < size:
        bands.append((start, size))
    return bands


def _rotation_order(cell: np.ndarray) -> int:
    if np.array_equal(cell, np.rot90(cell, 1)):
        return 4
    if np.array_equal(cell, np.rot90(cell, 2)):
        return 2
    return 1


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)

    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidates = (row_colors & col_colors) - {BLACK}
    assert candidates, "expected a non-black separator color"

    def score(color: int) -> tuple[int, int]:
        rows = sum(int(np.all(grid[r, :] == color)) for r in range(h))
        cols = sum(int(np.all(grid[:, c] == color)) for c in range(w))
        return rows + cols, int(color == GRAY)

    sep = max(candidates, key=score)
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    assert sep_rows and sep_cols, "expected horizontal and vertical separator lines"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) == 3, "puzzle uses three destination rows"
    assert col_bands, "expected at least one column slot"

    row_heights = {r1 - r0 for r0, r1 in row_bands}
    col_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1
    assert next(iter(row_heights)) == next(iter(col_widths)), "sub-regions must be equal squares"

    return int(sep), sep_rows, sep_cols, row_bands, col_bands


def _paint_cell(out: np.ndarray, band_r: tuple[int, int], band_c: tuple[int, int], cell: np.ndarray) -> None:
    r0, r1 = band_r
    c0, c1 = band_c
    out[r0:r1, c0:c1] = cell


def main(grid: np.ndarray) -> np.ndarray:
    """Sort lattice cells by rotational symmetry order and place them into rows 4, 2, 1."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, sep_rows, sep_cols, row_bands, col_bands = _parse_lattice(g)

    grouped: dict[int, list[np.ndarray]] = {4: [], 2: [], 1: []}
    for band_r in row_bands:
        for band_c in col_bands:
            r0, r1 = band_r
            c0, c1 = band_c
            cell = g[r0:r1, c0:c1].copy()
            grouped[_rotation_order(cell)].append(cell)

    out = np.full_like(g, BLACK)
    out[sep_rows, :] = sep
    out[:, sep_cols] = sep

    ncols = len(col_bands)
    for order in ORDERS:
        cells = grouped[order]
        assert len(cells) <= ncols
        dest_row = row_bands[ORDER_TO_ROW[order]]
        for i, cell in enumerate(cells):
            _paint_cell(out, dest_row, col_bands[i], cell)

    return out


def _blank_lattice(cell_size: int, cols: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    rows = 3
    stride = cell_size + 1
    h = rows * cell_size + (rows - 1)
    w = cols * cell_size + (cols - 1)
    grid = np.full((h, w), BLACK, dtype=int)

    sep_rows = [cell_size + i * stride for i in range(rows - 1)]
    sep_cols = [cell_size + i * stride for i in range(cols - 1)]
    grid[sep_rows, :] = GRAY
    grid[:, sep_cols] = GRAY

    row_bands = [(i * stride, i * stride + cell_size) for i in range(rows)]
    col_bands = [(i * stride, i * stride + cell_size) for i in range(cols)]
    return grid, row_bands, col_bands


def _plus_mask(n: int, arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    m[c, c - arm : c + arm + 1] = True
    m[c - arm : c + arm + 1, c] = True
    return m


def _x_mask(n: int, arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    for k in range(-arm, arm + 1):
        m[c + k, c + k] = True
        m[c + k, c - k] = True
    return m


def _diamond_mask(n: int, radius: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    rr, cc = np.indices((n, n))
    m[np.abs(rr - c) + np.abs(cc - c) <= radius] = True
    return m


def _ring_mask(n: int, radius: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    r0, r1 = c - radius, c + radius
    m[r0, r0 : r1 + 1] = True
    m[r1, r0 : r1 + 1] = True
    m[r0 : r1 + 1, r0] = True
    m[r0 : r1 + 1, r1] = True
    return m


def _star_mask(n: int, plus_arm: int, diag_arm: int) -> np.ndarray:
    return _plus_mask(n, plus_arm) | _x_mask(n, diag_arm)


def _hline_mask(n: int, arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    m[c, c - arm : c + arm + 1] = True
    return m


def _vline_mask(n: int, arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    m[c - arm : c + arm + 1, c] = True
    return m


def _diag_main_mask(n: int, arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    for k in range(-arm, arm + 1):
        m[c + k, c + k] = True
    return m


def _diag_anti_mask(n: int, arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    for k in range(-arm, arm + 1):
        m[c + k, c - k] = True
    return m


def _rect_mask(n: int, half_h: int, half_w: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    m[c - half_h : c + half_h + 1, c - half_w : c + half_w + 1] = True
    return m


def _barbell_mask(n: int, arm: int, slope: int) -> np.ndarray:
    m = _hline_mask(n, arm)
    c = n // 2
    m[c - 1, c - arm] = True
    m[c + slope, c + arm] = True
    return m


def _missing_arm_mask(n: int, arm: int, missing: int) -> np.ndarray:
    m = _plus_mask(n, arm)
    c = n // 2
    if missing == 0:
        m[c - arm : c, c] = False
    elif missing == 1:
        m[c + 1 : c + arm + 1, c] = False
    elif missing == 2:
        m[c, c - arm : c] = False
    else:
        m[c, c + 1 : c + arm + 1] = False
    return m


def _elbow_mask(n: int, v_arm: int, h_arm: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    m[c : c + v_arm + 1, c] = True
    m[c + v_arm, c : c + h_arm + 1] = True
    return m


def _hook_mask(n: int, arm: int) -> np.ndarray:
    m = _hline_mask(n, arm)
    c = n // 2
    m[c - 1, c - arm] = True
    return m


def _flag_mask(n: int, arm: int, side_len: int) -> np.ndarray:
    m = _vline_mask(n, arm)
    c = n // 2
    m[c - arm, c : c + side_len + 1] = True
    return m


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        out = np.fliplr(out)
    return out


def _sample_order4_mask(n: int) -> np.ndarray:
    d = n // 2
    choices = [
        lambda: _plus_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _x_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _diamond_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _ring_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _star_mask(
            n,
            int(_RNG.integers(1, d + 1)),
            int(_RNG.integers(1, d + 1)),
        ),
    ]
    for _ in range(100):
        mask = choices[int(_RNG.integers(0, len(choices)))]()
        if _rotation_order(mask.astype(int)) == 4:
            return mask
    raise RuntimeError("failed to sample order-4 mask")


def _sample_order2_mask(n: int) -> np.ndarray:
    d = n // 2
    choices = [
        lambda: _hline_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _vline_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _diag_main_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _diag_anti_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _rect_mask(
            n,
            int(_RNG.integers(0, d + 1)),
            int(_RNG.integers(0, d + 1)),
        ),
        lambda: _barbell_mask(n, int(_RNG.integers(1, d + 1)), int(_RNG.choice([-1, 1]))),
    ]
    for _ in range(200):
        mask = choices[int(_RNG.integers(0, len(choices)))]()
        mask = _random_transform(mask)
        pixels = int(mask.sum())
        if pixels < 3:
            continue
        if _rotation_order(mask.astype(int)) == 2:
            return mask
    raise RuntimeError("failed to sample order-2 mask")


def _sample_order1_mask(n: int) -> np.ndarray:
    d = n // 2
    choices = [
        lambda: _missing_arm_mask(n, int(_RNG.integers(1, d + 1)), int(_RNG.integers(0, 4))),
        lambda: _elbow_mask(
            n,
            int(_RNG.integers(1, d + 1)),
            int(_RNG.integers(1, d + 1)),
        ),
        lambda: _hook_mask(n, int(_RNG.integers(1, d + 1))),
        lambda: _flag_mask(
            n,
            int(_RNG.integers(1, d + 1)),
            int(_RNG.integers(1, d + 1)),
        ),
    ]
    for _ in range(200):
        mask = choices[int(_RNG.integers(0, len(choices)))]()
        mask = _random_transform(mask)
        pixels = int(mask.sum())
        if pixels < 3:
            continue
        if _rotation_order(mask.astype(int)) == 1:
            return mask
    raise RuntimeError("failed to sample order-1 mask")


def _sample_mask(order: int, n: int) -> np.ndarray:
    if order == 4:
        return _sample_order4_mask(n)
    if order == 2:
        return _sample_order2_mask(n)
    return _sample_order1_mask(n)


def _make_colored_cell(order: int, cell_size: int, color: int) -> np.ndarray:
    mask = _sample_mask(order, cell_size)
    cell = np.full((cell_size, cell_size), BLACK, dtype=int)
    cell[mask] = int(color)
    return cell


def _sample_layout(cols: int) -> list[list[int]] | None:
    labels = [4] * cols + [2] * cols + [1] * cols
    target = [[4] * cols, [2] * cols, [1] * cols]
    for _ in range(300):
        perm = list(_RNG.permutation(labels))
        rows = [perm[i * cols : (i + 1) * cols] for i in range(3)]
        if rows == target:
            continue
        if any(len(set(row)) < 2 for row in rows):
            continue
        if any(len({r for r, row in enumerate(rows) if order in row}) < 2 for order in ORDERS):
            continue
        return rows
    return None


def _build_expected(cells_by_order: dict[int, list[np.ndarray]], cell_size: int, cols: int) -> np.ndarray:
    out, row_bands, col_bands = _blank_lattice(cell_size, cols)
    for order in ORDERS:
        row_band = row_bands[ORDER_TO_ROW[order]]
        for i, cell in enumerate(cells_by_order[order]):
            _paint_cell(out, row_band, col_bands[i], cell)
    return out


def _gen_attempt() -> np.ndarray | None:
    cell_size, cols = VALID_LAYOUTS[int(_RNG.integers(0, len(VALID_LAYOUTS)))]
    layout = _sample_layout(cols)
    if layout is None:
        return None

    cells_by_order: dict[int, list[np.ndarray]] = {}
    for order in ORDERS:
        colors = list(_RNG.choice(COLOR_POOL, size=cols, replace=False))
        cells_by_order[order] = [_make_colored_cell(order, cell_size, int(color)) for color in colors]

    expected = _build_expected(cells_by_order, cell_size, cols)

    inp, row_bands, col_bands = _blank_lattice(cell_size, cols)
    queues = {order: list(cells_by_order[order]) for order in ORDERS}
    for ri, row in enumerate(layout):
        for ci, order in enumerate(row):
            cell = queues[order].pop(0)
            _paint_cell(inp, row_bands[ri], col_bands[ci], cell)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
