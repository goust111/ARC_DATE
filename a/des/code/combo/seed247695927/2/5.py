import numpy as np


BG = 0
MAROON = 9
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 300


def _find_separator_col(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = [c for c in range(w) if np.all(g[:, c] == g[0, c]) and g[0, c] != BG]
    preferred = [c for c in candidates if int(g[0, c]) == MAROON]
    pool = preferred if preferred else candidates
    for c in pool:
        if 0 < c < w - 1 and np.any(g[:, :c] != BG) and np.any(g[:, c + 1 :] != BG):
            return c
    raise AssertionError("expected a full-height separator column")


def _extract_table_structure(panel: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]], int]:
    colors, counts = np.unique(panel[panel != BG], return_counts=True)
    assert len(colors) > 0, "right panel must contain non-background pixels"
    grey = int(colors[np.argmax(counts)])
    grey_mask = panel == grey
    rows, cols = np.where(grey_mask)
    assert len(rows) > 0, "right panel must contain a grey grid"

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    box = panel[r0 : r1 + 1, c0 : c1 + 1]
    box_mask = box == grey

    row_lines = [r for r in range(box.shape[0]) if np.all(box_mask[r, :])]
    col_lines = [c for c in range(box.shape[1]) if np.all(box_mask[:, c])]
    assert len(row_lines) >= 3 and len(col_lines) >= 3, "grid must have outer border plus internal lines"

    row_bands = []
    for a, b in zip(row_lines, row_lines[1:]):
        if b - a > 1:
            row_bands.append((r0 + a + 1, r0 + b - 1))
    col_bands = []
    for a, b in zip(col_lines, col_lines[1:]):
        if b - a > 1:
            col_bands.append((c0 + a + 1, c0 + b - 1))

    assert row_bands and col_bands, "grid must contain cell interiors"
    return (r0, r1, c0, c1), row_bands, col_bands, grey


def main(grid: np.ndarray) -> np.ndarray:
    """
    Left sparse colored pixels define ranked row/column indices; fill the
    corresponding cells of the right grey table while preserving the grid lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    left_rows, left_cols = np.where((left != BG) & (left != MAROON))
    pixels = [(int(r), int(c), int(left[r, c])) for r, c in zip(left_rows, left_cols)]
    assert pixels, "left panel must contain colored pixels"

    active_rows = sorted({r for r, _, _ in pixels})
    active_cols = sorted({c for _, c, _ in pixels})
    assert len(active_rows) == len(pixels), "at most one colored pixel per row"
    assert len(active_cols) == len(pixels), "at most one colored pixel per column"

    _, row_bands, col_bands, grey = _extract_table_structure(right)
    assert len(row_bands) == len(active_rows), "table row count must match active left rows"
    assert len(col_bands) == len(active_cols), "table col count must match active left cols"

    row_rank = {r: i for i, r in enumerate(active_rows)}
    col_rank = {c: j for j, c in enumerate(active_cols)}

    out = g.copy()
    for r, c, color in pixels:
        rr0, rr1 = row_bands[row_rank[r]]
        cc0, cc1 = col_bands[col_rank[c]]
        cell = out[:, sep + 1 :]
        patch = cell[rr0 : rr1 + 1, cc0 : cc1 + 1]
        patch[patch != grey] = color

    return out


def _build_grid(
    height: int,
    left_w: int,
    top_margin: int,
    n: int,
    cell_h: int,
    cell_w: int,
    v_gap_left: int,
    bottom_margin: int,
    right_margin: int,
    rows: np.ndarray,
    cols: np.ndarray,
    perm: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    grid_h = top_margin + n * cell_h + (n + 1) + bottom_margin
    assert grid_h == height
    grid_w = left_w + 1 + v_gap_left + n * cell_w + (n + 1) + right_margin
    g = np.zeros((grid_h, grid_w), dtype=int)
    g[:, left_w] = MAROON

    right_r0 = top_margin
    right_c0 = left_w + 1 + v_gap_left
    right_h = n * cell_h + (n + 1)
    right_w = n * cell_w + (n + 1)
    g[right_r0 : right_r0 + right_h, right_c0 : right_c0 + right_w] = BG

    for k in range(n + 1):
        rr = right_r0 + k * (cell_h + 1)
        g[rr, right_c0 : right_c0 + right_w] = GREY
        cc = right_c0 + k * (cell_w + 1)
        g[right_r0 : right_r0 + right_h, cc] = GREY

    for idx, (r, c, color) in enumerate(zip(rows, cols[perm], colors)):
        assert idx < n
        g[int(r), int(c)] = int(color)

    return g


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n = int(rng.integers(2, 7))
    cell_h = int(rng.integers(2, 5))
    cell_w = int(rng.integers(2, 5))
    top_margin = int(rng.integers(0, 3))
    bottom_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    v_gap_left = int(rng.integers(0, 3))

    right_h = n * cell_h + (n + 1)
    height = top_margin + right_h + bottom_margin
    min_left_h = max(n + 2, 5)
    if height < min_left_h:
        return None

    left_w = int(rng.integers(max(n + 2, 5), max(n + 6, 9)))
    if left_w <= n:
        return None

    rows = np.sort(rng.choice(height, size=n, replace=False))
    cols = np.sort(rng.choice(left_w, size=n, replace=False))
    perm = rng.permutation(n)

    palette_size = int(rng.integers(2, min(len(PALETTE), n) + 1))
    palette = rng.choice(PALETTE, size=palette_size, replace=False)
    colors = rng.choice(palette, size=n, replace=True)

    g = _build_grid(
        height,
        left_w,
        top_margin,
        n,
        cell_h,
        cell_w,
        v_gap_left,
        bottom_margin,
        right_margin,
        rows,
        cols,
        perm,
        colors,
    )

    try:
        out = main(g)
    except AssertionError:
        return None

    if np.array_equal(out, g):
        return None

    changed = out != g
    if np.any(changed[:, : left_w + 1]):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
