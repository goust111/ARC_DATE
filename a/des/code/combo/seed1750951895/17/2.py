import numpy as np


BLUE = 1
BLACK = 0
SEPARATOR = 5
PALETTE = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _full_uniform_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if len(set(map(int, grid[r, :]))) == 1]
    return [int(c) for c in range(grid.shape[1]) if len(set(map(int, grid[:, c]))) == 1]


def _spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(int(x) for x in separators)) + [size]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    row_lines = _full_uniform_indices(g, axis=0)
    col_lines = _full_uniform_indices(g, axis=1)
    assert row_lines and col_lines, "expected separator rows and columns"

    sep_colors = {int(g[r, 0]) for r in row_lines} | {int(g[0, c]) for c in col_lines}
    assert len(sep_colors) == 1, "separator lines must share one color"
    sep = next(iter(sep_colors))

    assert all(np.all(g[r, :] == sep) for r in row_lines), "separator rows must be solid"
    assert all(np.all(g[:, c] == sep) for c in col_lines), "separator cols must be solid"

    row_spans = _spans(g.shape[0], row_lines)
    col_spans = _spans(g.shape[1], col_lines)
    assert row_spans and col_spans, "expected at least one cell span"

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "all cells must have equal size"
    return sep, row_spans, col_spans


def _cells(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[np.ndarray]:
    return [grid[r0 : r1 + 1, c0 : c1 + 1].copy() for r0, r1 in row_spans for c0, c1 in col_spans]


def _consensus_template(cells: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(cells, axis=0)
    h, w = stack.shape[1:]
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            vals, counts = np.unique(stack[:, r, c], return_counts=True)
            out[r, c] = int(vals[np.argmax(counts)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the consensus cell template from a separator-partitioned grid and
    mark the damaged cell's differing pixels in blue.
    """
    g = np.asarray(grid, dtype=int)
    _sep, row_spans, col_spans = _separator_structure(g)
    cells = _cells(g, row_spans, col_spans)
    template = _consensus_template(cells)

    mismatch_counts = [int(np.sum(cell != template)) for cell in cells]
    bad = [i for i, count in enumerate(mismatch_counts) if count > 0]
    assert len(bad) == 1, "expected exactly one damaged cell"

    out = template.copy()
    out[cells[bad[0]] != template] = BLUE
    return out


def _random_template(height: int, width: int) -> np.ndarray:
    color_count = int(np.random.randint(2, min(5, len(PALETTE)) + 1))
    colors = list(map(int, np.random.choice(PALETTE, size=color_count, replace=False)))
    template = np.full((height, width), BLACK, dtype=int)

    fill_count = int(np.random.randint(max(3, (height * width) // 4), max(4, (height * width * 2) // 3) + 1))
    positions = np.random.choice(height * width, size=fill_count, replace=False)
    for pos in positions:
        template.flat[int(pos)] = int(colors[np.random.randint(0, len(colors))])

    if len(set(map(int, template.ravel()))) < 2:
        template.flat[int(positions[0])] = colors[0]
        template.flat[int(positions[-1])] = colors[-1]

    rows_used = int(np.sum(np.any(template != BLACK, axis=1)))
    cols_used = int(np.sum(np.any(template != BLACK, axis=0)))
    if rows_used < 2:
        template[0, 0] = colors[0]
        template[-1, min(1, width - 1)] = colors[-1]
    if cols_used < 2:
        template[0, 0] = colors[0]
        template[min(1, height - 1), -1] = colors[-1]
    return template


def _damage_cell(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    damaged = template.copy()
    h, w = template.shape
    diff_mask = np.zeros((h, w), dtype=bool)
    change_count = int(np.random.randint(1, max(2, (h * w) // 3 + 1)))
    positions = np.random.choice(h * w, size=change_count, replace=False)
    branch = int(np.random.randint(0, 2))
    for pos in positions:
        r, c = divmod(int(pos), w)
        old = int(damaged[r, c])
        if branch == 0:
            new = BLACK if old != BLACK else int(np.random.choice(PALETTE))
        else:
            choices = [BLACK] + [c0 for c0 in PALETTE if int(c0) != old]
            new = int(np.random.choice(choices))
        if new == old:
            continue
        damaged[r, c] = new
        diff_mask[r, c] = True

    if not diff_mask.any():
        r = int(np.random.randint(0, h))
        c = int(np.random.randint(0, w))
        choices = [BLACK] + [c0 for c0 in PALETTE if int(c0) != int(template[r, c])]
        damaged[r, c] = int(np.random.choice(choices))
        diff_mask[r, c] = damaged[r, c] != template[r, c]

    return damaged, diff_mask


def _build_grid(template: np.ndarray, damaged: np.ndarray, bad_index: int, n_rows: int, n_cols: int) -> np.ndarray:
    cell_h, cell_w = template.shape
    grid_h = n_rows * cell_h + (n_rows + 1)
    grid_w = n_cols * cell_w + (n_cols + 1)
    grid = np.full((grid_h, grid_w), SEPARATOR, dtype=int)

    idx = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0 = 1 + rr * (cell_h + 1)
            c0 = 1 + cc * (cell_w + 1)
            tile = damaged if idx == bad_index else template
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = tile
            idx += 1
    return grid


def _expected_output(template: np.ndarray, diff_mask: np.ndarray) -> np.ndarray:
    out = template.copy()
    out[diff_mask] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = int(np.random.randint(2, 5))
    n_cols = int(np.random.randint(2, 5))
    if n_rows * n_cols < 3:
        return None

    cell_h = int(np.random.randint(3, 7))
    cell_w = int(np.random.randint(3, 7))
    template = _random_template(cell_h, cell_w)
    damaged, diff_mask = _damage_cell(template)

    bad_index = int(np.random.randint(0, n_rows * n_cols))
    grid = _build_grid(template, damaged, bad_index, n_rows, n_cols)

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = _expected_output(template, diff_mask)
    if not np.array_equal(out, expected):
        return None
    if BLUE in set(map(int, grid.ravel())):
        return None
    if np.array_equal(damaged, template):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
