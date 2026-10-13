import numpy as np

from arc_utils import get_bg


def _uniform_line_color(g: np.ndarray) -> int:
    row_colors = {
        int(g[r, 0])
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != 0
    }
    col_colors = {
        int(g[0, c])
        for c in range(g.shape[1])
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != 0
    }
    line_colors = row_colors & col_colors
    assert len(line_colors) == 1, line_colors
    return next(iter(line_colors))


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    edges = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(edges, edges[1:])]
    assert all(start < end for start, end in bands), bands
    assert len({end - start for start, end in bands}) == 1, bands
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the unique partition cell whose foreground pixels are all one color.

    Input structure
    - A black-background square grid is divided by full horizontal and vertical
      separator lines of one non-black color into equal square cells.
    - Most cells contain scattered foreground noise of several colors.
    - Exactly one cell has non-background pixels of a single color only.

    Rule
    - Ignore the separator lines and split the grid into its equal cells.
    - Return the unique cell whose non-background pixels are monochromatic,
      preserving its black pixels and foreground color.

    Key insight
    - The answer is not built from the whole grid; it is the only separator-free
      cell that is internally color-consistent, while the other cells mix colors.
    """
    g = np.asarray(grid, dtype=int)
    line_color = _uniform_line_color(g)

    row_separators = [
        r for r in range(g.shape[0]) if np.all(g[r, :] == line_color)
    ]
    col_separators = [
        c for c in range(g.shape[1]) if np.all(g[:, c] == line_color)
    ]
    assert row_separators and col_separators

    row_bands = _bands(g.shape[0], row_separators)
    col_bands = _bands(g.shape[1], col_separators)
    cell_heights = {end - start for start, end in row_bands}
    cell_widths = {end - start for start, end in col_bands}
    assert len(cell_heights) == 1 and len(cell_widths) == 1
    assert next(iter(cell_heights)) == next(iter(cell_widths))

    cell_rows = [r for start, end in row_bands for r in range(start, end)]
    cell_cols = [c for start, end in col_bands for c in range(start, end)]
    bg = get_bg(g[np.ix_(cell_rows, cell_cols)])
    assert bg != line_color

    candidates = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0:r1, c0:c1]
            fg_colors = set(map(int, np.unique(cell))) - {bg}
            if len(fg_colors) == 1:
                candidates.append(cell)

    assert len(candidates) == 1, len(candidates)
    return candidates[0].copy()

# --------------- generator (o5x) ---------------
# Canvas and palette bounds.
N_CELLS_RANGE = (4, 7)
CELL_SIDE_RANGE = (3, 6)
BACKGROUND_COLOR = 0
COLOR_RANGE = (1, 10)
FOREGROUND_COLORS = np.arange(*COLOR_RANGE)
# Foreground fill constraints.
MIN_FG_PIXELS = 2
MAX_FG_PERCENT = 44
TARGET_MIN_FG_PERCENT = 24
MAX_DISTRACTOR_COLORS = 7
# Generator retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _make_target_cell(cell_side: int, target_color: int) -> tuple[np.ndarray, int]:
    """Return the monochrome answer cell plus the shared foreground upper bound."""
    area = cell_side * cell_side
    max_fg = max(MIN_FG_PIXELS, (area * MAX_FG_PERCENT + 99) // 100)
    target_min_fg = max(MIN_FG_PIXELS, (area * TARGET_MIN_FG_PERCENT + 99) // 100)
    target_fg_count = int(np.random.randint(target_min_fg, max_fg + 1))
    target_cell = np.zeros((cell_side, cell_side), dtype=int)
    target_pos = np.random.choice(area, size=target_fg_count, replace=False)
    target_cell.flat[target_pos] = target_color
    return target_cell, max_fg

def _build_distractor_specs(n_cells: int, target_band_r: int, target_band_c: int, max_fg: int, target_color: int) -> list:
    """Sample non-target cell fill counts, color counts, and required color seeds."""
    cell_specs = []
    for br in range(n_cells):
        for bc in range(n_cells):
            if br == target_band_r and bc == target_band_c:
                continue
            fg_count = int(np.random.randint(2, max_fg + 1))
            color_count = int(np.random.randint(2, min(MAX_DISTRACTOR_COLORS, fg_count, len(FOREGROUND_COLORS)) + 1))
            cell_specs.append([br, bc, fg_count, color_count, set()])
    missing_colors = [color for color in range(*COLOR_RANGE) if color != target_color]
    np.random.shuffle(missing_colors)
    np.random.shuffle(cell_specs)
    for spec, color in zip(cell_specs, missing_colors):
        spec[4].add(int(color))
    return cell_specs

def _fill_distractors(grid: np.ndarray, cell_specs: list, cell_side: int) -> None:
    """Render each sampled distractor cell into its grid slot."""
    area = cell_side * cell_side
    for br, bc, fg_count, color_count, colors in cell_specs:
        while len(colors) < color_count:
            colors.add(int(np.random.choice(FOREGROUND_COLORS)))
        colors = list(colors)
        np.random.shuffle(colors)
        cell = np.zeros((cell_side, cell_side), dtype=int)
        pos = np.random.choice(area, size=fg_count, replace=False)
        cell.flat[pos[: len(colors)]] = colors
        if fg_count > len(colors):
            cell.flat[pos[len(colors):]] = np.random.choice(colors, size=fg_count - len(colors), replace=True)
        r0, c0 = br * (cell_side + 1), bc * (cell_side + 1)
        grid[r0 : r0 + cell_side, c0 : c0 + cell_side] = cell

def _gen_attempt() -> np.ndarray | None:
    n_cells = int(np.random.randint(*N_CELLS_RANGE))
    cell_side = int(np.random.randint(*CELL_SIDE_RANGE))
    grid_side = n_cells * cell_side + n_cells - 1
    sep_color = int(np.random.randint(*COLOR_RANGE))
    grid = np.full((grid_side, grid_side), BACKGROUND_COLOR, dtype=int)
    for band in range(1, n_cells):
        sep = band * cell_side + band - 1
        grid[sep, :] = sep_color
        grid[:, sep] = sep_color
    target_band_r = int(np.random.randint(n_cells))
    target_band_c = int(np.random.randint(n_cells))
    target_color = int(np.random.randint(*COLOR_RANGE))
    target_cell, max_fg = _make_target_cell(cell_side, target_color)
    cell_specs = _build_distractor_specs(n_cells, target_band_r, target_band_c, max_fg, target_color)
    _fill_distractors(grid, cell_specs, cell_side)
    r0, c0 = target_band_r * (cell_side + 1), target_band_c * (cell_side + 1)
    grid[r0 : r0 + cell_side, c0 : c0 + cell_side] = target_cell
    try:
        if not np.array_equal(main(grid), target_cell):
            return None
    except Exception:
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
