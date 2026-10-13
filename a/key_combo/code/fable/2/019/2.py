from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider_structure(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _get_bg(g)
    candidates: list[tuple[int, list[int], list[int], int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        row_lines = [r for r in range(height) if width > 1 and np.all(g[r, 1:] == color)]
        col_lines = [c for c in range(1, width) if np.all(g[:, c] == color)]
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        if row_lines[0] != 0 or row_lines[-1] != height - 1:
            continue
        if col_lines[0] != 1 or col_lines[-1] != width - 1:
            continue

        row_gaps = [row_lines[k + 1] - row_lines[k] - 1 for k in range(len(row_lines) - 1)]
        col_gaps = [col_lines[k + 1] - col_lines[k] - 1 for k in range(len(col_lines) - 1)]
        if not row_gaps or not col_gaps:
            continue
        if len(set(row_gaps)) != 1 or len(set(col_gaps)) != 1:
            continue

        m = int(row_gaps[0])
        n = int(col_gaps[0])
        if m <= 0 or n <= 0:
            continue
        if len(row_lines) != m + 1 or len(col_lines) != n + 1:
            continue
        candidates.append((color, row_lines, col_lines, m, n))

    assert len(candidates) == 1, "expected one divider color with consistent m-by-n cell geometry"
    divider, row_lines, col_lines, m, n = candidates[0]
    return bg, divider, row_lines, col_lines, m, n


def _cell_slices(
    row_lines: list[int],
    col_lines: list[int],
    cell_r: int,
    cell_c: int,
) -> tuple[slice, slice]:
    return (
        slice(row_lines[cell_r] + 1, row_lines[cell_r + 1]),
        slice(col_lines[cell_c] + 1, col_lines[cell_c + 1]),
    )


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], int, int, list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, divider, row_lines, col_lines, m, n = _find_divider_structure(g)
    row_colors: list[int] = []
    for cell_r in range(m):
        band = range(row_lines[cell_r] + 1, row_lines[cell_r + 1])
        marker_rows = [r for r in band if int(g[r, 0]) != bg]
        assert len(marker_rows) == 1, "expected exactly one row-marker pixel per cell row"
        color = int(g[marker_rows[0], 0])
        assert color not in (bg, divider), "row-marker color must differ from background and divider"
        row_colors.append(color)

    non_empty_cells: list[np.ndarray] = []
    for cell_r in range(m):
        for cell_c in range(n):
            rs, cs = _cell_slices(row_lines, col_lines, cell_r, cell_c)
            patch = g[rs, cs]
            assert patch.shape == (m, n), "each cell interior must be exactly m-by-n pixels"
            fg_mask = patch != bg
            if not fg_mask.any():
                continue
            fg_colors = set(map(int, np.unique(patch[fg_mask])))
            assert len(fg_colors) == 1, "template cell must contain a monochrome mask"
            assert divider not in fg_colors, "divider color cannot appear inside a cell interior"
            non_empty_cells.append(fg_mask.copy())

    assert len(non_empty_cells) == 1, "expected exactly one non-empty template cell"
    return bg, divider, row_lines, col_lines, m, n, row_colors, non_empty_cells[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique template mask as both stamp content and placement map inside the divider grid.

    Input structure
    - A dominant background surrounds one divider lattice whose full rows and columns split the board into m rows and n columns of equal cells.
    - The lattice has one extra marker column on the far left, with exactly one colored row-marker pixel for each cell row.
    - Every cell interior is exactly m by n pixels; exactly one cell contains a monochrome template mask and all other cell interiors are empty.

    Rule
    - Read the template cell as an m by n boolean mask.
    - Clear every cell interior, including the template cell itself, back to the background color.
    - For each foreground mask position (i, j), stamp the whole template mask into cell row i, column j, recolored to the marker color of row i.
    - Keep the divider lines and the left-side row markers unchanged.

    Key insight
    - The template is self-referential: its foreground pixels say which cells receive copies, while the copied shape is that same template mask recolored by the destination row palette.
    """
    g = np.asarray(grid, dtype=int)
    bg, _divider, row_lines, col_lines, m, n, row_colors, mask = _parse_scene(g)

    out = g.copy()
    for cell_r in range(m):
        for cell_c in range(n):
            rs, cs = _cell_slices(row_lines, col_lines, cell_r, cell_c)
            out[rs, cs] = bg

    for cell_r, row_color in enumerate(row_colors):
        if not bool(mask[cell_r].any()):
            continue
        stamp = np.full((m, n), bg, dtype=int)
        stamp[mask] = int(row_color)
        for cell_c in range(n):
            if not bool(mask[cell_r, cell_c]):
                continue
            rs, cs = _cell_slices(row_lines, col_lines, cell_r, cell_c)
            out[rs, cs] = stamp

    return out


_RNG = np.random.default_rng()
_ALL_COLORS = list(range(10))
_SIZE_OPTIONS = ((3, 4), (4, 3), (4, 4))
_GEN_ATTEMPTS = 400


def _sample_template_mask(m: int, n: int) -> np.ndarray:
    while True:
        mask = _RNG.random((m, n)) < 0.55
        if int(mask.sum()) < 2 or int(mask.sum()) >= m * n:
            continue
        if any(int(mask[r].sum()) == 0 for r in range(m)):
            continue
        if not any(int(mask[r].sum()) < n for r in range(m)):
            continue
        if not any(int(mask[r].sum()) >= 2 for r in range(m)):
            continue
        if not any(int(mask[:, c].sum()) < m for c in range(n)):
            continue
        return mask


def _render_input(
    bg: int,
    divider: int,
    row_colors: list[int],
    template_color: int,
    mask: np.ndarray,
    template_cell: tuple[int, int],
) -> np.ndarray:
    m, n = map(int, mask.shape)
    height = m * m + m + 1
    width = n * n + n + 2
    grid = np.full((height, width), bg, dtype=int)

    row_lines = [k * (m + 1) for k in range(m + 1)]
    col_lines = [1 + k * (n + 1) for k in range(n + 1)]
    for r in row_lines:
        grid[r, 1:] = divider
    for c in col_lines:
        grid[:, c] = divider

    for cell_r, color in enumerate(row_colors):
        marker_r = row_lines[cell_r] + 1 + int(_RNG.integers(m))
        grid[marker_r, 0] = int(color)

    cell_r, cell_c = template_cell
    rs, cs = _cell_slices(row_lines, col_lines, cell_r, cell_c)
    patch = grid[rs, cs].copy()
    patch[mask] = int(template_color)
    grid[rs, cs] = patch
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    bg, _divider, row_lines, col_lines, m, n, row_colors, mask = _parse_scene(grid)
    out = grid.copy()
    for cell_r in range(m):
        for cell_c in range(n):
            rs, cs = _cell_slices(row_lines, col_lines, cell_r, cell_c)
            out[rs, cs] = bg
    for cell_r, row_color in enumerate(row_colors):
        stamp = np.full((m, n), bg, dtype=int)
        stamp[mask] = int(row_color)
        for cell_c in range(n):
            if mask[cell_r, cell_c]:
                rs, cs = _cell_slices(row_lines, col_lines, cell_r, cell_c)
                out[rs, cs] = stamp
    return out


def _choose_template_cell(mask: np.ndarray) -> tuple[int, int]:
    m, n = mask.shape
    want_fg = bool(_RNG.integers(2))
    candidates = np.argwhere(mask if want_fg else ~mask)
    if candidates.size == 0:
        candidates = np.argwhere(~mask if want_fg else mask)
    choice = candidates[int(_RNG.integers(len(candidates)))]
    return int(choice[0]), int(choice[1])


def _gen_attempt() -> np.ndarray | None:
    m, n = _SIZE_OPTIONS[int(_RNG.integers(len(_SIZE_OPTIONS)))]

    needed = m + 3
    palette = list(_RNG.choice(_ALL_COLORS, size=needed, replace=False))
    bg = int(palette[0])
    divider = int(palette[1])
    template_color = int(palette[2])
    row_colors = [int(c) for c in palette[3:]]

    mask = _sample_template_mask(m, n)
    template_cell = _choose_template_cell(mask)
    grid = _render_input(bg, divider, row_colors, template_color, mask, template_cell)

    if max(grid.shape) > 30:
        return None
    if _get_bg(grid) != bg:
        return None
    expected = _expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-self-stamp input")
