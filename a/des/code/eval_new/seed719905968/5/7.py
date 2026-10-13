import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400


def _extract_palette_color(region: np.ndarray) -> int:
    colors = {
        int(color)
        for color in np.unique(region)
        if int(color) not in (BLACK, GREY, MAROON)
    }
    assert len(colors) == 1
    return colors.pop()


def _board_structure(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    grey_pixels = np.argwhere(grid == GREY)
    assert grey_pixels.size > 0

    r0, c0 = grey_pixels.min(axis=0)
    r1, c1 = grey_pixels.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    h_lines = [r for r in range(r0, r1 + 1) if bool((grid[r, c0 : c1 + 1] == GREY).all())]
    v_lines = [c for c in range(c0, c1 + 1) if bool((grid[r0 : r1 + 1, c] == GREY).all())]

    assert len(h_lines) >= 2 and len(v_lines) >= 2
    assert len(h_lines) == len(v_lines)

    h_steps = [b - a for a, b in zip(h_lines, h_lines[1:])]
    v_steps = [b - a for a, b in zip(v_lines, v_lines[1:])]
    assert len(set(h_steps)) == 1
    assert len(set(v_steps)) == 1
    assert h_steps[0] == v_steps[0]

    cell_size = h_steps[0] - 1
    assert cell_size in (2, 3)
    return h_lines, v_lines, cell_size


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill marked checkerboard cells from row/column palettes using row+column parity.

    Input structure:
    - The background is black. A large central square board is divided by grey grid lines into equal square cells of size 2x2 or 3x3.
    - Some board cells contain a maroon marker pixel inside the otherwise black cell interior.
    - A horizontal color palette sits above the board, with one aligned color block per board column.
    - A vertical color palette sits to the left of the board, with one aligned color block per board row.

    Rule:
    - For each marked board cell, read the aligned column-palette color and row-palette color for that cell's column and row.
    - If the cell's row index plus column index is even, fill the entire cell with the column color.
    - If the sum is odd, fill the entire cell with the row color.
    - Filling replaces the full cell interior, including the maroon marker; unmarked cells stay black and the palettes and grey grid remain unchanged.

    Key insight:
    - The palettes provide two independent color lookups, and the actual choice alternates by the parity of the marked cell's row/column coordinates rather than by a single palette alone.
    """
    g = np.asarray(grid, dtype=int)
    h_lines, v_lines, _cell_size = _board_structure(g)

    board_top = h_lines[0]
    board_left = v_lines[0]
    n = len(h_lines) - 1

    row_colors = []
    for i in range(n):
        r0 = h_lines[i] + 1
        r1 = h_lines[i + 1]
        row_colors.append(_extract_palette_color(g[r0:r1, :board_left]))

    col_colors = []
    for j in range(n):
        c0 = v_lines[j] + 1
        c1 = v_lines[j + 1]
        col_colors.append(_extract_palette_color(g[:board_top, c0:c1]))

    assert len(set(row_colors)) == n
    assert len(set(col_colors)) == n

    out = g.copy()
    for i in range(n):
        rs = slice(h_lines[i] + 1, h_lines[i + 1])
        for j in range(n):
            cs = slice(v_lines[j] + 1, v_lines[j + 1])
            cell = g[rs, cs]
            if not bool((cell == MAROON).any()):
                continue
            fill = col_colors[j] if (i + j) % 2 == 0 else row_colors[i]
            out[rs, cs] = fill

    return out


def _choose_palettes(n: int) -> tuple[list[int], list[int]] | None:
    row_colors = random.sample(PALETTE_COLORS, n)
    col_colors = random.sample(PALETTE_COLORS, n)

    even_mismatch = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r + c) % 2 == 0 and row_colors[r] != col_colors[c]
    ]
    odd_mismatch = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r + c) % 2 == 1 and row_colors[r] != col_colors[c]
    ]
    if not even_mismatch or not odd_mismatch:
        return None
    return row_colors, col_colors


def _choose_marked_cells(n: int, row_colors: list[int], col_colors: list[int]) -> list[tuple[int, int]] | None:
    even_cells = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r + c) % 2 == 0 and row_colors[r] != col_colors[c]
    ]
    odd_cells = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r + c) % 2 == 1 and row_colors[r] != col_colors[c]
    ]
    if not even_cells or not odd_cells:
        return None

    required = {random.choice(even_cells), random.choice(odd_cells)}
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    remaining = [cell for cell in all_cells if cell not in required]

    max_count = n * n - 1 if n * n > 2 else 2
    target_count = random.randint(2, max_count)
    extra_count = target_count - len(required)
    if extra_count > len(remaining):
        return None

    marked = list(required)
    if extra_count > 0:
        marked.extend(random.sample(remaining, extra_count))
    return marked


def _render_input(
    n: int,
    cell_size: int,
    row_colors: list[int],
    col_colors: list[int],
    marked_cells: list[tuple[int, int]],
    top_gap: int,
    left_gap: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    board_size = n * cell_size + (n + 1)
    board_top = top_margin + cell_size + top_gap
    board_left = left_margin + cell_size + left_gap

    height = board_top + board_size + bottom_margin
    width = board_left + board_size + right_margin
    grid = np.full((height, width), BLACK, dtype=int)

    for i in range(n + 1):
        r = board_top + i * (cell_size + 1)
        grid[r, board_left : board_left + board_size] = GREY
    for j in range(n + 1):
        c = board_left + j * (cell_size + 1)
        grid[board_top : board_top + board_size, c] = GREY

    top_palette_r0 = board_top - top_gap - cell_size
    left_palette_c0 = board_left - left_gap - cell_size
    for j, color in enumerate(col_colors):
        c0 = board_left + 1 + j * (cell_size + 1)
        grid[top_palette_r0 : top_palette_r0 + cell_size, c0 : c0 + cell_size] = color
    for i, color in enumerate(row_colors):
        r0 = board_top + 1 + i * (cell_size + 1)
        grid[r0 : r0 + cell_size, left_palette_c0 : left_palette_c0 + cell_size] = color

    marker_offset = cell_size // 2
    for r, c in marked_cells:
        rr = board_top + 1 + r * (cell_size + 1) + marker_offset
        cc = board_left + 1 + c * (cell_size + 1) + marker_offset
        grid[rr, cc] = MAROON

    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice([2, 3])
    top_gap = random.randint(1, 2)
    left_gap = random.randint(1, 2)
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)

    usable_h = 30 - top_margin - bottom_margin - cell_size - top_gap
    usable_w = 30 - left_margin - right_margin - cell_size - left_gap
    max_board = min(usable_h, usable_w)
    max_n = min(7, (max_board - 1) // (cell_size + 1))
    if max_n < 3:
        return None

    n = random.randint(3, max_n)
    palettes = _choose_palettes(n)
    if palettes is None:
        return None
    row_colors, col_colors = palettes

    marked_cells = _choose_marked_cells(n, row_colors, col_colors)
    if marked_cells is None:
        return None

    grid = _render_input(
        n=n,
        cell_size=cell_size,
        row_colors=row_colors,
        col_colors=col_colors,
        marked_cells=marked_cells,
        top_gap=top_gap,
        left_gap=left_gap,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    if max(grid.shape) > 30:
        return None

    try:
        out = _expected_output(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    changed_cells = 0
    for r, c in marked_cells:
        board_top = top_margin + cell_size + top_gap
        board_left = left_margin + cell_size + left_gap
        rs = slice(board_top + 1 + r * (cell_size + 1), board_top + 1 + r * (cell_size + 1) + cell_size)
        cs = slice(board_left + 1 + c * (cell_size + 1), board_left + 1 + c * (cell_size + 1) + cell_size)
        if not bool(np.array_equal(out[rs, cs], grid[rs, cs])):
            changed_cells += 1
    if changed_cells != len(marked_cells):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
