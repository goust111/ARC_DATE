from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
CELL = 4
TABLE_COLS = 3
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

CANVAS_SIZE_RANGE = (6, 10)
INSTRUCTION_COUNT_RANGE = (2, 5)
FRAME_GAP_RANGE = (1, 2)
TOP_PAD_RANGE = (0, 2)
BOTTOM_PAD_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _full_sep_lines(sub: np.ndarray) -> tuple[list[int], list[int]]:
    row_lines = [int(r) for r in range(sub.shape[0]) if np.all(sub[r, :] == SEP)]
    col_lines = [int(c) for c in range(sub.shape[1]) if np.all(sub[:, c] == SEP)]
    return row_lines, col_lines


def _extract_non_bg(cell: np.ndarray) -> np.ndarray:
    return cell[(cell != BG) & (cell != SEP)]


def _cell_color(cell: np.ndarray) -> int:
    vals = _extract_non_bg(cell)
    assert vals.size > 0, "each instruction cell must contain a colored mark"
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[counts.argmax()])


def _find_table(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    candidates = []
    for comp in _components(grid == SEP):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        row_lines, col_lines = _full_sep_lines(sub)
        if len(col_lines) != TABLE_COLS + 1 or len(row_lines) < 3:
            continue
        if not all(col_lines[i + 1] - col_lines[i] > 1 for i in range(len(col_lines) - 1)):
            continue
        if not all(row_lines[i + 1] - row_lines[i] > 1 for i in range(len(row_lines) - 1)):
            continue
        candidates.append((len(row_lines), sub.shape[0] * sub.shape[1], box, row_lines, col_lines))

    assert candidates, "could not locate the instruction table"
    _n_rows, _area, box, row_lines, col_lines = max(candidates, key=lambda item: (item[0], item[1]))
    return box, row_lines, col_lines


def _find_framed_canvas(
    grid: np.ndarray, table_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    best = None
    for comp in _components(grid == SEP):
        box = _bbox(comp)
        if box == table_box:
            continue
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h != w or h < 4:
            continue
        if not (
            np.all(sub[0, :] == SEP)
            and np.all(sub[-1, :] == SEP)
            and np.all(sub[:, 0] == SEP)
            and np.all(sub[:, -1] == SEP)
        ):
            continue
        if not np.all(sub[1:-1, 1:-1] == BG):
            continue
        inner_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        size = h - 2
        if best is None or size > best[0]:
            best = (size, inner_box)
    return None if best is None else best[1]


def _find_blank_canvas(
    grid: np.ndarray, table_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    _tr0, _tc0, _tr1, tc1 = table_box
    h, w = grid.shape
    for size in range(min(h, w), 0, -1):
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - size + 1):
            for c0 in range(tc1 + 1, w - size + 1):
                if np.all(grid[r0 : r0 + size, c0 : c0 + size] == BG):
                    candidates.append((r0, c0))
        if candidates:
            r0, c0 = max(candidates, key=lambda item: (item[1], -item[0]))
            return r0, c0, r0 + size - 1, c0 + size - 1
    raise AssertionError("could not locate the canvas")


def _find_canvas(grid: np.ndarray, table_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    framed = _find_framed_canvas(grid, table_box)
    if framed is not None:
        return framed
    return _find_blank_canvas(grid, table_box)


def _read_instructions(
    grid: np.ndarray,
    table_box: tuple[int, int, int, int],
    row_lines: list[int],
    col_lines: list[int],
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = table_box
    table = grid[r0 : r1 + 1, c0 : c1 + 1]
    instructions: list[tuple[int, int, int]] = []

    for i in range(len(row_lines) - 1):
        row_cells = []
        for j in range(len(col_lines) - 1):
            cell = table[row_lines[i] + 1 : row_lines[i + 1], col_lines[j] + 1 : col_lines[j + 1]]
            row_cells.append(cell)

        color = _cell_color(row_cells[0])
        width_vals = _extract_non_bg(row_cells[1])
        height_vals = _extract_non_bg(row_cells[2])

        assert width_vals.size > 0 and height_vals.size > 0
        assert len(np.unique(width_vals)) == 1 and len(np.unique(height_vals)) == 1
        assert int(np.unique(width_vals)[0]) == color and int(np.unique(height_vals)[0]) == color

        width = int(width_vals.size)
        height = int(height_vals.size)
        assert 1 <= width <= CELL and 1 <= height <= CELL
        instructions.append((color, width, height))

    assert instructions, "expected at least one drawing instruction"
    return instructions


def _paint_canvas(instructions: list[tuple[int, int, int]], size: int) -> np.ndarray:
    out = np.full((size, size), BG, dtype=int)
    row = 0
    col = 0
    for color, width, height in instructions:
        if col + width > size:
            row += 1
            col = 0
        assert row + height <= size, "instruction sequence must fit the canvas"
        out[row : row + height, col : col + width] = color
        col += width
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the instruction table and execute the rectangle painting program."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    table_box, row_lines, col_lines = _find_table(g)
    canvas_box = _find_canvas(g, table_box)
    cr0, cc0, cr1, cc1 = canvas_box
    canvas_size = int(cr1 - cr0 + 1)
    assert canvas_size == int(cc1 - cc0 + 1), "canvas must be square"

    instructions = _read_instructions(g, table_box, row_lines, col_lines)
    return _paint_canvas(instructions, canvas_size)


def _simulate_layout(
    instructions: list[tuple[int, int, int]], size: int
) -> tuple[np.ndarray, int, int] | None:
    canvas = np.full((size, size), BG, dtype=int)
    row = 0
    col = 0
    wraps = 0
    visible_overlap = 0

    for color, width, height in instructions:
        if col + width > size:
            row += 1
            col = 0
            wraps += 1
        if row + height > size:
            return None
        region = canvas[row : row + height, col : col + width]
        visible_overlap += int(np.count_nonzero((region != BG) & (region != color)))
        canvas[row : row + height, col : col + width] = color
        col += width

    return canvas, wraps, visible_overlap


def _sample_instructions(size: int, count: int, require_wrap: bool) -> tuple[list[tuple[int, int, int]], np.ndarray] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        colors = [random.choice(PALETTE) for _ in range(count)]
        if count > 1 and len(set(colors)) == 1:
            alt = random.choice([c for c in PALETTE if c != colors[0]])
            colors[random.randrange(count)] = alt

        instructions = [
            (int(colors[i]), random.randint(1, CELL), random.randint(1, CELL))
            for i in range(count)
        ]
        simulated = _simulate_layout(instructions, size)
        if simulated is None:
            continue
        canvas, wraps, visible_overlap = simulated
        if require_wrap:
            if wraps == 0 or visible_overlap == 0:
                continue
        elif wraps != 0:
            continue
        return instructions, canvas
    return None


def _draw_color_block(cell: np.ndarray, color: int) -> None:
    h = random.randint(2, CELL)
    w = random.randint(2, CELL)
    r0 = random.randint(0, CELL - h)
    c0 = random.randint(0, CELL - w)
    cell[r0 : r0 + h, c0 : c0 + w] = color


def _draw_hbar(cell: np.ndarray, color: int, width: int) -> None:
    row = random.randint(0, CELL - 1)
    col = random.randint(0, CELL - width)
    cell[row, col : col + width] = color


def _draw_vbar(cell: np.ndarray, color: int, height: int) -> None:
    row = random.randint(0, CELL - height)
    col = random.randint(0, CELL - 1)
    cell[row : row + height, col] = color


def _render_table(instructions: list[tuple[int, int, int]]) -> np.ndarray:
    n_rows = len(instructions)
    table_h = n_rows * CELL + (n_rows + 1)
    table_w = TABLE_COLS * CELL + (TABLE_COLS + 1)
    table = np.full((table_h, table_w), BG, dtype=int)

    row_lines = [i * (CELL + 1) for i in range(n_rows + 1)]
    col_lines = [i * (CELL + 1) for i in range(TABLE_COLS + 1)]
    for r in row_lines:
        table[r, :] = SEP
    for c in col_lines:
        table[:, c] = SEP

    for i, (color, width, height) in enumerate(instructions):
        cell0 = table[row_lines[i] + 1 : row_lines[i + 1], col_lines[0] + 1 : col_lines[1]]
        cell1 = table[row_lines[i] + 1 : row_lines[i + 1], col_lines[1] + 1 : col_lines[2]]
        cell2 = table[row_lines[i] + 1 : row_lines[i + 1], col_lines[2] + 1 : col_lines[3]]
        _draw_color_block(cell0, color)
        _draw_hbar(cell1, color, width)
        _draw_vbar(cell2, color, height)

    return table


def _render_canvas_frame(size: int) -> np.ndarray:
    frame = np.full((size + 2, size + 2), BG, dtype=int)
    frame[0, :] = SEP
    frame[-1, :] = SEP
    frame[:, 0] = SEP
    frame[:, -1] = SEP
    return frame


def _build_input_grid(table: np.ndarray, canvas_size: int) -> np.ndarray:
    frame = _render_canvas_frame(canvas_size)
    table_h, table_w = table.shape
    frame_h, frame_w = frame.shape
    gap = random.randint(*FRAME_GAP_RANGE)
    table_top = random.randint(*TOP_PAD_RANGE)
    frame_top = random.randint(*TOP_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)

    grid_h = max(table_top + table_h, frame_top + frame_h) + bottom_pad
    grid_w = table_w + gap + frame_w
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[table_top : table_top + table_h, 0:table_w] = table
    start_c = table_w + gap
    grid[frame_top : frame_top + frame_h, start_c : start_c + frame_w] = frame
    return grid


def _generate_attempt() -> np.ndarray | None:
    require_wrap = random.random() < 0.5
    canvas_size = random.randint(*CANVAS_SIZE_RANGE)
    if require_wrap:
        count_lo = max(3, INSTRUCTION_COUNT_RANGE[0])
        count = random.randint(count_lo, INSTRUCTION_COUNT_RANGE[1])
    else:
        count = random.randint(*INSTRUCTION_COUNT_RANGE)

    sampled = _sample_instructions(canvas_size, count, require_wrap)
    if sampled is None:
        return None
    instructions, expected = sampled
    table = _render_table(instructions)
    grid = _build_input_grid(table, canvas_size)

    try:
        observed = main(grid)
    except Exception:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
