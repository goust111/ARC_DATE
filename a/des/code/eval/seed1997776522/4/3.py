from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 9
INSTR = 8

SEED_COLORS = (1, 2, 3, 4, 5, 6, 7)
CELL_SIZES = (5, 7)
GRID_N_MIN = 3
GRID_N_MAX = 6
MAX_RETRIES = 400


def _cell_start(index: int, cell_size: int) -> int:
    return index * (cell_size + 1)


def _cell_slice(index: int, cell_size: int) -> slice:
    start = _cell_start(index, cell_size)
    return slice(start, start + cell_size)


def _detect_layout(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    best: tuple[int, int, int, int] | None = None

    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not rows or not cols or len(rows) != len(cols):
            continue

        if len(rows) == 1:
            cell_size = int(rows[0])
        else:
            row_diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
            col_diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
            if len(set(row_diffs)) != 1 or len(set(col_diffs)) != 1:
                continue
            if row_diffs[0] != col_diffs[0]:
                continue
            cell_size = int(row_diffs[0] - 1)

        if cell_size <= 0:
            continue
        if rows[0] != cell_size or cols[0] != cell_size:
            continue
        if h - rows[-1] - 1 != cell_size or w - cols[-1] - 1 != cell_size:
            continue
        expected_step = cell_size + 1
        if any(rows[i + 1] - rows[i] != expected_step for i in range(len(rows) - 1)):
            continue
        if any(cols[i + 1] - cols[i] != expected_step for i in range(len(cols) - 1)):
            continue

        n = len(rows) + 1
        if h != n * cell_size + (n - 1) or w != n * cell_size + (n - 1):
            continue

        score = len(rows) + len(cols)
        candidate = (score, color, cell_size, n)
        if best is None or candidate > best:
            best = candidate

    assert best is not None, "failed to detect separator layout"
    return int(best[1]), int(best[2]), int(best[3])


def _background_color(grid: np.ndarray, sep_color: int) -> int:
    values, counts = np.unique(grid[grid != sep_color], return_counts=True)
    assert values.size > 0
    return int(values[int(np.argmax(counts))])


def _instruction_color(
    grid: np.ndarray, bg: int, sep_color: int, cell_size: int, n: int
) -> int:
    colors: set[int] = set()
    for idx in range(n):
        top = grid[_cell_slice(0, cell_size), _cell_slice(idx, cell_size)]
        left = grid[_cell_slice(idx, cell_size), _cell_slice(0, cell_size)]
        for cell in (top, left):
            vals = set(map(int, np.unique(cell)))
            vals.discard(bg)
            vals.discard(sep_color)
            colors.update(vals)
    assert len(colors) == 1, "instruction cells must use one dedicated color"
    return int(next(iter(colors)))


def _parse_instruction(cell: np.ndarray, instr_color: int) -> str:
    coords = np.argwhere(cell == instr_color)
    assert 2 <= len(coords) <= 3, "instruction line must be 2-3 pixels long"

    center_r = cell.shape[0] // 2
    center_c = cell.shape[1] // 2
    points = {(int(r), int(c)) for r, c in coords}
    assert (center_r, center_c) in points, "instruction must start at the cell center"

    rows = sorted({int(r) for r, _ in points})
    cols = sorted({int(c) for _, c in points})

    if len(rows) == 1:
        assert rows[0] == center_r
        assert cols == list(range(cols[0], cols[-1] + 1))
        if cols[0] == center_c and cols[-1] > center_c:
            return "right"
        if cols[-1] == center_c and cols[0] < center_c:
            return "left"
    if len(cols) == 1:
        assert cols[0] == center_c
        assert rows == list(range(rows[0], rows[-1] + 1))
        if rows[0] == center_r and rows[-1] > center_r:
            return "down"
        if rows[-1] == center_r and rows[0] < center_r:
            return "up"

    raise AssertionError("invalid instruction geometry")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed diagonally inside its cell according to top-row and left-column instructions.

    The grid is partitioned by 1-pixel separator lines into equal square cells.
    Cells on the top row encode horizontal growth (left/right); cells on the
    left column encode vertical growth (up/down). Every non-instruction cell
    combines those two directions into a diagonal and extends each seed pixel
    along that diagonal until the cell boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_color, cell_size, n = _detect_layout(g)
    bg = _background_color(g, sep_color)
    instr_color = _instruction_color(g, bg, sep_color, cell_size, n)

    row_dirs: list[int] = []
    col_dirs: list[int] = []

    for row_idx in range(1, n):
        cell = g[_cell_slice(row_idx, cell_size), _cell_slice(0, cell_size)]
        direction = _parse_instruction(cell, instr_color)
        assert direction in {"up", "down"}
        row_dirs.append(-1 if direction == "up" else 1)

    for col_idx in range(1, n):
        cell = g[_cell_slice(0, cell_size), _cell_slice(col_idx, cell_size)]
        direction = _parse_instruction(cell, instr_color)
        assert direction in {"left", "right"}
        col_dirs.append(-1 if direction == "left" else 1)

    out = g.copy()
    for row_idx in range(1, n):
        for col_idx in range(1, n):
            rs = _cell_slice(row_idx, cell_size)
            cs = _cell_slice(col_idx, cell_size)
            cell = g[rs, cs]
            seed_mask = (cell != bg) & (cell != instr_color) & (cell != sep_color)
            seeds = np.argwhere(seed_mask)
            if seeds.size == 0:
                continue

            dr = row_dirs[row_idx - 1]
            dc = col_dirs[col_idx - 1]
            for sr, sc in seeds:
                color = int(cell[int(sr), int(sc)])
                r = int(sr)
                c = int(sc)
                while 0 <= r < cell_size and 0 <= c < cell_size:
                    out[rs.start + r, cs.start + c] = color
                    r += dr
                    c += dc

    return out


def _draw_instruction(
    grid: np.ndarray, row_idx: int, col_idx: int, cell_size: int, direction: str
) -> None:
    rs = _cell_slice(row_idx, cell_size)
    cs = _cell_slice(col_idx, cell_size)
    center = cell_size // 2
    length = random.randint(2, 3)

    dr, dc = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]

    for step in range(length):
        rr = rs.start + center + dr * step
        cc = cs.start + center + dc * step
        grid[rr, cc] = INSTR


def _balanced_signs(count: int) -> list[int]:
    for _ in range(MAX_RETRIES):
        signs = [random.choice((-1, 1)) for _ in range(count)]
        if count <= 1 or (-1 in signs and 1 in signs):
            return signs
    raise RuntimeError("failed to sample both instruction branches")


def _choose_active_cells(data_n: int, active_count: int) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(1, data_n + 1) for c in range(1, data_n + 1)]
    min_rows = 2 if data_n >= 2 else 1
    min_cols = 2 if data_n >= 2 else 1

    for _ in range(MAX_RETRIES):
        chosen = random.sample(all_cells, active_count)
        used_rows = {r for r, _ in chosen}
        used_cols = {c for _, c in chosen}
        if len(used_rows) >= min_rows and len(used_cols) >= min_cols:
            return chosen
    return random.sample(all_cells, active_count)


def _seed_candidates(
    cell_size: int, dr: int, dc: int
) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(cell_size):
        for c in range(cell_size):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < cell_size and 0 <= nc < cell_size):
                continue
            key = r - c if dr == dc else r + c
            groups.setdefault(key, []).append((r, c))
    assert groups, "every direction must admit at least one growing seed"
    return groups


def _build_input() -> np.ndarray:
    n = random.randint(GRID_N_MIN, GRID_N_MAX)
    cell_size = random.choice(CELL_SIZES)
    size = n * cell_size + (n - 1)
    grid = np.full((size, size), BG, dtype=int)

    for k in range(1, n):
        pos = k * cell_size + (k - 1)
        grid[pos, :] = SEP
        grid[:, pos] = SEP

    row_dirs = _balanced_signs(n - 1)
    col_dirs = _balanced_signs(n - 1)

    _draw_instruction(grid, 0, 0, cell_size, random.choice(("up", "down", "left", "right")))
    for row_idx, dr in enumerate(row_dirs, start=1):
        _draw_instruction(grid, row_idx, 0, cell_size, "up" if dr == -1 else "down")
    for col_idx, dc in enumerate(col_dirs, start=1):
        _draw_instruction(grid, 0, col_idx, cell_size, "left" if dc == -1 else "right")

    data_n = n - 1
    data_cells = data_n * data_n
    active_max = min(data_cells, max(2, 2 * data_n))
    active_count = random.randint(2, active_max)
    active_cells = _choose_active_cells(data_n, active_count)

    for row_idx, col_idx in active_cells:
        dr = row_dirs[row_idx - 1]
        dc = col_dirs[col_idx - 1]
        groups = _seed_candidates(cell_size, dr, dc)
        seed_count = random.randint(1, min(3, len(groups)))
        chosen_keys = random.sample(list(groups.keys()), seed_count)
        chosen_colors = random.sample(SEED_COLORS, seed_count)

        rs = _cell_slice(row_idx, cell_size)
        cs = _cell_slice(col_idx, cell_size)
        for key, color in zip(chosen_keys, chosen_colors, strict=False):
            r, c = random.choice(groups[key])
            grid[rs.start + r, cs.start + c] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_input()
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
