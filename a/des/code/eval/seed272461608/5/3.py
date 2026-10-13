from __future__ import annotations

import random

import numpy as np


BG = 0
ARROW = 9
COLOR_POOL = tuple(range(1, 9))

GRID_H_RANGE = (8, 15)
GRID_W_RANGE = (8, 15)
MAX_ROW_ARROWS = 2
MAX_COL_ARROWS = 2
GEN_ATTEMPTS = 600


def _arrow_cells(shape: tuple[int, int], axis: str, index: int, direction: str) -> tuple[tuple[int, int], ...]:
    h, w = shape
    if axis == "row" and direction == "right":
        return ((index, 0), (index - 1, 1), (index + 1, 1))
    if axis == "row" and direction == "left":
        return ((index, w - 1), (index - 1, w - 2), (index + 1, w - 2))
    if axis == "col" and direction == "down":
        return ((0, index), (1, index - 1), (1, index + 1))
    assert axis == "col" and direction == "up"
    return ((h - 1, index), (h - 2, index - 1), (h - 2, index + 1))


def _detect_arrows(grid: np.ndarray) -> tuple[list[tuple[int, str]], list[tuple[int, str]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_arrows: list[tuple[int, str]] = []
    col_arrows: list[tuple[int, str]] = []
    usage: dict[tuple[int, int], int] = {}

    if w >= 2:
        for r in range(1, h - 1):
            for direction in ("right", "left"):
                cells = _arrow_cells((h, w), "row", r, direction)
                if all(int(g[rr, cc]) == ARROW for rr, cc in cells):
                    row_arrows.append((int(r), direction))
                    for cell in cells:
                        usage[cell] = usage.get(cell, 0) + 1

    if h >= 2:
        for c in range(1, w - 1):
            for direction in ("down", "up"):
                cells = _arrow_cells((h, w), "col", c, direction)
                if all(int(g[rr, cc]) == ARROW for rr, cc in cells):
                    col_arrows.append((int(c), direction))
                    for cell in cells:
                        usage[cell] = usage.get(cell, 0) + 1

    maroon_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == ARROW)}
    assert maroon_cells == set(usage), "every maroon pixel must belong to exactly one arrow"
    assert all(count == 1 for count in usage.values()), "arrow pixels may not overlap"
    assert len({r for r, _direction in row_arrows}) == len(row_arrows), "at most one row arrow per row"
    assert len({c for c, _direction in col_arrows}) == len(col_arrows), "at most one column arrow per column"

    return row_arrows, col_arrows, maroon_cells


def _collect_row_values(row: np.ndarray, direction: str) -> list[int]:
    values = [int(v) for v in row if int(v) not in (BG, ARROW)]
    return values if direction == "right" else values[::-1]


def _collect_col_values(col: np.ndarray, direction: str) -> list[int]:
    values = [int(v) for v in col if int(v) not in (BG, ARROW)]
    return values if direction == "down" else values[::-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push colored singleton pixels along arrow-marked rows and columns.

    Maroon 3-pixel edge arrows encode the push direction. After removing the arrows,
    each affected row is processed first, then each affected column is processed.
    Pixels are read from the side opposite the arrow direction and then stacked
    tightly from the destination boundary inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_arrows, col_arrows, arrow_cells = _detect_arrows(g)
    out = g.copy()
    for r, c in arrow_cells:
        out[r, c] = BG

    h, w = out.shape

    for r, direction in row_arrows:
        values = _collect_row_values(out[r], direction)
        out[r, :] = BG
        if direction == "right":
            for i, value in enumerate(values):
                out[r, w - 1 - i] = int(value)
        else:
            for i, value in enumerate(values):
                out[r, i] = int(value)

    for c, direction in col_arrows:
        values = _collect_col_values(out[:, c], direction)
        out[:, c] = BG
        if direction == "down":
            for i, value in enumerate(values):
                out[h - 1 - i, c] = int(value)
        else:
            for i, value in enumerate(values):
                out[i, c] = int(value)

    return out


def _sample_arrow_specs(h: int, w: int) -> list[tuple[str, int, str]] | None:
    mode = random.choices(
        ["horizontal", "vertical", "mixed"],
        weights=[2, 2, 3],
        k=1,
    )[0]

    row_count = 0
    col_count = 0
    if mode in ("horizontal", "mixed"):
        row_count = random.randint(1, min(MAX_ROW_ARROWS, max(1, h - 4)))
    if mode in ("vertical", "mixed"):
        col_count = random.randint(1, min(MAX_COL_ARROWS, max(1, w - 4)))

    occupied: set[tuple[int, int]] = set()
    specs: list[tuple[str, int, str]] = []

    row_candidates = [(r, direction) for r in range(2, h - 2) for direction in ("left", "right")]
    col_candidates = [(c, direction) for c in range(2, w - 2) for direction in ("up", "down")]
    random.shuffle(row_candidates)
    random.shuffle(col_candidates)

    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for r, direction in row_candidates:
        if len([spec for spec in specs if spec[0] == "row"]) >= row_count:
            break
        if r in used_rows:
            continue
        cells = set(_arrow_cells((h, w), "row", r, direction))
        if occupied & cells:
            continue
        specs.append(("row", int(r), direction))
        occupied |= cells
        used_rows.add(int(r))

    for c, direction in col_candidates:
        if len([spec for spec in specs if spec[0] == "col"]) >= col_count:
            break
        if c in used_cols:
            continue
        cells = set(_arrow_cells((h, w), "col", c, direction))
        if occupied & cells:
            continue
        specs.append(("col", int(c), direction))
        occupied |= cells
        used_cols.add(int(c))

    if len([spec for spec in specs if spec[0] == "row"]) != row_count:
        return None
    if len([spec for spec in specs if spec[0] == "col"]) != col_count:
        return None
    if not specs:
        return None
    return specs


def _direction_far_positions(length: int, direction: str, candidates: list[int]) -> list[int]:
    if direction == "right":
        far = [pos for pos in candidates if pos < length - 2]
    elif direction == "left":
        far = [pos for pos in candidates if pos > 1]
    elif direction == "down":
        far = [pos for pos in candidates if pos < length - 2]
    else:
        far = [pos for pos in candidates if pos > 1]
    return far if far else candidates


def _assign_colors(grid: np.ndarray, cells: list[tuple[int, int]], active_colors: list[int]) -> None:
    if not cells:
        return
    if len(cells) <= len(active_colors):
        colors = random.sample(active_colors, len(cells))
    else:
        colors = [random.choice(active_colors) for _ in cells]
    random.shuffle(colors)
    for (r, c), color in zip(cells, colors):
        grid[r, c] = int(color)


def _row_pixel_positions(
    shape: tuple[int, int],
    row: int,
    direction: str,
    need: int,
    occupied: set[tuple[int, int]],
    favored_cols: set[int],
) -> list[tuple[int, int]] | None:
    _h, w = shape
    free_cols = [c for c in range(w) if (row, c) not in occupied]
    free_cols = _direction_far_positions(w, direction, free_cols)
    if len(free_cols) < need:
        return None

    chosen: list[int] = []
    exclusive = [c for c in free_cols if c not in favored_cols]
    if exclusive:
        c = random.choice(exclusive)
        chosen.append(int(c))
        free_cols.remove(c)

    favored = [c for c in free_cols if c in favored_cols]
    if favored and len(chosen) < need:
        c = random.choice(favored)
        chosen.append(int(c))
        free_cols.remove(c)

    random.shuffle(free_cols)
    chosen.extend(int(c) for c in free_cols[: need - len(chosen)])
    return [(int(row), c) for c in chosen]


def _col_pixel_positions(
    shape: tuple[int, int],
    col: int,
    direction: str,
    need: int,
    occupied: set[tuple[int, int]],
    favored_rows: set[int],
) -> list[tuple[int, int]] | None:
    h, _w = shape
    free_rows = [r for r in range(h) if (r, col) not in occupied]
    free_rows = _direction_far_positions(h, direction, free_rows)
    if len(free_rows) < need:
        return None

    chosen: list[int] = []
    exclusive = [r for r in free_rows if r not in favored_rows]
    if exclusive:
        r = random.choice(exclusive)
        chosen.append(int(r))
        free_rows.remove(r)

    favored = [r for r in free_rows if r in favored_rows]
    if favored and len(chosen) < need:
        r = random.choice(favored)
        chosen.append(int(r))
        free_rows.remove(r)

    random.shuffle(free_rows)
    chosen.extend(int(r) for r in free_rows[: need - len(chosen)])
    return [(r, int(col)) for r in chosen]


def _populate_pixels(grid: np.ndarray, specs: list[tuple[str, int, str]]) -> bool:
    h, w = grid.shape
    occupied = {cell for spec in specs for cell in _arrow_cells((h, w), spec[0], spec[1], spec[2])}
    row_specs = [(index, direction) for axis, index, direction in specs if axis == "row"]
    col_specs = [(index, direction) for axis, index, direction in specs if axis == "col"]
    row_set = {row for row, _direction in row_specs}
    col_set = {col for col, _direction in col_specs}

    active_colors = random.sample(list(COLOR_POOL), random.randint(3, min(6, len(COLOR_POOL))))

    for row, direction in row_specs:
        need = random.choice([1, 2, 2, 3])
        cells = _row_pixel_positions((h, w), row, direction, need, occupied, col_set)
        if cells is None:
            return False
        _assign_colors(grid, cells, active_colors)
        occupied.update(cells)

    for col, direction in col_specs:
        current = sum(1 for r in range(h) if int(grid[r, col]) not in (BG, ARROW))
        target = random.choice([1, 2, 2, 3])
        need = max(0, target - current)
        if need == 0:
            continue
        cells = _col_pixel_positions((h, w), col, direction, need, occupied, row_set)
        if cells is None:
            return False
        _assign_colors(grid, cells, active_colors)
        occupied.update(cells)

    extra_budget = random.randint(0, max(2, (h * w) // 20))
    free_cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
    random.shuffle(free_cells)
    extras = free_cells[:extra_budget]
    _assign_colors(grid, extras, active_colors)

    return True


def _render_input(h: int, w: int, specs: list[tuple[str, int, str]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for axis, index, direction in specs:
        for r, c in _arrow_cells((h, w), axis, index, direction):
            grid[r, c] = ARROW
    return grid


def _is_nontrivial(inp: np.ndarray, out: np.ndarray) -> bool:
    clean = inp.copy()
    clean[clean == ARROW] = BG
    if np.array_equal(clean, out):
        return False
    colors = set(map(int, np.unique(inp))) - {BG, ARROW}
    return len(colors) >= 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    specs = _sample_arrow_specs(h, w)
    if specs is None:
        return None

    grid = _render_input(h, w, specs)
    if not _populate_pixels(grid, specs):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not _is_nontrivial(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
