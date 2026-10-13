from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
COARSE_CELL = 4
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_FULL_ROWS = 3
MAX_FULL_ROWS = 6
MIN_FULL_COLS = 3
MAX_FULL_COLS = 6
MIN_OBJECTS = 3
MAX_OBJECTS = 8
MAX_GENERATION_ATTEMPTS = 300

_INNER_2X2 = ((0, 0), (0, 1), (1, 0), (1, 1))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    remaining = set(cells)
    stack = [next(iter(remaining))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nxt in _neighbors4(r, c):
            if nxt in remaining and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(remaining)


def _safe_shapes() -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes = []
    for mask in range(1, 1 << len(_INNER_2X2)):
        cells = tuple(_INNER_2X2[i] for i in range(len(_INNER_2X2)) if (mask >> i) & 1)
        if _is_connected(cells):
            shapes.append(cells)
    return tuple(shapes)


SAFE_SHAPES = _safe_shapes()


def _find_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            stack = [(r, c)]
            coords = []
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append({"coords": arr, "bbox": (r0, c0, r1, c1)})

    return components


def _coarse_parity(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    coarse_r = int(center_r // COARSE_CELL)
    coarse_c = int(center_c // COARSE_CELL)
    return (coarse_r + coarse_c) % 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split foreground objects by the parity of the 4x4 coarse cell containing each bbox center.

    Even-parity objects are drawn on the left panel, odd-parity objects on the
    right panel, and the two panels are separated by one grey column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = np.full((h, 2 * w + 1), BACKGROUND, dtype=int)
    out[:, w] = SEPARATOR

    for comp in _find_components(g, bg=BACKGROUND):
        coords = comp["coords"]
        panel_offset = 0 if _coarse_parity(comp["bbox"]) == 0 else w + 1
        out[coords[:, 0], coords[:, 1] + panel_offset] = g[coords[:, 0], coords[:, 1]]

    return out


def _choose_layout() -> tuple[int, int, int, int]:
    full_rows = random.randint(MIN_FULL_ROWS, MAX_FULL_ROWS)
    full_cols = random.randint(MIN_FULL_COLS, MAX_FULL_COLS)
    extra_rows = random.randint(0, COARSE_CELL - 1)
    extra_cols = random.randint(0, COARSE_CELL - 1)
    height = full_rows * COARSE_CELL + extra_rows
    width = full_cols * COARSE_CELL + extra_cols
    return height, width, full_rows, full_cols


def _choose_cells(full_rows: int, full_cols: int, count: int) -> list[tuple[int, int]]:
    even_cells = [(r, c) for r in range(full_rows) for c in range(full_cols) if (r + c) % 2 == 0]
    odd_cells = [(r, c) for r in range(full_rows) for c in range(full_cols) if (r + c) % 2 == 1]

    even_min = max(1, count - len(odd_cells))
    even_max = min(len(even_cells), count - 1)
    if even_min > even_max:
        return []

    even_count = random.randint(even_min, even_max)
    odd_count = count - even_count
    chosen = random.sample(even_cells, even_count) + random.sample(odd_cells, odd_count)
    random.shuffle(chosen)
    return chosen


def _paint_object(
    grid: np.ndarray,
    coarse_cell: tuple[int, int],
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> None:
    base_r = coarse_cell[0] * COARSE_CELL + 1
    base_c = coarse_cell[1] * COARSE_CELL + 1
    for dr, dc in shape:
        grid[base_r + dr, base_c + dc] = color


def _gen_attempt() -> np.ndarray | None:
    height, width, full_rows, full_cols = _choose_layout()
    grid = np.full((height, width), BACKGROUND, dtype=int)

    max_objects = min(MAX_OBJECTS, len(PALETTE), full_rows * full_cols)
    count = random.randint(MIN_OBJECTS, max_objects)
    cells = _choose_cells(full_rows, full_cols, count)
    if not cells:
        return None

    colors = random.sample(PALETTE, count)
    for coarse_cell, color in zip(cells, colors):
        _paint_object(grid, coarse_cell, random.choice(SAFE_SHAPES), color)

    if len({_coarse_parity(comp["bbox"]) for comp in _find_components(grid)}) != 2:
        return None

    return grid if np.count_nonzero(main(grid)[:, :width]) > 0 else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
