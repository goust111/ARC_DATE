from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _line_color_candidates(lines: list[np.ndarray]) -> set[int]:
    return {int(line[0]) for line in lines if line.size > 0 and np.all(line == line[0])}


def _infer_colors(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_candidates = _line_color_candidates([grid[r, :] for r in range(h)])
    col_candidates = _line_color_candidates([grid[:, c] for c in range(w)])
    shared = row_candidates & col_candidates
    assert len(shared) == 1, "expected exactly one grid-line color"
    line_color = next(iter(shared))

    vals, counts = np.unique(grid, return_counts=True)
    ordered = [int(v) for _, v in sorted(zip(counts.tolist(), vals.tolist()), reverse=True)]
    bg_candidates = [v for v in ordered if v != line_color]
    assert bg_candidates, "expected a background color distinct from grid lines"
    return line_color, bg_candidates[0]


def _bands(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in cuts:
        if start < cut:
            bands.append((start, cut))
        start = cut + 1
    if start < size:
        bands.append((start, size))
    return bands


def _cell_slices(grid: np.ndarray, line_color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_lines = [r for r in range(h) if np.all(grid[r, :] == line_color)]
    col_lines = [c for c in range(w) if np.all(grid[:, c] == line_color)]
    assert row_lines and col_lines, "expected full row/column lattice lines"

    row_bands = _bands(h, row_lines)
    col_bands = _bands(w, col_lines)
    assert row_bands and col_bands, "expected non-empty cells"

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "cells must be equally sized"
    assert next(iter(heights)) == next(iter(widths)), "cells must be square"
    return row_bands, col_bands


def _extract_cell_patterns(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], line_color: int, bg: int
) -> list[np.ndarray]:
    items: list[tuple[int, int, int, np.ndarray]] = []
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = grid[r0:r1, c0:c1]
            fg_mask = (cell != bg) & (cell != line_color)
            if not fg_mask.any():
                continue
            colors = {int(v) for v in np.unique(cell[fg_mask])}
            assert len(colors) == 1, "each pattern must be monochrome"
            cc = len(_connected_components(fg_mask))
            assert 1 <= cc <= 4, "component count must be within puzzle range"
            items.append((cc, ri * len(col_bands) + ci, int(next(iter(colors))), cell.copy()))

    assert items, "expected at least one non-empty cell"
    items.sort(key=lambda x: (x[0], x[1]))
    return [cell for _, _, _, cell in items]


def _render_row(cells: list[np.ndarray], line_color: int, bg: int) -> np.ndarray:
    cell_size = cells[0].shape[0]
    assert all(cell.shape == (cell_size, cell_size) for cell in cells), "all cells must share one size"

    out_h = cell_size + 2
    out_w = len(cells) * cell_size + len(cells) + 1
    out = np.full((out_h, out_w), bg, dtype=int)
    out[0, :] = line_color
    out[-1, :] = line_color

    x = 0
    for cell in cells:
        out[:, x] = line_color
        out[1:-1, x + 1 : x + 1 + cell_size] = cell
        x += cell_size + 1
    out[:, x] = line_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract all non-empty lattice cells, sort them by 4-connected component count, and render them in one row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    line_color, bg = _infer_colors(g)
    row_bands, col_bands = _cell_slices(g, line_color)
    cells = _extract_cell_patterns(g, row_bands, col_bands, line_color, bg)
    return _render_row(cells, line_color, bg)


COMPONENT_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)

GEN_ATTEMPTS = 400
CELL_SIZE_RANGE = (5, 7)
GRID_ROWS_RANGE = (2, 5)
GRID_COLS_RANGE = (3, 6)
NONEMPTY_RANGE = (3, 10)


def _place_component(mask: np.ndarray, shape: np.ndarray) -> bool:
    h, w = mask.shape
    sh, sw = shape.shape
    positions = []
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            ok = True
            for dr in range(sh):
                for dc in range(sw):
                    if not shape[dr, dc]:
                        continue
                    rr, cc = r + dr, c + dc
                    if mask[rr, cc]:
                        ok = False
                        break
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                positions.append((r, c))
    if not positions:
        return False
    r, c = random.choice(positions)
    mask[r : r + sh, c : c + sw] |= shape
    return True


def _make_pattern(cell_size: int, n_components: int) -> np.ndarray | None:
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    for _ in range(n_components):
        choices = list(COMPONENT_SHAPES)
        random.shuffle(choices)
        placed = False
        for shape in choices:
            if _place_component(mask, shape):
                placed = True
                break
        if not placed:
            return None
    if len(_connected_components(mask)) != n_components:
        return None
    return mask


def _build_input_from_specs(
    cell_size: int, rows: int, cols: int, specs: list[tuple[tuple[int, int], int]]
) -> np.ndarray | None:
    grid = np.full((rows * cell_size + rows + 1, cols * cell_size + cols + 1), GREY, dtype=int)
    for r in range(0, grid.shape[0], cell_size + 1):
        grid[r, :] = BLACK
    for c in range(0, grid.shape[1], cell_size + 1):
        grid[:, c] = BLACK

    used_colors: Counter[int] = Counter()
    for (ri, ci), n_components in specs:
        pattern = _make_pattern(cell_size, n_components)
        if pattern is None:
            return None
        available = [c for c in PALETTE if used_colors[c] == min(used_colors.values(), default=0)]
        color = random.choice(available if available else PALETTE)
        used_colors[color] += 1
        r0 = 1 + ri * (cell_size + 1)
        c0 = 1 + ci * (cell_size + 1)
        cell = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
        cell[pattern] = color
    return grid


def _sample_specs(rows: int, cols: int) -> list[tuple[tuple[int, int], int]]:
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(positions)
    max_nonempty = min(len(positions), NONEMPTY_RANGE[1])
    count = random.randint(NONEMPTY_RANGE[0], max_nonempty)
    chosen = positions[:count]

    components = [random.randint(1, 4) for _ in range(count)]
    if len(set(components)) == 1:
        idx = random.randrange(count)
        components[idx] = 1 + (components[idx] % 4)
    return list(zip(chosen, components))


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.randint(*CELL_SIZE_RANGE)
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    if rows * cols < NONEMPTY_RANGE[0]:
        return None

    specs = _sample_specs(rows, cols)
    grid = _build_input_from_specs(cell_size, rows, cols, specs)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    expected_count = len(specs)
    expected_shape = (cell_size + 2, expected_count * cell_size + expected_count + 1)
    if out.shape != expected_shape:
        return None

    counts = [cc for _, cc in specs]
    if len(set(counts)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
