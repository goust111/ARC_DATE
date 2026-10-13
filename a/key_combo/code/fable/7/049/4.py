from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _infer_separator_structure(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("expected a square 2D grid")

    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int], int]] = []
    for color in np.unique(g):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(rows) < 3 or len(rows) != len(cols):
            continue
        if rows[0] != 0 or cols[0] != 0 or rows[-1] != h - 1 or cols[-1] != w - 1:
            continue

        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(row_diffs.tolist())) != 1 or len(set(col_diffs.tolist())) != 1:
            continue
        step = int(row_diffs[0])
        if step != int(col_diffs[0]):
            continue

        n = len(rows) - 1
        if step != n + 1:
            continue
        expected = list(range(0, h, step))
        if rows != expected or cols != expected:
            continue
        candidates.append((int(color), rows, cols, n))

    if len(candidates) != 1:
        raise ValueError("failed to infer a unique separator grid")
    return candidates[0]


def _extract_cells(grid: np.ndarray, row_seps: list[int], col_seps: list[int], n: int) -> list[list[np.ndarray]]:
    cells: list[list[np.ndarray]] = []
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            r0, r1 = row_seps[i] + 1, row_seps[i + 1]
            c0, c1 = col_seps[j] + 1, col_seps[j + 1]
            cell = np.asarray(grid[r0:r1, c0:c1], dtype=int).copy()
            if cell.shape != (n, n):
                raise ValueError("cell interior shape does not match macro grid size")
            row.append(cell)
        cells.append(row)
    return cells


def _infer_background(cells: list[list[np.ndarray]], separator_color: int) -> int:
    presence = Counter()
    totals = Counter()
    for row in cells:
        for cell in row:
            vals, cnts = np.unique(cell, return_counts=True)
            for val, cnt in zip(vals.tolist(), cnts.tolist()):
                val = int(val)
                if val == separator_color:
                    continue
                presence[val] += 1
                totals[val] += int(cnt)
    if not presence:
        raise ValueError("no cell-interior colors found")
    max_presence = max(presence.values())
    candidates = [color for color, count in presence.items() if count == max_presence]
    return max(candidates, key=lambda color: totals[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the cell whose inner drawing is exactly the coarse map of non-empty cells, then output that drawing.

    Input structure:
    - A single-color separator grid divides the square canvas into N×N cells.
    - Each cell interior is itself an N×N patch.
    - Empty cells contain only the shared background color; every non-empty cell contains a unique single-color pattern on that background.

    Rule:
    - Infer the separator lattice and extract all N×N cell interiors.
    - Build the macro occupancy mask that marks which cells are non-empty.
    - Compare that macro mask to each non-empty cell's inner boolean pattern (foreground vs. background).
    - Exactly one cell's inner mask matches the macro mask; return that cell interior with its original color.

    Key insight:
    - The target is identified only by cross-scale self-reference: one micro-pattern literally draws the macro layout of the non-empty cells.
    """
    g = np.asarray(grid, dtype=int)
    separator_color, row_seps, col_seps, n = _infer_separator_structure(g)
    cells = _extract_cells(g, row_seps, col_seps, n)
    bg = _infer_background(cells, separator_color)

    macro_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            macro_mask[i, j] = bool(np.any(cells[i][j] != bg))

    matches: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if not macro_mask[i, j]:
                continue
            if np.array_equal(cells[i][j] != bg, macro_mask):
                matches.append((i, j))

    if len(matches) != 1:
        raise ValueError("expected exactly one self-referential cell")
    r, c = matches[0]
    return cells[r][c].copy()


def _random_nonfull_mask(n: int, banned: set[bytes]) -> np.ndarray | None:
    for _ in range(200):
        count = random.randint(1, n * n - 1)
        flat = np.zeros(n * n, dtype=bool)
        flat[random.sample(range(n * n), count)] = True
        mask = flat.reshape(n, n)
        key = mask.tobytes()
        if key not in banned:
            return mask
    return None


def _build_grid(
    n: int,
    separator_color: int,
    background_color: int,
    macro_mask: np.ndarray,
    target_pos: tuple[int, int],
    pattern_by_cell: dict[tuple[int, int], np.ndarray],
    color_by_cell: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    side = n * (n + 1) + 1
    grid = np.full((side, side), separator_color, dtype=int)

    target_output = None
    for r in range(n):
        for c in range(n):
            r0 = 1 + r * (n + 1)
            c0 = 1 + c * (n + 1)
            cell = np.full((n, n), background_color, dtype=int)
            if macro_mask[r, c]:
                mask = pattern_by_cell[(r, c)]
                color = color_by_cell[(r, c)]
                cell[mask] = color
                if (r, c) == target_pos:
                    target_output = cell.copy()
            grid[r0 : r0 + n, c0 : c0 + n] = cell

    if target_output is None:
        raise ValueError("target cell was not rendered")
    return grid, target_output


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((2, 3, 4))
    max_active = min(8, n * n - 1)
    active_count = random.randint(2, max_active)

    positions = [(r, c) for r in range(n) for c in range(n)]
    active_positions = random.sample(positions, active_count)
    macro_mask = np.zeros((n, n), dtype=bool)
    for pos in active_positions:
        macro_mask[pos] = True

    target_pos = random.choice(active_positions)
    colors = list(PALETTE)
    separator_color, background_color = random.sample(colors, 2)
    usable_colors = [color for color in colors if color not in {separator_color, background_color}]
    if active_count > len(usable_colors):
        return None
    random.shuffle(usable_colors)

    pattern_by_cell: dict[tuple[int, int], np.ndarray] = {target_pos: macro_mask.copy()}
    banned = {macro_mask.tobytes()}
    for pos in active_positions:
        if pos == target_pos:
            continue
        mask = _random_nonfull_mask(n, banned)
        if mask is None:
            return None
        pattern_by_cell[pos] = mask
        banned.add(mask.tobytes())

    color_by_cell = {
        pos: int(color)
        for pos, color in zip(active_positions, usable_colors[:active_count], strict=True)
    }
    grid, expected = _build_grid(
        n=n,
        separator_color=separator_color,
        background_color=background_color,
        macro_mask=macro_mask,
        target_pos=target_pos,
        pattern_by_cell=pattern_by_cell,
        color_by_cell=color_by_cell,
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential separator-grid puzzle")
