from __future__ import annotations

import random

import numpy as np


BLACK = 0
SEPARATOR_COLOR = 7
PANEL_SIZE_RANGE = (2, 5)
FOREGROUND_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_ATTEMPTS = 400


def _panel_foreground_color(panel: np.ndarray) -> int:
    colors = sorted(set(int(v) for v in np.unique(panel)) - {BLACK})
    assert len(colors) == 1, "each panel must use exactly one foreground color"
    return colors[0]


def _valid_panel(panel: np.ndarray) -> bool:
    if panel.ndim != 2 or panel.size == 0:
        return False
    colors = sorted(set(int(v) for v in np.unique(panel)) - {BLACK})
    return len(colors) == 1


def _find_separator_row(grid: np.ndarray) -> int:
    h, _w = grid.shape
    candidates: list[int] = []
    for r in range(h):
        color = int(grid[r, 0])
        if color == BLACK or not np.all(grid[r] == color):
            continue
        top = grid[:r]
        bottom = grid[r + 1 :]
        if top.shape != bottom.shape or top.size == 0:
            continue
        if not (_valid_panel(top) and _valid_panel(bottom)):
            continue
        top_color = _panel_foreground_color(top)
        bottom_color = _panel_foreground_color(bottom)
        if color in (top_color, bottom_color) or top_color == bottom_color:
            continue
        if np.count_nonzero(grid == color) != grid.shape[1]:
            continue
        candidates.append(r)
    assert len(candidates) == 1, "expected exactly one valid separator row"
    return candidates[0]


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator_row(grid)
    top = grid[:sep].copy()
    bottom = grid[sep + 1 :].copy()
    assert top.shape == bottom.shape and top.size > 0, "expected two equal non-empty panels"
    return top, bottom


def _render_output(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    n, m = top.shape
    top_color = _panel_foreground_color(top)
    bottom_color = _panel_foreground_color(bottom)
    assert top_color != bottom_color, "panel colors must be distinct"

    blueprint = (top == BLACK) & (bottom == BLACK)
    out = np.zeros((n * n, m * m), dtype=int)
    for br in range(n):
        for bc in range(m):
            block = top if blueprint[br, bc] else bottom
            r0 = br * n
            c0 = bc * m
            out[r0 : r0 + n, c0 : c0 + m] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the common empty cells of two stacked panels as a blueprint for fractal weaving.

    Input structure:
    - A unique solid non-black separator row splits the grid into two equal panels.
    - The top panel contains only black plus one foreground color A.
    - The bottom panel contains only black plus one foreground color B.

    Rule:
    - Remove the separator row and align the two panels cell by cell.
    - Mark each panel position whose top and bottom cells are both black; this boolean mask is the hidden blueprint.
    - Produce a black output of size (n*n) by (m*m), divided into n by m blocks of size n by m.
    - At blueprint-true block positions, copy the full top-panel pattern; otherwise copy the full bottom-panel pattern.

    Key insight:
    - The controlling object is the negative-space intersection of the two panels, not either visible foreground stamp by itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    top, bottom = _split_panels(g)
    return _render_output(top, bottom)


def _sample_panel_shape() -> tuple[int, int]:
    n = random.randint(*PANEL_SIZE_RANGE)
    m = random.randint(*PANEL_SIZE_RANGE)
    assert n * n <= 30 and m * m <= 30
    return n, m


def _category_grid(n: int, m: int) -> np.ndarray:
    total = n * m
    categories = np.empty(total, dtype=int)
    categories[:4] = np.array([0, 1, 2, 3], dtype=int)

    extra_weights = [1, 1, 1, 1]
    extras = random.choices([0, 1, 2, 3], weights=extra_weights, k=total - 4)
    categories[4:] = np.array(extras, dtype=int)
    category_list = categories.tolist()
    random.shuffle(category_list)
    categories = np.array(category_list, dtype=int).reshape(n, m)

    top_fg = (categories == 1) | (categories == 3)
    bottom_fg = (categories == 2) | (categories == 3)
    top_density = float(top_fg.mean())
    bottom_density = float(bottom_fg.mean())
    hidden_density = float((categories == 0).mean())
    if not (0.2 <= top_density <= 0.8):
        raise ValueError("top density out of range")
    if not (0.2 <= bottom_density <= 0.8):
        raise ValueError("bottom density out of range")
    if not (0.1 <= hidden_density <= 0.6):
        raise ValueError("hidden density out of range")
    return categories


def _build_panels(categories: np.ndarray, top_color: int, bottom_color: int) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros(categories.shape, dtype=int)
    bottom = np.zeros(categories.shape, dtype=int)
    top[(categories == 1) | (categories == 3)] = top_color
    bottom[(categories == 2) | (categories == 3)] = bottom_color
    return top, bottom


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    sep = _find_separator_row(grid)
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    n, m = top.shape
    hidden = (top == BLACK) & (bottom == BLACK)
    out = np.zeros((n * n, m * m), dtype=int)
    for r in range(n):
        for c in range(m):
            block = top if hidden[r, c] else bottom
            out[r * n : (r + 1) * n, c * m : (c + 1) * m] = block
    return out


def _check_semantics(grid: np.ndarray) -> bool:
    try:
        top, bottom = _split_panels(grid)
        if top.shape != bottom.shape:
            return False

        n, m = top.shape
        if max(grid.shape) > 30 or n * n > 30 or m * m > 30:
            return False

        top_color = _panel_foreground_color(top)
        bottom_color = _panel_foreground_color(bottom)
        if top_color == bottom_color or SEPARATOR_COLOR in (top_color, bottom_color):
            return False

        top_fg = top != BLACK
        bottom_fg = bottom != BLACK
        both_bg = (~top_fg) & (~bottom_fg)
        top_only = top_fg & (~bottom_fg)
        bottom_only = (~top_fg) & bottom_fg
        both_fg = top_fg & bottom_fg

        if not (both_bg.any() and top_only.any() and bottom_only.any() and both_fg.any()):
            return False
        if np.array_equal(both_bg, ~top_fg) or np.array_equal(both_bg, ~bottom_fg):
            return False

        out = main(grid)
        expected = _independent_expected(grid)
        if not np.array_equal(out, expected):
            return False

        hidden = both_bg
        for r in range(n):
            for c in range(m):
                block = out[r * n : (r + 1) * n, c * m : (c + 1) * m]
                if hidden[r, c]:
                    if not np.array_equal(block, top):
                        return False
                else:
                    if not np.array_equal(block, bottom):
                        return False

        if not hidden.any() or hidden.all():
            return False
        if not (~hidden).any():
            return False
        return True
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    n, m = _sample_panel_shape()
    if n * m < 4:
        return None

    top_color, bottom_color = random.sample(FOREGROUND_COLORS, 2)
    for _ in range(80):
        try:
            categories = _category_grid(n, m)
        except ValueError:
            continue
        top, bottom = _build_panels(categories, top_color, bottom_color)
        separator = np.full((1, m), SEPARATOR_COLOR, dtype=int)
        grid = np.vstack([top, separator, bottom])
        if _check_semantics(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
