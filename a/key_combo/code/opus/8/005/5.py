from __future__ import annotations

import numpy as np


CONTROL_BG = 6
CONTROL_MARK = 2
MASK = 5
STRIPE_COLORS = (1, 3, 4, 7)
_CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = np.random.default_rng()


def _infer_shift(grid: np.ndarray) -> int:
    marks = np.flatnonzero(grid[:, 0] == CONTROL_MARK)
    assert len(marks) == 1, "expected exactly one red marker in the control column"
    return int(marks[0])


def _strip_control_and_roll(grid: np.ndarray) -> np.ndarray:
    shift = _infer_shift(grid)
    main = np.asarray(grid[:, 1:], dtype=int)
    return np.roll(main, shift=shift, axis=0)


def _line_color(values: np.ndarray) -> int:
    visible = values[values != MASK]
    assert visible.size > 0, "every reference row/column must keep at least one visible stripe pixel"
    uniq = np.unique(visible)
    assert uniq.size == 1, "stripe references must be monochromatic after ignoring the mask"
    return int(uniq[0])


def _detect_orientation(grid: np.ndarray) -> str:
    horizontal = all(np.unique(row[row != MASK]).size == 1 for row in grid if np.any(row != MASK))
    vertical = all(np.unique(col[col != MASK]).size == 1 for col in grid.T if np.any(col != MASK))

    if horizontal and not vertical:
        return "horizontal"
    if vertical and not horizontal:
        return "vertical"
    if horizontal and vertical:
        row_colors = [_line_color(row) for row in grid if np.any(row != MASK)]
        col_colors = [_line_color(col) for col in grid.T if np.any(col != MASK)]
        if len(set(row_colors)) > 1 and len(set(col_colors)) == 1:
            return "horizontal"
        if len(set(col_colors)) > 1 and len(set(row_colors)) == 1:
            return "vertical"
    raise AssertionError("could not determine a unique stripe orientation")


def _repair_mask(grid: np.ndarray, orientation: str) -> np.ndarray:
    out = grid.copy()
    if orientation == "horizontal":
        row_colors = [_line_color(out[r, :]) for r in range(out.shape[0])]
        for r in range(out.shape[0]):
            out[r, out[r] == MASK] = row_colors[r]
        return out

    assert orientation == "vertical"
    col_colors = [_line_color(out[:, c]) for c in range(out.shape[1])]
    for c in range(out.shape[1]):
        out[out[:, c] == MASK, c] = col_colors[c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll the striped main region by the control-row amount and repair gray cross-mask pixels from the shifted stripe direction.

    Input structure:
    - The leftmost column is a magenta control column containing exactly one red marker pixel.
    - The remaining cells form a striped region whose non-gray cells are either row-wise monochromatic or column-wise monochromatic.
    - One or two gray plus-shaped mask fragments overwrite the underlying stripe colors inside the main region.

    Rule:
    - Read the red marker row index r from the control column, remove that column, and cyclically roll the remaining grid downward by r rows.
    - In the rolled grid, ignore gray pixels and determine whether the visible stripes are horizontal or vertical.
    - Replace every gray pixel with the color from its row when stripes are horizontal, or from its column when stripes are vertical.

    Key insight:
    - The control column changes the stripe locations before reconstruction, and the stripe orientation decides whether rows or columns are the valid source of missing colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a control column and a main region"

    rolled = _strip_control_and_roll(g)
    orientation = _detect_orientation(rolled)
    return _repair_mask(rolled, orientation)


def _make_sequence(length: int, palette: list[int]) -> list[int]:
    order = list(_RNG.permutation(palette))
    seq: list[int] = []
    prev = None
    unused = order.copy()
    while len(seq) < length:
        if unused:
            choices = [color for color in unused if color != prev]
            if not choices:
                choices = unused.copy()
            color = int(choices[int(_RNG.integers(len(choices)))])
            unused.remove(color)
        else:
            choices = [color for color in palette if color != prev]
            color = int(choices[int(_RNG.integers(len(choices)))])
        seq.append(color)
        prev = color
    return seq


def _paint_stripes(height: int, width: int, orientation: str, palette: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    if orientation == "horizontal":
        row_colors = _make_sequence(height, palette)
        for r, color in enumerate(row_colors):
            grid[r, :] = color
        return grid

    col_colors = _make_sequence(width, palette)
    for c, color in enumerate(col_colors):
        grid[:, c] = color
    return grid


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in _CROSS_OFFSETS}


def _place_masks(main_grid: np.ndarray, count: int) -> bool:
    height, width = main_grid.shape
    centers = [(r, c) for r in range(2, height - 2) for c in range(2, width - 2)]
    if not centers:
        return False

    chosen_cells: set[tuple[int, int]] = set()
    _RNG.shuffle(centers)
    placed = 0
    for center in centers:
        cells = _cross_cells(center)
        if chosen_cells & cells:
            continue
        for r, c in cells:
            main_grid[r, c] = MASK
        chosen_cells |= cells
        placed += 1
        if placed == count:
            return True
    return False


def _build_candidate() -> np.ndarray:
    height = int(_RNG.integers(7, 11))
    width = int(_RNG.integers(8, 13))
    orientation = "horizontal" if int(_RNG.integers(2)) == 0 else "vertical"

    max_palette = min(4, height if orientation == "horizontal" else width)
    palette_size = int(_RNG.integers(2, max_palette + 1))
    palette = [int(c) for c in _RNG.choice(STRIPE_COLORS, size=palette_size, replace=False)]

    main_grid = _paint_stripes(height, width, orientation, palette)

    mask_count = 1 if int(_RNG.integers(100)) < 55 else 2
    if not _place_masks(main_grid, mask_count):
        raise RuntimeError("failed to place masks")

    control = np.full((height, 1), CONTROL_BG, dtype=int)
    shift = int(_RNG.integers(height))
    control[shift, 0] = CONTROL_MARK
    return np.concatenate([control, main_grid], axis=1)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_candidate()
        out = main(grid)
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.any(out == MASK):
            continue
        if np.array_equal(out, grid[:, 1:]):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
