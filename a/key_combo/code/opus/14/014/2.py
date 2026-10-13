from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox, crop_bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BG = int(Color.BLACK)
SCAFFOLD = int(Color.GREY)
ROWCOL_BORDER = int(Color.RED)
ROWCOL_FILL = int(Color.GREEN)
SHAPE_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW))

GRID_SIDE_MIN = 3
GRID_SIDE_MAX = 6
GEN_ATTEMPTS = 400


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                key = (int(nr), int(nc))
                if key not in seen:
                    seen.add(key)
                    stack.append(key)
    return len(seen) == int(mask.sum())


def _lr_symmetric(mask: np.ndarray) -> bool:
    box = bbox(mask)
    assert box is not None, "shape mask must be non-empty"
    tight = crop_bbox(mask.astype(int), box).astype(bool)
    return bool(np.array_equal(tight, np.fliplr(tight)))


def _shape_catalog() -> tuple[list[dict], list[dict]]:
    prototypes = [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
    ]

    sym: list[dict] = []
    asym: list[dict] = []
    seen = set()
    for proto in prototypes:
        h, w = proto.shape
        for r0 in range(4 - h):
            for c0 in range(4 - w):
                patch = np.zeros((3, 3), dtype=bool)
                patch[r0 : r0 + h, c0 : c0 + w] = proto
                key = tuple(map(int, patch.flatten()))
                if key in seen:
                    continue
                seen.add(key)
                entry = {
                    "mask": patch,
                    "area": int(patch.sum()),
                    "includes_center": bool(patch[1, 1]),
                }
                if _lr_symmetric(patch):
                    sym.append(entry)
                else:
                    asym.append(entry)
    return sym, asym


SYMMETRIC_SHAPES, ASYMMETRIC_SHAPES = _shape_catalog()
SYMMETRIC_AREA3 = [shape for shape in SYMMETRIC_SHAPES if shape["area"] == 3]


def _build_scaffold(n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.zeros((4 * n_rows + 1, 4 * n_cols + 1), dtype=int)
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0 = 4 * rr + 1
            c0 = 4 * cc + 1
            grid[r0 : r0 + 3, c0 : c0 + 3] = SCAFFOLD
            grid[r0 + 1, c0 + 1] = BG
    return grid


def _boundary_lines(grid: np.ndarray, axis: int) -> list[int]:
    lines = []
    limit = grid.shape[axis]
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == BG):
            lines.append(int(idx))
    assert len(lines) >= 4, "expected at least a 3x3 scaffold"
    assert lines[0] == 0 and lines[-1] == limit - 1, "scaffold must touch the canvas border"
    assert all(b - a == 4 for a, b in zip(lines, lines[1:])), "boundary lines must be spaced by 4"
    return lines


def _frame_patch(grid: np.ndarray, row_idx: int, col_idx: int, row_bounds: list[int], col_bounds: list[int]) -> np.ndarray:
    r0 = row_bounds[row_idx] + 1
    c0 = col_bounds[col_idx] + 1
    return grid[r0 : r0 + 3, c0 : c0 + 3]


def _shape_mask_from_patch(patch: np.ndarray) -> np.ndarray:
    return ~np.isin(patch, (BG, SCAFFOLD))


def _analyze_grid(grid: np.ndarray) -> tuple[list[int], list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_bounds = _boundary_lines(g, axis=0)
    col_bounds = _boundary_lines(g, axis=1)
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    assert 3 <= n_rows <= 6 and 3 <= n_cols <= 6, "unexpected scaffold size"

    asymmetric_cells: list[tuple[int, int]] = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            patch = _frame_patch(g, rr, cc, row_bounds, col_bounds)
            assert patch.shape == (3, 3), "every scaffold cell must be 3x3"
            shape_mask = _shape_mask_from_patch(patch)
            area = int(shape_mask.sum())
            assert 1 <= area <= 3, "every frame must contain a 1-3 pixel figure"
            assert _connected4(shape_mask), "every figure must be 4-connected"
            shape_colors = {int(v) for v in np.unique(patch[shape_mask])}
            assert len(shape_colors) == 1, "each figure must be monochromatic"
            if not _lr_symmetric(shape_mask):
                asymmetric_cells.append((rr, cc))

    assert len(asymmetric_cells) == 1, "expected exactly one left-right-asymmetric figure"
    return row_bounds, col_bounds, asymmetric_cells[0][0], asymmetric_cells[0][1]


def _expected_output(grid: np.ndarray, target_row: int, target_col: int, row_bounds: list[int], col_bounds: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1

    for cc in range(n_cols):
        r0 = row_bounds[target_row] + 1
        c0 = col_bounds[cc] + 1
        patch = out[r0 : r0 + 3, c0 : c0 + 3]
        patch[patch == SCAFFOLD] = ROWCOL_BORDER
    for rr in range(n_rows):
        r0 = row_bounds[rr] + 1
        c0 = col_bounds[target_col] + 1
        patch = out[r0 : r0 + 3, c0 : c0 + 3]
        patch[patch == SCAFFOLD] = ROWCOL_BORDER

    for cc in range(n_cols):
        out[row_bounds[target_row] + 2, col_bounds[cc] + 2] = ROWCOL_FILL
    for rr in range(n_rows):
        out[row_bounds[rr] + 2, col_bounds[target_col] + 2] = ROWCOL_FILL
    out[row_bounds[target_row] + 2, col_bounds[target_col] + 2] = ROWCOL_BORDER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique left-right-asymmetric frame figure and expand a colored scaffold cross through its grid row and column.

    Input structure:
    - A black background contains a regular lattice of hollow 3x3 grey frames separated by all-black boundary rows and columns.
    - Every frame contains one monochromatic 1-3 pixel figure using one of the non-grey foreground colors.
    - Exactly one frame figure is not mirror-symmetric left-to-right inside its own tight bounding box.

    Rule:
    - Infer the scaffold row and column boundaries from the periodic all-black separator lines.
    - Inspect each frame's local figure mask, crop it to its tight bounding box, and find the unique left-right-asymmetric figure.
    - Let that figure sit at scaffold row R and column C; recolor every grey border segment in row R and column C to red.
    - Fill the center hole of every frame in row R or column C with green, except the intersection frame at (R, C), whose center becomes red.

    Key insight:
    - The asymmetric figure is only a locator: the solver must convert object-level asymmetry into scaffold coordinates, then use those coordinates to drive a separate row-and-column expansion.
    """
    row_bounds, col_bounds, target_row, target_col = _analyze_grid(grid)
    return _expected_output(grid, target_row, target_col, row_bounds, col_bounds)


def _place_shape(grid: np.ndarray, cell_row: int, cell_col: int, shape_mask: np.ndarray, color: int) -> None:
    r0 = 4 * cell_row + 1
    c0 = 4 * cell_col + 1
    patch = grid[r0 : r0 + 3, c0 : c0 + 3]
    patch[shape_mask] = int(color)


def _sample_cell_specs(n_rows: int, n_cols: int):
    target_row = random.randrange(n_rows)
    target_col = random.randrange(n_cols)
    target_color = random.choice(SHAPE_COLORS)
    target_shape = random.choice(ASYMMETRIC_SHAPES)
    specs = {}
    repeated_target = False

    for rr in range(n_rows):
        for cc in range(n_cols):
            if (rr, cc) == (target_row, target_col):
                specs[(rr, cc)] = (target_shape, target_color)
                continue
            if not repeated_target and random.random() < 0.35:
                shape = random.choice(SYMMETRIC_AREA3)
                color = target_color
                repeated_target = True
            else:
                shape = random.choice(SYMMETRIC_SHAPES)
                color = random.choice(SHAPE_COLORS)
            specs[(rr, cc)] = (shape, color)

    if not repeated_target:
        candidates = [(rr, cc) for rr in range(n_rows) for cc in range(n_cols) if (rr, cc) != (target_row, target_col)]
        chosen = random.choice(candidates)
        specs[chosen] = (random.choice(SYMMETRIC_AREA3), target_color)
        repeated_target = True

    colors_used = {color for _shape, color in specs.values()}
    if len(colors_used) < 2:
        candidates = [cell for cell in specs if cell != (target_row, target_col)]
        chosen = random.choice(candidates)
        new_color = random.choice([color for color in SHAPE_COLORS if color != target_color])
        specs[chosen] = (random.choice(SYMMETRIC_SHAPES), new_color)

    return specs, target_row, target_col, target_color


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, target_row: int, target_col: int, target_color: int) -> bool:
    try:
        row_bounds, col_bounds, found_row, found_col = _analyze_grid(inp)
    except AssertionError:
        return False
    if (found_row, found_col) != (target_row, target_col):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    if n_rows < 3 or n_cols < 3:
        return False

    cross_centers = {(row_bounds[target_row] + 2, col_bounds[cc] + 2) for cc in range(n_cols)}
    cross_centers |= {(row_bounds[rr] + 2, col_bounds[target_col] + 2) for rr in range(n_rows)}
    green_count = sum(int(solved[r, c] == ROWCOL_FILL) for r, c in cross_centers)
    red_count = sum(int(solved[r, c] == ROWCOL_BORDER) for r, c in cross_centers)
    if green_count != n_rows + n_cols - 2 or red_count != 1:
        return False

    shape_infos = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            patch = _frame_patch(inp, rr, cc, row_bounds, col_bounds)
            shape_mask = _shape_mask_from_patch(patch)
            color = int(next(iter(np.unique(patch[shape_mask]))))
            shape_infos.append((rr, cc, int(shape_mask.sum()), color, _lr_symmetric(shape_mask)))

    if sum(int(area == 3 and color == target_color and is_sym) for _rr, _cc, area, color, is_sym in shape_infos) < 1:
        return False
    if len({color for _rr, _cc, _area, color, _is_sym in shape_infos}) < 2:
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_rows = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    n_cols = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = _build_scaffold(n_rows, n_cols)
    specs, target_row, target_col, target_color = _sample_cell_specs(n_rows, n_cols)

    for (rr, cc), (shape, color) in specs.items():
        _place_shape(grid, rr, cc, shape["mask"], color)

    row_bounds = [4 * idx for idx in range(n_rows + 1)]
    col_bounds = [4 * idx for idx in range(n_cols + 1)]
    expected = _expected_output(grid, target_row, target_col, row_bounds, col_bounds)

    if not _semantic_checks(grid, expected, target_row, target_col, target_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid asymmetric-scaffold input")
