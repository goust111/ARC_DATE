from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREEN = 3
CONTENT_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _full_green_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREEN)]


def _full_green_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREEN)]


def _gcd_step(lines: list[int], limit: int) -> int:
    assert len(lines) >= 3, "need at least three observed separator lines"
    assert lines[0] == 0 and lines[-1] == limit - 1, "outer border lines must be visible"
    diffs = [b - a for a, b in zip(lines, lines[1:])]
    step = diffs[0]
    for diff in diffs[1:]:
        step = math.gcd(step, diff)
    assert step >= 3, "cell interior must be at least 2 px tall/wide"
    assert (limit - 1) % step == 0, "separator period must tile the canvas exactly"
    return step


def _infer_full_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = _full_green_rows(grid)
    cols = _full_green_cols(grid)
    row_step = _gcd_step(rows, grid.shape[0])
    col_step = _gcd_step(cols, grid.shape[1])
    full_rows = list(range(0, grid.shape[0], row_step))
    full_cols = list(range(0, grid.shape[1], col_step))
    assert full_rows[-1] == grid.shape[0] - 1
    assert full_cols[-1] == grid.shape[1] - 1
    return full_rows, full_cols


def _snake_order(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else range(n_cols - 1, -1, -1)
        for c in cols:
            order.append((r, c))
    return order


def _row_major_order(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _extract_patches(
    grid: np.ndarray, row_lines: list[int], col_lines: list[int]
) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for r in range(len(row_lines) - 1):
        for c in range(len(col_lines) - 1):
            r0, r1 = row_lines[r] + 1, row_lines[r + 1]
            c0, c1 = col_lines[c] + 1, col_lines[c + 1]
            patch = grid[r0:r1, c0:c1].copy()
            assert patch.size > 0
            assert not np.any(patch == GREEN), "cell interiors must not contain separator color"
            patches.append(patch)
    return patches


def _build_scaffold(shape: tuple[int, int], row_lines: list[int], col_lines: list[int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[row_lines, :] = GREEN
    grid[:, col_lines] = GREEN
    return grid


def _render_shifted(
    shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
    patches: list[np.ndarray],
    order: list[tuple[int, int]],
) -> np.ndarray:
    out = _build_scaffold(shape, row_lines, col_lines)
    patch_by_cell = {
        (r, c): patch for (r, c), patch in zip(_row_major_order(len(row_lines) - 1, len(col_lines) - 1), patches)
    }
    for idx, dst_cell in enumerate(order):
        src_cell = order[(idx - 1) % len(order)]
        patch = patch_by_cell[src_cell]
        r, c = dst_cell
        r0, r1 = row_lines[r] + 1, row_lines[r + 1]
        c0, c1 = col_lines[c] + 1, col_lines[c + 1]
        assert patch.shape == (r1 - r0, c1 - c0)
        out[r0:r1, c0:c1] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing periodic green grid lines, then rotate all cell contents one seat forward in boustrophedon order.

    Input structure:
    - A black background contains a green separator lattice whose horizontal and vertical lines are equally spaced, but some whole separator lines are absent.
    - The missing lines are inferred from the remaining periodic scaffold; each cell interior has at most one non-green monochrome sprite, and empty cells are allowed.
    - The grid uses a complete outer border, so the restored lattice partitions the canvas into equal-sized rectangular cells.

    Rule:
    - Recover every missing full green row and column implied by the observed lattice period.
    - Read the cells in snake order: first row left-to-right, second row right-to-left, then alternating; the last cell wraps around to the first.
    - Move each cell's entire interior content, including emptiness, to the next cell in that cyclic snake order while preserving its pixel offsets inside the cell; keep all green separator lines.

    Key insight:
    - The correct transport order is defined by the hidden fine grid, not by the larger merged regions visible before the missing lines are restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_lines, col_lines = _infer_full_lines(g)
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    assert n_rows >= 3 and n_cols >= 3

    row_step = row_lines[1] - row_lines[0]
    col_step = col_lines[1] - col_lines[0]
    assert all(b - a == row_step for a, b in zip(row_lines, row_lines[1:]))
    assert all(b - a == col_step for a, b in zip(col_lines, col_lines[1:]))

    patches = _extract_patches(g, row_lines, col_lines)
    order = _snake_order(n_rows, n_cols)
    return _render_shifted(g.shape, row_lines, col_lines, patches, order)


def _random_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    max_area = min(height * width - 1, 5)
    if max_area <= 0:
        max_area = 1
    target_area = random.randint(1, max_area)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    for r, c in cells:
        patch[r, c] = color
    return patch


def _sample_missing_indices(count: int, want_missing: bool) -> list[int]:
    if not want_missing:
        return []
    internal = list(range(1, count))
    if len(internal) <= 1:
        return []
    random.shuffle(internal)
    max_missing = min(len(internal) - 1, 2)
    missing_n = random.randint(1, max_missing)
    missing = sorted(internal[:missing_n])
    return missing


def _coarse_cell_id(
    full_rows: list[int],
    full_cols: list[int],
    observed_rows: list[int],
    observed_cols: list[int],
    cell: tuple[int, int],
) -> tuple[int, int]:
    r, c = cell
    rr = full_rows[r] + 1
    cc = full_cols[c] + 1
    coarse_r = next(i for i in range(len(observed_rows) - 1) if observed_rows[i] < rr < observed_rows[i + 1])
    coarse_c = next(i for i in range(len(observed_cols) - 1) if observed_cols[i] < cc < observed_cols[i + 1])
    return coarse_r, coarse_c


def _repair_is_necessary(
    filled_cells: set[tuple[int, int]],
    full_rows: list[int],
    full_cols: list[int],
    observed_rows: list[int],
    observed_cols: list[int],
) -> bool:
    order = _snake_order(len(full_rows) - 1, len(full_cols) - 1)
    coarse_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for cell in filled_cells:
        coarse_groups[_coarse_cell_id(full_rows, full_cols, observed_rows, observed_cols, cell)].append(cell)

    for cells in coarse_groups.values():
        non_empty = [cell for cell in cells if cell in filled_cells]
        if len(non_empty) < 2:
            continue
        dst_groups = set()
        for cell in non_empty:
            idx = order.index(cell)
            dst = order[(idx + 1) % len(order)]
            dst_groups.add(_coarse_cell_id(full_rows, full_cols, observed_rows, observed_cols, dst))
        if len(dst_groups) >= 2:
            return True
    return False


def _candidate_layout() -> tuple[int, int, int, int]:
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)
    cell_h = random.randint(2, 4)
    cell_w = random.randint(2, 4)
    return n_rows, n_cols, cell_h, cell_w


def _grid_lines(cell_count: int, cell_size: int) -> list[int]:
    step = cell_size + 1
    return list(range(0, cell_count * step + 1, step))


def _make_input_from_patches(
    patches_by_cell: dict[tuple[int, int], np.ndarray],
    row_lines: list[int],
    col_lines: list[int],
    observed_rows: list[int],
    observed_cols: list[int],
) -> np.ndarray:
    grid = _build_scaffold((row_lines[-1] + 1, col_lines[-1] + 1), observed_rows, observed_cols)
    for (r, c), patch in patches_by_cell.items():
        r0, r1 = row_lines[r] + 1, row_lines[r + 1]
        c0, c1 = col_lines[c] + 1, col_lines[c + 1]
        assert patch.shape == (r1 - r0, c1 - c0)
        grid[r0:r1, c0:c1] = patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols, cell_h, cell_w = _candidate_layout()
    row_lines = _grid_lines(n_rows, cell_h)
    col_lines = _grid_lines(n_cols, cell_w)

    branch = random.choices(
        ["h_only", "v_only", "both"],
        weights=[3, 3, 4],
        k=1,
    )[0]
    miss_h = branch in {"h_only", "both"}
    miss_v = branch in {"v_only", "both"}
    missing_h = _sample_missing_indices(n_rows, miss_h)
    missing_v = _sample_missing_indices(n_cols, miss_v)
    observed_rows = [line for i, line in enumerate(row_lines) if i not in missing_h]
    observed_cols = [line for i, line in enumerate(col_lines) if i not in missing_v]

    if _gcd_step(observed_rows, row_lines[-1] + 1) != cell_h + 1:
        return None
    if _gcd_step(observed_cols, col_lines[-1] + 1) != cell_w + 1:
        return None

    all_cells = _row_major_order(n_rows, n_cols)
    max_filled = min(len(all_cells) - 1, len(CONTENT_COLORS))
    if max_filled < 5:
        return None
    filled_n = random.randint(5, max_filled)
    random.shuffle(all_cells)
    filled_cells = set(all_cells[:filled_n])

    patches_by_cell: dict[tuple[int, int], np.ndarray] = {}
    available_colors = random.sample(CONTENT_COLORS, filled_n)
    color_iter = iter(available_colors)
    for cell in _row_major_order(n_rows, n_cols):
        if cell in filled_cells:
            color = next(color_iter)
            patch = _random_patch(cell_h, cell_w, color)
        else:
            patch = np.zeros((cell_h, cell_w), dtype=int)
        patches_by_cell[cell] = patch

    inp = _make_input_from_patches(patches_by_cell, row_lines, col_lines, observed_rows, observed_cols)
    expected_patches = [patches_by_cell[cell] for cell in _row_major_order(n_rows, n_cols)]
    expected = _render_shifted(inp.shape, row_lines, col_lines, expected_patches, _snake_order(n_rows, n_cols))

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None

    row_major_out = _render_shifted(inp.shape, row_lines, col_lines, expected_patches, _row_major_order(n_rows, n_cols))
    if np.array_equal(row_major_out, expected):
        return None

    if (missing_h or missing_v) and not _repair_is_necessary(filled_cells, row_lines, col_lines, observed_rows, observed_cols):
        return None

    if len(_full_green_rows(inp)) == len(row_lines) and len(_full_green_cols(inp)) == len(col_lines):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
