from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, components_mask
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the red shape across its touched vertical edge into a teal-and-black negative half.

    Input structure
    The input is a black grid containing red foreground pixels. The foreground
    touches exactly one vertical border, and that border is the reflection seam.

    Rule
    Horizontally reflect the input. In the reflected half, turn black pixels
    teal and all red foreground pixels black. If the red foreground touches the
    left border, place the reflected half on the left and the original on the
    right; if it touches the right border, place the original on the left and
    the reflected half on the right.

    Key insight
    The output is the original grid joined to a color-remapped mirror image
    across the single vertical edge touched by the red foreground.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = Color.BLACK
    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors == [Color.RED], "expected red foreground on black background"

    foreground = g != bg
    assert np.any(foreground), "expected at least one foreground pixel"
    touches_left = bool(np.any(foreground[:, 0]))
    touches_right = bool(np.any(foreground[:, -1]))
    assert touches_left != touches_right, "foreground must touch exactly one vertical border"

    reflected = np.fliplr(g)
    mirrored_half = np.full_like(reflected, Color.TEAL)
    mirrored_half[reflected != bg] = bg

    if touches_left:
        return np.concatenate((mirrored_half, g), axis=1)
    return np.concatenate((g, mirrored_half), axis=1)

# --------------- generator (o5x) ---------------

# Canvas and foreground bounds.
_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT = 3, 12
_GEN_MIN_WIDTH, _GEN_MAX_WIDTH = 3, 6
_GEN_MIN_RED_PIXELS, _GEN_MAX_RED_PIXELS = 4, 18

# Density and row-coverage constraints.
_GEN_MIN_DENSITY, _GEN_MAX_DENSITY = 0.25, 0.56
_GEN_MIN_DENSITY_NUM, _GEN_MIN_DENSITY_DEN = 1, 4
_GEN_MAX_DENSITY_NUM, _GEN_MAX_DENSITY_DEN = 56, 100
_GEN_MIN_ROW_COVERAGE_NUM, _GEN_MIN_ROW_COVERAGE_DEN = 3, 5

# Shape structure and retries.
_GEN_OPPOSITE_BORDER_EMPTY_COLS = 1
_GEN_MAX_SATELLITES = 4
_GEN_MIN_COMPONENTS, _GEN_MAX_COMPONENTS = 1, 5
_GEN_CONNECTIVITY = 4
_GEN_MAX_ATTEMPTS = 300

_GEN_NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_FLIP_CHOICES = (False, True)


def _gen_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gen_red_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == Color.RED))


def _gen_grow_main(grid, main_cells, target_main_count: int, top: int, bottom: int, allowed_cols: int) -> bool:
    """Grow the connected border component inside the covered row band."""
    while len(main_cells) < target_main_count:
        occupied = set(zip(*np.where(grid == Color.RED)))
        candidates: set[tuple[int, int]] = set()
        for r, c in main_cells:
            for dr, dc in _GEN_NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if top <= nr <= bottom and 0 <= nc < allowed_cols and (nr, nc) not in occupied:
                    candidates.add((nr, nc))
        frontier = list(candidates)
        if not frontier:
            return False
        r, c = random.choice(frontier)
        grid[r, c] = Color.RED
        main_cells.add((r, c))
    return True


def _gen_isolated_candidates(grid: np.ndarray, allowed_cols: int) -> list[tuple[int, int]]:
    h, _ = grid.shape
    out = []
    for r in range(h):
        for c in range(allowed_cols):
            if grid[r, c] != Color.BLACK:
                continue
            isolated = True
            for dr, dc in _GEN_NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < allowed_cols and grid[nr, nc] == Color.RED:
                    isolated = False
                    break
            if isolated:
                out.append((r, c))
    return out


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    reflected = np.fliplr(inp)
    mirrored_half = np.full_like(reflected, Color.TEAL)
    mirrored_half[reflected != Color.BLACK] = Color.BLACK
    if np.any(inp[:, 0] == Color.RED):
        return np.concatenate((mirrored_half, inp), axis=1)
    return np.concatenate((inp, mirrored_half), axis=1)


def _gen_feasible_counts(h: int, w: int, allowed_cols: int) -> tuple[int, list[int]]:
    """Return minimum covered rows and all feasible red-pixel totals."""
    area = h * w
    row_min = _gen_ceil_div(_GEN_MIN_ROW_COVERAGE_NUM * h, _GEN_MIN_ROW_COVERAGE_DEN)
    count_lo = max(_GEN_MIN_RED_PIXELS, _gen_ceil_div(_GEN_MIN_DENSITY_NUM * area, _GEN_MIN_DENSITY_DEN), row_min)
    count_hi = min(_GEN_MAX_RED_PIXELS, (_GEN_MAX_DENSITY_NUM * area) // _GEN_MAX_DENSITY_DEN, h * allowed_cols)
    feasible_counts = [
        k
        for k in range(count_lo, count_hi + 1)
        if max(row_min, _gen_ceil_div(k, allowed_cols)) <= min(h, k)
    ]
    return row_min, feasible_counts


def _gen_choose_covered_span(h: int, allowed_cols: int, row_min: int, red_total: int) -> tuple[int, int]:
    covered_lo = max(row_min, _gen_ceil_div(red_total, allowed_cols))
    covered_hi = min(h, red_total)
    covered_rows = random.randint(covered_lo, covered_hi)
    top = random.randint(0, h - covered_rows)
    return top, top + covered_rows - 1


def _gen_seed_main_component(grid: np.ndarray, red_total: int, top: int, bottom: int, allowed_cols: int) -> set[tuple[int, int]]:
    """Create the guaranteed border-touching component before optional flip."""
    main_cells: set[tuple[int, int]] = set()
    covered_rows = bottom - top + 1
    anchor_max = min(allowed_cols - 1, red_total - covered_rows)
    anchor_col = random.randint(0, anchor_max)
    arm_row = random.randint(top, bottom)

    for r in range(top, bottom + 1):
        grid[r, anchor_col] = Color.RED
        main_cells.add((r, anchor_col))
    for c in range(anchor_col + 1):
        grid[arm_row, c] = Color.RED
        main_cells.add((arm_row, c))
    return main_cells


def _gen_add_satellites(grid: np.ndarray, allowed_cols: int, desired_count: int) -> None:
    for _ in range(desired_count):
        candidates = _gen_isolated_candidates(grid, allowed_cols)
        if not candidates:
            break
        r, c = random.choice(candidates)
        grid[r, c] = Color.RED


def _gen_valid_input_constraints(grid: np.ndarray, row_min: int) -> bool:
    """Check the sampled grid against spec-level input constraints."""
    foreground = grid == Color.RED
    red_count = _gen_red_count(grid)
    density = red_count / grid.size

    if not (np.any(foreground[:, 0]) ^ np.any(foreground[:, -1])):
        return False
    if not (_GEN_MIN_RED_PIXELS <= red_count <= _GEN_MAX_RED_PIXELS):
        return False
    if not (_GEN_MIN_DENSITY <= density <= _GEN_MAX_DENSITY):
        return False
    if int(np.count_nonzero(np.any(foreground, axis=1))) < row_min:
        return False
    component_count = len(components_mask(foreground, connectivity=_GEN_CONNECTIVITY))
    return _GEN_MIN_COMPONENTS <= component_count <= _GEN_MAX_COMPONENTS


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT)
    w = random.randint(_GEN_MIN_WIDTH, _GEN_MAX_WIDTH)
    allowed_cols = w - _GEN_OPPOSITE_BORDER_EMPTY_COLS
    row_min, feasible_counts = _gen_feasible_counts(h, w, allowed_cols)
    if not feasible_counts:
        return None

    red_total = random.choice(feasible_counts)
    top, bottom = _gen_choose_covered_span(h, allowed_cols, row_min, red_total)
    grid = np.zeros((h, w), dtype=int)
    main_cells = _gen_seed_main_component(grid, red_total, top, bottom, allowed_cols)

    remaining = red_total - _gen_red_count(grid)
    desired_satellites = random.randint(0, min(_GEN_MAX_SATELLITES, remaining))
    main_target = red_total - desired_satellites
    if not _gen_grow_main(grid, main_cells, main_target, top, bottom, allowed_cols):
        return None

    _gen_add_satellites(grid, allowed_cols, desired_satellites)

    remaining = red_total - _gen_red_count(grid)
    if remaining:
        main_target = len(main_cells) + remaining
        if not _gen_grow_main(grid, main_cells, main_target, top, bottom, allowed_cols):
            return None

    if random.choice(_GEN_FLIP_CHOICES):
        grid = np.fliplr(grid).copy()
    if not _gen_valid_input_constraints(grid, row_min):
        return None

    expected = _gen_expected_output(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
