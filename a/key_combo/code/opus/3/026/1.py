from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, randomly_spaced_indices as _arc_randomly_spaced_indices
except Exception:
    _arc_randomly_spaced_indices = None

    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN

GRID_SIDE_RANGE = (13, 19)
SPACING_BORDER = 1
SPACING_PADDING = 1
VISIBLE_BLUE_PROBABILITY = 0.7
EXTRA_ACTIVE_RED_PROBABILITY = 0.55
MAX_GENERATION_ATTEMPTS = 400


SCHEMAS = (
    {
        "reds": (("r_a", "c_a"), ("r_b", "c_b"), ("r_d", "c_d")),
        "blues": (("r_a", "c_b"), ("r_b", "c_c"), ("r_c", "c_a")),
        "extra_red_col": "c_c",
    },
    {
        "reds": (("r_a", "c_a"), ("r_b", "c_c"), ("r_d", "c_d")),
        "blues": (("r_a", "c_b"), ("r_b", "c_a"), ("r_c", "c_c")),
        "extra_red_col": "c_b",
    },
)


def _fallback_spaced_indices(max_len: int, n_indices: int, border_size: int = 1, padding: int = 1) -> np.ndarray:
    max_len = int(max_len)
    n_indices = int(n_indices)
    border_size = int(border_size)
    padding = int(padding)
    if n_indices < 0:
        raise ValueError("n_indices must be non-negative")
    if n_indices == 0:
        return np.zeros(0, dtype=int)
    lo = border_size
    hi = max_len - border_size - 1
    if hi < lo:
        raise ValueError("no feasible placement")

    chosen: list[int] = []
    while len(chosen) < n_indices:
        possible = [
            idx
            for idx in range(lo, hi + 1)
            if all(abs(idx - other) > padding for other in chosen)
        ]
        if not possible:
            raise ValueError("no feasible placement")
        chosen.append(random.choice(possible))
    return np.array(sorted(chosen), dtype=int)


def _spaced_indices(max_len: int, n_indices: int, border_size: int = 1, padding: int = 1) -> np.ndarray:
    if _arc_randomly_spaced_indices is not None:
        return np.asarray(
            _arc_randomly_spaced_indices(
                max_len,
                n_indices,
                border_size=border_size,
                padding=padding,
            ),
            dtype=int,
        )
    return _fallback_spaced_indices(max_len, n_indices, border_size=border_size, padding=padding)


def _extract_seed_lists(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed = {BG, BLUE, RED}
    assert set(map(int, np.unique(g))) <= allowed
    reds = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    blues = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    return reds, blues


def _active_sets(
    red_seeds: list[tuple[int, int]],
    blue_seeds: list[tuple[int, int]],
) -> tuple[set[int], set[int]]:
    red_rows = {r for r, _c in red_seeds}
    blue_cols = {c for _r, c in blue_seeds}
    active_red_cols = {c for _r, c in red_seeds if c in blue_cols}
    active_blue_rows = {r for r, _c in blue_seeds if r in red_rows}
    return active_red_cols, active_blue_rows


def _render_full_output(
    shape: tuple[int, int],
    red_seeds: list[tuple[int, int]],
    blue_seeds: list[tuple[int, int]],
    active_red_cols: set[int],
    active_blue_rows: set[int],
    *,
    keep_inactive_seeds: bool,
    blend_intersections: bool,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    if keep_inactive_seeds:
        for r, c in red_seeds:
            if c not in active_red_cols:
                out[r, c] = RED
        for r, c in blue_seeds:
            if r not in active_blue_rows:
                out[r, c] = BLUE

    if active_red_cols:
        out[:, sorted(active_red_cols)] = RED
    if active_blue_rows:
        out[sorted(active_blue_rows), :] = BLUE
    if blend_intersections and active_red_cols and active_blue_rows:
        out[np.ix_(sorted(active_blue_rows), sorted(active_red_cols))] = GREEN
    return out


def _render_truncated_output(
    shape: tuple[int, int],
    red_seeds: list[tuple[int, int]],
    blue_seeds: list[tuple[int, int]],
    active_red_cols: set[int],
    active_blue_rows: set[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    red_mask = np.zeros(shape, dtype=bool)
    blue_mask = np.zeros(shape, dtype=bool)

    for r, c in red_seeds:
        if c not in active_red_cols:
            out[r, c] = RED
    for r, c in blue_seeds:
        if r not in active_blue_rows:
            out[r, c] = BLUE

    for col in active_red_cols:
        involved_rows = [r for r, c in red_seeds if c == col]
        involved_rows.extend(r for r, c in blue_seeds if c == col)
        red_mask[min(involved_rows) : max(involved_rows) + 1, col] = True

    for row in active_blue_rows:
        involved_cols = [c for r, c in blue_seeds if r == row]
        involved_cols.extend(c for r, c in red_seeds if r == row)
        blue_mask[row, min(involved_cols) : max(involved_cols) + 1] = True

    out[red_mask] = RED
    out[blue_mask] = BLUE
    out[red_mask & blue_mask] = GREEN
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    red_seeds, blue_seeds = _extract_seed_lists(grid)
    active_red_cols, active_blue_rows = _active_sets(red_seeds, blue_seeds)
    return _render_full_output(
        grid.shape,
        red_seeds,
        blue_seeds,
        active_red_cols,
        active_blue_rows,
        keep_inactive_seeds=True,
        blend_intersections=True,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Extend conditionally activated red/blue seed rays into a crossing line grid with green intersections.

    Input structure
    - The grid is a black background containing single-cell blue and red seed pixels.
    - Red seeds indicate potential vertical rays and blue seeds indicate potential horizontal rays.
    - Same-color seeds occupy distinct rows and distinct columns; cross-color seeds may share a row or a column.

    Rule
    - A red seed extends through its entire column only when that column also contains at least one blue seed; otherwise it stays a single red pixel.
    - A blue seed extends through its entire row only when that row also contains at least one red seed; otherwise it stays a single blue pixel.
    - Pixels covered by both an active red column ray and an active blue row ray become green instead of red or blue.
    - All other cells remain black.

    Key insight
    - Color decides direction, but row/column sharing with the opposite color decides whether the ray exists at all; the final green nodes come from combining those independently activated lines.
    """
    g = np.asarray(grid, dtype=int)
    return _solve(g)


def _sample_role_positions(length: int, count: int) -> list[int]:
    positions = list(map(int, _spaced_indices(length, count, border_size=SPACING_BORDER, padding=SPACING_PADDING)))
    random.shuffle(positions)
    return positions


def _place_from_schema(shape: tuple[int, int], schema: dict, *, add_visible_blue: bool, add_extra_active_red: bool) -> np.ndarray:
    h, w = shape
    n_rows = 4 + int(add_visible_blue) + int(add_extra_active_red)
    n_cols = 4 + int(add_visible_blue)
    row_values = _sample_role_positions(h, n_rows)
    col_values = _sample_role_positions(w, n_cols)

    row_roles = {
        "r_a": row_values[0],
        "r_b": row_values[1],
        "r_c": row_values[2],
        "r_d": row_values[3],
    }
    if add_visible_blue:
        row_roles["r_e"] = row_values[4]
    if add_extra_active_red:
        row_roles["r_f"] = row_values[-1]

    col_roles = {
        "c_a": col_values[0],
        "c_b": col_values[1],
        "c_c": col_values[2],
        "c_d": col_values[3],
    }
    if add_visible_blue:
        col_roles["c_e"] = col_values[4]

    grid = np.full((h, w), BG, dtype=int)
    red_seeds = [(row_roles[row_key], col_roles[col_key]) for row_key, col_key in schema["reds"]]
    blue_seeds = [(row_roles[row_key], col_roles[col_key]) for row_key, col_key in schema["blues"]]

    if add_visible_blue:
        blue_seeds.append((row_roles["r_e"], col_roles["c_e"]))
    if add_extra_active_red:
        blue_col = col_roles[schema["extra_red_col"]]
        red_seeds.append((row_roles["r_f"], blue_col))

    for r, c in blue_seeds:
        grid[r, c] = BLUE
    for r, c in red_seeds:
        grid[r, c] = RED
    return grid


def _non_seed_intersections(
    red_seeds: list[tuple[int, int]],
    blue_seeds: list[tuple[int, int]],
    active_red_cols: set[int],
    active_blue_rows: set[int],
) -> set[tuple[int, int]]:
    seed_cells = set(red_seeds) | set(blue_seeds)
    intersections = {(r, c) for r in active_blue_rows for c in active_red_cols}
    return intersections - seed_cells


def _input_constraints_ok(grid: np.ndarray) -> bool:
    red_seeds, blue_seeds = _extract_seed_lists(grid)
    if not (6 <= len(red_seeds) + len(blue_seeds) <= 8):
        return False
    if len({r for r, _c in red_seeds}) != len(red_seeds):
        return False
    if len({c for _r, c in red_seeds}) != len(red_seeds):
        return False
    if len({r for r, _c in blue_seeds}) != len(blue_seeds):
        return False
    if len({c for _r, c in blue_seeds}) != len(blue_seeds):
        return False

    active_red_cols, active_blue_rows = _active_sets(red_seeds, blue_seeds)
    active_reds = [(r, c) for r, c in red_seeds if c in active_red_cols]
    inactive_reds = [(r, c) for r, c in red_seeds if c not in active_red_cols]
    active_blues = [(r, c) for r, c in blue_seeds if r in active_blue_rows]
    inactive_blues = [(r, c) for r, c in blue_seeds if r not in active_blue_rows]

    if len(active_red_cols) < 2 or len(active_blue_rows) < 2:
        return False
    if not inactive_reds or not inactive_blues:
        return False
    if not active_reds or not active_blues:
        return False
    if not _non_seed_intersections(red_seeds, blue_seeds, active_red_cols, active_blue_rows):
        return False
    return True


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    red_seeds, blue_seeds = _extract_seed_lists(grid)
    active_red_cols, active_blue_rows = _active_sets(red_seeds, blue_seeds)

    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if not np.any(expected == GREEN):
        return False

    colored_fraction = float(np.count_nonzero(expected != BG)) / float(expected.size)
    if colored_fraction >= 0.5:
        return False

    extend_all = _render_full_output(
        grid.shape,
        red_seeds,
        blue_seeds,
        {c for _r, c in red_seeds},
        {r for r, _c in blue_seeds},
        keep_inactive_seeds=True,
        blend_intersections=True,
    )
    if np.array_equal(extend_all, expected):
        return False

    no_blend = _render_full_output(
        grid.shape,
        red_seeds,
        blue_seeds,
        active_red_cols,
        active_blue_rows,
        keep_inactive_seeds=True,
        blend_intersections=False,
    )
    if np.array_equal(no_blend, expected):
        return False

    no_singles = _render_full_output(
        grid.shape,
        red_seeds,
        blue_seeds,
        active_red_cols,
        active_blue_rows,
        keep_inactive_seeds=False,
        blend_intersections=True,
    )
    if np.array_equal(no_singles, expected):
        return False

    truncated = _render_truncated_output(grid.shape, red_seeds, blue_seeds, active_red_cols, active_blue_rows)
    if np.array_equal(truncated, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    schema = random.choice(SCHEMAS)
    add_visible_blue = random.random() < VISIBLE_BLUE_PROBABILITY
    add_extra_active_red = random.random() < EXTRA_ACTIVE_RED_PROBABILITY

    try:
        grid = _place_from_schema(
            (h, w),
            schema,
            add_visible_blue=add_visible_blue,
            add_extra_active_red=add_extra_active_red,
        )
    except ValueError:
        return None

    if not _input_constraints_ok(grid):
        return None

    expected = _solve(grid)
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the cross-ray puzzle")
