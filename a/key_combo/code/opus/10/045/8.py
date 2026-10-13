from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
MARKER = 1
OBSTACLE = 2

GRID_RANGE = (14, 16)
TOP_BAND_CHOICES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 600

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _marker_coords(grid: np.ndarray) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]


def _obstacle_coords(grid: np.ndarray) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == OBSTACLE)]


def _base_with_obstacles(grid: np.ndarray) -> np.ndarray:
    base = np.full(grid.shape, BG, dtype=int)
    base[grid == OBSTACLE] = OBSTACLE
    return base


def _protection_mask(shape: tuple[int, int], obstacles: list[Coord]) -> np.ndarray:
    protect = np.zeros(shape, dtype=bool)
    for r, c in obstacles:
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                protect[rr, cc] = True
    return protect


def _group_by_row_col(coords: list[Coord]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        by_row[r].append(c)
        by_col[c].append(r)
    for cols in by_row.values():
        cols.sort()
    for rows in by_col.values():
        rows.sort()
    return dict(by_row), dict(by_col)


def _settle_markers(grid: np.ndarray) -> tuple[np.ndarray, list[Coord], list[str]]:
    h, _w = grid.shape
    working = _base_with_obstacles(grid)
    initial = sorted(_marker_coords(grid), key=lambda rc: (-rc[0], rc[1]))
    settled: list[Coord] = []
    reasons: list[str] = []

    for r, c in initial:
        rr = r
        while rr + 1 < h and working[rr + 1, c] == BG:
            rr += 1
        if rr == h - 1:
            reasons.append("bottom")
        elif working[rr + 1, c] == OBSTACLE:
            reasons.append("obstacle")
        else:
            reasons.append("marker")
        working[rr, c] = MARKER
        settled.append((rr, c))

    settled.sort()
    return working, settled, reasons


def _connect_spans(
    base_grid: np.ndarray,
    marker_coords: list[Coord],
    *,
    use_protection: bool,
) -> np.ndarray:
    out = base_grid.copy()
    marker_set = set(marker_coords)
    obstacles = _obstacle_coords(base_grid)
    protect = _protection_mask(base_grid.shape, obstacles)
    by_row, by_col = _group_by_row_col(marker_coords)

    for r, cols in by_row.items():
        if len(cols) < 2:
            continue
        for c in range(cols[0], cols[-1] + 1):
            if out[r, c] == OBSTACLE:
                continue
            if use_protection and protect[r, c] and (r, c) not in marker_set:
                continue
            out[r, c] = MARKER

    for c, rows in by_col.items():
        if len(rows) < 2:
            continue
        for r in range(rows[0], rows[-1] + 1):
            if out[r, c] == OBSTACLE:
                continue
            if use_protection and protect[r, c] and (r, c) not in marker_set:
                continue
            out[r, c] = MARKER

    return out


def _transform(grid: np.ndarray, *, use_gravity: bool, use_protection: bool) -> np.ndarray:
    if use_gravity:
        base, settled, _ = _settle_markers(grid)
        return _connect_spans(base, settled, use_protection=use_protection)

    base = _base_with_obstacles(grid)
    markers = _marker_coords(grid)
    for r, c in markers:
        base[r, c] = MARKER
    return _connect_spans(base, sorted(markers), use_protection=use_protection)


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the blue markers, then connect their settled row/column spans while skipping obstacle protection cells.

    Input structure
    - The grid uses black background, isolated blue marker pixels near the top,
      and isolated red obstacle pixels in the lower half.
    - Multiple blue markers may share a column, so falling order matters.
    - Red pixels are fixed barriers; their four orthogonal neighbors form an
      invisible protection zone for later line drawing.

    Rule
    - Process blue markers from lower initial row to higher initial row.
    - For each marker, clear it from its start position and let it fall straight
      downward until the next cell would be outside the grid, a red obstacle, or
      an already settled non-background pixel; it stops on the current cell.
    - After all markers settle, connect every row span and every column span
      defined by the settled blue markers, but do not newly paint red cells or
      protected cells. Settled markers themselves remain blue.

    Key insight
    - The same red pixels affect the puzzle twice: first as gravity blockers that
      change the settled marker layout, then as nearby protection cells that cut
      the row/column span fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.size > 0
    return _transform(g, use_gravity=True, use_protection=True)


def _span_skip_cells(marker_coords: list[Coord], shape: tuple[int, int], obstacles: list[Coord]) -> set[Coord]:
    by_row, by_col = _group_by_row_col(marker_coords)
    protect = _protection_mask(shape, obstacles)
    marker_set = set(marker_coords)
    skipped: set[Coord] = set()

    for r, cols in by_row.items():
        if len(cols) < 2:
            continue
        for c in range(cols[0], cols[-1] + 1):
            if protect[r, c] and (r, c) not in marker_set and (r, c) not in obstacles:
                skipped.add((r, c))

    for c, rows in by_col.items():
        if len(rows) < 2:
            continue
        for r in range(rows[0], rows[-1] + 1):
            if protect[r, c] and (r, c) not in marker_set and (r, c) not in obstacles:
                skipped.add((r, c))

    return skipped


def _well_spaced_columns(width: int, need: int) -> list[int] | None:
    pool = list(range(1, width - 1))
    for _ in range(120):
        cols = sorted(random.sample(pool, need))
        if min(b - a for a, b in zip(cols, cols[1:])) >= 2 and cols[2] - cols[0] >= 4:
            return cols
    return None


def _pick_free_obstacle(shape: tuple[int, int], forbidden: set[Coord], forbidden_cols: set[int]) -> Coord | None:
    h, w = shape
    candidates: list[Coord] = []
    for r in range(h // 2, h - 1):
        for c in range(1, w - 1):
            cell = (r, c)
            if c in forbidden_cols or cell in forbidden:
                continue
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in forbidden):
                continue
            candidates.append(cell)
    return random.choice(candidates) if candidates else None


def _build_candidate_input() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    top_band = random.choice(TOP_BAND_CHOICES)
    stack_count = 3 if top_band == 5 and random.random() < 0.45 else 2

    extra_singletons = random.randint(0, 2)
    total_markers = stack_count + 2 + extra_singletons  # stack + left + obstacle-stop
    if not 4 <= total_markers <= 6:
        return None

    distinct_cols_needed = 4 + extra_singletons
    cols = _well_spaced_columns(w, distinct_cols_needed)
    if cols is None:
        return None

    left_col = cols[0]
    cut_col = cols[1]
    stack_col = cols[2]
    stop_col = cols[3]
    extra_cols = cols[4:]

    grid = np.full((h, w), BG, dtype=int)

    cut_obstacle = (h - 2, cut_col)
    stop_row = random.randint(max(h // 2, 6), h - 4)
    stop_obstacle = (stop_row, stop_col)
    if abs(cut_obstacle[0] - stop_obstacle[0]) + abs(cut_obstacle[1] - stop_obstacle[1]) == 1:
        return None

    obstacles = [cut_obstacle, stop_obstacle]
    if random.random() < 0.5:
        forbidden = set(obstacles)
        extra = _pick_free_obstacle((h, w), forbidden, {left_col, cut_col, stack_col, stop_col, *extra_cols})
        if extra is None:
            return None
        obstacles.append(extra)

    for r, c in obstacles:
        grid[r, c] = OBSTACLE

    stack_rows = [2 * i for i in range(stack_count)]
    if stack_rows[-1] >= top_band:
        return None

    used_positions = set()
    for r in stack_rows:
        used_positions.add((r, stack_col))

    marker_positions: list[Coord] = [(r, stack_col) for r in stack_rows]

    singleton_cols = [left_col, stop_col, *extra_cols[:extra_singletons]]

    candidate_rows = list(range(top_band))
    for col in singleton_cols:
        choices = [r for r in candidate_rows if (r, col) not in used_positions]
        if not choices:
            return None
        row = random.choice(choices)
        marker_positions.append((row, col))
        used_positions.add((row, col))

    for r, c in marker_positions:
        grid[r, c] = MARKER

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    markers = _marker_coords(g)
    obstacles = _obstacle_coords(g)
    if not (4 <= len(markers) <= 6 and 2 <= len(obstacles) <= 3):
        return False
    if any(r > 4 for r, _ in markers):
        return False
    if any(r < g.shape[0] // 2 or r >= g.shape[0] - 1 for r, _ in obstacles):
        return False
    if any(abs(r0 - r1) + abs(c0 - c1) == 1 for i, (r0, c0) in enumerate(obstacles) for r1, c1 in obstacles[i + 1:]):
        return False

    settled_grid, settled, reasons = _settle_markers(g)
    by_row, by_col = _group_by_row_col(settled)
    if not any(len(cols) >= 2 for cols in by_row.values()):
        return False
    if not any(len(rows) >= 2 for rows in by_col.values()):
        return False
    if not {"bottom", "obstacle", "marker"}.issubset(set(reasons)):
        return False

    out = main(g)
    no_gravity = _transform(g, use_gravity=False, use_protection=True)
    no_protection = _transform(g, use_gravity=True, use_protection=False)

    if np.array_equal(out, settled_grid):
        return False
    if np.array_equal(out, no_gravity):
        return False
    if np.array_equal(out, no_protection):
        return False

    skips = _span_skip_cells(settled, g.shape, obstacles)
    if not skips:
        return False

    expected = _connect_spans(settled_grid, settled, use_protection=True)
    if not np.array_equal(out, expected):
        return False

    added_blue = int(np.count_nonzero(out == MARKER) - np.count_nonzero(settled_grid == MARKER))
    if added_blue <= 0:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate_input()
        if grid is not None and _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
