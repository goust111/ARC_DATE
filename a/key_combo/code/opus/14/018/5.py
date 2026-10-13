from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
CONTROL_COLORS = (1, 3, 4, 7, 8, 9)

GRID_HEIGHT_RANGE = (12, 15)
GRID_WIDTH_RANGE = (12, 16)
CONTROL_COUNT_RANGE = (2, 3)
CONTROL_MIN_GAP = 4
GENERATION_ATTEMPTS = 500

# Every on-ray multi-component contains the anchor cell (0, 0), which is the
# red pixel seen by the ray in its control column.
ON_RAY_MULTI_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, -1), (0, 0)),
    ((0, 0), (1, 0)),
    ((0, -1), (0, 0), (0, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, -1), (0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, -1), (0, 0), (1, -1)),
    ((0, -1), (0, 0), (1, -1), (1, 0)),
    ((0, -1), (0, 0), (0, 1), (1, 0)),
)
HORIZONTAL_ONLY_SHAPES = tuple(
    shape for shape in ON_RAY_MULTI_SHAPES if len({dr for dr, _dc in shape}) == 1
)
GENERAL_MULTI_SHAPES = ON_RAY_MULTI_SHAPES + (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _red_components(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], list[set[tuple[int, int]]]]:
    """Return per-cell component sizes and the explicit 4-connected red components."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: dict[tuple[int, int], int] = {}
    comps: list[set[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != RED:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == RED:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
            size = len(comp)
            for cell in comp:
                sizes[cell] = size
    return sizes, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each top-row control pixel downward, letting singleton red components absorb the ray while larger red components let it pass through.

    Input structure:
    - The background is black.
    - The top row contains 2-3 non-red control pixels at distinct columns.
    - Rows below the controls contain only black background and red 4-connected components.
    - Some red components are isolated single pixels; others are multi-pixel shapes.

    Rule:
    - Process the top-row controls from left to right.
    - For each control, scan straight downward in its own column starting at row 1.
    - Black cells are painted with the control color and the ray continues.
    - When the ray hits red, inspect that red pixel's 4-connected red component.
      If the component has size 1, recolor that pixel to the control color and stop the ray.
      If the component has size greater than 1, leave it red and keep scanning below it.
    - Any non-black, non-red cell blocks the ray immediately.

    Key insight:
    - The stopping rule is not based on the first obstacle alone; the ray must classify the struck red component by 4-connected size, and that size determines whether the obstacle absorbs the ray or is transparent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    assert h >= 2, "expected at least one row below the control row"

    component_sizes, _components = _red_components(g)
    out = g.copy()
    control_cols = [int(c) for c in np.flatnonzero(g[0] != BLACK)]
    assert control_cols, "expected at least one control pixel"

    for c in control_cols:
        color = int(g[0, c])
        assert color not in (BLACK, RED), "control pixels must be non-black and non-red"
        for r in range(1, h):
            cell = int(out[r, c])
            if cell == BLACK:
                out[r, c] = color
                continue
            if cell == RED:
                if component_sizes[(r, c)] == 1:
                    out[r, c] = color
                    break
                continue
            break

    return out


def _choose_control_columns(width: int, count: int) -> list[int] | None:
    cols: list[int] = []
    candidates = list(range(1, width - 1))
    for _ in range(count):
        valid = [c for c in candidates if all(abs(c - chosen) >= CONTROL_MIN_GAP for chosen in cols)]
        if not valid:
            return None
        cols.append(random.choice(valid))
    cols.sort()
    return cols


def _can_place_component(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (1 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False

    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and int(grid[nr, nc]) == RED:
                return False
    return True


def _place_component(grid: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = RED


def _shape_cells(anchor_r: int, anchor_c: int, shape: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    return {(anchor_r + dr, anchor_c + dc) for dr, dc in shape}


def _place_on_ray_component(
    grid: np.ndarray,
    col: int,
    row: int,
    *,
    horizontal_only: bool = False,
) -> tuple[set[tuple[int, int]], bool] | None:
    shape_pool = HORIZONTAL_ONLY_SHAPES if horizontal_only else ON_RAY_MULTI_SHAPES
    order = list(shape_pool)
    random.shuffle(order)
    for shape in order:
        cells = _shape_cells(row, col, shape)
        if _can_place_component(grid, cells):
            _place_component(grid, cells)
            horizontal_hit = len({rr for rr, cc in cells if cc == col}) == 1 and any(cc != col for rr, cc in cells)
            return cells, horizontal_hit
    return None


def _place_general_component(
    grid: np.ndarray,
    *,
    include_singleton: bool,
    forbidden_cols: set[int],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    options: list[tuple[tuple[int, int], ...]] = [((0, 0),)] if include_singleton else []
    options.extend(GENERAL_MULTI_SHAPES)
    random.shuffle(options)

    anchors = [(r, c) for r in range(1, h) for c in range(w) if c not in forbidden_cols]
    random.shuffle(anchors)
    for shape in options:
        for r, c in anchors:
            cells = _shape_cells(r, c, shape)
            if any(cc in forbidden_cols for _rr, cc in cells):
                continue
            if _can_place_component(grid, cells):
                _place_component(grid, cells)
                return cells
    return None


def _solve_stop_on_first_red(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, _w = out.shape
    for c in np.flatnonzero(out[0] != BLACK):
        color = int(out[0, c])
        for r in range(1, h):
            cell = int(out[r, c])
            if cell == BLACK:
                out[r, c] = color
                continue
            if cell == RED:
                out[r, c] = color
            break
    return out


def _solve_column_run_heuristic(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, _w = g.shape
    for c in np.flatnonzero(g[0] != BLACK):
        color = int(g[0, c])
        for r in range(1, h):
            cell = int(out[r, c])
            if cell == BLACK:
                out[r, c] = color
                continue
            if cell == RED:
                vertical_neighbor = (r > 1 and int(g[r - 1, c]) == RED) or (r + 1 < h and int(g[r + 1, c]) == RED)
                if not vertical_neighbor:
                    out[r, c] = color
                    break
                continue
            break
    return out


def _solve_all_red_transparent(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, _w = out.shape
    for c in np.flatnonzero(out[0] != BLACK):
        color = int(out[0, c])
        for r in range(1, h):
            cell = int(out[r, c])
            if cell == BLACK:
                out[r, c] = color
                continue
            if cell == RED:
                continue
            break
    return out


def _control_metadata(grid: np.ndarray) -> tuple[list[int], dict[tuple[int, int], int], list[set[tuple[int, int]]]]:
    control_cols = [int(c) for c in np.flatnonzero(grid[0] != BLACK)]
    sizes, comps = _red_components(grid)
    return control_cols, sizes, comps


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.all(np.isin(g[1:], [BLACK, RED])):
        return False

    control_cols, sizes, comps = _control_metadata(g)
    if not (CONTROL_COUNT_RANGE[0] <= len(control_cols) <= CONTROL_COUNT_RANGE[1]):
        return False

    control_colors = [int(g[0, c]) for c in control_cols]
    if len(control_colors) != len(set(control_colors)):
        return False
    if any(color in (BLACK, RED) for color in control_colors):
        return False

    comp_intersections: list[int] = []
    has_off_ray_singleton = False
    has_horizontal_multi = False
    for comp in comps:
        hit_controls = sum(any(cc == c for _rr, cc in comp) for c in control_cols)
        if hit_controls > 1:
            return False
        comp_intersections.append(hit_controls)
        if len(comp) == 1 and hit_controls == 0:
            has_off_ray_singleton = True
        if len(comp) > 1:
            touched_cols = [c for c in control_cols if any(cc == c for _rr, cc in comp)]
            if touched_cols:
                c = touched_cols[0]
                if sum(cc == c for _rr, cc in comp) == 1 and any(cc != c for _rr, cc in comp):
                    has_horizontal_multi = True

    if not has_off_ray_singleton or not has_horizontal_multi:
        return False

    has_post_stop_red = False
    for c in control_cols:
        red_rows = [r for r in range(1, g.shape[0]) if int(g[r, c]) == RED]
        if not red_rows:
            return False
        first_singleton_row = None
        has_multi_before_singleton = False
        seen_comp: set[frozenset[tuple[int, int]]] = set()
        for r in red_rows:
            comp = next(comp for comp in comps if (r, c) in comp)
            comp_key = frozenset(comp)
            if comp_key in seen_comp:
                continue
            seen_comp.add(comp_key)
            if len(comp) == 1:
                first_singleton_row = r
                break
            has_multi_before_singleton = True
        if first_singleton_row is None or not has_multi_before_singleton:
            return False
        if any(r > first_singleton_row and int(g[r, c]) == RED for r in red_rows):
            has_post_stop_red = True

    if not has_post_stop_red:
        return False

    actual = main(g)
    if np.array_equal(actual, g):
        return False
    if np.array_equal(actual, _solve_stop_on_first_red(g)):
        return False
    if np.array_equal(actual, _solve_column_run_heuristic(g)):
        return False
    if np.array_equal(actual, _solve_all_red_transparent(g)):
        return False

    # The painted columns should end in distinct absorbing rows, preventing the
    # trivial "all rays have the same stop depth" presentation.
    stop_rows = []
    for c in control_cols:
        color = int(g[0, c])
        colored_rows = [r for r in range(1, g.shape[0]) if int(actual[r, c]) == color]
        if not colored_rows:
            return False
        stop_rows.append(max(colored_rows))
    if len(set(stop_rows)) == 1:
        return False

    # Reject visually flat cases where all red components are singletons or all
    # are multi-components.
    comp_sizes = Counter(len(comp) for comp in comps)
    if 1 not in comp_sizes or not any(size > 1 for size in comp_sizes):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    control_count = random.randint(*CONTROL_COUNT_RANGE)
    control_cols = _choose_control_columns(width, control_count)
    if control_cols is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    control_colors = random.sample(CONTROL_COLORS, k=control_count)
    for c, color in zip(control_cols, control_colors):
        grid[0, c] = color

    # Ensure the dataset contains at least one hit where the multi-component is
    # only horizontally connected to the ray cell.
    must_use_horizontal = random.randrange(control_count)
    hidden_post_count = 0

    for idx, c in enumerate(control_cols):
        cursor = random.randint(2, 3)
        pre_multi_count = random.randint(1, 2)
        for multi_idx in range(pre_multi_count):
            force_horizontal = idx == must_use_horizontal and multi_idx == 0
            placed = _place_on_ray_component(grid, c, cursor, horizontal_only=force_horizontal)
            if placed is None:
                return None
            cells, _horizontal_hit = placed
            cursor = max(r for r, _cc in cells) + random.randint(2, 3)

        if cursor >= height:
            return None
        singleton = {(cursor, c)}
        if not _can_place_component(grid, singleton):
            return None
        _place_component(grid, singleton)
        cursor += random.randint(2, 3)

        post_kind = random.choice(("single", "multi", "none"))
        if idx == control_count - 1 and hidden_post_count == 0:
            post_kind = random.choice(("single", "multi"))

        if post_kind != "none" and cursor < height:
            if post_kind == "single":
                cells = {(cursor, c)}
                if not _can_place_component(grid, cells):
                    return None
                _place_component(grid, cells)
                hidden_post_count += 1
            else:
                placed = _place_on_ray_component(grid, c, cursor, horizontal_only=False)
                if placed is None:
                    return None
                hidden_post_count += 1

    if hidden_post_count == 0:
        return None

    forbidden_cols = set(control_cols)
    if _place_general_component(grid, include_singleton=True, forbidden_cols=forbidden_cols) is None:
        return None

    extra_decoys = random.randint(1, 3)
    for _ in range(extra_decoys):
        include_singleton = random.random() < 0.4
        if _place_general_component(grid, include_singleton=include_singleton, forbidden_cols=forbidden_cols) is None:
            return None

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid size-gated ray input")
