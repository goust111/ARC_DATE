from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every unobstructed background row or column spanning between opposite colored border cells green.

    Input structure
    - The grid has a black background with red and teal foreground cells on the borders and as blockers.
    - Some full rows or columns have foreground cells at both opposite edges and only background cells between them.

    Rule
    - For each row, if the leftmost and rightmost cells are foreground and every interior cell is background, recolor that interior row green.
    - For each column, if the top and bottom cells are foreground and every interior cell is background, recolor that interior column green.
    - Preserve all original foreground cells and all blocked rows or columns.

    Key insight
    - The green cells mark edge-to-edge clear corridors, not enclosed areas; any original foreground cell inside the span blocks that row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "expected room for row and column interiors"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert not np.any(g == Color.GREEN), "green is reserved for the output corridors"

    out = g.copy()
    row_count = 0
    col_count = 0

    for r in range(height):
        if (
            int(g[r, 0]) != bg
            and int(g[r, width - 1]) != bg
            and np.all(g[r, 1 : width - 1] == bg)
        ):
            out[r, 1 : width - 1] = Color.GREEN
            row_count += 1

    for c in range(width):
        if (
            int(g[0, c]) != bg
            and int(g[height - 1, c]) != bg
            and np.all(g[1 : height - 1, c] == bg)
        ):
            out[1 : height - 1, c] = Color.GREEN
            col_count += 1

    assert row_count > 0, "expected at least one unobstructed row corridor"
    assert col_count > 0, "expected at least one unobstructed column corridor"
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 14
# Border partition sampling.
BORDER_STYLE_COUNT = 3
BORDER_STYLE_HORIZONTAL, BORDER_STYLE_VERTICAL = 0, 1
BORDER_L_CORNER_COUNT, BORDER_L_ARC_HIGH_OFFSET = 4, 3
BINARY_CHOICE_COUNT = 2
# Corridor row/column sampling.
CORRIDOR_MARGIN = 2
ROW_CORRIDOR_MIN, ROW_CORRIDOR_MAX = 1, 6
COL_CORRIDOR_MIN, COL_CORRIDOR_MAX = 1, 3
# Interior foreground density.
INTERIOR_DENSITY_MIN, INTERIOR_DENSITY_MAX = 0.18, 0.27
# Retry limits.
GENERATION_RETRIES = 300
FOREGROUND_COLORS = (Color.RED, Color.TEAL)
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_border_loop(n: int) -> list[tuple[int, int]]:
    coords = []
    coords.extend((0, c) for c in range(n))
    coords.extend((r, n - 1) for r in range(1, n))
    coords.extend((n - 1, c) for c in range(n - 2, -1, -1))
    coords.extend((r, 0) for r in range(n - 2, 0, -1))
    return coords


def _gen_transition_indices(
    n: int, rng: np.random.Generator, loop_index: dict[tuple[int, int], int]
) -> tuple[int, int]:
    style = int(rng.integers(0, BORDER_STYLE_COUNT))

    if style == BORDER_STYLE_HORIZONTAL:
        left_r = int(rng.integers(1, n - 1))
        right_r = int(rng.integers(1, n - 1))
        return loop_index[(left_r, 0)], loop_index[(right_r, n - 1)]

    if style == BORDER_STYLE_VERTICAL:
        top_c = int(rng.integers(1, n - 1))
        bottom_c = int(rng.integers(1, n - 1))
        return loop_index[(0, top_c)], loop_index[(n - 1, bottom_c)]

    corner = int(rng.integers(0, BORDER_L_CORNER_COUNT))
    arc_len = int(rng.integers(n, 2 * n - BORDER_L_ARC_HIGH_OFFSET))
    lo = max(1, arc_len - (n - 2))
    hi = min(n - 2, arc_len - 1)
    d1 = int(rng.integers(lo, hi + 1))
    d2 = arc_len - d1

    if corner == 0:
        return loop_index[(0, d1)], loop_index[(d2, 0)]
    if corner == 1:
        return loop_index[(0, n - 1 - d1)], loop_index[(d2, n - 1)]
    if corner == 2:
        return loop_index[(n - 1, n - 1 - d1)], loop_index[(n - 1 - d2, n - 1)]
    return loop_index[(n - 1, d1)], loop_index[(n - 1 - d2, 0)]


def _gen_paint_border(out: np.ndarray, rng: np.random.Generator) -> None:
    n = out.shape[0]
    loop = _gen_border_loop(n)
    loop_index = {coord: i for i, coord in enumerate(loop)}
    i, j = _gen_transition_indices(n, rng, loop_index)
    red_arc = bool(rng.integers(0, BINARY_CHOICE_COUNT))
    arc_color = Color.RED if red_arc else Color.TEAL
    other_color = Color.TEAL if red_arc else Color.RED

    for k, (r, c) in enumerate(loop):
        in_arc = (k - i) % len(loop) < (j - i) % len(loop)
        out[r, c] = arc_color if in_arc else other_color


def _gen_choose_corridors(
    n: int, rng: np.random.Generator
) -> tuple[set[int], set[int]]:
    corridor_pool = list(range(CORRIDOR_MARGIN, n - CORRIDOR_MARGIN))
    row_hi = min(ROW_CORRIDOR_MAX, len(corridor_pool)) + 1
    col_hi = min(COL_CORRIDOR_MAX, len(corridor_pool)) + 1
    row_count = int(rng.integers(ROW_CORRIDOR_MIN, row_hi))
    col_count = int(rng.integers(COL_CORRIDOR_MIN, col_hi))
    rows = set(map(int, rng.choice(corridor_pool, size=row_count, replace=False)))
    cols = set(map(int, rng.choice(corridor_pool, size=col_count, replace=False)))
    return rows, cols


def _gen_mark_corridors(out: np.ndarray, rows: set[int], cols: set[int]) -> None:
    n = out.shape[0]
    for r in rows:
        out[r, 1 : n - 1] = Color.GREEN
    for c in cols:
        out[1 : n - 1, c] = Color.GREEN


def _gen_place_cell(out: np.ndarray, r: int, c: int, color: int) -> bool:
    if out[r, c] != Color.BLACK:
        return False
    out[r, c] = color
    return True


def _gen_seed_required_blockers(
    out: np.ndarray, rows: set[int], cols: set[int], rng: np.random.Generator
) -> None:
    n = out.shape[0]
    interior_rows = [r for r in range(1, n - 1) if r not in rows]
    interior_cols = [c for c in range(1, n - 1) if c not in cols]

    for r in interior_rows:
        side = int(rng.integers(0, BINARY_CHOICE_COUNT))
        choices = [(r, 1, int(out[r, 0])), (r, n - 2, int(out[r, n - 1]))]
        for rr, cc, color in choices[side:] + choices[:side]:
            if _gen_place_cell(out, rr, cc, color):
                break

    for c in interior_cols:
        side = int(rng.integers(0, BINARY_CHOICE_COUNT))
        choices = [(1, c, int(out[0, c])), (n - 2, c, int(out[n - 1, c]))]
        for rr, cc, color in choices[side:] + choices[:side]:
            if _gen_place_cell(out, rr, cc, color):
                break


def _gen_frontier_cells(out: np.ndarray) -> list[tuple[int, int, int]]:
    n = out.shape[0]
    frontier = []
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if out[r, c] != Color.BLACK:
                continue
            adj_colors = set()
            for dr, dc in NEIGHBORS_4:
                color = out[r + dr, c + dc]
                if color in FOREGROUND_COLORS:
                    adj_colors.add(int(color))
            for color in adj_colors:
                frontier.append((r, c, color))
    return frontier


def _gen_interior_counts(grid: np.ndarray) -> dict[int, int]:
    inner = grid[1:-1, 1:-1]
    return {
        color: int(np.count_nonzero(inner == color)) for color in FOREGROUND_COLORS
    }


def _gen_grow_blockers(out: np.ndarray, target_count: int, rng: np.random.Generator) -> bool:
    current = sum(_gen_interior_counts(out).values())
    for _ in range(out.size):
        if current >= target_count:
            return True
        frontier = _gen_frontier_cells(out)
        if not frontier:
            return False
        r, c, color = frontier[int(rng.integers(0, len(frontier)))]
        out[r, c] = color
        current += 1
    return current >= target_count


def _gen_fill_to_density(out: np.ndarray, rng: np.random.Generator) -> bool:
    area = (out.shape[0] - 2) * (out.shape[1] - 2)
    min_count = int(np.ceil(INTERIOR_DENSITY_MIN * area))
    max_count = int(np.floor(INTERIOR_DENSITY_MAX * area))
    current = sum(_gen_interior_counts(out).values())
    if current > max_count:
        return False
    target = int(rng.integers(max(min_count, current), max_count + 1))
    return _gen_grow_blockers(out, target, rng)


def _gen_has_required_blockers(inp: np.ndarray, rows: set[int], cols: set[int]) -> bool:
    n = inp.shape[0]
    for r in range(1, n - 1):
        has_blocker = np.any(inp[r, 1 : n - 1] != Color.BLACK)
        if (r in rows) == has_blocker:
            return False
    for c in range(1, n - 1):
        has_blocker = np.any(inp[1 : n - 1, c] != Color.BLACK)
        if (c in cols) == has_blocker:
            return False
    return True


def _gen_components_ok(inp: np.ndarray) -> bool:
    comps = find_components(
        inp, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict"
    )
    for color in FOREGROUND_COLORS:
        if sum(1 for comp in comps if int(comp["color"]) == color) != 1:
            return False
    return True


def _gen_input_from_output(
    out: np.ndarray, rows: set[int], cols: set[int]
) -> np.ndarray | None:
    area = (out.shape[0] - 2) * (out.shape[1] - 2)
    counts = _gen_interior_counts(out)
    if counts[Color.RED] == 0 or counts[Color.TEAL] == 0:
        return None

    inp = out.copy()
    inp[inp == Color.GREEN] = Color.BLACK

    density = sum(_gen_interior_counts(inp).values()) / area
    if not (INTERIOR_DENSITY_MIN <= density <= INTERIOR_DENSITY_MAX):
        return None
    if not _gen_has_required_blockers(inp, rows, cols):
        return None
    if not _gen_components_ok(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    out = np.zeros((n, n), dtype=int)

    _gen_paint_border(out, rng)
    rows, cols = _gen_choose_corridors(n, rng)
    _gen_mark_corridors(out, rows, cols)
    _gen_seed_required_blockers(out, rows, cols, rng)

    if not _gen_fill_to_density(out, rng):
        return None
    return _gen_input_from_output(out, rows, cols)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 2bee17df input")
