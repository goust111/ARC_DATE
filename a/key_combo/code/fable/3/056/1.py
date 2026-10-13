from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
GREEN = 3
GREY = 5
DIR_CW = 1
DIR_CCW = -1
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _rotate_point(rel: tuple[int, int], direction: int) -> tuple[int, int]:
    dr, dc = rel
    if direction == DIR_CW:
        return dc, -dr
    if direction == DIR_CCW:
        return -dc, dr
    raise ValueError(f"unexpected direction: {direction}")


def _rotate_set(cells: set[tuple[int, int]], direction: int) -> set[tuple[int, int]]:
    return {_rotate_point(cell, direction) for cell in cells}


def _rotate_turns(cells: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range(turns % 4):
        out = _rotate_set(out, DIR_CW)
    return out


def _add_rel(
    pivot: tuple[int, int], rel_cells: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    pr, pc = pivot
    return {(pr + dr, pc + dc) for dr, dc in rel_cells}


def _connected_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _canonical_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _all_foreground(grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in np.argwhere(grid != bg)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance each pivoted sprite by one more quarter-turn using the green ghost and the red current state.

    Input structure:
    - The grid has one background color and several separated motion components.
    - Each component contains one grey pivot pixel, one red connected sprite, and one green connected sprite.
    - The green sprite is the same shape as the red sprite after a 90-degree rotation around that pivot.

    Rule:
    - For each pivot, find the red/green pair for which rotating the green sprite 90 degrees clockwise or
      counterclockwise lands exactly on the red sprite.
    - That successful quarter-turn direction is the component's motion direction.
    - Clear the green cells to background, recolor the red cells green, and draw the same sprite in red after
      one more 90-degree step in the same direction.

    Key insight:
    - The green sprite is the immediately previous frame rather than decoration: green-to-red reveals each
      component's private angular velocity, and the output is the next frame while keeping only the latest trace.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    pivots = [tuple(map(int, coord)) for coord in np.argwhere(g == GREY)]
    red_components = _connected_components(g == RED)
    green_components = _connected_components(g == GREEN)
    assert pivots, "expected at least one pivot"
    assert len(pivots) == len(red_components) == len(green_components), "expected one red and one green sprite per pivot"

    pivot_set = set(pivots)
    all_fg = _all_foreground(g, bg)
    candidate_lists: list[list[dict[str, object]]] = []

    for pivot in pivots:
        pr, pc = pivot
        candidates: list[dict[str, object]] = []
        for red_i, red_cells in enumerate(red_components):
            red_rel = {(r - pr, c - pc) for r, c in red_cells}
            if (0, 0) in red_rel:
                continue
            for green_i, green_cells in enumerate(green_components):
                if len(red_cells) != len(green_cells):
                    continue
                green_rel = {(r - pr, c - pc) for r, c in green_cells}
                if (0, 0) in green_rel:
                    continue

                matches: list[dict[str, object]] = []
                for direction in (DIR_CW, DIR_CCW):
                    if _rotate_set(green_rel, direction) != red_rel:
                        continue
                    future_rel = _rotate_set(red_rel, direction)
                    future_abs = _add_rel(pivot, future_rel)
                    if future_abs & red_cells:
                        continue
                    if future_abs & green_cells:
                        continue
                    if future_abs & pivot_set:
                        continue
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in future_abs):
                        continue
                    if future_abs & (all_fg - red_cells - green_cells - {pivot}):
                        continue
                    matches.append(
                        {
                            "red_i": red_i,
                            "green_i": green_i,
                            "direction": direction,
                            "future_abs": future_abs,
                        }
                    )
                if len(matches) == 1:
                    candidates.append(matches[0])
        assert candidates, f"could not match pivot at {pivot}"
        candidate_lists.append(candidates)

    order = sorted(range(len(pivots)), key=lambda idx: len(candidate_lists[idx]))
    assignments: list[dict[str, object] | None] = [None] * len(pivots)
    used_red: set[int] = set()
    used_green: set[int] = set()
    used_future: set[tuple[int, int]] = set()

    def search(depth: int) -> bool:
        if depth == len(order):
            return True
        pivot_i = order[depth]
        for candidate in candidate_lists[pivot_i]:
            red_i = int(candidate["red_i"])
            green_i = int(candidate["green_i"])
            future_abs = set(candidate["future_abs"])
            if red_i in used_red or green_i in used_green:
                continue
            if future_abs & used_future:
                continue

            assignments[pivot_i] = candidate
            used_red.add(red_i)
            used_green.add(green_i)
            used_future.update(future_abs)
            if search(depth + 1):
                return True
            assignments[pivot_i] = None
            used_red.remove(red_i)
            used_green.remove(green_i)
            used_future.difference_update(future_abs)
        return False

    assert search(0), "failed to find a consistent pivot/component assignment"
    assert all(assignment is not None for assignment in assignments)

    out = g.copy()
    out[g == GREEN] = bg
    out[g == RED] = GREEN
    for assignment in assignments:
        for r, c in assignment["future_abs"]:
            out[r, c] = RED
    return out


GEN_RETRIES = 400
GEN_SHAPE_RETRIES = 120
GRID_ROWS = (18, 30)
GRID_COLS = (18, 30)
COMPONENT_COUNT_CHOICES = (1, 2, 3, 4)
COMPONENT_COUNT_WEIGHTS = (1, 3, 3, 2)
BG_COLORS = [0, 1, 4, 6, 7, 8, 9]


def _neighbors4(cell: tuple[int, int], height: int, width: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _random_connected_shape(height: int, width: int, n_cells: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < n_cells:
        frontier = {
            nxt
            for cell in cells
            for nxt in _neighbors4(cell, height, width)
            if nxt not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(sorted(frontier)))
    return cells


def _expand_cells(
    cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1
) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _make_component_shape(used_shapes: set[tuple[tuple[int, int], ...]]) -> set[tuple[int, int]] | None:
    for _ in range(GEN_SHAPE_RETRIES):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        n_cells = random.randint(3, min(6, height * width))
        local = _random_connected_shape(height, width, n_cells)
        if local is None:
            continue

        if random.random() < 0.5:
            local = {(height - 1 - r, c) for r, c in local}
        if random.random() < 0.5:
            local = {(r, width - 1 - c) for r, c in local}

        signature = _canonical_shape(local)
        if signature in used_shapes:
            continue

        shift_r = random.randint(2, 4)
        shift_c = random.randint(2, 4)
        rel_cells = {(shift_r + r, shift_c + c) for r, c in local}
        rel_cells = _rotate_turns(rel_cells, random.randrange(4))
        used_shapes.add(signature)
        return rel_cells
    return None


def _direction_plan(n_components: int) -> list[int]:
    if n_components == 1:
        return [random.choice((DIR_CW, DIR_CCW))]
    directions = [DIR_CW, DIR_CCW]
    while len(directions) < n_components:
        directions.append(random.choice((DIR_CW, DIR_CCW)))
    random.shuffle(directions)
    return directions


def _build_expected(
    grid: np.ndarray,
    bg: int,
    placed: list[dict[str, set[tuple[int, int]] | tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    out[out == GREEN] = bg
    out[out == RED] = GREEN
    for item in placed:
        for r, c in item["future"]:
            out[r, c] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    bg = random.choice(BG_COLORS)
    n_components = random.choices(COMPONENT_COUNT_CHOICES, weights=COMPONENT_COUNT_WEIGHTS, k=1)[0]
    directions = _direction_plan(n_components)

    grid = np.full((h, w), bg, dtype=int)
    blocked: set[tuple[int, int]] = set()
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    placed: list[dict[str, set[tuple[int, int]] | tuple[int, int]]] = []

    for direction in directions:
        red_rel = _make_component_shape(used_shapes)
        if red_rel is None:
            return None

        green_rel = _rotate_set(red_rel, DIR_CCW if direction == DIR_CW else DIR_CW)
        future_rel = _rotate_set(red_rel, direction)
        footprint_rel = set(red_rel) | set(green_rel) | set(future_rel) | {(0, 0)}

        min_dr = min(dr for dr, _ in footprint_rel)
        max_dr = max(dr for dr, _ in footprint_rel)
        min_dc = min(dc for _, dc in footprint_rel)
        max_dc = max(dc for _, dc in footprint_rel)
        row_lo = -min_dr
        row_hi = h - 1 - max_dr
        col_lo = -min_dc
        col_hi = w - 1 - max_dc
        if row_lo > row_hi or col_lo > col_hi:
            return None

        positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(positions)

        chosen: tuple[int, int] | None = None
        chosen_cells: dict[str, set[tuple[int, int]] | tuple[int, int]] | None = None
        for pivot in positions:
            red_abs = _add_rel(pivot, red_rel)
            green_abs = _add_rel(pivot, green_rel)
            future_abs = _add_rel(pivot, future_rel)
            footprint_abs = red_abs | green_abs | future_abs | {pivot}
            if footprint_abs & blocked:
                continue
            if red_abs & green_abs or red_abs & future_abs or green_abs & future_abs:
                continue
            chosen = pivot
            chosen_cells = {
                "pivot": pivot,
                "red": red_abs,
                "green": green_abs,
                "future": future_abs,
            }
            break

        if chosen is None or chosen_cells is None:
            return None

        pr, pc = chosen
        grid[pr, pc] = GREY
        for r, c in chosen_cells["red"]:
            grid[r, c] = RED
        for r, c in chosen_cells["green"]:
            grid[r, c] = GREEN

        blocked |= _expand_cells(
            chosen_cells["red"] | chosen_cells["green"] | chosen_cells["future"] | {chosen},
            grid.shape,
            radius=1,
        )
        placed.append(chosen_cells)

    expected = _build_expected(grid, bg, placed)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-clock input")
