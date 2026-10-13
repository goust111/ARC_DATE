import itertools

import numpy as np


CONTROL_SIZE = 3
SPRITE_COLORS = (5, 6, 7, 8, 9)
OBSTACLE_COLORS = (1, 2, 3)
CONTROL_CODES = (0, 1, 2, 3, 4)
DIRS = {
    0: (0, 0),   # stay
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (-1, 0),  # up
    4: (0, -1),  # left
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= CONTROL_SIZE and g.shape[1] >= CONTROL_SIZE

    control = g[:CONTROL_SIZE, :CONTROL_SIZE].copy()
    assert set(np.unique(control)).issubset(CONTROL_CODES)

    sprite_mask = np.isin(g, SPRITE_COLORS)
    sprite_coords = np.argwhere(sprite_mask)
    assert sprite_coords.shape[0] == 9

    rows = sorted(int(v) for v in np.unique(sprite_coords[:, 0]))
    cols = sorted(int(v) for v in np.unique(sprite_coords[:, 1]))
    assert len(rows) == 3 and len(cols) == 3

    sprite_map: dict[tuple[int, int], int] = {}
    for r, c in sprite_coords:
        sprite_map[(int(r), int(c))] = int(g[r, c])

    assert len(sprite_map) == 9
    for r in rows:
        for c in cols:
            assert (r, c) in sprite_map

    obstacle_grid = np.where(np.isin(g, OBSTACLE_COLORS), g, 0)
    obstacle_grid[:CONTROL_SIZE, :CONTROL_SIZE] = 0
    return g, control, rows, cols, sprite_map, obstacle_grid


def _slide(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction_code: int,
    obstacle_grid: np.ndarray,
):
    r, c = start
    dr, dc = DIRS[int(direction_code)]
    if dr == 0 and dc == 0:
        return (r, c), None

    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(shape, nr, nc):
            return (r, c), None
        if obstacle_grid[nr, nc] != 0:
            return (r, c), int(obstacle_grid[nr, nc])
        r, c = nr, nc


def _transform_with_records(grid: np.ndarray):
    g, control, rows, cols, sprite_map, obstacle_grid = _extract_scene(grid)

    out = g.copy()
    out[:CONTROL_SIZE, :CONTROL_SIZE] = 0
    for pos in sprite_map:
        out[pos] = 0

    records = []
    used_stops = set()
    for rr, sr in enumerate(rows):
        for cc, sc in enumerate(cols):
            start = (sr, sc)
            original_color = sprite_map[start]
            code = int(control[rr, cc])
            stop, contact_color = _slide(g.shape, start, code, obstacle_grid)
            if stop in used_stops:
                raise AssertionError("sprites must end at distinct cells")
            used_stops.add(stop)
            final_color = original_color if contact_color is None else contact_color
            out[stop] = final_color
            records.append(
                {
                    "cell": (rr, cc),
                    "code": code,
                    "start": start,
                    "stop": stop,
                    "original_color": original_color,
                    "contact_color": contact_color,
                    "final_color": final_color,
                }
            )
    return out, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a 3x3 control matrix to assign sliding directions, then recolor sprites only when they stop against an obstacle bar.

    Input structure:
    - The top-left 3x3 region is a control matrix whose colors are drawn from black, blue, red, green, and yellow.
    - Elsewhere on the black background, nine single-pixel sprites of colors 5-9 occupy a 3x3 lattice in the lower-right area.
    - Short red, blue, or green obstacle bars lie away from the control matrix and may block some sprite lanes.

    Rule:
    - Match each control-matrix cell to the sprite at the same lattice row and column.
    - Interpret the control color as a direction: blue=right, red=down, green=up, yellow=left, black=stay.
    - Slide each sprite independently until the next step would leave the grid or enter an obstacle pixel.
    - If the stop is caused by an obstacle, recolor the sprite to that obstacle's color; otherwise keep the sprite's original color.
    - Erase the entire control matrix to black and leave the obstacle bars unchanged.

    Key insight:
    - The control matrix does not say the final color directly; it only chooses a motion direction, and the path outcome decides whether the sprite keeps its own color or inherits an obstacle color.
    """
    out, _ = _transform_with_records(grid)
    return out


def _build_direction_layout(rng: np.random.Generator):
    horiz_perm = rng.permutation(3)
    vert_choices = [
        np.array(p, dtype=int)
        for p in itertools.permutations(range(3))
        if all(p[r] != int(horiz_perm[r]) for r in range(3))
    ]
    vert_perm = vert_choices[int(rng.integers(len(vert_choices)))]

    control = np.zeros((3, 3), dtype=int)

    horiz_repeat = int(rng.choice([1, 4]))
    horiz_other = 4 if horiz_repeat == 1 else 1
    horiz_dirs = np.array([horiz_repeat, horiz_repeat, horiz_other], dtype=int)
    rng.shuffle(horiz_dirs)

    vert_repeat = int(rng.choice([2, 3]))
    vert_other = 3 if vert_repeat == 2 else 2
    vert_dirs = np.array([vert_repeat, vert_repeat, vert_other], dtype=int)
    rng.shuffle(vert_dirs)

    for r in range(3):
        control[r, int(horiz_perm[r])] = int(horiz_dirs[r])
        control[r, int(vert_perm[r])] = int(vert_dirs[r])

    outcomes: dict[tuple[int, int], str] = {}

    horiz_repeat_rows = [r for r in range(3) if int(horiz_dirs[r]) == horiz_repeat]
    rng.shuffle(horiz_repeat_rows)
    outcomes[(horiz_repeat_rows[0], int(horiz_perm[horiz_repeat_rows[0]]))] = "contact"
    outcomes[(horiz_repeat_rows[1], int(horiz_perm[horiz_repeat_rows[1]]))] = "edge"

    vert_repeat_rows = [r for r in range(3) if int(vert_dirs[r]) == vert_repeat]
    rng.shuffle(vert_repeat_rows)
    outcomes[(vert_repeat_rows[0], int(vert_perm[vert_repeat_rows[0]]))] = "contact"
    outcomes[(vert_repeat_rows[1], int(vert_perm[vert_repeat_rows[1]]))] = "edge"

    for r in range(3):
        for c in range(3):
            if control[r, c] != 0 and (r, c) not in outcomes:
                outcomes[(r, c)] = "contact" if rng.random() < 0.55 else "edge"

    return control, outcomes


def _bar_clear(cells, obstacle_mask: np.ndarray) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    h, w = obstacle_mask.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if obstacle_mask[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and obstacle_mask[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _vertical_bar_options(center_row: int, col: int, height: int):
    if height == 2:
        return [
            [(center_row - 1, col), (center_row, col)],
            [(center_row, col), (center_row + 1, col)],
        ]
    return [[(center_row - 1, col), (center_row, col), (center_row + 1, col)]]


def _horizontal_bar_options(row: int, center_col: int, width: int):
    if width == 2:
        return [
            [(row, center_col - 1), (row, center_col)],
            [(row, center_col), (row, center_col + 1)],
        ]
    return [[(row, center_col - 1), (row, center_col), (row, center_col + 1)]]


def _place_vertical_bar(
    rng: np.random.Generator,
    obstacle_grid: np.ndarray,
    row: int,
    start_col: int,
    direction: int,
    sprite_cols: list[int],
    color: int,
):
    h, w = obstacle_grid.shape
    if direction == 1:
        candidates = [c for c in range(start_col + 2, w) if c not in sprite_cols and (c - 1) not in sprite_cols]
    else:
        candidates = [c for c in range(0, start_col - 1) if c not in sprite_cols and (c + 1) not in sprite_cols]
    rng.shuffle(candidates)

    lengths = [2, 3]
    rng.shuffle(lengths)
    for barrier_col in candidates:
        for length in lengths:
            options = _vertical_bar_options(row, barrier_col, length)
            rng.shuffle(options)
            for cells in options:
                if _bar_clear(cells, obstacle_grid != 0):
                    for rr, cc in cells:
                        obstacle_grid[rr, cc] = color
                    return
    raise AssertionError("failed to place a vertical obstacle bar")


def _place_horizontal_bar(
    rng: np.random.Generator,
    obstacle_grid: np.ndarray,
    start_row: int,
    col: int,
    direction: int,
    sprite_rows: list[int],
    color: int,
):
    h, w = obstacle_grid.shape
    if direction == 2:
        candidates = [r for r in range(start_row + 2, h) if r not in sprite_rows and (r - 1) not in sprite_rows]
    else:
        candidates = [r for r in range(0, start_row - 1) if r not in sprite_rows and (r + 1) not in sprite_rows]
    rng.shuffle(candidates)

    lengths = [2, 3]
    rng.shuffle(lengths)
    for barrier_row in candidates:
        for length in lengths:
            options = _horizontal_bar_options(barrier_row, col, length)
            rng.shuffle(options)
            for cells in options:
                if _bar_clear(cells, obstacle_grid != 0):
                    for rr, cc in cells:
                        obstacle_grid[rr, cc] = color
                    return
    raise AssertionError("failed to place a horizontal obstacle bar")


def _generate_attempt(rng: np.random.Generator):
    h = int(rng.integers(16, 21))
    w = int(rng.integers(16, 21))
    row_gap = int(rng.choice([3, 4]))
    col_gap = int(rng.choice([3, 4]))

    row0_max = h - 1 - 2 * row_gap - 2
    col0_max = w - 1 - 2 * col_gap - 2
    if row0_max < 5 or col0_max < 5:
        return None

    row0 = int(rng.integers(5, row0_max + 1))
    col0 = int(rng.integers(5, col0_max + 1))
    sprite_rows = [row0 + i * row_gap for i in range(3)]
    sprite_cols = [col0 + i * col_gap for i in range(3)]

    control, outcomes = _build_direction_layout(rng)

    contact_cells = [cell for cell, mode in outcomes.items() if mode == "contact"]
    rng.shuffle(contact_cells)
    contact_colors: dict[tuple[int, int], int] = {}
    distinct = list(rng.permutation(np.array(OBSTACLE_COLORS, dtype=int))[:2])
    contact_colors[contact_cells[0]] = int(distinct[0])
    contact_colors[contact_cells[1]] = int(distinct[1])
    for cell in contact_cells[2:]:
        contact_colors[cell] = int(rng.choice(OBSTACLE_COLORS))

    sprite_values = list(SPRITE_COLORS) + list(rng.choice(SPRITE_COLORS, size=4, replace=True))
    rng.shuffle(sprite_values)

    grid = np.zeros((h, w), dtype=int)
    grid[:CONTROL_SIZE, :CONTROL_SIZE] = control

    idx = 0
    for r in range(3):
        for c in range(3):
            grid[sprite_rows[r], sprite_cols[c]] = int(sprite_values[idx])
            idx += 1

    obstacle_grid = np.zeros_like(grid)

    for r in range(3):
        for c in range(3):
            code = int(control[r, c])
            if code == 0 or outcomes.get((r, c)) != "contact":
                continue
            start_row = sprite_rows[r]
            start_col = sprite_cols[c]
            color = contact_colors[(r, c)]
            if code in (1, 4):
                _place_vertical_bar(rng, obstacle_grid, start_row, start_col, code, sprite_cols, color)
            else:
                _place_horizontal_bar(rng, obstacle_grid, start_row, start_col, code, sprite_rows, color)

    mask = obstacle_grid != 0
    if np.any(grid[mask] != 0):
        return None
    grid[mask] = obstacle_grid[mask]
    return grid


def _scene_is_semantically_valid(grid: np.ndarray) -> bool:
    try:
        g, control, rows, cols, sprite_map, obstacle_grid = _extract_scene(grid)
        out, records = _transform_with_records(grid)
    except AssertionError:
        return False

    if any(code not in set(control.flatten()) for code in CONTROL_CODES):
        return False

    if len({rec["stop"] for rec in records}) != 9:
        return False

    if np.array_equal(g, out):
        return False

    moved = [rec for rec in records if rec["start"] != rec["stop"]]
    if not moved:
        return False

    if not any(rec["contact_color"] is None for rec in moved):
        return False
    if not any(rec["contact_color"] is not None for rec in moved):
        return False

    contact_colors = {rec["contact_color"] for rec in records if rec["contact_color"] is not None}
    if len(contact_colors) < 2:
        return False

    outcome_by_dir: dict[int, set[str]] = {}
    for rec in records:
        code = int(rec["code"])
        if code == 0:
            continue
        outcome_by_dir.setdefault(code, set()).add("contact" if rec["contact_color"] is not None else "edge")
    if any(code not in outcome_by_dir for code in (1, 2, 3, 4)):
        return False
    if not any(len(v) > 1 for v in outcome_by_dir.values()):
        return False

    sprite_row_set = set(rows)
    sprite_col_set = set(cols)
    for rec in records:
        code = int(rec["code"])
        if code == 0:
            continue
        dr, dc = DIRS[code]
        rr, cc = rec["start"]
        hits = 0
        while True:
            rr += dr
            cc += dc
            if not _in_bounds(g.shape, rr, cc):
                break
            if obstacle_grid[rr, cc] != 0:
                hits += 1
        if hits > 1:
            return False
        if code in (1, 4) and rec["contact_color"] is not None and rec["stop"][1] in sprite_col_set:
            return False
        if code in (2, 3) and rec["contact_color"] is not None and rec["stop"][0] in sprite_row_set:
            return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is None:
            continue
        if _scene_is_semantically_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid sliding-direction ARC input")
