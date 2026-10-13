from __future__ import annotations

import numpy as np


CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
BG_COLOR = 0
GRID_COLOR = 5
SEED_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _detect_grid_structure(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    row_threshold = max(3, int(np.ceil(0.6 * w)))
    col_threshold = max(3, int(np.ceil(0.6 * h)))

    best: tuple[tuple[int, int], int, np.ndarray, np.ndarray] | None = None
    for color in np.unique(grid):
        row_ids = np.flatnonzero(np.count_nonzero(grid == color, axis=1) >= row_threshold)
        col_ids = np.flatnonzero(np.count_nonzero(grid == color, axis=0) >= col_threshold)
        if row_ids.size < 2 or col_ids.size < 2:
            continue
        if int(row_ids[0]) != 0 or int(row_ids[-1]) != h - 1:
            continue
        if int(col_ids[0]) != 0 or int(col_ids[-1]) != w - 1:
            continue

        row_steps = np.diff(row_ids)
        col_steps = np.diff(col_ids)
        if row_steps.size == 0 or col_steps.size == 0:
            continue
        if int(row_steps[0]) <= 1 or int(col_steps[0]) <= 1:
            continue
        if not np.all(row_steps == row_steps[0]) or not np.all(col_steps == col_steps[0]):
            continue

        score = (int(row_ids.size + col_ids.size), int(np.count_nonzero(grid == color)))
        if best is None or score > best[0]:
            best = (score, int(color), row_ids.astype(int), col_ids.astype(int))

    assert best is not None, "failed to detect the grey lattice"
    _, line_color, line_rows, line_cols = best
    other = grid[grid != line_color]
    assert other.size > 0, "grid must contain non-line pixels"
    bg = _most_common_color(other)
    return line_color, bg, line_rows, line_cols


def _cell_slice(line_rows: np.ndarray, line_cols: np.ndarray, row: int, col: int) -> tuple[slice, slice]:
    return slice(int(line_rows[row]) + 1, int(line_rows[row + 1])), slice(int(line_cols[col]) + 1, int(line_cols[col + 1]))


def _edge_is_open(
    grid: np.ndarray,
    bg: int,
    line_rows: np.ndarray,
    line_cols: np.ndarray,
    row: int,
    col: int,
    direction: tuple[int, int],
) -> bool:
    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    dr, dc = direction
    if dr == -1:
        if row == 0:
            return False
        segment = grid[int(line_rows[row]), int(line_cols[col]) + 1 : int(line_cols[col + 1])]
    elif dr == 1:
        if row + 1 >= n_rows:
            return False
        segment = grid[int(line_rows[row + 1]), int(line_cols[col]) + 1 : int(line_cols[col + 1])]
    elif dc == -1:
        if col == 0:
            return False
        segment = grid[int(line_rows[row]) + 1 : int(line_rows[row + 1]), int(line_cols[col])]
    else:
        if col + 1 >= n_cols:
            return False
        segment = grid[int(line_rows[row]) + 1 : int(line_rows[row + 1]), int(line_cols[col + 1])]
    return bool(np.any(segment == bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate seed-cell patterns through straight lattice channels opened by border gaps.

    Input structure:
    - Black background cells are separated by a grey rectangular lattice.
    - A few cells contain colored seed patterns.
    - Gaps are black openings on the middle of cell borders.

    Rule:
    - Each seed propagates its whole interior pattern along every gap direction.
    - Propagation continues straight while every next separator on that line also has a gap.
    - When several seeds can fill the same cell, keep the seed with the smaller hop count.
      If hop counts tie, keep the seed whose pattern has more non-background pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_color, bg, line_rows, line_cols = _detect_grid_structure(g)
    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    assert n_rows > 0 and n_cols > 0, "grid must contain at least one cell"

    seeds: list[dict[str, object]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            rs, cs = _cell_slice(line_rows, line_cols, row, col)
            pattern = g[rs, cs].copy()
            filled = (pattern != bg) & (pattern != line_color)
            if not bool(np.any(filled)):
                continue

            directions = [direction for direction in CARDINAL_DIRS if _edge_is_open(g, bg, line_rows, line_cols, row, col, direction)]
            seeds.append(
                {
                    "row": row,
                    "col": col,
                    "pattern": pattern,
                    "count": int(np.count_nonzero(filled)),
                    "directions": directions,
                }
            )

    assert seeds, "expected at least one seed cell"

    winners: dict[tuple[int, int], tuple[tuple[int, int, int], np.ndarray]] = {}

    def offer(cell_row: int, cell_col: int, distance: int, seed_index: int, seed: dict[str, object]) -> None:
        key = (int(distance), -int(seed["count"]), int(seed_index))
        cell = (int(cell_row), int(cell_col))
        current = winners.get(cell)
        if current is None or key < current[0]:
            winners[cell] = (key, np.asarray(seed["pattern"], dtype=int))

    for seed_index, seed in enumerate(seeds):
        row = int(seed["row"])
        col = int(seed["col"])
        offer(row, col, 0, seed_index, seed)
        for direction in seed["directions"]:
            dr, dc = direction
            cur_row, cur_col = row, col
            distance = 0
            while _edge_is_open(g, bg, line_rows, line_cols, cur_row, cur_col, direction):
                cur_row += dr
                cur_col += dc
                distance += 1
                offer(cur_row, cur_col, distance, seed_index, seed)

    out = g.copy()
    for (row, col), (_, pattern) in winners.items():
        rs, cs = _cell_slice(line_rows, line_cols, row, col)
        out[rs, cs] = pattern
    return out


def _blank_lattice(cell_rows: int, cell_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    h = cell_rows * (cell_h + 1) + 1
    w = cell_cols * (cell_w + 1) + 1
    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[:: cell_h + 1, :] = GRID_COLOR
    grid[:, :: cell_w + 1] = GRID_COLOR
    return grid


def _paint_gap(grid: np.ndarray, cell_h: int, cell_w: int, row: int, col: int, direction: tuple[int, int]) -> None:
    top = row * (cell_h + 1)
    left = col * (cell_w + 1)
    center_r = top + 1 + cell_h // 2
    center_c = left + 1 + cell_w // 2
    dr, dc = direction
    if dr == -1:
        grid[top, center_c] = BG_COLOR
    elif dr == 1:
        grid[top + cell_h + 1, center_c] = BG_COLOR
    elif dc == -1:
        grid[center_r, left] = BG_COLOR
    else:
        grid[center_r, left + cell_w + 1] = BG_COLOR


def _add_segment(
    open_right: np.ndarray,
    open_down: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    row, col = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for _ in range(int(length)):
        next_row = row + dr
        next_col = col + dc
        if dr == 0 and dc == 1:
            open_right[row, col] = True
        elif dr == 0 and dc == -1:
            open_right[row, col - 1] = True
        elif dr == 1 and dc == 0:
            open_down[row, col] = True
        else:
            open_down[row - 1, col] = True
        row, col = next_row, next_col
        cells.append((row, col))
    return cells


def _free_run(
    start: tuple[int, int],
    direction: tuple[int, int],
    reserved: set[tuple[int, int]],
    cell_rows: int,
    cell_cols: int,
) -> list[tuple[int, int]]:
    row, col = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while True:
        row += dr
        col += dc
        if not (0 <= row < cell_rows and 0 <= col < cell_cols):
            break
        if (row, col) in reserved:
            break
        cells.append((row, col))
    return cells


def _pick_length(rng: np.random.Generator, max_len: int) -> int:
    if max_len <= 1:
        return max_len
    low = 2 if rng.random() < 0.65 else 1
    return int(rng.integers(low, max_len + 1))


def _random_connected_mask(rng: np.random.Generator, height: int, width: int, count: int) -> np.ndarray:
    start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {start}
    while len(cells) < count:
        frontier: list[tuple[int, int]] = []
        for row, col in cells:
            for dr, dc in CARDINAL_DIRS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)

    mask = np.zeros((height, width), dtype=bool)
    for row, col in cells:
        mask[row, col] = True
    return mask


def _plan_conflict(
    rng: np.random.Generator,
    cell_rows: int,
    cell_cols: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    for _ in range(120):
        cross_row = int(rng.integers(0, cell_rows))
        cross_col = int(rng.integers(0, cell_cols))
        from_left = bool(int(rng.integers(0, 2)))
        from_top = bool(int(rng.integers(0, 2)))

        max_h = cross_col if from_left else (cell_cols - 1 - cross_col)
        max_v = cross_row if from_top else (cell_rows - 1 - cross_row)
        if max_h < 1 or max_v < 1:
            continue

        count_tie = bool(rng.random() < 0.5)
        if count_tie:
            shared = min(max_h, max_v)
            if shared < 1:
                continue
            dist_h = dist_v = int(rng.integers(1, shared + 1))
        else:
            dist_h = int(rng.integers(1, max_h + 1))
            dist_v = int(rng.integers(1, max_v + 1))
            if dist_h == dist_v:
                if dist_v < max_v:
                    dist_v += 1
                elif dist_h < max_h:
                    dist_h += 1
                else:
                    continue

        horiz_seed = (cross_row, cross_col - dist_h) if from_left else (cross_row, cross_col + dist_h)
        horiz_dir = (0, 1) if from_left else (0, -1)
        vert_seed = (cross_row - dist_v, cross_col) if from_top else (cross_row + dist_v, cross_col)
        vert_dir = (1, 0) if from_top else (-1, 0)
        if horiz_seed == vert_seed:
            continue

        extra_h_max = (cell_cols - 1 - cross_col) if horiz_dir == (0, 1) else cross_col
        extra_v_max = (cell_rows - 1 - cross_row) if vert_dir == (1, 0) else cross_row
        extra_h = int(rng.integers(0, extra_h_max + 1))
        extra_v = int(rng.integers(0, extra_v_max + 1))

        return (
            {
                "pos": horiz_seed,
                "direction": horiz_dir,
                "length": dist_h + extra_h,
            },
            {
                "pos": vert_seed,
                "direction": vert_dir,
                "length": dist_v + extra_v,
            },
        )
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_h = int(rng.choice(np.array([3, 5], dtype=int)))
    cell_w = int(rng.choice(np.array([3, 5], dtype=int)))
    cell_rows = int(rng.integers(3, 7 if cell_h == 3 else 6))
    cell_cols = int(rng.integers(3, 7 if cell_w == 3 else 6))
    total_cells = cell_rows * cell_cols
    seed_count = int(rng.integers(2, min(4, total_cells - 1) + 1))

    open_right = np.zeros((cell_rows, cell_cols - 1), dtype=bool)
    open_down = np.zeros((cell_rows - 1, cell_cols), dtype=bool)

    seeds: list[dict[str, object]] = []
    reserved: set[tuple[int, int]] = set()
    seed_dirs: dict[tuple[int, int], list[tuple[int, int]]] = {}

    if seed_count >= 2 and rng.random() < 0.5:
        conflict = _plan_conflict(rng, cell_rows, cell_cols)
        if conflict is None:
            return None
        for spec in conflict:
            pos = tuple(spec["pos"])
            seeds.append({"pos": pos})
            reserved.add(pos)
            seed_dirs[pos] = [tuple(spec["direction"])]
        for spec in conflict:
            path_cells = _add_segment(open_right, open_down, tuple(spec["pos"]), tuple(spec["direction"]), int(spec["length"]))
            reserved.update(path_cells)

    all_cells = [(row, col) for row in range(cell_rows) for col in range(cell_cols)]
    while len(seeds) < seed_count:
        candidates = []
        for cell in all_cells:
            if cell in reserved:
                continue
            if any(_free_run(cell, direction, reserved | {cell}, cell_rows, cell_cols) for direction in CARDINAL_DIRS):
                candidates.append(cell)
        if not candidates:
            return None
        pick = tuple(candidates[int(rng.integers(0, len(candidates)))])
        seeds.append({"pos": pick})
        reserved.add(pick)
        seed_dirs.setdefault(pick, [])

    for seed in seeds:
        pos = tuple(seed["pos"])
        used = list(seed_dirs.get(pos, []))

        if not used:
            options = []
            for direction in CARDINAL_DIRS:
                run = _free_run(pos, direction, reserved, cell_rows, cell_cols)
                if run:
                    options.append((direction, run))
            if not options:
                return None
            direction, run = options[int(rng.integers(0, len(options)))]
            length = _pick_length(rng, len(run))
            path_cells = _add_segment(open_right, open_down, pos, direction, length)
            reserved.update(path_cells)
            used.append(direction)

        if len(used) == 1 and rng.random() < 0.55:
            options = []
            for direction in CARDINAL_DIRS:
                if direction in used:
                    continue
                run = _free_run(pos, direction, reserved, cell_rows, cell_cols)
                if run:
                    options.append((direction, run))
            if options:
                direction, run = options[int(rng.integers(0, len(options)))]
                length = _pick_length(rng, len(run))
                path_cells = _add_segment(open_right, open_down, pos, direction, length)
                reserved.update(path_cells)
                used.append(direction)

        if len(used) == 2 and rng.random() < 0.25:
            options = []
            for direction in CARDINAL_DIRS:
                if direction in used:
                    continue
                run = _free_run(pos, direction, reserved, cell_rows, cell_cols)
                if run:
                    options.append((direction, run))
            if options:
                direction, run = options[int(rng.integers(0, len(options)))]
                length = _pick_length(rng, len(run))
                path_cells = _add_segment(open_right, open_down, pos, direction, length)
                reserved.update(path_cells)
                used.append(direction)

        seed_dirs[pos] = used

    area = cell_h * cell_w
    if seed_count > area - 2:
        return None
    colors = rng.choice(SEED_COLORS, size=seed_count, replace=False)
    counts = rng.choice(np.arange(2, area), size=seed_count, replace=False)

    grid = _blank_lattice(cell_rows, cell_cols, cell_h, cell_w)
    for row in range(cell_rows):
        for col in range(cell_cols - 1):
            if open_right[row, col]:
                _paint_gap(grid, cell_h, cell_w, row, col, (0, 1))
        for col in range(cell_cols):
            if row < cell_rows - 1 and open_down[row, col]:
                _paint_gap(grid, cell_h, cell_w, row, col, (1, 0))

    for index, seed in enumerate(seeds):
        mask = _random_connected_mask(rng, cell_h, cell_w, int(counts[index]))
        pattern = np.zeros((cell_h, cell_w), dtype=int)
        pattern[mask] = int(colors[index])
        row, col = tuple(seed["pos"])
        top = row * (cell_h + 1) + 1
        left = col * (cell_w + 1) + 1
        grid[top : top + cell_h, left : left + cell_w] = pattern

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
