import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GATE_PALETTE = [1, 2, 3, 6, 7, 8, 9]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _path_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != YELLOW)


def _path_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _trace_slider(grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int, int]:
    mask = _path_mask(grid)
    sr, sc = start
    options: list[tuple[int, int, int, int]] = []
    for rr, cc in _path_neighbors(mask, sr, sc):
        options.append((rr, cc, rr - sr, cc - sc))
    assert len(options) == 1, "each slider must touch exactly one channel endpoint"

    r, c, dr, dc = options[0]
    color = YELLOW
    if int(grid[r, c]) != GREY:
        color = int(grid[r, c])

    while True:
        rr = r + dr
        cc = c + dc
        if not _in_bounds(grid.shape, rr, cc) or not bool(mask[rr, cc]):
            return int(r), int(c), int(color)
        r, c = rr, cc
        if int(grid[r, c]) != GREY:
            color = int(grid[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each yellow token straight through the channel network, recoloring it at every dye gate and keeping only the last gate color.

    Input structure:
    - The background is black.
    - Grey pixels form 1-cell-wide horizontal and vertical channels, sometimes
      meeting at T-junctions or cross-junctions.
    - Some channel cells are replaced by non-grey, non-black, non-yellow dye
      gates that still belong to the channel.
    - Yellow pixels sit just outside dead-end channel endpoints on the same row
      or column as the endpoint they enter.

    Rule:
    - Each yellow slider enters its unique adjacent channel cell and moves in
      that initial direction.
    - At any junction the slider keeps going straight; it never turns. If the
      straight-ahead cell is not part of the channel, it stops on the current
      channel cell.
    - Passing through a dye gate does not stop the slider; it simply changes the
      slider's color to that gate's color, overwriting any previous color.
    - In the output, remove the starting yellow pixels, restore every dye gate
      cell to grey, and place one pixel of each slider's final color at its
      stopping position.

    Key insight:
    - Colored channel pixels are not blockers but pass-through recolor gates,
      and a slider's final color depends only on the last such gate encountered
      along its straight-line route.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    starts = [(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)]
    assert starts, "expected at least one slider"

    finals = [_trace_slider(g, start) for start in starts]

    out = g.copy()
    out[g == YELLOW] = BG
    gate_mask = _path_mask(g) & (g != GREY)
    out[gate_mask] = GREY

    occupied: dict[tuple[int, int], int] = {}
    for r, c, color in finals:
        prev = occupied.get((r, c))
        assert prev is None or prev == color, "sliders should not end in conflicting colors"
        occupied[(r, c)] = int(color)

    for (r, c), color in occupied.items():
        out[r, c] = int(color)

    return out


GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
MIN_VERTICAL_GAP = 3
SPUR_COUNT_RANGE = (0, 2)
GEN_MAX_ATTEMPTS = 1000


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    if hi - lo + 1 < count:
        return None
    for _ in range(200):
        vals = sorted(random.sample(range(lo, hi + 1), count))
        if all(b - a >= min_gap for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _draw_horizontal(cells: set[tuple[int, int]], row: int, c0: int, c1: int) -> None:
    for c in range(min(c0, c1), max(c0, c1) + 1):
        cells.add((int(row), int(c)))


def _draw_vertical(cells: set[tuple[int, int]], col: int, r0: int, r1: int) -> None:
    for r in range(min(r0, r1), max(r0, r1) + 1):
        cells.add((int(r), int(col)))


def _cell_degree(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    total = 0
    for dr, dc in DIR4:
        if (r + dr, c + dc) in cells:
            total += 1
    return total


def _sample_extra_col(lo: int, hi: int, used: list[int], min_gap: int) -> int | None:
    choices = [c for c in range(lo, hi + 1) if all(abs(c - u) >= min_gap for u in used)]
    return None if not choices else int(random.choice(choices))


def _add_random_spurs(
    cells: set[tuple[int, int]],
    verticals: list[dict[str, int]],
    main_row: int,
    width: int,
) -> None:
    for _ in range(random.randint(*SPUR_COUNT_RANGE)):
        candidates = []
        for v in verticals:
            rows = [r for r in range(v["r0"] + 1, v["r1"]) if abs(r - main_row) >= 2]
            if rows:
                candidates.append((v, rows))
        if not candidates:
            return

        v, rows = random.choice(candidates)
        anchor_r = int(random.choice(rows))
        side = random.choice([-1, 1])

        free = 0
        c = v["col"] + side
        while 0 <= c < width and (anchor_r, c) not in cells:
            free += 1
            c += side
        if free < 2:
            continue

        length = random.randint(2, min(5, free))
        for step in range(1, length + 1):
            cells.add((anchor_r, v["col"] + side * step))


def _expected_horizontal_stop(side: str, row: int, left: int, right: int) -> tuple[int, int]:
    return (row, right) if side == "left" else (row, left)


def _expected_vertical_stop(side: str, col: int, top: int, bottom: int) -> tuple[int, int]:
    return (bottom, col) if side == "top" else (top, col)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    main_row = random.randint(6, H - 7)
    left_end = random.randint(2, 4)
    min_right = left_end + 12
    max_right = W - 3
    if min_right > max_right:
        return None
    right_end = random.randint(min_right, max_right)

    base_cols = _sample_spaced_values(left_end + 3, right_end - 3, 3, MIN_VERTICAL_GAP)
    if base_cols is None:
        return None
    random.shuffle(base_cols)
    cross_col, top_t_col, bottom_t_col = map(int, base_cols)

    cross_top = random.randint(2, main_row - 3)
    cross_bottom = random.randint(main_row + 3, H - 3)
    top_t_top = random.randint(2, main_row - 3)
    bottom_t_bottom = random.randint(main_row + 3, H - 3)

    cells: set[tuple[int, int]] = set()
    _draw_horizontal(cells, main_row, left_end, right_end)
    _draw_vertical(cells, cross_col, cross_top, cross_bottom)
    _draw_vertical(cells, top_t_col, top_t_top, main_row)
    _draw_vertical(cells, bottom_t_col, main_row, bottom_t_bottom)

    verticals: list[dict[str, int]] = [
        {"col": cross_col, "r0": cross_top, "r1": cross_bottom},
        {"col": top_t_col, "r0": top_t_top, "r1": main_row},
        {"col": bottom_t_col, "r0": main_row, "r1": bottom_t_bottom},
    ]

    if random.random() < 0.55:
        extra_col = _sample_extra_col(left_end + 3, right_end - 3, [v["col"] for v in verticals], MIN_VERTICAL_GAP)
        if extra_col is not None:
            role = random.choice(["cross", "top_t", "bottom_t"])
            if role == "cross":
                r0 = random.randint(2, main_row - 3)
                r1 = random.randint(main_row + 3, H - 3)
            elif role == "top_t":
                r0 = random.randint(2, main_row - 3)
                r1 = main_row
            else:
                r0 = main_row
                r1 = random.randint(main_row + 3, H - 3)
            _draw_vertical(cells, extra_col, r0, r1)
            verticals.append({"col": int(extra_col), "r0": int(r0), "r1": int(r1)})

    _add_random_spurs(cells, verticals, main_row, W)

    vertical_cols = {v["col"] for v in verticals}
    horizontal_candidates = [
        (main_row, c)
        for c in range(left_end + 1, right_end)
        if (main_row, c) in cells and c not in vertical_cols and _cell_degree(cells, (main_row, c)) == 2
    ]
    if len(horizontal_candidates) < 3:
        return None

    horizontal_side = random.choice(["left", "right"])
    cross_side = random.choice(["top", "bottom"])
    t_role = random.choice(["top_t", "bottom_t"])

    multi_gate_count = random.choice([2, 3])
    if len(horizontal_candidates) < multi_gate_count:
        return None
    multi_gate_cells = sorted(random.sample(horizontal_candidates, multi_gate_count), key=lambda p: p[1])

    cross_candidates = [
        (r, cross_col)
        for r in range(cross_top + 1, cross_bottom)
        if r != main_row and (r, cross_col) in cells and _cell_degree(cells, (r, cross_col)) == 2
    ]
    if not cross_candidates:
        return None
    single_gate_cell = random.choice(cross_candidates)

    gate_colors = random.sample(GATE_PALETTE, multi_gate_count + 1)
    multi_gate_colors = gate_colors[:multi_gate_count]
    single_gate_color = gate_colors[-1]

    starts = [
        (main_row, left_end - 1) if horizontal_side == "left" else (main_row, right_end + 1),
        (cross_top - 1, cross_col) if cross_side == "top" else (cross_bottom + 1, cross_col),
        (top_t_top - 1, top_t_col) if t_role == "top_t" else (bottom_t_bottom + 1, bottom_t_col),
    ]
    if len(set(starts)) != len(starts):
        return None

    for sr, sc in starts:
        if not _in_bounds((H, W), sr, sc):
            return None
        if (sr, sc) in cells:
            return None

    grid = np.full((H, W), BG, dtype=int)
    for r, c in cells:
        grid[r, c] = GREY
    for (r, c), color in zip(multi_gate_cells, multi_gate_colors):
        grid[r, c] = int(color)
    grid[single_gate_cell] = int(single_gate_color)
    for sr, sc in starts:
        grid[sr, sc] = YELLOW

    try:
        finals = [_trace_slider(grid, start) for start in starts]
    except Exception:
        return None
    final_positions = {(r, c) for r, c, _color in finals}
    if len(final_positions) != len(finals):
        return None

    expected_horizontal_color = multi_gate_colors[-1] if horizontal_side == "left" else multi_gate_colors[0]
    expected_horizontal_pos = _expected_horizontal_stop(horizontal_side, main_row, left_end, right_end)
    expected_cross_pos = _expected_vertical_stop(cross_side, cross_col, cross_top, cross_bottom)
    expected_t_pos = (main_row, top_t_col) if t_role == "top_t" else (main_row, bottom_t_col)

    if finals[0] != (*expected_horizontal_pos, expected_horizontal_color):
        return None
    if finals[1] != (*expected_cross_pos, single_gate_color):
        return None
    if finals[2] != (*expected_t_pos, YELLOW):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if int(out[expected_horizontal_pos]) != int(expected_horizontal_color):
        return None
    if int(out[expected_cross_pos]) != int(single_gate_color):
        return None
    if int(out[expected_t_pos]) != int(YELLOW):
        return None
    if np.count_nonzero(grid == YELLOW) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dye-gate slider scene")
