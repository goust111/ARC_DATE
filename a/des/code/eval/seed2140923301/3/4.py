from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = (1, 2, 3, 4, 6, 7, 8)
ROOM_GRID_CHOICES = ((2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2))
ROOM_SIZE_RANGE = (6, 9)
GEN_ATTEMPTS = 400

DIR_TO_DELTA = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
DIR_TO_MISSING = {
    "U": (0, 0),
    "R": (0, 1),
    "D": (1, 1),
    "L": (1, 0),
}
MISSING_TO_DIR = {value: key for key, value in DIR_TO_MISSING.items()}


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        frontier_list = sorted(frontier)
        choice = frontier_list[int(rng.integers(0, len(frontier_list)))]
        cells.add(choice)
    shape = _normalize(cells)
    for _ in range(int(rng.integers(0, 4))):
        h = max(r for r, _ in shape) + 1
        shape = [(c, h - 1 - r) for r, c in shape]
        shape = _normalize(set(shape))
    return shape


def _room_bounds(
    room_r: int,
    room_c: int,
    wall_rows: list[int],
    wall_cols: list[int],
) -> tuple[int, int, int, int, int, int, int, int]:
    wt = wall_rows[room_r]
    wb = wall_rows[room_r + 1]
    wl = wall_cols[room_c]
    wr = wall_cols[room_c + 1]
    return wt, wb, wl, wr, wt + 1, wb - 1, wl + 1, wr - 1


def _find_walls(grid: np.ndarray, axis: int) -> list[int]:
    grey_mask = grid == GREY
    counts = np.sum(grey_mask, axis=1 - axis)
    span = grid.shape[1 - axis]
    threshold = span // 2 + 1
    walls = [int(i) for i, count in enumerate(counts) if int(count) >= threshold]
    assert len(walls) >= 3
    assert walls[0] == 0 and walls[-1] == grid.shape[axis] - 1
    assert all(b - a >= 2 for a, b in zip(walls, walls[1:]))
    return walls


def _arrow_direction(room: np.ndarray) -> str:
    coords = [tuple(map(int, pos)) for pos in np.argwhere(room == MAROON)]
    assert len(coords) == 3
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert (r1 - r0, c1 - c0) == (1, 1)
    present = {(r - r0, c - c0) for r, c in coords}
    missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - present).pop()
    return MISSING_TO_DIR[missing]


def _window_on_side(
    grid: np.ndarray,
    side: str,
    wt: int,
    wb: int,
    wl: int,
    wr: int,
) -> tuple[list[tuple[int, int]], int] | None:
    if side == "U":
        coords = [(wt, c) for c in range(wl + 1, wr)]
        axis_values = [c for _, c in coords]
    elif side == "D":
        coords = [(wb, c) for c in range(wl + 1, wr)]
        axis_values = [c for _, c in coords]
    elif side == "L":
        coords = [(r, wl) for r in range(wt + 1, wb)]
        axis_values = [r for r, _ in coords]
    else:
        coords = [(r, wr) for r in range(wt + 1, wb)]
        axis_values = [r for r, _ in coords]

    cells = [pos for pos in coords if int(grid[pos]) != GREY]
    if not cells:
        return None

    color = int(grid[cells[0]])
    assert color not in (BG, GREY, MAROON)
    assert all(int(grid[pos]) == color for pos in cells)

    used = sorted(axis_values[coords.index(pos)] for pos in cells)
    assert used[-1] - used[0] + 1 == len(used)
    return cells, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the maroon L-arrows, then project each room's interior motif into
    the adjacent room behind its colored window using the window color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_rows = _find_walls(g, axis=0)
    wall_cols = _find_walls(g, axis=1)

    out = g.copy()
    out[g == MAROON] = BG

    for room_r in range(len(wall_rows) - 1):
        for room_c in range(len(wall_cols) - 1):
            wt, wb, wl, wr, r0, r1, c0, c1 = _room_bounds(room_r, room_c, wall_rows, wall_cols)
            room = g[r0 : r1 + 1, c0 : c1 + 1]

            direction = _arrow_direction(room)

            windows = {}
            for side in "URDL":
                found = _window_on_side(g, side, wt, wb, wl, wr)
                if found is not None:
                    windows[side] = found
            assert len(windows) == 1
            assert direction in windows
            _, window_color = windows[direction]

            pattern = [
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if int(g[r, c]) not in (BG, MAROON)
            ]
            assert 2 <= len(pattern) <= 4

            if direction == "U":
                dr = (wt - 1) - max(r for r, _ in pattern)
                dc = 0
            elif direction == "D":
                dr = (wb + 1) - min(r for r, _ in pattern)
                dc = 0
            elif direction == "L":
                dr = 0
                dc = (wl - 1) - max(c for _, c in pattern)
            else:
                dr = 0
                dc = (wr + 1) - min(c for _, c in pattern)

            for r, c in pattern:
                rr = r + dr
                cc = c + dc
                assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
                assert int(g[rr, cc]) != GREY
                out[rr, cc] = window_color

    return out


def _grid_walls(sizes: list[int]) -> list[int]:
    walls = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        walls.append(pos)
    return walls


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _perfect_matching(rows: int, cols: int, rng: np.random.Generator) -> dict[tuple[int, int], tuple[int, int]] | None:
    assert (rows * cols) % 2 == 0
    all_cells = tuple((r, c) for r in range(rows) for c in range(cols))
    adjacency = {cell: _neighbors(cell, rows, cols) for cell in all_cells}

    def backtrack(unmatched: frozenset[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]] | None:
        if not unmatched:
            return {}
        cell = min(unmatched)
        nbrs = [n for n in adjacency[cell] if n in unmatched]
        rng.shuffle(nbrs)
        for nbr in nbrs:
            rest = unmatched - {cell, nbr}
            result = backtrack(frozenset(rest))
            if result is not None:
                result[cell] = nbr
                result[nbr] = cell
                return result
        return None

    return backtrack(frozenset(all_cells))


def _window_segment(
    pair: tuple[tuple[int, int], tuple[int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = sorted(pair)
    if r1 == r2:
        wt, wb, _, _, r0, r1i, _, _ = _room_bounds(r1, c1, wall_rows, wall_cols)
        del wt
        del wb
        wall_c = wall_cols[c1 + 1]
        max_len = min(3, r1i - r0 + 1)
        length = int(rng.integers(1, max_len + 1))
        start = int(rng.integers(r0, r1i - length + 2))
        return [(r, wall_c) for r in range(start, start + length)]

    _, _, wl, wr, _, _, c0, c1i = _room_bounds(r1, c1, wall_rows, wall_cols)
    del wl
    del wr
    wall_r = wall_rows[r1 + 1]
    max_len = min(3, c1i - c0 + 1)
    length = int(rng.integers(1, max_len + 1))
    start = int(rng.integers(c0, c1i - length + 2))
    return [(wall_r, c) for c in range(start, start + length)]


def _place_room_contents(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int, int, int, int, int],
    direction: str,
    window_color: int,
    rng: np.random.Generator,
) -> bool:
    _wt, _wb, _wl, _wr, r0, r1, c0, c1 = bounds
    pattern_color_choices = [color for color in PALETTE if color != window_color]
    pattern_color = int(pattern_color_choices[int(rng.integers(0, len(pattern_color_choices)))])

    shape = _random_shape(int(rng.integers(2, 5)), rng)
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1

    top_min = r0 + 1
    top_max = r1 - shape_h
    left_min = c0 + 1
    left_max = c1 - shape_w
    if top_min > top_max or left_min > left_max:
        return False

    arrow_missing = DIR_TO_MISSING[direction]
    arrow_offsets = [(dr, dc) for dr in (0, 1) for dc in (0, 1) if (dr, dc) != arrow_missing]
    arrow_top_min = r0 + 1
    arrow_top_max = r1 - 2
    arrow_left_min = c0 + 1
    arrow_left_max = c1 - 2
    if arrow_top_min > arrow_top_max or arrow_left_min > arrow_left_max:
        return False

    for _ in range(100):
        top = int(rng.integers(top_min, top_max + 1))
        left = int(rng.integers(left_min, left_max + 1))
        pattern_cells = {(top + dr, left + dc) for dr, dc in shape}

        arrow_top = int(rng.integers(arrow_top_min, arrow_top_max + 1))
        arrow_left = int(rng.integers(arrow_left_min, arrow_left_max + 1))
        arrow_cells = {(arrow_top + dr, arrow_left + dc) for dr, dc in arrow_offsets}

        if pattern_cells & arrow_cells:
            continue

        for r, c in pattern_cells:
            grid[r, c] = pattern_color
        for r, c in arrow_cells:
            grid[r, c] = MAROON
        return True

    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows, cols = ROOM_GRID_CHOICES[int(rng.integers(0, len(ROOM_GRID_CHOICES)))]
    room_heights = [int(rng.integers(*ROOM_SIZE_RANGE)) for _ in range(rows)]
    room_widths = [int(rng.integers(*ROOM_SIZE_RANGE)) for _ in range(cols)]
    wall_rows = _grid_walls(room_heights)
    wall_cols = _grid_walls(room_widths)

    grid = np.zeros((wall_rows[-1] + 1, wall_cols[-1] + 1), dtype=int)
    grid[wall_rows, :] = GREY
    grid[:, wall_cols] = GREY

    partners = _perfect_matching(rows, cols, rng)
    if partners is None:
        return None

    room_info: dict[tuple[int, int], dict[str, object]] = {}
    for room_r in range(rows):
        for room_c in range(cols):
            partner = partners[(room_r, room_c)]
            dr = partner[0] - room_r
            dc = partner[1] - room_c
            if (dr, dc) == (-1, 0):
                direction = "U"
            elif (dr, dc) == (1, 0):
                direction = "D"
            elif (dr, dc) == (0, -1):
                direction = "L"
            else:
                direction = "R"
            room_info[(room_r, room_c)] = {"direction": direction}

    seen_pairs: set[frozenset[tuple[int, int]]] = set()
    for room, info in room_info.items():
        partner = partners[room]
        pair_key = frozenset((room, partner))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        window_color = int(PALETTE[int(rng.integers(0, len(PALETTE)))])
        for r, c in _window_segment((room, partner), wall_rows, wall_cols, rng):
            grid[r, c] = window_color
        room_info[room]["window_color"] = window_color
        room_info[partner]["window_color"] = window_color

    for room_r in range(rows):
        for room_c in range(cols):
            info = room_info[(room_r, room_c)]
            ok = _place_room_contents(
                grid,
                _room_bounds(room_r, room_c, wall_rows, wall_cols),
                str(info["direction"]),
                int(info["window_color"]),
                rng,
            )
            if not ok:
                return None

    grid = np.rot90(grid, int(rng.integers(0, 4))).copy()

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-projection puzzle")
