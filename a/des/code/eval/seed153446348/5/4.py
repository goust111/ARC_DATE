from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT_TURN = {
    (-1, 0): (0, -1),
    (1, 0): (0, 1),
    (0, -1): (1, 0),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {
    (-1, 0): (0, 1),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
    (0, 1): (1, 0),
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _add(p: tuple[int, int], d: tuple[int, int]) -> tuple[int, int]:
    return p[0] + d[0], p[1] + d[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _wall_dirs(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]] | None:
    if int(grid[r, c]) != BLACK:
        return None
    walls = []
    h, w = grid.shape
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == GREY:
            walls.append((dr, dc))
    return walls if len(walls) == 2 else None


def _open_dirs(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]] | None:
    walls = _wall_dirs(grid, r, c)
    if walls is None:
        return None
    return [d for d in DIRS4 if d not in walls]


def _is_corridor_cell(grid: np.ndarray, r: int, c: int) -> bool:
    return _open_dirs(grid, r, c) is not None


def _trace_corridor_from_marker(
    grid: np.ndarray, marker: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    mr, mc = marker
    starts = []
    for dr, dc in DIRS4:
        nr, nc = mr + dr, mc + dc
        if not _in_bounds(h, w, nr, nc):
            continue
        if int(grid[nr, nc]) != BLACK:
            continue
        open_dirs = _open_dirs(grid, nr, nc)
        if open_dirs is None:
            continue
        if (-dr, -dc) in open_dirs:
            starts.append((nr, nc))
    if len(starts) != 1:
        return None

    start = starts[0]
    path = [start]
    visited = {start}
    prev = marker
    cur = start
    while True:
        open_dirs = _open_dirs(grid, cur[0], cur[1])
        if open_dirs is None:
            return None
        next_cells = []
        for dr, dc in open_dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if (nr, nc) == prev:
                continue
            if not _in_bounds(h, w, nr, nc):
                continue
            if int(grid[nr, nc]) != BLACK or not _is_corridor_cell(grid, nr, nc):
                continue
            neighbor_open = _open_dirs(grid, nr, nc)
            if neighbor_open is None or (-dr, -dc) not in neighbor_open:
                continue
            next_cells.append((nr, nc))
        if len(next_cells) != 1:
            break
        nxt = next_cells[0]
        if nxt in visited:
            return None
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
    return path


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = _sub(path[1], path[0])
    for i in range(1, len(path) - 1):
        cur_dir = _sub(path[i + 1], path[i])
        if cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BLACK, GREY):
                continue
            path = _trace_corridor_from_marker(g, (r, c))
            if not path:
                continue
            fill_len = min(len(path), _count_turns(path) + 1)
            for rr, cc in path[:fill_len]:
                out[rr, cc] = color
    return out


def _path_open_dirs(
    path: list[tuple[int, int]],
    idx: int,
    start_out: tuple[int, int],
    end_out: tuple[int, int],
) -> set[tuple[int, int]]:
    if idx == 0:
        return {start_out, _sub(path[1], path[0])}
    if idx == len(path) - 1:
        return {_sub(path[idx - 1], path[idx]), end_out}
    return {_sub(path[idx - 1], path[idx]), _sub(path[idx + 1], path[idx])}


def _build_relative_path(
    rng: random.Random, turn_mode: str
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    for _ in range(200):
        if turn_mode == "straight":
            turns = 0
        elif turn_mode == "turning":
            turns = rng.randint(1, 4)
        else:
            turns = rng.randint(0, 4)

        seg_dirs = [rng.choice(DIRS4)]
        for _ in range(turns):
            turn_fn = rng.choice((LEFT_TURN, RIGHT_TURN))
            seg_dirs.append(turn_fn[seg_dirs[-1]])

        seg_steps = [rng.randint(1, 5) for _ in seg_dirs]
        path = [(0, 0)]
        used = {(0, 0)}
        ok = True
        for direction, steps in zip(seg_dirs, seg_steps):
            for _ in range(steps):
                nxt = _add(path[-1], direction)
                if nxt in used:
                    ok = False
                    break
                for dr, dc in DIRS4:
                    adj = (nxt[0] + dr, nxt[1] + dc)
                    if adj in used and adj != path[-1]:
                        ok = False
                        break
                if not ok:
                    break
                path.append(nxt)
                used.add(nxt)
            if not ok:
                break
        if not ok or len(path) < 2:
            continue

        start_out = (-seg_dirs[0][0], -seg_dirs[0][1])
        end_out = seg_dirs[-1]
        return path, start_out, end_out
    return None


def _build_walls(
    path: list[tuple[int, int]], start_out: tuple[int, int], end_out: tuple[int, int]
) -> set[tuple[int, int]] | None:
    walls: set[tuple[int, int]] = set()
    path_set = set(path)
    for idx, cell in enumerate(path):
        open_dirs = _path_open_dirs(path, idx, start_out, end_out)
        for d in DIRS4:
            if d in open_dirs:
                continue
            wall = _add(cell, d)
            if wall in path_set:
                return None
            walls.add(wall)
    return walls


def _expanded(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _candidate_offsets(lo: int, hi: int, rng: random.Random) -> list[int]:
    vals = list(range(lo, hi + 1))
    rng.shuffle(vals)
    return vals


def _place_corridor(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    rng: random.Random,
    color: int,
    exit_mode: str,
    turn_mode: str,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(400):
        built = _build_relative_path(rng, turn_mode)
        if built is None:
            continue
        path, start_out, end_out = built
        walls = _build_walls(path, start_out, end_out)
        if walls is None:
            continue

        marker = _add(path[0], start_out)
        exit_open = _add(path[-1], end_out)
        occupied = set(path) | walls | {marker}
        if marker in walls or marker in set(path):
            continue
        if exit_mode == "open" and exit_open in occupied:
            continue

        fit_cells = set(occupied)
        if exit_mode == "open":
            fit_cells.add(exit_open)

        rows = [r for r, _ in fit_cells]
        cols = [c for _, c in fit_cells]
        row_lo = -min(rows)
        row_hi = h - 1 - max(rows)
        col_lo = -min(cols)
        col_hi = w - 1 - max(cols)
        if row_lo > row_hi or col_lo > col_hi:
            continue

        fixed_rows = None
        fixed_cols = None
        if exit_mode == "edge":
            end_r, end_c = path[-1]
            if end_out == (-1, 0):
                fixed_rows = [-end_r]
            elif end_out == (1, 0):
                fixed_rows = [h - 1 - end_r]
            elif end_out == (0, -1):
                fixed_cols = [-end_c]
            else:
                fixed_cols = [w - 1 - end_c]

        row_choices = (
            fixed_rows
            if fixed_rows is not None
            else _candidate_offsets(row_lo, row_hi, rng)
        )
        col_choices = (
            fixed_cols
            if fixed_cols is not None
            else _candidate_offsets(col_lo, col_hi, rng)
        )
        reserve_seed = occupied | ({exit_open} if exit_mode == "open" else set())
        reserve = _expanded(reserve_seed)

        for row_off in row_choices:
            if not row_lo <= row_off <= row_hi:
                continue
            for col_off in col_choices:
                if not col_lo <= col_off <= col_hi:
                    continue

                translated_occupied = {(r + row_off, c + col_off) for r, c in occupied}
                translated_reserve = {
                    (r + row_off, c + col_off)
                    for r, c in reserve
                    if _in_bounds(h, w, r + row_off, c + col_off)
                }
                if translated_reserve & blocked:
                    continue

                translated_path = [(r + row_off, c + col_off) for r, c in path]
                translated_walls = {(r + row_off, c + col_off) for r, c in walls}
                translated_marker = (marker[0] + row_off, marker[1] + col_off)
                for wr, wc in translated_walls:
                    grid[wr, wc] = GREY
                grid[translated_marker] = color
                blocked.update(translated_reserve)
                return {
                    "color": color,
                    "marker": translated_marker,
                    "path": translated_path,
                }
    return None


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        color = int(spec["color"])
        path = list(spec["path"])
        fill_len = min(len(path), _count_turns(path) + 1)
        for r, c in path[:fill_len]:
            out[r, c] = color
    return out


def _valid_instance(grid: np.ndarray, specs: list[dict[str, object]]) -> bool:
    markers_seen = 0
    for spec in specs:
        marker = tuple(spec["marker"])
        path = list(spec["path"])
        traced = _trace_corridor_from_marker(grid, marker)
        if traced != path:
            return False
        markers_seen += 1

    actual_markers = int(np.isin(grid, MARKER_COLORS).sum())
    if actual_markers != markers_seen:
        return False

    expected = _expected_output(grid, specs)
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        h = rng.randint(16, 28)
        w = rng.randint(16, 28)
        grid = np.zeros((h, w), dtype=int)
        blocked: set[tuple[int, int]] = set()

        corridor_count = rng.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0]
        colors = list(MARKER_COLORS)
        rng.shuffle(colors)

        if corridor_count == 1:
            exit_modes = [rng.choice(("open", "edge"))]
            turn_modes = [rng.choice(("straight", "turning"))]
        else:
            exit_modes = ["open", "edge"] + [
                rng.choice(("open", "edge")) for _ in range(corridor_count - 2)
            ]
            turn_modes = ["straight", "turning"] + [
                rng.choice(("straight", "turning")) for _ in range(corridor_count - 2)
            ]
            rng.shuffle(exit_modes)
            rng.shuffle(turn_modes)

        specs: list[dict[str, object]] = []
        ok = True
        for idx in range(corridor_count):
            spec = _place_corridor(
                grid=grid,
                blocked=blocked,
                rng=rng,
                color=colors[idx],
                exit_mode=exit_modes[idx],
                turn_mode=turn_modes[idx],
            )
            if spec is None:
                ok = False
                break
            specs.append(spec)

        if ok and _valid_instance(grid, specs):
            return grid
    raise RuntimeError("generate_input: failed to build a valid corridor puzzle")
