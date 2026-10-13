from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
BG = 0
WALL = 5
START = 3
SEQ_COLORS = [1, 2, 4, 6, 7, 8, 9]

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (20, 30)
TURN_COUNT_RANGE = (2, 5)
FORWARD_LEN_RANGE = (2, 4)
EXTRA_WALL_COUNT_RANGE = (2, 6)
MAX_LAYOUT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400

TURN_CASES = ("left_longer", "right_longer", "tie_left")


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _step(pos: tuple[int, int], dir_idx: int, dist: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return pos[0] + dr * dist, pos[1] + dc * dist


def _ray(pos: tuple[int, int], dir_idx: int, length: int) -> list[tuple[int, int]]:
    return [_step(pos, dir_idx, dist) for dist in range(1, length + 1)]


def _left(dir_idx: int) -> int:
    return (dir_idx - 1) % 4


def _right(dir_idx: int) -> int:
    return (dir_idx + 1) % 4


def _extract_sequence(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    c = 0
    seq: list[int] = []
    while c < g.shape[1] and int(g[0, c]) != bg:
        seq.append(int(g[0, c]))
        c += 1
    assert seq, "top-left color strip must be non-empty"
    work = g.copy()
    work[0, :c] = bg
    return seq, work


def _find_start_and_wall(work: np.ndarray, bg: int) -> tuple[tuple[int, int], int]:
    vals, cnts = np.unique(work[work != bg], return_counts=True)
    assert len(vals) >= 2, "expected wall pixels and a unique start marker"
    wall = int(vals[np.argmax(cnts)])
    start_pts = np.argwhere((work != bg) & (work != wall))
    assert start_pts.shape[0] == 1, "expected exactly one non-wall marker after removing the sequence"
    sr, sc = map(int, start_pts[0])
    return (sr, sc), wall


def _is_open(work: np.ndarray, wall: int, r: int, c: int) -> bool:
    h, w = work.shape
    return 0 <= r < h and 0 <= c < w and int(work[r, c]) != wall


def _ray_length(work: np.ndarray, wall: int, pos: tuple[int, int], dir_idx: int) -> int:
    length = 0
    r, c = pos
    dr, dc = DIRS[dir_idx]
    nr, nc = r + dr, c + dc
    while _is_open(work, wall, nr, nc):
        length += 1
        nr += dr
        nc += dc
    return length


def _initial_direction(work: np.ndarray, wall: int, start: tuple[int, int]) -> int:
    choices = []
    for dir_idx, (dr, dc) in enumerate(DIRS):
        nr, nc = start[0] + dr, start[1] + dc
        if _is_open(work, wall, nr, nc):
            choices.append(dir_idx)
    assert len(choices) == 1, f"expected exactly one initial travel direction, got {choices}"
    return choices[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the colored step sequence through a wall-constrained path, turning
    toward the side with the longer visible corridor whenever forward motion is blocked.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    seq, work = _extract_sequence(g, bg)
    start, wall = _find_start_and_wall(work, bg)

    out = work.copy()
    r, c = start
    dir_idx = _initial_direction(work, wall, start)

    for i, color in enumerate(seq):
        out[r, c] = int(color)
        if i == len(seq) - 1:
            break

        fr, fc = _step((r, c), dir_idx)
        if not _is_open(work, wall, fr, fc):
            left_dir = _left(dir_idx)
            right_dir = _right(dir_idx)
            left_len = _ray_length(work, wall, (r, c), left_dir)
            right_len = _ray_length(work, wall, (r, c), right_dir)
            assert left_len > 0 or right_len > 0, "blocked path must offer a left or right turn"
            dir_idx = left_dir if left_len >= right_len else right_dir
            fr, fc = _step((r, c), dir_idx)
            assert _is_open(work, wall, fr, fc), "chosen turn direction must be traversable"

        r, c = fr, fc

    return out


def _in_generation_bounds(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 2 <= r < h and 0 <= c < w


def _can_use_open(
    h: int,
    w: int,
    cells: list[tuple[int, int]],
    open_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> bool:
    return all(_in_generation_bounds(h, w, cell) and cell not in open_cells and cell not in wall_cells for cell in cells)


def _can_use_wall(
    h: int,
    w: int,
    cell: tuple[int, int],
    open_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> bool:
    return _in_generation_bounds(h, w, cell) and cell not in open_cells and cell not in wall_cells


def _commit_segment(
    h: int,
    w: int,
    current: tuple[int, int],
    dir_idx: int,
    length: int,
    open_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    path: list[tuple[int, int]],
) -> tuple[int, int] | None:
    cells = _ray(current, dir_idx, length)
    wall = _step(current, dir_idx, length + 1)
    if not _can_use_open(h, w, cells, open_cells, wall_cells):
        return None
    if not _can_use_wall(h, w, wall, open_cells, wall_cells):
        return None
    open_cells.update(cells)
    wall_cells.add(wall)
    path.extend(cells)
    return cells[-1]


def _commit_turn(
    h: int,
    w: int,
    current: tuple[int, int],
    dir_idx: int,
    open_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    path: list[tuple[int, int]],
) -> tuple[tuple[int, int], int, str] | None:
    cases = list(TURN_CASES)
    random.shuffle(cases)
    lengths = list(range(FORWARD_LEN_RANGE[0], FORWARD_LEN_RANGE[1] + 1))

    for case in cases:
        chosen_dir = _left(dir_idx) if case != "right_longer" else _right(dir_idx)
        decoy_dir = _right(dir_idx) if chosen_dir == _left(dir_idx) else _left(dir_idx)

        random.shuffle(lengths)
        for chosen_len in lengths:
            if case == "tie_left":
                decoy_options = [chosen_len]
            else:
                decoy_options = list(range(1, chosen_len))
                random.shuffle(decoy_options)
            for decoy_len in decoy_options:
                chosen_cells = _ray(current, chosen_dir, chosen_len)
                chosen_wall = _step(current, chosen_dir, chosen_len + 1)
                decoy_cells = _ray(current, decoy_dir, decoy_len)
                decoy_wall = _step(current, decoy_dir, decoy_len + 1)

                if not _can_use_open(h, w, chosen_cells, open_cells, wall_cells):
                    continue
                if not _can_use_open(h, w, decoy_cells, open_cells, wall_cells):
                    continue
                if not _can_use_wall(h, w, chosen_wall, open_cells, wall_cells):
                    continue
                if not _can_use_wall(h, w, decoy_wall, open_cells, wall_cells):
                    continue

                open_cells.update(chosen_cells)
                open_cells.update(decoy_cells)
                wall_cells.add(chosen_wall)
                wall_cells.add(decoy_wall)
                path.extend(chosen_cells)
                return chosen_cells[-1], chosen_dir, case
    return None


def _sprinkle_extra_walls(
    h: int,
    w: int,
    open_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> None:
    target = random.randint(*EXTRA_WALL_COUNT_RANGE)
    attempts = 0
    while target > 0 and attempts < 200:
        attempts += 1
        orient = random.choice([0, 1])
        seg_len = random.randint(1, 3)
        if orient == 0:
            r = random.randint(2, h - 1)
            c = random.randint(0, w - seg_len)
            cells = [(r, c + k) for k in range(seg_len)]
        else:
            r = random.randint(2, h - seg_len)
            c = random.randint(0, w - 1)
            cells = [(r + k, c) for k in range(seg_len)]
        if any(cell in open_cells or cell in wall_cells for cell in cells):
            continue
        wall_cells.update(cells)
        target -= 1


def _build_layout(h: int, w: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[str]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        start = (random.randint(3, h - 3), random.randint(2, w - 3))
        dir_idx = random.randrange(4)

        open_cells = {start}
        wall_cells: set[tuple[int, int]] = set()
        path = [start]

        # Force a unique initial direction.
        for block_dir in (_left(dir_idx), _right(dir_idx), (dir_idx + 2) % 4):
            block = _step(start, block_dir)
            if not _can_use_wall(h, w, block, open_cells, wall_cells):
                break
            wall_cells.add(block)
        else:
            initial_lengths = list(range(FORWARD_LEN_RANGE[0], FORWARD_LEN_RANGE[1] + 1))
            random.shuffle(initial_lengths)
            current = None
            for length in initial_lengths:
                current = _commit_segment(h, w, start, dir_idx, length, open_cells, wall_cells, path)
                if current is not None:
                    break
            if current is None:
                continue

            cases_used: list[str] = []
            ok = True
            for _turn in range(random.randint(*TURN_COUNT_RANGE)):
                res = _commit_turn(h, w, current, dir_idx, open_cells, wall_cells, path)
                if res is None:
                    ok = False
                    break
                current, dir_idx, case = res
                cases_used.append(case)
            if not ok:
                continue

            _sprinkle_extra_walls(h, w, open_cells, wall_cells)
            return path, wall_cells, cases_used
    return None


def _make_sequence(length: int) -> list[int]:
    motif_len = random.randint(2, 4)
    motif = random.sample(SEQ_COLORS, motif_len)
    return [motif[i % motif_len] for i in range(length)]


def _render_input(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    seq: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL
    sr, sc = path[0]
    grid[sr, sc] = START
    grid[0, : len(seq)] = np.array(seq, dtype=int)
    return grid


def _expected_output(inp: np.ndarray, path: list[tuple[int, int]], seq: list[int]) -> np.ndarray:
    out = inp.copy()
    out[0, : len(seq)] = BG
    for color, (r, c) in zip(seq, path):
        out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    built = _build_layout(h, w)
    if built is None:
        return None
    path, wall_cells, cases_used = built

    if len(path) > w:
        return None
    if not cases_used or len(set(cases_used)) < min(2, len(cases_used)):
        return None

    seq = _make_sequence(len(path))
    inp = _render_input(h, w, path, wall_cells, seq)
    expected = _expected_output(inp, path, seq)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
