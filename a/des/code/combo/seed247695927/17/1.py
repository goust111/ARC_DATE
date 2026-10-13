from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
CONTROL_COLORS = (1, 2, 3, 6, 8, 9)

ROW_RANGE = (12, 30)
COL_RANGE = (12, 30)
OUTPUT_CENTER_MARGIN = 2
FRAME_MARGIN = 1
FRAME_CLEARANCE = 4
DISPLACEMENT_MANHATTAN_RANGE = (2, 10)
PATH_LEN_RANGE = (3, 11)
MAX_PATH_TURNS = 2
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _draw_ray(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> None:
    h, w = grid.shape
    rr, cc = r, c
    while 0 <= rr < h and 0 <= cc < w:
        grid[rr, cc] = color
        rr += dr
        cc += dc


def _find_control_color(grid: np.ndarray) -> int:
    colors = set(int(x) for x in np.unique(grid))
    control_colors = colors - {BLACK, YELLOW, GREY}
    assert len(control_colors) == 1
    control = next(iter(control_colors))
    assert int((grid == control).sum()) == 2
    return control


def _find_frame_center(grid: np.ndarray, control: int) -> Coord:
    h, w = grid.shape
    centers: list[Coord] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != control:
                continue
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            if patch.shape != (3, 3):
                continue
            if int(patch[1, 1]) != control:
                continue
            border_ok = True
            for rr in range(3):
                for cc in range(3):
                    if (rr, cc) == (1, 1):
                        continue
                    if int(patch[rr, cc]) != YELLOW:
                        border_ok = False
                        break
                if not border_ok:
                    break
            if border_ok:
                centers.append((r, c))
    assert len(centers) == 1
    return centers[0]


def _gray_path_connects(grid: np.ndarray, start: Coord, end: Coord) -> bool:
    h, w = grid.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if nxt == end or int(grid[nr, nc]) == GREY:
                seen.add(nxt)
                stack.append(nxt)
    grey_seen = sum(1 for r, c in seen if int(grid[r, c]) == GREY)
    return end in seen and grey_seen == int((grid == GREY).sum())


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3

    control = _find_control_color(g)
    frame_center = _find_frame_center(g, control)
    fr, fc = frame_center

    frame_ring = {
        (r, c)
        for r in range(fr - 1, fr + 2)
        for c in range(fc - 1, fc + 2)
        if (r, c) != frame_center
    }

    yellow_cells = set(map(tuple, np.argwhere(g == YELLOW)))
    endpoint_cells = yellow_cells - frame_ring
    assert len(endpoint_cells) == 1
    endpoint = next(iter(endpoint_cells))

    control_cells = set(map(tuple, np.argwhere(g == control)))
    other_control = next(iter(control_cells - {frame_center}))
    assert _gray_path_connects(g, other_control, endpoint)

    dr = int(endpoint[0] - other_control[0])
    dc = int(endpoint[1] - other_control[1])
    out_r = fr + dr
    out_c = fc + dc
    assert 1 <= out_r < h - 1 and 1 <= out_c < w - 1

    out = np.full_like(g, control)
    out[out_r - 1 : out_r + 2, out_c - 1 : out_c + 2] = YELLOW
    out[out_r, out_c] = control

    for rr, cc, drr, dcc in (
        (out_r - 1, out_c - 1, -1, -1),
        (out_r - 1, out_c + 1, -1, 1),
        (out_r + 1, out_c - 1, 1, -1),
        (out_r + 1, out_c + 1, 1, 1),
    ):
        _draw_ray(out, rr, cc, drr, dcc, YELLOW)
    return out


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _path_from_waypoints(waypoints: list[Coord]) -> list[Coord]:
    r = 0
    c = 0
    path = [(r, c)]
    for nr, nc in waypoints:
        assert r == nr or c == nc
        while r != nr:
            r += _sgn(nr - r)
            path.append((r, c))
        while c != nc:
            c += _sgn(nc - c)
            path.append((r, c))
    return path


def _path_turns(path: list[Coord]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((_sgn(r1 - r0), _sgn(c1 - c0)))
    return sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)


def _candidate_paths(dr: int, dc: int) -> list[list[Coord]]:
    candidates: list[list[Coord]] = []

    if dr == 0:
        candidates.append(_path_from_waypoints([(0, dc)]))
        for off in (-1, 1):
            candidates.append(_path_from_waypoints([(off, 0), (off, dc), (0, dc)]))
    elif dc == 0:
        candidates.append(_path_from_waypoints([(dr, 0)]))
        for off in (-1, 1):
            candidates.append(_path_from_waypoints([(0, off), (dr, off), (dr, 0)]))
    else:
        candidates.append(_path_from_waypoints([(dr, 0), (dr, dc)]))
        candidates.append(_path_from_waypoints([(0, dc), (dr, dc)]))
        sr = _sgn(dr)
        sc = _sgn(dc)
        candidates.append(_path_from_waypoints([(-sr, 0), (-sr, dc), (dr, dc)]))
        candidates.append(_path_from_waypoints([(0, -sc), (dr, -sc), (dr, dc)]))
        candidates.append(_path_from_waypoints([(dr + sr, 0), (dr + sr, dc), (dr, dc)]))
        candidates.append(_path_from_waypoints([(0, dc + sc), (dr, dc + sc), (dr, dc)]))
        if abs(dr) >= 2:
            split = random.randint(1, abs(dr) - 1) * sr
            candidates.append(_path_from_waypoints([(split, 0), (split, dc), (dr, dc)]))
        if abs(dc) >= 2:
            split = random.randint(1, abs(dc) - 1) * sc
            candidates.append(_path_from_waypoints([(0, split), (dr, split), (dr, dc)]))

    valid = []
    for path in candidates:
        if len(set(path)) != len(path):
            continue
        if not (PATH_LEN_RANGE[0] <= len(path) <= PATH_LEN_RANGE[1]):
            continue
        if _path_turns(path) > MAX_PATH_TURNS:
            continue
        if len(path) < 3:
            continue
        valid.append(path)
    unique = []
    seen = set()
    for path in valid:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _draw_frame(grid: np.ndarray, r: int, c: int, control: int) -> None:
    grid[r - 1 : r + 2, c - 1 : c + 2] = YELLOW
    grid[r, c] = control


def _make_output(h: int, w: int, control: int, center: Coord) -> np.ndarray:
    out = np.full((h, w), control, dtype=int)
    r, c = center
    out[r - 1 : r + 2, c - 1 : c + 2] = YELLOW
    out[r, c] = control
    for rr, cc, dr, dc in (
        (r - 1, c - 1, -1, -1),
        (r - 1, c + 1, -1, 1),
        (r + 1, c - 1, 1, -1),
        (r + 1, c + 1, 1, 1),
    ):
        _draw_ray(out, rr, cc, dr, dc, YELLOW)
    return out


def _valid_displacements(h: int, w: int, out_r: int, out_c: int, manhattan: int) -> list[Coord]:
    choices = []
    for dr in range(-manhattan, manhattan + 1):
        dc_abs = manhattan - abs(dr)
        dc_values = [0] if dc_abs == 0 else [-dc_abs, dc_abs]
        for dc in dc_values:
            frame_r = out_r - dr
            frame_c = out_c - dc
            if FRAME_MARGIN <= frame_r < h - FRAME_MARGIN and FRAME_MARGIN <= frame_c < w - FRAME_MARGIN:
                choices.append((dr, dc))
    return choices


def _valid_path_starts(h: int, w: int, path: list[Coord], frame_center: Coord) -> list[Coord]:
    fr, fc = frame_center
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    r_lo, r_hi = -min(rs), h - 1 - max(rs)
    c_lo, c_hi = -min(cs), w - 1 - max(cs)
    starts = []
    for r0 in range(r_lo, r_hi + 1):
        for c0 in range(c_lo, c_hi + 1):
            ok = True
            for pr, pc in path:
                ar = r0 + pr
                ac = c0 + pc
                if max(abs(ar - fr), abs(ac - fc)) < FRAME_CLEARANCE:
                    ok = False
                    break
            if ok:
                starts.append((r0, c0))
    random.shuffle(starts)
    return starts


def _place_path(grid: np.ndarray, start: Coord, path: list[Coord], control: int) -> None:
    abs_path = [(start[0] + dr, start[1] + dc) for dr, dc in path]
    grid[abs_path[0]] = control
    for r, c in abs_path[1:-1]:
        grid[r, c] = GREY
    end_r, end_c = abs_path[-1]
    grid[end_r, end_c] = YELLOW


def _sample_output_scene() -> tuple[int, int, int, Coord, np.ndarray]:
    h = random.randint(*ROW_RANGE)
    w = random.randint(*COL_RANGE)
    control = random.choice(CONTROL_COLORS)
    out_r = random.randint(OUTPUT_CENTER_MARGIN, h - OUTPUT_CENTER_MARGIN - 1)
    out_c = random.randint(OUTPUT_CENTER_MARGIN, w - OUTPUT_CENTER_MARGIN - 1)
    center = (out_r, out_c)
    return h, w, control, center, _make_output(h, w, control, center)


def _gen_attempt() -> np.ndarray | None:
    h, w, control, out_center, expected = _sample_output_scene()
    out_r, out_c = out_center

    manhattan = random.randint(*DISPLACEMENT_MANHATTAN_RANGE)
    displacements = _valid_displacements(h, w, out_r, out_c, manhattan)
    if not displacements:
        return None
    dr, dc = random.choice(displacements)

    path_options = _candidate_paths(dr, dc)
    if not path_options:
        return None
    path = random.choice(path_options)

    frame_center = (out_r - dr, out_c - dc)
    starts = _valid_path_starts(h, w, path, frame_center)
    if not starts:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    _draw_frame(grid, frame_center[0], frame_center[1], control)
    _place_path(grid, starts[0], path, control)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
