from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5

FRAME_SIZE = 5
FRAME_CENTER_OFFSET = 2
INNER_SLICE = slice(1, 4)
OPENINGS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_OFFSETS = {
    "up": (0, 2),
    "down": (4, 2),
    "left": (2, 0),
    "right": (2, 4),
}
GAP_ENDPOINTS = {
    "up": ((0, 1), (0, 3)),
    "down": ((4, 1), (4, 3)),
    "left": ((1, 0), (3, 0)),
    "right": ((1, 4), (3, 4)),
}
DIAG_DIRECTIONS = {
    "up": ((-1, -1), (-1, 1)),
    "down": ((1, -1), (1, 1)),
    "left": ((-1, -1), (1, -1)),
    "right": ((-1, 1), (1, 1)),
}
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _frame_bbox(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    return r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1


def _bbox_overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _coord_in_expanded_bbox(coord: tuple[int, int], bbox: tuple[int, int, int, int], margin: int) -> bool:
    r, c = coord
    r0, c0, r1, c1 = bbox
    return r0 - margin <= r <= r1 + margin and c0 - margin <= c <= c1 + margin


def _outer_ring_coords(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    coords = []
    for r in range(r0, r0 + FRAME_SIZE):
        for c in range(c0, c0 + FRAME_SIZE):
            if r in (r0, r0 + FRAME_SIZE - 1) or c in (c0, c0 + FRAME_SIZE - 1):
                coords.append((r, c))
    return coords


def _frame_border_coords(top_left: tuple[int, int], opening: str) -> set[tuple[int, int]]:
    gap_r, gap_c = _gap_coord(top_left, opening)
    return {coord for coord in _outer_ring_coords(top_left) if coord != (gap_r, gap_c)}


def _gap_coord(top_left: tuple[int, int], opening: str) -> tuple[int, int]:
    r0, c0 = top_left
    dr, dc = GAP_OFFSETS[opening]
    return r0 + dr, c0 + dc


def _gap_endpoints(top_left: tuple[int, int], opening: str) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr, dc in GAP_ENDPOINTS[opening]]


def _frame_center(top_left: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = top_left
    return r0 + FRAME_CENTER_OFFSET, c0 + FRAME_CENTER_OFFSET


def _detect_frame(grid: np.ndarray) -> tuple[tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[tuple[int, int], str]] = []
    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            center = (r0 + FRAME_CENTER_OFFSET, c0 + FRAME_CENTER_OFFSET)
            if int(g[center]) != GREEN:
                continue
            for opening in OPENINGS:
                ok = True
                border = _frame_border_coords((r0, c0), opening)
                gap = _gap_coord((r0, c0), opening)
                for rr, cc in _outer_ring_coords((r0, c0)):
                    if (rr, cc) == gap:
                        if int(g[rr, cc]) == YELLOW:
                            ok = False
                            break
                    elif int(g[rr, cc]) != YELLOW:
                        ok = False
                        break
                if not ok:
                    continue
                for rr in range(r0 + 1, r0 + 4):
                    for cc in range(c0 + 1, c0 + 4):
                        if (rr, cc) == center:
                            continue
                        if int(g[rr, cc]) != BLACK:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and len(border) == 15:
                    candidates.append(((r0, c0), opening))
    assert len(candidates) == 1, f"expected exactly one valid frame, found {len(candidates)}"
    return candidates[0]


def _path_connected(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    g = np.asarray(grid, dtype=int)
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) in seen:
                continue
            if (nr, nc) == end or int(g[nr, nc]) == GREY:
                seen.add((nr, nc))
                q.append((nr, nc))
    grey_seen = sum(1 for r, c in seen if int(g[r, c]) == GREY)
    return end in seen and grey_seen == int((g == GREY).sum())


def _draw_ray(out: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> None:
    r, c = start
    dr, dc = direction
    while _in_bounds(out.shape, r, c):
        if int(out[r, c]) != YELLOW:
            out[r, c] = YELLOW
        r += dr
        c += dc


def _render_output(
    shape: tuple[int, int],
    dst_top_left: tuple[int, int],
    opening: str,
) -> np.ndarray:
    out = np.full(shape, GREEN, dtype=int)
    for r, c in _frame_border_coords(dst_top_left, opening):
        out[r, c] = YELLOW
    r0, c0 = dst_top_left
    out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = YELLOW
    _draw_ray(out, _gap_coord(dst_top_left, opening), DIR_VECTORS[opening])
    for endpoint, diag in zip(_gap_endpoints(dst_top_left, opening), DIAG_DIRECTIONS[opening]):
        start = (endpoint[0] + diag[0], endpoint[1] + diag[1])
        _draw_ray(out, start, diag)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate the notched yellow frame by the displacement encoded by the gray path, then fill it and emit rays through the notch.

    Input structure:
    - The background is black.
    - One 5x5 yellow hollow frame has a single missing border-center cell on one side and a green anchor at its center.
    - Elsewhere, a second green pixel is connected by a 4-connected gray path to one lone yellow pixel.

    Rule:
    - Identify the frame-center green as the frame anchor and the other green as the path start.
    - Compute the translation vector from the path start to the lone yellow endpoint, and move the entire frame shape by that vector.
    - Fill the whole output with green, draw the translated notched frame, fill its 3x3 interior with yellow, then emit one straight yellow ray from the notch and two diagonal yellow rays from the notch endpoints in the notch-facing direction.

    Key insight:
    - The gray path's geometry is irrelevant except for its two colored endpoints, while the frame's missing border cell independently encodes the ray direction after translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (src_top_left, opening) = _detect_frame(g)
    src_border = _frame_border_coords(src_top_left, opening)
    anchor = _frame_center(src_top_left)

    green_cells = [tuple(map(int, coord)) for coord in np.argwhere(g == GREEN)]
    assert len(green_cells) == 2, f"expected exactly two green cells, found {len(green_cells)}"
    assert anchor in green_cells
    path_start = next(cell for cell in green_cells if cell != anchor)

    yellow_cells = {tuple(map(int, coord)) for coord in np.argwhere(g == YELLOW)}
    endpoint_candidates = yellow_cells - src_border
    assert len(endpoint_candidates) == 1, f"expected one yellow endpoint outside frame, found {len(endpoint_candidates)}"
    path_end = endpoint_candidates.pop()

    assert int((g == GREY).sum()) >= 1
    assert _path_connected(g, path_start, path_end), "gray path must connect the outer green start to the yellow endpoint"

    dr = path_end[0] - path_start[0]
    dc = path_end[1] - path_start[1]
    dst_top_left = (src_top_left[0] + dr, src_top_left[1] + dc)
    r0, c0 = dst_top_left
    assert 0 <= r0 <= g.shape[0] - FRAME_SIZE and 0 <= c0 <= g.shape[1] - FRAME_SIZE

    return _render_output(g.shape, dst_top_left, opening)


def _sgn(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _path_from_waypoints(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r = 0
    c = 0
    path = [(r, c)]
    for nr, nc in waypoints:
        if r != nr and c != nc:
            raise ValueError("waypoints must define cardinal segments")
        while r != nr:
            r += _sgn(nr - r)
            path.append((r, c))
        while c != nc:
            c += _sgn(nc - c)
            path.append((r, c))
    return path


def _path_turns(path: list[tuple[int, int]]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((_sgn(r1 - r0), _sgn(c1 - c0)))
    return sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)


def _path_candidates(dr: int, dc: int) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []

    def add(waypoints: list[tuple[int, int]]) -> None:
        try:
            path = _path_from_waypoints(waypoints)
        except ValueError:
            return
        if path[-1] != (dr, dc):
            return
        if len(set(path)) != len(path):
            return
        if not (5 <= len(path) <= 13):
            return
        if _path_turns(path) > 3:
            return
        candidates.append(path)

    if dr == 0 or dc == 0:
        add([(dr, dc)])
        if dr == 0:
            for off in (-1, 1):
                add([(off, 0), (off, dc), (dr, dc)])
        else:
            for off in (-1, 1):
                add([(0, off), (dr, off), (dr, dc)])
    else:
        add([(dr, 0), (dr, dc)])
        add([(0, dc), (dr, dc)])
        for split in range(1, abs(dr)):
            sr = split * _sgn(dr)
            add([(sr, 0), (sr, dc), (dr, dc)])
        for split in range(1, abs(dc)):
            sc = split * _sgn(dc)
            add([(0, sc), (dr, sc), (dr, dc)])
        sr = _sgn(dr)
        sc = _sgn(dc)
        add([(0, -sc), (dr, -sc), (dr, dc)])
        add([(0, dc + sc), (dr, dc + sc), (dr, dc)])
        add([(-sr, 0), (-sr, dc), (dr, dc)])
        add([(dr + sr, 0), (dr + sr, dc), (dr, dc)])

    unique = []
    seen = set()
    for path in candidates:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _choose_output_top_left(h: int, w: int) -> tuple[int, int]:
    return random.randint(2, h - 7), random.randint(2, w - 7)


def _choose_displacement(
    h: int,
    w: int,
    dst_top_left: tuple[int, int],
) -> tuple[int, int] | None:
    dst_bbox = _frame_bbox(dst_top_left)
    candidates = []
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if dr == 0 and dc == 0:
                continue
            manhattan = abs(dr) + abs(dc)
            if not (4 <= manhattan <= 9):
                continue
            src_top_left = (dst_top_left[0] - dr, dst_top_left[1] - dc)
            sr, sc = src_top_left
            if not (1 <= sr <= h - 6 and 1 <= sc <= w - 6):
                continue
            src_bbox = _frame_bbox(src_top_left)
            if _bbox_overlaps(src_bbox, dst_bbox, margin=1):
                continue
            if not _path_candidates(dr, dc):
                continue
            candidates.append((dr, dc))
    if not candidates:
        return None
    axis = [vec for vec in candidates if vec[0] == 0 or vec[1] == 0]
    corner = [vec for vec in candidates if vec[0] != 0 and vec[1] != 0]
    pool = axis if axis and corner and random.random() < 0.35 else candidates
    if pool is candidates and corner and random.random() < 0.6:
        pool = corner
    return random.choice(pool)


def _choose_path(
    h: int,
    w: int,
    src_top_left: tuple[int, int],
    dst_top_left: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    src_bbox = _frame_bbox(src_top_left)
    dst_bbox = _frame_bbox(dst_top_left)
    path_options = _path_candidates(dr, dc)
    bent = [path for path in path_options if _path_turns(path) >= 1]
    if bent and random.random() < 0.75:
        random.shuffle(bent)
        path_order = bent + [path for path in path_options if path not in bent]
    else:
        random.shuffle(path_options)
        path_order = path_options

    for offsets in path_order:
        rs = [r for r, _ in offsets]
        cs = [c for _, c in offsets]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        origins = []
        for base_r in range(-min_r, h - max_r):
            for base_c in range(-min_c, w - max_c):
                absolute = [(base_r + r, base_c + c) for r, c in offsets]
                if any(
                    _coord_in_expanded_bbox(coord, src_bbox, margin=1)
                    or _coord_in_expanded_bbox(coord, dst_bbox, margin=1)
                    for coord in absolute
                ):
                    continue
                origins.append((base_r, base_c))
        if origins:
            return random.choice(origins), offsets
    return None


def _build_input(
    h: int,
    w: int,
    src_top_left: tuple[int, int],
    opening: str,
    path_start: tuple[int, int],
    path_offsets: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in _frame_border_coords(src_top_left, opening):
        grid[r, c] = YELLOW
    grid[_frame_center(src_top_left)] = GREEN

    abs_path = [(path_start[0] + r, path_start[1] + c) for r, c in path_offsets]
    endpoint = abs_path[-1]
    grid[path_start] = GREEN
    for coord in abs_path[1:-1]:
        grid[coord] = GREY
    grid[endpoint] = YELLOW
    return grid


def _validate_instance(
    grid: np.ndarray,
    src_top_left: tuple[int, int],
    dst_top_left: tuple[int, int],
    opening: str,
    path_start: tuple[int, int],
    path_offsets: list[tuple[int, int]],
) -> bool:
    expected_input_path = [(path_start[0] + r, path_start[1] + c) for r, c in path_offsets]
    expected_end = expected_input_path[-1]

    detected_top_left, detected_opening = _detect_frame(grid)
    if detected_top_left != src_top_left or detected_opening != opening:
        return False

    green_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid == GREEN)}
    yellow_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid == YELLOW)}
    grey_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid == GREY)}
    if green_cells != {_frame_center(src_top_left), path_start}:
        return False
    if grey_cells != set(expected_input_path[1:-1]):
        return False
    if yellow_cells != _frame_border_coords(src_top_left, opening) | {expected_end}:
        return False
    if not _path_connected(grid, path_start, expected_end):
        return False
    if _bbox_overlaps(_frame_bbox(src_top_left), _frame_bbox(dst_top_left), margin=0):
        return False

    out = main(grid)
    expected = _render_output(grid.shape, dst_top_left, opening)
    if not np.array_equal(out, expected):
        return False
    if set(map(int, np.unique(out))) != {GREEN, YELLOW}:
        return False

    gap = _gap_coord(dst_top_left, opening)
    diag_starts = [
        (endpoint[0] + diag[0], endpoint[1] + diag[1])
        for endpoint, diag in zip(_gap_endpoints(dst_top_left, opening), DIAG_DIRECTIONS[opening])
    ]
    if any(not _in_bounds(out.shape, r, c) for r, c in [gap, *diag_starts]):
        return False
    if any(int(out[r, c]) != YELLOW for r, c in [gap, *diag_starts]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    opening = random.choice(OPENINGS)
    dst_top_left = _choose_output_top_left(h, w)
    disp = _choose_displacement(h, w, dst_top_left)
    if disp is None:
        return None
    dr, dc = disp
    src_top_left = (dst_top_left[0] - dr, dst_top_left[1] - dc)

    chosen_path = _choose_path(h, w, src_top_left, dst_top_left, dr, dc)
    if chosen_path is None:
        return None
    path_start, path_offsets = chosen_path

    grid = _build_input(h, w, src_top_left, opening, path_start, path_offsets)
    if not _validate_instance(grid, src_top_left, dst_top_left, opening, path_start, path_offsets):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
