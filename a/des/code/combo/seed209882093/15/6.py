from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _wall_color(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> int:
    r0, c0, r1, c1 = frame_bbox
    border_values = [int(grid[r, c]) for r, c in _border_coords(r0, c0, r1, c1) if int(grid[r, c]) != bg]
    assert border_values, "expected a non-empty frame border"
    return Counter(border_values).most_common(1)[0][0]


def _ports(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in _border_coords(r0, c0, r1, c1):
        value = int(grid[r, c])
        if value in (bg, wall):
            continue
        by_color.setdefault(value, []).append((r, c))
    assert by_color, "expected at least one colored border port"
    assert all(len(cells) == 2 for cells in by_color.values()), "each port color must appear exactly twice"
    return by_color


def _find_path(
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = passable.shape
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not passable[nr, nc] or (nr, nc) in prev:
                continue
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))

    assert goal in prev, "paired ports must be connected through the maze"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color border port pair by filling the unique internal maze path.

    The input contains a grey rectangular frame with grey inner walls and black
    1-pixel corridors. Colored pixels on the frame border are ports; each color
    appears exactly twice. The output preserves every wall and port, and colors
    the black corridor cells along each paired path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg_mask = g != BLACK
    if not np.any(fg_mask):
        return g.copy()

    frame_bbox = _bbox(fg_mask)
    r0, c0, r1, c1 = frame_bbox
    bg = BLACK
    wall = _wall_color(g, frame_bbox, bg)
    ports = _ports(g, frame_bbox, bg, wall)

    out = g.copy()
    frame_passable = np.zeros_like(g, dtype=bool)
    frame_passable[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1] != wall

    all_port_cells = {cell for cells in ports.values() for cell in cells}
    for color, cells in ports.items():
        start, goal = cells
        passable = frame_passable.copy()
        for cell in all_port_cells:
            if cell not in (start, goal):
                passable[cell] = False
        path = _find_path(passable, start, goal)
        for r, c in path:
            if int(g[r, c]) == bg:
                out[r, c] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 80

INTERIOR_H_MIN, INTERIOR_H_MAX = 8, 16
INTERIOR_W_MIN, INTERIOR_W_MAX = 8, 18
EXTRA_MARGIN_MIN, EXTRA_MARGIN_MAX = 2, 6


def _jittered_split(total: int, rng: np.random.Generator) -> tuple[int, int]:
    base = total // 2
    lo = max(1, base - 1)
    hi = min(total - 1, base + 1)
    first = int(rng.integers(lo, hi + 1))
    return first, total - first


def _candidate_entries(
    side: str,
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    if side == "top":
        return [(0, c) for c in range(w) if not blocked[0, c]]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if not blocked[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if not blocked[r, 0]]
    if side == "right":
        return [(r, w - 1) for r in range(h) if not blocked[r, w - 1]]
    raise ValueError(f"unknown side {side}")


def _is_induced_path(path: list[tuple[int, int]]) -> bool:
    if len(path) < 2 or len(set(path)) != len(path):
        return False

    cells = set(path)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        degree = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                degree += 1
        degrees[(r, c)] = degree

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    mids_ok = all(degree == 2 for cell, degree in degrees.items() if cell not in endpoints)
    return len(endpoints) == 2 and mids_ok


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    return None


def _polyline_path(points: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    compact = [points[0]]
    for point in points[1:]:
        if point != compact[-1]:
            compact.append(point)

    path: list[tuple[int, int]] = []
    for a, b in zip(compact, compact[1:]):
        segment = _segment_cells(a, b)
        if segment is None:
            return None
        path.extend(segment if not path else segment[1:])
    return path


def _path_is_free(blocked: np.ndarray, path: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    return all(0 <= r < h and 0 <= c < w and not blocked[r, c] for r, c in path)


def _plan_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    sr, sc = start
    gr, gc = goal
    row_candidates = list(range(h))
    col_candidates = list(range(w))
    rng.shuffle(row_candidates)
    rng.shuffle(col_candidates)

    route_points: list[list[tuple[int, int]]] = []
    route_points.append([start, goal])
    route_points.append([start, (sr, gc), goal])
    route_points.append([start, (gr, sc), goal])

    for anchor_col in col_candidates[: min(8, len(col_candidates))]:
        route_points.append([start, (sr, anchor_col), (gr, anchor_col), goal])
    for anchor_row in row_candidates[: min(8, len(row_candidates))]:
        route_points.append([start, (anchor_row, sc), (anchor_row, gc), goal])

    for anchor_row in row_candidates[: min(5, len(row_candidates))]:
        for anchor_col in col_candidates[: min(5, len(col_candidates))]:
            route_points.append([start, (sr, anchor_col), (anchor_row, anchor_col), (anchor_row, gc), goal])
            route_points.append([start, (anchor_row, sc), (anchor_row, anchor_col), (gr, anchor_col), goal])

    rng.shuffle(route_points)
    for points in route_points:
        path = _polyline_path(points)
        if path is None or not _is_induced_path(path):
            continue
        if _path_is_free(blocked, path):
            return path
    return None


def _reserve_path(blocked: np.ndarray, path: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in path:
        blocked[r, c] = True
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _port_global(
    side: str,
    entry: tuple[int, int],
    frame_top: int,
    frame_left: int,
    interior_h: int,
    interior_w: int,
) -> tuple[int, int]:
    r, c = entry
    if side == "top":
        return frame_top, frame_left + 1 + c
    if side == "bottom":
        return frame_top + interior_h + 1, frame_left + 1 + c
    if side == "left":
        return frame_top + 1 + r, frame_left
    if side == "right":
        return frame_top + 1 + r, frame_left + interior_w + 1
    raise ValueError(f"unknown side {side}")


def _path_global(
    local_path: list[tuple[int, int]],
    frame_top: int,
    frame_left: int,
) -> list[tuple[int, int]]:
    return [(frame_top + 1 + r, frame_left + 1 + c) for r, c in local_path]


def _pair_count(interior_h: int, interior_w: int) -> int:
    max_pairs = 2
    if min(interior_h, interior_w) >= 10:
        max_pairs += 1
    if min(interior_h, interior_w) >= 14 and interior_h * interior_w >= 210:
        max_pairs += 1
    max_pairs = min(max_pairs, len(PATH_COLORS))
    return max_pairs


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    interior_h = int(rng.integers(INTERIOR_H_MIN, INTERIOR_H_MAX + 1))
    interior_w = int(rng.integers(INTERIOR_W_MIN, INTERIOR_W_MAX + 1))
    pair_max = _pair_count(interior_h, interior_w)
    if pair_max < 2:
        return None
    pair_count = int(rng.integers(2, pair_max + 1))

    frame_h = interior_h + 2
    frame_w = interior_w + 2
    extra_h = int(rng.integers(EXTRA_MARGIN_MIN, EXTRA_MARGIN_MAX + 1))
    extra_w = int(rng.integers(EXTRA_MARGIN_MIN, EXTRA_MARGIN_MAX + 1))
    top_margin, bottom_margin = _jittered_split(extra_h, rng)
    left_margin, right_margin = _jittered_split(extra_w, rng)

    grid_h = frame_h + top_margin + bottom_margin
    grid_w = frame_w + left_margin + right_margin
    frame_top = top_margin
    frame_left = left_margin

    blocked = np.zeros((interior_h, interior_w), dtype=bool)
    side_pairs = [(a, b) for i, a in enumerate(SIDES) for b in SIDES[i + 1 :]]
    rng.shuffle(side_pairs)
    colors = list(map(int, rng.choice(PATH_COLORS, size=pair_count, replace=False)))

    pair_specs: list[tuple[int, str, tuple[int, int], str, tuple[int, int], list[tuple[int, int]]]] = []
    for color in colors:
        placed = False
        local_side_pairs = list(side_pairs)
        rng.shuffle(local_side_pairs)
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            side_a, side_b = local_side_pairs[int(rng.integers(0, len(local_side_pairs)))]
            candidates_a = _candidate_entries(side_a, blocked)
            candidates_b = _candidate_entries(side_b, blocked)
            if not candidates_a or not candidates_b:
                continue
            rng.shuffle(candidates_a)
            rng.shuffle(candidates_b)

            sample_a = candidates_a[: min(len(candidates_a), 10)]
            sample_b = candidates_b[: min(len(candidates_b), 10)]
            for entry_a in sample_a:
                for entry_b in sample_b:
                    path = _plan_path(blocked, entry_a, entry_b, rng)
                    if path is None:
                        continue
                    pair_specs.append((color, side_a, entry_a, side_b, entry_b, path))
                    _reserve_path(blocked, path)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = GREY

    for color, side_a, entry_a, side_b, entry_b, path in pair_specs:
        for r, c in _path_global(path, frame_top, frame_left):
            grid[r, c] = BLACK
        pr0, pc0 = _port_global(side_a, entry_a, frame_top, frame_left, interior_h, interior_w)
        pr1, pc1 = _port_global(side_b, entry_b, frame_top, frame_left, interior_h, interior_w)
        grid[pr0, pc0] = color
        grid[pr1, pc1] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    port_color_count = Counter(int(v) for v in grid.ravel() if int(v) not in (BLACK, GREY))
    if set(port_color_count.values()) != {2}:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-port ARC input")
