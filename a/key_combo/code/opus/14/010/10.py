import random
from itertools import combinations

import numpy as np


BG = 0
WALL = 1
MARKER = 7
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _edge_direction(pos: Coord, shape: tuple[int, int]) -> Coord:
    r, c = pos
    h, w = shape
    dirs: list[Coord] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "marker must lie on exactly one edge"
    return dirs[0]


def _wall_segment(g: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    h, w = g.shape
    if dr != 0:
        lo = hi = c
        while lo - 1 >= 0 and g[r, lo - 1] == WALL:
            lo -= 1
        while hi + 1 < w and g[r, hi + 1] == WALL:
            hi += 1
        return lo, hi

    lo = hi = r
    while lo - 1 >= 0 and g[lo - 1, c] == WALL:
        lo -= 1
    while hi + 1 < h and g[hi + 1, c] == WALL:
        hi += 1
    return lo, hi


def _opening_branches(
    g: np.ndarray, hit: Coord, dr: int, dc: int, lo: int, hi: int
) -> list[Coord]:
    r, c = hit
    openings: list[Coord] = []
    if dr != 0:
        for oc in (lo - 1, hi + 1):
            if _in_bounds(g.shape, r, oc) and g[r, oc] == BG:
                openings.append((r, oc))
        length = hi - lo + 1
        if length % 2 == 1:
            return openings
        if not openings:
            return []
        openings.sort(key=lambda pos: (abs(pos[1] - c), pos[1]))
        return [openings[0]]

    for orow in (lo - 1, hi + 1):
        if _in_bounds(g.shape, orow, c) and g[orow, c] == BG:
            openings.append((orow, c))
    length = hi - lo + 1
    if length % 2 == 1:
        return openings
    if not openings:
        return []
    openings.sort(key=lambda pos: (abs(pos[0] - r), pos[0]))
    return [openings[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast inward rays whose wall-end branching is controlled by the struck wall segment's length parity.

    Input structure:
    - A black background contains exactly one orange single-pixel marker on one non-corner grid edge.
    - Blue interior wall segments are straight contiguous horizontal or vertical runs separated from each other.
    - Every ray travels perpendicular to the wall segments it can strike.

    Rule:
    - Start from the orange edge marker and move inward one cell at a time, recoloring every visited black cell orange.
    - When the next cell is blue, recover the full struck wall segment along the axis perpendicular to the ray, recolor only the struck blue hit cell orange, and inspect the two cells just beyond that segment's ends.
    - Odd-length wall segments launch rays from every in-bounds black end opening; even-length wall segments launch exactly one ray from the nearest opening, breaking ties toward the smaller row or column.
    - Each spawned ray keeps the same inward direction and continues until it reaches the opposite boundary or a struck wall offers no valid opening.

    Key insight:
    - Branching is decided by a hidden measured property of each struck wall segment, not by the visible gap pattern alone, so the solver must count wall length before choosing how many rays continue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_cells = np.argwhere(g == MARKER)
    assert len(marker_cells) == 1, "expected exactly one orange marker"
    start = tuple(map(int, marker_cells[0]))
    dr, dc = _edge_direction(start, g.shape)

    out = g.copy()
    stack: list[Coord] = [start]
    seen_starts: set[Coord] = set()

    while stack:
        r, c = stack.pop()
        if (r, c) in seen_starts:
            continue
        seen_starts.add((r, c))
        assert g[r, c] != WALL, "ray cannot start inside a wall"
        out[r, c] = MARKER

        cr, cc = r, c
        while True:
            nr, nc = cr + dr, cc + dc
            if not _in_bounds(g.shape, nr, nc):
                break
            if g[nr, nc] != WALL:
                out[nr, nc] = MARKER
                cr, cc = nr, nc
                continue

            out[nr, nc] = MARKER
            lo, hi = _wall_segment(g, nr, nc, dr, dc)
            for branch in reversed(_opening_branches(g, (nr, nc), dr, dc, lo, hi)):
                stack.append(branch)
            break

    return out


def _sample_rows(height: int, n_walls: int) -> list[int] | None:
    for _ in range(200):
        rows = [random.randint(2, 4)]
        for _ in range(1, n_walls):
            rows.append(rows[-1] + random.randint(2, 4))
        if rows[-1] <= height - 3:
            return rows
    return None


def _ensure_separated(cols: list[int]) -> bool:
    cols = sorted(cols)
    return all(b - a >= 2 for a, b in zip(cols, cols[1:]))


def _segment_candidates(
    width: int,
    active_cols: list[int],
    parent_col: int,
    length: int,
    odd_branching: bool,
) -> list[tuple[int, int, list[int]]]:
    others = [c for c in active_cols if c != parent_col]
    candidates: list[tuple[int, int, list[int]]] = []
    for left in range(2, width - length - 1):
        right = left + length - 1
        if not (left <= parent_col <= right):
            continue
        if any(left <= c <= right for c in others):
            continue

        openings = [left - 1, right + 1]
        if odd_branching:
            children = openings
        else:
            children = [min(openings, key=lambda x: (abs(x - parent_col), x))]

        merged = others + children
        if len(set(merged)) != len(merged):
            continue
        if not _ensure_separated(merged):
            continue
        candidates.append((left, right, children))
    return candidates


def _transform_from_top(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid)
    if edge == "left":
        return grid.T.copy()
    if edge == "right":
        return np.fliplr(grid.T)
    raise ValueError(edge)


def _build_canonical_grid() -> tuple[np.ndarray, list[int]]:
    height = random.randint(16, 22)
    width = random.randint(16, 22)
    n_walls = random.randint(3, 5)
    rows = _sample_rows(height, n_walls)
    if rows is None:
        raise RuntimeError("failed to sample wall rows")

    start_col = random.randint(3, width - 4)
    active_cols = [start_col]
    walls: list[tuple[int, int, int]] = []
    parities: list[int] = []

    forced = [random.choice([0, 1]), None]
    forced[1] = 1 - forced[0]

    for i, row in enumerate(rows):
        parent_col = random.choice(active_cols)
        desired_odd = forced[i] if i < 2 else random.choice([0, 1])
        parity_orders = [bool(desired_odd), not bool(desired_odd)]
        placed = False

        for odd_branching in parity_orders:
            lengths = [3, 5, 7] if odd_branching else [2, 4, 6]
            random.shuffle(lengths)
            for length in lengths:
                if length >= width - 3:
                    continue
                candidates = _segment_candidates(width, active_cols, parent_col, length, odd_branching)
                if not candidates:
                    continue
                left, right, children = random.choice(candidates)
                active_cols.remove(parent_col)
                active_cols.extend(children)
                active_cols.sort()
                walls.append((row, left, right))
                parities.append(length % 2)
                placed = True
                break
            if placed:
                break

        if not placed:
            raise RuntimeError("failed to place a wall")

    grid = np.full((height, width), BG, dtype=int)
    grid[0, start_col] = MARKER
    for row, left, right in walls:
        grid[row, left : right + 1] = WALL
    return grid, parities


def _trace_metrics(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    marker_cells = np.argwhere(g == MARKER)
    start = tuple(map(int, marker_cells[0]))
    dr, dc = _edge_direction(start, g.shape)

    stack: list[Coord] = [start]
    seen_starts: set[Coord] = set()
    hit_count = 0
    max_branches = 1
    exit_count = 0

    while stack:
        max_branches = max(max_branches, len(stack))
        r, c = stack.pop()
        if (r, c) in seen_starts:
            continue
        seen_starts.add((r, c))
        cr, cc = r, c
        while True:
            nr, nc = cr + dr, cc + dc
            if not _in_bounds(g.shape, nr, nc):
                exit_count += 1
                break
            if g[nr, nc] != WALL:
                cr, cc = nr, nc
                continue

            hit_count += 1
            lo, hi = _wall_segment(g, nr, nc, dr, dc)
            for branch in _opening_branches(g, (nr, nc), dr, dc, lo, hi):
                stack.append(branch)
            break

    return hit_count, max_branches, exit_count


def _attempt() -> np.ndarray | None:
    try:
        canonical, parities = _build_canonical_grid()
    except RuntimeError:
        return None

    if len(set(parities)) < 2:
        return None

    edge = random.choice(["top", "bottom", "left", "right"])
    grid = _transform_from_top(canonical, edge)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None

    hit_count, max_branches, exit_count = _trace_metrics(grid)
    if hit_count < 3:
        return None
    if max_branches < 2:
        return None
    if exit_count < 2:
        return None

    if np.count_nonzero(out == MARKER) <= np.count_nonzero(grid == MARKER):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-wall ray puzzle")
