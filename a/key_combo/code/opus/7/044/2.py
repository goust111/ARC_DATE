from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE_COLORS = (BLUE, RED, GREEN)
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[Coord]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _right_edge_palette(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    cols: list[int] = []
    for c in range(w - 1, -1, -1):
        vals = np.unique(grid[:, c])
        if len(vals) == 1 and int(vals[0]) != BLACK:
            cols.append(c)
        else:
            break
    cols.reverse()
    assert cols, "expected a right-edge palette"
    assert cols == list(range(cols[0], w)), "palette must be contiguous at the right edge"
    palette = [int(grid[0, c]) for c in cols]
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return cols, palette


def _forbidden_mask(shape: tuple[int, int], obstacles: list[Coord]) -> np.ndarray:
    forbidden = np.zeros(shape, dtype=bool)
    for r, c in obstacles:
        forbidden[r, c] = True
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                forbidden[nr, nc] = True
    return forbidden


def _adjacency(mask: np.ndarray) -> dict[Coord, list[Coord]]:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    cells = set(coords)
    adj: dict[Coord, list[Coord]] = {}
    for r, c in coords:
        nbrs = []
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                nbrs.append((nr, nc))
        adj[(r, c)] = sorted(nbrs)
    return adj


def _unique_simple_path(mask: np.ndarray, start: Coord, end: Coord) -> list[Coord]:
    adj = _adjacency(mask)
    assert start in adj and end in adj, "endpoints must remain on the safe track"
    paths: list[list[Coord]] = []
    seen = {start}
    stack = [start]

    def dfs(node: Coord) -> None:
        if len(paths) >= 2:
            return
        if node == end:
            paths.append(stack.copy())
            return
        for nxt in adj[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
            dfs(nxt)
            stack.pop()
            seen.remove(nxt)
            if len(paths) >= 2:
                return

    dfs(start)
    assert len(paths) == 1, "expected exactly one simple safe path"
    return paths[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the unique safe path between the maroon endpoints with the right-edge palette in cyclic order.

    Input structure
    - A black background contains a teal 4-connected track network, exactly two maroon endpoint pixels, one to three orange obstacle pixels, and a contiguous set of full-height palette columns on the right edge.

    Rule
    - Treat every orange obstacle and its 8-neighborhood as forbidden, but only for pathfinding.
    - In the remaining teal/maroon track graph, find the unique simple 4-connected path between the two maroon endpoints.
    - Read the right-edge palette colors from left to right, erase those palette columns to black, then traverse the safe path from the row-major-smaller endpoint and recolor each teal path pixel by cycling through the palette; keep the maroon endpoints unchanged.

    Key insight
    - The obstacle neighborhoods do not erase the drawing directly; they only prune the teal network so that one traversal order remains, and that order controls where the palette cycle lands in space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette_cols, palette = _right_edge_palette(g)
    endpoints = sorted(tuple(map(int, p)) for p in np.argwhere(g == MAROON))
    obstacles = [tuple(map(int, p)) for p in np.argwhere(g == ORANGE)]
    assert len(endpoints) == 2, "expected exactly two maroon endpoints"
    assert 1 <= len(obstacles) <= 3, "expected one to three obstacles"
    assert 2 <= len(palette) <= 4, "expected a short palette"

    forbidden = _forbidden_mask(g.shape, obstacles)
    assert all(not forbidden[r, c] for r, c in endpoints), "endpoint cannot be forbidden"

    passable = ((g == TEAL) | (g == MAROON)) & ~forbidden
    safe_path = _unique_simple_path(passable, endpoints[0], endpoints[1])

    out = g.copy()
    out[:, palette_cols] = BLACK

    color_idx = 0
    for r, c in safe_path:
        if g[r, c] == TEAL:
            out[r, c] = palette[color_idx % len(palette)]
            color_idx += 1
    return out


def _backbone_path(track_w: int) -> tuple[int, list[Coord], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    n_levels = random.randint(3, 5)
    h = 4 * (n_levels - 1) + 3
    rows = [1 + 4 * i for i in range(n_levels)]

    left_choices = list(range(1, max(3, track_w // 3) + 1))
    right_start = max(track_w // 2 + 1, track_w - max(3, track_w // 3) - 1)
    right_choices = list(range(right_start, track_w - 1))
    assert left_choices and right_choices

    start_left = bool(random.getrandbits(1))
    cols: list[int] = []
    prev: int | None = None
    for i in range(n_levels):
        band = left_choices if ((i % 2 == 0) == start_left) else right_choices
        candidates = band[:]
        random.shuffle(candidates)
        if prev is None:
            col = candidates[0]
        else:
            candidates.sort(key=lambda c: (abs(c - prev), random.random()), reverse=True)
            col = candidates[0]
        cols.append(col)
        prev = col

    path: list[Coord] = [(0, cols[0]), (1, cols[0])]
    horiz_segments: list[tuple[int, int, int]] = []
    vert_segments: list[tuple[int, int, int]] = []
    for i in range(n_levels - 1):
        r = rows[i]
        c0 = cols[i]
        c1 = cols[i + 1]
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            path.append((r, c))
        for rr in range(r + 1, rows[i + 1] + 1):
            path.append((rr, c1))
        horiz_segments.append((r, min(c0, c1), max(c0, c1)))
        vert_segments.append((c1, r, rows[i + 1]))

    for rr in range(rows[-1] + 1, h):
        path.append((rr, cols[-1]))
    vert_segments.append((cols[-1], rows[-1], h - 1))

    assert len(path) == len(set(path)), "backbone must be simple"
    return h, path, horiz_segments, vert_segments


def _loop_shape(row: int, base: int, side: str) -> tuple[list[Coord], Coord, set[Coord]]:
    if side == "down":
        detour = [
            (row + 1, base),
            (row + 2, base),
            (row + 3, base),
            (row + 3, base + 1),
            (row + 3, base + 2),
            (row + 2, base + 2),
            (row + 1, base + 2),
        ]
        obstacle = (row + 2, base + 1)
    else:
        detour = [
            (row - 1, base),
            (row - 2, base),
            (row - 3, base),
            (row - 3, base + 1),
            (row - 3, base + 2),
            (row - 2, base + 2),
            (row - 1, base + 2),
        ]
        obstacle = (row - 2, base + 1)
    top = {(row, base), (row, base + 1), (row, base + 2)}
    return detour, obstacle, top


def _can_place(cells: Iterable[Coord], occupied: set[Coord], h: int, track_w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < track_w):
            return False
        if (r, c) in occupied:
            return False
    return True


def _build_expected(inp: np.ndarray, palette_cols: list[int], palette: list[int], safe_path: list[Coord]) -> np.ndarray:
    out = inp.copy()
    out[:, palette_cols] = BLACK
    idx = 0
    for r, c in safe_path:
        if out[r, c] == TEAL:
            out[r, c] = palette[idx % len(palette)]
            idx += 1
    return out


def _generate_attempt() -> np.ndarray | None:
    palette = random.sample(PALETTE_COLORS, random.randint(2, 3))
    palette_w = len(palette)
    gap_w = random.randint(1, 3)
    track_w = random.randint(11, 17)
    total_w = track_w + gap_w + palette_w

    h, safe_path, horiz_segments, vert_segments = _backbone_path(track_w)
    safe_set = set(safe_path)
    occupied = set(safe_path)

    loop_candidates: list[tuple[list[Coord], Coord, set[Coord]]] = []
    for row, left, right in horiz_segments:
        if right - left < 4:
            continue
        for side in ("up", "down"):
            if side == "up" and row - 3 < 1:
                continue
            if side == "down" and row + 3 > h - 2:
                continue
            for base in range(left + 1, right - 2):
                loop_candidates.append(_loop_shape(row, base, side))

    random.shuffle(loop_candidates)
    desired_loops = min(random.choice([1, 2, 2, 3]), len(loop_candidates))
    loop_tracks: list[Coord] = []
    obstacles: list[Coord] = []
    reserved_backbone = set()
    for detour, obstacle, top in loop_candidates:
        if len(obstacles) >= desired_loops:
            break
        if top & reserved_backbone:
            continue
        if not _can_place(detour + [obstacle], occupied, h, track_w):
            continue
        if any(max(abs(orow - obstacle[0]), abs(ocol - obstacle[1])) <= 2 for orow, ocol in obstacles):
            continue
        loop_tracks.extend(detour)
        obstacles.append(obstacle)
        occupied.update(detour)
        occupied.add(obstacle)
        reserved_backbone.update(top)

    if not obstacles:
        return None

    branch_candidates: list[list[Coord]] = []
    for row, left, right in horiz_segments:
        for c in range(left + 1, right):
            if (row, c) in reserved_backbone:
                continue
            for side in ("up", "down"):
                for length in (3, 2):
                    cells = []
                    ok = True
                    for d in range(1, length + 1):
                        rr = row - d if side == "up" else row + d
                        cell = (rr, c)
                        if not (1 <= rr <= h - 2) or cell in occupied:
                            ok = False
                            break
                        cells.append(cell)
                    if ok:
                        branch_candidates.append(cells)
    for col, top, bottom in vert_segments:
        for r in range(top + 1, bottom):
            for side in ("left", "right"):
                cells = []
                ok = True
                for d in (1, 2):
                    cc = col - d if side == "left" else col + d
                    cell = (r, cc)
                    if not (0 <= cc < track_w) or cell in occupied:
                        ok = False
                        break
                    cells.append(cell)
                if ok:
                    branch_candidates.append(cells)

    random.shuffle(branch_candidates)
    desired_branches = random.choice([1, 1, 2, 3])
    branches: list[Coord] = []
    for cells in branch_candidates:
        if desired_branches <= 0:
            break
        if not _can_place(cells, occupied, h, track_w):
            continue
        branches.extend(cells)
        occupied.update(cells)
        desired_branches -= 1

    if not branches:
        return None

    grid = np.full((h, total_w), BLACK, dtype=int)
    for r, c in safe_path[1:-1]:
        grid[r, c] = TEAL
    grid[safe_path[0]] = MAROON
    grid[safe_path[-1]] = MAROON
    for r, c in loop_tracks + branches:
        grid[r, c] = TEAL
    for r, c in obstacles:
        grid[r, c] = ORANGE

    first_palette_col = track_w + gap_w
    palette_cols = list(range(first_palette_col, total_w))
    for i, color in enumerate(palette):
        grid[:, first_palette_col + i] = color

    forbidden = _forbidden_mask(grid.shape, obstacles)
    forbidden_track = {
        (r, c)
        for r, c in occupied
        if grid[r, c] in (TEAL, MAROON) and forbidden[r, c]
    }
    if len(forbidden_track) < 4:
        return None

    off_path = {
        (r, c)
        for r, c in occupied
        if grid[r, c] == TEAL and (r, c) not in safe_set
    }
    if len(off_path) < 4:
        return None

    passable = ((grid == TEAL) | (grid == MAROON)) & ~forbidden
    try:
        actual_path = _unique_simple_path(passable, safe_path[0], safe_path[-1])
    except AssertionError:
        return None
    if actual_path != safe_path:
        return None

    expected = _build_expected(grid, palette_cols, palette, safe_path)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
