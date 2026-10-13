import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"
SIDES = (TOP, BOTTOM, LEFT, RIGHT)
SIDE_TO_COLOR = {
    TOP: RED,
    BOTTOM: GREEN,
    LEFT: YELLOW,
    RIGHT: AZURE,
}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 400
LINE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _draw_walls(grid: np.ndarray) -> None:
    grid[0, :] = RED
    grid[-1, :] = GREEN
    grid[1:-1, 0] = YELLOW
    grid[1:-1, -1] = AZURE


def _check_walls(grid: np.ndarray) -> None:
    assert np.all(grid[0, 1:-1] == RED), "top wall must be red"
    assert np.all(grid[-1, 1:-1] == GREEN), "bottom wall must be green"
    assert np.all(grid[1:-1, 0] == YELLOW), "left wall must be yellow"
    assert np.all(grid[1:-1, -1] == AZURE), "right wall must be azure"


def _endpoint_side(grid: np.ndarray, cell: tuple[int, int]) -> str | None:
    h, w = grid.shape
    r, c = cell
    sides: list[str] = []
    if r == 1 and int(grid[0, c]) == RED:
        sides.append(TOP)
    if r == h - 2 and int(grid[h - 1, c]) == GREEN:
        sides.append(BOTTOM)
    if c == 1 and int(grid[r, 0]) == YELLOW:
        sides.append(LEFT)
    if c == w - 2 and int(grid[r, w - 1]) == AZURE:
        sides.append(RIGHT)
    assert len(sides) <= 1, "wall endpoint must not touch two walls"
    return sides[0] if sides else None


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _parse_polylines(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _check_walls(g)

    parts = _components(g == GREY)
    assert parts, "expected at least one grey polyline"

    parsed: list[dict] = []
    for comp in parts:
        pts = set(comp)
        deg: dict[tuple[int, int], int] = {}
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cell in pts:
            neigh = [nb for nb in _neighbors4(*cell) if nb in pts]
            deg[cell] = len(neigh)
            adj[cell] = neigh

        assert all(v in (1, 2) for v in deg.values()), "polyline must be a simple 1px path"
        endpoints = [cell for cell, d in deg.items() if d == 1]
        assert len(endpoints) == 2, "polyline must have exactly two endpoints"

        wall_endpoints = [cell for cell in endpoints if _endpoint_side(g, cell) is not None]
        assert len(wall_endpoints) == 1, "each line needs exactly one wall-adjacent endpoint"
        start = wall_endpoints[0]
        wall_side = _endpoint_side(g, start)
        assert wall_side is not None

        path = [start]
        prev = None
        cur = start
        while True:
            nxts = [nb for nb in adj[cur] if nb != prev]
            if not nxts:
                break
            assert len(nxts) == 1, "polyline path must be unambiguous"
            prev, cur = cur, nxts[0]
            path.append(cur)
        assert len(path) == len(pts), "polyline traversal must visit every grey cell"

        corners = 0
        for i in range(1, len(path) - 1):
            in_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            out_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if in_dir != out_dir:
                corners += 1
        assert corners >= 2, "each polyline must contain at least two corners"

        parsed.append(
            {
                "path": path,
                "wall_side": wall_side,
                "wall_color": SIDE_TO_COLOR[wall_side],
            }
        )

    parsed.sort(key=lambda item: item["path"][0])
    return parsed


def _apply_branches(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    meta: list[dict] = []

    for info in _parse_polylines(g):
        path = info["path"]
        branch_color = int(info["wall_color"])
        branches: list[dict] = []
        corner_idx = 0
        for i in range(1, len(path) - 1):
            in_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            out_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if in_dir == out_dir:
                continue
            corner_idx += 1
            branch_dir = _left_of(in_dir) if corner_idx % 2 == 1 else _right_of(in_dir)
            grown: list[tuple[int, int]] = []
            for step in (1, 2):
                nr = path[i][0] + branch_dir[0] * step
                nc = path[i][1] + branch_dir[1] * step
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    break
                if int(out[nr, nc]) != BLACK:
                    break
                out[nr, nc] = branch_color
                grown.append((int(nr), int(nc)))
            branches.append({"index": corner_idx, "corner": path[i], "cells": grown})
        meta.append(
            {
                "path": path,
                "wall_side": info["wall_side"],
                "wall_color": branch_color,
                "branches": branches,
            }
        )

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating side branches from each grey wall-anchored polyline corner.

    Input structure:
    - Black background.
    - Red / green / yellow / azure border walls on the four sides.
    - One-pixel-wide grey orthogonal polylines inside the walls.

    Rule:
    - For each grey polyline, start at the endpoint that touches a wall.
    - Visit corners in path order.
    - At corner 1 grow 2 cells to the left of the incoming direction, at corner 2 grow to
      the right, then keep alternating.
    - Branch color is the color of the touched wall.
    - Stop a branch early if the next cell is out of bounds or not black.
    """
    return _apply_branches(grid)


def _canonical_relative_path(steps: list[int]) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    r = 0
    c = 0
    for seg_idx, seg_steps in enumerate(steps):
        if seg_idx % 2 == 0:
            dr, dc = 1, 0
        else:
            dr, dc = (0, -1) if (seg_idx // 2) % 2 == 0 else (0, 1)
        for _ in range(seg_steps):
            r += dr
            c += dc
            cells.append((r, c))
    return cells


def _transform_rel(cell: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = cell
    if side == TOP:
        return (r, c)
    if side == BOTTOM:
        return (-r, c)
    if side == LEFT:
        return (-c, r)
    assert side == RIGHT
    return (c, -r)


def _candidate_starts(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == TOP:
        return [(1, c) for c in range(2, w - 2)]
    if side == BOTTOM:
        return [(h - 2, c) for c in range(2, w - 2)]
    if side == LEFT:
        return [(r, 1) for r in range(2, h - 2)]
    return [(r, w - 2) for r in range(2, h - 2)]


def _translate_path(rel_path: list[tuple[int, int]], start: tuple[int, int], side: str) -> list[tuple[int, int]]:
    sr, sc = start
    out = []
    for cell in rel_path:
        dr, dc = _transform_rel(cell, side)
        out.append((sr + dr, sc + dc))
    return out


def _touches_wall(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r in (1, h - 2) or c in (1, w - 2)


def _wall_touch_sides(cell: tuple[int, int], h: int, w: int) -> list[str]:
    r, c = cell
    sides: list[str] = []
    if r == 1:
        sides.append(TOP)
    if r == h - 2:
        sides.append(BOTTOM)
    if c == 1:
        sides.append(LEFT)
    if c == w - 2:
        sides.append(RIGHT)
    return sides


def _valid_path_shape(path: list[tuple[int, int]], blocked: set[tuple[int, int]], h: int, w: int) -> bool:
    if len(path) != len(set(path)):
        return False
    if any(cell in blocked for cell in path):
        return False
    for idx, (r, c) in enumerate(path):
        if not (1 <= r <= h - 2 and 1 <= c <= w - 2):
            return False
        if idx == 0:
            if len(_wall_touch_sides((r, c), h, w)) != 1:
                return False
        elif _touches_wall((r, c), h, w):
            return False

    pts = set(path)
    for cell in pts:
        deg = sum(1 for nb in _neighbors4(*cell) if nb in pts)
        if deg not in (1, 2):
            return False
    if sum(1 for cell in pts if sum(1 for nb in _neighbors4(*cell) if nb in pts) == 1) != 2:
        return False

    for i, (r1, c1) in enumerate(path):
        for j in range(i + 2, len(path)):
            r2, c2 = path[j]
            if j == i + 2 and abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    corners = 0
    for i in range(1, len(path) - 1):
        in_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        out_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if in_dir != out_dir:
            corners += 1
    return corners >= 2


def _chebyshev_block(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((r + dr, c + dc))
    return blocked


def _sample_steps(corners: int, h: int, w: int) -> list[int]:
    steps: list[int] = []
    for seg_idx in range(corners + 1):
        if seg_idx % 2 == 0:
            hi = max(1, min(4, h - 6))
            steps.append(random.randint(1, hi))
        else:
            hi = max(1, min(5, w - 6))
            steps.append(random.randint(1, hi))
    return steps


def _try_place_line(
    h: int,
    w: int,
    side: str,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(LINE_ATTEMPTS):
        corners = random.randint(2, 5)
        rel_path = _canonical_relative_path(_sample_steps(corners, h, w))
        starts = _candidate_starts(side, h, w)
        random.shuffle(starts)
        for start in starts:
            path = _translate_path(rel_path, start, side)
            if not _valid_path_shape(path, blocked, h, w):
                continue
            return path
    return None


def _validate_generated(grid: np.ndarray, expected_lines: int) -> bool:
    try:
        parsed = _parse_polylines(grid)
        if len(parsed) != expected_lines:
            return False
        out, meta = _apply_branches(grid, return_meta=True)
    except AssertionError:
        return False

    added = (grid == BLACK) & (out != BLACK)
    if int(np.count_nonzero(added)) < max(3, expected_lines * 2):
        return False

    odd_visible = False
    even_visible = False
    longest = 0
    for item in meta:
        for branch in item["branches"]:
            longest = max(longest, len(branch["cells"]))
            if branch["cells"]:
                if branch["index"] % 2 == 1:
                    odd_visible = True
                else:
                    even_visible = True
    if longest < 2 or not odd_visible or not even_visible:
        return False
    branch_colors = {int(v) for v in out[added]}
    return branch_colors <= {RED, GREEN, YELLOW, AZURE}


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    max_lines = 2 if min(h, w) < 15 else 3 if min(h, w) < 19 else 4
    line_count = random.randint(1, max_lines)
    chosen_sides = random.sample(list(SIDES), line_count)

    grid = np.full((h, w), BLACK, dtype=int)
    _draw_walls(grid)

    blocked: set[tuple[int, int]] = set()
    for side in chosen_sides:
        path = _try_place_line(h, w, side, blocked)
        if path is None:
            return None
        for r, c in path:
            grid[r, c] = GREY
        blocked |= _chebyshev_block(path)

    if not _validate_generated(grid, line_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
