from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREEN = 3
MAROON = 9
DOOR_COLORS = [0, 1, 2, 4, 5, 6, 7]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]

COARSE_H_RANGE = (5, 10)
COARSE_W_RANGE = (5, 11)
PATH_SEARCH_ATTEMPTS = 300
PATH_DFS_RETRIES = 120
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _component_from_start(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) == MAROON:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _trace_path_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    starts = np.argwhere(grid == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start pixel"
    start = tuple(map(int, starts[0]))
    component = _component_from_start(grid, start)
    assert len(component) >= 3, "path must contain start, exit, and at least one interior cell"

    degree: dict[tuple[int, int], int] = {}
    for cell in component:
        r, c = cell
        degree[cell] = sum((nr, nc) in component for nr, nc in _neighbors4(r, c))

    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "path component must be a single simple path"
    assert start in endpoints, "green start must lie on a path endpoint"
    assert all(deg in (1, 2) for deg in degree.values()), "path must be unbranched"
    assert sum(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in endpoints) == 2

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [cell for cell in _neighbors4(*cur) if cell in component and cell != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path walk must have a unique next step"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(component), "walk must cover the whole path component"
    end = path[-1]
    assert end != start and degree[end] == 1, "walk must end at the other endpoint"
    assert end[0] in (0, grid.shape[0] - 1) or end[1] in (0, grid.shape[1] - 1), "exit must be on the border"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique walled path marked by a green start pixel.
    Recolor each path segment using the color of the next door encountered along the route.
    The last segment adopts the last door's color, while door pixels themselves stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    path = _trace_path_cells(g)

    door_positions = [
        idx
        for idx, (r, c) in enumerate(path)
        if int(g[r, c]) not in (AZURE, GREEN, MAROON)
    ]
    if not door_positions:
        return out

    segment_start = 0
    for door_idx in door_positions:
        door_r, door_c = path[door_idx]
        fill_color = int(g[door_r, door_c])
        for idx in range(segment_start, door_idx):
            r, c = path[idx]
            out[r, c] = fill_color
        segment_start = door_idx + 1

    last_color = int(g[path[door_positions[-1]]])
    for idx in range(segment_start, len(path)):
        r, c = path[idx]
        out[r, c] = last_color

    return out


def _coarse_side(h: int, w: int, cell: tuple[int, int]) -> str | None:
    r, c = cell
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    if (on_top or on_bottom) and (on_left or on_right):
        return None
    if on_top:
        return "top"
    if on_bottom:
        return "bottom"
    if on_left:
        return "left"
    if on_right:
        return "right"
    return None


def _coarse_corners(h: int, w: int) -> set[tuple[int, int]]:
    return {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _border_cells_for_side(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    raise ValueError(side)


def _search_coarse_path(h: int, w: int) -> list[tuple[int, int]]:
    cells = h * w
    corners = _coarse_corners(h, w)

    for _ in range(PATH_SEARCH_ATTEMPTS):
        start_side = random.choice(SIDES)
        start = random.choice(_border_cells_for_side(h, w, start_side))
        min_len = random.randint(max(4, min(h, w)), max(4, min(cells // 3, h + w + 1)))
        target_len = random.randint(min_len, max(min_len, min(cells // 2, min_len + h + w)))
        max_len = min(cells - 1, target_len + random.randint(2, h + w + 2))

        best: list[tuple[int, int]] | None = None

        def free_degree(cell: tuple[int, int], used: set[tuple[int, int]]) -> int:
            r, c = cell
            total = 0
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(h, w, nr, nc):
                    continue
                if (nr, nc) in corners or (nr, nc) in used:
                    continue
                total += 1
            return total

        def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
            nonlocal best
            cur = path[-1]
            cur_side = _coarse_side(h, w, cur)
            acceptable_end = cur_side is not None and cur_side != start_side and len(path) >= min_len

            if acceptable_end:
                if best is None or abs(len(path) - target_len) < abs(len(best) - target_len):
                    best = list(path)
                if len(path) >= target_len or random.random() < 0.18:
                    return list(path)

            if len(path) >= max_len:
                return None

            nbrs = []
            for nr, nc in _neighbors4(*cur):
                if not _in_bounds(h, w, nr, nc):
                    continue
                if (nr, nc) in corners or (nr, nc) in used:
                    continue
                nbrs.append((nr, nc))

            random.shuffle(nbrs)
            nbrs.sort(
                key=lambda cell: (
                    2 if (_coarse_side(h, w, cell) not in (None, start_side) and len(path) + 1 >= min_len) else 0,
                    free_degree(cell, used),
                    random.random(),
                ),
                reverse=True,
            )

            for nxt in nbrs:
                used.add(nxt)
                path.append(nxt)
                result = dfs(path, used)
                if result is not None:
                    return result
                path.pop()
                used.remove(nxt)

            return list(path) if acceptable_end else None

        for _ in range(PATH_DFS_RETRIES):
            result = dfs([start], {start})
            if result is not None and _coarse_side(h, w, result[-1]) not in (None, start_side):
                return result
            if best is not None and len(best) >= min_len:
                return best

    raise RuntimeError("failed to generate a coarse border-to-border path")


def _coarse_to_fine(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return 2 * r + 1, 2 * c + 1


def _border_extension(h: int, w: int, cell: tuple[int, int]) -> tuple[int, int]:
    fr, fc = _coarse_to_fine(cell)
    side = _coarse_side(h, w, cell)
    assert side is not None, "path endpoint must be a non-corner border cell"
    if side == "top":
        return 0, fc
    if side == "bottom":
        return 2 * h, fc
    if side == "left":
        return fr, 0
    return fr, 2 * w


def _render_fine_path(coarse_path: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    fine_path = [_border_extension(h, w, coarse_path[0]), _coarse_to_fine(coarse_path[0])]
    prev = _coarse_to_fine(coarse_path[0])
    for coarse_cell in coarse_path[1:]:
        cur = _coarse_to_fine(coarse_cell)
        fine_path.append(((prev[0] + cur[0]) // 2, (prev[1] + cur[1]) // 2))
        fine_path.append(cur)
        prev = cur
    fine_path.append(_border_extension(h, w, coarse_path[-1]))
    return fine_path


def _sample_door_indices(path_len: int, door_count: int) -> list[int] | None:
    candidates = list(range(2, path_len - 2))
    if not candidates:
        return None
    for _ in range(200):
        picks = sorted(random.sample(candidates, door_count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:], strict=False)):
            return picks
    return None


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert int(np.sum(grid == AZURE)) > int(np.sum(grid == MAROON)), "azure must remain the dominant background"
    path = _trace_path_cells(grid)
    assert path[0][0] in (0, grid.shape[0] - 1) or path[0][1] in (0, grid.shape[1] - 1)
    door_colors = [int(grid[r, c]) for r, c in path if int(grid[r, c]) not in (AZURE, GREEN, MAROON)]
    assert door_colors, "path must contain at least one door"
    assert len(door_colors) == len(set(door_colors)), "door colors must be distinct"
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "transformation should recolor the path"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        coarse_h = random.randint(*COARSE_H_RANGE)
        coarse_w = random.randint(*COARSE_W_RANGE)
        if coarse_h < 3 or coarse_w < 3:
            continue

        coarse_path = _search_coarse_path(coarse_h, coarse_w)
        fine_path = _render_fine_path(coarse_path, coarse_h, coarse_w)
        path_set = set(fine_path)

        h = 2 * coarse_h + 1
        w = 2 * coarse_w + 1
        grid = np.full((h, w), AZURE, dtype=int)

        walls: set[tuple[int, int]] = set()
        for r, c in fine_path:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in path_set:
                    walls.add((nr, nc))
        for r, c in walls:
            grid[r, c] = MAROON

        if int(np.sum(grid == AZURE)) <= int(np.sum(grid == MAROON)):
            continue

        max_doors = min(len(DOOR_COLORS), max(1, (len(fine_path) - 4 + 1) // 2))
        door_count = random.randint(1, min(5, max_doors))
        door_indices = _sample_door_indices(len(fine_path), door_count)
        if door_indices is None:
            continue

        door_palette = random.sample(DOOR_COLORS, door_count)
        for idx, color in zip(door_indices, door_palette, strict=False):
            r, c = fine_path[idx]
            grid[r, c] = color

        sr, sc = fine_path[0]
        er, ec = fine_path[-1]
        if not (
            sr in (0, h - 1) or sc in (0, w - 1)
        ) or not (
            er in (0, h - 1) or ec in (0, w - 1)
        ):
            continue

        grid[sr, sc] = GREEN

        try:
            _validate_grid(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input after repeated retries")
