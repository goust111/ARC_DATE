from __future__ import annotations

import random
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, crop_bbox, in_bounds, neighbors4, neighbors8


PATTERN_HEIGHT = 2
MIN_PERIOD = 4
MAX_PERIOD = 6

GEN_H_RANGE = (16, 24)
GEN_W_RANGE = (16, 24)
GEN_PATTERN_WIDTHS = (2, 3)
GEN_MIN_PATH_LEN = 12
GEN_MIN_TURNS = 2
GEN_MAX_ATTEMPTS = 400
GEN_EDGE_SAMPLE = 8


def _component_bbox(shape: tuple[int, int], coords: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    box = bbox(mask)
    assert box is not None
    return box


def _find_template_and_periods(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    grey_components = components_mask(grid == Color.GREY, connectivity=4)
    assert len(grey_components) == 2, "expected one horizontal and one vertical grey control bar"

    horizontal_box = None
    vertical_box = None
    for coords in grey_components:
        box = _component_bbox(grid.shape, coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 1 and MIN_PERIOD <= w <= MAX_PERIOD:
            horizontal_box = box
        elif w == 1 and MIN_PERIOD <= h <= MAX_PERIOD:
            vertical_box = box
    assert horizontal_box is not None and vertical_box is not None, "could not identify control bars"

    _, pattern_w, _, _ = horizontal_box
    pattern_h, _, _, _ = vertical_box
    assert pattern_w >= 2 and pattern_h >= 2

    template = crop_bbox(grid, (0, 0, pattern_h - 1, pattern_w - 1))
    orange_cells = [tuple(map(int, p)) for p in np.argwhere(template == Color.ORANGE)]
    assert 1 <= len(orange_cells) <= 3, "template must contain 1..3 orange pixels"
    assert set(map(int, np.unique(template))) <= {Color.TEAL, Color.ORANGE}

    period_x = horizontal_box[3] - horizontal_box[1] + 1
    period_y = vertical_box[2] - vertical_box[0] + 1
    return template, orange_cells, int(period_x), int(period_y)


def _implicit_obstacles(
    grid: np.ndarray, orange_cells: list[tuple[int, int]], period_x: int, period_y: int
) -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    h, w = grid.shape
    for base_r, base_c in orange_cells:
        for r in range(base_r, h, period_y):
            for c in range(base_c, w, period_x):
                if grid[r, c] in (Color.TEAL, Color.ORANGE):
                    obstacles.add((int(r), int(c)))
    return obstacles


def _forbidden_mask(grid: np.ndarray, obstacles: set[tuple[int, int]]) -> np.ndarray:
    forbidden = np.zeros(grid.shape, dtype=bool)
    for r, c in obstacles:
        forbidden[r, c] = True
        for nr, nc in neighbors8(r, c):
            if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == Color.TEAL:
                forbidden[nr, nc] = True
    return forbidden


def _passable_mask(grid: np.ndarray, forbidden: np.ndarray) -> np.ndarray:
    return ((grid == Color.TEAL) | (grid == Color.MAROON)) & ~forbidden


def _bfs_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]] | None:
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in neighbors4(*cur):
            nr, nc = nxt
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    if end not in parent:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _bridge_edges(passable: np.ndarray, start: tuple[int, int]) -> set[frozenset[tuple[int, int]]]:
    sys.setrecursionlimit(10000)
    entry: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    bridges: set[frozenset[tuple[int, int]]] = set()
    timer = 0

    def dfs(node: tuple[int, int]) -> None:
        nonlocal timer
        entry[node] = timer
        low[node] = timer
        timer += 1
        for nxt in neighbors4(*node):
            nr, nc = nxt
            if not (in_bounds(passable.shape, nr, nc) and passable[nr, nc]):
                continue
            if nxt == parent[node]:
                continue
            if nxt in entry:
                low[node] = min(low[node], entry[nxt])
                continue
            parent[nxt] = node
            dfs(nxt)
            low[node] = min(low[node], low[nxt])
            if low[nxt] > entry[node]:
                bridges.add(frozenset((node, nxt)))

    dfs(start)
    return bridges


def _unique_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]] | None:
    path = _bfs_path(passable, start, end)
    if path is None:
        return None
    bridges = _bridge_edges(passable, start)
    for a, b in zip(path, path[1:]):
        if frozenset((a, b)) not in bridges:
            return None
    return path


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the periodic obstacle field, forbid every obstacle and its 8-neighborhood, then paint the unique safe route between the maroon endpoints.

    Input structure:
    - The whole canvas starts teal.
    - In the upper-left corner, a 2x2 or 2x3 template contains 1..3 orange obstacle pixels on teal.
    - A one-cell-high grey bar immediately to the template's right gives the horizontal repeat period, and a one-cell-wide grey bar immediately below the template gives the vertical repeat period.
    - Two maroon single-pixel endpoints sit on border cells outside the implicit forbidden zone.

    Rule:
    - Read the template's orange pixels and repeat only those orange positions from the template origin using the two grey-bar periods across the whole grid; copies only occupy cells that were teal/orange in the visible input.
    - Treat every repeated orange cell and every teal cell in its 8-neighborhood as forbidden for routing.
    - In the remaining 4-connected teal corridor together with the two maroon endpoints, find the unique simple path between the endpoints and repaint its teal cells maroon.
    - Leave the endpoints, visible orange template pixels, grey control bars, and all non-path cells unchanged.

    Key insight:
    - The actual maze is not drawn explicitly: the small orange template plus the two control-bar lengths define a hidden periodic obstacle field, and only after reconstructing that field can the unique safe route be recovered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template, orange_cells, period_x, period_y = _find_template_and_periods(g)
    _ = template  # retained for symmetry with the decoded structure and doc indexing
    obstacles = _implicit_obstacles(g, orange_cells, period_x, period_y)
    forbidden = _forbidden_mask(g, obstacles)

    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == Color.MAROON)]
    assert len(endpoints) == 2, "expected exactly two maroon endpoints"
    assert all(not forbidden[p] for p in endpoints), "endpoints must stay outside forbidden cells"

    passable = _passable_mask(g, forbidden)
    path = _unique_path(passable, endpoints[0], endpoints[1])
    assert path is not None, "expected a unique safe path between the endpoints"

    out = g.copy()
    for r, c in path:
        if out[r, c] == Color.TEAL:
            out[r, c] = Color.MAROON
    return out


def _render_base_grid(
    h: int,
    w: int,
    pattern_w: int,
    period_x: int,
    period_y: int,
    orange_cells: tuple[tuple[int, int], ...],
) -> np.ndarray:
    grid = np.full((h, w), Color.TEAL, dtype=int)
    for r, c in orange_cells:
        grid[r, c] = Color.ORANGE
    grid[0, pattern_w : pattern_w + period_x] = Color.GREY
    grid[PATTERN_HEIGHT : PATTERN_HEIGHT + period_y, 0] = Color.GREY
    return grid


def _component_is_tree(cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    edges = 0
    for cell in cells:
        for nxt in neighbors4(*cell):
            if nxt in cell_set:
                edges += 1
    return edges // 2 == len(cells) - 1


def _sample_cells(cells: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    if len(cells) <= limit:
        return cells[:]
    return random.sample(cells, limit)


def _candidate_endpoint_pairs(passable: np.ndarray) -> list[tuple[int, int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]]:
    h, w = passable.shape
    candidates = []
    specs = [
        (lambda p: p[1] == 0 or p[0] == h - 1, lambda p: p[0] == 0 or p[1] == w - 1),
        (lambda p: p[1] == 0, lambda p: p[1] == w - 1),
        (lambda p: p[0] == h - 1, lambda p: p[0] == 0),
        (lambda p: p[1] == w - 1 or p[0] == h - 1, lambda p: p[0] == 0 or p[1] == 0),
    ]

    for comp_coords in components_mask(passable, connectivity=4):
        cells = [tuple(map(int, p)) for p in comp_coords]
        if len(cells) < GEN_MIN_PATH_LEN + 1 or not _component_is_tree(cells):
            continue
        cell_set = set(cells)
        border_cells = [p for p in cells if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
        if len(border_cells) < 2:
            continue

        for is_start, is_end in specs:
            start_pool = [p for p in border_cells if is_start(p)]
            end_pool = [p for p in border_cells if is_end(p)]
            if not start_pool or not end_pool:
                continue

            for start in _sample_cells(start_pool, GEN_EDGE_SAMPLE):
                q = deque([start])
                dist = {start: 0}
                parent = {start: None}
                while q:
                    cur = q.popleft()
                    for nxt in neighbors4(*cur):
                        if nxt in cell_set and nxt not in dist:
                            dist[nxt] = dist[cur] + 1
                            parent[nxt] = cur
                            q.append(nxt)

                for end in end_pool:
                    if end == start or end not in dist or dist[end] < GEN_MIN_PATH_LEN:
                        continue
                    path = []
                    cur = end
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    turns = _path_turns(path)
                    if turns < GEN_MIN_TURNS:
                        continue
                    score = dist[end] * 10 + turns
                    candidates.append((score, turns, path, start, end))
    return candidates


def _validate_generated_case(grid: np.ndarray) -> bool:
    template, orange_cells, period_x, period_y = _find_template_and_periods(grid)
    assert template.shape[0] == PATTERN_HEIGHT
    obstacles = _implicit_obstacles(grid, orange_cells, period_x, period_y)
    forbidden = _forbidden_mask(grid, obstacles)
    endpoints = [tuple(map(int, p)) for p in np.argwhere(grid == Color.MAROON)]
    if len(endpoints) != 2 or any(forbidden[p] for p in endpoints):
        return False

    passable = _passable_mask(grid, forbidden)
    path = _unique_path(passable, endpoints[0], endpoints[1])
    if path is None or len(path) - 1 < GEN_MIN_PATH_LEN or _path_turns(path) < GEN_MIN_TURNS:
        return False

    out = main(grid)
    changed = int(np.sum((grid == Color.TEAL) & (out == Color.MAROON)))
    return changed == len(path) - 2 and changed > 0


def _gen_attempt(
    *,
    period_x: int | None = None,
    period_y: int | None = None,
    pattern_w: int | None = None,
    orange_count: int | None = None,
) -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    pattern_w = random.choice(GEN_PATTERN_WIDTHS) if pattern_w is None else int(pattern_w)
    period_x = random.randint(MIN_PERIOD, MAX_PERIOD) if period_x is None else int(period_x)
    period_y = random.randint(MIN_PERIOD, MAX_PERIOD) if period_y is None else int(period_y)

    template_cells = [(r, c) for r in range(PATTERN_HEIGHT) for c in range(pattern_w)]
    max_oranges = min(3, len(template_cells))
    if orange_count is None:
        orange_count = random.randint(1, max_oranges)
    orange_count = min(int(orange_count), max_oranges)
    orange_cells = tuple(sorted(random.sample(template_cells, orange_count)))

    grid = _render_base_grid(h, w, pattern_w, period_x, period_y, orange_cells)
    obstacles = _implicit_obstacles(grid, list(orange_cells), period_x, period_y)
    forbidden = _forbidden_mask(grid, obstacles)
    passable = _passable_mask(grid, forbidden)

    candidates = _candidate_endpoint_pairs(passable)
    if not candidates:
        return None

    best_score = max(score for score, _, _, _, _ in candidates)
    shortlist = [item for item in candidates if item[0] >= best_score - 20]
    _, _, path, start, end = random.choice(shortlist)

    out = grid.copy()
    out[start] = Color.MAROON
    out[end] = Color.MAROON

    if len(path) - 1 < GEN_MIN_PATH_LEN or _path_turns(path) < GEN_MIN_TURNS:
        return None
    if not _validate_generated_case(out):
        return None
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-obstacle routing puzzle")
