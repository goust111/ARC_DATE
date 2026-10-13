from __future__ import annotations

from collections import deque
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import Color, find_components


BG = int(Color.BLACK)
PATH = int(Color.BLUE)
ENDPOINT = int(Color.RED)
WALL = int(Color.GREY)

BASE_CONNECTOR_COLS = (3, 13, 5, 15)
BASE_BAND_ROWS = (2, 6, 10, 14)
VALID_WIDTHS = (20, 21, 22, 23, 24)
VALID_VERTICAL_SHIFTS = (0, 1, 2, 3, 4, 5)


def _endpoint_coords(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(grid == ENDPOINT)
    assert coords.shape == (2, 2), "expected exactly two red endpoints"
    pts = [tuple(map(int, rc)) for rc in coords]
    pts.sort()
    return pts[0], pts[1]


def _crop_walls(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(grid, dtype=int).copy()
    erased = np.zeros(out.shape, dtype=bool)
    components = find_components(out, bg=BG, colors=[WALL], connectivity=4, monochromatic=True)
    assert components, "expected at least one grey wall component"

    for comp in components:
        coords = comp["coords"]
        _r0, c0, _r1, _c1 = map(int, comp["bbox"])
        erase_mask = coords[:, 1] < c0 + 2
        erase = coords[erase_mask]
        out[erase[:, 0], erase[:, 1]] = BG
        erased[erase[:, 0], erase[:, 1]] = True

    return out, erased


def _bfs(passable: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=object)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        next_dist = int(dist[r, c]) + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not passable[nr, nc]:
                continue
            if dist[nr, nc] < 0:
                dist[nr, nc] = next_dist
                ways[nr, nc] = ways[r, c]
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
            elif dist[nr, nc] == next_dist:
                ways[nr, nc] += ways[r, c]

    return dist, ways, parent


def _shortest_path(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    dist, ways, parent = _bfs(passable, start)
    assert int(dist[end]) >= 0, "cropped maze must connect the endpoints"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, int(ways[end])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each grey wall locally from its own left edge, then paint the endpoint-to-endpoint shortest route through the opened maze.

    Input structure:
    - Black is the background, grey marks wall objects, and exactly two red pixels are the endpoints.
    - Grey walls are separated 4-connected objects whose local left edges determine where the crop happens.
    - The red endpoints sit in opposite broad regions of the grid and remain red in the output.

    Rule:
    - Find every 4-connected grey wall component and erase the pixels that belong to that component inside the two leftmost columns of its own bbox.
    - Treat the remaining grey pixels as blocked and every non-grey pixel as passable.
    - Compute a 4-connected shortest path between the two red endpoints in the cropped maze.
    - Paint every background cell on that route blue while keeping the red endpoints and the remaining grey walls unchanged.

    Key insight:
    - The opening is not measured in global coordinates: each wall must be cropped relative to its own bbox before pathfinding, because those local openings define the traversable maze topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start, end = _endpoint_coords(g)
    cropped, _erased = _crop_walls(g)
    passable = cropped != WALL
    path, _ways = _shortest_path(passable, start, end)

    out = cropped.copy()
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = PATH
    out[start] = ENDPOINT
    out[end] = ENDPOINT
    return out


def _build_template(width: int, shift: int, end_col: int) -> np.ndarray:
    height = 20 + shift
    grid = np.full((height, width), BG, dtype=int)
    rows = [r + shift for r in BASE_BAND_ROWS]

    for i in range(1, len(rows)):
        r0 = rows[i - 1] + 1
        r1 = rows[i] - 1
        col = BASE_CONNECTOR_COLS[i]
        if col - 2 >= 2:
            grid[r0 : r1 + 1, 0 : col - 1] = WALL
        if col + 2 <= width - 3:
            grid[r0 : r1 + 1, col : width - 2] = WALL

    start_ref = (rows[0], BASE_CONNECTOR_COLS[0])
    end_ref = (rows[-1], width - 3)
    for i, r in enumerate(rows):
        if i == 0:
            left = min(start_ref[1], BASE_CONNECTOR_COLS[1])
            right = max(start_ref[1], BASE_CONNECTOR_COLS[1])
        elif i == len(rows) - 1:
            left = min(BASE_CONNECTOR_COLS[i], end_ref[1])
            right = max(BASE_CONNECTOR_COLS[i], end_ref[1])
        else:
            left = min(BASE_CONNECTOR_COLS[i], BASE_CONNECTOR_COLS[i + 1])
            right = max(BASE_CONNECTOR_COLS[i], BASE_CONNECTOR_COLS[i + 1])
        if left - 1 >= 2:
            grid[r, 0 : left - 1] = WALL
        if right + 1 <= width - 2:
            grid[r, right - 1 : width - 2] = WALL

    start = (shift, 1)
    end = (15 + shift, end_col)
    grid[start] = ENDPOINT
    grid[end] = ENDPOINT
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    start, end = _endpoint_coords(grid)
    components = find_components(grid, bg=BG, colors=[WALL], connectivity=4, monochromatic=True)
    assert len(components) >= 2, "generator must keep multiple local wall objects"

    cropped, erased = _crop_walls(grid)
    passable_before = grid != WALL
    passable_after = cropped != WALL
    before_path, before_ways = _shortest_path(passable_before, start, end)
    after_path, after_ways = _shortest_path(passable_after, start, end)

    assert before_ways >= 1, "generated input must remain solvable for deterministic validation"
    assert after_ways == 1, "cropped maze must have a unique shortest path"
    assert len(after_path) < len(before_path), "local crop must strictly improve the route"
    assert any(erased[r, c] for r, c in after_path), "the unique route must traverse erased wall cells"

    out = main(grid)
    blue_count = int(np.count_nonzero(out == PATH))
    painted_steps = sum(int(cropped[r, c] == BG) for r, c in after_path)
    assert blue_count == painted_steps, "solver must paint exactly the background cells on the route"


def generate_input() -> np.ndarray:
    width = random.choice(VALID_WIDTHS)
    shift = random.choice(VALID_VERTICAL_SHIFTS)
    end_col = random.randint(14, width - 5)
    grid = _build_template(width, shift, end_col)
    _validate_generated(grid)
    return grid
