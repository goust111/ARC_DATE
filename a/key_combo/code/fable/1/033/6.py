from __future__ import annotations

from collections import deque
import random

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREY = 5
    TEAL = 8


Coord = tuple[int, int]


def _marker_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == Color.RED) | (grid == Color.GREY)


def _visibility_graph(grid: np.ndarray) -> tuple[dict[Coord, set[Coord]], Coord]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(_marker_mask(grid))]
    sources = [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.RED)]
    assert len(sources) == 1, "expected exactly one red source marker"
    source = sources[0]
    assert source in markers, "the source must also be a marker"

    walls = grid == Color.TEAL
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in markers:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    graph = {marker: set() for marker in markers}
    for r, cols in by_row.items():
        ordered = sorted(cols)
        for c0, c1 in zip(ordered, ordered[1:]):
            if not np.any(walls[r, c0 + 1 : c1]):
                a = (r, c0)
                b = (r, c1)
                graph[a].add(b)
                graph[b].add(a)

    for c, rows in by_col.items():
        ordered = sorted(rows)
        for r0, r1 in zip(ordered, ordered[1:]):
            if not np.any(walls[r0 + 1 : r1, c]):
                a = (r0, c)
                b = (r1, c)
                graph[a].add(b)
                graph[b].add(a)

    return graph, source


def _reachable_distances(grid: np.ndarray) -> dict[Coord, int]:
    graph, source = _visibility_graph(grid)
    dist = {source: 0}
    q: deque[Coord] = deque([source])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate red reachability across the hidden row/column visibility graph and light every reachable gray marker.

    Input structure:
    - The grid contains single-pixel markers on a plain background: exactly one red source and any number of gray markers.
    - Teal wall pixels or short teal wall bars may lie between some aligned markers.
    - Two markers are potential neighbors only when they share a row or a column.

    Rule:
    - Build the implicit visibility graph on the markers: along each row and each column, only consecutive markers can connect.
    - A consecutive aligned marker pair is adjacent iff the cells strictly between them contain no teal wall pixels.
    - Starting from the unique red source, traverse that graph and recolor every reachable gray marker to red.

    Key insight:
    - The links are invisible: connectivity depends only on unobstructed aligned visibility between consecutive markers, so walls and intermediate markers sever longer straight-line paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    reachable = _reachable_distances(g)
    out = g.copy()
    for (r, c), _dist in reachable.items():
        if out[r, c] == Color.GREY:
            out[r, c] = Color.RED
    return out


# --------------- generator ---------------

_GEN_BG = Color.BLACK
_GEN_MIN_STEP = 3
_GEN_MAX_STEP = 6
_GEN_MARGIN = 1
_GEN_AXIS_SLACK = 8
_GEN_RETRY_LIMIT = 300
_GEN_MODE_BUILDERS = []
_GEN_MODE_BAG: list[int] = []


def _scene_corner_gate() -> tuple[dict[str, tuple[int, int]], str, list[tuple[str, str]]]:
    markers = {
        "S": (1, 0),
        "A": (1, 1),
        "B": (2, 1),
        "C": (2, 2),
        "U": (1, 2),
    }
    walls = [("A", "U"), ("C", "U")]
    if random.random() < 0.7:
        markers["R1"] = (0, 1)
    if random.random() < 0.55:
        markers["R2"] = (3, 2)
    if random.random() < 0.65:
        markers["U2"] = (0, 2)
        if "R1" in markers:
            walls.append(("R1", "U2"))
    return markers, "S", walls


def _scene_tee_gate() -> tuple[dict[str, tuple[int, int]], str, list[tuple[str, str]]]:
    markers = {
        "S": (1, 0),
        "A": (1, 1),
        "B": (1, 2),
        "C": (0, 1),
        "D": (2, 1),
        "U": (2, 2),
    }
    walls = [("D", "U"), ("B", "U")]
    if random.random() < 0.7:
        markers["R1"] = (3, 1)
    if random.random() < 0.55:
        markers["R2"] = (0, 2)
    if random.random() < 0.65:
        markers["U2"] = (3, 2)
        if "R1" in markers:
            walls.append(("R1", "U2"))
    return markers, "S", walls


_GEN_MODE_BUILDERS = [_scene_corner_gate, _scene_tee_gate]


def _next_mode() -> int:
    global _GEN_MODE_BAG
    if not _GEN_MODE_BAG:
        _GEN_MODE_BAG = random.sample(range(len(_GEN_MODE_BUILDERS)), len(_GEN_MODE_BUILDERS))
    return _GEN_MODE_BAG.pop()


def _transform_points(points: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    transform = random.choice(transforms)
    raw = {name: transform(r, c) for name, (r, c) in points.items()}

    rows = {r for r, _c in raw.values()}
    cols = {c for _r, c in raw.values()}
    row_map = {r: i for i, r in enumerate(sorted(rows))}
    col_map = {c: i for i, c in enumerate(sorted(cols))}
    return {name: (row_map[r], col_map[c]) for name, (r, c) in raw.items()}


def _sample_axis_positions(count: int) -> tuple[int, list[int]]:
    assert count >= 1
    steps = [random.randint(_GEN_MIN_STEP, _GEN_MAX_STEP) for _ in range(count - 1)]
    span = sum(steps)
    min_size = span + 2 * _GEN_MARGIN + 1
    max_size = min(30, min_size + _GEN_AXIS_SLACK)
    size = random.randint(min_size, max_size)
    start = random.randint(_GEN_MARGIN, size - _GEN_MARGIN - 1 - span)
    positions = [start]
    for step in steps:
        positions.append(positions[-1] + step)
    return size, positions


def _paint_wall_between(grid: np.ndarray, a: Coord, b: Coord) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        gap = hi - lo - 1
        assert gap >= 1, "wall pairs must leave space for a blocker"
        length = random.randint(1, min(3, gap))
        start = random.randint(lo + 1, hi - length)
        grid[r0, start : start + length] = Color.TEAL
        return

    assert c0 == c1, "walls must connect aligned marker pairs"
    lo, hi = sorted((r0, r1))
    gap = hi - lo - 1
    assert gap >= 1, "wall pairs must leave space for a blocker"
    length = random.randint(1, min(3, gap))
    start = random.randint(lo + 1, hi - length)
    grid[start : start + length, c0] = Color.TEAL


def _render_scene() -> np.ndarray:
    builder = _GEN_MODE_BUILDERS[_next_mode()]
    abstract_markers, source_name, wall_pairs = builder()
    markers = _transform_points(abstract_markers)

    row_count = 1 + max(r for r, _c in markers.values())
    col_count = 1 + max(c for _r, c in markers.values())
    height, row_positions = _sample_axis_positions(row_count)
    width, col_positions = _sample_axis_positions(col_count)
    grid = np.full((height, width), _GEN_BG, dtype=int)

    actual: dict[str, Coord] = {}
    for name, (r_idx, c_idx) in markers.items():
        coord = (row_positions[r_idx], col_positions[c_idx])
        actual[name] = coord
        grid[coord] = Color.RED if name == source_name else Color.GREY

    for left, right in wall_pairs:
        _paint_wall_between(grid, actual[left], actual[right])

    return grid


def _valid_generated_sample(grid: np.ndarray) -> bool:
    out = main(grid)
    recolored = int(np.count_nonzero((grid == Color.GREY) & (out == Color.RED)))
    stuck = int(np.count_nonzero((grid == Color.GREY) & (out == Color.GREY)))
    wall_pixels = int(np.count_nonzero(grid == Color.TEAL))
    distances = _reachable_distances(grid)
    return (
        recolored >= 2
        and stuck >= 1
        and wall_pixels >= 2
        and max(distances.values()) >= 2
        and np.any(out != grid)
    )


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _render_scene()
        if _valid_generated_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid visibility-propagation sample")
