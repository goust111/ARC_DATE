from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, in_bounds, neighbors4


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

RING_SIDE_CHOICES = ("N", "E", "S", "W")
RING_COLOR_POOL = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
RING_COUNT_CHOICES = (4, 5)
GEN_ATTEMPTS = 400


def _boundary_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            cells.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            cells.append((r, c0))
    return cells


def _side_gap_choices(box: BBox, side: str) -> list[Coord]:
    r0, c0, r1, c1 = box
    if side == "N":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "E":
        return [(r, c1) for r in range(r0 + 1, r1)]
    if side == "S":
        return [(r1, c) for c in range(c0 + 1, c1)]
    return [(r, c0) for r in range(r0 + 1, r1)]


def _paint_ring(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_rings(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    ring_colors = [int(c) for c in np.unique(g) if c not in (Color.BLACK, Color.PINK)]
    rings: list[dict[str, object]] = []
    for color in ring_colors:
        mask = g == color
        comps = components_mask(mask, connectivity=4)
        assert len(comps) == 1, "each ring color must form one connected component"
        box = bbox(mask)
        assert box is not None, "ring must have a bounding box"
        boundary = _boundary_cells(box)
        ring_cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
        assert all(cell in boundary for cell in ring_cells), "ring pixels must stay on their bbox boundary"
        gap_cells = [cell for cell in boundary if cell not in ring_cells]
        assert len(gap_cells) == 1, "each ring must have exactly one boundary gap"
        gap_r, gap_c = gap_cells[0]
        r0, c0, r1, c1 = box
        assert (gap_r not in (r0, r1)) or (gap_c not in (c0, c1)), "gap must lie on a side, not a corner"
        rings.append({"color": color, "bbox": box, "gap": gap_cells[0]})

    rings.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))  # outer to inner
    assert rings, "expected at least one ring"
    return rings


def _passable_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return (g == Color.BLACK) | (g == Color.PINK)


def _unique_shortest_path(grid: np.ndarray, start: Coord, goal: Coord) -> list[Coord]:
    passable = _passable_mask(grid)
    H, W = map(int, grid.shape)
    dist: dict[Coord, int] = {start: 0}
    count: dict[Coord, int] = {start: 1}
    prev: dict[Coord, Coord | None] = {start: None}
    q: deque[Coord] = deque([start])

    while q:
        r, c = q.popleft()
        base = dist[(r, c)] + 1
        for nr, nc in neighbors4(r, c):
            if not in_bounds((H, W), nr, nc) or not bool(passable[nr, nc]):
                continue
            nxt = (int(nr), int(nc))
            if nxt not in dist:
                dist[nxt] = base
                count[nxt] = count[(r, c)]
                prev[nxt] = (r, c)
                q.append(nxt)
            elif dist[nxt] == base:
                count[nxt] = min(2, count[nxt] + count[(r, c)])
                if count[nxt] > 1:
                    prev[nxt] = None

    assert goal in dist, "endpoints must be connected"
    assert count[goal] == 1, "expected a unique shortest path"

    full_path = [goal]
    cur = goal
    while cur != start:
        predecessor = prev[cur]
        assert predecessor is not None, "unique path reconstruction failed"
        full_path.append(predecessor)
        cur = predecessor
    full_path.reverse()
    return full_path[1:-1]


def _depth_for_cell(cell: Coord, boxes: list[BBox]) -> int:
    r, c = cell
    depth = sum(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes) - 1
    return int(depth)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique route through a concentric ring maze by reversed ring depth.

    Input structure:
    - Black background contains 4 to 5 concentric one-cell-thick rectangular rings,
      each in a distinct non-black color and each with exactly one one-cell side gap.
    - Adjacent rings are separated by one-cell black corridors.
    - Exactly two magenta endpoint pixels lie on different corridor depths, and ring
      pixels are the only obstacles.

    Rule:
    - Keep every ring pixel and both magenta endpoints unchanged.
    - Read the ring colors from outermost to innermost, reverse that sequence, and
      use it as the depth palette.
    - Find the unique shortest 4-connected path between the two endpoints through
      black cells and ring gaps.
    - For each black path pixel, count how many ring bboxes contain it (inclusive),
      subtract one to get its corridor depth, and recolor it with the reversed
      ring color at that depth.

    Key insight:
    - The same concentric geometry plays two roles at once: it defines the maze
      walls and also defines the outside-in depth order whose colors are mirrored
      onto the traversed path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.PINK)]
    assert len(endpoints) == 2, "expected exactly two endpoints"

    rings = _extract_rings(g)
    boxes = [tuple(int(v) for v in ring["bbox"]) for ring in rings]
    reversed_colors = [int(ring["color"]) for ring in reversed(rings)]

    path = _unique_shortest_path(g, endpoints[0], endpoints[1])
    out = g.copy()
    for cell in path:
        r, c = cell
        assert int(g[r, c]) == Color.BLACK, "path interior must use only black cells"
        depth = _depth_for_cell(cell, boxes)
        assert 0 <= depth < len(reversed_colors), "path cell must lie in a valid corridor layer"
        out[r, c] = reversed_colors[depth]
    return out


def _ring_boxes(ring_count: int) -> list[BBox]:
    side = 4 * ring_count - 1
    return [(2 * i, 2 * i, side - 1 - 2 * i, side - 1 - 2 * i) for i in range(ring_count)]


def _sample_gaps(boxes: list[BBox]) -> tuple[list[Coord], list[str]]:
    gaps: list[Coord] = []
    sides: list[str] = []
    prev_side: str | None = None
    for box in boxes:
        side_choices = list(RING_SIDE_CHOICES)
        if prev_side is not None and random.random() < 0.8:
            side_choices = [side for side in side_choices if side != prev_side]
        side = random.choice(side_choices)
        gap = random.choice(_side_gap_choices(box, side))
        gaps.append(gap)
        sides.append(side)
        prev_side = side
    return gaps, sides


def _build_candidate() -> tuple[np.ndarray, list[Coord], list[str]]:
    ring_count = random.choice(RING_COUNT_CHOICES)
    boxes = _ring_boxes(ring_count)
    side = 4 * ring_count - 1
    colors = random.sample(RING_COLOR_POOL, ring_count)
    gaps, gap_sides = _sample_gaps(boxes)

    grid = np.zeros((side, side), dtype=int)
    for box, color, gap in zip(boxes, colors, gaps):
        _paint_ring(grid, box, color)
        grid[gap] = Color.BLACK

    center = (side // 2, side // 2)
    grid[gaps[0]] = Color.PINK
    grid[center] = Color.PINK
    return grid, gaps, gap_sides


def _validate_candidate(grid: np.ndarray, sampled_gaps: list[Coord], gap_sides: list[str]) -> bool:
    try:
        rings = _extract_rings(grid)
    except AssertionError:
        return False

    ring_count = len(rings)
    if ring_count not in RING_COUNT_CHOICES:
        return False
    if len(set(gap_sides)) < 2:
        return False

    boxes = [tuple(int(v) for v in ring["bbox"]) for ring in rings]
    expected_boxes = _ring_boxes(ring_count)
    if boxes != expected_boxes:
        return False

    actual_gaps = [tuple(int(v) for v in ring["gap"]) for ring in rings]
    if actual_gaps != [tuple(map(int, gap)) for gap in sampled_gaps]:
        return False

    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.PINK)]
    if len(endpoints) != 2:
        return False

    endpoint_depths = sorted(_depth_for_cell(cell, boxes) for cell in endpoints)
    if endpoint_depths != [0, ring_count - 1]:
        return False

    try:
        path = _unique_shortest_path(grid, endpoints[0], endpoints[1])
        out = main(grid)
    except AssertionError:
        return False

    if not path:
        return False

    path_depths = [_depth_for_cell(cell, boxes) for cell in path]
    if set(path_depths) != set(range(ring_count)):
        return False

    reversed_colors = [int(ring["color"]) for ring in reversed(rings)]
    path_set = set(path)
    for r, c in path:
        if int(grid[r, c]) != Color.BLACK:
            return False
        if int(out[r, c]) != reversed_colors[_depth_for_cell((r, c), boxes)]:
            return False

    for r, c in np.argwhere(grid != Color.BLACK):
        cell = (int(r), int(c))
        if cell in path_set:
            return False
        if int(out[r, c]) != int(grid[r, c]):
            return False

    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, gaps, gap_sides = _build_candidate()
        if _validate_candidate(grid, gaps, gap_sides):
            return grid
    raise RuntimeError("failed to generate a valid concentric-maze instance")
