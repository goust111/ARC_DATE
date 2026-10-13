from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 7
RECT_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
PATH_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up
DIR_LABELS = ("R", "D", "L", "U")
DIR_TO_VEC = dict(zip(DIR_LABELS, PATH_DIRS))
NEXT_DIRS = {
    "R": ("R", "U", "D"),
    "D": ("D", "R", "L"),
    "L": ("L", "D", "U"),
    "U": ("U", "L", "R"),
}
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _markers(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == MARKER)]
    assert len(coords) == 2, "expected exactly two marker pixels"
    return coords


def _find_rectangles(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, MARKER):
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] >= 3 and patch.shape[1] >= 3, "rectangles must be at least 3x3"
        assert np.all((patch == color) | (patch == MARKER)), "each color must occupy one solid rectangle"
        rectangles.append((int(r0), int(c0), int(r1), int(c1), color))
    assert 3 <= len(rectangles) <= 5, "expected 3 to 5 rectangles"
    return rectangles


def _hollow_rectangles(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    occupied = (g != bg) & (g != MARKER)
    out = np.full_like(g, bg)

    for r0, c0, r1, c1, color in _find_rectangles(g, bg):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r0 < r < r1 and c0 < c < c1:
                    continue
                outward: list[tuple[int, int]] = []
                if r == r0:
                    outward.append((r - 1, c))
                if r == r1:
                    outward.append((r + 1, c))
                if c == c0:
                    outward.append((r, c - 1))
                if c == c1:
                    outward.append((r, c + 1))
                exposed = any(
                    (not _in_bounds(g.shape, nr, nc)) or (not occupied[nr, nc]) for nr, nc in outward
                )
                if exposed:
                    out[r, c] = color

    for r, c in _markers(g):
        out[r, c] = MARKER
    return out


def _bfs_path(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = sorted(_markers(grid))
    start, goal = markers
    bg = _most_common_color(grid)
    passable = (grid == bg) | (grid == MARKER)
    dist = np.full(grid.shape, -1, dtype=int)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue: deque[tuple[int, int]] = deque([start])
    dist[start] = 0

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            break
        for dr, dc in PATH_DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and passable[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    assert dist[goal] >= 0, "markers must be connected after hollowing"
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow the solid rectangles into colored walls, then paint the marker path through the resulting corridor network.

    Input structure:
    - The grid has black background, 3 to 5 differently colored solid axis-aligned rectangles, and exactly two orange markers.
    - Each marker lies strictly inside a different rectangle.
    - Rectangles may be isolated, separated by one-cell gaps, or touch along full sides so the hollowed layout forms corridors.

    Rule:
    - Convert every solid rectangle into a hollow frame by clearing its strict interior to background.
    - Keep only the rectangle-border pixels that remain exposed to background; side-to-side contacts between rectangles open into passages.
    - Keep the two orange markers fixed, then find a shortest orthogonal path between them through background cells.
    - Paint the entire chosen path orange, using right, then down, then left, then up as the tie-break order among equal-length shortest routes.

    Key insight:
    - The rectangle hollowing step is not the final output: it creates the traversable topology, and the orange route is the shortest path inside that induced maze.
    """
    hollowed = _hollow_rectangles(np.asarray(grid, dtype=int))
    out = hollowed.copy()
    for r, c in _bfs_path(hollowed):
        out[r, c] = MARKER
    return out


def _advance(point: tuple[int, int], direction: str, steps: int) -> tuple[int, int]:
    dr, dc = DIR_TO_VEC[direction]
    return point[0] + dr * steps, point[1] + dc * steps


def _segment_rect(
    start: tuple[int, int], direction: str, length: int
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    r, c = start
    if direction == "R":
        return (r - 1, c - 1, r + 1, c + length), (r, c + length - 1)
    if direction == "L":
        return (r - 1, c - length, r + 1, c + 1), (r, c - length + 1)
    if direction == "D":
        return (r - 1, c - 1, r + length, c + 1), (r + length - 1, c)
    assert direction == "U"
    return (r - length, c - 1, r + 1, c + 1), (r - length + 1, c)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _build_layout(rng: random.Random) -> tuple[list[tuple[int, int, int, int]], tuple[int, int], tuple[int, int]] | None:
    n_rects = rng.randint(3, 5)
    directions = [rng.choice(DIR_LABELS)]
    for _ in range(1, n_rects):
        prev = directions[-1]
        choices = list(NEXT_DIRS[prev])
        weights = (1, 2, 2) if len(set(choices)) == 3 else (1, 1, 1)
        directions.append(rng.choices(choices, weights=weights, k=1)[0])

    lengths = [rng.randint(2, 5) for _ in range(n_rects)]
    start = (0, 0)
    marker_start = start
    marker_end = start
    rects: list[tuple[int, int, int, int]] = []

    for i, (direction, length) in enumerate(zip(directions, lengths)):
        rect, segment_end = _segment_rect(start, direction, length)
        if any(_overlap(rect, older) for older in rects):
            return None
        rects.append(rect)
        marker_end = segment_end
        if i + 1 < n_rects:
            start = _advance(segment_end, direction, 3)

    for i, rect_i in enumerate(rects):
        for j in range(i + 1, len(rects)):
            if j == i + 1:
                continue
            grown = (rect_i[0] - 1, rect_i[1] - 1, rect_i[2] + 1, rect_i[3] + 1)
            if _overlap(grown, rects[j]):
                return None

    return rects, marker_start, marker_end


def _shift_layout(
    rects: list[tuple[int, int, int, int]],
    marker_start: tuple[int, int],
    marker_end: tuple[int, int],
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int], tuple[int, int], tuple[int, int]]:
    min_r = min(r0 for r0, _c0, _r1, _c1 in rects)
    min_c = min(c0 for _r0, c0, _r1, _c1 in rects)
    max_r = max(r1 for _r0, _c0, r1, _c1 in rects)
    max_c = max(c1 for _r0, _c0, _r1, c1 in rects)
    rect_h = max_r - min_r + 1
    rect_w = max_c - min_c + 1

    top = rng.randint(1, max(1, 30 - rect_h - 1))
    left = rng.randint(1, max(1, 30 - rect_w - 1))
    bottom = rng.randint(1, max(1, 30 - rect_h - top))
    right = rng.randint(1, max(1, 30 - rect_w - left))

    row_shift = top - min_r
    col_shift = left - min_c
    shifted_rects = [(r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift) for r0, c0, r1, c1 in rects]
    shifted_start = (marker_start[0] + row_shift, marker_start[1] + col_shift)
    shifted_end = (marker_end[0] + row_shift, marker_end[1] + col_shift)
    shape = (rect_h + top + bottom, rect_w + left + right)
    return shifted_rects, shifted_start, shifted_end, shape


def _count_shortest_paths(grid: np.ndarray) -> int:
    markers = sorted(_markers(grid))
    start, goal = markers
    bg = _most_common_color(grid)
    passable = (grid == bg) | (grid == MARKER)
    dist = np.full(grid.shape, -1, dtype=int)
    queue: deque[tuple[int, int]] = deque([start])
    dist[start] = 0

    while queue:
        r, c = queue.popleft()
        for dr, dc in PATH_DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and passable[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    if dist[goal] == -1:
        return 0

    order = sorted(
        ((int(dist[r, c]), int(r), int(c)) for r, c in zip(*np.where(dist >= 0))),
        key=lambda item: (item[0], item[1], item[2]),
    )
    counts = {start: 1}
    for _d, r, c in order:
        cur = counts.get((r, c), 0)
        if cur == 0:
            continue
        for dr, dc in PATH_DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and dist[nr, nc] == dist[r, c] + 1:
                counts[(nr, nc)] = min(2, counts.get((nr, nc), 0) + cur)
    return counts.get(goal, 0)


def _render_input(
    rects: list[tuple[int, int, int, int]],
    marker_start: tuple[int, int],
    marker_end: tuple[int, int],
    shape: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    colors = rng.sample(list(RECT_COLORS), len(rects))
    grid = np.full(shape, BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG)
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    grid[marker_start] = MARKER
    grid[marker_end] = MARKER
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _build_layout(rng)
    if layout is None:
        return None
    rects, marker_start, marker_end = layout
    rects, marker_start, marker_end, shape = _shift_layout(rects, marker_start, marker_end, rng)
    if max(shape) > 30:
        return None

    grid = _render_input(rects, marker_start, marker_end, shape, rng)
    hollowed = _hollow_rectangles(grid)
    if _count_shortest_paths(hollowed) != 1:
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-maze instance")
