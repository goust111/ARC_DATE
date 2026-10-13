from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COLOR_POOL = tuple(c for c in range(10) if c != BG)
MAX_ATTEMPTS = 500


def _bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_from(grid: np.ndarray, start: tuple[int, int], color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    out: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return out


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rows = sorted(set(int(r) for r in coords[:, 0]))
        cols = sorted(set(int(c) for c in coords[:, 1]))
        coord_set = {(int(r), int(c)) for r, c in coords}
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if not 4 <= r1 - r0 + 1 <= 10:
                    continue
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1 :]:
                        if not 4 <= c1 - c0 + 1 <= 10:
                            continue
                        border = _border_cells((r0, c0, r1, c1))
                        if not border <= coord_set:
                            continue
                        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
                        if np.any(inner == color):
                            continue
                        frames.append({"color": color, "box": (r0, c0, r1, c1)})

    frames.sort(key=lambda item: (item["box"][2] - item["box"][0] + 1) * (item["box"][3] - item["box"][1] + 1), reverse=True)
    selected: list[dict] = []
    used: set[tuple[int, int]] = set()
    for frame in frames:
        cells = _border_cells(frame["box"])
        if cells & used:
            continue
        selected.append(frame)
        used |= cells
    return selected


def _arrow_for_frame(grid: np.ndarray, frame: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["box"]
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    checks = [
        ((-1, 0), [(r0, c) for c in range(c0 + 1, c1)]),
        ((1, 0), [(r1, c) for c in range(c0 + 1, c1)]),
        ((0, -1), [(r, c0) for r in range(r0 + 1, r1)]),
        ((0, 1), [(r, c1) for r in range(r0 + 1, r1)]),
    ]
    for direction, anchors in checks:
        dr, dc = direction
        for ar, ac in anchors:
            p1 = (ar + dr, ac + dc)
            p2 = (ar + 2 * dr, ac + 2 * dc)
            if _inside(grid.shape, *p1) and _inside(grid.shape, *p2):
                if int(grid[p1]) == color and int(grid[p2]) == color:
                    candidates.append(((ar, ac), direction))
    if not candidates:
        return None
    return candidates[0]


def _marker_count(grid: np.ndarray, frame: dict, bg: int) -> int:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["box"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return int(np.count_nonzero((inner != bg) & (inner != color)))


def _target_for_arrow(grid: np.ndarray, anchor: tuple[int, int], direction: tuple[int, int], bg: int, arrow_color: int) -> tuple[int, list[tuple[int, int]]] | None:
    dr, dc = direction
    r, c = anchor[0] + 3 * dr, anchor[1] + 3 * dc
    while _inside(grid.shape, r, c) and int(grid[r, c]) == bg:
        r += dr
        c += dc
    if not _inside(grid.shape, r, c):
        return None
    color = int(grid[r, c])
    if color in (bg, arrow_color):
        return None
    return color, _component_from(grid, (r, c), color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    operations: list[tuple[int, int, tuple[int, int], list[tuple[int, int]]]] = []

    for frame in _find_frames(g, bg):
        arrow = _arrow_for_frame(g, frame)
        if arrow is None:
            continue
        anchor, direction = arrow
        distance = _marker_count(g, frame, bg)
        target = _target_for_arrow(g, anchor, direction, bg, int(frame["color"]))
        if distance <= 0 or target is None:
            continue
        target_color, cells = target
        operations.append((int(frame["color"]), target_color, direction, cells, distance))

    for arrow_color, target_color, direction, cells, distance in operations:
        dr, dc = direction
        for step in range(distance):
            for r, c in cells:
                nr, nc = r + step * dr, c + step * dc
                if _inside(out.shape, nr, nc):
                    out[nr, nc] = arrow_color
        for r, c in cells:
            nr, nc = r + distance * dr, c + distance * dc
            if _inside(out.shape, nr, nc):
                out[nr, nc] = target_color
    return out


SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)


def _norm(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rot(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _norm([(c, -r) for r, c in cells])


def _sample_shape() -> tuple[tuple[int, int], ...]:
    cells = tuple(random.choice(SHAPES))
    for _ in range(random.randrange(4)):
        cells = _rot(cells)
    return cells


def _edge_cell_for_direction(shape: tuple[tuple[int, int], ...], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if dr < 0:
        edge = max(r for r, _ in shape)
        return sorted((r, c) for r, c in shape if r == edge)[0]
    if dr > 0:
        edge = min(r for r, _ in shape)
        return sorted((r, c) for r, c in shape if r == edge)[0]
    if dc < 0:
        edge = max(c for _, c in shape)
        return sorted((r, c) for r, c in shape if c == edge)[0]
    edge = min(c for _, c in shape)
    return sorted((r, c) for r, c in shape if c == edge)[0]


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if _inside(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _try_group(grid: np.ndarray, occupied: set[tuple[int, int]], colors: list[int]) -> bool:
    h, w = grid.shape
    frame_color = colors.pop()
    marker_color = colors.pop()
    object_color = colors.pop()
    direction = random.choice(tuple(DIRS.values()))
    dr, dc = direction
    fh = random.randint(5, 8)
    fw = random.randint(5, 8)
    distance = random.randint(1, 4)
    gap = random.randint(1, 3)
    shape = _sample_shape()

    for _ in range(150):
        r0 = random.randint(3, h - fh - 4)
        c0 = random.randint(3, w - fw - 4)
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        if direction == DIRS["up"]:
            anchor = (r0, random.randint(c0 + 1, c1 - 1))
        elif direction == DIRS["down"]:
            anchor = (r1, random.randint(c0 + 1, c1 - 1))
        elif direction == DIRS["left"]:
            anchor = (random.randint(r0 + 1, r1 - 1), c0)
        else:
            anchor = (random.randint(r0 + 1, r1 - 1), c1)

        ray = (anchor[0] + (3 + gap) * dr, anchor[1] + (3 + gap) * dc)
        edge = _edge_cell_for_direction(shape, direction)
        top = ray[0] - edge[0]
        left = ray[1] - edge[1]
        object_cells = {(top + r, left + c) for r, c in shape}
        shifted = [
            {(r + step * dr, c + step * dc) for r, c in object_cells}
            for step in range(distance + 1)
        ]
        arrow = {(anchor[0] + i * dr, anchor[1] + i * dc) for i in range(3)}
        frame = _border_cells((r0, c0, r1, c1))
        inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        markers = set(random.sample(inner, distance))
        reserve = set().union(frame, markers, arrow, *shifted)

        if any(not _inside(grid.shape, r, c) for r, c in reserve):
            continue
        moving_later = set().union(*shifted) - object_cells
        if moving_later & (frame | markers | arrow):
            continue
        if (arrow - {anchor}) & (frame | markers):
            continue
        if _expanded(reserve, grid.shape, 1) & occupied:
            continue

        for r, c in frame:
            grid[r, c] = frame_color
        for r, c in markers:
            grid[r, c] = marker_color
        for r, c in arrow:
            grid[r, c] = frame_color
        for r, c in object_cells:
            grid[r, c] = object_color
        occupied.update(_expanded(reserve, grid.shape, 1))
        return True

    colors.extend([frame_color, marker_color, object_color])
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(28, 36)
    w = random.randint(28, 36)
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    count = random.randint(1, 3)
    if len(colors) < 3 * count:
        return None
    for _ in range(count):
        if not _try_group(grid, occupied, colors):
            return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid push-arrow input")
