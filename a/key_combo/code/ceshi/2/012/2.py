"""Topology-based obstacle classification and path tracing."""

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        component = []
        while queue:
            r, c = queue.pop()
            component.append((r, c))
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        result.append(component)
    return result


def _has_hole(shape, component):
    blocked = np.zeros(shape, dtype=bool)
    for point in component:
        blocked[point] = True
    exterior = np.zeros(shape, dtype=bool)
    queue = deque()
    h, w = shape
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if not blocked[nr, nc] and not exterior[nr, nc]:
                    exterior[nr, nc] = True
                    queue.append((nr, nc))
    return bool((~blocked & ~exterior).any())


def _closed_object_mask(grid):
    walls = np.zeros(grid.shape, dtype=bool)
    for color in set(map(int, grid.flat)) - {BG, RED}:
        for component in _components(grid == color):
            if _has_hole(grid.shape, component):
                for point in component:
                    walls[point] = True
    return walls


def _shortest(walls, start, goal, *, count_paths=False, blocked_edge=None):
    h, w = walls.shape
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=np.int64)
    previous = {}
    dist[start] = 0
    ways[start] = 1
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        r, c = cell
        for dr, dc in DIRS:
            nxt = r + dr, c + dc
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w) or walls[nxt]:
                continue
            if blocked_edge is not None and frozenset((cell, nxt)) == blocked_edge:
                continue
            candidate = dist[cell] + 1
            if dist[nxt] < 0:
                dist[nxt] = candidate
                ways[nxt] = ways[cell]
                previous[nxt] = cell
                queue.append(nxt)
            elif dist[nxt] == candidate:
                ways[nxt] = min(2, ways[nxt] + ways[cell])
    if dist[goal] < 0:
        return None, 0, None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return int(dist[goal]), int(ways[goal]) if count_paths else 1, path


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the unique shortest path while only closed objects act as walls.

    Input structure: A black grid contains two isolated red endpoints and several
    monochrome four-connected objects. Some objects enclose background holes,
    while open line-like objects do not; object colors do not encode their type.
    Rule: Mark every pixel of each hole-enclosing object impassable, allow travel
    through background and open-object pixels, then recolor the intermediate cells
    of the unique shortest four-neighbor endpoint path yellow.
    Key insight: Passability is a topological property of each individual object,
    not a color property; open foreground pixels remain part of the path graph.
    """
    grid = np.asarray(grid)
    endpoints = [tuple(x) for x in np.argwhere(grid == RED)]
    if len(endpoints) != 2:
        raise ValueError("expected exactly two red endpoints")
    walls = _closed_object_mask(grid)
    _, _, path = _shortest(walls, endpoints[0], endpoints[1])
    if path is None:
        raise ValueError("endpoints are disconnected")
    out = grid.copy()
    for point in path[1:-1]:
        out[point] = YELLOW
    return out


def _paint_frame(grid, top, left, height, width, color):
    grid[top, left:left + width] = color
    grid[top + height - 1, left:left + width] = color
    grid[top:top + height, left] = color
    grid[top:top + height, left + width - 1] = color


def _expanded4(mask):
    padded = np.pad(mask, 1)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _has_longer_alternative(walls, path, distance):
    for a, b in zip(path, path[1:]):
        alt, _, _ = _shortest(walls, path[0], path[-1], blocked_edge=frozenset((a, b)))
        if alt is not None and alt > distance:
            return True
    return False


def _place_open_line(grid, color, forbidden, min_length=2, max_length=4):
    h, w = grid.shape
    for _ in range(150):
        horizontal = random.choice((False, True))
        length = random.randint(min_length, max_length)
        if horizontal:
            r = random.randrange(h)
            c = random.randrange(w - length + 1)
            cells = [(r, c + i) for i in range(length)]
        else:
            r = random.randrange(h - length + 1)
            c = random.randrange(w)
            cells = [(r + i, c) for i in range(length)]
        if any(grid[p] != BG or p in forbidden for p in cells):
            continue
        # Keep a same-color open line separate from same-color closed objects.
        if any(
            0 <= p[0] + dr < h
            and 0 <= p[1] + dc < w
            and grid[p[0] + dr, p[1] + dc] == color
            for p in cells
            for dr, dc in DIRS
        ):
            continue
        for point in cells:
            grid[point] = color
        return set(cells)
    return None


def generate_input() -> np.ndarray:
    palette = [1, 3, 5, 6, 7, 8, 9]
    for _ in range(6000):
        h = random.randint(14, 19)
        w = random.randint(14, 19)
        walls = np.zeros((h, w), dtype=bool)
        frame_masks = []
        for _ in range(random.randint(5, 8)):
            for _ in range(50):
                frame_h = random.randint(3, min(8, h - 2))
                frame_w = random.randint(3, min(7, w - 2))
                top = random.randint(0, h - frame_h)
                left = random.randint(0, w - frame_w)
                candidate = np.zeros((h, w), dtype=bool)
                _paint_frame(candidate, top, left, frame_h, frame_w, True)
                if (_expanded4(candidate) & walls).any():
                    continue
                walls |= candidate
                frame_masks.append(candidate)
                break
        if len(frame_masks) < 4:
            continue

        border = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r in (0, h - 1) or c in (0, w - 1)) and not walls[r, c]
        ]
        chosen = None
        for _ in range(60):
            start, goal = random.sample(border, 2)
            direct = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if direct < 8:
                continue
            distance, ways, path = _shortest(walls, start, goal, count_paths=True)
            if path is None or ways != 1 or distance < direct + 4:
                continue
            if not _has_longer_alternative(walls, path, distance):
                continue
            path_cells = set(path)
            adjacent_frames = sum(
                any(
                    (r + dr, c + dc) in path_cells
                    for r, c in map(tuple, np.argwhere(mask))
                    for dr, dc in DIRS
                )
                for mask in frame_masks
            )
            if adjacent_frames < 2:
                continue
            chosen = start, goal, distance, path
            break
        if chosen is None:
            continue

        start, goal, distance, path = chosen
        grid = np.zeros((h, w), dtype=int)
        frame_palette = random.sample(palette, random.randint(2, min(4, len(frame_masks))))
        frame_colors = []
        for mask in frame_masks:
            color = random.choice(frame_palette)
            frame_colors.append(color)
            grid[mask] = color
        grid[start] = RED
        grid[goal] = RED

        # Paint a connected portion of the true route as a traversable open object.
        path_length = random.randint(2, min(4, len(path) - 2))
        path_start = random.randint(1, len(path) - path_length - 1)
        route_object = path[path_start:path_start + path_length]
        open_choices = list(set(palette) - set(frame_palette))
        route_color = random.choice(open_choices)
        for point in route_object:
            grid[point] = route_color

        # Reuse a closed-frame color on a separate open line to defeat color rules.
        shared_color = random.choice(frame_colors)
        forbidden = set(path) | {start, goal}
        if _place_open_line(grid, shared_color, forbidden) is None:
            continue
        if random.random() < 0.6:
            extra_choices = list(set(palette) - {route_color, shared_color})
            if not extra_choices or _place_open_line(grid, random.choice(extra_choices), forbidden) is None:
                continue

        detected_walls = _closed_object_mask(grid)
        if not np.array_equal(detected_walls, walls):
            continue
        all_foreground = (grid != BG) & (grid != RED)
        all_distance, _, _ = _shortest(all_foreground, start, goal)
        if all_distance is not None and all_distance <= distance:
            continue
        open_pixels = (grid != BG) & (grid != RED) & ~detected_walls
        if not any(open_pixels[p] for p in path[1:-1]):
            continue
        return grid
    raise RuntimeError("failed to generate a non-degenerate topology maze")
