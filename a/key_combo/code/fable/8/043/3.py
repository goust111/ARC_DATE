from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
COLORS = list(range(1, 10))
D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in D4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _full_2x2(cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for r0 in (r - 1, r):
            for c0 in (c - 1, c):
                square = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
                if square <= cells:
                    return True
    return False


def _is_simple_path(coords: np.ndarray) -> bool:
    if len(coords) < 2:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    degs = []
    for r, c in cells:
        degs.append(sum((nr, nc) in cells for nr, nc in _neighbors4(r, c)))
    return degs.count(1) == 2 and all(d in (1, 2) for d in degs)


def _trace_path(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    ends = [rc for rc in cells if sum((nr, nc) in cells for nr, nc in _neighbors4(*rc)) == 1]
    assert len(ends) == 2
    start = min(ends)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        nxts = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in cells and (nr, nc) != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
    assert len(order) == len(coords)
    return order


def _find_snake(grid: np.ndarray) -> np.ndarray:
    fg_parts = _components_mask(grid != BACKGROUND)
    candidates = []
    for part in fg_parts:
        cols = [int(grid[r, c]) for r, c in part]
        if len(cols) != len(set(cols)):
            continue
        if _is_simple_path(part):
            candidates.append(part)
    assert len(candidates) == 1
    return candidates[0]


def _column_from_order(color_order: list[int], area_by_color: dict[int, int]) -> np.ndarray:
    out = []
    for color in color_order:
        out.extend([int(color)] * int(area_by_color[color]))
    return np.array(out, dtype=int).reshape(-1, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the snake's color order into a single vertical quantity bar made from matching object areas.

    Input structure:
    - The background is black.
    - Exactly one foreground component is a multicolor 4-connected snake with exactly two endpoints.
    - Every snake cell has a distinct color, and each snake color also appears in exactly one separate monochrome object.
    - Additional monochrome distractor objects may use colors that never appear on the snake.

    Rule:
    - Find the snake and start from its top-left-most endpoint.
    - Trace the snake cell by cell.
    - For each visited snake color c, find the separate monochrome object of color c and append a vertical bar of color c whose length equals that object's area.
    - Ignore every object whose color is not on the snake.

    Key insight:
    - The snake supplies the output order, while the separate same-color objects supply the bar lengths; both clues are necessary and must be joined by color.
    """
    g = np.asarray(grid, dtype=int)
    snake = _find_snake(g)
    order = _trace_path(snake)

    out: list[int] = []
    for rc in order:
        r, c = rc
        color = int(g[r, c])
        parts = _components_mask(g == color)
        object_part = None
        snake_pixel_parts = 0
        for part in parts:
            cells = {tuple(map(int, xy)) for xy in part}
            if rc in cells:
                snake_pixel_parts += 1
                assert len(part) == 1
            else:
                assert object_part is None
                object_part = part
        assert snake_pixel_parts == 1
        assert object_part is not None
        out.extend([color] * len(object_part))

    return np.array(out, dtype=int).reshape(-1, 1)


def _turn_count(path: list[tuple[int, int]]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _random_snake_path(h: int, w: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        path = [start]
        seen = {start}

        def dfs() -> bool:
            if len(path) == length:
                rs = [r for r, _ in path]
                cs = [c for _, c in path]
                return _turn_count(path) >= 2 and (max(rs) > min(rs)) and (max(cs) > min(cs))

            cur = path[-1]
            prev = path[-2] if len(path) >= 2 else None
            candidates = []
            for nr, nc in _neighbors4(*cur):
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                    continue
                if (nr, nc) in seen:
                    continue
                visited_neighbors = [(ar, ac) for ar, ac in _neighbors4(nr, nc) if (ar, ac) in seen]
                if set(visited_neighbors) != {cur}:
                    continue
                if _full_2x2(seen | {(nr, nc)}):
                    continue
                straight = prev is not None and (nr - cur[0], nc - cur[1]) == (cur[0] - prev[0], cur[1] - prev[1])
                candidates.append(((nr, nc), straight))

            random.shuffle(candidates)
            candidates.sort(key=lambda item: item[1])
            for (nr, nc), _straight in candidates:
                seen.add((nr, nc))
                path.append((nr, nc))
                if dfs():
                    return True
                path.pop()
                seen.remove((nr, nc))
            return False

        if dfs():
            return path
    return None


def _random_connected_mask(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(list(frontier)))

    pts = np.array(sorted(cells), dtype=int)
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True

    rot = random.randint(0, 3)
    mask = np.rot90(mask, rot)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _can_place(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray | None:
    h, w = grid.shape
    mh, mw = mask.shape
    rel = np.argwhere(mask)
    if h - mh - 1 < 1 or w - mw - 1 < 1:
        return None
    for _ in range(300):
        r0 = random.randint(1, h - mh - 1)
        c0 = random.randint(1, w - mw - 1)
        coords = rel + np.array([r0, c0], dtype=int)
        if not _can_place(coords, occupied):
            continue
        grid[coords[:, 0], coords[:, 1]] = int(color)
        occupied[coords[:, 0], coords[:, 1]] = True
        return coords
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        snake = _find_snake(grid)
        order = _trace_path(snake)
        snake_colors = [int(grid[r, c]) for r, c in order]
        if len(snake_colors) < 4:
            return False
        if len(set(snake_colors)) != len(snake_colors):
            return False

        area_by_color: dict[int, int] = {}
        top_lefts: list[tuple[int, int]] = []
        for rc, color in zip(order, snake_colors):
            parts = _components_mask(grid == color)
            if len(parts) != 2:
                return False
            object_part = None
            hits = 0
            for part in parts:
                cells = {tuple(map(int, xy)) for xy in part}
                if rc in cells:
                    hits += 1
                    if len(part) != 1:
                        return False
                else:
                    object_part = part
            if hits != 1 or object_part is None:
                return False
            area_by_color[color] = len(object_part)
            top_lefts.append(_bbox_coords(object_part)[:2])

        counts = [area_by_color[color] for color in snake_colors]
        if len(set(counts)) != len(counts):
            return False
        if counts == sorted(counts) or counts == sorted(counts, reverse=True):
            return False

        pos_order = sorted(range(len(top_lefts)), key=lambda i: top_lefts[i])
        if pos_order == list(range(len(top_lefts))):
            return False

        distractor_colors = set(map(int, np.unique(grid))) - {BACKGROUND} - set(snake_colors)
        for color in distractor_colors:
            parts = _components_mask(grid == color)
            if not parts:
                return False

        actual = main(grid)
        if actual.ndim != 2 or actual.shape[1] != 1 or actual.shape[0] > 30:
            return False
        if actual.shape[0] <= len(snake_colors):
            return False

        size_desc = sorted(snake_colors, key=lambda color: area_by_color[color], reverse=True)
        size_asc = sorted(snake_colors, key=lambda color: area_by_color[color])
        pos_sorted = [snake_colors[i] for i in pos_order]
        if np.array_equal(actual, _column_from_order(size_desc, area_by_color)):
            return False
        if np.array_equal(actual, _column_from_order(size_asc, area_by_color)):
            return False
        if np.array_equal(actual, _column_from_order(pos_sorted, area_by_color)):
            return False
        if np.array_equal(actual, np.array(snake_colors, dtype=int).reshape(-1, 1)):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    snake_len = random.randint(4, 6)

    path = _random_snake_path(h, w, snake_len)
    if path is None:
        return None

    ordered_path = _trace_path(np.array(path, dtype=int))
    snake_colors = random.sample(COLORS, snake_len)

    area_choices = random.sample([2, 3, 4, 5, 6, 7], snake_len)
    random.shuffle(area_choices)
    if area_choices == sorted(area_choices) or area_choices == sorted(area_choices, reverse=True):
        return None
    area_by_color = {color: area for color, area in zip(snake_colors, area_choices)}

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    coord_to_color = {coord: color for coord, color in zip(ordered_path, snake_colors)}
    for r, c in path:
        grid[r, c] = int(coord_to_color[(r, c)])
        occupied[r, c] = True

    top_lefts = []
    for color in snake_colors:
        mask = _random_connected_mask(area_by_color[color])
        coords = _place_mask(grid, occupied, mask, color)
        if coords is None:
            return None
        top_lefts.append(_bbox_coords(coords)[:2])

    remaining_colors = [color for color in COLORS if color not in snake_colors]
    max_distractors = min(3, len(remaining_colors))
    if max_distractors > 0:
        distractor_count = random.choice([0, 1, 1, 2, 2, max_distractors])
        distractor_count = min(distractor_count, max_distractors)
        random.shuffle(remaining_colors)
        for color in remaining_colors[:distractor_count]:
            area = random.randint(1, 5)
            mask = _random_connected_mask(area)
            coords = _place_mask(grid, occupied, mask, color)
            if coords is None:
                return None

    pos_order = sorted(range(len(top_lefts)), key=lambda i: top_lefts[i])
    if pos_order == list(range(len(top_lefts))):
        return None

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
