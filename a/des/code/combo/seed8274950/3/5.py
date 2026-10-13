from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _traverse_path(mask: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one non-background component"
    comp = comps[0]
    points = set(comp)
    h, w = mask.shape

    degrees: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg = sum((nr, nc) in points for nr, nc in _neighbors4(r, c, h, w))
        assert deg in (1, 2), "component must be a simple path"
        degrees[(r, c)] = deg

    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start = min(endpoints)
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [pt for pt in _neighbors4(cur[0], cur[1], h, w) if mask[pt] and pt != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxt[0]
        path.append(cur)

    assert len(path) == len(comp), "path traversal did not cover the full component"
    return path


def _segment_runs(colors: list[int]) -> list[tuple[int, int]]:
    assert colors, "path must contain at least one cell"
    runs: list[tuple[int, int]] = []
    cur = colors[0]
    length = 1
    for color in colors[1:]:
        if color == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = color
            length = 1
    runs.append((cur, length))
    return runs


def _render_output(segments: list[tuple[int, int]], bg: int) -> np.ndarray:
    height = 2 if any(length % 2 == 0 for _, length in segments) else 1
    out = np.full((height, len(segments)), bg, dtype=int)
    base_row = height - 1
    for idx, (color, length) in enumerate(segments):
        out[base_row, idx] = color
        if height == 2 and length % 2 == 0:
            out[0, idx] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a multi-color edge-to-edge path into path-ordered color blocks.

    The non-background cells form a single 4-connected simple path with exactly
    two endpoints. Traversal starts from the lexicographically smaller endpoint
    (topmost, then leftmost), then consecutive same-color runs are collapsed to
    one output column per run. Even-length runs get an extra pixel above.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    path_mask = g != bg
    assert np.any(path_mask), "input must contain a path"

    path = _traverse_path(path_mask)
    colors = [int(g[r, c]) for r, c in path]
    segments = _segment_runs(colors)
    return _render_output(segments, bg)


# --------------- generator ---------------

_GEN_ATTEMPTS = 400
_PATH_ATTEMPTS = 80
_GRID_MIN = 7
_GRID_MAX = 13
_MAX_SEGMENTS = 6


def _is_edge(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _visited_neighbor_count(cell: tuple[int, int], visited: set[tuple[int, int]], h: int, w: int) -> int:
    return sum((nr, nc) in visited for nr, nc in _neighbors4(cell[0], cell[1], h, w))


def _orient_top_left(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return path if path[0] <= path[-1] else list(reversed(path))


def _random_induced_path(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    edge_cells = [(r, c) for r in range(h) for c in range(w) if _is_edge((r, c), h, w)]

    for _ in range(_PATH_ATTEMPTS):
        start = rng.choice(edge_cells)
        min_len = rng.randint(max(7, min(h, w)), min(24, max(8, (h * w) // 4)))
        max_len = min((h * w) // 3, min_len + rng.randint(4, 10))
        path = [start]
        visited = {start}

        def onward_score(cell: tuple[int, int]) -> int:
            shadow = visited | {cell}
            score = 0
            for nr, nc in _neighbors4(cell[0], cell[1], h, w):
                if (nr, nc) in shadow or _is_edge((nr, nc), h, w):
                    continue
                if _visited_neighbor_count((nr, nc), shadow, h, w) == 1:
                    score += 1
            return score

        def dfs(cur: tuple[int, int]) -> bool:
            if len(path) >= min_len and cur != start and _is_edge(cur, h, w):
                return True
            if len(path) >= max_len:
                return False

            finish_moves: list[tuple[int, int]] = []
            interior_moves: list[tuple[int, int]] = []
            for nr, nc in _neighbors4(cur[0], cur[1], h, w):
                nxt = (nr, nc)
                if nxt in visited:
                    continue
                if _visited_neighbor_count(nxt, visited, h, w) != 1:
                    continue
                if _is_edge(nxt, h, w):
                    if len(path) + 1 >= min_len:
                        finish_moves.append(nxt)
                else:
                    interior_moves.append(nxt)

            rng.shuffle(finish_moves)
            rng.shuffle(interior_moves)
            interior_moves.sort(key=onward_score, reverse=True)

            if len(path) >= min_len and finish_moves and rng.random() < 0.7:
                candidates = finish_moves + interior_moves
            else:
                candidates = interior_moves + finish_moves

            for nxt in candidates:
                visited.add(nxt)
                path.append(nxt)
                if dfs(nxt):
                    return True
                path.pop()
                visited.remove(nxt)
            return False

        if dfs(start):
            edge_count = sum(_is_edge(cell, h, w) for cell in path)
            if edge_count == 2:
                return _orient_top_left(path)
    return None


def _odd_partition(total: int, rng: random.Random) -> list[int]:
    candidates = [k for k in range(2, min(_MAX_SEGMENTS, total) + 1) if k % 2 == total % 2]
    assert candidates
    k = rng.choice(candidates)
    extras = [0] * k
    for _ in range((total - k) // 2):
        extras[rng.randrange(k)] += 1
    lengths = [1 + 2 * extra for extra in extras]
    rng.shuffle(lengths)
    return lengths


def _mixed_parity_partition(total: int, rng: random.Random) -> list[int]:
    candidates = []
    for k in range(2, min(_MAX_SEGMENTS, total) + 1):
        if total % 2 == 0 and k < 3:
            continue
        candidates.append(k)
    assert candidates
    k = rng.choice(candidates)
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, total), k - 1))
        lengths = []
        prev = 0
        for cut in cuts + [total]:
            lengths.append(cut - prev)
            prev = cut
        if any(length % 2 == 0 for length in lengths) and any(length % 2 == 1 for length in lengths):
            return lengths
    raise RuntimeError("failed to create mixed-parity segment lengths")


def _segment_colors(count: int, bg: int, rng: random.Random) -> list[int]:
    palette = [color for color in range(10) if color != bg]
    colors: list[int] = []
    for _ in range(count):
        choices = [color for color in palette if color != (colors[-1] if colors else None)]
        colors.append(rng.choice(choices))
    return colors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(_GRID_MIN, _GRID_MAX)
    w = rng.randint(_GRID_MIN, _GRID_MAX)
    path = _random_induced_path(h, w, rng)
    if path is None:
        return None

    path_len = len(path)
    if path_len < 8:
        return None

    bg = rng.randrange(10)
    if rng.random() < 0.5:
        lengths = _odd_partition(path_len, rng)
    else:
        lengths = _mixed_parity_partition(path_len, rng)

    colors = _segment_colors(len(lengths), bg, rng)
    path_colors: list[int] = []
    for color, length in zip(colors, lengths):
        path_colors.extend([color] * length)
    assert len(path_colors) == path_len

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(path, path_colors):
        grid[r, c] = color

    if _most_common_color(grid) != bg:
        return None

    if sum(_is_edge((r, c), h, w) for r, c in path) != 2:
        return None

    expected = _render_output(list(zip(colors, lengths)), bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
