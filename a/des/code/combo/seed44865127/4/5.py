from __future__ import annotations

import random
from collections import Counter

import numpy as np


ARC_COLORS = tuple(range(10))
MIN_SEGMENTS = 2
MAX_SEGMENTS = 5
MIN_TOTAL_LENGTH = 7
MAX_TOTAL_LENGTH = 18
MIN_SIDE = 5
MAX_SIDE = 22
MAX_WALK_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _foreground_path(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fg_cells = [tuple(map(int, rc)) for rc in np.argwhere(g != bg)]
    assert len(fg_cells) >= 2, "expected a nontrivial path"

    fg = set(fg_cells)
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in fg:
        r, c = cell
        neighbors = []
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) in fg:
                neighbors.append((rr, cc))
        degree = len(neighbors)
        assert degree in (1, 2), "path pixels must have degree 1 or 2"
        neighbor_map[cell] = neighbors

    endpoints = [cell for cell, neighbors in neighbor_map.items() if len(neighbors) == 1]
    assert len(endpoints) == 2, "expected exactly two endpoints"

    seen = {endpoints[0]}
    order = [endpoints[0]]
    prev = None
    current = endpoints[0]
    while True:
        next_cells = [cell for cell in neighbor_map[current] if cell != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path must not branch"
        nxt = next_cells[0]
        assert nxt not in seen, "path must not cycle"
        order.append(nxt)
        seen.add(nxt)
        prev, current = current, nxt

    assert len(order) == len(fg), "path traversal must cover all foreground cells"
    return order


def _segment_path(grid: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    segments: list[tuple[int, int]] = []
    current_color = int(g[path[0]])
    current_len = 1

    for r, c in path[1:]:
        color = int(g[r, c])
        if color == current_color:
            current_len += 1
            continue
        segments.append((current_len, current_color))
        current_color = color
        current_len = 1
    segments.append((current_len, current_color))
    return segments


def _sorted_output(segments: list[tuple[int, int]], width_hint: int, height_hint: int) -> np.ndarray:
    ordered = sorted(segments, key=lambda seg: seg[0])
    total = sum(length for length, _color in ordered)

    if total <= width_hint:
        out = np.empty((1, total), dtype=int)
        cursor = 0
        for length, color in ordered:
            out[0, cursor : cursor + length] = int(color)
            cursor += length
        return out

    assert total <= height_hint, "path length must fit at least one straight orientation"
    out = np.empty((total, 1), dtype=int)
    cursor = 0
    for length, color in ordered:
        out[cursor : cursor + length, 0] = int(color)
        cursor += length
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the single multicolor path into same-color runs, sort the runs by
    length, and emit the colors as one straight line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "expected a nonempty canvas"

    bg = _dominant_color(g)
    path = _foreground_path(g, bg)
    segments = _segment_path(g, path)
    return _sorted_output(segments, width_hint=w, height_hint=h)


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            turns += 1
    return turns


def _coarse_future_choices(
    cell: tuple[int, int],
    visited: np.ndarray,
) -> int:
    count = 0
    for dr, dc in NEIGHBORS4:
        rr, cc = cell[0] + dr, cell[1] + dc
        if 0 <= rr < visited.shape[0] and 0 <= cc < visited.shape[1] and not visited[rr, cc]:
            count += 1
    return count


def _random_coarse_walk(h: int, w: int, node_count: int) -> list[tuple[int, int]] | None:
    min_turns = 1 if node_count >= 4 else 0

    for _ in range(MAX_WALK_ATTEMPTS):
        start = (random.randrange(h), random.randrange(w))
        visited = np.zeros((h, w), dtype=bool)
        visited[start] = True
        path = [start]

        def dfs(current: tuple[int, int]) -> bool:
            if len(path) == node_count:
                return _turn_count(path) >= min_turns

            options = []
            for dr, dc in NEIGHBORS4:
                nxt = (current[0] + dr, current[1] + dc)
                rr, cc = nxt
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if visited[rr, cc]:
                    continue
                score = _coarse_future_choices(nxt, visited)
                options.append((score, random.random(), nxt))
            options.sort(key=lambda item: (-item[0], item[1]))

            for _score, _noise, nxt in options:
                visited[nxt] = True
                path.append(nxt)
                if dfs(nxt):
                    return True
                path.pop()
                visited[nxt] = False
            return False

        if dfs(start):
            return path
    return None


def _expand_coarse_path(
    coarse_path: list[tuple[int, int]],
    total_length: int,
) -> list[tuple[int, int]]:
    fine_path = [(2 * coarse_path[0][0], 2 * coarse_path[0][1])]
    for (r0, c0), (r1, c1) in zip(coarse_path, coarse_path[1:]):
        fine_path.append((r0 + r1, c0 + c1))
        if len(fine_path) == total_length:
            break
        fine_path.append((2 * r1, 2 * c1))
        if len(fine_path) == total_length:
            break
    assert len(fine_path) == total_length
    return fine_path


def _translate_path(
    path: list[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    rows = [r for r, _c in path]
    cols = [c for _r, c in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    row_slack = h - (max_r - min_r + 1)
    col_slack = w - (max_c - min_c + 1)
    shift_r = random.randint(0, row_slack)
    shift_c = random.randint(0, col_slack)
    return [(r - min_r + shift_r, c - min_c + shift_c) for r, c in path]


def _sample_lengths() -> list[int] | None:
    for _ in range(120):
        n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
        lengths = random.sample(range(1, 9), n_segments)
        total = sum(lengths)
        if not (MIN_TOTAL_LENGTH <= total <= MAX_TOTAL_LENGTH):
            continue
        random.shuffle(lengths)
        if lengths != sorted(lengths):
            return lengths
    return None


def _sample_canvas(total: int, branch: str) -> tuple[int, int] | None:
    if branch == "horizontal":
        width = random.randint(total, min(MAX_SIDE, total + 6))
        height_hi = min(MAX_SIDE, total - 1)
        if MIN_SIDE > height_hi:
            return None
        height = random.randint(MIN_SIDE, height_hi)
        if ((height + 1) // 2) * ((width + 1) // 2) >= total // 2 + 1:
            return height, width
        return None

    height = random.randint(total, min(MAX_SIDE, total + 6))
    width_hi = min(MAX_SIDE, total - 1)
    if MIN_SIDE > width_hi:
        return None
    width = random.randint(MIN_SIDE, width_hi)
    if ((height + 1) // 2) * ((width + 1) // 2) >= total // 2 + 1:
        return height, width
    return None


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> None:
    cursor = 0
    for length, color in zip(lengths, colors):
        for r, c in path[cursor : cursor + length]:
            grid[r, c] = int(color)
        cursor += length


def _expected_output(lengths: list[int], colors: list[int], branch: str) -> np.ndarray:
    segments = list(zip(lengths, colors))
    ordered = sorted(segments, key=lambda seg: seg[0])
    total = sum(lengths)
    if branch == "horizontal":
        out = np.empty((1, total), dtype=int)
        cursor = 0
        for length, color in ordered:
            out[0, cursor : cursor + length] = int(color)
            cursor += length
        return out

    out = np.empty((total, 1), dtype=int)
    cursor = 0
    for length, color in ordered:
        out[cursor : cursor + length, 0] = int(color)
        cursor += length
    return out


def _gen_attempt() -> np.ndarray | None:
    lengths = _sample_lengths()
    if lengths is None:
        return None

    total = sum(lengths)
    branch = random.choice(("horizontal", "vertical"))
    dims = _sample_canvas(total, branch)
    if dims is None:
        return None
    h, w = dims

    coarse_h = (h + 1) // 2
    coarse_w = (w + 1) // 2
    coarse_path = _random_coarse_walk(coarse_h, coarse_w, total // 2 + 1)
    if coarse_path is None:
        return None
    path = _expand_coarse_path(coarse_path, total)
    path = _translate_path(path, h, w)
    if _turn_count(path) == 0 and total >= 6:
        return None

    bg = random.choice(ARC_COLORS)
    colors = random.sample([color for color in ARC_COLORS if color != bg], len(lengths))

    grid = np.full((h, w), bg, dtype=int)
    _paint_path(grid, path, lengths, colors)

    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= max(lengths):
        return None

    expected = _expected_output(lengths, colors, branch)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
