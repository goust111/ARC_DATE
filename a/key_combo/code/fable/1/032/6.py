from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
SEED_INDICES = (0, 1, 3, 6)
GEN_RETRY_LIMIT = 500
WALK_RETRY_LIMIT = 200
MIN_COARSE_VERTICES = 10


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    height, width = shape
    for dr, dc in DIRS4:
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _trace_path(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    current = start

    while True:
        next_cells = [
            cell
            for cell in _neighbors4(mask.shape, current[0], current[1])
            if bool(mask[cell]) and cell != prev
        ]
        if prev is None:
            assert len(next_cells) == 1, "path endpoint must have degree 1"
        else:
            assert len(next_cells) <= 1, "path must not branch"
        if not next_cells:
            break
        prev, current = current, next_cells[0]
        path.append(current)

    return path


def _extract_path(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    bg = _get_bg(grid)
    mask = grid != bg
    assert bool(mask.any()), "expected a non-empty path"

    degrees: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        degree = sum(bool(mask[nbr]) for nbr in _neighbors4(mask.shape, cell[0], cell[1]))
        degrees[cell] = degree
        assert degree in (1, 2), "path must be a 1-pixel-wide open chain"
        if degree == 1:
            endpoints.append(cell)

    assert len(endpoints) == 2, "expected exactly two path endpoints"
    path = _trace_path(mask, endpoints[0])
    assert len(path) == int(mask.sum()), "all non-background cells must belong to one path"

    non_bg = [int(v) for v in grid[mask]]
    counts = Counter(non_bg)
    path_color, _count = counts.most_common(1)[0]
    return int(bg), int(path_color), path


def _find_seeded_order(grid: np.ndarray, path_color: int, path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int, int]]:
    assert len(path) > SEED_INDICES[-1], "path must be long enough to expose the four seeds"

    for ordered in (path, list(reversed(path))):
        marked = {
            idx
            for idx, (r, c) in enumerate(ordered)
            if int(grid[r, c]) != int(path_color)
        }
        if marked != set(SEED_INDICES):
            continue

        colors = [int(grid[ordered[idx]]) for idx in SEED_INDICES]
        if colors[0] != colors[3]:
            continue
        cycle = (colors[0], colors[1], colors[2])
        if len(set(cycle)) != 3:
            continue
        if any(color == int(path_color) for color in cycle):
            continue
        return ordered, cycle

    raise AssertionError("could not identify the seeded endpoint and 3-color cycle")


def _paint_triangular_cycle(
    out: np.ndarray,
    ordered_path: list[tuple[int, int]],
    cycle: tuple[int, int, int],
) -> np.ndarray:
    position = 0
    step = 1
    color_index = 0
    while position < len(ordered_path):
        r, c = ordered_path[position]
        out[r, c] = int(cycle[color_index % len(cycle)])
        color_index += 1
        position += step
        step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a 3-color cycle along the path's triangular-number arc-length positions.

    Input structure:
    - The background is the most common color.
    - All non-background pixels form one 1-pixel-wide, 4-connected, open path with two endpoints.
    - Near one endpoint, the path cells at arc-length positions 0, 1, 3, and 6 are recolored as
      four seeds, with the fourth seed repeating the first.

    Rule:
    - Identify the endpoint whose ordered path begins with the four seeded triangular positions.
    - Read the three-color cycle from those seeds in order.
    - Recolor every path cell at triangular-number positions 0, 1, 3, 6, 10, 15, ... from that
      endpoint using the continuing cycle; leave the rest of the path and the background unchanged.

    Key insight:
    - Both the active positions and the color sequence are defined along path arc length, not by
      straight-line distance; the repeated fourth seed reveals the cycle length of 3.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, path_color, path = _extract_path(g)
    ordered_path, cycle = _find_seeded_order(g, path_color, path)

    out = g.copy()
    return _paint_triangular_cycle(out, ordered_path, cycle)


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    directions = [
        (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        for idx in range(len(path) - 1)
    ]
    return sum(directions[idx] != directions[idx - 1] for idx in range(1, len(directions)))


def _grow_coarse_walk(
    height: int,
    width: int,
    target_vertices: int,
    min_turns: int,
) -> list[tuple[int, int]] | None:
    for _ in range(WALK_RETRY_LIMIT):
        start = (random.randrange(height), random.randrange(width))
        path = [start]
        visited = {start}
        prev_dir: tuple[int, int] | None = None

        while len(path) < target_vertices:
            r, c = path[-1]
            options: list[tuple[float, tuple[int, int], tuple[int, int], int]] = []
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                cell = (nr, nc)
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if cell in visited:
                    continue

                visited.add(cell)
                onward = 0
                for rr, cc in _neighbors4((height, width), nr, nc):
                    if (rr, cc) not in visited:
                        onward += 1
                visited.remove(cell)

                turn_bonus = 1.7 if prev_dir is not None and (dr, dc) != prev_dir else 0.4
                edge_dist = min(nr, nc, height - 1 - nr, width - 1 - nc)
                score = 2.5 * onward + turn_bonus + 0.15 * edge_dist + random.random()
                options.append((score, cell, (dr, dc), onward))

            if not options:
                break

            remaining_vertices = target_vertices - len(path)
            if remaining_vertices > 1:
                non_dead = [item for item in options if item[3] > 0]
                if non_dead:
                    options = non_dead

            options.sort(key=lambda item: item[0], reverse=True)
            _score, cell, step_dir, _onward = random.choice(options[: min(3, len(options))])
            visited.add(cell)
            path.append(cell)
            prev_dir = step_dir

        if len(path) == target_vertices and _count_turns(path) >= min_turns:
            return path

    return None


def _expand_path(coarse_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    fine_path: list[tuple[int, int]] = []
    for idx, (r, c) in enumerate(coarse_path):
        vertex = (2 * int(r), 2 * int(c))
        if idx == 0:
            fine_path.append(vertex)
            continue
        pr, pc = coarse_path[idx - 1]
        midpoint = (int(pr) + int(r), int(pc) + int(c))
        fine_path.append(midpoint)
        fine_path.append(vertex)
    return fine_path


def _normalize_path(path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rows = [r for r, _c in path]
    cols = [c for _r, c in path]
    r0 = min(rows)
    c0 = min(cols)
    normalized = [(r - r0, c - c0) for r, c in path]
    height = max(r for r, _c in normalized) + 1
    width = max(c for _r, c in normalized) + 1
    return normalized, int(height), int(width)


def _generate_attempt() -> np.ndarray | None:
    coarse_h = random.randint(5, 11)
    coarse_w = random.randint(5, 11)
    max_vertices = min(coarse_h * coarse_w // 2 + 2, 20)
    if max_vertices < MIN_COARSE_VERTICES:
        return None

    target_vertices = random.randint(MIN_COARSE_VERTICES, max_vertices)
    min_turns = random.randint(4, min(8, target_vertices - 2))
    coarse_path = _grow_coarse_walk(coarse_h, coarse_w, target_vertices, min_turns)
    if coarse_path is None:
        return None

    fine_path, fine_h, fine_w = _normalize_path(_expand_path(coarse_path))
    canvas_h = random.randint(fine_h, min(30, fine_h + 6))
    canvas_w = random.randint(fine_w, min(30, fine_w + 6))
    top = random.randint(0, canvas_h - fine_h)
    left = random.randint(0, canvas_w - fine_w)
    placed_path = [(r + top, c + left) for r, c in fine_path]

    bg, path_color, seed_a, seed_b, seed_c = random.sample(PALETTE, 5)
    grid = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    for r, c in placed_path:
        grid[r, c] = int(path_color)

    ordered = placed_path if random.random() < 0.5 else list(reversed(placed_path))
    seed_colors = (seed_a, seed_b, seed_c, seed_a)
    for idx, color in zip(SEED_INDICES, seed_colors):
        r, c = ordered[idx]
        grid[r, c] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid triangular-cycle path instance")
