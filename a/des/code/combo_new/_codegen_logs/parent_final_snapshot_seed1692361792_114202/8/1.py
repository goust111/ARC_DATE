import itertools
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
OFFSETS: tuple[Coord, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
STEP_CHOICES: tuple[Coord, ...] = ((0, 2), (2, 0), (1, 1))
OBJECT_COUNT_RANGE = (2, 5)
PADDING_RANGE = (0, 6)
GENERATION_RETRIES = 500


def _cross_cells(center: Coord) -> tuple[Coord, ...]:
    r, c = center
    return tuple((r + dr, c + dc) for dr, dc in OFFSETS)


def _in_bounds(shape: tuple[int, int], cells: Iterable[Coord]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _candidate_centers(coords: np.ndarray, shape: tuple[int, int]) -> list[Coord]:
    visible = {tuple(map(int, coord)) for coord in coords}
    assert visible, "every color must remain visible"

    candidates: set[Coord] = set()
    for vr, vc in visible:
        for dr, dc in OFFSETS:
            center = (vr - dr, vc - dc)
            cells = _cross_cells(center)
            if not _in_bounds(shape, cells):
                continue
            if visible.issubset(cells):
                candidates.add(center)
    return sorted(candidates)


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int] | None:
    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indegree[dst] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        zeros = [color for color in colors if color in remaining and indegree[color] == 0]
        if len(zeros) != 1:
            return None
        color = zeros[0]
        remaining.remove(color)
        order.append(color)
        for dst in edges[color]:
            indegree[dst] -= 1
    return order


def _render_solution(shape: tuple[int, int], centers: dict[int, Coord], order: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color in order:
        for r, c in _cross_cells(centers[color]):
            out[r, c] = int(color)
    return out


def _validate_assignment(grid: np.ndarray, colors: list[int], centers: dict[int, Coord]) -> list[int] | None:
    coverers: dict[Coord, list[int]] = {}
    for color in colors:
        for cell in _cross_cells(centers[color]):
            coverers.setdefault(cell, []).append(color)

    edges = {color: set() for color in colors}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            obs = int(grid[r, c])
            cell = (r, c)
            if obs == 0:
                if cell in coverers:
                    return None
                continue
            if cell not in coverers:
                return None
            if obs not in coverers[cell]:
                return None
            for lower in coverers[cell]:
                if lower != obs:
                    edges[lower].add(obs)

    order = _unique_topological_order(colors, edges)
    if order is None:
        return None
    if not np.array_equal(_render_solution(grid.shape, centers, order), grid):
        return None
    return order


def _solve_instance(grid: np.ndarray) -> tuple[dict[int, Coord], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert colors, "expected at least one foreground color"

    candidates_by_color = {color: _candidate_centers(np.argwhere(g == color), g.shape) for color in colors}
    assert all(candidates_by_color[color] for color in colors), "each color must admit at least one center"

    search_colors = sorted(colors, key=lambda color: (len(candidates_by_color[color]), color))
    solutions: list[tuple[dict[int, Coord], list[int]]] = []
    for chosen_centers in itertools.product(*(candidates_by_color[color] for color in search_colors)):
        centers = {color: center for color, center in zip(search_colors, chosen_centers)}
        order = _validate_assignment(g, colors, centers)
        if order is None:
            continue
        solutions.append((centers, order))
        if len(solutions) > 1:
            break

    assert len(solutions) == 1, "instance must have a unique center assignment and layer order"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the layer order of overlapping cross fragments, then redraw the full crosses in that order.

    Input structure:
    - The background is black.
    - Each non-black color belongs to exactly one 5-pixel plus shape: a 3x3 box with the four corners removed.
    - Several such plus shapes overlap on the same canvas, so lower colors may be visible only as fragments.

    Rule:
    - For each color, enumerate the plus-template centers that could explain its visible pixels.
    - Keep only the global center assignment whose covered cells exactly explain the input and whose occlusion constraints give a unique bottom-to-top order.
    - Starting from a black canvas of the same size, redraw every full plus from the lowest layer to the highest layer so later colors overwrite earlier ones.

    Key insight:
    - A fragment cannot be completed locally in isolation: its missing cells are determined by which other colors sit on its plus template, so center recovery and layer-order recovery must be solved together.
    """
    g = np.asarray(grid, dtype=int)
    centers, order = _solve_instance(g)
    return _render_solution(g.shape, centers, order)


def _transform_point(point: Coord, rot_k: int, flip: bool) -> Coord:
    r, c = point
    if flip:
        c = -c
    for _ in range(rot_k % 4):
        r, c = c, -r
    return int(r), int(c)


def _make_relative_chain(count: int) -> list[Coord]:
    centers = [(0, 0)]
    for _ in range(count - 1):
        dr, dc = random.choice(STEP_CHOICES)
        pr, pc = centers[-1]
        centers.append((pr + dr, pc + dc))
    return centers


def _place_centers(relative_centers: list[Coord]) -> tuple[tuple[int, int], list[Coord]]:
    rot_k = random.randrange(4)
    flip = random.choice((False, True))
    transformed = [_transform_point(point, rot_k, flip) for point in relative_centers]

    min_r = min(r for r, _c in transformed)
    max_r = max(r for r, _c in transformed)
    min_c = min(c for _r, c in transformed)
    max_c = max(c for _r, c in transformed)

    cluster_h = max_r - min_r + 3
    cluster_w = max_c - min_c + 3

    pad_top = random.randint(*PADDING_RANGE)
    pad_bottom = random.randint(*PADDING_RANGE)
    pad_left = random.randint(*PADDING_RANGE)
    pad_right = random.randint(*PADDING_RANGE)

    h = cluster_h + pad_top + pad_bottom
    w = cluster_w + pad_left + pad_right
    if h > 30 or w > 30:
        raise ValueError("generated canvas exceeds ARC size limit")

    shift_r = pad_top + 1 - min_r
    shift_c = pad_left + 1 - min_c
    centers = [(r + shift_r, c + shift_c) for r, c in transformed]
    return (h, w), centers


def _visible_count(grid: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(grid == color))


def _generate_case() -> tuple[np.ndarray, dict[int, Coord], list[int]]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    order = random.sample(PALETTE, count)
    relative_centers = _make_relative_chain(count)
    shape, centers_in_order = _place_centers(relative_centers)

    centers = {color: center for color, center in zip(order, centers_in_order)}
    grid = _render_solution(shape, centers, order)
    assert max(grid.shape) <= 30
    assert any(_visible_count(grid, color) < 5 for color in order[:-1]), "at least one cross should be occluded"
    return grid, centers, order


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            grid, expected_centers, expected_order = _generate_case()
            found_centers, found_order = _solve_instance(grid)
        except (AssertionError, ValueError):
            continue

        if found_order != expected_order:
            continue
        if any(found_centers[color] != expected_centers[color] for color in expected_order):
            continue
        return grid
    raise RuntimeError("failed to generate a valid overlapping-cross instance")
