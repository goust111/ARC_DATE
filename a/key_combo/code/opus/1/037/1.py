from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
PIVOT = 5
ORANGE = 7

MARKER_PALETTE = (BLUE, GREEN, ORANGE)
GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (14, 20)
THREE_MARKER_PROB = 0.65
EDGE_PIVOT_PROB = 0.45

Coord = tuple[int, int]
Marker = tuple[int, int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _reflect_marker(marker: Marker, pivot: Coord) -> Marker:
    r, c, color = marker
    pr, pc = pivot
    return 2 * pr - r, 2 * pc - c, color


def _scene_from_grid(grid: np.ndarray) -> tuple[Coord, list[Marker]]:
    pivot_coords = np.argwhere(grid == PIVOT)
    assert pivot_coords.shape == (1, 2), "expected exactly one gray pivot pixel"
    pivot = tuple(map(int, pivot_coords[0]))

    marker_coords = np.argwhere((grid != BG) & (grid != PIVOT))
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in marker_coords]
    assert 2 <= len(markers) <= 3, "expected two or three colored marker pixels"
    assert len({color for _r, _c, color in markers}) == len(markers), "marker colors must be distinct"
    return pivot, sorted(markers)


def _validate_layout(shape: tuple[int, int], pivot: Coord, markers: list[Marker]) -> list[Marker]:
    pr, pc = pivot
    reflected = [_reflect_marker(marker, pivot) for marker in markers]
    seeds = markers + reflected

    for r, c, _color in markers:
        assert r != pr and c != pc, "original marker must not share pivot row or column"
    for r, c, _color in seeds:
        assert _in_bounds(shape, r, c), "all original and reflected markers must stay in bounds"

    rows = [r for r, _c, _color in seeds]
    cols = [c for _r, c, _color in seeds]
    assert len(rows) == len(set(rows)), "all original/reflected markers must have distinct rows"
    assert len(cols) == len(set(cols)), "all original/reflected markers must have distinct columns"
    return reflected


def _render_from_scene(shape: tuple[int, int], pivot: Coord, markers: list[Marker]) -> np.ndarray:
    reflected = _validate_layout(shape, pivot, markers)
    seeds = markers + reflected

    out = np.full(shape, BG, dtype=int)
    out[pivot] = PIVOT
    for r, c, color in markers:
        out[r, c] = color
    for r, c, color in reflected:
        out[r, c] = color

    for r, _c, color in seeds:
        out[r, :] = color
    for _r, c, color in seeds:
        out[:, c] = color

    for marker, mirror in zip(markers, reflected):
        out[marker[0], mirror[1]] = RED
        out[mirror[0], marker[1]] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored marker through the gray pivot, cast full row-and-column rays from originals and reflections, and recolor only the paired intersections red.

    Input structure:
    - A black background contains one gray pivot pixel and two or three colored single-pixel markers.
    - Marker colors are distinct, and no original or reflected marker shares a row or column with any other marker.
    - Every reflected marker position given by 180-degree symmetry through the pivot stays inside the grid.

    Rule:
    - Reflect each colored marker through the pivot to create a same-color partner at the opposite point.
    - From every original and reflected marker, paint its entire row in that marker's color, then paint its entire column in that marker's color so vertical lines overwrite horizontal ones.
    - After all lines are drawn, recolor only the two off-marker rectangle corners formed by each original marker and its own reflection to red.

    Key insight:
    - The pivot does not emit lines; it only pairs each marker with an unseen symmetric partner, and only those partner-specific row/column crossings become red while all other line intersections keep their line color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    pivot, markers = _scene_from_grid(g)
    reflected = _validate_layout(g.shape, pivot, markers)
    seeds = markers + reflected

    out = g.copy()
    for r, c, color in reflected:
        out[r, c] = color
    for r, _c, color in seeds:
        out[r, :] = color
    for _r, c, color in seeds:
        out[:, c] = color
    for marker, mirror in zip(markers, reflected):
        out[marker[0], mirror[1]] = RED
        out[mirror[0], marker[1]] = RED
    return out


def _choose_marker_count() -> int:
    return 3 if random.random() < THREE_MARKER_PROB else 2


def _choose_pivot_coord(limit: int, marker_count: int) -> int:
    low = marker_count
    high = limit - 1 - marker_count
    assert low <= high, "grid must have room for all symmetric distances"

    if random.random() < EDGE_PIVOT_PROB:
        options = {low, min(high, low + 1), max(low, high - 1), high}
        return random.choice(sorted(options))
    return random.randint(low, high)


def _sample_distances(cap: int, marker_count: int) -> list[int]:
    assert cap >= marker_count
    return random.sample(range(1, cap + 1), marker_count)


def _build_markers(shape: tuple[int, int], pivot: Coord, colors: list[int]) -> list[Marker]:
    pr, pc = pivot
    row_cap = min(pr, shape[0] - 1 - pr)
    col_cap = min(pc, shape[1] - 1 - pc)
    marker_count = len(colors)

    row_dists = _sample_distances(row_cap, marker_count)
    col_dists = _sample_distances(col_cap, marker_count)
    random.shuffle(row_dists)
    random.shuffle(col_dists)

    markers: list[Marker] = []
    for color, row_dist, col_dist in zip(colors, row_dists, col_dists):
        row_sign = random.choice((-1, 1))
        col_sign = random.choice((-1, 1))
        r = pr + row_sign * row_dist
        c = pc + col_sign * col_dist
        markers.append((r, c, color))

    return markers


def _scene_has_distractors(markers: list[Marker], reflected: list[Marker]) -> bool:
    red_cells = set()
    for marker, mirror in zip(markers, reflected):
        red_cells.add((marker[0], mirror[1]))
        red_cells.add((mirror[0], marker[1]))

    distractors = set()
    for i, marker in enumerate(markers):
        for j, mirror in enumerate(reflected):
            if i == j:
                continue
            distractors.add((marker[0], mirror[1]))
            distractors.add((mirror[0], marker[1]))

    return bool(distractors - red_cells)


def _validate_generated_scene(shape: tuple[int, int], pivot: Coord, markers: list[Marker]) -> None:
    reflected = _validate_layout(shape, pivot, markers)
    seeds = {(r, c) for r, c, _color in markers + reflected}
    red_cells = {(marker[0], mirror[1]) for marker, mirror in zip(markers, reflected)}
    red_cells |= {(mirror[0], marker[1]) for marker, mirror in zip(markers, reflected)}

    assert len(red_cells) == 2 * len(markers), "each original-reflection pair must contribute two unique red cells"
    assert pivot not in seeds and pivot not in red_cells, "pivot must remain structurally distinct"
    assert seeds.isdisjoint(red_cells), "red cells must be different from marker locations"
    assert _scene_has_distractors(markers, reflected), "scene must contain non-red line intersections from unpaired markers"


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    marker_count = _choose_marker_count()
    height = random.randint(max(GRID_HEIGHT_RANGE[0], 2 * marker_count + 3), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(GRID_WIDTH_RANGE[0], 2 * marker_count + 3), GRID_WIDTH_RANGE[1])
    pivot = (
        _choose_pivot_coord(height, marker_count),
        _choose_pivot_coord(width, marker_count),
    )

    colors = random.sample(MARKER_PALETTE, marker_count)
    markers = _build_markers((height, width), pivot, colors)
    _validate_generated_scene((height, width), pivot, markers)

    inp = np.full((height, width), BG, dtype=int)
    inp[pivot] = PIVOT
    for r, c, color in markers:
        inp[r, c] = color

    expected = _render_from_scene(inp.shape, pivot, markers)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected = _make_case()
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid pivot-reflection ray scene")
