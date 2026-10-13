from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np


GRID_SHAPE = (8, 8)
BG_COLOR = 0
PIVOT_COLOR = 5
MARKER_PALETTE = (1, 2, 3, 4)
CORNER_COORDS = {
    "top_left": (0, 0),
    "top_right": (0, 7),
    "bottom_left": (7, 0),
    "bottom_right": (7, 7),
}
CORNER_TEMPLATES = {
    "top_left": ((1, 1), (1, 2), (2, 1), (2, 2)),
    "top_right": ((1, 5), (1, 6), (2, 5), (2, 6)),
    "bottom_left": ((6, 0), (6, 1), (5, 0)),
    "bottom_right": ((6, 7), (6, 6), (5, 7)),
}
INTERIOR_CELLS = tuple((r, c) for r in range(1, 7) for c in range(1, 7))
MARKER_COUNT_CHOICES = (2, 3)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < GRID_SHAPE[0] and 0 <= c < GRID_SHAPE[1]


def _reflect_cell(cell: tuple[int, int], pivot: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    pr, pc = pivot
    return 2 * pr - r, 2 * pc - c


def _normalized_marker_items(markers: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(markers.items()))


def _template_reflection(corner_name: str, pivot: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(_reflect_cell(cell, pivot) for cell in CORNER_TEMPLATES[corner_name])


def _structure_is_valid(corner_names: tuple[str, ...], pivot: tuple[int, int]) -> bool:
    used: set[tuple[int, int]] = set()
    for corner_name in corner_names:
        for reflected in _template_reflection(corner_name, pivot):
            if not _in_bounds(*reflected):
                return False
            if reflected == pivot:
                return False
            if reflected in used:
                return False
            used.add(reflected)
    return True


def _render_input(markers: dict[str, int], pivot: tuple[int, int]) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    pr, pc = pivot
    grid[pr, pc] = PIVOT_COLOR
    for corner_name, color in markers.items():
        r, c = CORNER_COORDS[corner_name]
        grid[r, c] = int(color)
    return grid


def _build_output(markers: dict[str, int], pivot: tuple[int, int]) -> np.ndarray:
    out = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    pr, pc = pivot
    out[pr, pc] = PIVOT_COLOR

    for corner_name, color in markers.items():
        for rr, cc in _template_reflection(corner_name, pivot):
            if not _in_bounds(rr, cc):
                continue
            if (rr, cc) == pivot:
                continue
            existing = int(out[rr, cc])
            assert existing in (BG_COLOR, color), "reflected templates must not conflict"
            out[rr, cc] = int(color)

    return out


def _parse_input(grid: np.ndarray) -> tuple[tuple[int, int], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == GRID_SHAPE, "expected the fixed 8x8 task grid"
    assert np.all((0 <= g) & (g <= 9)), "colors must be ARC ids"

    pivot_cells = np.argwhere(g == PIVOT_COLOR)
    assert len(pivot_cells) == 1, "expected exactly one grey pivot"
    pivot = tuple(map(int, pivot_cells[0]))
    assert pivot in INTERIOR_CELLS, "pivot must be strictly inside the grid"

    markers: dict[str, int] = {}
    allowed_foreground = {pivot}
    for corner_name, coord in CORNER_COORDS.items():
        color = int(g[coord])
        if color == BG_COLOR:
            continue
        assert color in MARKER_PALETTE, "corner markers must use colors 1..4"
        markers[corner_name] = color
        allowed_foreground.add(coord)

    assert len(markers) in MARKER_COUNT_CHOICES, "expected 2 or 3 corner markers"
    assert len(set(markers.values())) == len(markers), "corner marker colors must be distinct"

    foreground = {tuple(map(int, rc)) for rc in np.argwhere(g != BG_COLOR)}
    assert foreground == allowed_foreground, "only the pivot and corner markers may be non-black"
    return pivot, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate corner-selected inward templates, then relocate them by a 180-degree turn around the grey pivot.

    Input structure:
    - The grid is an 8x8 black canvas with exactly one grey pivot pixel strictly inside the border.
    - Exactly 2 or 3 corners contain one-pixel markers whose distinct colors come from blue, red, green, and yellow.
    - No other foreground objects are present.

    Rule:
    - Use each marked corner to choose its fixed inward template and paint that template with the marker color.
    - Erase the original corner markers.
    - Reflect every template pixel through the grey pivot with the mapping (r, c) -> (2*pr-r, 2*pc-c), discarding any out-of-bounds reflections and keeping the pivot grey.
    - The output contains only the reflected templates plus the unchanged grey pivot.

    Key insight:
    - The corner decides which template shape exists, while the pivot decides where that generated template lands after the half-turn, so both cues are required.
    """
    pivot, markers = _parse_input(grid)
    return _build_output(markers, pivot)


def _spec_has_necessary_dependencies(
    pivot: tuple[int, int], marker_items: tuple[tuple[str, int], ...]
) -> bool:
    markers = dict(marker_items)
    base_output = _build_output(markers, pivot)

    for corner_name in markers:
        reduced = {name: color for name, color in markers.items() if name != corner_name}
        if np.array_equal(base_output, _build_output(reduced, pivot)):
            return False

    corner_names = tuple(name for name, _ in marker_items)
    for alt_pivot in INTERIOR_CELLS:
        if alt_pivot == pivot or not _structure_is_valid(corner_names, alt_pivot):
            continue
        if np.array_equal(base_output, _build_output(markers, alt_pivot)):
            return False

    return True


def _enumerate_candidate_groups() -> dict[int, dict[tuple[str, ...], list[tuple[tuple[int, int], tuple[tuple[str, int], ...]]]]]:
    specs: list[tuple[tuple[int, int], tuple[tuple[str, int], ...]]] = []
    outputs_to_specs: defaultdict[bytes, list[tuple[tuple[int, int], tuple[tuple[str, int], ...]]]] = defaultdict(list)

    for marker_count in MARKER_COUNT_CHOICES:
        for corner_names in itertools.combinations(CORNER_COORDS.keys(), marker_count):
            for pivot in INTERIOR_CELLS:
                if not _structure_is_valid(corner_names, pivot):
                    continue
                for colors in itertools.permutations(MARKER_PALETTE, marker_count):
                    markers = dict(zip(corner_names, colors))
                    marker_items = _normalized_marker_items(markers)
                    input_grid = _render_input(markers, pivot)
                    output_grid = _build_output(markers, pivot)

                    assert not np.array_equal(input_grid, output_grid), "task must change the grid"
                    assert np.array_equal(main(input_grid), output_grid), "generator and solver disagree"

                    spec = (pivot, marker_items)
                    specs.append(spec)
                    outputs_to_specs[output_grid.tobytes()].append(spec)

    grouped: dict[int, dict[tuple[str, ...], list[tuple[tuple[int, int], tuple[tuple[str, int], ...]]]]] = {
        2: defaultdict(list),
        3: defaultdict(list),
    }
    for pivot, marker_items in specs:
        markers = dict(marker_items)
        output_sig = _build_output(markers, pivot).tobytes()
        if len(outputs_to_specs[output_sig]) != 1:
            continue
        if not _spec_has_necessary_dependencies(pivot, marker_items):
            continue
        corner_key = tuple(name for name, _ in marker_items)
        grouped[len(marker_items)][corner_key].append((pivot, marker_items))

    assert grouped[2] and grouped[3], "expected valid 2-marker and 3-marker samples"
    return grouped


_CANDIDATE_GROUPS = _enumerate_candidate_groups()


def generate_input() -> np.ndarray:
    marker_count = random.choice(MARKER_COUNT_CHOICES)
    group_key = random.choice(tuple(_CANDIDATE_GROUPS[marker_count].keys()))
    pivot, marker_items = random.choice(_CANDIDATE_GROUPS[marker_count][group_key])
    return _render_input(dict(marker_items), pivot)
