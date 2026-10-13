from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
import random


def _is_corner_bracket(component: dict) -> bool:
    coords = {tuple(map(int, p)) for p in component["coords"]}
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or component["area"] != h + w - 1:
        return False

    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    return any(coords == candidate for candidate in candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every pixel whose color matches the marker inside the corner bracket, except for the marker itself.

    Input structure
    - The background is the most common color.
    - There is exactly one monochromatic 4-connected corner bracket object.
    - Inside the bracket bbox, outside the bracket cells, there is exactly one foreground marker pixel.
    Rule
    - Read the marker color.
    - Recolor every other pixel of that color to the background.
    - Leave the bracket, the marker, and all other colors unchanged.
    Key insight
    - The bracket is a color selector: its enclosed marker names the color to delete everywhere else.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, min_area=3)
    brackets = [component for component in components if _is_corner_bracket(component)]
    assert len(brackets) == 1, "expected exactly one corner bracket"

    bracket = brackets[0]
    bracket_coords = {tuple(map(int, p)) for p in bracket["coords"]}
    r0, c0, r1, c1 = bracket["bbox"]
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_coords and g[r, c] != bg
    ]
    assert len(markers) == 1, "expected one marker inside the bracket"

    marker = markers[0]
    marker_color = int(g[marker])
    assert marker_color != bg and marker_color != int(bracket["color"])

    out = g.copy()
    out[out == marker_color] = bg
    out[marker] = marker_color
    return out

# --------------- generator (o5x) ---------------
# Canvas and fixed selector geometry.
_GEN_H = 15
_GEN_W = 15
_GEN_SHAPE = (_GEN_H, _GEN_W)
_GEN_BG = 0
_GEN_COLORS = tuple(range(1, 10))
_GEN_BRACKET_COLOR = 5
_GEN_BRACKET_SIZE = 4
_GEN_BRACKET_EDGE = _GEN_BRACKET_SIZE - 1
_GEN_MARKER_POS = (1, 1)
_GEN_MARKER_COLORS = tuple(c for c in _GEN_COLORS if c != _GEN_BRACKET_COLOR)

# Scatter layout: sparse singletons plus a small number of adjacent pairs.
_GEN_SCATTER_TOTAL_RANGE = (16, 32)
_GEN_PAIR_COUNT_RANGE = (2, 6)
_GEN_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_PAIR_DIRS = ((1, 0), (0, 1))

# Scatter color multiplicities.
_GEN_MARKER_SCATTER_COUNT_RANGE = (3, 5)
_GEN_INCLUDE_BRACKET_COLOR_PROB = 0.18
_GEN_MAX_OTHER_COLOR_COUNT = 8

# Generation retry bound.
_GEN_MAX_ATTEMPTS = 300

_GEN_BRACKET_CELLS = frozenset(
    {(r, _GEN_BRACKET_EDGE) for r in range(_GEN_BRACKET_SIZE)}
    | {(_GEN_BRACKET_EDGE, c) for c in range(_GEN_BRACKET_SIZE)}
)
_GEN_SCATTER_CELLS = tuple(
    (r, c)
    for r in range(_GEN_H)
    for c in range(_GEN_W)
    if not (r < _GEN_BRACKET_SIZE and c < _GEN_BRACKET_SIZE)
)
_GEN_SCATTER_SET = frozenset(_GEN_SCATTER_CELLS)
_GEN_BRACKET_TOUCH_CELLS = frozenset(
    (r + dr, c + dc)
    for r, c in _GEN_BRACKET_CELLS
    for dr, dc in _GEN_DIRS4
    if 0 <= r + dr < _GEN_H
    and 0 <= c + dc < _GEN_W
    and (r + dr, c + dc) in _GEN_SCATTER_SET
)


def _gen_neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr, dc in _GEN_DIRS4
        if 0 <= r + dr < _GEN_H and 0 <= c + dc < _GEN_W
    ]


def _gen_is_clear_scatter_cell(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    return cell not in occupied and all(n not in occupied for n in _gen_neighbors4(cell))


def _gen_pair_candidates(
    occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates = []
    for a in _GEN_SCATTER_CELLS:
        ar, ac = a
        for dr, dc in _GEN_PAIR_DIRS:
            b = (ar + dr, ac + dc)
            if b not in _GEN_SCATTER_SET:
                continue
            if not _gen_is_clear_scatter_cell(a, occupied):
                continue
            if not _gen_is_clear_scatter_cell(b, occupied):
                continue
            candidates.append((a, b))
    return candidates


def _gen_single_candidates(
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [
        cell
        for cell in _GEN_SCATTER_CELLS
        if _gen_is_clear_scatter_cell(cell, occupied)
    ]


def _gen_scatter_cells(total_pixels: int) -> list[tuple[int, int]] | None:
    min_pairs, max_pairs = _GEN_PAIR_COUNT_RANGE
    pair_count = random.randint(min_pairs, min(max_pairs, total_pixels // 2))
    occupied: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []

    for _ in range(pair_count):
        candidates = _gen_pair_candidates(occupied)
        if not candidates:
            return None
        pair = random.choice(candidates)
        occupied.update(pair)
        cells.extend(pair)

    for _ in range(total_pixels - 2 * pair_count):
        candidates = _gen_single_candidates(occupied)
        if not candidates:
            return None
        cell = random.choice(candidates)
        occupied.add(cell)
        cells.append(cell)

    random.shuffle(cells)
    return cells


def _gen_color_counts(total_pixels: int, marker_color: int) -> dict[int, int]:
    marker_count = random.randint(*_GEN_MARKER_SCATTER_COUNT_RANGE)
    other_colors = [c for c in _GEN_COLORS if c not in (marker_color, _GEN_BRACKET_COLOR)]
    include_bracket_color = random.random() < _GEN_INCLUDE_BRACKET_COLOR_PROB

    counts = {marker_color: marker_count}
    for color in other_colors:
        counts[color] = 1
    if include_bracket_color:
        counts[_GEN_BRACKET_COLOR] = 1

    remaining = total_pixels - sum(counts.values())
    recipients = other_colors + ([_GEN_BRACKET_COLOR] if include_bracket_color else [])
    while remaining > 0:
        eligible = [color for color in recipients if counts[color] < _GEN_MAX_OTHER_COLOR_COUNT]
        color = random.choice(eligible)
        counts[color] += 1
        remaining -= 1
    return counts


def _gen_assign_colors(
    cells: list[tuple[int, int]],
    counts: dict[int, int],
) -> dict[tuple[int, int], int] | None:
    assignments: dict[tuple[int, int], int] = {}
    remaining_cells = list(cells)

    if _GEN_BRACKET_COLOR in counts:
        safe_cells = [
            cell for cell in remaining_cells if cell not in _GEN_BRACKET_TOUCH_CELLS
        ]
        if len(safe_cells) < counts[_GEN_BRACKET_COLOR]:
            return None
        chosen = set(random.sample(safe_cells, counts[_GEN_BRACKET_COLOR]))
        for cell in chosen:
            assignments[cell] = _GEN_BRACKET_COLOR
        remaining_cells = [cell for cell in remaining_cells if cell not in chosen]

    non_bracket_colors = [c for c in counts if c != _GEN_BRACKET_COLOR]
    for color in random.sample(non_bracket_colors, len(non_bracket_colors)):
        chosen = set(random.sample(remaining_cells, counts[color]))
        for cell in chosen:
            assignments[cell] = color
        remaining_cells = [cell for cell in remaining_cells if cell not in chosen]

    return assignments


def _gen_base_grid(marker_color: int) -> np.ndarray:
    grid = np.full(_GEN_SHAPE, _GEN_BG, dtype=int)
    for r, c in _GEN_BRACKET_CELLS:
        grid[r, c] = _GEN_BRACKET_COLOR
    grid[_GEN_MARKER_POS] = marker_color
    return grid


def _gen_expected_output(grid: np.ndarray, marker_color: int) -> np.ndarray:
    expected = grid.copy()
    expected[expected == marker_color] = _GEN_BG
    expected[_GEN_MARKER_POS] = marker_color
    return expected


def _gen_attempt() -> np.ndarray | None:
    marker_color = random.choice(_GEN_MARKER_COLORS)
    grid = _gen_base_grid(marker_color)
    total_pixels = random.randint(*_GEN_SCATTER_TOTAL_RANGE)
    counts = _gen_color_counts(total_pixels, marker_color)
    cells = _gen_scatter_cells(total_pixels)
    if cells is None:
        return None

    assignments = _gen_assign_colors(cells, counts)
    if assignments is None:
        return None
    for (r, c), color in assignments.items():
        grid[r, c] = color

    if not np.array_equal(main(grid), _gen_expected_output(grid, marker_color)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate valid input for 1e81d6f9")
