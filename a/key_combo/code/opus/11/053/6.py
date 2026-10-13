import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

CELL_SIZE = 3
GRID_SIZE = 11
DIVIDER_ROWS = (3, 7)
DIVIDER_COLS = (3, 7)
GEN_ATTEMPTS = 400

TEMPLATE_COLORS = (BLUE, RED)

# Profiles are chosen so the lower median varies across examples.
COUNT_PROFILES = (
    (1, 1, 2, 3, 4, 5, 6, 7),  # median 3
    (1, 2, 2, 3, 4, 5, 6, 7),  # median 3
    (1, 2, 3, 4, 5, 6, 6, 7),  # median 4
    (1, 2, 3, 4, 5, 6, 7, 7),  # median 4
    (1, 2, 3, 5, 6, 6, 7, 8),  # median 5
    (1, 2, 4, 5, 6, 6, 7, 8),  # median 5
    (2, 3, 4, 5, 6, 7, 7, 8),  # median 5
    (2, 3, 5, 6, 7, 7, 8, 8),  # median 6
)


def _uniform_full_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if np.all(grid[r] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) >= 2 and len(cols) >= 2:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one divider color with full rows and cols"
    return candidates[0]


def _cell_slices(divider_rows: list[int], divider_cols: list[int]) -> list[tuple[slice, slice]]:
    r0, r1 = map(int, divider_rows)
    c0, c1 = map(int, divider_cols)
    row_spans = [slice(0, r0), slice(r0 + 1, r1), slice(r1 + 1, None)]
    col_spans = [slice(0, c0), slice(c0 + 1, c1), slice(c1 + 1, None)]
    return [(rs, cs) for rs in row_spans for cs in col_spans]


def _lower_median(values: list[int]) -> int:
    ordered = sorted(map(int, values))
    return int(ordered[len(ordered) // 2 - 1])


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w == GRID_SIZE

    divider, divider_rows, divider_cols = _uniform_full_lines(g)
    assert divider == GREEN
    assert divider_rows == list(DIVIDER_ROWS)
    assert divider_cols == list(DIVIDER_COLS)

    row_sizes = [divider_rows[0], divider_rows[1] - divider_rows[0] - 1, h - divider_rows[1] - 1]
    col_sizes = [divider_cols[0], divider_cols[1] - divider_cols[0] - 1, w - divider_cols[1] - 1]
    assert row_sizes == [CELL_SIZE, CELL_SIZE, CELL_SIZE]
    assert col_sizes == [CELL_SIZE, CELL_SIZE, CELL_SIZE]

    cells = [g[rs, cs].copy() for rs, cs in _cell_slices(divider_rows, divider_cols)]
    template = cells[0]
    template_mask = template != BLACK
    template_colors = set(map(int, template[template_mask]))
    assert divider not in template_colors
    assert template_mask.shape == (CELL_SIZE, CELL_SIZE)
    assert int(template_mask.sum()) == 8
    assert template_colors == set(TEMPLATE_COLORS)

    counts: list[int] = []
    for idx, cell in enumerate(cells[1:], start=1):
        assert cell.shape == (CELL_SIZE, CELL_SIZE)
        assert not np.any(cell == divider), f"divider color may only appear on full separator lines, found in cell {idx}"
        visible = cell != BLACK
        assert np.all(visible <= template_mask), f"region {idx} contains a pixel outside the template mask"
        if np.any(visible):
            assert np.array_equal(cell[visible], template[visible]), f"region {idx} must keep original template colors"
        counts.append(int(visible.sum()))

    return {
        "grid": g,
        "divider": divider,
        "divider_rows": divider_rows,
        "divider_cols": divider_cols,
        "cells": cells,
        "template": template,
        "template_mask": template_mask,
        "counts": counts,
    }


def _solve_from_parsed(parsed: dict) -> np.ndarray:
    g = parsed["grid"]
    divider = int(parsed["divider"])
    template = parsed["template"]
    template_mask = parsed["template_mask"]
    counts = parsed["counts"]
    threshold = _lower_median(counts)

    out = g.copy()
    for (rs, cs), count in zip(_cell_slices(parsed["divider_rows"], parsed["divider_cols"])[1:], counts):
        cell = out[rs, cs].copy()
        if int(count) > threshold:
            missing = template_mask & (cell == BLACK)
            cell[missing] = divider
        else:
            cell[cell != BLACK] = BLACK
        out[rs, cs] = cell
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete only the template copies whose surviving pixel count beats the global median, and erase the rest.

    Input structure:
    - The grid is an 11x11 board where two full green separator rows and two full green separator columns split the board into a 3x3 array of 3x3 cells.
    - The top-left cell is the complete template: an 8-pixel red/blue pattern on black with exactly one black hole.
    - Each of the other eight cells contains a partial copy of that template formed only by deleting some template pixels back to black.

    Rule:
    - Detect the green divider lines and crop the nine 3x3 cells.
    - Use the top-left cell as the template and count, for each of the other eight cells, how many non-black non-divider pixels remain.
    - Compute the lower median of those eight counts.
    - If a cell's count is strictly greater than that median, fill every missing template pixel position in that cell with green while keeping any surviving red/blue pixels unchanged; otherwise erase the whole cell to black.
    - Keep the template cell and the divider lines unchanged.

    Key insight:
    - The decision is global rather than local: you must compare every partial copy's surviving count against the median count across all eight copies before knowing whether that cell is completed or erased.
    """
    parsed = _parse_grid(grid)
    return _solve_from_parsed(parsed)


def _is_asymmetric_template(template: np.ndarray) -> bool:
    variants = []
    for k in range(4):
        rot = np.rot90(template, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    unique = {
        tuple(map(int, variant.flatten()))
        for variant in variants
    }
    return len(unique) >= 4


def _sample_template() -> np.ndarray:
    for _ in range(200):
        template = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
        hole = random.randrange(CELL_SIZE * CELL_SIZE)
        positions = [(idx // CELL_SIZE, idx % CELL_SIZE) for idx in range(CELL_SIZE * CELL_SIZE) if idx != hole]
        red_count = random.randint(3, 5)
        colors = [RED] * red_count + [BLUE] * (len(positions) - red_count)
        random.shuffle(colors)
        for (r, c), color in zip(positions, colors):
            template[r, c] = int(color)
        if _is_asymmetric_template(template):
            return template
    raise RuntimeError("failed to sample an asymmetric 8-pixel template")


def _sample_visible_positions(template_mask: np.ndarray, count: int) -> set[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
    chosen = random.sample(coords, k=int(count))
    return set(chosen)


def _render_input(template: np.ndarray, counts: list[int], visible_sets: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[DIVIDER_ROWS[0], :] = GREEN
    grid[DIVIDER_ROWS[1], :] = GREEN
    grid[:, DIVIDER_COLS[0]] = GREEN
    grid[:, DIVIDER_COLS[1]] = GREEN

    slices = _cell_slices(list(DIVIDER_ROWS), list(DIVIDER_COLS))
    grid[slices[0][0], slices[0][1]] = template
    template_mask = template != BLACK

    for (rs, cs), count, visible_set in zip(slices[1:], counts, visible_sets):
        cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
        for r, c in visible_set:
            assert template_mask[r, c]
            cell[r, c] = int(template[r, c])
        assert int(np.count_nonzero(cell)) == int(count)
        grid[rs, cs] = cell

    return grid


def _validate_case(inp: np.ndarray) -> None:
    parsed = _parse_grid(inp)
    counts = parsed["counts"]
    ordered = sorted(counts)
    assert len(counts) == 8
    assert all(1 <= count <= 8 for count in counts)
    assert ordered[3] < ordered[4], "median threshold must be unique across the middle split"

    freq = Counter(counts)
    assert max(freq.values()) <= 2, "at most two regions may share the same residual count"

    threshold = _lower_median(counts)
    high = [count for count in counts if count > threshold]
    low = [count for count in counts if count <= threshold]
    assert high and low, "both completion and erase branches must appear"
    assert any(count < int(parsed["template_mask"].sum()) for count in high), "at least one completed region must gain new pixels"

    expected = _solve_from_parsed(parsed)
    actual = main(inp)
    assert np.array_equal(actual, expected)

    slices = _cell_slices(parsed["divider_rows"], parsed["divider_cols"])
    template_mask = parsed["template_mask"]
    for idx, ((rs, cs), count) in enumerate(zip(slices[1:], counts), start=1):
        in_cell = inp[rs, cs]
        out_cell = actual[rs, cs]
        if count > threshold:
            missing = template_mask & (in_cell == BLACK)
            assert np.all(out_cell[missing] == GREEN), f"completed cell {idx} must fill missing template pixels with green"
            assert np.array_equal(out_cell[in_cell != BLACK], in_cell[in_cell != BLACK]), f"completed cell {idx} must preserve surviving colors"
        else:
            assert np.all(out_cell == BLACK), f"erased cell {idx} must become all black"

    assert np.array_equal(actual[0:CELL_SIZE, 0:CELL_SIZE], parsed["template"])
    assert np.all(actual[DIVIDER_ROWS[0], :] == GREEN)
    assert np.all(actual[DIVIDER_ROWS[1], :] == GREEN)
    assert np.all(actual[:, DIVIDER_COLS[0]] == GREEN)
    assert np.all(actual[:, DIVIDER_COLS[1]] == GREEN)


def _gen_attempt() -> np.ndarray | None:
    template = _sample_template()
    counts = list(random.choice(COUNT_PROFILES))
    random.shuffle(counts)

    template_mask = template != BLACK
    visible_sets: list[set[tuple[int, int]]] = []
    for count in counts:
        visible_sets.append(_sample_visible_positions(template_mask, count))

    # Reject exact duplicate partial cells too often; they hide the count signal.
    cell_signatures = [tuple(sorted(visible)) for visible in visible_sets]
    if len(set(cell_signatures)) < 6:
        return None

    inp = _render_input(template, counts, visible_sets)
    try:
        _validate_case(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-template-completion example")
