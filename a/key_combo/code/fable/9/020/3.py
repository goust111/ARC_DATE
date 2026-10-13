from __future__ import annotations

from collections import Counter
import random

import numpy as np


CELL_SIZE = 3
STRIDE = 4
BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ROW_OPTIONS = range(2, 6)
COL_OPTIONS = range(2, 6)
MAX_GENERATION_ATTEMPTS = 400
DEMO_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
NON_CENTER_COORDS = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE) if (r, c) != (1, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    for idx in dividers:
        assert start < idx
        parts.append((start, idx))
        start = idx + 1
    assert start < length
    parts.append((start, length))
    return parts


def _divider_grid(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    h, w = grid.shape
    for color in _colors_present(grid, exclude={bg}):
        rows = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
        cols = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1
    return candidates[0]


def _follow_chain(mapping: dict[int, int], start: int) -> int:
    seen: set[int] = set()
    cur = int(start)
    while cur in mapping:
        assert cur not in seen, "mapping chain must be acyclic"
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    divider, divider_rows, divider_cols = _divider_grid(g, bg)
    row_segments = _segments(g.shape[0], divider_rows)
    col_segments = _segments(g.shape[1], divider_cols)

    assert all(r1 - r0 == CELL_SIZE for r0, r1 in row_segments)
    assert all(c1 - c0 == CELL_SIZE for c0, c1 in col_segments)

    mapping: dict[int, int] = {}
    cells: list[dict] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            cell = g[r0:r1, c0:c1]
            assert cell.shape == (CELL_SIZE, CELL_SIZE)
            assert not bool(np.any(cell == divider))

            marker = int(cell[1, 1])
            assert marker not in {bg, divider}

            others = cell.copy()
            others[1, 1] = bg
            other_colors = _colors_present(others, exclude={bg})
            assert len(other_colors) <= 1

            demo = None
            if other_colors:
                demo = int(other_colors[0])
                assert demo not in {marker, divider}
                if marker in mapping:
                    assert mapping[marker] == demo
                else:
                    mapping[marker] = demo

            cells.append(
                {
                    "bounds": (r0, r1, c0, c1),
                    "marker": marker,
                    "demo": demo,
                }
            )

    return {
        "bg": bg,
        "divider": divider,
        "row_segments": row_segments,
        "col_segments": col_segments,
        "cells": cells,
        "mapping": mapping,
    }


def _expected_output_from_analysis(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mapping = analysis["mapping"]
    for cell in analysis["cells"]:
        r0, r1, c0, c1 = cell["bounds"]
        out[r0:r1, c0:c1] = _follow_chain(mapping, cell["marker"])
    return out


def _one_step_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mapping = analysis["mapping"]
    for cell in analysis["cells"]:
        r0, r1, c0, c1 = cell["bounds"]
        out[r0:r1, c0:c1] = mapping.get(cell["marker"], cell["marker"])
    return out


def _local_demo_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for cell in analysis["cells"]:
        r0, r1, c0, c1 = cell["bounds"]
        out[r0:r1, c0:c1] = cell["demo"] if cell["demo"] is not None else cell["marker"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Read in-window relay examples, then fill each window with its marker color's terminal relay color.

    Input structure:
    - Grey full rows and columns split the canvas into equal 3x3 windows.
    - Every window has exactly one colored marker at its center.
    - Some windows also contain one or more non-center demo pixels, all in a single color.

    Rule:
    - Treat each demo window as one directed pair: center marker color -> demo color.
    - Merge all such pairs into one in-grid mapping table; the pairs are consistent and acyclic.
    - For every window, start from its center marker color and repeatedly follow the mapping
      until reaching a color with no outgoing pair, then recolor the full 3x3 window to that terminal color.
    - Keep all separator rows and columns unchanged.

    Key insight:
    - The visible examples only teach one relay step at a time, but the required output is the
      final color reached after chasing the whole chain; even demo windows themselves are recolored
      to that terminal endpoint.
    """
    analysis = _analyze_grid(grid)
    return _expected_output_from_analysis(grid, analysis)


def _sample_mapping_graph() -> tuple[list[int], dict[int, int], list[int]]:
    n_colors = random.randint(4, 6)
    colors = random.sample(PALETTE, n_colors)

    primary_len = random.randint(3, n_colors - 1)
    primary_chain = colors[:primary_len]
    extras = colors[primary_len:]
    assert extras, "need at least one secondary terminal"

    mapping = {primary_chain[i]: primary_chain[i + 1] for i in range(primary_len - 1)}
    secondary_terminal = extras[0]
    for color in extras[1:]:
        if random.random() < 0.7:
            mapping[color] = random.choice(primary_chain[1:] + [secondary_terminal])

    return colors, mapping, primary_chain


def _choose_dims(required_cells: int) -> tuple[int, int]:
    feasible = [
        (rows, cols)
        for rows in ROW_OPTIONS
        for cols in COL_OPTIONS
        if rows * cols >= required_cells and (STRIDE * rows - 1) <= 30 and (STRIDE * cols - 1) <= 30
    ]
    assert feasible
    return random.choice(feasible)


def _build_cell_specs(
    colors: list[int],
    mapping: dict[int, int],
    primary_chain: list[int],
    capacity: int,
) -> list[tuple[int, int | None]]:
    specs: list[tuple[int, int | None]] = [(color, None) for color in colors]
    specs.extend((start, target) for start, target in mapping.items())

    required = len(specs)
    assert capacity >= required
    extra_count = capacity - required
    weighted_markers = primary_chain[:-1] + colors + colors
    for _ in range(extra_count):
        marker = random.choice(weighted_markers)
        if marker in mapping and random.random() < 0.45:
            specs.append((marker, mapping[marker]))
        else:
            specs.append((marker, None))

    random.shuffle(specs)
    return specs


def _make_base_grid(rows: int, cols: int) -> np.ndarray:
    h = STRIDE * rows - 1
    w = STRIDE * cols - 1
    grid = np.full((h, w), BG, dtype=int)
    grid[CELL_SIZE::STRIDE, :] = SEP
    grid[:, CELL_SIZE::STRIDE] = SEP
    return grid


def _render_demo_pixels(cell: np.ndarray, color: int) -> None:
    count = random.choice(DEMO_COUNT_CHOICES)
    for r, c in random.sample(NON_CENTER_COORDS, count):
        cell[r, c] = int(color)


def _render_input(rows: int, cols: int, specs: list[tuple[int, int | None]]) -> np.ndarray:
    grid = _make_base_grid(rows, cols)
    assert len(specs) <= rows * cols

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * STRIDE
            c0 = cc * STRIDE
            cell = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            marker, demo = specs[idx]
            idx += 1
            cell[1, 1] = int(marker)
            if demo is not None:
                _render_demo_pixels(cell, int(demo))
    return grid


def _validate_instance(
    grid: np.ndarray,
    planned_colors: list[int],
    planned_mapping: dict[int, int],
    primary_chain: list[int],
) -> bool:
    analysis = _analyze_grid(grid)
    actual_mapping = analysis["mapping"]
    cells = analysis["cells"]

    if actual_mapping != planned_mapping:
        return False
    if not all(any(cell["marker"] == color for cell in cells) for color in planned_colors):
        return False

    head = primary_chain[0]
    if not any(cell["marker"] == head and cell["demo"] is None for cell in cells):
        return False
    if not any(cell["marker"] == head and cell["demo"] == planned_mapping[head] for cell in cells):
        return False

    terminals = {_follow_chain(actual_mapping, color) for color in planned_colors}
    if len(terminals) < 2:
        return False
    if _follow_chain(actual_mapping, head) == actual_mapping[head]:
        return False

    expected = _expected_output_from_analysis(grid, analysis)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, _one_step_output(grid, analysis)):
        return False
    if np.array_equal(expected, _local_demo_output(grid, analysis)):
        return False

    fill_colors = _colors_present(expected, exclude={analysis["divider"]})
    if len(fill_colors) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    colors, mapping, primary_chain = _sample_mapping_graph()
    required_cells = len(colors) + len(mapping)
    rows, cols = _choose_dims(required_cells)
    specs = _build_cell_specs(colors, mapping, primary_chain, rows * cols)
    grid = _render_input(rows, cols, specs)
    if _validate_instance(grid, colors, mapping, primary_chain):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
