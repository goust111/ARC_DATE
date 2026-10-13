from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
READ_ORDER = (
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (0, 0),
    (0, 1),
    (0, 2),
)
DRAW_DIRS = (
    (0, -1),   # left
    (1, 0),    # down
    (0, 1),    # right
    (-1, 0),   # up
)
GEN_ATTEMPTS = 400


def _full_color_rows(grid: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]


def _full_color_cols(grid: np.ndarray, color: int) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _intervals(size: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(int(x) for x in separators), int(size)]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 1:
            raise AssertionError("separator lines collapse a region")
        spans.append((a + 1, b))
    return spans


def _compress_left_partition(left: np.ndarray) -> np.ndarray:
    row_seps = _full_color_rows(left, GREY)
    col_seps = _full_color_cols(left, GREY)
    assert len(row_seps) == 2, "expected exactly two horizontal separator rows"
    assert len(col_seps) == 2, "expected exactly two vertical separator columns"

    row_spans = _intervals(left.shape[0], row_seps)
    col_spans = _intervals(left.shape[1], col_seps)
    assert len(row_spans) == 3 and len(col_spans) == 3

    cmap = np.zeros((3, 3), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            block = left[r0:r1, c0:c1]
            colors = np.unique(block)
            assert len(colors) == 1, "each region must compress to one color"
            cmap[i, j] = int(colors[0])

    assert len(set(map(int, cmap.flat))) == 9, "the 3x3 color map must use nine distinct colors"
    return cmap


def _spiral_read(cmap: np.ndarray) -> list[int]:
    assert cmap.shape == (3, 3)
    return [int(cmap[r, c]) for r, c in READ_ORDER]


def _draw_spiral(
    shape: tuple[int, int],
    colors: list[int],
    *,
    dirs: tuple[tuple[int, int], ...] = DRAW_DIRS,
    return_trace: bool = False,
):
    h, w = map(int, shape)
    assert h % 2 == 1 and w % 2 == 1, "canvas center must be unique"
    assert colors, "spiral colors must be non-empty"

    out = np.zeros((h, w), dtype=int)
    center = (h // 2, w // 2)
    out[center] = int(colors[0])

    trace = [{"color_index": 0, "cells": (center,)}]
    pos = center
    seg_len = 1
    dir_idx = 0
    seg_no = 1

    while True:
        color_index = seg_no % len(colors)
        color = int(colors[color_index])
        dr, dc = dirs[dir_idx % len(dirs)]
        cells = []
        for step in range(seg_len):
            nr = pos[0] + dr
            nc = pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                trace.append({"color_index": color_index, "cells": tuple(cells)})
                if return_trace:
                    return out, trace, {
                        "segment_no": seg_no,
                        "direction_index": dir_idx,
                        "step_in_segment": step,
                        "partial_segment": bool(cells),
                    }
                return out
            out[nr, nc] = color
            cells.append((nr, nc))
            pos = (nr, nc)

        trace.append({"color_index": color_index, "cells": tuple(cells)})
        dir_idx += 1
        if dir_idx % 2 == 0:
            seg_len += 1
        seg_no += 1


def _parse(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_cols = _full_color_cols(g, GREY)
    assert grey_cols, "expected at least one full-height separator column"
    sep_col = max(grey_cols)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.size and right.size, "expected non-empty left partition and right canvas"
    assert right.shape[0] % 2 == 1 and right.shape[1] % 2 == 1, "right canvas must have a unique center"

    cmap = _compress_left_partition(left)
    center = (right.shape[0] // 2, right.shape[1] // 2)
    assert int(right[center]) == int(cmap[1, 1]), "center marker must match the color-map center"
    assert np.count_nonzero(right) == 1, "right canvas should contain only the center marker before drawing"
    return cmap, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left partition grid into a 3x3 color map and use its spiral readout to color a truncated spiral on the right canvas.

    Input structure:
    - The grid is split by a full-height grey separator column into a left partition block and a right black canvas.
    - The left block is a 3x3 array of uniform color regions separated by two grey rows and two grey columns.
    - The right block is an odd-sized black canvas containing exactly one colored center marker.

    Rule:
    - Compress each left-side region to one pixel, yielding a 3x3 color map.
    - Read that color map in clockwise spiral order from the center: center, right, down, left, left, up, up, right, right.
    - On the right canvas, keep the center marker as the first color, then draw a counterclockwise spiral with directions left, down, right, up and lengths 1, 1, 2, 2, 3, 3, ... using the color sequence cyclically.
    - If a segment would leave the canvas, draw only the in-bounds prefix of that segment and stop immediately.

    Key insight:
    - The left side turns spatial arrangement into a time-ordered color sequence, and the right side turns that sequence back into space via a growing spiral.
    """
    cmap, right = _parse(grid)
    colors = _spiral_read(cmap)
    return _draw_spiral(right.shape, colors)


def _build_left_grid(cell_size: int, cmap: np.ndarray) -> np.ndarray:
    side = 3 * cell_size + 2
    left = np.zeros((side, side), dtype=int)
    left[cell_size, :] = GREY
    left[2 * cell_size + 1, :] = GREY
    left[:, cell_size] = GREY
    left[:, 2 * cell_size + 1] = GREY

    for i in range(3):
        for j in range(3):
            r0 = i * (cell_size + 1)
            c0 = j * (cell_size + 1)
            left[r0 : r0 + cell_size, c0 : c0 + cell_size] = int(cmap[i, j])
    return left


def _build_input(cell_size: int, right_width: int, cmap: np.ndarray) -> np.ndarray:
    left = _build_left_grid(cell_size, cmap)
    h = left.shape[0]
    grid = np.zeros((h, h + 1 + right_width), dtype=int)
    grid[:, :h] = left
    grid[:, h] = GREY
    grid[h // 2, h + 1 + right_width // 2] = int(cmap[1, 1])
    return grid


def _alt_row_major_output(shape: tuple[int, int], cmap: np.ndarray) -> np.ndarray:
    return _draw_spiral(shape, [int(x) for x in cmap.flatten()])


def _alt_clockwise_output(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    clockwise = (
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
    )
    return _draw_spiral(shape, colors, dirs=clockwise)


def _used_color_indices(trace: list[dict]) -> set[int]:
    used = set()
    for item in trace:
        if item["cells"]:
            used.add(int(item["color_index"]))
    return used


def _candidate_ok(grid: np.ndarray) -> bool:
    out = main(grid)
    cmap, right = _parse(grid)
    colors = _spiral_read(cmap)
    expected, trace, stop = _draw_spiral(right.shape, colors, return_trace=True)
    if not np.array_equal(out, expected):
        return False

    if out.shape != right.shape or np.count_nonzero(out) <= 1:
        return False

    used_indices = _used_color_indices(trace)
    if used_indices != set(range(9)):
        return False

    nonempty_segments = sum(1 for item in trace[1:] if item["cells"])
    if nonempty_segments < 12:
        return False

    if not stop["partial_segment"]:
        return False

    center = (out.shape[0] // 2, out.shape[1] // 2)
    if int(out[center]) != int(cmap[1, 1]):
        return False

    if np.array_equal(out, _alt_row_major_output(out.shape, cmap)):
        return False
    if np.array_equal(out, _alt_clockwise_output(out.shape, colors)):
        return False

    swapped = cmap.copy()
    swapped[0, 0], swapped[0, 1] = swapped[0, 1], swapped[0, 0]
    if np.array_equal(out, _draw_spiral(out.shape, _spiral_read(swapped))):
        return False

    only_center = np.zeros_like(out)
    only_center[center] = int(cmap[1, 1])
    if np.array_equal(out, only_center):
        return False

    tall = out.shape[0] > out.shape[1]
    stopped_vertical = stop["direction_index"] % 4 in (1, 3)
    if tall == stopped_vertical:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        cell_size = random.choice((3, 5))
        side = 3 * cell_size + 2

        max_width = 30 - side - 1
        width_choices = [w for w in range(9, max_width + 1, 2)]
        assert width_choices

        branch = random.choice(("wide", "tall"))
        if branch == "wide":
            preferred = [w for w in width_choices if w > side]
            if preferred:
                width_choices = preferred
        else:
            preferred = [w for w in width_choices if w < side]
            if preferred:
                width_choices = preferred
        right_width = random.choice(width_choices)

        cmap = np.array(random.sample(range(1, 10), 9), dtype=int).reshape(3, 3)
        grid = _build_input(cell_size, right_width, cmap)
        if _candidate_ok(grid):
            return grid

    raise RuntimeError("failed to generate a valid spiral-compression ARC input")
