from __future__ import annotations

import random

import numpy as np


BG = 0
GAP_COLOR = 9
SIDE_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
GRID_MIN = 12
GRID_MAX = 30
FRAME_MIN = 7
GEN_RETRIES = 300


def _dominant_color(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid, dtype=int).ravel()).argmax())


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != bg)
    assert coords.size, "expected a non-background frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _side_color(values: np.ndarray, bg: int, gap_color: int) -> int:
    colors = {int(v) for v in np.unique(values) if int(v) not in (bg, gap_color)}
    assert len(colors) == 1, "each side must have exactly one non-gap color"
    return int(next(iter(colors)))


def _find_window(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    r0, c0, r1, c1 = _non_bg_bbox(g, bg)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a nonempty interior"
    assert bool((g[r0, c0 : c1 + 1] != bg).all())
    assert bool((g[r1, c0 : c1 + 1] != bg).all())
    assert bool((g[r0 + 1 : r1, c0] != bg).all())
    assert bool((g[r0 + 1 : r1, c1] != bg).all())

    side_sets = [
        {int(v) for v in np.unique(g[r0, c0 : c1 + 1]) if int(v) != bg},
        {int(v) for v in np.unique(g[r1, c0 : c1 + 1]) if int(v) != bg},
        {int(v) for v in np.unique(g[r0 + 1 : r1, c0]) if int(v) != bg},
        {int(v) for v in np.unique(g[r0 + 1 : r1, c1]) if int(v) != bg},
    ]
    common = side_sets[0] & side_sets[1] & side_sets[2] & side_sets[3]
    assert len(common) == 1, "expected one shared gap-marker color on all four sides"
    gap_color = int(next(iter(common)))

    top = _side_color(g[r0, c0 : c1 + 1], bg, gap_color)
    bottom = _side_color(g[r1, c0 : c1 + 1], bg, gap_color)
    left = _side_color(g[r0 + 1 : r1, c0], bg, gap_color)
    right = _side_color(g[r0 + 1 : r1, c1], bg, gap_color)
    assert len({top, bottom, left, right}) == 4, "window sides must use four distinct colors"
    assert bool((g[r0 + 1 : r1, c0 + 1 : c1] == bg).all()), "window interior must start empty"
    return (r0, c0, r1, c1), gap_color


def _apply_rays(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap_color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = bbox

    for r in range(r0 + 1, r1):
        if int(out[r, c0]) != gap_color:
            out[r, c0 + 1 : c1] = int(out[r, c0])
        if int(out[r, c1]) != gap_color:
            out[r, c0 + 1 : c1] = int(out[r, c1])

    for c in range(c0 + 1, c1):
        if int(out[r0, c]) != gap_color:
            out[r0 + 1 : r1, c] = int(out[r0, c])
        if int(out[r1, c]) != gap_color:
            out[r0 + 1 : r1, c] = int(out[r1, c])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of a colored window by projecting side-colored stripes through its gap-marked border.

    Input structure:
    - The background is black.
    - One large rectangular window sits inside the grid, with a one-pixel border.
    - The top, bottom, left, and right border sides use four distinct colors.
    - Some border pixels are replaced by a shared maroon gap-marker color; the corners belong to the horizontal sides.

    Rule:
    - Find the rectangular border and identify the shared gap-marker color.
    - Every non-gap pixel on the left or right side projects a horizontal stripe of that side's color across the full window interior row.
    - Every non-gap pixel on the top or bottom side projects a vertical stripe of that side's color across the full window interior column, overriding any horizontal stripe at crossings.
    - Gap-marker pixels do not project, and the border itself stays unchanged.

    Key insight:
    - The border is both legend and emitter: each side color defines a stripe direction, while the maroon gaps suppress specific rows or columns so the final interior is a priority-ordered stripe grid.
    """
    g = np.asarray(grid, dtype=int)
    bbox, gap_color = _find_window(g)
    return _apply_rays(g, bbox, gap_color)


def _sample_three_way_labels(length: int) -> list[int]:
    assert length >= 3, "need room for top/bottom/none or left/right/none"
    cuts = sorted(random.sample(range(1, length), 2))
    counts = [cuts[0], cuts[1] - cuts[0], length - cuts[1]]
    labels: list[int] = []
    for label, count in enumerate(counts):
        labels.extend([label] * int(count))
    random.shuffle(labels)
    return labels


def _centered_start(total: int, size: int) -> int:
    lo = 1
    hi = total - size - 1
    assert lo <= hi
    center = (total - size) // 2
    return min(max(center + random.randint(-2, 2), lo), hi)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)

        frame_h = random.randint(max(FRAME_MIN, height // 2), height - 2)
        frame_w = random.randint(max(FRAME_MIN, width // 2), width - 2)
        inner_h = frame_h - 2
        inner_w = frame_w - 2
        if inner_h < 3 or inner_w < 3:
            continue

        r0 = _centered_start(height, frame_h)
        c0 = _centered_start(width, frame_w)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        top_color, bottom_color, left_color, right_color = random.sample(SIDE_PALETTE, 4)
        col_labels = _sample_three_way_labels(inner_w)
        row_labels = _sample_three_way_labels(inner_h)

        grid = np.full((height, width), BG, dtype=int)
        grid[r0, c0 : c1 + 1] = top_color
        grid[r1, c0 : c1 + 1] = bottom_color
        grid[r0 + 1 : r1, c0] = left_color
        grid[r0 + 1 : r1, c1] = right_color

        for offset, label in enumerate(col_labels, start=1):
            c = c0 + offset
            grid[r0, c] = top_color if label == 0 else GAP_COLOR
            grid[r1, c] = bottom_color if label == 1 else GAP_COLOR

        for offset, label in enumerate(row_labels, start=1):
            r = r0 + offset
            grid[r, c0] = left_color if label == 0 else GAP_COLOR
            grid[r, c1] = right_color if label == 1 else GAP_COLOR

        expected = _apply_rays(grid, (r0, c0, r1, c1), GAP_COLOR)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        inner = actual[r0 + 1 : r1, c0 + 1 : c1]
        present = {int(v) for v in np.unique(inner)}
        if {top_color, bottom_color, left_color, right_color, BG} <= present:
            return grid

    raise RuntimeError("failed to generate a valid window instance")
