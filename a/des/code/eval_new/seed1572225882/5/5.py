from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _clockwise_boundary(height: int, width: int):
    for col in range(width):
        yield 0, col, "top", col
    for row in range(1, height):
        yield row, width - 1, "right", row
    if height > 1:
        for col in range(width - 2, -1, -1):
            yield height - 1, col, "bottom", col
    if width > 1:
        for row in range(height - 2, 0, -1):
            yield row, 0, "left", row


def _edge_anchors(grid: np.ndarray, bg: int):
    height, width = grid.shape
    anchors = []
    for row, col, edge, step in _clockwise_boundary(height, width):
        color = int(grid[row, col])
        if color != bg:
            anchors.append((edge, step, color, row, col))
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """Split a multicolor object into color layers, shift each layer by its edge anchor, and recombine them.

    Input structure:
    The background is the most common color. A central composite pattern contains several interwoven non-background colors. The boundary contains one single-pixel anchor for each pattern color; the anchor color matches its layer. Anchors on the top, bottom, left, and right edges encode upward, downward, leftward, and rightward motion respectively.

    Rule:
    Remove all boundary anchors from the working grid. For every anchor encountered while scanning the grid border clockwise from the upper-left corner, isolate all non-anchor pixels of the same color, translate that whole color layer in the direction indicated by the anchor edge, using the anchor's row or column offset as the distance, and paint it onto a background grid. Later anchors in the clockwise scan order overwrite earlier translated pixels when layers overlap.

    Key insight:
    The border pixels are not noise: their colors identify object layers, their edge positions encode each layer's movement, and their clockwise order determines overlap priority.
    """
    grid = np.asarray(grid)
    bg = _background_color(grid)
    anchors = _edge_anchors(grid, bg)
    output = np.full_like(grid, bg)

    anchor_mask = np.zeros(grid.shape, dtype=bool)
    for _edge, _step, _color, row, col in anchors:
        anchor_mask[row, col] = True

    deltas = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    height, width = grid.shape
    for edge, step, color, _row, _col in anchors:
        coords = np.argwhere((grid == color) & ~anchor_mask)
        if coords.size == 0:
            continue
        dr, dc = deltas[edge]
        rows = coords[:, 0] + dr * step
        cols = coords[:, 1] + dc * step
        keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        output[rows[keep], cols[keep]] = color
    return output


def _choose_spaced_steps(limit: int, count: int):
    values = list(range(1, limit + 1))
    random.shuffle(values)
    chosen = []
    for value in values:
        if all(abs(value - other) >= 2 for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    for value in values:
        if value not in chosen:
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    return chosen


def _assign_pattern_cells(height: int, width: int, color_count: int):
    mode = random.randrange(3)
    offset = random.randrange(color_count)
    assigned = {}
    for row in range(height):
        for col in range(width):
            if mode == 0:
                use = ((row + col + offset) % 2 == 0) or random.random() < 0.18
                color_index = (row + 2 * col + offset) % color_count
            elif mode == 1:
                use = ((2 * row + col + offset) % 3 != 1) and random.random() < 0.78
                color_index = (2 * row + col + offset) % color_count
            else:
                center_dist = abs(row - height // 2) + abs(col - width // 2)
                use = ((center_dist + offset) % 2 == 0) or random.random() < 0.22
                color_index = (row // 2 + col + offset) % color_count
            if use and random.random() < 0.9:
                assigned[(row, col)] = color_index

    min_count = 3 if height * width >= color_count * 5 else 2
    counts = Counter(assigned.values())
    empty = [(r, c) for r in range(height) for c in range(width) if (r, c) not in assigned]
    random.shuffle(empty)
    for color_index in range(color_count):
        while counts[color_index] < min_count and empty:
            cell = empty.pop()
            assigned[cell] = color_index
            counts[color_index] += 1
    return assigned


def generate_input() -> np.ndarray:
    bg = 0
    edge_names = ["top", "right", "bottom", "left"]
    color_count = random.choice([4, 4, 5, 5, 6])
    edges = edge_names[:]
    while len(edges) < color_count:
        edges.append(random.choice(edge_names))
    random.shuffle(edges)

    edge_counts = Counter(edges)
    largest_edge_group = max(edge_counts.values())
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    max_shift_limit = min(7, (height - 8) // 2, (width - 8) // 2)
    max_shift = random.randint(max(2, largest_edge_group), max_shift_limit)
    margin = max_shift + 1

    inner_height = height - 2 * margin
    inner_width = width - 2 * margin
    min_side = 6 if color_count >= 5 else 5
    pattern_height = random.randint(min_side, min(10, inner_height))
    pattern_width = random.randint(min_side, min(10, inner_width))
    top = random.randint(margin, height - margin - pattern_height)
    left = random.randint(margin, width - margin - pattern_width)

    colors = random.sample(range(1, 10), color_count)
    grid = np.full((height, width), bg, dtype=int)

    pattern = _assign_pattern_cells(pattern_height, pattern_width, color_count)
    for (row, col), color_index in pattern.items():
        grid[top + row, left + col] = colors[color_index]

    steps_by_edge = {}
    for edge, count in edge_counts.items():
        steps_by_edge[edge] = _choose_spaced_steps(max_shift, count)

    used_per_edge = defaultdict(int)
    for color, edge in zip(colors, edges):
        step = steps_by_edge[edge][used_per_edge[edge]]
        used_per_edge[edge] += 1
        if edge == "top":
            grid[0, step] = color
        elif edge == "bottom":
            grid[height - 1, step] = color
        elif edge == "left":
            grid[step, 0] = color
        else:
            grid[step, width - 1] = color

    return grid
