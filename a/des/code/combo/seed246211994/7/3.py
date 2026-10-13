from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


MAROON = 9


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    blocked = exclude or set()
    return [int(value) for value in np.unique(grid) if int(value) not in blocked]


def _infer_direction(coords: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    rows = coords[:, 0]
    cols = coords[:, 1]

    if np.all(rows == rows[0]):
        step = (0, 1)
        order = np.argsort(cols)
    elif np.all(cols == cols[0]):
        step = (1, 0)
        order = np.argsort(rows)
    elif np.all(rows - cols == rows[0] - cols[0]):
        step = (1, 1)
        order = np.argsort(rows)
    elif np.all(rows + cols == rows[0] + cols[0]):
        step = (1, -1)
        order = np.argsort(rows)
    else:
        raise AssertionError("foreground color is not a straight 1-pixel line segment")

    ordered = [tuple(map(int, rc)) for rc in coords[order]]
    r, c = ordered[0]
    expected = []
    for _ in range(len(ordered)):
        expected.append((r, c))
        r += step[0]
        c += step[1]
    assert ordered == expected, "line segment must be contiguous without gaps"
    return step, ordered


def _full_line_cells(shape: tuple[int, int], anchor: tuple[int, int], step: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    height, width = shape
    dr, dc = step
    r, c = anchor
    while 0 <= r - dr < height and 0 <= c - dc < width:
        r -= dr
        c -= dc

    cells = []
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return tuple(cells)


def _extract_segments(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = get_bg(grid)
    segments: list[dict[str, object]] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        assert len(coords) >= 2, "every segment must contain at least two pixels"
        step, ordered = _infer_direction(coords)
        segments.append(
            {
                "color": int(color),
                "original": frozenset(ordered),
                "full": _full_line_cells(grid.shape, ordered[0], step),
            }
        )
    return bg, segments


def _paint_extensions(shape: tuple[int, int], bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    original_owner: dict[tuple[int, int], int] = {}
    extension_hits: dict[tuple[int, int], set[int]] = defaultdict(set)

    for segment in segments:
        color = int(segment["color"])
        original = segment["original"]
        for cell in original:
            assert cell not in original_owner, "original segments may not overlap"
            original_owner[cell] = color
            out[cell] = color

    for segment in segments:
        color = int(segment["color"])
        original = segment["original"]
        for cell in segment["full"]:
            if cell in original:
                continue
            extension_hits[cell].add(color)

    for cell, hit_colors in extension_hits.items():
        if cell in original_owner:
            continue
        out[cell] = MAROON if len(hit_colors) >= 2 else next(iter(hit_colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored line segment to the grid edges and mark extension-only crossings in maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, segments = _extract_segments(g)
    return _paint_extensions(g.shape, bg, segments)


MIN_SIZE = 9
MAX_SIZE = 18
MIN_SEGMENTS = 2
MAX_SEGMENTS = 5
MIN_FULL_LINE = 4
MIN_SEG_LEN = 2
MAX_SEG_LEN = 6
MAX_GENERATION_ATTEMPTS = 400
INPUT_COLORS = tuple(color for color in range(10) if color != MAROON)


def _enumerate_lines(shape: tuple[int, int]) -> list[dict[str, object]]:
    height, width = shape
    lines: list[dict[str, object]] = []

    for r in range(height):
        cells = tuple((r, c) for c in range(width))
        if len(cells) >= MIN_FULL_LINE:
            lines.append({"id": ("h", r), "kind": "h", "cells": cells, "index": {cell: i for i, cell in enumerate(cells)}})

    for c in range(width):
        cells = tuple((r, c) for r in range(height))
        if len(cells) >= MIN_FULL_LINE:
            lines.append({"id": ("v", c), "kind": "v", "cells": cells, "index": {cell: i for i, cell in enumerate(cells)}})

    for diag in range(-(width - 1), height):
        cells = tuple((r, r - diag) for r in range(height) if 0 <= r - diag < width)
        if len(cells) >= MIN_FULL_LINE:
            lines.append({"id": ("d1", diag), "kind": "d1", "cells": cells, "index": {cell: i for i, cell in enumerate(cells)}})

    for diag in range(height + width - 1):
        cells = tuple((r, diag - r) for r in range(height) if 0 <= diag - r < width)
        if len(cells) >= MIN_FULL_LINE:
            lines.append({"id": ("d2", diag), "kind": "d2", "cells": cells, "index": {cell: i for i, cell in enumerate(cells)}})

    return lines


def _segment_options(line: dict[str, object], color: int, banned_cells: set[tuple[int, int]]) -> list[dict[str, object]]:
    cells = line["cells"]
    index = line["index"]
    line_len = len(cells)
    banned_indices = {index[cell] for cell in banned_cells if cell in index}

    interior: list[dict[str, object]] = []
    edge: list[dict[str, object]] = []
    max_len = min(MAX_SEG_LEN, line_len - 1)
    for start in range(line_len):
        max_end = min(line_len - 1, start + max_len - 1)
        for end in range(start + MIN_SEG_LEN - 1, max_end + 1):
            if end - start + 1 >= line_len:
                continue
            if any(start <= idx <= end for idx in banned_indices):
                continue
            original = frozenset(cells[start : end + 1])
            segment = {
                "color": color,
                "original": original,
                "full": cells,
                "line_id": line["id"],
            }
            if start > 0 and end < line_len - 1:
                interior.append(segment)
            else:
                edge.append(segment)

    random.shuffle(interior)
    random.shuffle(edge)
    if interior and random.random() < 0.7:
        return interior + edge
    options = interior + edge
    random.shuffle(options)
    return options


def _render_input(shape: tuple[int, int], bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for segment in segments:
        color = int(segment["color"])
        for r, c in segment["original"]:
            assert grid[r, c] == bg, "segments may not overlap in the input"
            grid[r, c] = color
    return grid


def _place_segment(
    lines: list[dict[str, object]],
    used_line_ids: set[tuple[str, int]],
    occupied: set[tuple[int, int]],
    color: int,
    protected_cells: set[tuple[int, int]],
    shape: tuple[int, int],
    bg: int,
    segments: list[dict[str, object]],
    avoid_intersections: bool,
) -> dict[str, object] | None:
    for line in random.sample(lines, len(lines)):
        if line["id"] in used_line_ids:
            continue
        for candidate in _segment_options(line, color, protected_cells):
            if candidate["original"] & occupied:
                continue
            if avoid_intersections:
                trial = segments + [candidate]
                if np.any(_paint_extensions(shape, bg, trial) == MAROON):
                    continue
            return candidate
    return None


def _intersecting_pairs(lines: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object], tuple[int, int]]]:
    pairs = []
    for i, left in enumerate(lines):
        left_set = set(left["cells"])
        for right in lines[i + 1 :]:
            if left["kind"] == right["kind"]:
                continue
            common = left_set & set(right["cells"])
            if len(common) == 1:
                pairs.append((left, right, next(iter(common))))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(MIN_SIZE, MAX_SIZE), random.randint(MIN_SIZE, MAX_SIZE))
    bg = random.choice(INPUT_COLORS)
    line_colors = [color for color in INPUT_COLORS if color != bg]
    n_segments = random.randint(MIN_SEGMENTS, min(MAX_SEGMENTS, len(line_colors)))
    colors = random.sample(line_colors, n_segments)
    want_intersection = random.choice([False, True])

    lines = _enumerate_lines(shape)
    segments: list[dict[str, object]] = []
    used_line_ids: set[tuple[str, int]] = set()
    occupied: set[tuple[int, int]] = set()
    protected_cells: set[tuple[int, int]] = set()
    color_index = 0

    if want_intersection:
        pairs = _intersecting_pairs(lines)
        random.shuffle(pairs)
        for left, right, intersection in pairs:
            left_options = _segment_options(left, colors[color_index], {intersection})
            right_options = _segment_options(right, colors[color_index + 1], {intersection})
            if not left_options or not right_options:
                continue
            picked = None
            for left_segment in left_options[:12]:
                if left_segment["original"] & occupied:
                    continue
                for right_segment in right_options[:12]:
                    if right_segment["original"] & occupied:
                        continue
                    if left_segment["original"] & right_segment["original"]:
                        continue
                    picked = (left_segment, right_segment)
                    break
                if picked is not None:
                    break
            if picked is None:
                continue
            for segment in picked:
                segments.append(segment)
                used_line_ids.add(segment["line_id"])
                occupied |= set(segment["original"])
            protected_cells.add(intersection)
            color_index += 2
            break
        if len(segments) < 2:
            return None

    while color_index < n_segments:
        segment = _place_segment(
            lines,
            used_line_ids,
            occupied,
            colors[color_index],
            protected_cells,
            shape,
            bg,
            segments,
            avoid_intersections=not want_intersection,
        )
        if segment is None:
            return None
        segments.append(segment)
        used_line_ids.add(segment["line_id"])
        occupied |= set(segment["original"])
        color_index += 1

    grid = _render_input(shape, bg, segments)
    expected = _paint_extensions(shape, bg, segments)
    has_intersection = bool(np.any(expected == MAROON))
    if has_intersection != want_intersection:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
