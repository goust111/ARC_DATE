from __future__ import annotations

import random

import numpy as np


GEN_BG = 8
GEN_MARKER = 9
GEN_SEGMENT_COLORS = tuple(color for color in range(10) if color not in (GEN_BG, GEN_MARKER))
GEN_GROUP_COUNT_RANGE = (2, 4)
GEN_SEGMENTS_PER_GROUP_RANGE = (2, 4)
GEN_LENGTH_CHOICES = (2, 3, 4, 5, 6)
GEN_PLACEMENT_ATTEMPTS = 500
GEN_MAX_ATTEMPTS = 300


def _color_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) >= 1
    if len(order) >= 2:
        assert int(counts[order[0]]) > int(counts[order[1]]), "expected a unique dominant background"
    return int(values[order[0]])


def _line_component_info(coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    length = int(coords.shape[0])

    if np.all(rows == rows[0]):
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert length == c1 - c0 + 1, "horizontal segment must be contiguous"
        cells = [(row, c) for c in range(c0, c1 + 1)]
        return {
            "orientation": "h",
            "length": length,
            "cells": cells,
            "endpoints": ((row, c0), (row, c1)),
        }

    if np.all(cols == cols[0]):
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert length == r1 - r0 + 1, "vertical segment must be contiguous"
        cells = [(r, col) for r in range(r0, r1 + 1)]
        return {
            "orientation": "v",
            "length": length,
            "cells": cells,
            "endpoints": ((r0, col), (r1, col)),
        }

    raise AssertionError("foreground components must be straight line segments")


def _segment_with_marker(info: dict, marker_positions: set[tuple[int, int]]) -> dict:
    (r0, c0), (r1, c1) = info["endpoints"]
    choices: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    if info["orientation"] == "h":
        left_marker = (r0, c0 - 1)
        right_marker = (r1, c1 + 1)
        if left_marker in marker_positions:
            choices.append((left_marker, (r0, c0), (r1, c1), (0, 1)))
        if right_marker in marker_positions:
            choices.append((right_marker, (r1, c1), (r0, c0), (0, -1)))
    else:
        top_marker = (r0 - 1, c0)
        bottom_marker = (r1 + 1, c1)
        if top_marker in marker_positions:
            choices.append((top_marker, (r0, c0), (r1, c1), (1, 0)))
        if bottom_marker in marker_positions:
            choices.append((bottom_marker, (r1, c1), (r0, c0), (-1, 0)))

    assert len(choices) == 1, "each segment must have exactly one head marker"
    marker, head, tail, step = choices[0]
    return {
        **info,
        "marker": marker,
        "head": head,
        "tail": tail,
        "step": step,
    }


def _parse_groups(grid: np.ndarray, bg: int, marker_color: int) -> dict[int, list[dict]]:
    marker_components = _color_components(grid == marker_color)
    assert marker_components, "expected at least one marker"
    assert all(component.shape[0] == 1 for component in marker_components), "markers must be single pixels"
    marker_positions = {tuple(map(int, component[0])) for component in marker_components}

    groups: dict[int, list[dict]] = {}
    used_markers: set[tuple[int, int]] = set()

    colors = [int(color) for color in np.unique(grid) if int(color) not in (bg, marker_color)]
    assert colors, "expected at least one segment color"

    for color in colors:
        components = _color_components(grid == color)
        assert 2 <= len(components) <= 4, "each color must appear in two to four segments"
        segments: list[dict] = []
        for component in components:
            info = _line_component_info(component)
            assert 2 <= int(info["length"]) <= 6, "segment lengths must lie in 2..6"
            segment = _segment_with_marker(info, marker_positions)
            assert segment["marker"] not in used_markers, "markers must belong to unique segments"
            used_markers.add(segment["marker"])
            segment["color"] = color
            segments.append(segment)

        segments.sort(
            key=lambda seg: (
                int(seg["length"]),
                int(seg["head"][0]),
                int(seg["head"][1]),
                int(seg["tail"][0]),
                int(seg["tail"][1]),
            )
        )
        groups[color] = segments

    assert used_markers == marker_positions, "every marker must annotate exactly one segment"
    return groups


def _infer_groups(grid: np.ndarray) -> dict[int, list[dict]]:
    bg = _dominant_color(grid)
    foreground = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(foreground) >= 2, "expected marker color plus at least one segment color"

    singleton_colors = []
    for color in foreground:
        components = _color_components(grid == color)
        if components and all(component.shape[0] == 1 for component in components):
            singleton_colors.append(color)

    if GEN_MARKER in singleton_colors:
        singleton_colors.remove(GEN_MARKER)
        singleton_colors.insert(0, GEN_MARKER)

    for marker_color in singleton_colors:
        try:
            return _parse_groups(grid, bg, marker_color)
        except AssertionError:
            continue

    raise AssertionError("failed to identify the marker color and segment groups")


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize each color-group path by sorting segment groups into a single output color column.

    Input structure:
    - The background is a dominant uniform color.
    - Foreground contains several monochrome horizontal or vertical line segments,
      each two to six cells long and one cell thick.
    - Every segment has one adjacent single-pixel marker placed one step beyond
      exactly one endpoint on the segment's extension line; colors group segments.

    Rule:
    - Identify the marker color and use each marker to determine the head endpoint
      of its neighboring segment.
    - Group segments by their own color, then sort each group from shortest to
      longest segment, using the visible head positions only as a stable tie-break.
    - Treat that order as the concatenation order of one abstract path per color
      group and compute each path's total length as the sum of its segment lengths.
    - Return an N x 1 grid whose rows are the group colors sorted from largest
      total path length to smallest; markers disappear in the output.

    Key insight:
    - The isolated marker pixels encode direction at the segment level, while the
      segment lengths encode the within-color ordering, so the final answer is a
      length-ranked color summary rather than a reconstructed drawing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    groups = _infer_groups(g)
    paths: list[tuple[int, int, tuple[int, int]]] = []
    for color, segments in groups.items():
        total_length = sum(int(segment["length"]) for segment in segments)
        start_head = tuple(map(int, segments[0]["head"]))
        paths.append((int(color), total_length, start_head))

    paths.sort(key=lambda item: (-int(item[1]), int(item[2][0]), int(item[2][1])))
    return np.array([[color] for color, _total, _start in paths], dtype=int)


def _random_group_lengths() -> list[int]:
    count = random.randint(*GEN_SEGMENTS_PER_GROUP_RANGE)
    lengths = random.sample(GEN_LENGTH_CHOICES, k=count)
    lengths.sort()
    return lengths


def _choose_group_specs() -> list[dict]:
    for _ in range(200):
        group_count = random.randint(*GEN_GROUP_COUNT_RANGE)
        colors = random.sample(GEN_SEGMENT_COLORS, k=group_count)
        groups = []
        totals = set()
        for color in colors:
            lengths = _random_group_lengths()
            total = sum(lengths)
            groups.append({"color": color, "lengths": lengths, "total": total})
            totals.add(total)
        if len(totals) == group_count:
            return groups
    raise RuntimeError("failed to sample unique total lengths")


def _object_cells(
    orientation: str,
    head_side: str,
    length: int,
    row: int,
    col: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if orientation == "h":
        segment_cells = [(row, col + offset) for offset in range(length)]
        if head_side == "left":
            marker = (row, col - 1)
        else:
            marker = (row, col + length)
        return segment_cells, marker

    segment_cells = [(row + offset, col) for offset in range(length)]
    if head_side == "up":
        marker = (row - 1, col)
    else:
        marker = (row + length, col)
    return segment_cells, marker


def _can_place(mask: np.ndarray, occupied: list[tuple[int, int]]) -> bool:
    height, width = mask.shape
    for r, c in occupied:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if mask[r, c]:
            return False

    for r, c in occupied:
        r0 = max(0, r - 1)
        r1 = min(height, r + 2)
        c0 = max(0, c - 1)
        c1 = min(width, c + 2)
        if np.any(mask[r0:r1, c0:c1]):
            return False

    return True


def _place_segment(
    grid: np.ndarray,
    occupied_mask: np.ndarray,
    color: int,
    length: int,
    orientation: str,
    head_side: str,
) -> bool:
    height, width = grid.shape
    for _ in range(GEN_PLACEMENT_ATTEMPTS):
        if orientation == "h":
            row = random.randint(0, height - 1)
            if head_side == "left":
                if width < length + 1:
                    return False
                col = random.randint(1, width - length)
            else:
                if width < length + 1:
                    return False
                col = random.randint(0, width - length - 1)
        else:
            col = random.randint(0, width - 1)
            if head_side == "up":
                if height < length + 1:
                    return False
                row = random.randint(1, height - length)
            else:
                if height < length + 1:
                    return False
                row = random.randint(0, height - length - 1)

        segment_cells, marker = _object_cells(orientation, head_side, length, row, col)
        all_cells = segment_cells + [marker]
        if not _can_place(occupied_mask, all_cells):
            continue

        for rr, cc in segment_cells:
            grid[rr, cc] = color
            occupied_mask[rr, cc] = True
        mr, mc = marker
        grid[mr, mc] = GEN_MARKER
        occupied_mask[mr, mc] = True
        return True

    return False


def _grid_shape_for(groups: list[dict]) -> tuple[int, int]:
    object_count = sum(len(group["lengths"]) for group in groups)
    min_dim = max(12, int(np.ceil(np.sqrt(object_count * 32))))
    if min_dim > 30:
        min_dim = 30
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)
    return height, width


def _build_attempt() -> np.ndarray | None:
    groups = _choose_group_specs()
    height, width = _grid_shape_for(groups)
    grid = np.full((height, width), GEN_BG, dtype=int)
    occupied_mask = np.zeros((height, width), dtype=bool)

    specs: list[tuple[int, int, str, str]] = []
    for group in groups:
        local_specs = []
        for length in group["lengths"]:
            orientation = random.choice(("h", "v"))
            if orientation == "h":
                head_side = random.choice(("left", "right"))
            else:
                head_side = random.choice(("up", "down"))
            local_specs.append((group["color"], length, orientation, head_side))

        if len(local_specs) >= 2 and all(spec[2] == local_specs[0][2] for spec in local_specs):
            color, length, orientation, head_side = local_specs[0]
            flipped = "v" if orientation == "h" else "h"
            flipped_head = random.choice(("up", "down")) if flipped == "v" else random.choice(("left", "right"))
            local_specs[0] = (color, length, flipped, flipped_head)

        specs.extend(local_specs)

    random.shuffle(specs)
    for color, length, orientation, head_side in specs:
        if not _place_segment(grid, occupied_mask, color, length, orientation, head_side):
            return None

    expected = np.array(
        [[group["color"]] for group in sorted(groups, key=lambda item: -int(item["total"]))],
        dtype=int,
    )

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
