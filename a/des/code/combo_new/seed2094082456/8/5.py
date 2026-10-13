from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(1, 10))
SEGMENT_COUNT_RANGE = (1, 6)
SEGMENT_LENGTH_RANGE = (2, 6)
MAX_SHAPE_SPAN = 26
MAX_GRID_SPAN = 30
EXTRA_MARGIN_RANGE = (0, 4)
GENERATION_ATTEMPTS = 500


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _monochrome_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(g.shape, rr, cc):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _find_start_block(g: np.ndarray) -> dict:
    candidates = []
    for comp in _monochrome_components(g, BG):
        if int(comp["area"]) != 4:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])):
            candidates.append(comp)

    assert len(candidates) == 1, "expected exactly one solid 2x2 start block"
    return candidates[0]


def _trace_chain(active: np.ndarray, first: tuple[int, int]) -> list[tuple[int, int]]:
    coords = set(map(tuple, np.argwhere(active)))
    assert first in coords

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = first
    while True:
        order.append(cur)
        cur_neighbors = [
            nbr
            for nbr in _neighbors(active.shape, cur[0], cur[1])
            if nbr in coords
        ]
        assert 1 <= len(cur_neighbors) <= 2, "active cells must form a simple path"

        next_cells = [nbr for nbr in cur_neighbors if nbr != prev]
        if prev is None:
            assert len(next_cells) == 1, "path must leave the start block in one direction"
        else:
            assert len(next_cells) <= 1, "path branches are not allowed"

        if not next_cells:
            break

        nxt = next_cells[0]
        assert nxt not in order, "path may not self-intersect"
        prev, cur = cur, nxt

    assert len(order) == int(active.sum()), "path traversal must visit every active cell"
    return order


def _segment_groups(
    g: np.ndarray, path_cells: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    assert path_cells, "expected at least one path cell before the end marker"
    groups: list[list[tuple[int, int]]] = []
    for cell in path_cells:
        color = int(g[cell])
        if not groups or int(g[groups[-1][0]]) != color:
            groups.append([cell])
        else:
            groups[-1].append(cell)

    seen_colors: set[int] = set()
    for group in groups:
        color = int(g[group[0]])
        assert color not in seen_colors, "path segment colors must be unique"
        seen_colors.add(color)

        rows = {r for r, _c in group}
        cols = {c for _r, c in group}
        assert len(rows) == 1 or len(cols) == 1, "each color segment must be straight"
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate colors forward along a segmented path from a 2x2 start block to a single end marker.

    Input structure:
    - The background is black.
    - One special monochrome 2x2 block marks the path start, and a single non-black
      pixel marks the path end.
    - Between them lies one connected one-cell-wide orthogonal path made of straight
      horizontal or vertical color segments whose colors are all distinct.

    Rule:
    - Follow the path in order starting from the cell adjacent to the 2x2 block and
      ending at the terminal marker pixel.
    - Recolor the first path segment with the start-block color, then recolor each
      later segment with the original color of the previous segment.
    - Recolor the entire 2x2 start block with the original color of the last path
      segment, and leave the end marker unchanged.

    Key insight:
    - The transformation is a cyclic color shift along an ordered polyline, so the
      crucial step is recovering the unique traversal order from the block-connected
      endpoint through the path to the terminal marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start = _find_start_block(g)
    start_color = int(start["color"])
    start_mask = np.zeros(g.shape, dtype=bool)
    start_mask[start["coords"][:, 0], start["coords"][:, 1]] = True

    active = (g != BG) & (~start_mask)
    assert int(active.sum()) >= 2, "expected at least one path cell plus one end marker"

    start_neighbors = {
        nbr
        for r, c in start["coords"]
        for nbr in _neighbors(g.shape, int(r), int(c))
        if active[nbr]
    }
    assert len(start_neighbors) == 1, "path must touch the start block at exactly one cell"
    first_cell = next(iter(start_neighbors))

    order = _trace_chain(active, first_cell)
    path_cells = order[:-1]
    assert path_cells, "the end marker may not be the only active cell"
    groups = _segment_groups(g, path_cells)
    last_segment_color = int(g[groups[-1][0]])

    out = g.copy()
    carry = start_color
    for group in groups:
        original = int(g[group[0]])
        for r, c in group:
            out[r, c] = carry
        carry = original

    out[start["coords"][:, 0], start["coords"][:, 1]] = last_segment_color
    return out


def _orthogonals(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _block_layout(direction: tuple[int, int], variant: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if direction == (0, 1):
        top_left = (0, -2) if variant == 0 else (-1, -2)
        attach = (0, -1)
    elif direction == (0, -1):
        top_left = (0, 1) if variant == 0 else (-1, 1)
        attach = (0, 1)
    elif direction == (1, 0):
        top_left = (-2, 0) if variant == 0 else (-2, -1)
        attach = (-1, 0)
    else:
        assert direction == (-1, 0)
        top_left = (1, 0) if variant == 0 else (1, -1)
        attach = (1, 0)

    cells = {
        (top_left[0] + dr, top_left[1] + dc)
        for dr in (0, 1)
        for dc in (0, 1)
    }
    return cells, attach


def _touching_occupied(
    cell: tuple[int, int], occupied: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    r, c = cell
    return {
        (r + dr, c + dc)
        for dr, dc in DIRS
        if (r + dr, c + dc) in occupied
    }


def _can_place(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    allowed_neighbors: set[tuple[int, int]],
) -> bool:
    if cell in occupied:
        return False
    return _touching_occupied(cell, occupied) <= allowed_neighbors


def _extend_line(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    prev = start
    built: list[tuple[int, int]] = []
    local_occupied = set(occupied)
    for _ in range(steps):
        nxt = (prev[0] + direction[0], prev[1] + direction[1])
        if not _can_place(nxt, local_occupied, {prev}):
            return None
        built.append(nxt)
        local_occupied.add(nxt)
        prev = nxt
    return built


def _build_relative_instance() -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]], tuple[int, int]] | None:
    segment_count = random.randint(*SEGMENT_COUNT_RANGE)
    first_direction = random.choice(DIRS)
    block_cells, attach = _block_layout(first_direction, random.randrange(2))

    first_cell = (0, 0)
    if not _can_place(first_cell, block_cells, {attach}):
        return None

    occupied = set(block_cells)
    occupied.add(first_cell)
    segments: list[list[tuple[int, int]]] = []
    current = first_cell
    prev_direction = first_direction

    for seg_idx in range(segment_count):
        direction_options = [first_direction] if seg_idx == 0 else _orthogonals(prev_direction)
        random.shuffle(direction_options)
        length_options = list(range(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1))
        random.shuffle(length_options)

        chosen_cells: list[tuple[int, int]] | None = None
        chosen_direction: tuple[int, int] | None = None
        for direction in direction_options:
            for seg_len in length_options:
                steps = seg_len - 1 if seg_idx == 0 else seg_len
                extension = _extend_line(current, direction, steps, occupied)
                if extension is None:
                    continue

                segment_cells = [current] + extension if seg_idx == 0 else extension
                end_cell = segment_cells[-1]
                if seg_idx == segment_count - 1:
                    marker = (end_cell[0] + direction[0], end_cell[1] + direction[1])
                    trial_occupied = occupied | set(extension)
                    if not _can_place(marker, trial_occupied, {end_cell}):
                        continue

                chosen_cells = segment_cells
                chosen_direction = direction
                break
            if chosen_cells is not None:
                break

        if chosen_cells is None or chosen_direction is None:
            return None

        segments.append(chosen_cells)
        new_cells = chosen_cells if seg_idx == 0 else chosen_cells
        occupied.update(new_cells[1:] if seg_idx == 0 else new_cells)
        current = chosen_cells[-1]
        prev_direction = chosen_direction

    marker = (current[0] + prev_direction[0], current[1] + prev_direction[1])
    if not _can_place(marker, occupied, {current}):
        return None

    all_coords = set(block_cells) | {marker}
    for segment in segments:
        all_coords.update(segment)

    rows = [r for r, _c in all_coords]
    cols = [c for _r, c in all_coords]
    if max(rows) - min(rows) + 1 > MAX_SHAPE_SPAN or max(cols) - min(cols) + 1 > MAX_SHAPE_SPAN:
        return None
    return block_cells, segments, marker


def _shifted_layout(
    block_cells: set[tuple[int, int]],
    segments: list[list[tuple[int, int]]],
    marker: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], list[list[tuple[int, int]]], tuple[int, int]]:
    all_coords = set(block_cells) | {marker}
    for segment in segments:
        all_coords.update(segment)

    min_r = min(r for r, _c in all_coords)
    max_r = max(r for r, _c in all_coords)
    min_c = min(c for _r, c in all_coords)
    max_c = max(c for _r, c in all_coords)
    shape_h = max_r - min_r + 1
    shape_w = max_c - min_c + 1

    extra_h = random.randint(EXTRA_MARGIN_RANGE[0], min(EXTRA_MARGIN_RANGE[1], MAX_GRID_SPAN - shape_h))
    extra_w = random.randint(EXTRA_MARGIN_RANGE[0], min(EXTRA_MARGIN_RANGE[1], MAX_GRID_SPAN - shape_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    shift_r = top - min_r
    shift_c = left - min_c
    grid_h = shape_h + extra_h
    grid_w = shape_w + extra_w

    shifted_block = {(r + shift_r, c + shift_c) for r, c in block_cells}
    shifted_segments = [
        [(r + shift_r, c + shift_c) for r, c in segment]
        for segment in segments
    ]
    shifted_marker = (marker[0] + shift_r, marker[1] + shift_c)
    return (grid_h, grid_w), shifted_block, shifted_segments, shifted_marker


def _render_instance(
    shape: tuple[int, int],
    block_cells: set[tuple[int, int]],
    segments: list[list[tuple[int, int]]],
    marker: tuple[int, int],
) -> np.ndarray:
    segment_count = len(segments)
    colors = random.sample(COLOR_POOL, segment_count + 2)
    start_color = colors[0]
    end_color = colors[1]
    segment_colors = colors[2:]

    grid = np.full(shape, BG, dtype=int)
    for r, c in block_cells:
        grid[r, c] = start_color
    for segment, color in zip(segments, segment_colors):
        for r, c in segment:
            grid[r, c] = color
    grid[marker] = end_color
    return grid


def _generate_attempt() -> np.ndarray | None:
    built = _build_relative_instance()
    if built is None:
        return None

    block_cells, segments, marker = built
    shape, block_cells, segments, marker = _shifted_layout(block_cells, segments, marker)
    grid = _render_instance(shape, block_cells, segments, marker)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the color-shift path task")
