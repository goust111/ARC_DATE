import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SOURCE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_RETRIES = 400
SHAPE_RETRIES = 300


def _contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size) and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1))


def _find_quadrants(grid: np.ndarray) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], int]:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert _contiguous(rows), "divider rows must form one contiguous grey band"
    assert _contiguous(cols), "divider cols must form one contiguous grey band"

    top_h = int(rows[0])
    bottom_start = int(rows[-1] + 1)
    left_w = int(cols[0])
    right_start = int(cols[-1] + 1)
    bottom_h = int(grid.shape[0] - bottom_start)
    right_w = int(grid.shape[1] - right_start)
    assert top_h > 0 and left_w > 0, "quadrants must be non-empty"
    assert top_h == bottom_h and left_w == right_w, "dividers must split the grid into equal halves"
    assert top_h == left_w, "quadrants must be square"

    bounds = {
        (0, 0): (0, top_h, 0, left_w),
        (0, 1): (0, top_h, right_start, grid.shape[1]),
        (1, 0): (bottom_start, grid.shape[0], 0, left_w),
        (1, 1): (bottom_start, grid.shape[0], right_start, grid.shape[1]),
    }
    return bounds, top_h


def _project_local(
    local_r: int,
    local_c: int,
    quadrant_size: int,
    src_key: tuple[int, int],
    tgt_key: tuple[int, int],
) -> tuple[int, int]:
    rr = local_r if src_key[0] == tgt_key[0] else quadrant_size - 1 - local_r
    cc = local_c if src_key[1] == tgt_key[1] else quadrant_size - 1 - local_c
    return rr, cc


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    bounds, quadrant_size = _find_quadrants(g)
    occupied: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    empty_keys: list[tuple[int, int]] = []

    for key, (r0, r1, c0, c1) in bounds.items():
        quad = g[r0:r1, c0:c1]
        colors = [int(v) for v in np.unique(quad) if int(v) != BLACK]
        if colors:
            assert len(colors) == 1, "each source quadrant must contain a single pattern color"
            color = colors[0]
            assert color not in {GREY, YELLOW}, "source colors must differ from grey divider and yellow overlap"
            occupied[key] = (quad, color)
        else:
            empty_keys.append(key)

    assert len(occupied) == 2 and len(empty_keys) == 2, "expected exactly two occupied and two empty quadrants"
    occupied_keys = set(occupied)
    assert occupied_keys in ({(0, 0), (1, 1)}, {(0, 1), (1, 0)}), "occupied quadrants must be diagonal"
    for key in empty_keys:
        r0, r1, c0, c1 = bounds[key]
        assert np.all(g[r0:r1, c0:c1] == BLACK), "empty quadrants must be pure background"

    source_colors = {color for _, color in occupied.values()}
    assert len(source_colors) == 2, "the two source patterns must use different colors"

    out = g.copy()
    for src_key, (src_quad, src_color) in occupied.items():
        src_coords = np.argwhere(src_quad == src_color)
        for tgt_key in empty_keys:
            assert (src_key[0] == tgt_key[0]) ^ (src_key[1] == tgt_key[1]), "sources only project into orthogonally adjacent quadrants"
            tr0, _tr1, tc0, _tc1 = bounds[tgt_key]
            for local_r_raw, local_c_raw in src_coords:
                local_r = int(local_r_raw)
                local_c = int(local_c_raw)
                rr, cc = _project_local(local_r, local_c, quadrant_size, src_key, tgt_key)
                gr = tr0 + rr
                gc = tc0 + cc
                if out[gr, gc] == BLACK:
                    out[gr, gc] = src_color
                elif out[gr, gc] == YELLOW:
                    continue
                elif out[gr, gc] != src_color:
                    out[gr, gc] = YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the two diagonal source patterns into the empty quadrants by axis-specific mirroring and mix overlaps into yellow.

    Input structure:
    - The grid is a square split into four equal quadrants by a central grey horizontal divider and a central grey vertical divider.
    - The background is black.
    - Exactly two diagonal quadrants contain one small monochrome pattern each, and the other diagonal pair is empty.
    - The two source pattern colors are distinct and neither source uses grey or yellow.

    Rule:
    - Keep the original source quadrants and the grey dividers unchanged.
    - Mirror each source pattern into the empty quadrant across the vertical divider and into the empty quadrant across the horizontal divider.
    - Moving to the opposite side flips local columns, and moving to the opposite vertical half flips local rows.
    - When the two projected copies land on the same target cell, recolor that cell yellow; otherwise keep the projected source color.

    Key insight:
    - Each empty quadrant is an overlay of two mirrored source patterns, and the needed flip axis is determined only by the relative source and target quadrant positions.
    """
    return _solve(grid)


def _rot180(shape: set[tuple[int, int]], quadrant_size: int) -> set[tuple[int, int]]:
    return {(quadrant_size - 1 - r, quadrant_size - 1 - c) for r, c in shape}


def _is_connected(shape: Iterable[tuple[int, int]]) -> bool:
    cells = set(shape)
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _is_irregular(shape: set[tuple[int, int]]) -> bool:
    if len(shape) == 2:
        return True
    rows = {r for r, _ in shape}
    cols = {c for _, c in shape}
    if len(rows) == 1 or len(cols) == 1:
        return False
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)
    area = (max_r - min_r + 1) * (max_c - min_c + 1)
    return area != len(shape)


def _frontier(shape: set[tuple[int, int]], quadrant_size: int, banned: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in shape:
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < quadrant_size and 0 <= cc < quadrant_size and (rr, cc) not in shape and (rr, cc) not in banned:
                cells.add((rr, cc))
    return list(cells)


def _random_connected_shape(
    quadrant_size: int,
    size: int,
    required: set[tuple[int, int]] | None = None,
    banned: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    banned = set() if banned is None else set(banned)
    required = set() if required is None else set(required)
    if required & banned:
        return None
    for r, c in required:
        if not (0 <= r < quadrant_size and 0 <= c < quadrant_size):
            return None
    if len(required) > size:
        return None

    allowed = [(r, c) for r in range(quadrant_size) for c in range(quadrant_size) if (r, c) not in banned]
    if not allowed:
        return None

    for _ in range(SHAPE_RETRIES):
        if required:
            shape = set(required)
            if not _is_connected(shape):
                continue
        else:
            shape = {random.choice(allowed)}

        while len(shape) < size:
            frontier = _frontier(shape, quadrant_size, banned)
            if not frontier:
                break
            shape.add(random.choice(frontier))

        if len(shape) != size:
            continue
        if not _is_connected(shape):
            continue
        if not _is_irregular(shape):
            continue
        return shape
    return None


def _paint_shape(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    shape: set[tuple[int, int]],
    color: int,
) -> None:
    for r, c in shape:
        grid[start_r + r, start_c + c] = color


def _build_scene(
    quadrant_size: int,
    occupied_keys: tuple[tuple[int, int], tuple[int, int]],
    shapes: tuple[set[tuple[int, int]], set[tuple[int, int]]],
    colors: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    divider_width = 2
    side = 2 * quadrant_size + divider_width
    input_grid = np.full((side, side), BLACK, dtype=int)
    input_grid[quadrant_size : quadrant_size + divider_width, :] = GREY
    input_grid[:, quadrant_size : quadrant_size + divider_width] = GREY

    starts = {
        (0, 0): (0, 0),
        (0, 1): (0, quadrant_size + divider_width),
        (1, 0): (quadrant_size + divider_width, 0),
        (1, 1): (quadrant_size + divider_width, quadrant_size + divider_width),
    }
    for key, shape, color in zip(occupied_keys, shapes, colors):
        sr, sc = starts[key]
        _paint_shape(input_grid, sr, sc, shape, color)

    output_grid = input_grid.copy()
    empty_keys = tuple(key for key in starts if key not in occupied_keys)
    for src_key, shape, color in zip(occupied_keys, shapes, colors):
        for tgt_key in empty_keys:
            rr0, cc0 = starts[tgt_key]
            for local_r, local_c in shape:
                rr, cc = _project_local(local_r, local_c, quadrant_size, src_key, tgt_key)
                gr = rr0 + rr
                gc = cc0 + cc
                if output_grid[gr, gc] == BLACK:
                    output_grid[gr, gc] = color
                elif output_grid[gr, gc] != color:
                    output_grid[gr, gc] = YELLOW

    return input_grid, output_grid


def _gen_attempt() -> np.ndarray | None:
    quadrant_size = random.randint(5, 14)
    has_overlap = random.random() < 0.5
    use_main_diagonal = random.random() < 0.5
    occupied_keys = ((0, 0), (1, 1)) if use_main_diagonal else ((0, 1), (1, 0))
    colors = tuple(random.sample(SOURCE_COLORS, 2))

    shape_a = _random_connected_shape(quadrant_size, random.randint(2, 6))
    if shape_a is None:
        return None

    rotated_a = _rot180(shape_a, quadrant_size)
    if has_overlap:
        required = {random.choice(tuple(rotated_a))}
        shape_b = _random_connected_shape(quadrant_size, random.randint(2, 6), required=required)
    else:
        shape_b = _random_connected_shape(quadrant_size, random.randint(2, 6), banned=rotated_a)
    if shape_b is None:
        return None

    overlap_count = len(shape_b & rotated_a)
    if has_overlap != (overlap_count > 0):
        return None

    input_grid, expected_output = _build_scene(quadrant_size, occupied_keys, (shape_a, shape_b), colors)
    if has_overlap:
        assert np.count_nonzero(expected_output == YELLOW) == 2 * overlap_count
    else:
        assert np.count_nonzero(expected_output == YELLOW) == 0

    try:
        actual_output = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(actual_output, expected_output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
