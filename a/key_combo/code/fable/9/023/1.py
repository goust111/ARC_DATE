from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
ROD = 2
GREEN = 3
ENDPOINT = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _advance(point: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return point[0] + direction[0] * steps, point[1] + direction[1] * steps


def _neighbor_count(points: set[tuple[int, int]], point: tuple[int, int]) -> int:
    r, c = point
    return sum((r + dr, c + dc) in points for dr, dc in DIR4)


def _path_from_relative_spec(
    segment_lengths: list[int],
    hinge_colors: list[int],
    start_dir: tuple[int, int],
) -> list[tuple[tuple[int, int], int]]:
    cells: list[tuple[tuple[int, int], int]] = [((0, 0), ENDPOINT)]
    visited = {(0, 0)}
    current = (0, 0)
    direction = start_dir

    for seg_len, hinge_color in zip(segment_lengths[:-1], hinge_colors):
        for step in range(1, seg_len + 1):
            current = _advance(current, direction)
            if current in visited:
                raise ValueError("path self-intersects")
            visited.add(current)
            cells.append((current, ROD if step < seg_len else hinge_color))
        direction = _turn_left(direction) if hinge_color == BLUE else _turn_right(direction)

    for _ in range(segment_lengths[-1]):
        current = _advance(current, direction)
        if current in visited:
            raise ValueError("path self-intersects")
        visited.add(current)
        cells.append((current, ROD))
    return cells


def _validate_simple_path(cells: list[tuple[tuple[int, int], int]]) -> None:
    points = {pt for pt, _color in cells}
    colors = dict(cells)
    endpoint_points = [pt for pt, color in cells if color == ENDPOINT]
    hinge_points = [pt for pt, color in cells if color in (BLUE, GREEN)]
    rod_points = [pt for pt, color in cells if color == ROD]
    assert len(endpoint_points) == 1
    assert hinge_points
    degrees = {pt: _neighbor_count(points, pt) for pt in points}

    for pt in endpoint_points:
        assert degrees[pt] == 1
    body_end = [pt for pt in rod_points if degrees[pt] == 1]
    assert len(body_end) == 1
    for pt in hinge_points:
        assert degrees[pt] == 2
    for pt in rod_points:
        if pt != body_end[0]:
            assert degrees[pt] == 2
    assert set(colors.values()) <= {ENDPOINT, BLUE, GREEN, ROD}


def _extract_ordered_line(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    endpoints = np.argwhere(grid == ENDPOINT)
    assert endpoints.shape == (1, 2), "expected exactly one gray endpoint"
    endpoint = tuple(map(int, endpoints[0]))

    non_bg = {tuple(map(int, pt)) for pt in np.argwhere(grid != BG)}
    neighbors = []
    for dr, dc in DIR4:
        nxt = (endpoint[0] + dr, endpoint[1] + dc)
        if nxt in non_bg:
            neighbors.append((dr, dc))
    assert len(neighbors) == 1, "endpoint must touch the rod in exactly one direction"
    direction = neighbors[0]

    ordered = [endpoint]
    current = endpoint
    while True:
        nxt = _advance(current, direction)
        if nxt not in non_bg:
            break
        ordered.append(nxt)
        current = nxt

    assert set(ordered) == non_bg, "foreground must be a single straight rod"
    rows = {r for r, _c in ordered}
    cols = {c for _r, c in ordered}
    assert len(rows) == 1 or len(cols) == 1
    return ordered, direction


def _draw_output(shape: tuple[int, int], endpoint: tuple[int, int], direction: tuple[int, int], segment_lengths: list[int], hinge_colors: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    current = endpoint
    out[current] = ENDPOINT
    facing = direction

    for seg_len, hinge_color in zip(segment_lengths[:-1], hinge_colors):
        for step in range(1, seg_len):
            rr, cc = _advance(current, facing, step)
            assert _in_bounds(shape, rr, cc)
            out[rr, cc] = ROD
        current = _advance(current, facing, seg_len)
        assert _in_bounds(shape, *current)
        out[current] = hinge_color
        facing = _turn_left(facing) if hinge_color == BLUE else _turn_right(facing)

    for step in range(1, segment_lengths[-1] + 1):
        rr, cc = _advance(current, facing, step)
        assert _in_bounds(shape, rr, cc)
        out[rr, cc] = ROD
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the straight rod into a polyline by applying each hinge's colored turn in sequence.

    Input structure:
    - A black background contains one horizontal or vertical red rod.
    - One end of the rod is a single gray endpoint pixel.
    - Interior rod pixels may include blue or green hinge markers; all remaining rod pixels are red.

    Rule:
    - Start at the gray endpoint and move along the rod's original direction.
    - Whenever a blue hinge is reached, turn left; whenever a green hinge is reached, turn right,
      always relative to the current travel direction.
    - Keep every segment length between consecutive special points unchanged, redraw the bent rod in
      place from the fixed gray endpoint, and erase the original straight rod elsewhere.

    Key insight:
    - The input rod is the unfolded net of the output polyline, and each hinge color encodes a
      relative ninety-degree turn rather than a fixed global direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, BLUE, ROD, GREEN, ENDPOINT}

    ordered, direction = _extract_ordered_line(g)
    assert len(ordered) >= 4, "rod must include endpoint, at least one hinge, and tail"

    hinge_indices: list[int] = []
    hinge_colors: list[int] = []
    for idx, point in enumerate(ordered[1:-1], start=1):
        color = int(g[point])
        if color in (BLUE, GREEN):
            hinge_indices.append(idx)
            hinge_colors.append(color)
        else:
            assert color == ROD, "non-hinge interior cells must be rod color"

    assert hinge_indices, "expected at least one hinge"
    key_indices = [0, *hinge_indices, len(ordered) - 1]
    segment_lengths = [b - a for a, b in zip(key_indices, key_indices[1:])]
    assert min(segment_lengths) >= 1
    endpoint = ordered[0]

    out = _draw_output(g.shape, endpoint, direction, segment_lengths, hinge_colors)
    return out


def _scenario_start_dir(scenario: int) -> tuple[int, int]:
    return [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (0, 1),
        (1, 0),
    ][scenario % 6]


def _sample_hinges(rng: np.random.Generator, scenario: int) -> tuple[list[int], list[int]]:
    if scenario < 4:
        hinge_count = int(rng.integers(2, 6))
        hinge_colors = [BLUE, GREEN]
        if hinge_count > 2:
            extra = rng.choice([BLUE, GREEN], size=hinge_count - 2, replace=True).astype(int).tolist()
            hinge_colors.extend(extra)
        rng.shuffle(hinge_colors)
    elif scenario == 4:
        hinge_count = 1
        hinge_colors = [BLUE]
    else:
        hinge_count = 1
        hinge_colors = [GREEN]

    segment_lengths = rng.integers(2, 6, size=hinge_count + 1).astype(int).tolist()
    if hinge_count >= 2 and len(set(segment_lengths)) == 1:
        pivot = int(rng.integers(0, len(segment_lengths)))
        segment_lengths[pivot] = 5 if segment_lengths[pivot] < 5 else 2
    return segment_lengths, hinge_colors


def _relative_bounds(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _translate_cells(
    rel_cells: list[tuple[tuple[int, int], int]],
    shape: tuple[int, int],
    shift: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for (r, c), color in rel_cells:
        rr = r + shift[0]
        cc = c + shift[1]
        assert _in_bounds(shape, rr, cc)
        assert grid[rr, cc] == BG
        grid[rr, cc] = color
    return grid


def _input_relative_cells(
    total_len: int,
    hinge_indices: list[int],
    hinge_colors: list[int],
    start_dir: tuple[int, int],
) -> list[tuple[tuple[int, int], int]]:
    rel_cells: list[tuple[tuple[int, int], int]] = [((0, 0), ENDPOINT)]
    hinge_at = dict(zip(hinge_indices, hinge_colors))
    for step in range(1, total_len + 1):
        point = (start_dir[0] * step, start_dir[1] * step)
        rel_cells.append((point, hinge_at.get(step, ROD)))
    return rel_cells


def _build_case(rng: np.random.Generator, scenario: int) -> tuple[np.ndarray, np.ndarray]:
    start_dir = _scenario_start_dir(scenario)

    for _ in range(400):
        segment_lengths, hinge_colors = _sample_hinges(rng, scenario)
        try:
            rel_out_cells = _path_from_relative_spec(segment_lengths, hinge_colors, start_dir)
        except ValueError:
            continue

        try:
            _validate_simple_path(rel_out_cells)
        except AssertionError:
            continue
        total_len = sum(segment_lengths)
        hinge_indices = np.cumsum(segment_lengths[:-1]).astype(int).tolist()
        rel_in_cells = _input_relative_cells(total_len, hinge_indices, hinge_colors, start_dir)

        out_points = [pt for pt, _color in rel_out_cells]
        in_points = [pt for pt, _color in rel_in_cells]
        r0a, c0a, r1a, c1a = _relative_bounds(out_points)
        r0b, c0b, r1b, c1b = _relative_bounds(in_points)
        r0 = min(r0a, r0b)
        c0 = min(c0a, c0b)
        r1 = max(r1a, r1b)
        c1 = max(c1a, c1b)
        need_h = r1 - r0 + 1
        need_w = c1 - c0 + 1
        if need_h > 30 or need_w > 30:
            continue

        extra_h = int(rng.integers(0, min(7, 31 - need_h)))
        extra_w = int(rng.integers(0, min(7, 31 - need_w)))
        H = need_h + extra_h
        W = need_w + extra_w
        if H > 30 or W > 30:
            continue

        row_shift_min = -r0
        row_shift_max = H - 1 - r1
        col_shift_min = -c0
        col_shift_max = W - 1 - c1
        if row_shift_min > row_shift_max or col_shift_min > col_shift_max:
            continue
        row_shift = int(rng.integers(row_shift_min, row_shift_max + 1))
        col_shift = int(rng.integers(col_shift_min, col_shift_max + 1))
        shift = (row_shift, col_shift)

        inp = _translate_cells(rel_in_cells, (H, W), shift)
        expected = _translate_cells(rel_out_cells, (H, W), shift)
        _validate_case(inp, expected, segment_lengths, hinge_colors)
        return inp, expected

    raise RuntimeError("failed to construct a valid unfolded-rod sample")


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    segment_lengths: list[int],
    hinge_colors: list[int],
) -> None:
    actual = main(inp)
    assert np.array_equal(actual, expected), "solver mismatch on generated sample"
    assert inp.shape == expected.shape
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert np.count_nonzero(inp != BG) == np.count_nonzero(expected != BG)
    assert np.count_nonzero(inp == ENDPOINT) == 1
    assert np.count_nonzero(inp == BLUE) + np.count_nonzero(inp == GREEN) == len(hinge_colors)
    assert len(segment_lengths) == len(hinge_colors) + 1
    assert min(segment_lengths) >= 2
    assert not np.array_equal(inp, expected)

    out_cells = [(tuple(map(int, pt)), int(expected[tuple(pt)])) for pt in np.argwhere(expected != BG)]
    _validate_simple_path(out_cells)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    rng = np.random.default_rng()
    scenario = _GEN_CALL_COUNT % 6
    _GEN_CALL_COUNT += 1
    inp, _expected = _build_case(rng, scenario)
    return inp
