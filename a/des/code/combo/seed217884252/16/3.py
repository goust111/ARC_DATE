from __future__ import annotations

import numpy as np


BG = 0
PATH_COLOR_POOL = tuple(range(1, 10))
PATH_MIN_LEN = 5
PATH_MAX_LEN = 28
MAX_SEGMENTS = 7
TEMP_HEIGHT_RANGE = (5, 13)
TEMP_WIDTH_RANGE = (5, 13)
MARGIN_RANGE = (1, 5)
PATH_BUILD_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _path_order(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = {tuple(map(int, cell)) for cell in np.argwhere(grid != BG)}
    assert len(coords) >= 2, "expected a nontrivial path"

    degree: dict[tuple[int, int], int] = {}
    for cell in coords:
        degree[cell] = sum(neighbor in coords for neighbor in _neighbors4(cell))

    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degree.values()), "path must be branch-free"

    start = min(endpoints)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        order.append(cur)
        next_cells = [neighbor for neighbor in _neighbors4(cur) if neighbor in coords and neighbor != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unambiguous"
        prev, cur = cur, next_cells[0]

    assert len(order) == len(coords), "path traversal did not cover every cell"
    return order


def _segment_specs(grid: np.ndarray, order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    run_color = None
    run_len = 0

    for cell in order:
        color = int(grid[cell])
        assert color != BG
        if color == run_color:
            run_len += 1
        else:
            if run_color is not None:
                specs.append((int(run_color), int(run_len)))
            run_color = color
            run_len = 1

    assert run_color is not None
    specs.append((int(run_color), int(run_len)))
    return specs


def _render_output(specs: list[tuple[int, int]]) -> np.ndarray:
    assert specs
    height = max(length for _color, length in specs)
    width = len(specs)
    out = np.full((height, width), BG, dtype=int)
    for col, (color, length) in enumerate(specs):
        out[height - length :, col] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Convert a multicolor 4-connected path into bottom-aligned segment bars."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    order = _path_order(g)
    specs = _segment_specs(g, order)
    return _render_output(specs)


def _extension_candidates(
    tail: tuple[int, int],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(tail):
        if not (0 <= nr < height and 0 <= nc < width):
            continue
        nxt = (nr, nc)
        if nxt in occupied:
            continue
        occupied_neighbors = sum(neighbor in occupied for neighbor in _neighbors4(nxt))
        if occupied_neighbors == 1:
            candidates.append(nxt)
    return candidates


def _greedy_full_path(height: int, width: int) -> list[tuple[int, int]]:
    start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
    path = [start]
    occupied = {start}

    while True:
        tail = path[-1]
        candidates = _extension_candidates(tail, occupied, height, width)
        if not candidates:
            return path

        scored: list[tuple[int, float, tuple[int, int]]] = []
        for candidate in candidates:
            occupied.add(candidate)
            onward = len(_extension_candidates(candidate, occupied, height, width))
            occupied.remove(candidate)
            scored.append((int(onward), float(_RNG.random()), candidate))

        best_onward = max(score[0] for score in scored)
        near_best = [candidate for onward, _tie, candidate in scored if onward >= best_onward - 1]
        choice = near_best[int(_RNG.integers(len(near_best)))]
        occupied.add(choice)
        path.append(choice)


def _sample_subpath() -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_ATTEMPTS):
        height = int(_RNG.integers(*TEMP_HEIGHT_RANGE))
        width = int(_RNG.integers(*TEMP_WIDTH_RANGE))
        full_path = _greedy_full_path(height, width)
        if len(full_path) < PATH_MIN_LEN:
            continue

        max_len = min(len(full_path), PATH_MAX_LEN)
        sub_len = int(_RNG.integers(PATH_MIN_LEN, max_len + 1))
        start_idx = int(_RNG.integers(0, len(full_path) - sub_len + 1))
        return full_path[start_idx : start_idx + sub_len]
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]

    cuts = sorted(int(v) for v in _RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    last = 0
    lengths: list[int] = []
    for cut in cuts + [total]:
        lengths.append(cut - last)
        last = cut
    return lengths


def _sample_segment_specs(path_len: int) -> list[tuple[int, int]]:
    max_parts = min(MAX_SEGMENTS, path_len - 1)
    part_count = int(_RNG.integers(2, max_parts + 1))
    lengths = _random_composition(path_len, part_count)
    assert max(lengths) >= 2

    specs: list[tuple[int, int]] = []
    prev_color = None
    for length in lengths:
        choices = [color for color in PATH_COLOR_POOL if color != prev_color]
        color = int(choices[int(_RNG.integers(len(choices)))])
        specs.append((color, int(length)))
        prev_color = color
    return specs


def _embed_path(path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rows = [r for r, _c in path]
    cols = [c for _r, c in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    top = int(_RNG.integers(*MARGIN_RANGE))
    bottom = int(_RNG.integers(*MARGIN_RANGE))
    left = int(_RNG.integers(*MARGIN_RANGE))
    right = int(_RNG.integers(*MARGIN_RANGE))
    height = max_r - min_r + 1 + top + bottom
    width = max_c - min_c + 1 + left + right
    assert height <= 30 and width <= 30

    shifted = [(r - min_r + top, c - min_c + left) for r, c in path]
    return shifted, (height, width)


def _paint_segments(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    specs: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    idx = 0
    for color, length in specs:
        for _ in range(length):
            r, c = order[idx]
            grid[r, c] = int(color)
            idx += 1
    assert idx == len(order)
    return grid


def _expected_specs_for_top_left(
    path: list[tuple[int, int]],
    paint_order: list[tuple[int, int]],
    specs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    top_left_endpoint = min(path[0], path[-1])
    return specs if paint_order[0] == top_left_endpoint else list(reversed(specs))


def _gen_attempt() -> np.ndarray | None:
    base_path = _sample_subpath()
    if base_path is None:
        return None

    path, shape = _embed_path(base_path)
    specs = _sample_segment_specs(len(path))
    paint_order = path if bool(_RNG.integers(0, 2)) else list(reversed(path))

    grid = _paint_segments(shape, paint_order, specs)

    expected_specs = _expected_specs_for_top_left(path, paint_order, specs)
    expected = _render_output(expected_specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
