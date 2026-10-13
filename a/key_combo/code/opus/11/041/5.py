from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
SEP = 5
PATH_COLORS = (1, 2, 3)
MIN_PATH_LEN = 3
MAX_PATH_LEN = 7
MIN_PATHS = 2
MAX_PATHS = 3
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _flood_component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    return cols[0]


def _path_metrics(cells: set[tuple[int, int]]) -> dict[str, int | tuple[int, int]]:
    assert cells
    degs = {}
    for r, c in cells:
        degs[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
    degree_values = sorted(degs.values())
    assert degree_values.count(1) == 2, "path must have exactly two endpoints"
    assert degree_values[-1] <= 2, "path must be unbranched"

    endpoints = [rc for rc, deg in degs.items() if deg == 1]
    (r0, c0), (r1, c1) = endpoints
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox_h = max(rows) - min(rows) + 1
    bbox_w = max(cols) - min(cols) + 1
    manhattan = abs(r0 - r1) + abs(c0 - c1) + 1
    return {
        "length": len(cells),
        "top": min(rows),
        "left": min(cols),
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "endpoint_span": manhattan,
    }


def _extract_path_info(left: np.ndarray) -> dict[int, dict[str, int]]:
    colors = sorted(set(int(v) for v in np.unique(left)) - {BG, SEP})
    assert MIN_PATHS <= len(colors) <= MAX_PATHS, "expected 2 or 3 path colors in the left half"
    info: dict[int, dict[str, int]] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(left == color)]
        assert coords, "every path color must appear"
        comp = _flood_component(left == color, coords[0])
        assert len(comp) == len(coords), "each path color must form exactly one 4-connected component"
        metrics = _path_metrics(comp)
        assert MIN_PATH_LEN <= int(metrics["length"]) <= MAX_PATH_LEN, "path length out of range"
        info[color] = metrics

    lengths = [int(metrics["length"]) for metrics in info.values()]
    assert len(set(lengths)) == len(lengths), "path lengths must be distinct"
    return info


def _extract_bands(right: np.ndarray) -> list[tuple[int, np.ndarray, int]]:
    active_rows = [r for r in range(right.shape[0]) if np.any(right[r] != BG)]
    assert MIN_PATHS <= len(active_rows) <= MAX_PATHS, "expected 2 or 3 right-half bands"
    assert all(active_rows[i + 1] - active_rows[i] >= 2 for i in range(len(active_rows) - 1)), "bands must be separated by background rows"

    bands: list[tuple[int, np.ndarray, int]] = []
    seen_colors: set[int] = set()
    for r in active_rows:
        row = right[r]
        vals = set(int(v) for v in row)
        assert len(vals) == 1, "each band row must be monochrome"
        color = next(iter(vals))
        assert color not in {BG, SEP}, "band rows must be non-background colors"
        assert color not in seen_colors, "band colors must be unique"
        seen_colors.add(color)
        bands.append((color, row.copy(), r))
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure each left-side path length and use it as the repetition count for the matching right-side row band.

    Input structure:
    - The background is black (0), and one full-height grey (5) column splits the grid into a left path half and a right band half.
    - The left half contains 2 or 3 different-colored single-pixel-wide 4-connected paths using colors from {1,2,3}.
    - The right half contains one full-width monochrome row band for each path color; each band is exactly one row tall and bands are separated by black rows.

    Rule:
    - Identify the unique grey separator column to locate the two halves.
    - In the left half, trace each colored 4-connected path and count its pixels; that count is the repeat factor for that color.
    - In the right half, read the row bands from top to bottom; for each band, vertically repeat that exact row by the repeat factor of the path with the same color.
    - Stack the repeated bands in that right-half top-to-bottom order and return only the stacked right-half content.

    Key insight:
    - The path geometry is distractor noise: only the pixel count of each colored path matters, and color is the key that links each count to the correct output band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0, "separator must leave non-empty halves"

    path_info = _extract_path_info(left)
    bands = _extract_bands(right)
    assert set(path_info) == {color for color, _, _ in bands}, "band colors must match path colors exactly"

    out_rows: list[np.ndarray] = []
    for color, row, _r in bands:
        out_rows.extend([row.copy() for _ in range(int(path_info[color]["length"]))])
    return np.stack(out_rows, axis=0)


def _dilate8(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for nr in range(max(0, r - 1), min(blocked.shape[0], r + 2)):
            for nc in range(max(0, c - 1), min(blocked.shape[1], c + 2)):
                blocked[nr, nc] = True


def _build_path(
    shape: tuple[int, int],
    blocked: np.ndarray,
    length: int,
    require_turn: bool,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if not blocked[r, c]]
    random.shuffle(candidates)

    def dfs(path: list[tuple[int, int]], seen: set[tuple[int, int]], prev_dir: tuple[int, int] | None, turns: int) -> list[tuple[int, int]] | None:
        if len(path) == length:
            if require_turn and turns == 0:
                return None
            return path.copy()

        r, c = path[-1]
        opts: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(shape, nr, nc) or blocked[nr, nc] or (nr, nc) in seen:
                continue
            visited_nbrs = sum((ar, ac) in seen for ar, ac in _neighbors4(nr, nc))
            if visited_nbrs != 1:
                continue
            step_dir = (nr - r, nc - c)
            next_turns = turns + int(prev_dir is not None and step_dir != prev_dir)
            opts.append(((nr, nc), step_dir, next_turns))

        random.shuffle(opts)
        if require_turn and turns == 0:
            opts.sort(key=lambda item: item[2], reverse=True)

        for cell, step_dir, next_turns in opts:
            path.append(cell)
            seen.add(cell)
            solved = dfs(path, seen, step_dir, next_turns)
            if solved is not None:
                return solved
            seen.remove(cell)
            path.pop()
        return None

    for start in candidates:
        solved = dfs([start], {start}, None, 0)
        if solved is not None:
            return solved
    return None


def _path_order_key(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols)


def _path_col_order_key(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(cols), min(rows)


def _pick_band_rows(height: int, count: int) -> tuple[int, ...] | None:
    options = [rows for rows in combinations(range(height), count) if all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1))]
    if not options:
        return None
    return random.choice(options)


def _stack_from_pairs(pairs: list[tuple[int, int]], width: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for color, reps in pairs:
        rows.extend([np.full((width,), color, dtype=int) for _ in range(reps)])
    return np.stack(rows, axis=0)


def _validate_non_degenerate(inp: np.ndarray, placed_paths: dict[int, list[tuple[int, int]]], band_order: list[int], lengths: dict[int, int]) -> None:
    sep = _find_separator_col(inp)
    left = inp[:, :sep]
    right = inp[:, sep + 1 :]
    path_info = _extract_path_info(left)
    bands = _extract_bands(right)

    assert band_order == [color for color, _, _ in bands]
    assert lengths == {color: int(info["length"]) for color, info in path_info.items()}

    expected = _stack_from_pairs([(color, lengths[color]) for color in band_order], right.shape[1])
    actual = main(inp)
    assert np.array_equal(actual, expected), "oracle mismatch"
    assert actual.shape[0] == sum(lengths.values()), "output height must equal sum of path lengths"

    top_order = [color for color, _cells in sorted(placed_paths.items(), key=lambda item: _path_order_key(item[1]))]
    left_order = [color for color, _cells in sorted(placed_paths.items(), key=lambda item: _path_col_order_key(item[1]))]
    assert band_order != top_order, "band order must not be recoverable from path top-to-bottom order"
    assert band_order != left_order, "band order must not be recoverable from path left-to-right order"

    bbox_height_guess = _stack_from_pairs([(color, int(path_info[color]["bbox_h"])) for color in band_order], right.shape[1])
    endpoint_guess = _stack_from_pairs([(color, int(path_info[color]["endpoint_span"])) for color in band_order], right.shape[1])
    max_bbox_guess = _stack_from_pairs(
        [(color, max(int(path_info[color]["bbox_h"]), int(path_info[color]["bbox_w"]))) for color in band_order],
        right.shape[1],
    )
    assert not np.array_equal(expected, bbox_height_guess), "counting bbox height must not solve the task"
    assert not np.array_equal(expected, endpoint_guess), "counting endpoint span must not solve the task"
    assert not np.array_equal(expected, max_bbox_guess), "counting max bbox extent must not solve the task"

    assert any(int(path_info[color]["length"]) > int(path_info[color]["endpoint_span"]) for color in path_info), "at least one path must visibly bend"


def _gen_attempt() -> np.ndarray | None:
    n_paths = random.randint(MIN_PATHS, MAX_PATHS)
    colors = random.sample(PATH_COLORS, n_paths)
    lengths = {color: length for color, length in zip(colors, random.sample(range(MIN_PATH_LEN, MAX_PATH_LEN + 1), n_paths))}
    band_order = random.sample(colors, len(colors))

    output_height = sum(lengths.values())
    right_w = random.randint(4, 10)
    left_w = random.randint(6, 11)
    width = left_w + 1 + right_w

    min_h = max(10, 2 * n_paths - 1, max(lengths.values()) + 3)
    max_h = min(24, 30)
    if min_h > max_h or width > 30 or output_height > 30:
        return None
    height = random.randint(min_h, max_h)

    band_rows = _pick_band_rows(height, n_paths)
    if band_rows is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_w] = SEP
    for r, color in zip(band_rows, band_order):
        grid[r, left_w + 1 :] = color

    usable_left_w = left_w - 1
    if usable_left_w < 3:
        return None
    blocked = np.zeros((height, usable_left_w), dtype=bool)
    placed_paths: dict[int, list[tuple[int, int]]] = {}
    turn_color = random.choice(colors)

    for color in sorted(colors, key=lambda c: lengths[c], reverse=True):
        path = _build_path((height, usable_left_w), blocked, lengths[color], require_turn=(color == turn_color))
        if path is None:
            return None
        placed_paths[color] = path
        for r, c in path:
            grid[r, c] = color
        _dilate8(blocked, path)

    try:
        _validate_non_degenerate(grid, placed_paths, band_order, lengths)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate sample")
