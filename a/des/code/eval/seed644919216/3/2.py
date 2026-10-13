from __future__ import annotations

import random

import numpy as np


BG = 8
PATH_COLOR_POOL = (1, 2, 3, 4)
RESULT_COLOR_POOL = (0, 5, 6, 7, 9)
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _parse_mix_table(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], dict[tuple[int, int], int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must be large enough for a 2x2 mix table"
    assert int(g[0, 0]) == bg, "top-left table corner must be background"

    row_headers: list[int] = []
    c = 1
    while c < w and int(g[0, c]) != bg:
        row_headers.append(int(g[0, c]))
        c += 1

    col_headers: list[int] = []
    r = 1
    while r < h and int(g[r, 0]) != bg:
        col_headers.append(int(g[r, 0]))
        r += 1

    n = len(row_headers)
    assert n >= 2, "expected at least two path colors in the mix table"
    assert n == len(col_headers), "row and column header counts must match"
    assert len(set(row_headers)) == n, "row headers must be unique path colors"
    assert len(set(col_headers)) == n, "column headers must be unique path colors"
    assert set(row_headers) == set(col_headers), "row/column headers must name the same paths"
    assert 1 + n <= h and 1 + n <= w, "mix table must fit in the grid"

    mapping: dict[tuple[int, int], int] = {}
    for i, row_color in enumerate(col_headers, start=1):
        for j, col_color in enumerate(row_headers, start=1):
            value = int(g[i, j])
            assert value != bg, "mix table cells must be non-background"
            mapping[(int(row_color), int(col_color))] = value

    return row_headers, col_headers, mapping, (n + 1, n + 1)


def _mix_lookup(mapping: dict[tuple[int, int], int], color_a: int, color_b: int) -> int:
    direct = mapping.get((int(color_a), int(color_b)))
    reverse = mapping.get((int(color_b), int(color_a)))
    assert direct is not None or reverse is not None, "missing mix-table entry"
    if direct is None:
        return int(reverse)
    if reverse is None:
        return int(direct)
    assert int(direct) == int(reverse), "cross-color mix table must be symmetric"
    return int(direct)


def _crossing_result(
    grid: np.ndarray,
    r: int,
    c: int,
    path_colors: set[int],
    mapping: dict[tuple[int, int], int],
) -> int | None:
    g = np.asarray(grid, dtype=int)
    center = int(g[r, c])
    if center not in path_colors:
        return None

    left = int(g[r, c - 1])
    right = int(g[r, c + 1])
    up = int(g[r - 1, c])
    down = int(g[r + 1, c])

    if left != right or up != down:
        return None
    if left not in path_colors or up not in path_colors:
        return None
    if left == up:
        return None
    if center not in {left, up}:
        return None

    return _mix_lookup(mapping, left, up)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect pairwise path crossings and recolor each crossing via the top-left mix table.

    The table format is a 1-pixel-cell square anchored at the top-left corner:
    - (0, 0) is background.
    - row 0 and column 0 list the path colors.
    - interior cells give the symmetric mix result for each color pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    row_headers, _col_headers, mapping, table_shape = _parse_mix_table(g, bg)
    table_h, table_w = table_shape
    path_colors = {int(color) for color in row_headers}

    out = g.copy()
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            if r < table_h and c < table_w:
                continue
            mixed = _crossing_result(g, r, c, path_colors, mapping)
            if mixed is not None:
                out[r, c] = int(mixed)

    return out


def _sample_spaced_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    if count <= 0:
        return []
    if high < low:
        return None

    values = list(range(int(low), int(high) + 1))
    for _ in range(200):
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - other) >= min_gap for other in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    chosen.sort()
                    return chosen
    return None


def _random_mix_matrix(path_colors: list[int]) -> np.ndarray:
    n = len(path_colors)
    mix = np.zeros((n, n), dtype=int)
    off_diagonal: list[int] = []
    for i in range(n):
        mix[i, i] = int(path_colors[i])
        for j in range(i + 1, n):
            value = int(random.choice(RESULT_COLOR_POOL))
            mix[i, j] = value
            mix[j, i] = value
            off_diagonal.append(value)

    if len(off_diagonal) >= 2 and len(set(off_diagonal)) == 1:
        i = 0
        j = 1
        alternatives = [color for color in RESULT_COLOR_POOL if color != int(mix[i, j])]
        replacement = int(random.choice(alternatives))
        mix[i, j] = replacement
        mix[j, i] = replacement

    return mix


def _expected_output(
    grid: np.ndarray,
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    mix_by_color: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for h_path in horizontals:
        for v_path in verticals:
            r = int(h_path["row"])
            c = int(v_path["col"])
            out[r, c] = _mix_lookup(mix_by_color, int(h_path["color"]), int(v_path["color"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    total_paths = random.randint(2, len(PATH_COLOR_POOL))
    split_options = [(h_count, total_paths - h_count) for h_count in range(1, total_paths)]
    h_count, v_count = random.choice(split_options)

    table_size = total_paths + 1
    min_h = table_size + 2 * h_count + 6
    min_w = table_size + 2 * v_count + 6
    height = random.randint(min_h, min_h + 8)
    width = random.randint(min_w, min_w + 8)

    row_low = table_size + 2
    row_high = height - 3
    col_low = table_size + 2
    col_high = width - 3

    h_rows = _sample_spaced_positions(row_low, row_high, h_count, 2)
    v_cols = _sample_spaced_positions(col_low, col_high, v_count, 2)
    if h_rows is None or v_cols is None:
        return None

    path_colors = random.sample(PATH_COLOR_POOL, total_paths)
    mix = _random_mix_matrix(path_colors)
    mix_by_color: dict[tuple[int, int], int] = {}
    for i, row_color in enumerate(path_colors):
        for j, col_color in enumerate(path_colors):
            mix_by_color[(int(row_color), int(col_color))] = int(mix[i, j])

    grid = np.full((height, width), BG, dtype=int)
    grid[0, 1 : total_paths + 1] = np.array(path_colors, dtype=int)
    grid[1 : total_paths + 1, 0] = np.array(path_colors, dtype=int)
    grid[1 : total_paths + 1, 1 : total_paths + 1] = mix

    horizontals: list[dict[str, int]] = []
    for idx, row in enumerate(h_rows):
        left = random.randint(table_size + 1, min(v_cols) - 1)
        right = random.randint(max(v_cols) + 1, width - 2)
        horizontals.append(
            {
                "color": int(path_colors[idx]),
                "row": int(row),
                "left": int(left),
                "right": int(right),
            }
        )

    verticals: list[dict[str, int]] = []
    for offset, col in enumerate(v_cols):
        top = random.randint(table_size + 1, min(h_rows) - 1)
        bottom = random.randint(max(h_rows) + 1, height - 2)
        verticals.append(
            {
                "color": int(path_colors[h_count + offset]),
                "col": int(col),
                "top": int(top),
                "bottom": int(bottom),
            }
        )

    draw_order: list[tuple[str, dict[str, int]]] = [("h", path) for path in horizontals]
    draw_order.extend(("v", path) for path in verticals)
    random.shuffle(draw_order)

    for kind, path in draw_order:
        if kind == "h":
            r = int(path["row"])
            grid[r, int(path["left"]) : int(path["right"]) + 1] = int(path["color"])
        else:
            c = int(path["col"])
            grid[int(path["top"]) : int(path["bottom"]) + 1, c] = int(path["color"])

    expected = _expected_output(grid, horizontals, verticals, mix_by_color)
    produced = main(grid)

    if not np.array_equal(produced, expected):
        return None

    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid crossing-mix ARC input")
