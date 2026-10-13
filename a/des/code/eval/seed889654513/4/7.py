from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

EDGE_ORDER = ((0, 1), (1, 2), (2, 1), (1, 0))

BODY_SIZE_MIN = 3
BODY_SIZE_MAX = 5
MAX_GENERATION_ATTEMPTS = 300
MAX_CODE_ATTEMPTS = 200


def _window_from_code(code: int) -> np.ndarray:
    tile = np.full((3, 3), GREEN, dtype=int)
    tile[1, 1] = BLACK
    for bit, (r, c) in enumerate(EDGE_ORDER):
        tile[r, c] = YELLOW if code & (1 << bit) else BLUE
    return tile


def _code_from_window(tile: np.ndarray) -> int | None:
    if tile.shape != (3, 3):
        return None
    if int(tile[1, 1]) != BLACK:
        return None
    corners = (tile[0, 0], tile[0, 2], tile[2, 0], tile[2, 2])
    if any(int(v) != GREEN for v in corners):
        return None

    code = 0
    for bit, (r, c) in enumerate(EDGE_ORDER):
        value = int(tile[r, c])
        if value == YELLOW:
            code |= 1 << bit
        elif value != BLUE:
            return None
    return code


def _find_window_tiles(grid: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    found: dict[tuple[int, int], int] = {}
    for r in range(h - 2):
        for c in range(w - 2):
            code = _code_from_window(grid[r : r + 3, c : c + 3])
            if code is not None:
                found[(r, c)] = code
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the query window as the OR-combination of one top header and one left
    header, then fill the intersecting body cell in red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    windows = _find_window_tiles(g)
    assert len(windows) >= 3, "expected header windows and one query window"

    query_pos = max(windows, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    query_code = windows.pop(query_pos)

    min_row = min(r for r, _ in windows)
    min_col = min(c for _, c in windows)

    top_headers = sorted(
        [(c, code) for (r, c), code in windows.items() if r == min_row],
        key=lambda item: item[0],
    )
    left_headers = sorted(
        [(r, code) for (r, c), code in windows.items() if c == min_col],
        key=lambda item: item[0],
    )

    assert top_headers, "expected windows along the top header row"
    assert left_headers, "expected windows along the left header column"
    assert len(top_headers) == len(left_headers), "row and column counts must match"

    row_codes = [code for _, code in top_headers]
    col_codes = [code for _, code in left_headers]
    matches: list[tuple[int, int]] = []
    for row_idx, row_code in enumerate(row_codes):
        for col_idx, col_code in enumerate(col_codes):
            if (row_code | col_code) == query_code:
                matches.append((row_idx, col_idx))

    assert len(matches) == 1, "query decomposition must be unique"
    row_idx, col_idx = matches[0]

    body_row_starts = [r for r, _ in left_headers]
    body_col_starts = [c for c, _ in top_headers]

    out = g.copy()
    rr = body_row_starts[row_idx]
    cc = body_col_starts[col_idx]
    out[rr : rr + 3, cc : cc + 3] = RED
    return out


def _sample_codes(
    body_size: int, rng: random.Random
) -> tuple[tuple[int, ...], tuple[int, ...], int, int, int]:
    base_codes = list(range(1, 15))
    for _ in range(MAX_CODE_ATTEMPTS):
        rng.shuffle(base_codes)
        row_codes = tuple(base_codes[:body_size])
        col_codes = tuple(base_codes[body_size : 2 * body_size])

        unions: dict[int, list[tuple[int, int]]] = {}
        for row_idx, row_code in enumerate(row_codes):
            for col_idx, col_code in enumerate(col_codes):
                unions.setdefault(row_code | col_code, []).append((row_idx, col_idx))

        unique_unions = [
            (union_code, pairs[0]) for union_code, pairs in unions.items() if len(pairs) == 1
        ]
        if not unique_unions:
            continue

        query_code, (row_idx, col_idx) = rng.choice(unique_unions)
        return row_codes, col_codes, row_idx, col_idx, query_code

    raise RuntimeError("failed to sample a uniquely decodable query window")


def _build_canvas(body_size: int) -> tuple[np.ndarray, list[int], int]:
    main_cells = body_size + 1
    main_size = 4 * main_cells + 1
    full_size = main_size + 4

    grid = np.full((full_size, full_size), BLACK, dtype=int)
    starts = [1 + 4 * idx for idx in range(main_cells)]

    for r0 in starts:
        for c0 in starts:
            grid[r0 : r0 + 3, c0 : c0 + 3] = GREEN

    query_start = main_size + 1
    grid[query_start : query_start + 3, query_start : query_start + 3] = GREEN
    return grid, starts, query_start


def _make_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    body_size = rng.randint(BODY_SIZE_MIN, BODY_SIZE_MAX)
    row_codes, col_codes, row_idx, col_idx, query_code = _sample_codes(body_size, rng)

    grid, starts, query_start = _build_canvas(body_size)

    top_header_cols = starts[1:]
    left_header_rows = starts[1:]
    for idx, code in enumerate(row_codes):
        c0 = top_header_cols[idx]
        grid[starts[0] : starts[0] + 3, c0 : c0 + 3] = _window_from_code(code)
    for idx, code in enumerate(col_codes):
        r0 = left_header_rows[idx]
        grid[r0 : r0 + 3, starts[0] : starts[0] + 3] = _window_from_code(code)

    grid[query_start : query_start + 3, query_start : query_start + 3] = _window_from_code(
        query_code
    )

    expected = grid.copy()
    target_r = left_header_rows[row_idx]
    target_c = top_header_cols[col_idx]
    expected[target_r : target_r + 3, target_c : target_c + 3] = RED
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected = _make_example(rng)
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
