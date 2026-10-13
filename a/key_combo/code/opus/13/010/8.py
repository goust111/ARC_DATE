import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == BLACK)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == BLACK)]
    keep_rows = [r for r in range(g.shape[0]) if r not in set(sep_rows)]
    keep_cols = [c for c in range(g.shape[1]) if c not in set(sep_cols)]
    assert keep_rows and keep_cols, "compression removed the whole grid"
    return g[np.ix_(keep_rows, keep_cols)]


def _square_data(marker_coords: np.ndarray) -> tuple[int, int, int, int, set[tuple[int, int]], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in marker_coords]
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    assert len(rows) == 2 and len(cols) == 2, "markers must occupy two rows and two cols"
    r0, r1 = rows
    c0, c1 = cols
    side = r1 - r0
    assert side == c1 - c0, "compressed markers must define a square"
    assert side >= 4 and side % 2 == 0, "square side must be even and at least 4"
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    points = set(coords)
    assert len(points) == 3 and points < corners, "markers must be exactly three square corners"
    missing = next(iter(corners - points))
    return r0, r1, c0, c1, corners, missing


def _solve_compressed(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(set(map(int, g.ravel())) - {BLACK, GREY})
    assert len(colors) == 1, "expected one marker color after compression"
    marker = int(colors[0])
    assert np.all((g == GREY) | (g == marker)), "compressed grid may only contain grey background and markers"

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.shape[0] == 3, "expected exactly three markers"
    r0, r1, c0, c1, _corners, missing = _square_data(marker_coords)

    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    out = g.copy()
    out[missing] = marker
    out[center_r, center_c] = BLACK

    for corner_r in (r0, r1):
        for corner_c in (c0, c1):
            dr = -1 if corner_r < center_r else 1
            dc = -1 if corner_c < center_c else 1
            out[center_r + dr, center_c + dc] = marker

    return out


def _is_three_square_corners(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 3 or len(set(coords)) != 3:
        return False
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) != 2 or len(cols) != 2:
        return False
    corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    return set(coords) < corners and rows[1] - rows[0] == cols[1] - cols[0]


def _axis_layout(length: int, sep_gaps: list[int]) -> tuple[list[int], list[int], int]:
    keep = []
    separators = []
    sep_set = set(sep_gaps)
    raw_index = 0
    for idx in range(length):
        keep.append(raw_index)
        raw_index += 1
        if idx < length - 1 and idx in sep_set:
            separators.append(raw_index)
            raw_index += 1
    return keep, separators, raw_index


def _build_raw_grid(
    compressed: np.ndarray,
    row_sep_gaps: list[int],
    col_sep_gaps: list[int],
) -> np.ndarray:
    row_keep, sep_rows, raw_h = _axis_layout(compressed.shape[0], row_sep_gaps)
    col_keep, sep_cols, raw_w = _axis_layout(compressed.shape[1], col_sep_gaps)
    raw = np.full((raw_h, raw_w), GREY, dtype=int)
    raw[sep_rows, :] = BLACK
    raw[:, sep_cols] = BLACK

    for r in range(compressed.shape[0]):
        for c in range(compressed.shape[1]):
            raw[row_keep[r], col_keep[c]] = int(compressed[r, c])
    return raw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress away full-black separator lines, reveal the hidden square corners, and draw a one-step inward copy around the square center.

    Input structure:
    - The grid has black full-row and full-column separators scattered through a grey field.
    - Outside those separator lines, the non-separator cells are grey except for exactly three same-colored single-pixel markers.
    - In raw coordinates the three markers do not form an axis-aligned square, but after removing every full-black row and column they become three corners of one axis-aligned square.

    Rule:
    - Identify every full-black row and column and remove them, keeping the remaining cells in their original order.
    - In the compressed grid, treat the three colored pixels as three corners of a square and infer the missing fourth corner.
    - Paint that missing corner in the same color, paint the square center black, and paint the four diagonal neighbors of the center toward the four corners in the marker color.
    - Return only the compressed grid; do not restore separator lines.

    Key insight:
    The separator lines are not noise: deleting them changes the coordinate system so the three markers finally become square corners, which then determines both the missing corner and the miniature copy around the center.
    """
    return _solve_compressed(_compress_grid(np.asarray(grid, dtype=int)))


def _sample_compressed_grid() -> tuple[np.ndarray, list[tuple[int, int]]]:
    height = random.randint(9, 14)
    width = random.randint(9, 14)

    valid_sides = []
    max_side = min(height, width) - 3
    for side in range(4, max_side + 1, 2):
        if height - side >= 3 and width - side >= 3:
            valid_sides.append(side)
    assert valid_sides

    side = random.choice(valid_sides)
    top = random.randint(1, height - side - 2)
    left = random.randint(1, width - side - 2)

    corners = [
        (top, left),
        (top, left + side),
        (top + side, left),
        (top + side, left + side),
    ]
    missing_index = random.randrange(4)
    marker_coords = [coord for i, coord in enumerate(corners) if i != missing_index]
    marker_color = random.choice(MARKER_PALETTE)

    compressed = np.full((height, width), GREY, dtype=int)
    for r, c in marker_coords:
        compressed[r, c] = marker_color
    return compressed, marker_coords


def _sample_separator_gaps(
    height: int,
    width: int,
    marker_coords: list[tuple[int, int]],
) -> tuple[list[int], list[int]] | None:
    rows = sorted({r for r, _c in marker_coords})
    cols = sorted({c for _r, c in marker_coords})
    r0, r1 = rows
    c0, c1 = cols

    row_gap_pool = list(range(height - 1))
    col_gap_pool = list(range(width - 1))
    total_lines = random.randint(2, 4)

    for _ in range(80):
        row_count = random.randint(0, total_lines)
        col_count = total_lines - row_count
        if row_count > len(row_gap_pool) or col_count > len(col_gap_pool):
            continue
        row_gaps = sorted(random.sample(row_gap_pool, row_count))
        col_gaps = sorted(random.sample(col_gap_pool, col_count))
        row_between = sum(r0 <= gap < r1 for gap in row_gaps)
        col_between = sum(c0 <= gap < c1 for gap in col_gaps)
        if row_between == col_between:
            continue
        return row_gaps, col_gaps
    return None


def _gen_attempt() -> np.ndarray | None:
    compressed, marker_coords = _sample_compressed_grid()
    sampled = _sample_separator_gaps(compressed.shape[0], compressed.shape[1], marker_coords)
    if sampled is None:
        return None
    row_gaps, col_gaps = sampled
    raw = _build_raw_grid(compressed, row_gaps, col_gaps)

    raw_marker_color = int(next(iter(set(map(int, compressed.ravel())) - {GREY})))
    raw_marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(raw == raw_marker_color)]
    if _is_three_square_corners(raw_marker_coords):
        return None
    if np.any(np.all(raw == BLACK, axis=1)) != (len(row_gaps) > 0):
        return None
    if np.any(np.all(raw == BLACK, axis=0)) != (len(col_gaps) > 0):
        return None

    expected = _solve_compressed(compressed)
    try:
        solved = main(raw)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return raw


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
