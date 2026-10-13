from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PARTITION_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_divider_column(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected one full-height grey divider column"
    return int(candidates[0])


def _partition_starts(area: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = area.shape
    row_starts = [0]
    for r in range(1, h):
        if np.any(area[r] != area[r - 1]):
            row_starts.append(r)
    col_starts = [0]
    for c in range(1, w):
        if np.any(area[:, c] != area[:, c - 1]):
            col_starts.append(c)
    return row_starts, col_starts


def _validate_partition(area: np.ndarray, row_starts: list[int], col_starts: list[int]) -> None:
    row_ends = row_starts[1:] + [area.shape[0]]
    col_ends = col_starts[1:] + [area.shape[1]]
    for i, (r0, r1) in enumerate(zip(row_starts, row_ends)):
        for j, (c0, c1) in enumerate(zip(col_starts, col_ends)):
            block = area[r0:r1, c0:c1]
            assert block.size > 0
            color = int(block[0, 0])
            assert color != BLACK, "partition blocks must be non-black"
            assert np.all(block == color), "each partition cell must be monochrome"
            if i + 1 < len(row_starts):
                below = area[row_starts[i + 1], c0:c1]
                if np.any(below != color):
                    break
        else:
            continue
    assert len(row_starts) >= 3 and len(col_starts) >= 3, "expected a 3x3 to 5x5 partition grid"


def _extract_compact_matrix(grid: np.ndarray) -> tuple[np.ndarray, int]:
    divider_c = _find_divider_column(grid)
    left = grid[:, :divider_c]
    bb = _bbox(left != BLACK)
    assert bb is not None, "expected a colored partition on the left"
    r0, c0, r1, c1 = bb
    area = left[r0 : r1 + 1, c0 : c1 + 1]
    row_starts, col_starts = _partition_starts(area)
    _validate_partition(area, row_starts, col_starts)
    compact = area[np.ix_(row_starts, col_starts)]
    return compact, divider_c


def _spiral_clockwise_from_tl(matrix: np.ndarray) -> list[int]:
    top, bottom = 0, matrix.shape[0] - 1
    left, right = 0, matrix.shape[1] - 1
    out: list[int] = []
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            out.append(int(matrix[top, c]))
        top += 1
        for r in range(top, bottom + 1):
            out.append(int(matrix[r, right]))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                out.append(int(matrix[bottom, c]))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                out.append(int(matrix[r, left]))
            left += 1
    return out


def _spiral_counterclockwise_from_tl(matrix: np.ndarray) -> list[int]:
    top, bottom = 0, matrix.shape[0] - 1
    left, right = 0, matrix.shape[1] - 1
    out: list[int] = []
    while top <= bottom and left <= right:
        for r in range(top, bottom + 1):
            out.append(int(matrix[r, left]))
        left += 1
        for c in range(left, right + 1):
            out.append(int(matrix[bottom, c]))
        bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                out.append(int(matrix[r, right]))
            right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                out.append(int(matrix[top, c]))
            top += 1
    return out


def _dedupe_adjacent(seq: list[int]) -> list[int]:
    palette: list[int] = []
    for color in seq:
        if not palette or palette[-1] != color:
            palette.append(int(color))
    assert palette, "palette must be non-empty"
    return palette


def _find_square_bbox(grid: np.ndarray, divider_c: int) -> tuple[int, int, int, int]:
    right = grid[:, divider_c + 1 :]
    bb = _bbox(right == MAROON)
    assert bb is not None, "expected one maroon square on the right"
    rr0, cc0, rr1, cc1 = bb
    r0, c0, r1, c1 = rr0, divider_c + 1 + cc0, rr1, divider_c + 1 + cc1
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h >= 5, "maroon border must be a square frame"
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                assert grid[r, c] == MAROON, "square perimeter must be maroon"
            else:
                assert grid[r, c] != MAROON, "maroon cells may only lie on the square border"
    extra_maroon = np.argwhere(grid == MAROON)
    for r, c in extra_maroon:
        assert r0 <= r <= r1 and c0 <= c <= c1, "unexpected extra maroon pixels"
    return r0, c0, r1, c1


def _fill_square_rings(out: np.ndarray, square_bb: tuple[int, int, int, int], palette: list[int]) -> None:
    r0, c0, r1, c1 = square_bb
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            depth = min(r - r0 - 1, r1 - r - 1, c - c0 - 1, c1 - c - 1)
            out[r, c] = int(palette[depth % len(palette)])


def _ring_sequence(palette: list[int], max_depth: int) -> tuple[int, ...]:
    return tuple(int(palette[d % len(palette)]) for d in range(max_depth + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left partition grid into a spiral palette and use it to fill the right square with concentric rings.

    Input structure:
    - A black background is split by a full-height grey divider column into a left partition area and a right framed square.
    - The left side contains a tightly packed rectangular grid of monochrome color blocks with shared row heights and shared column widths.
    - The right side contains a maroon 1-pixel square border whose interior is black.

    Rule:
    - Recover the left partition rows and columns from its visible color-change boundaries, then sample each block's upper-left color to obtain the compact partition matrix.
    - Read that compact matrix in clockwise spiral order starting at its top-left cell, and remove only adjacent repeated colors to obtain the palette.
    - Keep the left side and the grey divider unchanged, keep the maroon border unchanged, and fill the square interior by concentric square rings using palette[depth mod len(palette)], where depth is the pixel's Chebyshev distance from the square's inner boundary.

    Key insight:
    - The partition must first be compressed to one color per block, because the compact matrix's spatial arrangement determines the spiral palette order that drives the ring colors.
    """
    g = np.asarray(grid, dtype=int)
    compact, divider_c = _extract_compact_matrix(g)
    palette = _dedupe_adjacent(_spiral_clockwise_from_tl(compact))
    square_bb = _find_square_bbox(g, divider_c)
    out = g.copy()
    _fill_square_rings(out, square_bb, palette)
    return out


def _sample_sizes(rng: random.Random, count: int, lo: int, hi: int, max_total: int) -> list[int] | None:
    for _ in range(200):
        sizes = [rng.randint(lo, hi) for _ in range(count)]
        if sum(sizes) <= max_total:
            return sizes
    return None


def _boundary_visibility_ok(matrix: np.ndarray) -> bool:
    rows, cols = matrix.shape
    for r in range(rows - 1):
        if not np.any(matrix[r] != matrix[r + 1]):
            return False
    for c in range(cols - 1):
        if not np.any(matrix[:, c] != matrix[:, c + 1]):
            return False
    return True


def _render_partition(matrix: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    h = sum(row_sizes)
    w = sum(col_sizes)
    patch = np.zeros((h, w), dtype=int)
    r = 0
    for i, rh in enumerate(row_sizes):
        c = 0
        for j, cw in enumerate(col_sizes):
            patch[r : r + rh, c : c + cw] = int(matrix[i, j])
            c += cw
        r += rh
    return patch


def _build_square(side: int) -> np.ndarray:
    square = np.zeros((side, side), dtype=int)
    square[0, :] = MAROON
    square[-1, :] = MAROON
    square[:, 0] = MAROON
    square[:, -1] = MAROON
    return square


def _random_compact_matrix(rng: random.Random, rows: int, cols: int) -> np.ndarray | None:
    max_depth = max(rows, cols)
    for _ in range(400):
        matrix = np.array(rng.choices(PARTITION_COLORS, k=rows * cols), dtype=int).reshape(rows, cols)
        if len(set(map(int, matrix.ravel()))) < 4:
            continue
        if not _boundary_visibility_ok(matrix):
            continue
        spiral = _spiral_clockwise_from_tl(matrix)
        if not any(a == b for a, b in zip(spiral, spiral[1:])):
            continue
        palette = _dedupe_adjacent(spiral)
        if len(palette) < 3:
            continue
        true_rings = _ring_sequence(palette, max_depth)
        if len(set(true_rings)) < 3:
            continue

        row_major = _dedupe_adjacent(matrix.ravel().tolist())
        ccw = _dedupe_adjacent(_spiral_counterclockwise_from_tl(matrix))
        no_dedup = spiral
        if true_rings == _ring_sequence(row_major, max_depth):
            continue
        if true_rings == _ring_sequence(ccw, max_depth):
            continue
        if true_rings == _ring_sequence(no_dedup, max_depth):
            continue
        return matrix
    return None


def _place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch


def _make_input_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 5)
    cols = rng.randint(3, 5)
    matrix = _random_compact_matrix(rng, rows, cols)
    if matrix is None:
        return None

    row_sizes = _sample_sizes(rng, rows, 2, 4, 16)
    col_sizes = _sample_sizes(rng, cols, 1, 3, 10)
    if row_sizes is None or col_sizes is None:
        return None
    if len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1:
        return None

    partition = _render_partition(matrix, row_sizes, col_sizes)
    side = 2 * max(rows, cols) + 3
    square = _build_square(side)

    left_region_w = partition.shape[1] + rng.randint(0, 1) + rng.randint(0, 1)
    right_region_w = side + rng.randint(1, 2) + rng.randint(1, 2)
    total_w = left_region_w + 1 + right_region_w
    if total_w > 30:
        return None

    min_h = max(partition.shape[0] + rng.randint(0, 2) + rng.randint(0, 2), side + rng.randint(1, 2) + rng.randint(1, 2))
    if min_h > 30:
        return None
    total_h = min_h + rng.randint(0, min(3, 30 - min_h))

    divider_c = left_region_w
    left_max_col = left_region_w - partition.shape[1]
    right_origin = divider_c + 1
    right_region_w_actual = total_w - right_origin
    right_max_col = right_region_w_actual - side
    if left_max_col < 0 or right_max_col < 0:
        return None

    part_top = rng.randint(0, total_h - partition.shape[0])
    part_left = rng.randint(0, left_max_col)
    square_top = rng.randint(0, total_h - side)
    square_left = right_origin + rng.randint(0, right_max_col)

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[:, divider_c] = GREY
    _place_patch(grid, partition, part_top, part_left)
    _place_patch(grid, square, square_top, square_left)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        compact, divider_c = _extract_compact_matrix(grid)
        square_bb = _find_square_bbox(grid, divider_c)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(grid, out):
        return False

    rows, cols = compact.shape
    if not (3 <= rows <= 5 and 3 <= cols <= 5):
        return False
    if len(set(map(int, compact.ravel()))) < 4:
        return False
    if not _boundary_visibility_ok(compact):
        return False

    spiral = _spiral_clockwise_from_tl(compact)
    if not any(a == b for a, b in zip(spiral, spiral[1:])):
        return False
    palette = _dedupe_adjacent(spiral)
    max_depth = max(rows, cols)
    true_rings = _ring_sequence(palette, max_depth)
    if len(set(true_rings)) < 3:
        return False

    row_major = _dedupe_adjacent(compact.ravel().tolist())
    ccw = _dedupe_adjacent(_spiral_counterclockwise_from_tl(compact))
    no_dedup = spiral
    if true_rings == _ring_sequence(row_major, max_depth):
        return False
    if true_rings == _ring_sequence(ccw, max_depth):
        return False
    if true_rings == _ring_sequence(no_dedup, max_depth):
        return False

    r0, c0, r1, c1 = square_bb
    if r1 - r0 + 1 != 2 * max(rows, cols) + 3:
        return False
    seen = set()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            depth = min(r - r0 - 1, r1 - r - 1, c - c0 - 1, c1 - c - 1)
            expected = palette[depth % len(palette)]
            if int(out[r, c]) != int(expected):
                return False
            seen.add(int(out[r, c]))
    if len(seen) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) + attempt)
        grid = _make_input_attempt(rng)
        if grid is None:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
