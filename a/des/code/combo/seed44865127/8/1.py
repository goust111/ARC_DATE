import random

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
CELL_SIZES = (5, 7)
GRID_DIM_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _segments_from_dividers(length: int, dividers: list[int]) -> tuple[list[tuple[int, int]], int] | None:
    bounds = [-1] + dividers + [length]
    spans = [bounds[i + 1] - bounds[i] - 1 for i in range(len(bounds) - 1)]
    if any(span <= 0 for span in spans):
        return None
    if len(set(spans)) != 1:
        return None
    segments = [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    return segments, spans[0]


def _find_partition(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = grid.shape
    row_by_color: dict[int, list[int]] = {}
    col_by_color: dict[int, list[int]] = {}

    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            row_by_color.setdefault(color, []).append(r)
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            col_by_color.setdefault(color, []).append(c)

    candidates = []
    for color in sorted(set(row_by_color) & set(col_by_color)):
        row_info = _segments_from_dividers(h, sorted(row_by_color[color]))
        col_info = _segments_from_dividers(w, sorted(col_by_color[color]))
        if row_info is None or col_info is None:
            continue
        row_segments, cell_h = row_info
        col_segments, cell_w = col_info
        if len(row_segments) < 2 or len(col_segments) < 2:
            continue
        if cell_h != cell_w or cell_h < 3 or cell_h % 2 == 0:
            continue
        candidates.append((color, row_segments, col_segments, cell_h))

    assert len(candidates) == 1, "expected one valid square-cell partition"
    return candidates[0]


def _marker_positions(cell_size: int) -> dict[tuple[int, int], tuple[int, int]]:
    mid = cell_size // 2
    return {
        (0, mid): (-1, 0),
        (cell_size - 1, mid): (1, 0),
        (mid, 0): (0, -1),
        (mid, cell_size - 1): (0, 1),
    }


def _cell_patch(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    row_idx: int,
    col_idx: int,
) -> np.ndarray:
    r0, r1 = row_segments[row_idx]
    c0, c1 = col_segments[col_idx]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _infer_marker_color(
    grid: np.ndarray,
    bg: int,
    divider: int,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    cell_size: int,
) -> int:
    allowed = set(_marker_positions(cell_size))
    colors = [int(c) for c in np.unique(grid) if int(c) not in (bg, divider)]
    candidates: list[tuple[tuple[int, int], int]] = []

    for color in colors:
        total = 0
        ok = True
        for rr in range(len(row_segments)):
            for cc in range(len(col_segments)):
                patch = _cell_patch(grid, row_segments, col_segments, rr, cc)
                coords = np.argwhere(patch == color)
                if coords.size == 0:
                    continue
                total += int(coords.shape[0])
                if int(coords.shape[0]) != 1:
                    ok = False
                    break
                pos = tuple(map(int, coords[0]))
                if pos not in allowed:
                    ok = False
                    break
            if not ok:
                break
        if ok and total > 0:
            candidates.append(((1 if color == GREY else 0, total), color))

    assert candidates, "expected a marker color confined to cell-edge midpoints"
    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    best_colors = [color for score, color in candidates if score == best_score]
    assert len(best_colors) == 1, "marker color must be uniquely identifiable"
    return best_colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each marked cell's pattern repeatedly toward the grid edge indicated by its edge-midpoint marker.

    The grid is partitioned into equal square cells by full-row/full-column dividers.
    A single marker pixel in a cell sits at one side midpoint:
    top -> up, bottom -> down, left -> left, right -> right.
    The cell's pattern excluding the marker is copied into every later cell on that ray.
    Target cell interiors are fully replaced, dividers stay unchanged, and all markers are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    divider, row_segments, col_segments, cell_size = _find_partition(g, bg)
    marker_positions = _marker_positions(cell_size)
    marker = _infer_marker_color(g, bg, divider, row_segments, col_segments, cell_size)

    out = g.copy()
    sources: list[tuple[int, int, int, int, np.ndarray]] = []

    for rr in range(len(row_segments)):
        for cc in range(len(col_segments)):
            patch = _cell_patch(g, row_segments, col_segments, rr, cc)
            marker_coords = np.argwhere(patch == marker)
            assert int(marker_coords.shape[0]) <= 1, "each cell may contain at most one marker"
            if marker_coords.size == 0:
                continue

            pos = tuple(map(int, marker_coords[0]))
            assert pos in marker_positions, "marker must lie on a cell-edge midpoint"
            dr, dc = marker_positions[pos]

            stamp = patch.copy()
            stamp[pos] = bg
            sources.append((rr, cc, dr, dc, stamp))

            r0, _r1 = row_segments[rr]
            c0, _c1 = col_segments[cc]
            out[r0 + pos[0], c0 + pos[1]] = bg

    planned: dict[tuple[int, int], np.ndarray] = {}
    n_rows = len(row_segments)
    n_cols = len(col_segments)
    for rr, cc, dr, dc, stamp in sources:
        tr = rr + dr
        tc = cc + dc
        while 0 <= tr < n_rows and 0 <= tc < n_cols:
            key = (tr, tc)
            if key in planned:
                assert np.array_equal(planned[key], stamp), "conflicting overlapping stamps"
            else:
                planned[key] = stamp
            tr += dr
            tc += dc

    for (rr, cc), stamp in planned.items():
        r0, r1 = row_segments[rr]
        c0, c1 = col_segments[cc]
        out[r0 : r1 + 1, c0 : c1 + 1] = stamp

    return out


def _pattern_shapes() -> tuple[tuple[tuple[int, int], ...], ...]:
    return (
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    )


def _build_pattern(cell_size: int, bg: int, palette: list[int]) -> np.ndarray | None:
    patch = np.full((cell_size, cell_size), bg, dtype=int)
    blocked = set(_marker_positions(cell_size))
    occupied: set[tuple[int, int]] = set()
    shapes = list(_pattern_shapes())
    colors = random.sample(palette, random.randint(2, min(3, len(palette))))
    used_colors: set[int] = set()
    target_shapes = random.randint(len(colors), len(colors) + 2)

    for i in range(target_shapes):
        shape = random.choice(shapes)
        color = colors[i] if i < len(colors) else random.choice(colors)
        placed = False

        for _ in range(40):
            max_r = cell_size - max(dr for dr, _ in shape) - 1
            max_c = cell_size - max(dc for _, dc in shape) - 1
            r0 = random.randint(0, max_r)
            c0 = random.randint(0, max_c)
            coords = {(r0 + dr, c0 + dc) for dr, dc in shape}
            if coords & blocked or coords & occupied:
                continue
            for r, c in coords:
                patch[r, c] = color
            occupied |= coords
            used_colors.add(color)
            placed = True
            break

        if not placed and i < len(colors):
            return None

    if len(used_colors) < 2 or len(occupied) < 3:
        return None
    return patch


def _build_grid_shape(axis: int) -> tuple[int, int, int]:
    cell_size = random.choice(CELL_SIZES)
    n_rows = random.randint(*GRID_DIM_RANGE)
    n_cols = random.randint(*GRID_DIM_RANGE)
    if axis == 0:
        n_rows = max(n_rows, 2)
    else:
        n_cols = max(n_cols, 2)
    return cell_size, n_rows, n_cols


def _active_sources(axis: int, n_rows: int, n_cols: int) -> list[tuple[int, int, tuple[int, int]]]:
    sources: list[tuple[int, int, tuple[int, int]]] = []
    if axis == 0:
        active_cols = random.sample(range(n_cols), random.randint(1, n_cols))
        for col in active_cols:
            directions = []
            if n_rows >= 2:
                directions.extend(((-1, 0), (1, 0)))
            dr, dc = random.choice(directions)
            if dr < 0:
                row = random.randint(1, n_rows - 1)
            else:
                row = random.randint(0, n_rows - 2)
            sources.append((row, col, (dr, dc)))
    else:
        active_rows = random.sample(range(n_rows), random.randint(1, n_rows))
        for row in active_rows:
            directions = []
            if n_cols >= 2:
                directions.extend(((0, -1), (0, 1)))
            dr, dc = random.choice(directions)
            if dc < 0:
                col = random.randint(1, n_cols - 1)
            else:
                col = random.randint(0, n_cols - 2)
            sources.append((row, col, (dr, dc)))
    return sources


def _direction_to_marker(cell_size: int, direction: tuple[int, int]) -> tuple[int, int]:
    reverse = {value: key for key, value in _marker_positions(cell_size).items()}
    return reverse[direction]


def _place_cells(
    cell_size: int,
    n_rows: int,
    n_cols: int,
    bg: int,
    divider: int,
    pattern_palette: list[int],
    sources: list[tuple[int, int, tuple[int, int]]],
) -> np.ndarray | None:
    h = n_rows * cell_size + (n_rows - 1)
    w = n_cols * cell_size + (n_cols - 1)
    grid = np.full((h, w), bg, dtype=int)

    for r in range(1, n_rows):
        grid[r * cell_size + (r - 1), :] = divider
    for c in range(1, n_cols):
        grid[:, c * cell_size + (c - 1)] = divider

    source_lookup = {(r, c): direction for r, c, direction in sources}
    for rr in range(n_rows):
        for cc in range(n_cols):
            patch = _build_pattern(cell_size, bg, pattern_palette)
            if patch is None:
                return None
            if (rr, cc) in source_lookup:
                mr, mc = _direction_to_marker(cell_size, source_lookup[(rr, cc)])
                patch[mr, mc] = GREY
            r0 = rr * (cell_size + 1)
            c0 = cc * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch

    return grid


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice((0, 1))
    cell_size, n_rows, n_cols = _build_grid_shape(axis)

    bg_choices = [color for color in PALETTE if color != GREY]
    bg = random.choice(bg_choices)
    divider_choices = [color for color in PALETTE if color not in (bg, GREY)]
    divider = random.choice(divider_choices)
    pattern_palette = [color for color in PALETTE if color not in (bg, divider, GREY)]
    if len(pattern_palette) < 3:
        return None

    sources = _active_sources(axis, n_rows, n_cols)
    grid = _place_cells(cell_size, n_rows, n_cols, bg, divider, pattern_palette, sources)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
