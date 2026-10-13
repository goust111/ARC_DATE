from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEPARATOR = 7
SEED_COLORS = (1, 2)
GEN_RETRIES = 400
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
LAYOUTS = [(4, 4), (4, 5), (4, 6), (5, 4)]  # (cell_count, cell_size), side <= 30.


def _cell_runs(lines: np.ndarray, limit: int) -> list[tuple[int, int]]:
    lines = np.asarray(lines, dtype=int)
    assert len(lines) >= 2, "need at least a border scaffold"
    assert int(lines[0]) == 0 and int(lines[-1]) == limit - 1, "scaffold must frame the array"
    runs = []
    for a, b in zip(lines[:-1], lines[1:]):
        assert int(b) > int(a) + 1, "cells must have positive area"
        runs.append((int(a) + 1, int(b) - 1))
    lengths = {r1 - r0 + 1 for r0, r1 in runs}
    assert len(lengths) == 1, "cells must all be the same size"
    return runs


def _swap_seed_color(color: int) -> int:
    if color == 1:
        return 2
    if color == 2:
        return 1
    return int(color)


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _bbox_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    pts = set(coords)
    start = next(iter(pts))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _transform_set(
    coords: tuple[tuple[int, int], ...], h: int, w: int, name: str
) -> tuple[tuple[int, int], ...]:
    if name == "id":
        out = coords
    elif name == "flipud":
        out = tuple((h - 1 - r, c) for r, c in coords)
    elif name == "fliplr":
        out = tuple((r, w - 1 - c) for r, c in coords)
    elif name == "rot180":
        out = tuple((h - 1 - r, w - 1 - c) for r, c in coords)
    elif name == "transpose":
        out = tuple((c, r) for r, c in coords)
    elif name == "anti":
        out = tuple((w - 1 - c, h - 1 - r) for r, c in coords)
    elif name == "rot90":
        out = tuple((c, h - 1 - r) for r, c in coords)
    elif name == "rot270":
        out = tuple((w - 1 - c, r) for r, c in coords)
    else:
        raise ValueError(name)
    return tuple(sorted(out))


def _is_fully_asymmetric(coords: tuple[tuple[int, int], ...]) -> bool:
    h, w = _bbox_dims(coords)
    tests = ["flipud", "fliplr", "rot180"]
    if h == w:
        tests.extend(["transpose", "anti", "rot90", "rot270"])
    return all(_transform_set(coords, h, w, name) != coords for name in tests)


def _enumerate_base_shapes() -> list[tuple[tuple[int, int], ...]]:
    shapes = set()
    cells = [(r, c) for r in range(3) for c in range(3)]
    for size in range(3, 7):
        for combo in itertools.combinations(cells, size):
            norm = _normalize(combo)
            h, w = _bbox_dims(norm)
            if h < 2 or w < 2:
                continue
            if not _is_connected(norm):
                continue
            if not _is_fully_asymmetric(norm):
                continue
            shapes.add(norm)
    assert shapes, "need asymmetric seed motifs"
    return sorted(shapes)


BASE_SHAPES = _enumerate_base_shapes()


def _build_scaffold(cell_count: int, cell_size: int) -> np.ndarray:
    side = cell_count * cell_size + cell_count + 1
    grid = np.full((side, side), BG, dtype=int)
    lines = [k * (cell_size + 1) for k in range(cell_count + 1)]
    grid[lines, :] = SEPARATOR
    grid[:, lines] = SEPARATOR
    return grid


def _shape_centroid(points: list[tuple[int, int, int]]) -> tuple[float, float]:
    return (
        sum(r for r, _, _ in points) / len(points),
        sum(c for _, c, _ in points) / len(points),
    )


def _direction_from_centroid(points: list[tuple[int, int, int]], cell_size: int) -> tuple[int, int]:
    center = (cell_size - 1) / 2.0
    centroid_r, centroid_c = _shape_centroid(points)
    assert centroid_r != center and centroid_c != center, "seed centroid must encode a unique direction"
    return (1 if centroid_r < center else -1, 1 if centroid_c < center else -1)


def _place_seed(
    grid: np.ndarray,
    cell_rows: list[tuple[int, int]],
    cell_cols: list[tuple[int, int]],
    cell_rc: tuple[int, int],
    points: list[tuple[int, int, int]],
) -> None:
    src_i, src_j = cell_rc
    r0 = cell_rows[src_i][0]
    c0 = cell_cols[src_j][0]
    for dr, dc, color in points:
        grid[r0 + dr, c0 + dc] = int(color)


def _propagate(
    grid: np.ndarray,
    cell_rows: list[tuple[int, int]],
    cell_cols: list[tuple[int, int]],
    cell_rc: tuple[int, int],
    points: list[tuple[int, int, int]],
    delta: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    src_i, src_j = cell_rc
    di, dj = delta
    steps = 1
    while True:
        dst_i = src_i + steps * di
        dst_j = src_j + steps * dj
        if not (0 <= dst_i < len(cell_rows) and 0 <= dst_j < len(cell_cols)):
            break
        r0 = cell_rows[dst_i][0]
        c0 = cell_cols[dst_j][0]
        for dr, dc, color in points:
            new_color = _swap_seed_color(color) if steps % 2 == 1 else int(color)
            out[r0 + dr, c0 + dc] = new_color
        steps += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the bicolored seed motif along the scaffold cell diagonal opposite its corner bias.

    Input structure:
    - Orange full-row and full-column separator lines frame an array of equal square cells.
    - Exactly one cell contains a blue/red seed motif on black background; all other cells are empty.
    - The seed motif is a small off-center connected shape whose position inside the cell is biased toward one corner.

    Rule:
    - Use the separator lines to recover the cell lattice and identify the unique seed cell.
    - Measure the seed motif centroid relative to the cell center; a seed biased toward one corner propagates toward the opposite diagonal direction.
    - Copy the seed into each later cell on that diagonal, swapping blue and red on odd propagation steps and keeping the original colors on even steps.

    Key insight:
    - The propagation direction is not marked explicitly; it is encoded by which corner of the source cell the seed cluster leans toward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == SEPARATOR), "missing scaffold separators"

    row_lines = np.flatnonzero(np.all(g == SEPARATOR, axis=1))
    col_lines = np.flatnonzero(np.all(g == SEPARATOR, axis=0))
    cell_rows = _cell_runs(row_lines, g.shape[0])
    cell_cols = _cell_runs(col_lines, g.shape[1])
    assert len(cell_rows) >= 2 and len(cell_cols) >= 2
    cell_h = cell_rows[0][1] - cell_rows[0][0] + 1
    cell_w = cell_cols[0][1] - cell_cols[0][0] + 1
    assert cell_h == cell_w, "cells must be square"

    seed_cells = []
    for i, (r0, r1) in enumerate(cell_rows):
        for j, (c0, c1) in enumerate(cell_cols):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            coords = np.argwhere((patch == SEED_COLORS[0]) | (patch == SEED_COLORS[1]))
            if len(coords) > 0:
                seed_cells.append((i, j, coords))
    assert len(seed_cells) == 1, "expected exactly one seed cell"

    src_i, src_j, coords = seed_cells[0]
    patch = g[
        cell_rows[src_i][0] : cell_rows[src_i][1] + 1,
        cell_cols[src_j][0] : cell_cols[src_j][1] + 1,
    ]
    points = [(int(r), int(c), int(patch[r, c])) for r, c in coords]
    colors = {color for _, _, color in points}
    assert colors == {1, 2}, "seed must use blue and red"

    delta = _direction_from_centroid(points, cell_h)
    return _propagate(g, cell_rows, cell_cols, (src_i, src_j), points, delta)


def _candidate_sources(cell_count: int, delta: tuple[int, int]) -> list[tuple[int, int]]:
    di, dj = delta
    corners = {
        (0, 0),
        (0, cell_count - 1),
        (cell_count - 1, 0),
        (cell_count - 1, cell_count - 1),
    }
    out = []
    for i in range(cell_count):
        for j in range(cell_count):
            if (i, j) in corners:
                continue
            max_i = cell_count - 1 - i if di == 1 else i
            max_j = cell_count - 1 - j if dj == 1 else j
            if min(max_i, max_j) >= 2:
                out.append((i, j))
    return out


def _color_shape(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int, int]]:
    coords = list(shape)
    random.shuffle(coords)
    color1_count = random.randint(1, len(coords) - 1)
    blue_cells = set(coords[:color1_count])
    return [(r, c, 1 if (r, c) in blue_cells else 2) for r, c in sorted(coords)]


def _biased_seed_points(
    cell_size: int, delta: tuple[int, int]
) -> list[tuple[int, int, int]] | None:
    shape = random.choice(BASE_SHAPES)
    points = _color_shape(shape)
    rows = [r for r, _, _ in points]
    cols = [c for _, c, _ in points]
    h = max(rows) + 1
    w = max(cols) + 1

    bias_top = delta[0] == 1
    bias_left = delta[1] == 1
    if bias_top:
        row_choices = range(0, min(cell_size - h, 1) + 1)
    else:
        row_choices = range(max(0, cell_size - h - 1), cell_size - h + 1)
    if bias_left:
        col_choices = range(0, min(cell_size - w, 1) + 1)
    else:
        col_choices = range(max(0, cell_size - w - 1), cell_size - w + 1)

    candidates = []
    for r0 in row_choices:
        for c0 in col_choices:
            shifted = [(r + r0, c + c0, color) for r, c, color in points]
            try:
                inferred = _direction_from_centroid(shifted, cell_size)
            except AssertionError:
                continue
            if inferred == delta:
                candidates.append(shifted)
    return random.choice(candidates) if candidates else None


def _oracle_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    cell_count, cell_size = random.choice(LAYOUTS)
    delta = random.choice(DIAGONALS)
    source_choices = _candidate_sources(cell_count, delta)
    if not source_choices:
        return None
    source_cell = random.choice(source_choices)
    points = _biased_seed_points(cell_size, delta)
    if points is None:
        return None

    inp = _build_scaffold(cell_count, cell_size)
    row_lines = np.flatnonzero(np.all(inp == SEPARATOR, axis=1))
    col_lines = np.flatnonzero(np.all(inp == SEPARATOR, axis=0))
    cell_rows = _cell_runs(row_lines, inp.shape[0])
    cell_cols = _cell_runs(col_lines, inp.shape[1])
    _place_seed(inp, cell_rows, cell_cols, source_cell, points)

    expected = _propagate(inp, cell_rows, cell_cols, source_cell, points, delta)
    propagation_steps = min(
        (cell_count - 1 - source_cell[0]) if delta[0] == 1 else source_cell[0],
        (cell_count - 1 - source_cell[1]) if delta[1] == 1 else source_cell[1],
    )
    assert propagation_steps >= 2
    if np.array_equal(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _oracle_attempt()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 10")
