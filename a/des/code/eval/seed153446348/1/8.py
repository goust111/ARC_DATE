from __future__ import annotations

import random

import numpy as np


BACKGROUND_COLOR = 0
MIRROR_COLOR = 5
MARK_COLOR = 2
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_mirror(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    row_candidates = [
        r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(row_candidates) + len(col_candidates) == 1, "expected exactly one mirror line"

    if row_candidates:
        axis = int(row_candidates[0])
        return "horizontal", axis, int(grid[axis, 0])

    axis = int(col_candidates[0])
    return "vertical", axis, int(grid[0, axis])


def _pattern_color(grid: np.ndarray, bg: int, mirror_color: int) -> int:
    colors = set(map(int, np.unique(grid))) - {bg, mirror_color}
    assert len(colors) == 1, "expected one pattern color"
    color = colors.pop()
    assert color != MARK_COLOR, "input pattern color must differ from the output marker color"
    return color


def _mirrored_coord(orientation: str, axis: int, r: int, c: int) -> tuple[int, int]:
    if orientation == "vertical":
        return r, 2 * axis - c
    return 2 * axis - r, c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the grey mirror line and mark one-sided pattern pixels in red.

    The input contains a black background, one horizontal or vertical grey
    mirror line, and one non-grey pattern color. Pattern pixels that have a
    mirrored partner across the line stay unchanged. Pattern pixels whose
    mirrored location is background become red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _dominant_color(g)
    orientation, axis, mirror_color = _detect_mirror(g, bg)
    pattern_color = _pattern_color(g, bg, mirror_color)

    out = g.copy()
    for r, c in np.argwhere(g == pattern_color):
        rr, cc = _mirrored_coord(orientation, axis, int(r), int(c))
        if not (0 <= rr < h and 0 <= cc < w) or int(g[rr, cc]) != pattern_color:
            out[r, c] = MARK_COLOR
    return out


def _neighbors(cell: tuple[int, int], u_len: int, depth: int) -> list[tuple[int, int]]:
    u, v = cell
    out: list[tuple[int, int]] = []
    if u > 0:
        out.append((u - 1, v))
    if u + 1 < u_len:
        out.append((u + 1, v))
    if v > 0:
        out.append((u, v - 1))
    if v + 1 < depth:
        out.append((u, v + 1))
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    pieces = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    random.shuffle(pieces)
    return pieces


def _grow_component(
    free: set[tuple[int, int]],
    u_len: int,
    depth: int,
    size: int,
    attach_pool: set[tuple[int, int]] | None,
    attach_bias: float,
) -> set[tuple[int, int]] | None:
    if len(free) < size:
        return None

    free = set(free)
    for _ in range(40):
        if attach_pool and random.random() < attach_bias:
            starts = [cell for cell in free if any(n in attach_pool for n in _neighbors(cell, u_len, depth))]
            start = random.choice(starts) if starts else random.choice(tuple(free))
        else:
            start = random.choice(tuple(free))

        comp = {start}
        frontier = {n for n in _neighbors(start, u_len, depth) if n in free}
        while len(comp) < size and frontier:
            options = list(frontier)
            weights = [
                1 + sum(nn in comp for nn in _neighbors(cell, u_len, depth))
                for cell in options
            ]
            cell = random.choices(options, weights=weights, k=1)[0]
            frontier.remove(cell)
            comp.add(cell)
            for nbr in _neighbors(cell, u_len, depth):
                if nbr in free and nbr not in comp:
                    frontier.add(nbr)

        if len(comp) == size:
            return comp
    return None


def _place_cells(
    all_cells: set[tuple[int, int]],
    u_len: int,
    depth: int,
    total: int,
    clusters: int,
    used: set[tuple[int, int]],
    attach_pool: set[tuple[int, int]] | None = None,
    attach_bias: float = 0.0,
) -> set[tuple[int, int]] | None:
    if total == 0:
        return set()
    if clusters > total:
        return None

    reserved = set(used)
    result: set[tuple[int, int]] = set()
    for size in _random_partition(total, clusters):
        free = all_cells - reserved
        pool = None if attach_pool is None else (set(attach_pool) | result)
        comp = _grow_component(free, u_len, depth, size, pool, attach_bias)
        if comp is None:
            return None
        result |= comp
        reserved |= comp
    return result


def _sample_canvas(orientation: str) -> tuple[int, int, int, int, int]:
    rows = random.randint(9, 16)
    cols = random.randint(9, 16)
    if orientation == "vertical":
        axis = random.randint(3, cols - 4)
        depth = min(axis, cols - axis - 1)
        return rows, cols, axis, rows, depth

    axis = random.randint(3, rows - 4)
    depth = min(axis, rows - axis - 1)
    return rows, cols, axis, cols, depth


def _to_grid(orientation: str, axis: int, side: int, u: int, v: int) -> tuple[int, int]:
    dist = v + 1
    if orientation == "vertical":
        return u, axis - dist if side == 0 else axis + dist
    return axis - dist if side == 0 else axis + dist, u


def _render_grid(
    rows: int,
    cols: int,
    orientation: str,
    axis: int,
    pattern_color: int,
    side_a: set[tuple[int, int]],
    side_b: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    if orientation == "vertical":
        grid[:, axis] = MIRROR_COLOR
    else:
        grid[axis, :] = MIRROR_COLOR

    for u, v in side_a:
        r, c = _to_grid(orientation, axis, 0, u, v)
        grid[r, c] = pattern_color
    for u, v in side_b:
        r, c = _to_grid(orientation, axis, 1, u, v)
        grid[r, c] = pattern_color
    return grid


def _render_expected(
    rows: int,
    cols: int,
    orientation: str,
    axis: int,
    pattern_color: int,
    side_a: set[tuple[int, int]],
    side_b: set[tuple[int, int]],
) -> np.ndarray:
    out = _render_grid(rows, cols, orientation, axis, pattern_color, side_a, side_b)
    for u, v in side_a - side_b:
        r, c = _to_grid(orientation, axis, 0, u, v)
        out[r, c] = MARK_COLOR
    for u, v in side_b - side_a:
        r, c = _to_grid(orientation, axis, 1, u, v)
        out[r, c] = MARK_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("vertical", "horizontal"))
    rows, cols, axis, u_len, depth = _sample_canvas(orientation)
    capacity = u_len * depth
    if capacity < 8:
        return None

    all_cells = {(u, v) for u in range(u_len) for v in range(depth)}
    base_total = random.randint(3, min(12, capacity - 2))
    base_clusters = random.randint(1, min(3, base_total))
    base = _place_cells(all_cells, u_len, depth, base_total, base_clusters, set())
    if not base:
        return None

    side_a = set(base)
    side_b = set(base)
    used = set(base)
    branch = random.choice((0, 1, 2))
    free_count = capacity - len(used)

    if branch in (0, 1):
        if free_count < 1:
            return None
        extra_total = random.randint(1, min(5, free_count))
        extra_clusters = random.randint(1, min(2, extra_total))
        extra = _place_cells(
            all_cells,
            u_len,
            depth,
            extra_total,
            extra_clusters,
            used,
            attach_pool=base,
            attach_bias=0.75,
        )
        if extra is None:
            return None
        if branch == 0:
            side_a |= extra
        else:
            side_b |= extra
        used |= extra
    else:
        if free_count < 2:
            return None
        extra_a_total = random.randint(1, min(3, free_count - 1))
        extra_a = _place_cells(
            all_cells,
            u_len,
            depth,
            extra_a_total,
            random.randint(1, min(2, extra_a_total)),
            used,
            attach_pool=base,
            attach_bias=0.75,
        )
        if extra_a is None:
            return None
        used |= extra_a

        free_count = capacity - len(used)
        if free_count < 1:
            return None
        extra_b_total = random.randint(1, min(3, free_count))
        extra_b = _place_cells(
            all_cells,
            u_len,
            depth,
            extra_b_total,
            random.randint(1, min(2, extra_b_total)),
            used,
            attach_pool=base,
            attach_bias=0.75,
        )
        if extra_b is None:
            return None
        side_a |= extra_a
        side_b |= extra_b
        used |= extra_b

    pattern_color = random.choice(PATTERN_COLORS)
    grid = _render_grid(rows, cols, orientation, axis, pattern_color, side_a, side_b)
    expected = _render_expected(rows, cols, orientation, axis, pattern_color, side_a, side_b)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.any(expected == MARK_COLOR):
        return None
    if not np.any(expected == pattern_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-difference input")
