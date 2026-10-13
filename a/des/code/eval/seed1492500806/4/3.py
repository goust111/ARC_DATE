import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2))
TARGET_SIZE_RANGE = (1, 4)
MAX_INTERIOR_SIDE = 28
GEN_ATTEMPTS = 400


def _parse_ruler(line: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    i = 0
    n = int(line.size)
    while i < n:
        color = int(line[i])
        if color == BLACK:
            i += 1
            continue
        j = i + 1
        while j < n and int(line[j]) == color:
            j += 1
        assert color not in mapping, "each ruler color must appear in one contiguous segment"
        mapping[color] = j - i
        i = j
    assert mapping, "expected at least one colored ruler segment"
    return mapping


def _band_edges_from_rows(interior: np.ndarray) -> list[int]:
    h = int(interior.shape[0])
    edges = [0]
    for r in range(1, h):
        if not np.array_equal(interior[r], interior[r - 1]):
            edges.append(r)
    edges.append(h)
    return edges


def _band_edges_from_cols(interior: np.ndarray) -> list[int]:
    w = int(interior.shape[1])
    edges = [0]
    for c in range(1, w):
        if not np.array_equal(interior[:, c], interior[:, c - 1]):
            edges.append(c)
    edges.append(w)
    return edges


def _cell_colors(interior: np.ndarray, row_edges: list[int], col_edges: list[int]) -> np.ndarray:
    rows = len(row_edges) - 1
    cols = len(col_edges) - 1
    colors = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        r0, r1 = row_edges[i], row_edges[i + 1]
        for j in range(cols):
            c0, c1 = col_edges[j], col_edges[j + 1]
            patch = interior[r0:r1, c0:c1]
            color = int(patch[0, 0])
            assert color != BLACK, "interior must be fully tiled by colored rectangles"
            assert np.all(patch == color), "each grid cell must be a solid rectangle"
            colors[i, j] = color
    return colors


def _render(color_grid: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    out = np.zeros((sum(row_sizes), sum(col_sizes)), dtype=int)
    r0 = 0
    for i, height in enumerate(row_sizes):
        c0 = 0
        for j, width in enumerate(col_sizes):
            out[r0 : r0 + height, c0 : c0 + width] = int(color_grid[i, j])
            c0 += width
        r0 += height
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top-row and left-column color rulers, then resize each interior
    rectangular block to the width/height encoded by the matching ruler color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 3, "grid must contain rulers and interior"

    width_by_color = _parse_ruler(g[0, 1:])
    height_by_color = _parse_ruler(g[1:, 0])

    interior = g[1:, 1:]
    row_edges = _band_edges_from_rows(interior)
    col_edges = _band_edges_from_cols(interior)
    colors = _cell_colors(interior, row_edges, col_edges)

    interior_colors = {int(v) for v in np.unique(colors)}
    assert interior_colors <= set(width_by_color), "missing top ruler colors"
    assert interior_colors <= set(height_by_color), "missing left ruler colors"

    row_sizes: list[int] = []
    for i in range(colors.shape[0]):
        vals = {height_by_color[int(color)] for color in colors[i]}
        assert len(vals) == 1, "all blocks in one source row band must share the target height"
        row_sizes.append(vals.pop())

    col_sizes: list[int] = []
    for j in range(colors.shape[1]):
        vals = {width_by_color[int(color)] for color in colors[:, j]}
        assert len(vals) == 1, "all blocks in one source column band must share the target width"
        col_sizes.append(vals.pop())

    return _render(colors, row_sizes, col_sizes)


def _random_sizes(rng: random.Random, count: int) -> list[int]:
    lo, hi = TARGET_SIZE_RANGE
    return [rng.randint(lo, hi) for _ in range(count)]


def _random_composition(rng: random.Random, total: int, parts: int) -> list[int]:
    assert total >= parts
    remaining = total
    out: list[int] = []
    for i in range(parts - 1):
        max_here = remaining - (parts - i - 1)
        value = rng.randint(1, max_here)
        out.append(value)
        remaining -= value
    out.append(remaining)
    rng.shuffle(out)
    return out


def _sample_source_sizes(rng: random.Random, total: int, targets: list[int]) -> list[int]:
    require_expansion = rng.random() < 0.5 and any(target > 1 for target in targets)
    last = None
    for _ in range(200):
        sizes = _random_composition(rng, total, len(targets))
        last = sizes
        if not any(size > target for size, target in zip(sizes, targets)):
            continue
        if require_expansion and not any(size < target for size, target in zip(sizes, targets)):
            continue
        return sizes
    assert last is not None
    return last


def _build_expected(color_grid: np.ndarray, target_rows: list[int], target_cols: list[int]) -> np.ndarray:
    return _render(color_grid, target_rows, target_cols)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows, n_cols = LAYOUT_OPTIONS[rng.randrange(len(LAYOUT_OPTIONS))]
    target_rows = _random_sizes(rng, n_rows)
    target_cols = _random_sizes(rng, n_cols)

    interior_h = n_cols * sum(target_rows)
    interior_w = n_rows * sum(target_cols)
    if not (4 <= interior_h <= MAX_INTERIOR_SIDE and 4 <= interior_w <= MAX_INTERIOR_SIDE):
        return None

    source_rows = _sample_source_sizes(rng, interior_h, target_rows)
    source_cols = _sample_source_sizes(rng, interior_w, target_cols)

    colors = rng.sample(PALETTE, n_rows * n_cols)
    color_grid = np.array(colors, dtype=int).reshape(n_rows, n_cols)

    width_by_color = {
        int(color_grid[i, j]): int(target_cols[j]) for i in range(n_rows) for j in range(n_cols)
    }
    height_by_color = {
        int(color_grid[i, j]): int(target_rows[i]) for i in range(n_rows) for j in range(n_cols)
    }

    grid = np.zeros((interior_h + 1, interior_w + 1), dtype=int)
    grid[1:, 1:] = _render(color_grid, source_rows, source_cols)

    top_order = colors[:]
    left_order = colors[:]
    rng.shuffle(top_order)
    rng.shuffle(left_order)

    c = 1
    for color in top_order:
        width = width_by_color[int(color)]
        grid[0, c : c + width] = int(color)
        c += width
    assert c == grid.shape[1], "top ruler must exactly fill the available row"

    r = 1
    for color in left_order:
        height = height_by_color[int(color)]
        grid[r : r + height, 0] = int(color)
        r += height
    assert r == grid.shape[0], "left ruler must exactly fill the available column"

    expected = _build_expected(color_grid, target_rows, target_cols)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
