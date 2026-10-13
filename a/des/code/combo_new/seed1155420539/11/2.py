import random

import numpy as np


BLACK = 0
BLUE = 1
MAROON = 9
FILL_COLORS = (2, 3, 4, 5, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _non_separator_spans(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    sep = {int(x) for x in separators}
    spans: list[tuple[int, int]] = []
    start = None
    for idx in range(length):
        if idx in sep:
            if start is not None:
                spans.append((start, idx - 1))
                start = None
            continue
        if start is None:
            start = idx
    if start is not None:
        spans.append((start, length - 1))
    return spans


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates = []
    for color in map(int, np.unique(grid)):
        sep_rows = np.where(np.all(grid == color, axis=1))[0]
        sep_cols = np.where(np.all(grid == color, axis=0))[0]
        if len(sep_rows) == 0 or len(sep_cols) == 0:
            continue
        row_spans = _non_separator_spans(grid.shape[0], sep_rows)
        col_spans = _non_separator_spans(grid.shape[1], sep_cols)
        if len(row_spans) < 1 or len(col_spans) < 2:
            continue
        if len({r1 - r0 + 1 for r0, r1 in row_spans}) != 1:
            continue
        if len({c1 - c0 + 1 for c0, c1 in col_spans}) != 1:
            continue
        candidates.append((color, row_spans, col_spans))
    assert len(candidates) == 1, "expected a unique separator color and equal tile partition"
    return candidates[0]


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    return tuple(sorted((int(r - rows.min()), int(c - cols.min())) for r, c in zip(rows, cols)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each non-empty work tile background with the color named by the matching legend shape.

    Input structure:
    - The background is black.
    - Full maroon rows and columns partition the canvas into equal rectangular tiles.
    - The leftmost tile column is a legend: each legend tile contains one small blue seed shape and one non-blue color pixel in the bottom-right corner.
    - Every other tile is either all black or contains a translated copy of one legend blue shape.

    Rule:
    - Normalize each blue shape by translating it so its top-left occupied cell becomes (0, 0).
    - In the legend column, map each normalized blue shape to the bottom-right color pixel in the same tile.
    - For every non-empty work tile, find the matching legend entry and recolor only that tile's black background cells to the mapped color.
    - Keep the blue seed pixels, legend tiles, empty tiles, and separator lines unchanged.

    Key insight:
    - The left column is a shape-to-color dictionary; matching must ignore absolute placement inside a tile and compare only normalized blue-shape geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_spans, col_spans = _separator_structure(g)
    bg = _mode_color(g[g != separator])
    legend_c0, legend_c1 = col_spans[0]
    work_cols = col_spans[1:]
    assert work_cols, "expected at least one work-column tile"

    seed_colors: set[int] = set()
    for r0, r1 in row_spans:
        for c0, c1 in work_cols:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(tile == separator), "separator color must stay outside tile interiors"
            colors = {int(v) for v in np.unique(tile)}
            colors.discard(bg)
            seed_colors.update(colors)
    assert len(seed_colors) == 1, "expected a single work-area seed color"
    seed = next(iter(seed_colors))
    assert seed not in {bg, separator}

    legend: dict[tuple[tuple[int, int], ...], int] = {}
    for r0, r1 in row_spans:
        tile = g[r0 : r1 + 1, legend_c0 : legend_c1 + 1]
        label = int(tile[-1, -1])
        assert label not in {bg, seed, separator}, "legend label must be a distinct non-background color"
        assert np.count_nonzero(tile == label) == 1, "legend label must be a singleton pixel"

        shape = np.argwhere(tile == seed)
        assert 2 <= len(shape) <= 5, "legend shapes must be small blue clusters"
        key = _normalize_shape(shape)
        assert key not in legend, "legend shapes must be unique"

        leftovers = {int(v) for v in np.unique(tile)} - {bg, seed, label}
        assert not leftovers, "legend tile contains unexpected colors"
        legend[key] = label

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in work_cols:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            shape = np.argwhere(tile == seed)
            if len(shape) == 0:
                assert np.all(tile == bg), "empty work tiles must be pure background"
                continue

            colors = {int(v) for v in np.unique(tile)}
            assert colors <= {bg, seed}, "work tiles may only contain background and seed pixels"
            key = _normalize_shape(shape)
            assert key in legend, "every work-tile shape must appear in the legend"

            fill = legend[key]
            new_tile = tile.copy()
            new_tile[tile == bg] = fill
            out[r0 : r1 + 1, c0 : c1 + 1] = new_tile
    return out


# --------------- generator ---------------

TILE_SIZE_CHOICES = (4, 5, 6)
MAX_TILE_ROWS = 4
MAX_TILE_COLS = 5
GEN_RETRIES = 400


def _canon_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _enumerate_shapes(max_size: int = 5) -> tuple[tuple[tuple[int, int], ...], ...]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, max_size + 1):
        shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in by_size[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            for nxt in frontier:
                shapes.add(_canon_shape(cells | {nxt}))
        by_size[size] = shapes
    out = []
    for size in range(2, max_size + 1):
        out.extend(sorted(by_size[size]))
    return tuple(out)


SHAPE_LIBRARY = _enumerate_shapes()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    return tile_r * (tile_h + 1), tile_c * (tile_w + 1)


def _base_grid(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    height = tile_rows * tile_h + (tile_rows - 1)
    width = tile_cols * tile_w + (tile_cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[tile_h :: tile_h + 1, :] = MAROON
    grid[:, tile_w :: tile_w + 1] = MAROON
    return grid


def _paste_tile(grid: np.ndarray, tile: np.ndarray, tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> None:
    r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w)
    grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile


def _render_shape_tile(
    tile_h: int,
    tile_w: int,
    shape: tuple[tuple[int, int], ...],
    color: int,
    reserve_bottom_right: bool,
) -> np.ndarray | None:
    shape_h, shape_w = _shape_dims(shape)
    if shape_h > tile_h or shape_w > tile_w:
        return None

    offsets = []
    for r0 in range(tile_h - shape_h + 1):
        for c0 in range(tile_w - shape_w + 1):
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if reserve_bottom_right and (tile_h - 1, tile_w - 1) in cells:
                continue
            offsets.append((r0, c0))
    if not offsets:
        return None

    r0, c0 = random.choice(offsets)
    tile = np.full((tile_h, tile_w), BLACK, dtype=int)
    for r, c in shape:
        tile[r0 + r, c0 + c] = color
    return tile


def _choose_layout() -> tuple[int, int, int, int]:
    tile_h = random.choice(TILE_SIZE_CHOICES)
    tile_w = random.choice(TILE_SIZE_CHOICES)
    row_max = min(MAX_TILE_ROWS, 31 // (tile_h + 1))
    col_max = min(MAX_TILE_COLS, 31 // (tile_w + 1))
    tile_rows = random.randint(2, row_max)
    tile_cols = random.randint(3, col_max)
    return tile_rows, tile_cols, tile_h, tile_w


def _eligible_shapes(tile_h: int, tile_w: int, reserve_bottom_right: bool) -> list[tuple[tuple[int, int], ...]]:
    max_h = tile_h - 1 if reserve_bottom_right else tile_h
    max_w = tile_w - 1 if reserve_bottom_right else tile_w
    return [shape for shape in SHAPE_LIBRARY if _shape_dims(shape)[0] <= max_h and _shape_dims(shape)[1] <= max_w]


def _oracle_output(
    inp: np.ndarray,
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    shape_to_color: dict[tuple[tuple[int, int], ...], int],
) -> np.ndarray:
    out = inp.copy()
    for tile_r in range(tile_rows):
        for tile_c in range(1, tile_cols):
            r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w)
            tile = out[r0 : r0 + tile_h, c0 : c0 + tile_w]
            shape = np.argwhere(tile == BLUE)
            if len(shape) == 0:
                continue
            fill = shape_to_color[_normalize_shape(shape)]
            tile[tile == BLACK] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols, tile_h, tile_w = _choose_layout()
    legend_shapes = _eligible_shapes(tile_h, tile_w, reserve_bottom_right=True)
    if len(legend_shapes) < tile_rows:
        return None

    chosen_shapes = random.sample(legend_shapes, tile_rows)
    legend_colors = random.sample(FILL_COLORS, tile_rows)
    shape_to_color = {shape: color for shape, color in zip(chosen_shapes, legend_colors)}

    inp = _base_grid(tile_rows, tile_cols, tile_h, tile_w)

    for tile_r, (shape, color) in enumerate(zip(chosen_shapes, legend_colors)):
        tile = _render_shape_tile(tile_h, tile_w, shape, BLUE, reserve_bottom_right=True)
        if tile is None:
            return None
        tile[-1, -1] = color
        _paste_tile(inp, tile, tile_r, 0, tile_h, tile_w)

    work_positions = [(tile_r, tile_c) for tile_r in range(tile_rows) for tile_c in range(1, tile_cols)]
    if len(work_positions) < tile_rows + 1:
        return None

    random.shuffle(work_positions)
    assignments: dict[tuple[int, int], tuple[tuple[int, int], ...] | None] = {pos: None for pos in work_positions}

    for pos, shape in zip(work_positions, chosen_shapes):
        assignments[pos] = shape

    empties = 0
    for pos in work_positions[tile_rows:]:
        if random.random() < 0.5:
            empties += 1
            continue
        assignments[pos] = random.choice(chosen_shapes)

    if empties == 0:
        free_choices = [pos for pos in work_positions[tile_rows:] if assignments[pos] is not None]
        if not free_choices:
            return None
        assignments[random.choice(free_choices)] = None

    for (tile_r, tile_c), shape in assignments.items():
        if shape is None:
            continue
        tile = _render_shape_tile(tile_h, tile_w, shape, BLUE, reserve_bottom_right=False)
        if tile is None:
            return None
        _paste_tile(inp, tile, tile_r, tile_c, tile_h, tile_w)

    expected = _oracle_output(inp, tile_rows, tile_cols, tile_h, tile_w, shape_to_color)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
