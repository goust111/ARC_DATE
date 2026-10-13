import numpy as np


BACKGROUND = 0
PLACEHOLDER = 2
SEPARATOR = 5
LEGEND_PALETTE = (1, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400

SHAPE_LIBRARY = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _separator_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in _colors_present(grid, exclude={bg}):
        row_hits = sum(np.all(grid[r, :] == color) for r in range(grid.shape[0]))
        col_hits = sum(np.all(grid[:, c] == color) for c in range(grid.shape[1]))
        if row_hits >= 2 or col_hits >= 2:
            candidates.append(color)
    assert len(candidates) == 1, "expected a unique separator color"
    return int(candidates[0])


def _parse_layout(grid: np.ndarray) -> dict:
    h, w = grid.shape
    bg = _get_bg(grid)
    sep = _separator_color(grid, bg)
    rows = [r for r in range(h) if np.all(grid[r, :] == sep)]
    cols = [c for c in range(w) if np.all(grid[:, c] == sep)]

    if len(rows) == 3 and len(cols) == 2:
        step = rows[1] - rows[0]
        assert rows[2] - rows[1] == step
        assert cols[1] - cols[0] == step
        tile_size = step - 1
        assert rows == [tile_size, 2 * tile_size + 1, 3 * tile_size + 2]
        assert cols == [tile_size, 2 * tile_size + 1]
        tile_rows = [(0, tile_size), (tile_size + 1, 2 * tile_size + 1), (2 * tile_size + 2, 3 * tile_size + 2)]
        tile_cols = [(0, tile_size), (tile_size + 1, 2 * tile_size + 1), (2 * tile_size + 2, 3 * tile_size + 2)]
        legend_slice = (3 * tile_size + 3, h, 0, w)
        orientation = "below"
    elif len(rows) == 2 and len(cols) == 3:
        step = cols[1] - cols[0]
        assert cols[2] - cols[1] == step
        assert rows[1] - rows[0] == step
        tile_size = step - 1
        assert rows == [tile_size, 2 * tile_size + 1]
        assert cols == [tile_size, 2 * tile_size + 1, 3 * tile_size + 2]
        tile_rows = [(0, tile_size), (tile_size + 1, 2 * tile_size + 1), (2 * tile_size + 2, 3 * tile_size + 2)]
        tile_cols = [(0, tile_size), (tile_size + 1, 2 * tile_size + 1), (2 * tile_size + 2, 3 * tile_size + 2)]
        legend_slice = (0, h, 3 * tile_size + 3, w)
        orientation = "right"
    else:
        raise AssertionError("unrecognized tile-and-legend layout")

    lr0, lr1, lc0, lc1 = legend_slice
    assert lr1 > lr0 and lc1 > lc0
    return {
        "bg": bg,
        "sep": sep,
        "tile_size": tile_size,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "legend_slice": legend_slice,
        "orientation": orientation,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the 3x3 tile lattice and recolor each occupied cell by matching its normalized shape to the external legend.

    Input structure:
    - The background is black, and single-pixel separator lines split the canvas into a 3x3 tile lattice plus a separate legend region.
    - Non-empty tiles contain exactly one 4-connected placeholder object in a shared placeholder color.
    - The legend region contains 2-3 monochromatic reference objects in distinct colors.

    Rule:
    - Detect the 3x3 tile lattice from the separator lines and inspect each tile independently.
    - Normalize every non-empty tile object by translating its tight mask to the origin, then find the legend object with the same normalized shape.
    - Return a 3x3 output grid whose empty cells stay black and whose occupied cells take the matched legend color.

    Key insight:
    - The tile lattice contributes only coarse position, while the legend contributes color semantics; the placeholder color itself is a decoy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _parse_layout(g)
    bg = int(info["bg"])
    sep = int(info["sep"])
    lr0, lr1, lc0, lc1 = info["legend_slice"]
    legend = g[lr0:lr1, lc0:lc1]

    legend_map: dict[tuple[tuple[int, int], ...], int] = {}
    for color in _colors_present(legend, exclude={bg, sep}):
        comps = _components(legend == color)
        for coords in comps:
            sig = _normalize_coords(coords)
            assert sig not in legend_map, "legend shapes must be distinct"
            legend_map[sig] = int(color)
    assert 2 <= len(legend_map) <= 3

    out = np.full((3, 3), bg, dtype=int)
    placeholder_colors = set()
    for tr, (r0, r1) in enumerate(info["tile_rows"]):
        for tc, (c0, c1) in enumerate(info["tile_cols"]):
            tile = g[r0:r1, c0:c1]
            colors = _colors_present(tile, exclude={bg})
            assert len(colors) <= 1, "each tile should be empty or monochromatic"
            if not colors:
                continue
            placeholder_colors.add(int(colors[0]))
            comps = _components(tile == colors[0])
            assert len(comps) == 1, "each non-empty tile should contain exactly one object"
            sig = _normalize_coords(comps[0])
            assert sig in legend_map, "tile shape must appear in the legend"
            out[tr, tc] = legend_map[sig]

    assert len(placeholder_colors) <= 1
    return out


def _paint_shape(grid: np.ndarray, top: int, left: int, shape: tuple[tuple[int, int], ...], color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _segment_starts(tile_size: int) -> list[int]:
    return [0, tile_size + 1, 2 * tile_size + 2]


def _sample_shape_set(rng: np.random.Generator, tile_size: int, count: int) -> list[tuple[tuple[int, int], ...]]:
    candidates = [shape for shape in SHAPE_LIBRARY if max(_shape_dims(shape)) <= tile_size]
    while True:
        picks = rng.choice(len(candidates), size=count, replace=False)
        shapes = [candidates[int(i)] for i in picks]
        pairs_share_proxy = False
        for i in range(count):
            for j in range(i + 1, count):
                if len(shapes[i]) == len(shapes[j]) or _shape_dims(shapes[i]) == _shape_dims(shapes[j]):
                    pairs_share_proxy = True
        if pairs_share_proxy:
            return shapes


def _sample_occupied_cells(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    while True:
        picks = [cells[int(i)] for i in rng.choice(len(cells), size=count, replace=False)]
        rows = {r for r, _ in picks}
        cols = {c for _, c in picks}
        if len(rows) == 3 and len(cols) == 3:
            return picks


def _tile_offsets(rng: np.random.Generator, tile_size: int, shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h, w = _shape_dims(shape)
    return int(rng.integers(0, tile_size - h + 1)), int(rng.integers(0, tile_size - w + 1))


def _build_grid(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[tuple[int, int], ...]], dict[tuple[int, int], tuple[tuple[int, int], ...]], dict[tuple[int, int], tuple[int, int]]]:
    tile_size = int(rng.integers(4, 6))
    legend_shape_count = int(rng.integers(2, 4))
    shapes = _sample_shape_set(rng, tile_size, legend_shape_count)
    legend_colors = [int(x) for x in rng.choice(LEGEND_PALETTE, size=legend_shape_count, replace=False)]
    color_to_shape = {color: shape for color, shape in zip(legend_colors, shapes)}
    shape_to_color = {shape: color for color, shape in color_to_shape.items()}

    occupied_count = int(rng.integers(4, 8))
    occupied_cells = _sample_occupied_cells(rng, occupied_count)
    assignments = list(shapes)
    extra_shape_ids = rng.choice(len(shapes), size=occupied_count - len(shapes), replace=True)
    assignments.extend(shapes[int(i)] for i in extra_shape_ids)
    rng.shuffle(assignments)

    orientation = "below" if rng.random() < 0.5 else "right"
    if orientation == "below":
        grid = np.full((4 * tile_size + 3, 3 * tile_size + 2), BACKGROUND, dtype=int)
        grid[[tile_size, 2 * tile_size + 1, 3 * tile_size + 2], :] = SEPARATOR
        grid[:, [tile_size, 2 * tile_size + 1]] = SEPARATOR
    else:
        grid = np.full((3 * tile_size + 2, 4 * tile_size + 3), BACKGROUND, dtype=int)
        grid[[tile_size, 2 * tile_size + 1], :] = SEPARATOR
        grid[:, [tile_size, 2 * tile_size + 1, 3 * tile_size + 2]] = SEPARATOR

    tile_starts = _segment_starts(tile_size)
    expected = np.full((3, 3), BACKGROUND, dtype=int)
    cell_to_shape: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    cell_to_offset: dict[tuple[int, int], tuple[int, int]] = {}
    for (tr, tc), shape in zip(occupied_cells, assignments):
        dr, dc = _tile_offsets(rng, tile_size, shape)
        top = tile_starts[tr] + dr
        left = tile_starts[tc] + dc
        _paint_shape(grid, top, left, shape, PLACEHOLDER)
        expected[tr, tc] = shape_to_color[shape]
        cell_to_shape[(tr, tc)] = shape
        cell_to_offset[(tr, tc)] = (dr, dc)

    legend_slots = [int(x) for x in rng.choice(3, size=legend_shape_count, replace=False)]
    for slot, color, shape in zip(legend_slots, legend_colors, shapes):
        dr, dc = _tile_offsets(rng, tile_size, shape)
        if orientation == "below":
            top = 3 * tile_size + 3 + dr
            left = tile_starts[slot] + dc
        else:
            top = tile_starts[slot] + dr
            left = 3 * tile_size + 3 + dc
        _paint_shape(grid, top, left, shape, color)

    return grid, expected, color_to_shape, cell_to_shape, cell_to_offset


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    color_to_shape: dict[int, tuple[tuple[int, int], ...]],
    cell_to_shape: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    cell_to_offset: dict[tuple[int, int], tuple[int, int]],
) -> bool:
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    non_bg = [int(x) for x in np.unique(expected) if int(x) != BACKGROUND]
    if len(non_bg) != len(color_to_shape):
        return False

    if np.count_nonzero(expected != BACKGROUND) < 4 or np.count_nonzero(expected == BACKGROUND) < 2:
        return False

    shapes = list(color_to_shape.values())
    if not any(len(a) == len(b) or _shape_dims(a) == _shape_dims(b) for i, a in enumerate(shapes) for b in shapes[i + 1 :]):
        return False

    offsets_by_shape: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]] = {}
    for cell, shape in cell_to_shape.items():
        offsets_by_shape.setdefault(shape, set()).add(cell_to_offset[cell])
    if not any(len(offsets) >= 2 for offsets in offsets_by_shape.values()):
        return False

    info = _parse_layout(grid)
    bg = int(info["bg"])
    sep = int(info["sep"])
    lr0, lr1, lc0, lc1 = info["legend_slice"]
    legend = grid[lr0:lr1, lc0:lc1]
    legend_components = 0
    for color in _colors_present(legend, exclude={bg, sep}):
        comps = _components(legend == color)
        if len(comps) != 1:
            return False
        legend_components += 1
        if not 3 <= len(comps[0]) <= 6:
            return False
    if legend_components != len(color_to_shape):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid, expected, color_to_shape, cell_to_shape, cell_to_offset = _build_grid(rng)
        if _semantic_ok(grid, expected, color_to_shape, cell_to_shape, cell_to_offset):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
