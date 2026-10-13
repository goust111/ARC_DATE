from __future__ import annotations

import numpy as np


_BG = 0
_MARKER = 2
_OBJECT_COLORS = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_MAX_DIM = 30
_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == _BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) == _BG:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(cells)
    return components


def _parse_object(grid: np.ndarray, cells: list[tuple[int, int]]):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    assert np.all(patch != _BG), "each component must be a filled rectangle"
    marker_locs = np.argwhere(patch == _MARKER)
    assert marker_locs.shape == (1, 2), "each rectangle must contain exactly one red marker"

    non_marker_colors = sorted(set(int(x) for x in np.unique(patch)) - {_MARKER})
    assert len(non_marker_colors) == 1, "each rectangle must have one base color"
    base_color = non_marker_colors[0]

    mr, mc = map(int, marker_locs[0])
    return (r0, c0, r1, c1), (mr, mc), int(base_color), patch.shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reduce each marked rectangle to the row, column, or cross indicated by the marker's local coordinates.

    Input structure:
    - The background is black.
    - Several filled rectangular objects appear in different colors, and all of them have the same height and width.
    - Inside each object, exactly one cell is replaced by a red marker, and the objects do not touch.

    Rule:
    - Treat each non-black connected component as one filled rectangle and locate its red marker in local object coordinates (r, c).
    - If r < c, keep only the marker's full row from that rectangle.
    - If r > c, keep only the marker's full column from that rectangle.
    - If r = c, keep both that row and that column as a cross.
    - Restore the marker cell to the rectangle's original color and erase the rest of the rectangle to background.

    Key insight:
    - The shared rectangle shape makes the red marker meaningful by position, and comparing its local row and column selects the slice orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full(g.shape, _BG, dtype=int)
    components = _find_components(g)
    assert components, "expected at least one foreground object"

    shapes = set()
    for cells in components:
        (r0, c0, r1, c1), (mr, mc), base_color, shape = _parse_object(g, cells)
        shapes.add(tuple(map(int, shape)))
        if mr < mc:
            out[r0 + mr, c0 : c1 + 1] = base_color
        elif mr > mc:
            out[r0 : r1 + 1, c0 + mc] = base_color
        else:
            out[r0 + mr, c0 : c1 + 1] = base_color
            out[r0 : r1 + 1, c0 + mc] = base_color

    assert len(shapes) == 1, "all rectangles must share one shape"
    return out


def _candidate_positions(height: int, width: int) -> dict[str, list[tuple[int, int]]]:
    row_positions: list[tuple[int, int]] = []
    col_positions: list[tuple[int, int]] = []
    cross_positions: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if r < c:
                row_positions.append((r, c))
            elif r > c:
                col_positions.append((r, c))
            else:
                cross_positions.append((r, c))
    assert row_positions and col_positions and cross_positions
    return {
        "row": row_positions,
        "col": col_positions,
        "cross": cross_positions,
    }


def _sample_layout(
    rng: np.random.Generator, count: int, height: int, width: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    for _ in range(100):
        rows = int(rng.integers(1, min(count, 4) + 1))
        cols = (count + rows - 1) // rows

        gaps_y = [int(rng.integers(1, 5)) for _ in range(max(0, rows - 1))]
        gaps_x = [int(rng.integers(1, 5)) for _ in range(max(0, cols - 1))]
        margin_top = int(rng.integers(0, 4))
        margin_bottom = int(rng.integers(0, 4))
        margin_left = int(rng.integers(0, 4))
        margin_right = int(rng.integers(0, 4))

        grid_h = margin_top + rows * height + sum(gaps_y) + margin_bottom
        grid_w = margin_left + cols * width + sum(gaps_x) + margin_right
        if grid_h > _MAX_DIM or grid_w > _MAX_DIM:
            continue

        row_starts = [margin_top]
        for gap in gaps_y:
            row_starts.append(row_starts[-1] + height + gap)
        col_starts = [margin_left]
        for gap in gaps_x:
            col_starts.append(col_starts[-1] + width + gap)

        slots = [(row_starts[r], col_starts[c]) for r in range(rows) for c in range(cols)]
        if len(slots) < count:
            continue
        chosen = rng.choice(len(slots), size=count, replace=False)
        positions = [slots[int(i)] for i in chosen]
        rng.shuffle(positions)
        return int(grid_h), int(grid_w), positions
    return None


def _draw_expected_slice(
    out: np.ndarray, top: int, left: int, height: int, width: int, marker_pos: tuple[int, int], color: int
) -> None:
    mr, mc = marker_pos
    if mr < mc:
        out[top + mr, left : left + width] = color
    elif mr > mc:
        out[top : top + height, left + mc] = color
    else:
        out[top + mr, left : left + width] = color
        out[top : top + height, left + mc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    count = int(rng.integers(3, 7))
    height = int(rng.integers(2, 7))
    width = int(rng.integers(2, 7))
    positions_by_relation = _candidate_positions(height, width)

    layout = _sample_layout(rng, count, height, width)
    if layout is None:
        return None
    grid_h, grid_w, placements = layout

    colors = [int(c) for c in rng.choice(_OBJECT_COLORS, size=count, replace=False)]
    relations = ["row", "col", "cross"]
    while len(relations) < count:
        relations.append(str(rng.choice(np.array(["row", "col", "cross"], dtype=object))))
    rng.shuffle(relations)

    inp = np.full((grid_h, grid_w), _BG, dtype=int)
    expected = np.full((grid_h, grid_w), _BG, dtype=int)

    for (top, left), color, relation in zip(placements, colors, relations):
        marker_pos = positions_by_relation[relation][
            int(rng.integers(0, len(positions_by_relation[relation])))
        ]
        inp[top : top + height, left : left + width] = color
        inp[top + marker_pos[0], left + marker_pos[1]] = _MARKER
        _draw_expected_slice(expected, top, left, height, width, marker_pos, color)

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
