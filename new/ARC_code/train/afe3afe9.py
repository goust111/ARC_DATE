import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
import random


def _marker_side(g: np.ndarray, bg: int) -> tuple[str, int]:
    sides: list[tuple[str, int]] = []
    for color in set(map(int, g.ravel())) - {bg}:
        if np.all(g[0, :] == color):
            sides.append(("top", color))
        if np.all(g[:, -1] == color):
            sides.append(("right", color))
        if np.all(g[-1, :] == color):
            sides.append(("bottom", color))
        if np.all(g[:, 0] == color):
            sides.append(("left", color))

    assert len(sides) == 1, sides
    return sides[0]


def _symbolic_grid(g: np.ndarray) -> tuple[np.ndarray, str, int]:
    bg = get_bg(g)
    side, marker = _marker_side(g, bg)
    assert marker == 1, marker
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    objects: list[tuple[int, int, int]] = []
    for component in components:
        color = component["color"]
        if color == marker:
            continue

        b = component["bbox"]
        assert component["area"] == 8, component
        assert bbox_h(b) == 3 and bbox_w(b) == 3, component

        r0, c0, r1, c1 = b
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.full((3, 3), color, dtype=int)
        expected[1, 1] = bg
        assert np.array_equal(patch, expected), patch
        objects.append((r0, c0, color))

    assert objects
    rows = sorted({r for r, _c, _color in objects})
    cols = sorted({c for _r, c, _color in objects})
    symbolic = np.full((len(rows), len(cols)), bg, dtype=int)
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}

    for r, c, color in objects:
        rr = row_index[r]
        cc = col_index[c]
        assert symbolic[rr, cc] == bg
        symbolic[rr, cc] = color

    if side in ("top", "bottom"):
        dominant_cols = range(3, 6) if side == "top" else range(0, 3)
        dominant_values = [
            int(symbolic[r, c])
            for r in range(symbolic.shape[0])
            for c in dominant_cols
            if int(symbolic[r, c]) != bg
        ]
    else:
        dominant_rows = range(0, 3) if side == "left" else range(3, 6)
        dominant_values = [
            int(symbolic[r, c])
            for r in dominant_rows
            for c in range(symbolic.shape[1])
            if int(symbolic[r, c]) != bg
        ]
    assert dominant_values and set(dominant_values) == {8}, dominant_values

    return symbolic, side, bg


def _compact(symbolic: np.ndarray, side: str, bg: int) -> np.ndarray:
    out = np.full_like(symbolic, bg)

    if side == "top":
        for r, row in enumerate(symbolic):
            values = [int(v) for v in row if int(v) != bg]
            out[r, symbolic.shape[1] - len(values) :] = values
    elif side == "right":
        for c in range(symbolic.shape[1]):
            values = [int(v) for v in symbolic[:, c] if int(v) != bg]
            out[symbolic.shape[0] - len(values) :, c] = values
    elif side == "bottom":
        for r, row in enumerate(symbolic):
            values = [int(v) for v in row if int(v) != bg]
            out[r, : len(values)] = values
    elif side == "left":
        for c in range(symbolic.shape[1]):
            values = [int(v) for v in symbolic[:, c] if int(v) != bg]
            out[: len(values), c] = values
    else:
        raise AssertionError(side)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Convert each hollow 3x3 symbol to one pixel and slide the pixels clockwise from the marked edge.

    Input structure
    - The grid has a background, one full-edge marker line, and monochrome hollow 3x3 foreground symbols.
    - The symbols lie on a coarse row/column lattice; fully empty lattice rows or columns are absent from the output.

    Rule
    - Replace each hollow 3x3 symbol with one pixel of its color in the corresponding coarse-grid cell.
    - The marker edge gives the compaction direction by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
    - Preserve the order of non-background pixels while sliding them as far as possible in that direction.

    Key insight
    - The large picture is a scaled symbolic grid, and the blue edge marker acts as a gravity direction rather than as output content.
    """
    g = np.asarray(grid, dtype=int)
    symbolic, side, bg = _symbolic_grid(g)
    return _compact(symbolic, side, bg)

# --------------- generator (att2) ---------------

# Canvas and colors.
GEN_BG = 0
GEN_GRID_SHAPE = (30, 30)
GEN_MARKER_COLOR = 1
GEN_DOMINANT_COLOR = 8
GEN_ACCENT_COLORS = [2, 3, 4, 5, 6, 7, 9]

# Symbolic lattice layout.
GEN_SIDES = ["top", "right", "bottom", "left"]
GEN_SYMBOLIC_HW = (7, 6)
GEN_SYMBOLIC_VW = (6, 7)
GEN_DOMINANT_COUNT_RANGE = (13, 17)
GEN_ACCENT_COUNT_RANGE = (6, 8)
GEN_SPLIT_INDEX = 3

# Physical lattice coordinates.
GEN_GAP_SIX_POSITIONS = [1, 5, 9, 17, 21, 25]
GEN_SEVEN_FROM_MARKER = [2, 6, 10, 14, 18, 22, 26]
GEN_SEVEN_FROM_OTHER = [1, 5, 9, 13, 17, 21, 25]
GEN_SYMBOL_PATCH_SIZE = 3

# Generation retries.
GEN_MAX_ATTEMPTS = 300

def _gen_compact(symbolic, side, bg):
    out = np.full_like(symbolic, bg)

    if side == "top":
        for r, row in enumerate(symbolic):
            values = [int(v) for v in row if int(v) != bg]
            out[r, symbolic.shape[1] - len(values) :] = values
    elif side == "right":
        for c in range(symbolic.shape[1]):
            values = [int(v) for v in symbolic[:, c] if int(v) != bg]
            out[symbolic.shape[0] - len(values) :, c] = values
    elif side == "bottom":
        for r, row in enumerate(symbolic):
            values = [int(v) for v in row if int(v) != bg]
            out[r, : len(values)] = values
    elif side == "left":
        for c in range(symbolic.shape[1]):
            values = [int(v) for v in symbolic[:, c] if int(v) != bg]
            out[: len(values), c] = values
    else:
        raise AssertionError(side)

    return out


def _gen_covering_cells(rows, cols, count):
    rows = list(rows)
    cols = list(cols)
    cells = set()

    shuffled_rows = rows[:]
    random.shuffle(shuffled_rows)
    for r in shuffled_rows:
        cells.add((r, random.choice(cols)))

    for c in cols:
        if all(cc != c for _rr, cc in cells):
            cells.add((random.choice(rows), c))

    remaining = [(r, c) for r in rows for c in cols if (r, c) not in cells]
    random.shuffle(remaining)
    cells.update(remaining[: count - len(cells)])
    return cells


def _gen_fill(symbolic, rows, cols, count, color):
    for r, c in _gen_covering_cells(rows, cols, count):
        symbolic[r, c] = color


def _gen_symbolic(side, bg):
    color_a, color_b = random.sample(GEN_ACCENT_COLORS, 2)

    if side in ("top", "bottom"):
        symbolic = np.full(GEN_SYMBOLIC_HW, bg, dtype=int)
        dominant_cols = range(GEN_SPLIT_INDEX, GEN_SYMBOLIC_HW[1]) if side == "top" else range(0, GEN_SPLIT_INDEX)
        accent_cols = range(0, GEN_SPLIT_INDEX) if side == "top" else range(GEN_SPLIT_INDEX, GEN_SYMBOLIC_HW[1])
        _gen_fill(symbolic, range(GEN_SYMBOLIC_HW[0]), dominant_cols, random.randint(*GEN_DOMINANT_COUNT_RANGE), GEN_DOMINANT_COLOR)
        _gen_fill(symbolic, range(0, GEN_SPLIT_INDEX), accent_cols, random.randint(*GEN_ACCENT_COUNT_RANGE), color_a)
        _gen_fill(symbolic, range(GEN_SPLIT_INDEX + 1, GEN_SYMBOLIC_HW[0]), accent_cols, random.randint(*GEN_ACCENT_COUNT_RANGE), color_b)
    else:
        symbolic = np.full(GEN_SYMBOLIC_VW, bg, dtype=int)
        dominant_rows = range(0, GEN_SPLIT_INDEX) if side == "left" else range(GEN_SPLIT_INDEX, GEN_SYMBOLIC_VW[0])
        accent_rows = range(GEN_SPLIT_INDEX, GEN_SYMBOLIC_VW[0]) if side == "left" else range(0, GEN_SPLIT_INDEX)
        _gen_fill(symbolic, dominant_rows, range(GEN_SYMBOLIC_VW[1]), random.randint(*GEN_DOMINANT_COUNT_RANGE), GEN_DOMINANT_COLOR)
        _gen_fill(symbolic, accent_rows, range(0, GEN_SPLIT_INDEX), random.randint(*GEN_ACCENT_COUNT_RANGE), color_a)
        _gen_fill(symbolic, accent_rows, range(GEN_SPLIT_INDEX + 1, GEN_SYMBOLIC_VW[1]), random.randint(*GEN_ACCENT_COUNT_RANGE), color_b)

    return symbolic


def _gen_lattice_positions(side):
    if side == "top":
        return GEN_SEVEN_FROM_MARKER, GEN_GAP_SIX_POSITIONS
    if side == "bottom":
        return GEN_SEVEN_FROM_OTHER, GEN_GAP_SIX_POSITIONS
    if side == "left":
        return GEN_GAP_SIX_POSITIONS, GEN_SEVEN_FROM_MARKER
    return GEN_GAP_SIX_POSITIONS, GEN_SEVEN_FROM_OTHER


def _gen_place_marker(grid, side):
    if side == "top":
        grid[0, :] = GEN_MARKER_COLOR
    elif side == "right":
        grid[:, -1] = GEN_MARKER_COLOR
    elif side == "bottom":
        grid[-1, :] = GEN_MARKER_COLOR
    else:
        grid[:, 0] = GEN_MARKER_COLOR


def _gen_hollow_patch(color, bg):
    patch = np.full((GEN_SYMBOL_PATCH_SIZE, GEN_SYMBOL_PATCH_SIZE), color, dtype=int)
    patch[GEN_SYMBOL_PATCH_SIZE // 2, GEN_SYMBOL_PATCH_SIZE // 2] = bg
    return patch


def _gen_expand(symbolic, side, bg):
    grid = np.full(GEN_GRID_SHAPE, bg, dtype=int)
    _gen_place_marker(grid, side)

    row_positions, col_positions = _gen_lattice_positions(side)
    for r in range(symbolic.shape[0]):
        for c in range(symbolic.shape[1]):
            color = int(symbolic[r, c])
            if color == bg:
                continue
            paste(grid, _gen_hollow_patch(color, bg), row_positions[r], col_positions[c])
    return grid


def _gen_attempt():
    side = random.choice(GEN_SIDES)
    symbolic = _gen_symbolic(side, GEN_BG)
    expected = _gen_compact(symbolic, side, GEN_BG)
    if np.array_equal(symbolic, expected):
        return None

    inp = _gen_expand(symbolic, side, GEN_BG)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate afe3afe9 input")
