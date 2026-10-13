import numpy as np

from arc_utils import Color, colors_present, get_bg


Tile = tuple[frozenset[tuple[int, int]], int]

_TILE_PATTERNS: tuple[Tile, ...] = (
    (frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}), Color.BLUE),
    (frozenset({(0, 1), (1, 0), (1, 1)}), Color.RED),
    (frozenset({(0, 0), (0, 1), (1, 1)}), Color.GREEN),
    (frozenset({(0, 0), (1, 0), (1, 1)}), Color.YELLOW),
)


def _unique_tile_cover(g: np.ndarray, fg_color: int) -> list[Tile]:
    h, w = g.shape
    fg = frozenset((int(r), int(c)) for r, c in np.argwhere(g == fg_color))
    assert fg

    candidates: list[Tile] = []
    for r in range(h - 1):
        for c in range(w - 1):
            for rels, color in _TILE_PATTERNS:
                cells = frozenset((r + dr, c + dc) for dr, dc in rels)
                if cells <= fg:
                    candidates.append((cells, color))
    assert candidates

    by_cell: dict[tuple[int, int], list[int]] = {cell: [] for cell in fg}
    for idx, (cells, _) in enumerate(candidates):
        for cell in cells:
            by_cell[cell].append(idx)
    assert all(by_cell[cell] for cell in fg)

    solutions: list[list[int]] = []

    def search(remaining: frozenset[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return

        cell = min(
            remaining,
            key=lambda p: sum(1 for idx in by_cell[p] if candidates[idx][0] <= remaining),
        )
        options = [idx for idx in by_cell[cell] if candidates[idx][0] <= remaining]
        for idx in options:
            search(remaining - candidates[idx][0], chosen + [idx])

    search(fg, [])
    assert len(solutions) == 1
    return [candidates[idx] for idx in solutions[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the monochrome foreground mask by its unique tiling into 2x2 squares and three 2x2 L-tromino orientations.

    Input structure
    - The background is the dominant color.
    - All non-background pixels have one foreground color and form a mask that can be tiled by small 2x2-based sprites.

    Rule
    - Cover the foreground exactly once with translated tiles: a filled 2x2 square, an L missing the upper-left cell, an L missing the lower-left cell, or an L missing the upper-right cell.
    - Recolor those four tile shapes blue, red, green, and yellow respectively.
    - Keep background pixels unchanged.

    Key insight
    - Some L-trominoes interlock, so their missing corner may be occupied by another tile; the invariant is the unique exact cover of foreground cells, not local 2x2 background.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1
    fg_color = colors[0]

    out = g.copy()
    for cells, color in _unique_tile_cover(g, fg_color):
        for r, c in cells:
            out[r, c] = color

    assert not np.any(out == fg_color)
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and placement bounds.
_GEN_GRID_SIZE = 7
_GEN_ANCHOR_SPAN = _GEN_GRID_SIZE - 1
_GEN_BACKGROUND_COLOR = 0
_GEN_FOREGROUND_COLOR = 8
# Tile definitions: relative cells and output color by tile type.
_GEN_TILE_PATTERNS = (
    (((0, 0), (0, 1), (1, 0), (1, 1)), 1),
    (((0, 1), (1, 0), (1, 1)), 2),
    (((0, 0), (0, 1), (1, 1)), 3),
    (((0, 0), (1, 0), (1, 1)), 4),
)
_GEN_BASE_TILE_TYPES = (0, 1, 2, 3)
# Tile-count sampling: optional fifth tile and its type bias.
_GEN_EXTRA_TILE_PROB = 0.68
_GEN_EXTRA_SQUARE_PROB = 0.53
_GEN_EXTRA_L_TYPE_RANGE = (1, 4)
_GEN_MAX_ATTEMPTS = 300


def _gen_tile_cells(tile_type, r0, c0):
    return frozenset(
        (r0 + dr, c0 + dc) for dr, dc in _GEN_TILE_PATTERNS[tile_type][0]
    )


def _gen_is_touching(cells, occupied):
    return any(
        abs(r - rr) <= 1 and abs(c - cc) <= 1
        for r, c in cells
        for rr, cc in occupied
    )


def _gen_tile_type_order():
    tile_types = list(_GEN_BASE_TILE_TYPES)
    if np.random.random() < _GEN_EXTRA_TILE_PROB:
        if np.random.random() < _GEN_EXTRA_SQUARE_PROB:
            tile_types.append(0)
        else:
            tile_types.append(int(np.random.randint(*_GEN_EXTRA_L_TYPE_RANGE)))
    np.random.shuffle(tile_types)
    return tile_types


def _gen_place_tiles(tile_types):
    placements = []

    def backtrack(index, occupied):
        if index == len(tile_types):
            return True

        tile_type = tile_types[index]
        anchors = [
            (r, c)
            for r in range(_GEN_ANCHOR_SPAN)
            for c in range(_GEN_ANCHOR_SPAN)
        ]
        np.random.shuffle(anchors)

        for r0, c0 in anchors:
            cells = _gen_tile_cells(tile_type, r0, c0)
            if cells & occupied:
                continue
            if occupied and not _gen_is_touching(cells, occupied):
                continue

            placements.append((tile_type, cells))
            if backtrack(index + 1, occupied | cells):
                return True
            placements.pop()

        return False

    if not backtrack(0, frozenset()):
        return None
    return placements


def _gen_render_output(placements):
    out = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    for tile_type, cells in placements:
        color = _GEN_TILE_PATTERNS[tile_type][1]
        for r, c in cells:
            out[r, c] = color
    return out


def _gen_validate_output(out):
    inp = np.where(
        out == _GEN_BACKGROUND_COLOR,
        _GEN_BACKGROUND_COLOR,
        _GEN_FOREGROUND_COLOR,
    )
    try:
        predicted = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(predicted, out):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    placements = _gen_place_tiles(_gen_tile_type_order())
    if placements is None:
        return None
    return _gen_validate_output(_gen_render_output(placements))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a unique 626c0bcc tiling")
