import numpy as np

from arc_utils import (
    Color,
    bbox_coords,
    bbox_h,
    bbox_w,
    colors_present,
    components_mask,
    get_bg,
)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()) == 0
        or int(cols.min()) == 0
        or int(rows.max()) == height - 1
        or int(cols.max()) == width - 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed square hole in the gray outlines with the color assigned to that hole size.

    Input structure
    - The grid is a black canvas containing gray outline objects.
    - Each relevant interior is a 4-connected black component fully enclosed by gray pixels.
    - Enclosed interiors are solid squares of side length 1, 2, or 3.

    Rule
    - Keep the gray outlines and the exterior black background unchanged.
    - Fill a 1x1 enclosed square with pink, a 2x2 enclosed square with orange, and a 3x3 enclosed square with teal.

    Key insight
    - The fill color is determined by the size of the enclosed background region, not by the outline's position or full shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black to be the empty color"
    assert colors_present(g, exclude={bg}) == [Color.GREY], "expected only gray foreground outlines"

    side_to_color = {
        1: Color.PINK,
        2: Color.ORANGE,
        3: Color.TEAL,
    }

    out = g.copy()
    filled_any = False
    for coords in components_mask(g == bg, connectivity=4):
        if _touches_border(coords, g.shape):
            continue

        box = bbox_coords(coords)
        height = bbox_h(box)
        width = bbox_w(box)
        assert height == width, "enclosed region must be square"
        assert coords.shape[0] == height * width, "enclosed square must be solid"
        assert height in side_to_color, "unexpected enclosed square size"

        out[coords[:, 0], coords[:, 1]] = side_to_color[height]
        filled_any = True

    assert filled_any, "expected at least one enclosed square"
    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste

# Canvas and color constants.
GEN_GRID_SIZE = 12
GEN_GRID_SHAPE = (GEN_GRID_SIZE, GEN_GRID_SIZE)
GEN_BG_COLOR = Color.BLACK
GEN_FRAME_COLOR = Color.GREY
GEN_FILL_COLORS = {
    1: Color.PINK,
    2: Color.ORANGE,
    3: Color.TEAL,
}

# Frame size and spacing constraints.
GEN_INTERIOR_SIDES = (1, 2, 3)
GEN_FRAME_COUNT_OPTIONS = (2, 3)
GEN_FRAME_THICKNESS = 1
GEN_MIN_FRAME_GAP = 2

# Border sampling and retry controls.
GEN_ALLOW_BORDER_PROB = 0.55
GEN_MAX_ATTEMPTS = 300


def _gen_outer_size(side: int) -> int:
    return int(side) + 2 * GEN_FRAME_THICKNESS


def _gen_frame_patch(side: int) -> np.ndarray:
    outer = _gen_outer_size(side)
    patch = np.full((outer, outer), GEN_BG_COLOR, dtype=int)
    patch[0, :] = GEN_FRAME_COLOR
    patch[-1, :] = GEN_FRAME_COLOR
    patch[:, 0] = GEN_FRAME_COLOR
    patch[:, -1] = GEN_FRAME_COLOR
    return patch


def _gen_frames_are_separated(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar0, ac0, aside = a
    br0, bc0, bside = b
    ar1 = ar0 + _gen_outer_size(aside) - 1
    ac1 = ac0 + _gen_outer_size(aside) - 1
    br1 = br0 + _gen_outer_size(bside) - 1
    bc1 = bc0 + _gen_outer_size(bside) - 1
    row_gap = max(ar0 - br1, br0 - ar1, 0)
    col_gap = max(ac0 - bc1, bc0 - ac1, 0)
    return max(row_gap, col_gap) >= GEN_MIN_FRAME_GAP


def _gen_touches_grid_edge(r0: int, c0: int, side: int) -> bool:
    outer = _gen_outer_size(side)
    return (
        r0 == 0
        or c0 == 0
        or r0 + outer == GEN_GRID_SIZE
        or c0 + outer == GEN_GRID_SIZE
    )


def _gen_valid_positions(
    placed: list[tuple[int, int, int]], side: int, border_used: bool, allow_border: bool
) -> list[tuple[int, int]]:
    """Return top-left positions that fit while preserving spacing and border limits."""
    side = int(side)
    outer = _gen_outer_size(side)
    valid = []
    for r0 in range(GEN_GRID_SIZE - outer + 1):
        for c0 in range(GEN_GRID_SIZE - outer + 1):
            touches_border = _gen_touches_grid_edge(r0, c0, side)
            if touches_border and (border_used or not allow_border):
                continue
            candidate = (r0, c0, side)
            if all(_gen_frames_are_separated(candidate, prev) for prev in placed):
                valid.append((r0, c0))
    return valid


def _gen_expected_output(grid: np.ndarray, placed: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, side in placed:
        r1 = r0 + GEN_FRAME_THICKNESS
        c1 = c0 + GEN_FRAME_THICKNESS
        out[r1 : r1 + side, c1 : c1 + side] = GEN_FILL_COLORS[side]
    return out


def _gen_sample_sides() -> list[int]:
    """Sample distinct interior side lengths and randomize placement order."""
    count = int(np.random.choice(GEN_FRAME_COUNT_OPTIONS))
    sizes = np.random.choice(GEN_INTERIOR_SIDES, size=count, replace=False).tolist()
    np.random.shuffle(sizes)
    return sizes


def _gen_place_frames(
    grid: np.ndarray, sizes: list[int], allow_border: bool
) -> list[tuple[int, int, int]] | None:
    """Draw sampled frames in order, returning placements or None if any frame cannot fit."""
    placed = []
    border_used = False
    for side in sizes:
        valid = _gen_valid_positions(placed, int(side), border_used, allow_border)
        if not valid:
            return None
        r0, c0 = valid[int(np.random.randint(len(valid)))]
        paste(grid, _gen_frame_patch(int(side)), r0, c0)
        placed.append((r0, c0, int(side)))
        border_used = border_used or _gen_touches_grid_edge(r0, c0, int(side))
    return placed


def _gen_oracle_accepts(grid: np.ndarray, placed: list[tuple[int, int, int]]) -> bool:
    """Check the generated input against the solver's expected output."""
    expected = _gen_expected_output(grid, placed)
    try:
        return bool(np.array_equal(main(grid), expected))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    grid = np.full(GEN_GRID_SHAPE, GEN_BG_COLOR, dtype=int)
    sizes = _gen_sample_sides()
    allow_border = bool(np.random.random() < GEN_ALLOW_BORDER_PROB)
    placed = _gen_place_frames(grid, sizes, allow_border)
    if placed is None or not _gen_oracle_accepts(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c0f76784 input")
