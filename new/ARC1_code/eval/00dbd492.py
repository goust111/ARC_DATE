import numpy as np

from arc_utils import Color, find_components, get_bg


FILL_BY_SIDE = {
    5: Color.TEAL,
    7: Color.YELLOW,
    9: Color.GREEN,
}


def _is_square_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or h != w:
        return False

    if int(component["area"]) != 2 * h + 2 * w - 4:
        return False

    coords = component["coords"]
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow red square with the size-specific color while preserving its centered red marker.

    Input structure
    The grid has a black background and red foreground made of one or more
    hollow square frames plus one isolated red marker centered inside each frame.

    Rule
    Copy the input. For every red square frame, recolor all black interior
    pixels: 5x5 frames become teal, 7x7 frames become yellow, and 9x9 frames
    become green. Red border and marker pixels remain unchanged.

    Key insight
    The isolated red marker belongs to the hollow frame that contains it, and
    the frame side length determines the interior fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(map(int, np.unique(g))).issubset({bg, Color.RED})

    out = g.copy()
    frames = []
    markers = set()
    for component in find_components(g, bg=bg, connectivity=4, colors=[Color.RED]):
        if _is_square_frame(component):
            frames.append(component["bbox"])
        elif int(component["area"]) == 1:
            markers.add(tuple(map(int, component["coords"][0])))
        else:
            raise AssertionError("unexpected red component")

    assert frames
    used_markers = set()
    for r0, c0, r1, c1 in frames:
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1
        assert side in FILL_BY_SIDE

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere(interior == Color.RED)
        assert marker_locs.shape == (1, 2)
        mr, mc = map(int, marker_locs[0])
        assert (mr, mc) == ((side - 2) // 2, (side - 2) // 2)

        marker = (r0 + 1 + mr, c0 + 1 + mc)
        assert marker in markers
        used_markers.add(marker)

        fill_mask = interior == bg
        assert int(np.count_nonzero(fill_mask)) == (side - 2) ** 2 - 1
        out_view = out[r0 + 1 : r1, c0 + 1 : c1]
        out_view[fill_mask] = FILL_BY_SIDE[side]

    assert used_markers == markers
    return out

# --------------- generator (o5x) ---------------

# Palette used by generated inputs and expected outputs.
_GEN_BLACK = 0
_GEN_RED = 2
_GEN_FILL_BY_SIDE = {5: 8, 7: 4, 9: 3}

# Frame count and side-length sampling.
_GEN_SIDES = (5, 7, 9)
_GEN_FRAME_COUNT_MIN = 1
_GEN_FRAME_COUNT_MAX = 4

# Square canvas bounds and random slack beyond the minimum packable side.
_GEN_GRID_SIDE_MIN = 7
_GEN_GRID_SIDE_MAX = 20
_GEN_GRID_PADDING_MIN = 0
_GEN_GRID_PADDING_MAX = 3

# Packing feasibility and retry controls.
_GEN_MIN_BBOX_GAP = 1
_GEN_RED_AREA_HEADROOM = 2
_GEN_MAX_ATTEMPTS = 300

_GEN_RNG = np.random.default_rng()


def _gen_has_required_gap(box_a, box_b):
    return (
        box_a[2] + _GEN_MIN_BBOX_GAP < box_b[0]
        or box_b[2] + _GEN_MIN_BBOX_GAP < box_a[0]
        or box_a[3] + _GEN_MIN_BBOX_GAP < box_b[1]
        or box_b[3] + _GEN_MIN_BBOX_GAP < box_a[1]
    )


def _gen_fits_with_boxes(box, boxes):
    return all(_gen_has_required_gap(box, prev) for prev in boxes)


def _gen_box(r, c, side):
    return (int(r), int(c), int(r + side - 1), int(c + side - 1))


def _gen_place_sides(sides, grid_side, randomize):
    ordered_sides = tuple(sorted(map(int, sides), reverse=True))
    candidates_by_side = {
        side: [(r, c) for r in range(grid_side - side + 1) for c in range(grid_side - side + 1)]
        for side in set(ordered_sides)
    }
    memo = {}

    def can_complete(index, boxes):
        key = (index, tuple(sorted(boxes)))
        if key in memo:
            return memo[key]
        if index == len(ordered_sides):
            memo[key] = True
            return True

        side = ordered_sides[index]
        for r, c in candidates_by_side[side]:
            box = _gen_box(r, c, side)
            if _gen_fits_with_boxes(box, boxes) and can_complete(index + 1, boxes + (box,)):
                memo[key] = True
                return True
        memo[key] = False
        return False

    boxes = tuple()
    placements = []
    for index, side in enumerate(ordered_sides):
        positions = list(candidates_by_side[side])
        if randomize:
            _GEN_RNG.shuffle(positions)

        viable = []
        for r, c in positions:
            box = _gen_box(r, c, side)
            if _gen_fits_with_boxes(box, boxes) and can_complete(index + 1, boxes + (box,)):
                viable.append((r, c, box))

        if not viable:
            return None

        r, c, box = viable[int(_GEN_RNG.integers(len(viable)))] if randomize else viable[0]
        placements.append((int(r), int(c), int(side)))
        boxes = boxes + (box,)

    return placements


def _gen_min_grid_side(sides):
    red_pixels = sum(4 * int(side) - 3 for side in sides)
    lower = max(_GEN_GRID_SIDE_MIN, max(map(int, sides)))
    while lower * lower <= _GEN_RED_AREA_HEADROOM * red_pixels:
        lower += 1

    for grid_side in range(lower, _GEN_GRID_SIDE_MAX + 1):
        if _gen_place_sides(sides, grid_side, randomize=False) is not None:
            return grid_side
    return None


def _gen_draw_frame(grid, r, c, side):
    grid[r, c : c + side] = _GEN_RED
    grid[r + side - 1, c : c + side] = _GEN_RED
    grid[r : r + side, c] = _GEN_RED
    grid[r : r + side, c + side - 1] = _GEN_RED
    grid[r + side // 2, c + side // 2] = _GEN_RED


def _gen_expected_output(grid, placements):
    out = grid.copy()
    for r, c, side in placements:
        fill_color = _GEN_FILL_BY_SIDE[side]
        out[r + 1 : r + side - 1, c + 1 : c + side - 1] = fill_color
        out[r + side // 2, c + side // 2] = _GEN_RED
    return out


def _gen_passes_oracle(grid, placements):
    expected = _gen_expected_output(grid, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    num_frames = int(_GEN_RNG.integers(_GEN_FRAME_COUNT_MIN, _GEN_FRAME_COUNT_MAX + 1))
    sides = [int(_GEN_RNG.choice(_GEN_SIDES)) for _ in range(num_frames)]
    min_side = _gen_min_grid_side(sides)
    if min_side is None:
        return None

    padding = int(_GEN_RNG.integers(_GEN_GRID_PADDING_MIN, _GEN_GRID_PADDING_MAX + 1))
    grid_side = min(_GEN_GRID_SIDE_MAX, min_side + padding)
    placements = _gen_place_sides(sides, grid_side, randomize=True)
    if placements is None:
        return None

    grid = np.full((grid_side, grid_side), _GEN_BLACK, dtype=int)
    for r, c, side in placements:
        _gen_draw_frame(grid, r, c, side)

    if not _gen_passes_oracle(grid, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 00dbd492 input")
