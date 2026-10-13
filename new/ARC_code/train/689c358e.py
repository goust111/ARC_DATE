from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste
import random


def _solid_frame_color(g: np.ndarray) -> int:
    H, W = g.shape
    assert H >= 3 and W >= 3, "expected room for a frame and an interior"

    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    colors = set(map(int, border))
    assert len(colors) == 1, "expected a one-color rectangular frame"
    return int(colors.pop())


def _sprite_endpoints(g: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    H, W = g.shape
    pts = [tuple(map(int, p)) for p in coords]

    row_counts = {r: sum(1 for rr, _cc in pts if rr == r) for r in sorted({r for r, _c in pts})}
    col_counts = {c: sum(1 for _rr, cc in pts if cc == c) for c in sorted({c for _r, c in pts})}

    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]

    is_horizontal = len(long_rows) == 1 and len(cross_cols) == 1 and not long_cols and not cross_rows
    is_vertical = len(long_cols) == 1 and len(cross_rows) == 1 and not long_rows and not cross_cols
    assert is_horizontal ^ is_vertical, "expected a six-pixel asymmetric T sprite"

    if is_horizontal:
        r = int(long_rows[0])
        cross_c = int(cross_cols[0])
        line_cols = sorted(c for rr, c in pts if rr == r)
        cross_rs = sorted(rr for rr, c in pts if c == cross_c)

        assert line_cols == list(range(line_cols[0], line_cols[0] + 4)), "expected a length-4 horizontal bar"
        assert cross_rs == [r - 1, r, r + 1], "expected a centered length-3 crossbar"
        assert cross_c in line_cols, "crossbar must intersect the long bar"
        assert 0 < r < H - 1, "sprite axis must point to frame rows"

        left_dist = cross_c - line_cols[0]
        right_dist = line_cols[-1] - cross_c
        assert {left_dist, right_dist} == {1, 2}, "crossbar must be one cell closer to one end"
        if left_dist < right_dist:
            return (r, 0), (r, W - 1)
        return (r, W - 1), (r, 0)

    c = int(long_cols[0])
    cross_r = int(cross_rows[0])
    line_rows = sorted(r for r, cc in pts if cc == c)
    cross_cs = sorted(cc for r, cc in pts if r == cross_r)

    assert line_rows == list(range(line_rows[0], line_rows[0] + 4)), "expected a length-4 vertical bar"
    assert cross_cs == [c - 1, c, c + 1], "expected a centered length-3 crossbar"
    assert cross_r in line_rows, "crossbar must intersect the long bar"
    assert 0 < c < W - 1, "sprite axis must point to frame columns"

    up_dist = cross_r - line_rows[0]
    down_dist = line_rows[-1] - cross_r
    assert {up_dist, down_dist} == {1, 2}, "crossbar must be one cell closer to one end"
    if up_dist < down_dist:
        return (0, c), (H - 1, c)
    return (H - 1, c), (0, c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each arrow-like sprite's pointed frame endpoint with its color and the opposite endpoint black.

    Input structure:
    - The grid has a one-pixel solid rectangular frame around a dominant interior background.
    - Inside the frame are monochrome six-pixel sprites shaped as asymmetric T arrows.
    - Each sprite has a length-4 bar and a length-3 crossbar one cell closer to the pointed end.

    Rule:
    - For each sprite, follow its bar axis from the shorter end to the frame.
    - Paint that frame endpoint with the sprite color.
    - Paint the opposite endpoint on the same row or column black.
    - Leave the interior background, sprites, and all other frame pixels unchanged.

    Key insight:
    - The direction is encoded by the crossbar's off-center position: the nearer end is the pointing end.
    """
    g = np.asarray(grid, dtype=int)
    frame = _solid_frame_color(g)
    bg = int(get_bg(g[1:-1, 1:-1]))
    assert bg != frame, "interior background must differ from the frame"

    sprite_colors = colors_present(g, exclude={bg, frame})
    assert sprite_colors, "expected at least one sprite"
    assert Color.BLACK not in set(sprite_colors) | {bg, frame}, "black is reserved for opposite endpoints"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=sprite_colors)
    assert len(components) == len(sprite_colors), "expected one 4-connected object per sprite color"

    assignments: dict[tuple[int, int], int] = {}
    for component in components:
        assert int(component["area"]) == 6, "each sprite must have six pixels"
        color = int(component["color"])
        pointed, opposite = _sprite_endpoints(g, component["coords"])

        for coord, value in ((pointed, color), (opposite, Color.BLACK)):
            assert int(g[coord]) == frame, "sprite endpoints must land on frame pixels"
            assert coord not in assignments, "frame endpoint assignments must not overlap"
            assignments[coord] = int(value)

    out = g.copy()
    for (r, c), value in assignments.items():
        out[r, c] = value
    return out

# --------------- generator (att2) ---------------

# Canvas and fixed colors.
GRID_SIZE, FRAME_COLOR, BACKGROUND_COLOR = 11, 6, 7
LAST = GRID_SIZE - 1

# Sprite sampling and shape.
SPRITE_COUNT, SPRITE_COLORS = 4, [1, 2, 3, 4, 5, 8, 9]
ORIENTATIONS, DIRECTIONS = ("H", "V"), {"H": ("left", "right"), "V": ("up", "down")}
BAR_LENGTH, CROSSBAR_LENGTH = 4, 3
CROSSBAR_MID, NEAR_END_OFFSET, FAR_END_OFFSET = 1, 1, 2
TRANSPARENT = -1

# Interior placement and retry bounds.
AXIS_LINES, MAX_GENERATION_ATTEMPTS = range(2, 9), 300
BAR_START_MIN, BAR_START_MAX = 1, 6

def _make_sprite(color: int, orientation: str, direction: str) -> np.ndarray:
    """Return one transparent-backed asymmetric T sprite patch."""
    patch = np.full((CROSSBAR_LENGTH, BAR_LENGTH) if orientation == "H" else (BAR_LENGTH, CROSSBAR_LENGTH), TRANSPARENT, dtype=int)
    if orientation == "H":
        offset = NEAR_END_OFFSET if direction == DIRECTIONS["H"][0] else FAR_END_OFFSET
        patch[CROSSBAR_MID, :] = color
        patch[:, offset] = color
    else:
        offset = NEAR_END_OFFSET if direction == DIRECTIONS["V"][0] else FAR_END_OFFSET
        patch[:, CROSSBAR_MID] = color
        patch[offset, :] = color
    return patch

def _sample_plan():
    """Sample layout-independent choices in the original random order."""
    colors = random.sample(SPRITE_COLORS, SPRITE_COUNT)
    orientations = [random.choice(ORIENTATIONS) for _ in range(SPRITE_COUNT)]
    axes_by_orientation = {
        orient: random.sample(AXIS_LINES, orientations.count(orient))
        for orient in ORIENTATIONS
    }
    order = list(range(SPRITE_COUNT))
    random.shuffle(order)
    return colors, orientations, axes_by_orientation, order

def _try_place_sprite(grid, color, orientation, axis, placed_endpoints) -> bool:
    """Sample one sprite's direction/start and paste it if pixels/endpoints are free."""
    direction = random.choice(DIRECTIONS[orientation])
    bar_start = random.randint(BAR_START_MIN, BAR_START_MAX)
    if orientation == "H":
        top, left = axis - CROSSBAR_MID, bar_start
        near_frame, far_frame = (axis, 0), (axis, LAST)
    else:
        top, left = bar_start, axis - CROSSBAR_MID
        near_frame, far_frame = (0, axis), (LAST, axis)
    pointed_frame, opposite_frame = (near_frame, far_frame) if direction == DIRECTIONS[orientation][0] else (far_frame, near_frame)
    if pointed_frame in placed_endpoints or opposite_frame in placed_endpoints:
        return False

    patch = _make_sprite(color, orientation, direction)
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != TRANSPARENT
    if view.shape != patch.shape or np.any(view[mask] != BACKGROUND_COLOR):
        return False

    paste(grid, patch, top, left, transparent=TRANSPARENT)
    placed_endpoints[pointed_frame] = color
    placed_endpoints[opposite_frame] = Color.BLACK
    return True

def _matches_solver_oracle(grid, placed_endpoints) -> bool:
    """Check that the solver agrees with the recorded frame endpoint edits."""
    expected = grid.copy()
    for coord, color in placed_endpoints.items():
        expected[coord] = color
    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False

def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = FRAME_COLOR
    colors, orientations, axes_by_orientation, order = _sample_plan()
    placed_endpoints: dict[tuple[int, int], int] = {}

    for i in order:
        orientation = orientations[i]
        axis = axes_by_orientation[orientation].pop()
        if not _try_place_sprite(grid, colors[i], orientation, axis, placed_endpoints):
            return None

    return grid if _matches_solver_oracle(grid, placed_endpoints) else None

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 689c358e input")
