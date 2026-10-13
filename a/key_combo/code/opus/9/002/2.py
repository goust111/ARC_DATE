from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_SPAN = 5
GRID_SIDE_RANGE = (16, 27)
MAX_SPAN = 13
GENERATION_RETRIES = 400


def _parse_markers(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coords = np.argwhere(g != BACKGROUND)
    assert coords.shape[0] == 4, "expected exactly four foreground markers"

    colors = [int(g[r, c]) for r, c in coords]
    uniques, counts = np.unique(colors, return_counts=True)
    assert len(uniques) == 2 and set(map(int, counts)) == {2}, "expected two marker colors with two cells each"

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    r_min, c_min, r_max, c_max = map(int, (r_min, c_min, r_max, c_max))
    assert r_max - r_min >= MIN_SPAN and c_max - c_min >= MIN_SPAN, "rectangle must be large enough"

    corners = {
        (r_min, c_min),
        (r_min, c_max),
        (r_max, c_min),
        (r_max, c_max),
    }
    color_at = {(int(r), int(c)): int(g[r, c]) for r, c in coords}
    assert set(color_at) == corners, "markers must occupy the four rectangle corners"

    tl = color_at[(r_min, c_min)]
    tr = color_at[(r_min, c_max)]
    bl = color_at[(r_max, c_min)]
    br = color_at[(r_max, c_max)]
    assert tl == br and tr == bl and tl != tr, "marker colors must pair across the two diagonals"

    return (r_min, c_min, r_max, c_max), color_at


def _render_output(
    shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    color_at: dict[tuple[int, int], int],
    *,
    interior_mode: str = "edge_sum",
    boundary_mode: str = "even",
) -> np.ndarray:
    r_min, c_min, r_max, c_max = bounds
    tl_color = int(color_at[(r_min, c_min)])
    other_color = int(color_at[(r_min, c_max)])
    out = np.zeros(shape, dtype=int)

    for (r, c), color in color_at.items():
        out[r, c] = int(color)

    def pick_interior(r: int, c: int, dr: int, dc: int) -> int:
        if interior_mode == "edge_sum":
            parity = (dr + dc) % 2
        elif interior_mode == "local_checker":
            parity = ((r - r_min) + (c - c_min)) % 2
        elif interior_mode == "dr_only":
            parity = dr % 2
        elif interior_mode == "dc_only":
            parity = dc % 2
        elif interior_mode == "min_edge":
            parity = min(dr, dc) % 2
        elif interior_mode == "max_edge":
            parity = max(dr, dc) % 2
        elif interior_mode == "product":
            parity = (dr * dc) % 2
        else:
            raise ValueError(f"unknown interior mode: {interior_mode}")
        return tl_color if parity == 0 else other_color

    for c in range(c_min + 1, c_max):
        dist = min(c - c_min, c_max - c)
        should_draw = boundary_mode == "solid" or (dist % 2 == 0) == (boundary_mode == "even")
        if should_draw:
            out[r_min, c] = GREY
            out[r_max, c] = GREY

    for r in range(r_min + 1, r_max):
        dist = min(r - r_min, r_max - r)
        should_draw = boundary_mode == "solid" or (dist % 2 == 0) == (boundary_mode == "even")
        if should_draw:
            out[r, c_min] = GREY
            out[r, c_max] = GREY

    for r in range(r_min + 1, r_max):
        for c in range(c_min + 1, c_max):
            dr = min(r - r_min, r_max - r)
            dc = min(c - c_min, c_max - c)
            out[r, c] = pick_interior(r, c, dr, dc)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the rectangle implied by two diagonal marker pairs with a border-anchored woven pattern.

    Input structure:
    - The grid has black background and exactly four non-background single-cell markers.
    - The four markers sit on the corners of one implicit axis-aligned rectangle.
    - One diagonal pair shares one color and the other diagonal pair shares a second color.

    Rule:
    - Use the marker min/max rows and columns as the rectangle bounds.
    - Keep the four corner markers unchanged, and draw the non-corner boundary cells in grey when their Chebyshev distance to the nearer corner is even; leave the odd ones black to make a dashed frame.
    - For each strict interior cell, compute its distances to the nearer horizontal and vertical rectangle edges; color it with the top-left corner color when the distance sum is even, otherwise use the other marker color.

    Key insight:
    - The diagonal color pairing determines both the rectangle geometry and which of the two colors is the even-parity weave color, while the parity itself is anchored to distances from the rectangle edges rather than to absolute grid coordinates.
    """
    bounds, color_at = _parse_markers(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int).shape, bounds, color_at)


def _sample_interval(limit: int) -> tuple[int, int] | None:
    max_span = min(MAX_SPAN, limit - 3)
    if max_span < MIN_SPAN:
        return None
    span = random.randint(MIN_SPAN, max_span)
    start = random.randint(1, limit - span - 2)
    return start, start + span


def _make_input(
    shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    colors: tuple[int, int],
    tl_uses_first: bool,
) -> np.ndarray:
    h, w = shape
    r_min, c_min, r_max, c_max = bounds
    color_a, color_b = colors
    tl_color = color_a if tl_uses_first else color_b
    other = color_b if tl_uses_first else color_a

    g = np.zeros((h, w), dtype=int)
    g[r_min, c_min] = tl_color
    g[r_max, c_max] = tl_color
    g[r_min, c_max] = other
    g[r_max, c_min] = other
    return g


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        bounds, color_at = _parse_markers(inp)
        out = main(inp)
    except AssertionError:
        return False

    expected = _render_output(inp.shape, bounds, color_at)
    if not np.array_equal(out, expected):
        return False

    r_min, c_min, r_max, c_max = bounds
    interior = out[r_min + 1 : r_max, c_min + 1 : c_max]
    if interior.size == 0:
        return False

    tl_color = color_at[(r_min, c_min)]
    other_color = color_at[(r_min, c_max)]
    if set(map(int, np.unique(interior))) != {tl_color, other_color}:
        return False

    variants = [
        _render_output(inp.shape, bounds, color_at, interior_mode="local_checker"),
        _render_output(inp.shape, bounds, color_at, interior_mode="dr_only"),
        _render_output(inp.shape, bounds, color_at, interior_mode="dc_only"),
        _render_output(inp.shape, bounds, color_at, interior_mode="min_edge"),
        _render_output(inp.shape, bounds, color_at, interior_mode="max_edge"),
        _render_output(inp.shape, bounds, color_at, interior_mode="product"),
        _render_output(inp.shape, bounds, color_at, boundary_mode="solid"),
        _render_output(inp.shape, bounds, color_at, boundary_mode="odd"),
    ]
    if any(np.array_equal(out, alt) for alt in variants):
        return False

    top_edge = out[r_min, c_min + 1 : c_max]
    left_edge = out[r_min + 1 : r_max, c_min]
    if GREY not in top_edge or BACKGROUND not in top_edge:
        return False
    if GREY not in left_edge or BACKGROUND not in left_edge:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    row_interval = _sample_interval(shape[0])
    col_interval = _sample_interval(shape[1])
    if row_interval is None or col_interval is None:
        return None

    r_min, r_max = row_interval
    c_min, c_max = col_interval
    colors = tuple(random.sample(MARKER_PALETTE, 2))
    tl_uses_first = bool(random.getrandbits(1))
    inp = _make_input(shape, (r_min, c_min, r_max, c_max), colors, tl_uses_first)
    return inp if _semantic_checks(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid rectangle-weave ARC input")
