from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6


def _marker_path(color: int, r: int, c: int):
    if color == BLUE:
        return [(r, c + 1), (r, c + 2)], (r - 1, c + 2), None
    if color == RED:
        return [(r, c - 1), (r, c - 2), (r, c - 3)], (r, c - 4), [
            (r, cc) for cc in range(c - 4, c + 1)
        ]
    if color == GREEN:
        return [(r + 1, c), (r + 2, c)], (r + 3, c), [
            (rr, c) for rr in range(r, r + 4)
        ]
    if color == PINK:
        return [(rr, c) for rr in range(r - 1, r - 6, -1)], (r - 6, c), [
            (rr, c) for rr in range(r - 6, r + 1)
        ]
    raise AssertionError(f"unsupported marker color: {color}")


def main(grid: np.ndarray) -> np.ndarray:
    """Draw each colored marker's fixed trail, then turn red/vertical trail crossings into yellow down-left diagonals.

    Input structure
    - A black background contains one pixel of each present marker color drawn from blue, red, green, and pink.
    - The grey and yellow colors are absent from the input and are used only in the output trail drawing.

    Rule
    - Blue paints two grey pixels to the right and places a blue endpoint one row above the second grey pixel.
    - Red paints three grey pixels to the left and places a red endpoint four columns left.
    - Green paints two grey pixels downward and places a green endpoint three rows down.
    - Pink paints five grey pixels upward and places a pink endpoint six rows up.
    - Where a red horizontal segment crosses a green or pink vertical segment, paint yellow there and continue yellow diagonally down-left to the grid edge.

    Key insight
    - Marker color identifies a fixed motion/trail, and yellow is the collision effect between the red westward trail and a vertical trail.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == 0

    out = g.copy()
    red_segments = []
    vertical_segments = []

    for color in colors_present(g, exclude={bg}):
        assert color in {BLUE, RED, GREEN, PINK}
        coords = np.argwhere(g == color)
        assert len(coords) == 1
        r, c = map(int, coords[0])

        grey_cells, endpoint, segment = _marker_path(color, r, c)
        for rr, cc in grey_cells:
            assert in_bounds(g.shape, rr, cc)
            assert g[rr, cc] == bg
            assert out[rr, cc] in (bg, GREY)
            out[rr, cc] = GREY

        er, ec = endpoint
        assert in_bounds(g.shape, er, ec)
        assert g[er, ec] == bg
        assert out[er, ec] == bg
        out[er, ec] = color

        if color == RED:
            red_segments.append(set(segment))
        elif color in {GREEN, PINK}:
            vertical_segments.append(set(segment))

    crossings = []
    for red_segment in red_segments:
        for vertical_segment in vertical_segments:
            for rr, cc in sorted(red_segment & vertical_segment):
                assert g[rr, cc] == bg
                crossings.append((rr, cc))

    for rr, cc in crossings:
        while in_bounds(g.shape, rr, cc):
            assert g[rr, cc] == bg
            assert out[rr, cc] in (bg, GREY, YELLOW)
            out[rr, cc] = YELLOW
            rr += 1
            cc -= 1

    return out

# --------------- generator (o5x) ---------------
# Generator CFG: canvas and marker subset sampling.
GRID_SIZE_RANGE, MARKER_COUNT_RANGE = (9, 12), (1, 5)
MARKER_COLORS = [BLUE, RED, GREEN, PINK]
GENERATION_MAX_ATTEMPTS = 300

# Source bounds and crossing-biased pair placement.
VERTICAL_COLORS = (GREEN, PINK)
BLUE_MIN_ROW, BLUE_RIGHT_TRAIL = 1, 2
RED_LEFT_TRAIL, GREEN_DOWN_TRAIL, PINK_UP_TRAIL = 4, 3, 6
CROSSING_PAIR_PROB = 0.55
CROSSING_COL_BACK_RANGE = (1, 4)
GREEN_RED_ROW_OFFSET_RANGE, PINK_RED_ROW_OFFSET_RANGE = (1, 3), (1, 6)


def _gen_source_position(color: int, n: int) -> tuple[int, int]:
    if color == BLUE:
        return int(np.random.randint(BLUE_MIN_ROW, n)), int(np.random.randint(0, n - BLUE_RIGHT_TRAIL))
    if color == RED:
        return int(np.random.randint(0, n)), int(np.random.randint(RED_LEFT_TRAIL, n))
    if color == GREEN:
        return int(np.random.randint(0, n - GREEN_DOWN_TRAIL)), int(np.random.randint(0, n))
    if color == PINK:
        return int(np.random.randint(PINK_UP_TRAIL, n)), int(np.random.randint(0, n))
    raise AssertionError(f"unsupported marker color: {color}")


def _gen_crossing_sources(vertical_color: int, n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    red_c = int(np.random.randint(RED_LEFT_TRAIL, n))
    vertical_c = red_c - int(np.random.randint(*CROSSING_COL_BACK_RANGE))
    if vertical_color == GREEN:
        vertical_r = int(np.random.randint(0, n - GREEN_DOWN_TRAIL))
        red_r = vertical_r + int(np.random.randint(*GREEN_RED_ROW_OFFSET_RANGE))
    elif vertical_color == PINK:
        vertical_r = int(np.random.randint(PINK_UP_TRAIL, n))
        red_r = vertical_r - int(np.random.randint(*PINK_RED_ROW_OFFSET_RANGE))
    else:
        raise AssertionError(f"unsupported vertical color: {vertical_color}")
    return (red_r, red_c), (vertical_r, vertical_c)


def _choose_sources(chosen: list[int], n: int):
    """Sample source pixels, optionally biasing one red/vertical pair to cross."""
    sources = {}
    source_cells = set()
    verticals = [c for c in chosen if c in VERTICAL_COLORS]
    if RED in chosen and verticals and np.random.random() < CROSSING_PAIR_PROB:
        vertical_color = int(np.random.choice(verticals))
        red_source, vertical_source = _gen_crossing_sources(vertical_color, n)
        sources[RED] = red_source
        sources[vertical_color] = vertical_source
        source_cells.update((red_source, vertical_source))

    for color in chosen:
        if color in sources:
            continue
        pos = _gen_source_position(color, n)
        if pos in source_cells:
            return None
        sources[color] = pos
        source_cells.add(pos)
    return sources, source_cells


def _build_trail_records(sources: dict, source_cells: set):
    """Return per-color trail metadata and endpoints, or None on a collision."""
    records = {}
    grey_owners = {}
    endpoints = set()
    for color, (r, c) in sources.items():
        grey_cells, endpoint, segment = _marker_path(color, r, c)
        grey_cells = [tuple(cell) for cell in grey_cells]
        endpoint = tuple(endpoint)
        segment = set(segment) if segment is not None else None

        if endpoint in source_cells or endpoint in endpoints or any(cell in source_cells for cell in grey_cells):
            return None
        endpoints.add(endpoint)

        for cell in grey_cells:
            grey_owners.setdefault(cell, []).append(color)
        records[color] = {"grey": grey_cells, "endpoint": endpoint, "segment": segment}

    if any(endpoint in grey_owners for endpoint in endpoints):
        return None

    for owners in grey_owners.values():
        if len(owners) <= 1:
            continue
        owner_set = set(owners)
        if len(owner_set) != 2 or RED not in owner_set or not (owner_set - {RED}) <= set(VERTICAL_COLORS):
            return None
    return records, endpoints


def _collect_yellow_cells(n: int, records: dict, source_cells: set, endpoints: set):
    """Return yellow ray cells, or None for invalid red/vertical intersections."""
    yellow_cells = set()
    if RED not in records:
        return yellow_cells
    red_grey = set(records[RED]["grey"])
    red_segment = records[RED]["segment"]
    for color in VERTICAL_COLORS:
        if color not in records:
            continue
        vertical_grey = set(records[color]["grey"])
        vertical_segment = records[color]["segment"]
        for cell in sorted(red_segment & vertical_segment):
            if cell not in red_grey or cell not in vertical_grey:
                return None
            r, c = cell
            while 0 <= r < n and 0 <= c < n:
                ray_cell = (r, c)
                if ray_cell in source_cells or ray_cell in endpoints:
                    return None
                yellow_cells.add(ray_cell)
                r += 1
                c -= 1
    return yellow_cells


def _render_grids(n: int, sources: dict, records: dict, yellow_cells: set) -> tuple[np.ndarray, np.ndarray]:
    """Render accepted metadata into the input grid and oracle-checked output grid."""
    out = np.zeros((n, n), dtype=int)
    inp = np.zeros_like(out)
    for color, (r, c) in sources.items():
        out[r, c] = color
        inp[r, c] = color
    for color in sorted(records):
        for r, c in records[color]["grey"]:
            out[r, c] = GREY
        er, ec = records[color]["endpoint"]
        out[er, ec] = color
    for r, c in yellow_cells:
        out[r, c] = YELLOW

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*GRID_SIZE_RANGE))
    count = int(np.random.randint(*MARKER_COUNT_RANGE))
    chosen = [int(c) for c in np.random.choice(MARKER_COLORS, size=count, replace=False)]

    source_data = _choose_sources(chosen, n)
    if source_data is None:
        return None
    sources, source_cells = source_data

    trail_data = _build_trail_records(sources, source_cells)
    if trail_data is None:
        return None
    records, endpoints = trail_data

    yellow_cells = _collect_yellow_cells(n, records, source_cells, endpoints)
    if yellow_cells is None:
        return None

    inp, out = _render_grids(n, sources, records, yellow_cells)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_MAX_ATTEMPTS} attempts")
