from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
HEADER_PALETTE = (1, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (8, 24)
GRID_WIDTH_RANGE = (8, 24)
HEADER_LEN_RANGE = (2, 6)
RING_COUNT_RANGE = (3, 5)
MAX_GRAY_FRACTION = 0.40
GENERATION_RETRIES = 400

Coord = tuple[int, int]


def _parse_header(row: np.ndarray) -> list[int]:
    end = 0
    while end < row.size and int(row[end]) != BLACK:
        end += 1
    assert end > 0, "expected a non-empty top-row palette sequence"
    assert np.all(row[end:] == BLACK), "expected the top-row palette to be contiguous from the left"
    return [int(color) for color in row[:end]]


def main(grid: np.ndarray) -> np.ndarray:
    """Use the top-row palette as a cyclic distance lookup to recolor each gray pixel by its Chebyshev distance from the lone red marker.

    Input structure:
    - The top row starts with a contiguous non-black palette sequence and is black elsewhere.
    - Below the top row, the canvas is black except for exactly one red marker pixel and one or more gray pixels.
    - Gray pixels may lie farther from the marker than the palette length.

    Rule:
    - Read the top-row palette from left to right as colors 1..L.
    - For each gray pixel, compute its Chebyshev distance d from the red marker: `max(abs(dr), abs(dc))`.
    - Recolor that gray pixel with palette color `((d - 1) mod L)`, keep the red marker unchanged, and clear the entire top row to black.

    Key insight:
    - The header is a cyclic lookup table indexed by Chebyshev distance, so equal-color pixels belong to concentric square rings around the marker rather than diamond-shaped Manhattan rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a header row and a canvas"

    header = _parse_header(g[0])
    canvas = g[1:]

    marker_coords = np.argwhere(canvas == RED)
    assert marker_coords.shape[0] == 1, "expected exactly one red marker below the header"
    gray_coords = np.argwhere(canvas == GRAY)
    assert gray_coords.shape[0] >= 1, "expected at least one gray pixel below the header"

    allowed_canvas_colors = {BLACK, RED, GRAY}
    present_canvas_colors = set(map(int, np.unique(canvas)))
    assert present_canvas_colors <= allowed_canvas_colors, "unexpected non-header colors in the canvas"

    marker_r = int(marker_coords[0, 0]) + 1
    marker_c = int(marker_coords[0, 1])
    palette_len = len(header)

    out = np.full_like(g, BLACK)
    out[marker_r, marker_c] = RED

    for gray_r0, gray_c in gray_coords:
        gray_r = int(gray_r0) + 1
        d = max(abs(gray_r - marker_r), abs(int(gray_c) - marker_c))
        assert d >= 1, "gray pixels must not coincide with the marker"
        out[gray_r, int(gray_c)] = header[(d - 1) % palette_len]

    return out


def _max_canvas_distance(height: int, width: int, marker: Coord) -> int:
    mr, mc = marker
    return max(mr - 1, height - 1 - mr, mc, width - 1 - mc)


def _ring_cells(height: int, width: int, marker: Coord, dist: int) -> list[Coord]:
    mr, mc = marker
    cells: list[Coord] = []
    for r in range(max(1, mr - dist), min(height - 1, mr + dist) + 1):
        for c in range(max(0, mc - dist), min(width - 1, mc + dist) + 1):
            if max(abs(r - mr), abs(c - mc)) == dist:
                cells.append((r, c))
    return cells


def _ring_hint_cells(height: int, width: int, marker: Coord, dist: int) -> list[Coord]:
    mr, mc = marker
    hints = [
        (mr - dist, mc - dist),
        (mr - dist, mc),
        (mr - dist, mc + dist),
        (mr, mc - dist),
        (mr, mc + dist),
        (mr + dist, mc - dist),
        (mr + dist, mc),
        (mr + dist, mc + dist),
    ]
    seen: set[Coord] = set()
    out: list[Coord] = []
    for r, c in hints:
        if 1 <= r < height and 0 <= c < width and max(abs(r - mr), abs(c - mc)) == dist:
            cell = (r, c)
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
    return out


def _ring_side_cells(height: int, width: int, marker: Coord, dist: int, side: str) -> list[Coord]:
    mr, mc = marker
    cells: list[Coord] = []
    if side == "top":
        r = mr - dist
        for c in range(mc - dist, mc + dist + 1):
            if 1 <= r < height and 0 <= c < width:
                cells.append((r, c))
    elif side == "bottom":
        r = mr + dist
        for c in range(mc - dist, mc + dist + 1):
            if 1 <= r < height and 0 <= c < width:
                cells.append((r, c))
    elif side == "left":
        c = mc - dist
        for r in range(mr - dist, mr + dist + 1):
            if 1 <= r < height and 0 <= c < width:
                cells.append((r, c))
    else:
        c = mc + dist
        for r in range(mr - dist, mr + dist + 1):
            if 1 <= r < height and 0 <= c < width:
                cells.append((r, c))
    return cells


def _sample_ring_subset(height: int, width: int, marker: Coord, dist: int, *, force_full: bool) -> list[Coord]:
    ring = _ring_cells(height, width, marker, dist)
    assert ring, "every chosen ring distance must have visible cells"
    if force_full:
        return ring

    ring_set = set(ring)
    style = random.choices(
        ["sides", "sparse", "full"],
        weights=[4, 4, 1 if len(ring) > 24 else 2],
        k=1,
    )[0]
    chosen = set(_ring_hint_cells(height, width, marker, dist))

    if style == "full":
        return ring

    if style == "sides":
        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(2, 4)]:
            side_cells = [cell for cell in _ring_side_cells(height, width, marker, dist, side) if cell in ring_set]
            if len(side_cells) <= 2 or random.random() < 0.5:
                chosen.update(side_cells)
                continue
            seg_len = random.randint(max(2, len(side_cells) // 2), len(side_cells))
            start = random.randint(0, len(side_cells) - seg_len)
            chosen.update(side_cells[start : start + seg_len])
        return sorted(chosen)

    remainder = [cell for cell in ring if cell not in chosen]
    if remainder:
        extra_min = max(1, len(remainder) // 5)
        extra_max = max(extra_min, len(remainder) // 2)
        chosen.update(random.sample(remainder, random.randint(extra_min, extra_max)))
    return sorted(chosen)


def _choose_marker(height: int, width: int, header_len: int) -> Coord | None:
    row_choices = list(range(2, height - 1))
    col_choices = list(range(1, width - 1))
    random.shuffle(row_choices)
    random.shuffle(col_choices)
    for mr in row_choices:
        for mc in col_choices:
            marker = (mr, mc)
            if _max_canvas_distance(height, width, marker) >= header_len + 1:
                return marker
    return None


def _choose_distances(max_dist: int, header_len: int) -> list[int]:
    target = random.randint(RING_COUNT_RANGE[0], min(RING_COUNT_RANGE[1], max_dist))
    distances = {1, header_len}
    if header_len + 1 <= max_dist:
        distances.add(header_len + 1)
    while len(distances) < target:
        distances.add(random.randint(1, max_dist))
    return sorted(d for d in distances if 1 <= d <= max_dist)


def _gen_attempt() -> np.ndarray | None:
    width_min = max(GRID_WIDTH_RANGE[0], HEADER_LEN_RANGE[0] + 3)
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(width_min, GRID_WIDTH_RANGE[1])

    max_header_len = min(HEADER_LEN_RANGE[1], width - 2)
    if max_header_len < HEADER_LEN_RANGE[0]:
        return None
    header_len = random.randint(HEADER_LEN_RANGE[0], max_header_len)

    marker = _choose_marker(height, width, header_len)
    if marker is None:
        return None
    max_dist = _max_canvas_distance(height, width, marker)
    if max_dist < header_len + 1 or max_dist < RING_COUNT_RANGE[0]:
        return None

    header = random.sample(HEADER_PALETTE, header_len)
    distances = _choose_distances(max_dist, header_len)

    gray_cells: set[Coord] = set()
    for idx, dist in enumerate(distances):
        force_full = idx == 0 and len(_ring_cells(height, width, marker, dist)) <= 28
        gray_cells.update(_sample_ring_subset(height, width, marker, dist, force_full=force_full))

    if any(cell == marker for cell in gray_cells):
        return None
    if not gray_cells:
        return None
    if not any(max(abs(r - marker[0]), abs(c - marker[1])) > header_len for r, c in gray_cells):
        return None

    canvas_area = (height - 1) * width
    if len(gray_cells) > int(canvas_area * MAX_GRAY_FRACTION):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :header_len] = header
    grid[marker] = RED
    for r, c in gray_cells:
        grid[r, c] = GRAY

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.all(out[0] == BLACK):
        return None
    if out[marker] != RED:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring palette input")
