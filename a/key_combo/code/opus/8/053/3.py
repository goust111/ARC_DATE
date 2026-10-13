from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

CORNER_COLORS = (BLUE, RED, GREEN)
SCATTER_COLORS = (YELLOW, ORANGE)
GRID_SIDE_RANGE = (18, 24)
SQUARE_SIDE_OPTIONS = (8, 10, 12)
COUNT_RANGE = (2, 6)
MAX_ATTEMPTS = 500

Coord = tuple[int, int]


def _infer_square(grid: np.ndarray) -> dict[str, object]:
    corner_positions: dict[int, Coord] = {}
    for color in CORNER_COLORS:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 1, "each corner color must appear exactly once"
        r, c = map(int, coords[0])
        corner_positions[int(color)] = (r, c)

    coords = np.array(list(corner_positions.values()), dtype=int)
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    assert len(rows) == 2 and len(cols) == 2, "three corner markers must occupy three square corners"

    r0, r1 = rows
    c0, c1 = cols
    side_r = r1 - r0
    side_c = c1 - c0
    assert side_r == side_c and side_r > 1, "corner bbox must be a non-degenerate square"
    assert side_r % 2 == 0, "square side must give an integer center"

    full_corners = {(r, c) for r in rows for c in cols}
    observed = set(corner_positions.values())
    assert len(observed) == 3 and observed < full_corners, "must observe exactly three of the four square corners"
    missing = next(iter(full_corners - observed))

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return {
        "corner_positions": corner_positions,
        "bbox": (r0, c0, r1, c1),
        "side": side_r,
        "missing": missing,
        "center": center,
    }


def _count_parameters(grid: np.ndarray) -> tuple[int, int]:
    yellow_count = int(np.count_nonzero(grid == YELLOW))
    orange_count = int(np.count_nonzero(grid == ORANGE))
    assert yellow_count > 0 and orange_count > 0, "both scatter colors must be present"
    assert yellow_count != orange_count, "scatter-color counts must be distinct"
    return tuple(sorted((yellow_count, orange_count)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing square corner, complete the square frame, and fill a count-sized inner rectangle.

    Input structure:
    - A black background contains exactly three single-pixel corner markers in blue, red, and green.
    - Those three pixels occupy three corners of one axis-aligned square, with the fourth corner absent.
    - Yellow and orange single-pixel scatter markers lie outside that square and encode two distinct counts.

    Rule:
    - Use the three colored corners to infer the full square, its missing corner, and its geometric center.
    - Count the yellow markers and the orange markers; let the smaller count be the fill height and the larger count be the fill width.
    - Draw the square boundary in grey, keep the original three colored corner markers, fill a grey solid rectangle inside the square with its lower-left corner aligned to the square's inner lower-left corner, and keep the missing corner grey.
    - Paint the square center magenta and preserve every yellow and orange scatter marker.

    Key insight:
    - The square geometry and the rectangle size come from two independent signal sources, and both must be combined to determine where the grey fill belongs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    square = _infer_square(g)
    fill_h, fill_w = _count_parameters(g)
    r0, c0, r1, c1 = map(int, square["bbox"])
    center_r, center_c = map(int, square["center"])
    side = int(square["side"])
    assert fill_h <= side - 1 and fill_w <= side - 1, "fill rectangle must fit strictly inside the square"

    out = g.copy()
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY

    out[r1 - fill_h : r1, c0 + 1 : c0 + 1 + fill_w] = GREY

    for color, (r, c) in square["corner_positions"].items():
        out[int(r), int(c)] = int(color)

    missing_r, missing_c = map(int, square["missing"])
    out[missing_r, missing_c] = GREY
    out[center_r, center_c] = MAGENTA
    return out


def _zone_for_cell(cell: Coord, bbox: tuple[int, int, int, int]) -> str:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if r < r0:
        return "top"
    if r > r1:
        return "bottom"
    if c < c0:
        return "left"
    return "right"


def _scatter_candidates(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> list[Coord]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = bbox
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            cells.append((r, c))
    return cells


def _place_scatter(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    yellow_count: int,
    orange_count: int,
) -> bool:
    candidates = _scatter_candidates(grid.shape, bbox)
    if len(candidates) < yellow_count + orange_count:
        return False

    random.shuffle(candidates)
    chosen = candidates[: yellow_count + orange_count]
    zones = {_zone_for_cell(cell, bbox) for cell in chosen}
    if len(zones) < 2:
        return False

    yellow_cells = chosen[:yellow_count]
    orange_cells = chosen[yellow_count:]
    if len({_zone_for_cell(cell, bbox) for cell in yellow_cells}) < 1:
        return False
    if len({_zone_for_cell(cell, bbox) for cell in orange_cells}) < 1:
        return False

    for r, c in yellow_cells:
        grid[r, c] = YELLOW
    for r, c in orange_cells:
        grid[r, c] = ORANGE
    return True


def _expected_output_from_input(inp: np.ndarray) -> np.ndarray:
    square = _infer_square(inp)
    fill_h, fill_w = _count_parameters(inp)
    out = inp.copy()
    r0, c0, r1, c1 = map(int, square["bbox"])
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r1 - fill_h : r1, c0 + 1 : c0 + 1 + fill_w] = GREY
    for color, (r, c) in square["corner_positions"].items():
        out[int(r), int(c)] = int(color)
    mr, mc = map(int, square["missing"])
    cr, cc = map(int, square["center"])
    out[mr, mc] = GREY
    out[cr, cc] = MAGENTA
    return out


def _semantic_check(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if max(map(int, inp.shape)) > 30:
        return False

    present = set(map(int, np.unique(inp)))
    if not present <= {BG, BLUE, RED, GREEN, YELLOW, ORANGE}:
        return False

    try:
        square = _infer_square(inp)
        fill_h, fill_w = _count_parameters(inp)
    except AssertionError:
        return False

    r0, c0, r1, c1 = map(int, square["bbox"])
    side = int(square["side"])
    if side not in SQUARE_SIDE_OPTIONS:
        return False
    if not (COUNT_RANGE[0] <= fill_h <= COUNT_RANGE[1] and COUNT_RANGE[0] <= fill_w <= COUNT_RANGE[1]):
        return False
    if fill_w > side - 1:
        return False

    scatter_cells = np.argwhere(np.isin(inp, SCATTER_COLORS))
    if scatter_cells.shape[0] != fill_h + fill_w:
        return False
    zones = set()
    for r, c in scatter_cells:
        rr, cc = int(r), int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            return False
        if r0 - 1 <= rr <= r1 + 1 and c0 - 1 <= cc <= c1 + 1:
            return False
        zones.add(_zone_for_cell((rr, cc), square["bbox"]))
    if len(zones) < 2:
        return False

    center_r, center_c = map(int, square["center"])
    if inp[center_r, center_c] != BG:
        return False

    out = main(inp)
    expected = _expected_output_from_input(inp)
    if not np.array_equal(out, expected):
        return False

    boundary = np.zeros_like(inp, dtype=bool)
    boundary[r0, c0 : c1 + 1] = True
    boundary[r1, c0 : c1 + 1] = True
    boundary[r0 : r1 + 1, c0] = True
    boundary[r0 : r1 + 1, c1] = True
    corner_set = set(square["corner_positions"].values())
    for r, c in np.argwhere(boundary):
        rr, cc = int(r), int(c)
        if (rr, cc) in corner_set:
            if int(out[rr, cc]) != int(inp[rr, cc]):
                return False
        elif (rr, cc) == tuple(square["missing"]):
            if int(out[rr, cc]) != GREY:
                return False
        elif int(out[rr, cc]) != GREY:
            return False

    rect_mask = np.zeros_like(inp, dtype=bool)
    rect_mask[r1 - fill_h : r1, c0 + 1 : c0 + 1 + fill_w] = True
    for r, c in np.argwhere(rect_mask):
        rr, cc = int(r), int(c)
        if (rr, cc) == (center_r, center_c):
            if int(out[rr, cc]) != MAGENTA:
                return False
        elif int(out[rr, cc]) != GREY:
            return False

    if int(out[center_r, center_c]) != MAGENTA:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    side = random.choice(SQUARE_SIDE_OPTIONS)
    if h <= side + 2 or w <= side + 2:
        return None

    r0 = random.randint(1, h - side - 2)
    c0 = random.randint(1, w - side - 2)
    r1 = r0 + side
    c1 = c0 + side

    fill_pair = random.sample(range(COUNT_RANGE[0], COUNT_RANGE[1] + 1), 2)
    if random.random() < 0.5:
        yellow_count, orange_count = min(fill_pair), max(fill_pair)
    else:
        yellow_count, orange_count = max(fill_pair), min(fill_pair)

    grid = np.zeros((h, w), dtype=int)
    all_corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing_idx = random.randrange(4)
    present_corners = [corner for i, corner in enumerate(all_corners) if i != missing_idx]

    colors = list(CORNER_COLORS)
    random.shuffle(colors)
    for color, (r, c) in zip(colors, present_corners):
        grid[r, c] = color

    if not _place_scatter(grid, (r0, c0, r1, c1), yellow_count, orange_count):
        return None

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
