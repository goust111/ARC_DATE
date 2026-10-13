from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEAL = 8
FILL_COLORS = np.array([1, 2, 3, 4, 6], dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    pts: list[tuple[int, int]] = []
    while True:
        pts.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    return pts


def _axis_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c0 <= c1 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r0 <= r1 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("expected an axis-aligned segment")


def _extract_grey_structure(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.shape == (4, 2), "expected exactly four grey pixels"

    r0, c0, r1, c1 = _bbox(grey_coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and 3 <= side_h <= 5, "grey markers must define a 3..5 square bbox"

    corners = {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }
    grey_set = {tuple(map(int, pt)) for pt in grey_coords}
    corner_marks = grey_set & corners
    assert len(corner_marks) == 3, "expected three grey corners and one missing corner"

    missing_corner = next(iter(corners - corner_marks))
    inner_marks = list(grey_set - corners)
    assert len(inner_marks) == 1, "expected one interior grey distractor"
    inner_r, inner_c = inner_marks[0]
    assert r0 < inner_r < r1 and c0 < inner_c < c1, "distractor must be strictly inside the square bbox"

    adjacent_corners = [
        corner
        for corner in corner_marks
        if corner[0] == missing_corner[0] or corner[1] == missing_corner[1]
    ]
    assert len(adjacent_corners) == 2, "missing corner should have two adjacent marked corners"
    return grey_set, missing_corner, adjacent_corners


def _triangle_edge_pixels(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    grey_set, missing_corner, adjacent_corners = _extract_grey_structure(grid)
    edge_pixels: set[tuple[int, int]] = set()

    for corner in adjacent_corners:
        edge_pixels.update(_axis_line(missing_corner, corner))
    edge_pixels.update(_bresenham_line(adjacent_corners[0], adjacent_corners[1]))

    cyan_pixels = set(edge_pixels)
    cyan_pixels.update(grey_set)
    return edge_pixels, cyan_pixels


def _find_orange_block(grid: np.ndarray) -> tuple[list[int], list[int]]:
    orange_coords = np.argwhere(grid == ORANGE)
    assert orange_coords.shape == (4, 2), "expected exactly one 2x2 orange block"
    r0, c0, r1, c1 = _bbox(orange_coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "orange marker must be 2x2"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE), "orange marker must fill its bbox"
    return [r0, r1], [c0, c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the missing-corner triangle in teal, then erase a block-defined cross while preserving teal pixels.

    Input structure:
    - The grid is mostly filled with non-black colors.
    - Exactly four grey pixels exist: three occupy corners of a square bbox and one is a strict interior distractor.
    - Exactly one orange 2x2 block exists elsewhere in the grid.

    Rule:
    - Identify the square bbox of the grey pixels and find which of its four corners is missing a grey marker.
    - Treat that missing corner as the right-angle vertex; connect it to the two adjacent marked corners with horizontal/vertical teal edges, and connect those two marked corners with a Bresenham diagonal to form the triangle outline.
    - Recolor every original grey pixel to teal as well.
    - Take the two rows and two columns occupied by the orange 2x2 block and turn every non-teal cell on those rows or columns black.

    Key insight:
    - The grey markers first define a geometric teal shield, and that newly drawn shield is exactly what survives the later cross-shaped erase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _edge_pixels, cyan_pixels = _triangle_edge_pixels(g)
    erase_rows, erase_cols = _find_orange_block(g)

    out = g.copy()
    for r, c in cyan_pixels:
        out[r, c] = TEAL

    erase_mask = np.zeros_like(out, dtype=bool)
    erase_mask[erase_rows, :] = True
    erase_mask[:, erase_cols] = True
    out[erase_mask & (out != TEAL)] = BLACK
    return out


def _pick_geometry(
    rng: np.random.Generator, height: int, width: int
) -> tuple[set[tuple[int, int]], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    side = int(rng.choice(np.array([3, 4, 4, 5, 5], dtype=int)))
    r0 = int(rng.integers(0, height - side + 1))
    c0 = int(rng.integers(0, width - side + 1))
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    corners = [
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    ]
    missing_corner = corners[int(rng.integers(0, 4))]
    present_corners = [corner for corner in corners if corner != missing_corner]
    adjacent_corners = [
        corner
        for corner in present_corners
        if corner[0] == missing_corner[0] or corner[1] == missing_corner[1]
    ]
    assert len(adjacent_corners) == 2

    edge_pixels: set[tuple[int, int]] = set()
    for corner in adjacent_corners:
        edge_pixels.update(_axis_line(missing_corner, corner))
    edge_pixels.update(_bresenham_line(adjacent_corners[0], adjacent_corners[1]))

    interior_cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    assert interior_cells
    off_edge_choices = [cell for cell in interior_cells if cell not in edge_pixels]
    distractor = off_edge_choices[int(rng.integers(0, len(off_edge_choices)))] if off_edge_choices else interior_cells[0]

    grey_pixels = set(present_corners)
    grey_pixels.add(distractor)
    cyan_pixels = set(edge_pixels)
    cyan_pixels.update(grey_pixels)
    return grey_pixels, missing_corner, edge_pixels, cyan_pixels


def _orange_candidates(
    height: int,
    width: int,
    missing_corner: tuple[int, int],
    edge_pixels: set[tuple[int, int]],
    cyan_pixels: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    miss_r, miss_c = missing_corner
    for br in range(height - 1):
        for bc in range(width - 1):
            rows = {br, br + 1}
            cols = {bc, bc + 1}
            if miss_r in rows or miss_c in cols:
                continue
            block_cells = {(br, bc), (br + 1, bc), (br, bc + 1), (br + 1, bc + 1)}
            if block_cells & cyan_pixels:
                continue
            if any((r in rows) or (c in cols) for r, c in edge_pixels):
                candidates.append((br, bc))
    return candidates


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        edge_pixels, cyan_pixels = _triangle_edge_pixels(grid)
        erase_rows, erase_cols = _find_orange_block(grid)
    except AssertionError:
        return False

    grey_coords = np.argwhere(grid == GREY)
    orange_coords = np.argwhere(grid == ORANGE)
    if grey_coords.shape != (4, 2) or orange_coords.shape != (4, 2):
        return False

    cross_hits = sum((r in erase_rows) or (c in erase_cols) for r, c in edge_pixels)
    if cross_hits == 0:
        return False

    out = main(grid)
    if not np.any(out == TEAL):
        return False
    if not np.any(out == BLACK):
        return False
    cross_mask = np.zeros_like(out, dtype=bool)
    cross_mask[erase_rows, :] = True
    cross_mask[:, erase_cols] = True
    if not np.any((out == TEAL) & cross_mask):
        return False
    if cyan_pixels & {tuple(map(int, pt)) for pt in orange_coords}:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        height = int(rng.integers(11, 17))
        width = int(rng.integers(11, 17))
        grid = rng.choice(FILL_COLORS, size=(height, width)).astype(int)

        try:
            grey_pixels, missing_corner, edge_pixels, cyan_pixels = _pick_geometry(rng, height, width)
        except AssertionError:
            continue

        orange_spots = _orange_candidates(height, width, missing_corner, edge_pixels, cyan_pixels)
        if not orange_spots:
            continue
        br, bc = orange_spots[int(rng.integers(0, len(orange_spots)))]

        for r, c in grey_pixels:
            grid[r, c] = GREY
        grid[br : br + 2, bc : bc + 2] = ORANGE

        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
