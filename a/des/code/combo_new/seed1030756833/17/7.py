import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _source_components(g: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere((g != BLACK) & (g != GREY) & (g != MAROON)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        color = int(g[sr, sc])
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _region_marker_direction(g: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int]:
    h, w = g.shape
    hits: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for r, c in comp:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == MAROON:
                hits.append(((nr, nc), (dr, dc), (r, c)))

    markers = {marker for marker, _, _ in hits}
    assert len(markers) == 1, "each colored region must touch exactly one maroon marker"
    assert len(hits) == 1, "the maroon marker must touch exactly one pixel of its region"

    marker, direction, _ = hits[0]
    mr, mc = marker
    dr, dc = direction
    r0, c0, r1, c1 = _bbox(comp)
    if (dr, dc) == (-1, 0):
        assert mr == r0 - 1 and c0 <= mc <= c1, "top marker must sit just outside the region bbox"
    elif (dr, dc) == (1, 0):
        assert mr == r1 + 1 and c0 <= mc <= c1, "bottom marker must sit just outside the region bbox"
    elif (dr, dc) == (0, -1):
        assert mc == c0 - 1 and r0 <= mr <= r1, "left marker must sit just outside the region bbox"
    else:
        assert mc == c1 + 1 and r0 <= mr <= r1, "right marker must sit just outside the region bbox"
    return direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw grey shadow projections from each colored region in its maroon-marker direction.

    Input structure:
    - The background is black.
    - Several non-overlapping 4-connected single-color regions are present; source
      colors are distinct and are not black, grey, or maroon.
    - Each region has exactly one 4-adjacent single maroon marker just outside the
      top, bottom, left, or right side of its bounding box.

    Rule:
    - For every colored region, find the adjacent maroon marker and use its relative
      side as the projection direction.
    - Let N be the number of pixels in that region.
    - From every pixel of the region, walk N cells in that direction and paint grey
      on black cells only.
    - Existing colored region pixels and maroon markers remain unchanged; any
      non-black cell along a ray is skipped while the ray continues.

    Key insight: The external maroon marker encodes direction, while the component
    area encodes the projection length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    comps = _source_components(g)
    assert comps, "at least one colored region is required"

    for comp in comps:
        dr, dc = _region_marker_direction(g, comp)
        length = len(comp)
        for r, c in comp:
            for step in range(1, length + 1):
                rr = r + dr * step
                cc = c + dc * step
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == BLACK:
                    out[rr, cc] = GREY

    return out


def _normalize(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return sorted((r - min_r, c - min_c) for r, c in points)


def _random_shape(size: int, rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(100):
        pts = {(0, 0)}
        while len(pts) < size:
            r, c = rng.choice(tuple(pts))
            dr, dc = rng.choice(DIRS)
            pts.add((r + dr, c + dc))
        shape = _normalize(pts)
        r0, c0, r1, c1 = _bbox(shape)
        if r1 - r0 <= 4 and c1 - c0 <= 4:
            return shape
    return shape


def _marker_for_shape(
    shape: list[tuple[int, int]],
    direction: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        choices = [(r, c) for r, c in shape if r == r0]
    elif (dr, dc) == (1, 0):
        choices = [(r, c) for r, c in shape if r == r1]
    elif (dr, dc) == (0, -1):
        choices = [(r, c) for r, c in shape if c == c0]
    else:
        choices = [(r, c) for r, c in shape if c == c1]
    r, c = rng.choice(choices)
    return r + dr, c + dc


def _local_required_cells(
    shape: list[tuple[int, int]],
    marker: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    length = len(shape)
    cells = list(shape) + [marker]
    for r, c in shape:
        for step in range(1, length + 1):
            cells.append((r + dr * step, c + dc * step))
    return cells


def _can_place(
    grid: np.ndarray,
    reserved: np.ndarray,
    shape: list[tuple[int, int]],
    marker: tuple[int, int],
    top: int,
    left: int,
) -> bool:
    actual = [(top + r, left + c) for r, c in shape + [marker]]
    for r, c in actual:
        if bool(reserved[r, c]) or int(grid[r, c]) != BLACK:
            return False
    return True


def _reserve_cells(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                reserved[rr, cc] = True


def _place_one(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: tuple[int, int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(100):
        size = rng.randint(3, 8)
        shape = _random_shape(size, rng)
        marker = _marker_for_shape(shape, direction, rng)
        needed = _local_required_cells(shape, marker, direction)
        min_r, min_c, max_r, max_c = _bbox(needed)
        top_lo, top_hi = -min_r, h - 1 - max_r
        left_lo, left_hi = -min_c, w - 1 - max_c
        if top_lo > top_hi or left_lo > left_hi:
            continue
        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        if not _can_place(grid, reserved, shape, marker, top, left):
            continue

        actual_shape = [(top + r, left + c) for r, c in shape]
        actual_marker = (top + marker[0], left + marker[1])
        for r, c in actual_shape:
            grid[r, c] = int(color)
        grid[actual_marker] = MAROON
        _reserve_cells(reserved, actual_shape + [actual_marker])
        return True
    return False


def _valid_generated(grid: np.ndarray, expected_regions: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    comps = _source_components(grid)
    if len(comps) != expected_regions:
        return False
    if not all(3 <= len(comp) <= 8 for comp in comps):
        return False

    marker_pts = [(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    if len(marker_pts) != expected_regions:
        return False
    source_colors = {int(grid[r, c]) for comp in comps for r, c in comp}
    for mr, mc in marker_pts:
        adj = 0
        for nr, nc in _neighbors4(mr, mc):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) in source_colors:
                adj += 1
        if adj != 1:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return bool(np.any((grid == BLACK) & (out == GREY)))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(17, 30)
    w = rng.randint(17, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    count = rng.randint(4, 6)
    colors = rng.sample(SOURCE_COLORS, count)
    directions = list(DIRS)
    while len(directions) < count:
        directions.append(rng.choice(DIRS))
    rng.shuffle(directions)

    for color, direction in zip(colors, directions):
        if not _place_one(grid, reserved, color, direction, rng):
            return None

    if not _valid_generated(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow projection input")
