from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

MIN_SIZE = 11
MAX_SIZE = 16
MIN_DIAG_LEN = 8
MAX_DIAG_LEN = 13
MAX_MARGIN = 2
MIN_BLOCKER_GAP = 3
MAX_GENERATION_ATTEMPTS = 500
MIN_REGION_COUNT = 4
MIN_REGION_AREA = 2


def _diag_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    assert dr > 0 and dr == dc
    return [(ar + i, ac + i) for i in range(dr + 1)]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _parse_input(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == BLUE)]
    red_pts = {tuple(map(int, pt)) for pt in np.argwhere(g == RED)}
    nonzero = set(tuple(map(int, pt)) for pt in np.argwhere(g != BLACK))

    assert len(blue_pts) == 2
    assert 2 <= len(red_pts) <= 3
    assert len(nonzero) == 2 + len(red_pts)

    start, end = sorted(blue_pts)
    segment = _diag_segment(start, end)
    interior = set(segment[1:-1])
    assert red_pts <= interior
    blockers = [pt for pt in segment[1:-1] if pt in red_pts]
    assert len(blockers) == len(red_pts)

    blocker_steps = [r - start[0] for r, _c in blockers]
    assert blocker_steps[0] >= 2
    assert blocker_steps[-1] <= len(segment) - 3
    for a, b in zip(blocker_steps, blocker_steps[1:]):
        assert b - a >= MIN_BLOCKER_GAP

    return start, end, segment, blockers


def _render_lines(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = map(int, out.shape)
    _start, _end, segment, blockers = _parse_input(out)

    for r, c in segment:
        if int(out[r, c]) == BLACK:
            out[r, c] = BLUE
        else:
            assert int(out[r, c]) in (BLUE, RED)

    for br, bc in blockers:
        for dr, dc in ((-1, 1), (1, -1)):
            r, c = br + dr, bc + dc
            while 0 <= r < height and 0 <= c < width:
                if int(out[r, c]) in (BLUE, RED):
                    break
                out[r, c] = RED
                r += dr
                c += dc

    return out, blockers


def _background_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) != BLACK:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == BLACK:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _region_key(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return min(coords)


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int], edge: str) -> bool:
    height, width = shape
    if edge == "top":
        return any(r == 0 for r, _c in coords)
    if edge == "bottom":
        return any(r == height - 1 for r, _c in coords)
    if edge == "left":
        return any(c == 0 for _r, c in coords)
    if edge == "right":
        return any(c == width - 1 for _r, c in coords)
    raise ValueError("unknown edge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the main blue diagonal, let each red blocker emit a perpendicular diagonal, then fill every other enclosed region green.

    Input structure:
    - The grid is mostly black background with exactly two blue endpoints on one descending diagonal.
    - Two or three red blocker pixels lie on the interior cells of that same diagonal, spaced apart.
    - No other foreground pixels are present in the input; the blockers mark where perpendicular rays must be emitted.

    Rule:
    - Draw the full blue descending diagonal segment between the two blue endpoints, preserving any red blockers already on it.
    - From each red blocker, cast red rays in the two perpendicular diagonal directions (up-right and down-left), stopping just before any existing blue or red pixel or at the grid edge.
    - Treat the remaining black cells as 4-connected background regions separated by the drawn lines and the grid boundary.
    - Sort those background regions by the smallest `(row, col)` cell they contain, and fill the 1st, 3rd, 5th, ... regions with green while leaving the even-indexed ones black.

    Key insight:
    - The output depends on the whole dependency chain: the blue segment determines where blockers sit, the blockers determine the red diagonal barriers, and the final green pattern comes from globally ordering the resulting enclosed background regions in row-major scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out, _blockers = _render_lines(g)
    regions = sorted(_background_components(out), key=_region_key)
    for index, coords in enumerate(regions, start=1):
        if index % 2 == 1:
            for r, c in coords:
                out[r, c] = GREEN
    return out


def _sample_geometry() -> tuple[int, int, int, int, int]:
    while True:
        diag_len = int(np.random.randint(MIN_DIAG_LEN, MAX_DIAG_LEN + 1))
        top = int(np.random.randint(0, MAX_MARGIN + 1))
        bottom = int(np.random.randint(0, MAX_MARGIN + 1))
        left = int(np.random.randint(0, MAX_MARGIN + 1))
        right = int(np.random.randint(0, MAX_MARGIN + 1))
        height = diag_len + top + bottom
        width = diag_len + left + right
        if MIN_SIZE <= height <= MAX_SIZE and MIN_SIZE <= width <= MAX_SIZE:
            return height, width, diag_len, top, left


def _sample_blocker_indices(diag_len: int) -> list[int] | None:
    if diag_len < 8:
        return None

    if diag_len >= 11:
        blocker_count = 3 if float(np.random.random()) < 0.6 else 2
    else:
        blocker_count = 2

    valid = list(range(2, diag_len - 2))
    for _ in range(200):
        picks = sorted(int(v) for v in np.random.choice(valid, size=blocker_count, replace=False))
        if all(b - a >= MIN_BLOCKER_GAP for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_input(height: int, width: int, diag_len: int, top: int, left: int, blocker_indices: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    start = (top, left)
    end = (top + diag_len - 1, left + diag_len - 1)
    grid[start] = BLUE
    grid[end] = BLUE
    for step in blocker_indices:
        grid[top + step, left + step] = RED
    return grid


def _has_scan_tie(regions: list[list[tuple[int, int]]]) -> bool:
    keys = [_region_key(coords) for coords in regions]
    rows = [r for r, _c in keys]
    cols = [c for _r, c in keys]
    return len(set(rows)) < len(rows) or len(set(cols)) < len(cols)


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        _start, _end, segment, blockers = _parse_input(grid)
    except AssertionError:
        return False

    if len(segment) > 30:
        return False

    lines_only, _ = _render_lines(grid)
    regions = sorted(_background_components(lines_only), key=_region_key)
    if len(regions) < MIN_REGION_COUNT:
        return False
    if any(len(coords) < MIN_REGION_AREA for coords in regions):
        return False

    odd_regions = regions[::2]
    even_regions = regions[1::2]
    if not odd_regions or not even_regions:
        return False

    shape = tuple(map(int, grid.shape))
    if not any(_touches_edge(coords, shape, "top") for coords in regions):
        return False
    if not any(_touches_edge(coords, shape, "bottom") for coords in regions):
        return False
    if not any(_touches_edge(coords, shape, "left") for coords in regions):
        return False
    if not any(_touches_edge(coords, shape, "right") for coords in regions):
        return False

    blocker_steps = [r - _start[0] for r, _c in blockers]
    if len(set(np.diff(blocker_steps))) == 1 and len(blocker_steps) > 1 and not _has_scan_tie(regions):
        return False

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return False
    if int(np.count_nonzero(out == GREEN)) == 0:
        return False
    if not np.any(out == BLACK):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height, width, diag_len, top, left = _sample_geometry()
        blocker_indices = _sample_blocker_indices(diag_len)
        if blocker_indices is None:
            continue
        grid = _build_input(height, width, diag_len, top, left, blocker_indices)
        if _is_valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
