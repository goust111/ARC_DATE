from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MAX_GENERATION_ATTEMPTS = 400
MIN_MARKER_GAP = 4


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int, np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == RED)]
    assert marker_coords, "expected at least one red marker"
    assert len({r for r, _c in marker_coords}) == len(marker_coords), "marker rows must be distinct"

    marker_cols = {c for _r, c in marker_coords}
    assert len(marker_cols) == 1, "all markers must lie in one column"
    marker_col = next(iter(marker_cols))

    blue_components = _components_mask(g == BLUE)
    assert len(blue_components) == len(marker_coords), "expected one blue component per marker"

    assignments: list[tuple[int, int, np.ndarray]] = []
    used_components: set[int] = set()
    for r, c in sorted(marker_coords):
        matching: list[tuple[int, np.ndarray]] = []
        for idx, comp in enumerate(blue_components):
            if idx in used_components:
                continue
            if int(comp[:, 1].max()) >= c:
                continue
            if np.any(comp[:, 0] == r):
                matching.append((idx, comp))
        assert len(matching) == 1, "each marker must match exactly one left-side blue component"
        idx, comp = matching[0]
        used_components.add(idx)
        assignments.append((r, c, comp))

    areas = [int(comp.shape[0]) for _r, _c, comp in assignments]
    assert len(set(areas)) == len(areas), "blue component areas must be distinct"
    assert all(2 <= area <= 5 for area in areas), "blue component areas must lie in [2, 5]"
    return g, marker_col, assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each red marker rightward with a finite alternating ray whose length is twice its adjacent blue component area.

    Input structure:
    - A black background contains 3 or 4 red marker pixels, each on a different row but all in the same column.
    - To the left of each red marker is exactly one blue 4-connected component, and each component has a distinct area from 2 to 5.
    - The right side of the grid is empty black space wide enough to hold the longest generated ray.

    Rule:
    - For each red marker, find the unique blue 4-connected component strictly to its left that also occupies the marker's row.
    - Let N be that component's area in pixels; overwrite offsets 0 through 2N-1 on the marker's row with an alternating pattern starting at the marker.
    - Even offsets are red, odd offsets are grey, cells at offset 2N and beyond remain unchanged, and every blue component stays as it was.

    Key insight:
    - The blue shape is only a parameter carrier: its pixel count, not its outline, determines exactly how far the alternating red-grey ray should extend.
    """
    g, _marker_col, assignments = _parse_instance(grid)
    out = g.copy()
    width = g.shape[1]

    for r, c, comp in assignments:
        area = int(comp.shape[0])
        ray_len = 2 * area
        assert c + ray_len <= width, "ray would exceed the right boundary"
        for offset in range(ray_len):
            out[r, c + offset] = RED if offset % 2 == 0 else GREY

    return out


def _choose_marker_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(1, height - 1))
    for _ in range(200):
        rows = sorted(random.sample(candidates, count))
        if all(rows[i + 1] - rows[i] >= MIN_MARKER_GAP for i in range(len(rows) - 1)):
            return rows
    return None


def _grow_component(
    area: int,
    anchor: tuple[int, int],
    allowed_rows: range,
    allowed_cols: range,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    allowed = {
        (r, c)
        for r in allowed_rows
        for c in allowed_cols
        if (r, c) not in occupied
    }
    if anchor not in allowed:
        return None

    cells = {anchor}
    frontier = {
        nbr
        for nbr in _neighbors4(*anchor)
        if nbr in allowed and nbr not in cells
    }
    while len(cells) < area:
        if not frontier:
            return None
        choice = random.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for nbr in _neighbors4(*choice):
            if nbr in allowed and nbr not in cells:
                frontier.add(nbr)
    return cells


def _build_input() -> np.ndarray | None:
    marker_count = random.choice([3, 4])
    areas = random.sample([2, 3, 4, 5], marker_count)
    random.shuffle(areas)

    max_area = max(areas)
    min_height = max(12, marker_count * MIN_MARKER_GAP + 2)
    height = random.randint(min_height, 24)
    rows = _choose_marker_rows(height, marker_count)
    if rows is None:
        return None

    min_width = max(18, 10 + max_area * 2)
    width = random.randint(min_width, 28)
    marker_col = random.randint(4, min(11, width - 2 * max_area))

    grid = np.full((height, width), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for row, area in zip(rows, areas):
        grid[row, marker_col] = RED
        occupied.add((row, marker_col))

        row_pad = random.choice([1, 1, 2])
        row_lo = max(0, row - row_pad)
        row_hi = min(height - 1, row + row_pad)
        if not (row_lo <= row <= row_hi):
            return None

        if marker_col < 2:
            return None
        anchor_col = random.randint(0, marker_col - 1)
        component = _grow_component(
            area=area,
            anchor=(row, anchor_col),
            allowed_rows=range(row_lo, row_hi + 1),
            allowed_cols=range(0, marker_col),
            occupied=occupied,
        )
        if component is None:
            return None

        for rr, cc in component:
            grid[rr, cc] = BLUE
        occupied.update(component)

    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        parsed_grid, marker_col, assignments = _parse_instance(grid)
    except AssertionError:
        return False

    if not (3 <= len(assignments) <= 4):
        return False

    width = parsed_grid.shape[1]
    max_area = max(int(comp.shape[0]) for _r, _c, comp in assignments)
    if width - marker_col < 2 * max_area:
        return False

    out = main(parsed_grid)
    ray_mask = np.zeros_like(parsed_grid, dtype=bool)

    for r, c, comp in assignments:
        area = int(comp.shape[0])
        ray_len = 2 * area
        if not np.any(comp[:, 0] == r):
            return False
        expected = np.array([RED if i % 2 == 0 else GREY for i in range(ray_len)], dtype=int)
        if not np.array_equal(out[r, c : c + ray_len], expected):
            return False
        if not np.all(out[r, c + ray_len :] == BLACK):
            return False
        ray_mask[r, c : c + ray_len] = True

    if not np.array_equal(out[parsed_grid == BLUE], parsed_grid[parsed_grid == BLUE]):
        return False
    if not np.array_equal(out[~ray_mask], parsed_grid[~ray_mask]):
        return False
    if np.array_equal(out, parsed_grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None and _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
