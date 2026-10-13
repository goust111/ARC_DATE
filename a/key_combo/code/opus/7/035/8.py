from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 7)

GEN_SIZE_MIN = 11
GEN_SIZE_MAX = 15
GEN_COUNTS = [(1, 2), (2, 1), (2, 2), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_domino_walls(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    comps = _components(grid == WALL)
    row_walls: list[int] = []
    col_walls: list[int] = []
    for comp in comps:
        assert len(comp) == 2, "each grey object must be a two-pixel domino"
        (r0, c0), (r1, c1) = sorted(comp)
        if r0 == r1 and abs(c0 - c1) == 1:
            row_walls.append(int(r0))
        elif c0 == c1 and abs(r0 - r1) == 1:
            col_walls.append(int(c0))
        else:
            raise AssertionError("grey domino must be horizontal or vertical")
    assert row_walls and col_walls, "need at least one horizontal and one vertical domino"
    return sorted(set(row_walls)), sorted(set(col_walls)), len(comps)


def _draw_walls(grid: np.ndarray, row_walls: list[int], col_walls: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for c in col_walls:
        out[:, c] = WALL
    for r in row_walls:
        out[r, :] = WALL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend grey domino rays into walls, then flood-fill each marked chamber.

    Input structure:
    - The background is black.
    - Grey foreground objects are isolated two-pixel dominoes, each horizontal or vertical.
    - Additional singleton colored marker pixels lie away from the rows and columns implied by those dominoes.

    Rule:
    - Extend every vertical grey domino to a full grey column and every horizontal grey domino to a full grey row.
    - Treat the resulting grey rows and columns as barriers that partition the grid into rectangular chambers.
    - For each 4-connected non-grey chamber, if it contains exactly one colored marker pixel, flood-fill the whole chamber with that marker color; leave unmarked chambers black.

    Key insight:
    - The dominoes do not directly encode the output image; they first generate a wall layout, and only those induced walls determine which marker colors can spread through which chambers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_walls, col_walls, _ = _extract_domino_walls(g)
    prefill = _draw_walls(g, row_walls, col_walls)
    out = prefill.copy()

    for comp in _components(prefill != WALL):
        markers = [(r, c) for r, c in comp if prefill[r, c] not in (BG, WALL)]
        assert len(markers) <= 1, "each chamber may contain at most one marker"
        if not markers:
            continue
        color = int(prefill[markers[0]])
        for r, c in comp:
            out[r, c] = color

    return out


def _sample_axis(length: int, count: int) -> list[int] | None:
    candidates = list(range(1, length - 1))
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        bounds = [-1, *picks, length]
        sizes = [bounds[i + 1] - bounds[i] - 1 for i in range(len(bounds) - 1)]
        if min(sizes) >= 2:
            return picks
    return None


def _intervals(length: int, walls: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *walls, length]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def _place_dominoes(
    shape: tuple[int, int], row_walls: list[int], col_walls: list[int]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[tuple[int, int], tuple[int, int]]] = []
    specs = [("h", r) for r in row_walls] + [("v", c) for c in col_walls]
    random.shuffle(specs)

    def can_place(cells: list[tuple[int, int]]) -> bool:
        for cell in cells:
            if cell in occupied:
                return False
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in occupied and (nr, nc) not in cells:
                    return False
        return True

    for orient, index in specs:
        candidates: list[list[tuple[int, int]]] = []
        if orient == "h":
            for c in range(w - 1):
                if c in col_walls or c + 1 in col_walls:
                    continue
                cells = [(index, c), (index, c + 1)]
                candidates.append(cells)
        else:
            for r in range(h - 1):
                if r in row_walls or r + 1 in row_walls:
                    continue
                cells = [(r, index), (r + 1, index)]
                candidates.append(cells)
        random.shuffle(candidates)
        chosen = None
        for cells in candidates:
            if can_place(cells):
                chosen = cells
                break
        if chosen is None:
            raise RuntimeError("failed to place domino")
        placements.append((chosen[0], chosen[1]))
        occupied.update(chosen)
    return placements


def _choose_markers(
    shape: tuple[int, int], row_walls: list[int], col_walls: list[int]
) -> list[tuple[int, int, int]]:
    h, w = shape
    row_regions = _intervals(h, row_walls)
    col_regions = _intervals(w, col_walls)
    regions = [
        (r0, r1, c0, c1)
        for r0, r1 in row_regions
        for c0, c1 in col_regions
        if r0 <= r1 and c0 <= c1
    ]
    region_count = len(regions)
    assert region_count >= 4
    marked_count = random.randint(3, min(len(MARKER_COLORS), region_count - 1))
    chosen_regions = random.sample(regions, marked_count)
    colors = random.sample(list(MARKER_COLORS), marked_count)

    markers: list[tuple[int, int, int]] = []
    for (r0, r1, c0, c1), color in zip(chosen_regions, colors):
        r = random.randint(r0, r1)
        c = random.randint(c0, c1)
        markers.append((r, c, int(color)))
    return markers


def _make_attempt() -> np.ndarray | None:
    h = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    w = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    h_count, v_count = random.choice(GEN_COUNTS)

    row_walls = _sample_axis(h, h_count)
    col_walls = _sample_axis(w, v_count)
    if row_walls is None or col_walls is None:
        return None

    try:
        dominoes = _place_dominoes((h, w), row_walls, col_walls)
    except RuntimeError:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for (r0, c0), (r1, c1) in dominoes:
        grid[r0, c0] = WALL
        grid[r1, c1] = WALL

    for r, c, color in _choose_markers((h, w), row_walls, col_walls):
        grid[r, c] = int(color)

    return grid if _valid_input(grid) else None


def _valid_input(grid: np.ndarray) -> bool:
    try:
        row_walls, col_walls, domino_count = _extract_domino_walls(grid)
    except AssertionError:
        return False

    if not (3 <= domino_count <= 5):
        return False

    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != BG) & (grid != WALL))]
    if len(markers) < 3:
        return False
    if any(color not in MARKER_COLORS for _, _, color in markers):
        return False
    row_set, col_set = set(row_walls), set(col_walls)
    if any(r in row_set or c in col_set for r, c, _ in markers):
        return False

    walled = _draw_walls(grid, row_walls, col_walls)
    region_count = 0
    marked_regions = 0
    unmarked_regions = 0
    for comp in _components(walled != WALL):
        region_count += 1
        comp_markers = [(r, c) for r, c in comp if walled[r, c] not in (BG, WALL)]
        if len(comp_markers) > 1:
            return False
        if comp_markers:
            marked_regions += 1
        else:
            unmarked_regions += 1

    if region_count < 4 or marked_regions < 3 or unmarked_regions < 1:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
