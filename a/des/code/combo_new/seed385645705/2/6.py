from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")


def _plus_cells(r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _find_anchor_center(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    grey_cells = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    assert len(grey_cells) == 5, "expected one five-cell grey plus anchor"

    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grey_cells == _plus_cells(r, c):
                centers.append((int(r), int(c)))

    assert len(centers) == 1, "expected a unique grey plus center"
    return centers[0]


def _ray_dirs_for_marker(r: int, c: int, cr: int, cc: int) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
    if c == cc and r != cr:
        return abs(int(r - cr)), ((0, -1), (0, 1))
    if r == cr and c != cc:
        return abs(int(c - cc)), ((-1, 0), (1, 0))
    raise AssertionError("each colored marker must lie on one of the anchor axes")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw equal-length perpendicular rays from every axis-aligned marker around a grey anchor.

    Input structure:
    - The background is black.
    - A unique grey five-cell plus marks the anchor center and its four axis neighbors.
    - Every other non-background cell is a colored singleton marker lying exactly
      above, below, left, or right of the anchor center.
    - Marker rays fit in the grid and do not overlap the anchor or one another.

    Rule:
    - For each marker, measure its axis distance k from the grey anchor center.
    - If the marker is above or below the anchor, draw k cells left and k cells
      right from that marker in the marker color.
    - If the marker is left or right of the anchor, draw k cells up and k cells
      down from that marker in the marker color.
    - Keep the grey anchor and the original marker pixels unchanged.

    Key insight:
    - The marker's relative position supplies two separate instructions: the
      output direction is perpendicular to the anchor axis, while the ray length
      is the marker-anchor distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must be at most 30x30"

    cr, cc = _find_anchor_center(g)
    marker_coords = [
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != GREY))
    ]
    assert marker_coords, "expected at least one colored marker"

    marker_colors = [int(g[r, c]) for r, c in marker_coords]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be distinct"

    out = g.copy()
    for r, c in marker_coords:
        color = int(g[r, c])
        k, ray_dirs = _ray_dirs_for_marker(r, c, cr, cc)
        assert k >= 2, "markers must sit outside the grey plus arms"

        for dr, dc in ray_dirs:
            for step in range(1, k + 1):
                rr = int(r + dr * step)
                cc2 = int(c + dc * step)
                assert 0 <= rr < h and 0 <= cc2 < w, "ray must stay inside the grid"
                assert int(out[rr, cc2]) == BG, "rays must not overwrite existing structure"
                out[rr, cc2] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 300
SIZE_OPTIONS = tuple(range(13, 30, 2))
EXTRA_MARKER_RANGE = (0, 4)


def _marker_position(direction: str, cr: int, cc: int, k: int) -> tuple[int, int]:
    if direction == "up":
        return cr - k, cc
    if direction == "down":
        return cr + k, cc
    if direction == "left":
        return cr, cc - k
    assert direction == "right"
    return cr, cc + k


def _max_distance(h: int, w: int, cr: int, cc: int, direction: str) -> int:
    if direction == "up":
        return min(cr - 1, cc - 1, w - cc - 2)
    if direction == "down":
        return min(h - cr - 2, cc - 1, w - cc - 2)
    if direction == "left":
        return min(cc - 1, cr - 1, h - cr - 2)
    assert direction == "right"
    return min(w - cc - 2, cr - 1, h - cr - 2)


def _footprint(direction: str, cr: int, cc: int, k: int) -> set[tuple[int, int]]:
    r, c = _marker_position(direction, cr, cc, k)
    if direction in {"up", "down"}:
        ray_dirs = ((0, -1), (0, 1))
    else:
        ray_dirs = ((-1, 0), (1, 0))

    cells = {(int(r), int(c))}
    for dr, dc in ray_dirs:
        for step in range(1, k + 1):
            cells.add((int(r + dr * step), int(c + dc * step)))
    return cells


def _candidate_specs(
    h: int,
    w: int,
    cr: int,
    cc: int,
    direction: str,
) -> list[tuple[str, int, tuple[int, int], set[tuple[int, int]]]]:
    specs = []
    for k in range(2, _max_distance(h, w, cr, cc, direction) + 1):
        cells = _footprint(direction, cr, cc, k)
        if all(0 <= r < h and 0 <= c < w for r, c in cells):
            specs.append((direction, int(k), _marker_position(direction, cr, cc, k), cells))
    random.shuffle(specs)
    return specs


def _choose_marker_specs(
    h: int,
    w: int,
    cr: int,
    cc: int,
    requested_dirs: list[str],
) -> list[tuple[str, int, tuple[int, int]]] | None:
    by_dir = {d: _candidate_specs(h, w, cr, cc, d) for d in DIRECTIONS}
    if any(not by_dir[d] for d in DIRECTIONS):
        return None

    anchor = _plus_cells(cr, cc)

    def search(
        i: int,
        occupied: set[tuple[int, int]],
        chosen: list[tuple[str, int, tuple[int, int]]],
    ) -> list[tuple[str, int, tuple[int, int]]] | None:
        if i == len(requested_dirs):
            return chosen

        direction = requested_dirs[i]
        candidates = by_dir[direction][:]
        random.shuffle(candidates)
        for cand_dir, k, pos, cells in candidates:
            if cells & occupied:
                continue
            result = search(
                i + 1,
                occupied | cells,
                chosen + [(cand_dir, int(k), pos)],
            )
            if result is not None:
                return result
        return None

    return search(0, set(anchor), [])


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(SIZE_OPTIONS)
    w = random.choice(SIZE_OPTIONS)
    cr, cc = h // 2, w // 2

    requested_dirs = list(DIRECTIONS)
    extra_count = random.randint(*EXTRA_MARKER_RANGE)
    requested_dirs.extend(random.choice(DIRECTIONS) for _ in range(extra_count))
    random.shuffle(requested_dirs)

    specs = _choose_marker_specs(h, w, cr, cc, requested_dirs)
    if specs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r, c in _plus_cells(cr, cc):
        grid[r, c] = GREY

    colors = random.sample(MARKER_COLORS, len(specs))
    for (_direction, _k, (r, c)), color in zip(specs, colors):
        grid[int(r), int(c)] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid perpendicular-ray marker input")
