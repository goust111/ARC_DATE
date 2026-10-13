from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS = ("left", "right", "up", "down")

# --------------- solver ---------------


def _most_common_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0, c0 in np.argwhere(g == int(color)):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue

        q: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []

        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or int(g[nr, nc]) != int(color):
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))

        br0, bc0, br1, bc1 = _bbox(coords)
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (br0, bc0, br1, bc1),
                "area": len(coords),
            }
        )

    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _line_orientation(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) != 3:
        return None
    rows = sorted(r for r, _ in coords)
    cols = sorted(c for _, c in coords)
    if rows[0] == rows[1] == rows[2] and cols == [cols[0], cols[0] + 1, cols[0] + 2]:
        return "h"
    if cols[0] == cols[1] == cols[2] and rows == [rows[0], rows[0] + 1, rows[0] + 2]:
        return "v"
    return None


def _direction_from_pair(obj: dict, arrow: dict) -> str:
    or0, oc0, or1, oc1 = obj["bbox"]
    ar0, ac0, ar1, ac1 = arrow["bbox"]
    orient = _line_orientation(arrow["coords"])
    assert orient is not None, "arrow must be a 3-pixel straight segment"

    if orient == "h":
        assert not (ar1 < or0 or ar0 > or1), "horizontal arrow must overlap object rows"
        if ac0 > oc1:
            return "right"
        if ac1 < oc0:
            return "left"
    else:
        assert not (ac1 < oc0 or ac0 > oc1), "vertical arrow must overlap object columns"
        if ar0 > or1:
            return "down"
        if ar1 < or0:
            return "up"
    raise AssertionError("arrow must sit on one side of its object")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored object onto the inner wall indicated by its same-color arrow.

    Input structure
    - The background is black.
    - The outer border is a 1-pixel grey wall.
    - Every non-wall foreground color appears exactly twice as 4-connected components:
      one small object and one 3-pixel straight arrow marker of the same color.

    Rule
    - Determine the direction from the arrow's axis and which side of the object it occupies.
    - Compress the object into a 1D silhouette on the inner wall in that direction:
      left/right use occupied rows; up/down use occupied columns.
    - Keep the original objects and arrows unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7

    wall = int(g[0, 0])
    assert np.all(g[0, :] == wall)
    assert np.all(g[-1, :] == wall)
    assert np.all(g[:, 0] == wall)
    assert np.all(g[:, -1] == wall)

    interior = g[1:-1, 1:-1]
    assert interior.size > 0
    bg = _most_common_color(interior)
    assert bg != wall

    colors = [int(color) for color in np.unique(g) if color not in (bg, wall)]
    assert colors, "expected at least one object-arrow pair"

    out = g.copy()
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each color must form exactly one object and one arrow"

        arrow_candidates = [comp for comp in comps if _line_orientation(comp["coords"]) is not None]
        assert len(arrow_candidates) == 1, "each color must have exactly one 3-cell line arrow"
        arrow = arrow_candidates[0]
        obj = comps[0] if comps[1] is arrow else comps[1]

        direction = _direction_from_pair(obj, arrow)
        rows = sorted({r for r, _ in obj["coords"]})
        cols = sorted({c for _, c in obj["coords"]})

        if direction == "left":
            for r in rows:
                assert out[r, 1] in (bg, color)
                out[r, 1] = color
        elif direction == "right":
            for r in rows:
                assert out[r, w - 2] in (bg, color)
                out[r, w - 2] = color
        elif direction == "up":
            for c in cols:
                assert out[1, c] in (bg, color)
                out[1, c] = color
        else:
            for c in cols:
                assert out[h - 2, c] in (bg, color)
                out[h - 2, c] = color

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 5
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_H_MIN = 11
GEN_H_MAX = 17
GEN_W_MIN = 11
GEN_W_MAX = 17
GEN_ATTEMPTS = 300
PAIR_ATTEMPTS = 120
SHAPE_ATTEMPTS = 80


def _normalize_coords(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _random_shape(rng: np.random.Generator) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = int(rng.integers(2, 5))
        box_w = int(rng.integers(2, 5))
        target = int(rng.integers(2, 6))

        start = (int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(0, len(frontier)))]
            cells.add(pick)

        if len(cells) != target:
            continue

        shape = _normalize_coords(cells)
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if target == 3 and (len(rows) == 1 or len(cols) == 1):
            continue
        return shape
    return None


def _shape_size(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _mark_halo(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _draw_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _choose_count(rng: np.random.Generator, h: int, w: int) -> int:
    m = min(h, w)
    if m <= 12:
        max_pairs = 2
    elif m <= 14:
        max_pairs = 3
    else:
        max_pairs = 4
    return int(rng.integers(1, max_pairs + 1))


def _place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: str,
    shape: list[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    shape_h, shape_w = _shape_size(shape)
    row_offsets = sorted({r for r, _ in shape})
    col_offsets = sorted({c for _, c in shape})

    for _ in range(PAIR_ATTEMPTS):
        gap = int(rng.integers(1, 3))

        if direction == "right":
            r_lo, r_hi = 2, h - shape_h - 2
            c_lo, c_hi = 2, w - shape_w - gap - 5
        elif direction == "left":
            r_lo, r_hi = 2, h - shape_h - 2
            c_lo, c_hi = gap + 5, w - shape_w - 2
        elif direction == "down":
            r_lo, r_hi = 2, h - shape_h - gap - 5
            c_lo, c_hi = 2, w - shape_w - 2
        else:
            r_lo, r_hi = gap + 5, h - shape_h - 2
            c_lo, c_hi = 2, w - shape_w - 2

        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        obj_cells = {(r0 + r, c0 + c) for r, c in shape}

        if direction == "right":
            arrow_row = r0 + int(rng.choice(row_offsets))
            start_c = c0 + shape_w - 1 + gap + 1
            arrow_cells = {(arrow_row, start_c + dc) for dc in range(3)}
        elif direction == "left":
            arrow_row = r0 + int(rng.choice(row_offsets))
            start_c = c0 - gap - 3
            arrow_cells = {(arrow_row, start_c + dc) for dc in range(3)}
        elif direction == "down":
            arrow_col = c0 + int(rng.choice(col_offsets))
            start_r = r0 + shape_h - 1 + gap + 1
            arrow_cells = {(start_r + dr, arrow_col) for dr in range(3)}
        else:
            arrow_col = c0 + int(rng.choice(col_offsets))
            start_r = r0 - gap - 3
            arrow_cells = {(start_r + dr, arrow_col) for dr in range(3)}

        all_cells = obj_cells | arrow_cells
        if any(blocked[r, c] for r, c in all_cells):
            continue

        _draw_cells(grid, obj_cells, color)
        _draw_cells(grid, arrow_cells, color)
        _mark_halo(blocked, all_cells)
        return True

    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GEN_H_MIN, GEN_H_MAX + 1))
    w = int(rng.integers(GEN_W_MIN, GEN_W_MAX + 1))
    grid = np.full((h, w), GEN_BG, dtype=int)
    grid[0, :] = GEN_WALL
    grid[-1, :] = GEN_WALL
    grid[:, 0] = GEN_WALL
    grid[:, -1] = GEN_WALL

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    blocked[1, :] = True
    blocked[h - 2, :] = True
    blocked[:, 1] = True
    blocked[:, w - 2] = True

    count = _choose_count(rng, h, w)
    directions = list(rng.permutation(DIRS)[:count])
    colors = list(rng.choice(np.array(GEN_COLORS, dtype=int), size=count, replace=False))
    order = list(rng.permutation(count))

    for idx in order:
        shape = _random_shape(rng)
        if shape is None:
            return None
        if not _place_pair(grid, blocked, int(colors[idx]), directions[idx], shape, rng):
            return None

    assert np.all(grid[1, 1:-1] == GEN_BG)
    assert np.all(grid[h - 2, 1:-1] == GEN_BG)
    assert np.all(grid[1:-1, 1] == GEN_BG)
    assert np.all(grid[1:-1, w - 2] == GEN_BG)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
