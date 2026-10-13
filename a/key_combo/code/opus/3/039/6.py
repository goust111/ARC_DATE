import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _paint_rays(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    ray_mask = np.zeros((h, w), dtype=bool)

    if w < 3:
        return out, ray_mask

    marker_rows = [r for r in range(h) if int(g[r, 1]) == ORANGE and int(g[r, 0]) == GREEN]
    for r in marker_rows:
        stop = w - 1
        blocked = False
        for c in range(2, w):
            if int(g[r, c]) == GREY:
                stop = c
                blocked = True
                break
        end_c = stop - 1
        if end_c < 1:
            continue
        color = ORANGE if blocked else GREEN
        out[r, 1 : end_c + 1] = color
        ray_mask[r, 1 : end_c + 1] = True
    return out, ray_mask


def _region_fill_color(ray_colors: set[int]) -> int:
    if ray_colors == {GREEN}:
        return BLUE
    if ray_colors == {ORANGE, GREEN}:
        return RED
    return YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend edge markers into wall-stopped or wall-free rays, then flood-fill each partition by adjacent ray types.

    Input structure:
    - Black background with full green bars on the left and right borders.
    - Orange marker pixels appear only in the column just inside the left green bar.
    - Inside the grid are grey one-cell-wide vertical wall segments that do not touch the side bars.

    Rule:
    - Each orange marker grows a horizontal ray to the right until the first grey wall in its row or, if none exists, until the cell just before the right green bar.
    - A wall-stopped ray stays orange; a ray that reaches the far side is recolored green.
    - After all rays are drawn, flood-fill every remaining black 4-connected region: regions adjacent only to orange rays (or to no rays) become yellow, regions adjacent only to green rays become blue, and regions adjacent to both ray types become red.

    Key insight:
    - The grey walls simultaneously stop rays and split the background into regions, while the ray color after the stop/no-stop test determines the color used to fill each region.
    """
    ray_grid, ray_mask = _paint_rays(grid)
    out = ray_grid.copy()

    for comp in _components(out == BLACK):
        touching_rays: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and bool(ray_mask[nr, nc]):
                    touching_rays.add(int(out[nr, nc]))
        fill = _region_fill_color(touching_rays)
        for r, c in comp:
            out[r, c] = fill

    return out


_GEN_H_RANGE = (9, 12)
_GEN_W_RANGE = (11, 15)
_GEN_RETRIES = 400
_GEN_EXTRA_WALL_ATTEMPTS = 80


def _sample_marker_rows(h: int, k: int) -> tuple[list[int], int] | None:
    interior = list(range(1, h - 1))
    for _ in range(100):
        rows = sorted(random.sample(interior, k))
        anchors = [i for i in range(k - 2) if rows[i + 1] - rows[i] >= 2 and rows[i + 2] - rows[i + 1] >= 2]
        if anchors:
            return rows, random.choice(anchors)
    return None


def _segment_cells(r0: int, length: int, c: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c) for dr in range(length)]


def _can_place_segment(
    h: int,
    w: int,
    cells: list[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    marker_rows: set[int],
    *,
    allow_marker_row: int | None = None,
) -> bool:
    for r, c in cells:
        if not (1 <= r < h - 1 and 2 <= c < w - 1):
            return False
        if r in marker_rows and r != allow_marker_row:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in wall_cells:
                    return False
    return True


def _add_wall(grid: np.ndarray, wall_cells: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = GREY
        wall_cells.add((r, c))


def _first_wall_col(grid: np.ndarray, row: int) -> int | None:
    for c in range(2, grid.shape[1] - 1):
        if int(grid[row, c]) == GREY:
            return c
    return None


def _pick_anchor_columns(w: int) -> tuple[int, int]:
    c1 = random.randint(3, w - 6)
    c2 = random.randint(c1 + 2, w - 3)
    return c1, c2


def _add_anchor_walls(
    grid: np.ndarray,
    marker_rows: set[int],
    anchor_rows: list[int],
    pattern: str,
) -> tuple[set[int], set[tuple[int, int]]] | None:
    h, w = grid.shape
    wall_cells: set[tuple[int, int]] = set()
    blocked_rows: set[int] = set()
    c1, c2 = _pick_anchor_columns(w)

    if pattern == "BBG":
        r1, r2, r3 = anchor_rows
        blocked_rows = {r1, r2}
        lengths = [
            random.randint(2, min(4, r2 - r1)),
            random.randint(2, min(4, r3 - r2)),
        ]
        segments = [
            _segment_cells(r1, lengths[0], c1),
            _segment_cells(r2, lengths[1], c2),
        ]
    else:
        r1, r2, r3 = anchor_rows
        blocked_rows = {r2, r3}
        len1 = random.randint(2, min(4, r2 - r1))
        len2 = random.randint(2, min(4, r3 - r2))
        segments = [
            _segment_cells(r2 - len1 + 1, len1, c1),
            _segment_cells(r3 - len2 + 1, len2, c2),
        ]

    for row, cells in zip(sorted(blocked_rows), segments):
        if not _can_place_segment(h, w, cells, wall_cells, marker_rows, allow_marker_row=row):
            return None
        _add_wall(grid, wall_cells, cells)
    return blocked_rows, wall_cells


def _add_extra_walls(grid: np.ndarray, wall_cells: set[tuple[int, int]], marker_rows: set[int], count: int) -> None:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _ in range(_GEN_EXTRA_WALL_ATTEMPTS):
            length = random.randint(2, 4)
            r0 = random.randint(1, h - length - 1)
            c = random.randint(2, w - 2)
            cells = _segment_cells(r0, length, c)
            if not _can_place_segment(h, w, cells, wall_cells, marker_rows):
                continue
            _add_wall(grid, wall_cells, cells)
            placed = True
            break
        if not placed:
            return


def _status_ok(grid: np.ndarray, blocked_rows: set[int], marker_rows: list[int]) -> bool:
    for row in marker_rows:
        has_wall = _first_wall_col(grid, row) is not None
        if (row in blocked_rows) != has_wall:
            return False
    return True


def _fill_colors_present(grid: np.ndarray) -> set[int]:
    return set(int(v) for v in np.unique(grid) if int(v) in {BLUE, RED, YELLOW})


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GEN_H_RANGE)
    w = random.randint(*_GEN_W_RANGE)
    k_lo = max(4, math.ceil(0.4 * h))
    k_hi = min(6, math.floor(0.6 * h), h - 2)
    if k_lo > k_hi:
        return None

    sample = _sample_marker_rows(h, random.randint(k_lo, k_hi))
    if sample is None:
        return None
    marker_rows, anchor_idx = sample
    anchor_rows = marker_rows[anchor_idx : anchor_idx + 3]
    pattern = random.choice(["BBG", "GBB"])

    grid = np.full((h, w), BLACK, dtype=int)
    grid[:, 0] = GREEN
    grid[:, -1] = GREEN
    for row in marker_rows:
        grid[row, 1] = ORANGE

    anchor_result = _add_anchor_walls(grid, set(marker_rows), anchor_rows, pattern)
    if anchor_result is None:
        return None
    blocked_rows, wall_cells = anchor_result

    extra_walls = random.randint(0, 2)
    _add_extra_walls(grid, wall_cells, set(marker_rows), extra_walls)

    if not _status_ok(grid, blocked_rows, marker_rows):
        return None

    ray_grid, ray_mask = _paint_rays(grid)
    if not (np.any(ray_mask & (ray_grid == ORANGE)) and np.any(ray_mask & (ray_grid == GREEN))):
        return None

    out = main(grid)
    if _fill_colors_present(out) != {BLUE, RED, YELLOW}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid sample")
