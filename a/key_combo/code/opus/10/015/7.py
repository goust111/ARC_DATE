from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 400
GRID_SIZE_RANGE = (15, 18)
DIAMOND_RADIUS_RANGE = (5, 7)

SPRITE_SHAPES = {
    "h2": ((0, 0), (0, 1)),
    "v2": ((0, 0), (1, 0)),
    "h3": ((0, -1), (0, 0), (0, 1)),
    "v3": ((-1, 0), (0, 0), (1, 0)),
    "h4": ((0, -1), (0, 0), (0, 1), (0, 2)),
    "v4": ((-1, 0), (0, 0), (1, 0), (2, 0)),
    "l3_se": ((0, 0), (1, 0), (1, 1)),
    "l3_sw": ((0, 0), (1, 0), (1, -1)),
    "l3_ne": ((0, 0), (-1, 0), (-1, 1)),
    "l3_nw": ((0, 0), (-1, 0), (-1, -1)),
}

SPRITE_RECIPES = (
    (((0, 0), (4, 4)), (("h2", "h2"), ("h2", "v2"), ("h2", "h3"), ("h2", "v3"), ("v2", "h2"), ("v2", "v2"), ("v2", "h3"), ("v2", "v3"), ("h2", "l3_se"), ("h2", "l3_sw"), ("h2", "l3_ne"), ("h2", "l3_nw"), ("h2", "h4"), ("h2", "v4"))),
    (((0, 0), (4, 4), (0, 4)), (("h2", "l3_sw", "h2"), ("h2", "l3_sw", "v2"), ("h2", "l3_sw", "h3"), ("h2", "l3_sw", "v3"), ("h2", "l3_sw", "l3_se"), ("h2", "l3_sw", "h4"), ("h2", "l3_sw", "v4"), ("h2", "l3_nw", "h2"))),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return (r0 + r1) // 2, (c0 + c1) // 2


def _bresenham_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return points


def _draw_polyline(grid: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        for r, c in _bresenham_points(r0, c0, r1, c1):
            grid[r, c] = color


def _scanline_fill(base: np.ndarray, working: np.ndarray) -> np.ndarray:
    out = working.copy()
    h, w = out.shape
    for r in range(h):
        cols = np.flatnonzero(base[r] == GREEN)
        for left, right in zip(cols, cols[1:]):
            if right - left <= 1:
                continue
            c = int(left) + 1
            stop = int(right)
            while c < stop:
                while c < stop and out[r, c] != BLACK:
                    c += 1
                start = c
                while c < stop and out[r, c] == BLACK:
                    c += 1
                if start < c:
                    out[r, start:c] = RED
    return out


def _transform(grid: np.ndarray, *, use_dilation: bool, use_connections: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprites = _components(g == GREY)

    if use_dilation:
        new_grey: set[tuple[int, int]] = set()
        for coords in sprites:
            for r_raw, c_raw in coords:
                r = int(r_raw)
                c = int(c_raw)
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLACK:
                        new_grey.add((nr, nc))
        for r, c in new_grey:
            out[r, c] = GREY

    if use_connections and len(sprites) >= 2:
        ordered = sorted(sprites, key=lambda coords: (_bbox_center(coords)[1], _bbox_center(coords)[0]))
        centers = [_bbox_center(coords) for coords in ordered]
        for (r0, c0), (r1, c1) in zip(centers, centers[1:]):
            for r, c in _bresenham_points(r0, c0, r1, c1):
                if out[r, c] == BLACK:
                    out[r, c] = YELLOW

    return _scanline_fill(g, out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dilate the gray blocker sprites, connect adjacent sprite centers with yellow diagonals, then scan-fill only the unblocked parts of each green-bounded row span.

    Input structure
    - A black-background grid contains a green diagonal outline whose rows expose horizontal spans between consecutive outline pixels.
    - Inside the outline sit several small gray 4-connected blocker sprites.
    - Gray sprites are disjoint, and no yellow or red pixels appear in the input.

    Rule
    - Dilate each gray sprite by one 4-neighbor layer into black cells only, never overwriting green outline pixels or existing gray pixels.
    - Sort the original gray sprites by bounding-box center from left to right, then connect each adjacent pair of centers with a yellow Bresenham line drawn only on black cells.
    - For every row, treat each interval strictly between consecutive green pixels as a potential scanline span; gray and yellow pixels split that span into subspans.
    - Fill only the black subspans with red, while preserving all green, gray, and yellow pixels.

    Key insight
    - The result is determined by the cooperative barrier: dilation makes each sprite wide enough to matter locally, and the center-to-center diagonal links merge those local blockers into a stronger scanline stop set.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = set(map(int, np.unique(g)))
    assert colors.issubset({BLACK, GREEN, GREY})
    assert GREEN in colors and GREY in colors
    return _transform(g, use_dilation=True, use_connections=True)


def _diamond_outline(shape: tuple[int, int], cr: int, cc: int, radius: int) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    top = (cr - radius, cc)
    left = (cr, cc - radius)
    bottom = (cr + radius, cc)
    right = (cr, cc + radius)
    _draw_polyline(grid, [top, left, bottom], GREEN)
    _draw_polyline(grid, [top, right, bottom], GREEN)
    return grid


def _inside_outline(shape: tuple[int, int], cr: int, cc: int, radius: int, r: int, c: int) -> bool:
    if not _in_bounds(shape, r, c):
        return False
    return abs(r - cr) + abs(c - cc) < radius


def _place_sprite_cells(center: tuple[int, int], shape_name: str) -> list[tuple[int, int]]:
    r0, c0 = center
    return [(r0 + dr, c0 + dc) for dr, dc in SPRITE_SHAPES[shape_name]]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _outline_interior_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    interior = np.zeros((h, w), dtype=bool)
    for r in range(h):
        cols = np.flatnonzero(grid[r] == GREEN)
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                interior[r, int(left) + 1 : int(right)] = True
    return interior


def _sprites_are_disjoint(sprite_cells: list[list[tuple[int, int]]]) -> bool:
    occupied: dict[tuple[int, int], int] = {}
    for idx, cells in enumerate(sprite_cells):
        for cell in cells:
            if cell in occupied:
                return False
            occupied[cell] = idx
    for idx, cells in enumerate(sprite_cells):
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                other = occupied.get((nr, nc))
                if other is not None and other != idx:
                    return False
    return True


def _has_diagonal_neighbor_pair(centers: list[tuple[int, int]]) -> bool:
    ordered = sorted(centers, key=lambda rc: (rc[1], rc[0]))
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        if 2 <= dc <= 4 and abs(dr - dc) <= 1:
            return True
    return False


def _row_outline_constraint(grid: np.ndarray) -> bool:
    return all(int(np.count_nonzero(grid[r] == GREEN)) <= 2 for r in range(grid.shape[0]))


def _has_split_span(full_out: np.ndarray, base_grid: np.ndarray) -> bool:
    for r in range(full_out.shape[0]):
        cols = np.flatnonzero(base_grid[r] == GREEN)
        for left, right in zip(cols, cols[1:]):
            segment = full_out[r, int(left) + 1 : int(right)]
            red_runs = 0
            in_red = False
            for value in segment:
                if int(value) == RED:
                    if not in_red:
                        red_runs += 1
                        in_red = True
                else:
                    in_red = False
            if red_runs >= 2:
                return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    if not _row_outline_constraint(grid):
        return False

    sprites = _components(grid == GREY)
    if not (2 <= len(sprites) <= 3):
        return False
    if any(not (2 <= int(coords.shape[0]) <= 4) for coords in sprites):
        return False

    interior = _outline_interior_mask(grid)
    grey_coords = np.argwhere(grid == GREY)
    if grey_coords.size == 0:
        return False
    if not all(bool(interior[r, c]) for r, c in grey_coords):
        return False

    centers = [_bbox_center(coords) for coords in sprites]
    if not _has_diagonal_neighbor_pair(centers):
        return False
    for i, (r0, c0) in enumerate(centers):
        for r1, c1 in centers[i + 1 :]:
            if max(abs(r1 - r0), abs(c1 - c0)) > 4:
                return False

    full = main(grid)
    no_dilation = _transform(grid, use_dilation=False, use_connections=True)
    no_connections = _transform(grid, use_dilation=True, use_connections=False)
    bare_outline = np.where(grid == GREEN, GREEN, BLACK)
    outline_only = _scanline_fill(bare_outline, bare_outline.copy())

    if np.array_equal(full, no_dilation):
        return False
    if np.array_equal(full, no_connections):
        return False
    if np.array_equal(full, outline_only):
        return False
    if not _has_split_span(full, grid):
        return False
    return True


def _random_reflection(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, copy=True)


def _candidate_grid() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)

    max_radius = min(DIAMOND_RADIUS_RANGE[1], (h - 3) // 2, (w - 3) // 2)
    if max_radius < DIAMOND_RADIUS_RANGE[0]:
        return None
    radius = random.randint(DIAMOND_RADIUS_RANGE[0], max_radius)

    cr = random.randint(radius + 1, h - radius - 2)
    cc = random.randint(radius + 1, w - radius - 2)
    base_grid = _diamond_outline((h, w), cr, cc, radius)

    anchors = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(anchors)
    recipes = list(SPRITE_RECIPES)
    random.shuffle(recipes)

    for rel_centers, shape_options in recipes:
        shape_choices = list(shape_options)
        random.shuffle(shape_choices)
        for shape_names in shape_choices:
            for anchor_r, anchor_c in anchors:
                centers = [(anchor_r + dr, anchor_c + dc) for dr, dc in rel_centers]
                sprite_cells = [_place_sprite_cells(center, shape_name) for center, shape_name in zip(centers, shape_names)]
                if not all(_inside_outline(base_grid.shape, cr, cc, radius, r, c) for cells in sprite_cells for r, c in cells):
                    continue
                if not _sprites_are_disjoint(sprite_cells):
                    continue

                grid = base_grid.copy()
                for cells in sprite_cells:
                    _paint_cells(grid, cells, GREY)

                if not _semantic_checks(grid):
                    continue

                reflected = _random_reflection(grid)
                if _semantic_checks(reflected):
                    return reflected
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _candidate_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scanline-blocker ARC input")
