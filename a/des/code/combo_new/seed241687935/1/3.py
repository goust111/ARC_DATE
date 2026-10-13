import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

MARKER_TO_STEP = {
    RED: (0, -1),
    BLUE: (1, 0),
    GREEN: (0, 1),
    YELLOW: (-1, 0),
}
MARKER_COLORS = tuple(MARKER_TO_STEP)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def touches_edge(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def boundary_mask(filled: np.ndarray) -> np.ndarray:
    filled = np.asarray(filled, dtype=bool)
    out = np.zeros_like(filled)
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if not in_bounds(filled.shape, nr, nc) or not filled[nr, nc]:
                out[rr, cc] = True
                break
    return out


def mask_to_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}


def first_boundary_on_ray(
    boundary_cells: set[tuple[int, int]], marker: tuple[int, int], color: int
) -> tuple[int, int] | None:
    r, c = marker
    if color == GREEN:
        cols = [bc for br, bc in boundary_cells if br == r and bc < c]
        return None if not cols else (r, min(cols))
    if color == RED:
        cols = [bc for br, bc in boundary_cells if br == r and bc > c]
        return None if not cols else (r, max(cols))
    if color == BLUE:
        rows = [br for br, bc in boundary_cells if bc == c and br < r]
        return None if not rows else (min(rows), c)
    rows = [br for br, bc in boundary_cells if bc == c and br > r]
    return None if not rows else (max(rows), c)


def segment_from_edge(
    shape: tuple[int, int], hit: tuple[int, int], color: int
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = hit
    if color == GREEN:
        return [(r, cc) for cc in range(0, c + 1)]
    if color == RED:
        return [(r, cc) for cc in range(w - 1, c - 1, -1)]
    if color == BLUE:
        return [(rr, c) for rr in range(0, r + 1)]
    return [(rr, c) for rr in range(h - 1, r - 1, -1)]


def extract_enclosed_regions(grid: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    passable = g != GREY
    regions: list[tuple[tuple[int, int], int]] = []

    for comp in components_mask(passable):
        if touches_edge(comp, g.shape):
            continue
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) in MARKER_COLORS]
        assert len(marker_cells) == 1, "each enclosed region must contain exactly one marker"
        marker = marker_cells[0]
        marker_color = int(g[marker])
        for r, c in comp:
            color = int(g[r, c])
            assert color in (BLACK, marker_color), "only black cells and the single marker may appear inside a region"
        regions.append((marker, marker_color))

    assert regions, "expected at least one enclosed region"
    regions.sort()
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire edge-origin rays toward grey enclosures using each interior marker color as the direction code.

    Input structure:
    - The background is black.
    - One or more grey 4-connected closed outline objects enclose black interior regions.
    - Each enclosed region contains exactly one colored marker pixel: red, blue,
      green, or yellow.
    - Orange pixels may appear outside the enclosures as external obstacles.

    Rule:
    - Read each enclosed marker color as a direction: red means from the right
      edge leftward, blue from the top edge downward, green from the left edge
      rightward, and yellow from the bottom edge upward.
    - Launch a ray from the corresponding grid edge along the marker's row or
      column.
    - Recolor every black pixel on that ray to the marker color until the ray
      first hits a grey outline pixel or an orange obstacle.
    - Do not cover the blocking pixel; instead recolor that blocking grey/orange
      pixel to maroon.

    Key insight:
    - The colored pixel inside each closed region is not the ray source: it only
      encodes which edge to use and which row or column the edge-origin ray must
      follow, and the struck blocker changes to maroon at the stopping point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for marker, color in extract_enclosed_regions(g):
        start = marker[0], 0
        if color == RED:
            start = marker[0], g.shape[1] - 1
        elif color == BLUE:
            start = 0, marker[1]
        elif color == YELLOW:
            start = g.shape[0] - 1, marker[1]

        dr, dc = MARKER_TO_STEP[color]
        r, c = start
        hit_blocker = False
        while in_bounds(g.shape, r, c):
            cell = int(g[r, c])
            if cell == BLACK:
                current = int(out[r, c])
                assert current in (BLACK, color), "different rays may not conflict on black cells"
                out[r, c] = color
                r += dr
                c += dc
                continue

            assert cell in (GREY, ORANGE), "rays must stop on grey or orange blockers"
            current = int(out[r, c])
            assert current in (GREY, ORANGE, MAROON), "blocking cells may only be grey, orange, or already maroon"
            out[r, c] = MAROON
            hit_blocker = True
            break

        assert hit_blocker, "every ray must hit a blocker before leaving the grid"

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 16, 28
REGION_COUNT_RANGE = (1, 4)
SHAPE_ATTEMPTS = 40
PLACE_ATTEMPTS = 120
GEN_ATTEMPTS = 400

RECT_SIZE_RANGE = (5, 8)
TAB_SIZE_RANGE = (5, 8)
TAB_DEPTH_RANGE = (3, 5)
EXTRA_ORANGE_RANGE = (0, 5)


def sample_filled_shape() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_ATTEMPTS):
        if random.random() < 0.5:
            h = random.randint(*RECT_SIZE_RANGE)
            w = random.randint(*RECT_SIZE_RANGE)
            filled = np.ones((h, w), dtype=bool)
        else:
            base_h = random.randint(*TAB_SIZE_RANGE)
            base_w = random.randint(*TAB_SIZE_RANGE)
            depth = random.randint(*TAB_DEPTH_RANGE)
            side = random.choice(("left", "right", "top", "bottom"))
            if side in ("left", "right"):
                span = random.randint(3, base_h)
                start = random.randint(0, base_h - span)
                filled = np.zeros((base_h, base_w + depth), dtype=bool)
                if side == "left":
                    filled[:, depth:] = True
                    filled[start : start + span, : depth + 1] = True
                else:
                    filled[:, :base_w] = True
                    filled[start : start + span, base_w - 1 :] = True
            else:
                span = random.randint(3, base_w)
                start = random.randint(0, base_w - span)
                filled = np.zeros((base_h + depth, base_w), dtype=bool)
                if side == "top":
                    filled[depth:, :] = True
                    filled[: depth + 1, start : start + span] = True
                else:
                    filled[:base_h, :] = True
                    filled[base_h - 1 :, start : start + span] = True

        boundary = boundary_mask(filled)
        interior = filled & ~boundary
        if np.count_nonzero(interior) < 3:
            continue
        if len(components_mask(boundary)) != 1:
            continue
        if len(components_mask(interior)) != 1:
            continue
        return boundary, interior
    return None


def random_origin(
    grid_shape: tuple[int, int], local_shape: tuple[int, int], color: int
) -> tuple[int, int] | None:
    h, w = grid_shape
    sh, sw = local_shape
    row_lo, row_hi = 1, h - sh - 1
    col_lo, col_hi = 1, w - sw - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    pref_row_lo, pref_row_hi = row_lo, row_hi
    pref_col_lo, pref_col_hi = col_lo, col_hi
    if color == GREEN:
        pref_col_lo = max(col_lo, w // 2 - sw // 2)
    elif color == RED:
        pref_col_hi = min(col_hi, max(col_lo, w // 2 - 1))
    elif color == BLUE:
        pref_row_lo = max(row_lo, h // 2 - sh // 2)
    else:
        pref_row_hi = min(row_hi, max(row_lo, h // 2 - 1))

    if pref_row_lo <= pref_row_hi and pref_col_lo <= pref_col_hi and random.random() < 0.8:
        return random.randint(pref_row_lo, pref_row_hi), random.randint(pref_col_lo, pref_col_hi)
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def choose_marker(
    interior: np.ndarray,
    r0: int,
    c0: int,
    boundary_cells: set[tuple[int, int]],
    color: int,
) -> tuple[int, int] | None:
    coords = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(interior)]
    random.shuffle(coords)
    for marker in coords:
        if first_boundary_on_ray(boundary_cells, marker, color) is not None:
            return marker
    return None


def place_region(
    grid_shape: tuple[int, int],
    occupied_nonblack: set[tuple[int, int]],
    occupied_filled: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    color: int,
    use_orange: bool,
) -> dict[str, object] | None:
    for _ in range(PLACE_ATTEMPTS):
        parts = sample_filled_shape()
        if parts is None:
            continue
        boundary, interior = parts
        origin = random_origin(grid_shape, boundary.shape, color)
        if origin is None:
            continue
        r0, c0 = origin

        filled_cells = mask_to_cells(boundary | interior, r0, c0)
        boundary_cells = mask_to_cells(boundary, r0, c0)
        if filled_cells & occupied_filled:
            continue
        if filled_cells & occupied_nonblack:
            continue
        if filled_cells & reserved_paths:
            continue

        marker = choose_marker(interior, r0, c0, boundary_cells, color)
        if marker is None:
            continue

        hit = first_boundary_on_ray(boundary_cells, marker, color)
        if hit is None:
            continue
        full_segment = segment_from_edge(grid_shape, hit, color)
        if use_orange and len(full_segment) < 3:
            continue
        if not use_orange and len(full_segment) < 2:
            continue

        seg_wo_hit = set(full_segment[:-1])
        if seg_wo_hit & occupied_nonblack:
            continue
        if seg_wo_hit & occupied_filled:
            continue
        if set(full_segment) & reserved_paths:
            continue

        orange_cell = None
        if use_orange:
            orange_cell = random.choice(full_segment[1:-1])

        candidate_nonblack = set(boundary_cells)
        candidate_nonblack.add(marker)
        if orange_cell is not None:
            candidate_nonblack.add(orange_cell)
        if candidate_nonblack & occupied_nonblack:
            continue

        return {
            "boundary_cells": boundary_cells,
            "filled_cells": filled_cells,
            "marker": marker,
            "marker_color": color,
            "orange_cell": orange_cell,
            "reserved_path": set(full_segment),
        }
    return None


def render_input(regions: list[dict[str, object]], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(grid_shape, BLACK, dtype=int)
    for region in regions:
        for r, c in region["boundary_cells"]:
            grid[r, c] = GREY
        mr, mc = region["marker"]
        grid[mr, mc] = int(region["marker_color"])
        orange_cell = region["orange_cell"]
        if orange_cell is not None:
            grid[orange_cell] = ORANGE
    return grid


def add_extra_oranges(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> None:
    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == BLACK and (r, c) not in forbidden
    ]
    random.shuffle(candidates)
    count = random.randint(*EXTRA_ORANGE_RANGE)
    for r, c in candidates[:count]:
        grid[r, c] = ORANGE


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    h = random.randint(max(GRID_MIN, 11 + 2 * region_count), GRID_MAX)
    w = random.randint(max(GRID_MIN, 11 + 2 * region_count), GRID_MAX)
    directions = random.sample(list(MARKER_COLORS), region_count)

    occupied_nonblack: set[tuple[int, int]] = set()
    occupied_filled: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    regions: list[dict[str, object]] = []

    order = directions[:]
    random.shuffle(order)
    for color in order:
        region = place_region(
            (h, w),
            occupied_nonblack,
            occupied_filled,
            reserved_paths,
            color,
            use_orange=bool(random.getrandbits(1)),
        )
        if region is None:
            return None
        regions.append(region)
        occupied_nonblack |= set(region["boundary_cells"])
        occupied_nonblack.add(region["marker"])
        if region["orange_cell"] is not None:
            occupied_nonblack.add(region["orange_cell"])
        occupied_filled |= set(region["filled_cells"])
        reserved_paths |= set(region["reserved_path"])

    inp = render_input(regions, (h, w))
    add_extra_oranges(inp, forbidden=occupied_filled | reserved_paths)

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if len(extract_enclosed_regions(inp)) != region_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
