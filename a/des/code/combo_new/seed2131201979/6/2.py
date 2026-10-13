from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


AZURE = 8
GREY = 5
RED = 2
GREEN = 3

PATH_COLOR_POOL = [0, 1, 4, 6, 7, 9]
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GEN_RETRIES = 400
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 30)
LOOP_COUNT_RANGE = (1, 4)
PLACEMENT_PADDING = 2


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _legend_positions(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, _ = shape
    return (h - 2, 0), (h - 1, 0)


def _repair_breaks(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    legend_top, legend_bottom = _legend_positions(g.shape)
    ignore = {legend_top, legend_bottom}

    for _ in range(8):
        changed = False
        grey_cells = np.argwhere(g == GREY)
        if grey_cells.size == 0:
            break
        for r, c in grey_cells:
            cell = (int(r), int(c))
            if cell in ignore:
                continue
            neighbor_colors = []
            for nr, nc in _neighbors4(cell[0], cell[1], *g.shape):
                if (nr, nc) in ignore:
                    continue
                col = int(g[nr, nc])
                if col not in (AZURE, GREY):
                    neighbor_colors.append(col)
            counts = Counter(neighbor_colors)
            candidates = [col for col, cnt in counts.items() if cnt >= 2]
            if len(candidates) == 1:
                g[cell] = int(candidates[0])
                changed = True
        if not changed:
            break

    unresolved = np.argwhere(g == GREY)
    for r, c in unresolved:
        if (int(r), int(c)) not in ignore:
            raise AssertionError("all non-legend grey break cells must be repairable")
    return g


def _find_loop_components(
    repaired: np.ndarray,
) -> list[dict[str, object]]:
    h, w = repaired.shape
    legend_top, legend_bottom = _legend_positions(repaired.shape)
    fg = (repaired != AZURE) & (repaired != GREY)
    fg[legend_top] = False
    fg[legend_bottom] = False
    seen = np.zeros_like(fg, dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            color = int(repaired[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if fg[nr, nc] and not seen[nr, nc] and int(repaired[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _component_orientation(
    coords: list[tuple[int, int]],
    anchors: list[tuple[int, int]],
    shape: tuple[int, int],
) -> str:
    h, w = shape
    coord_set = set(coords)
    for r, c in coords:
        deg = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c, h, w))
        if deg != 2:
            raise AssertionError("each repaired path cell must have degree 2")

    start = min(anchors if anchors else coords)
    start_neighbors = sorted((nr, nc) for nr, nc in _neighbors4(start[0], start[1], h, w) if (nr, nc) in coord_set)
    if len(start_neighbors) != 2:
        raise AssertionError("path tracing requires exactly two neighbors at start")

    cycle = [start]
    prev = start
    curr = start_neighbors[0]
    visited = {start}
    limit = len(coords) + 2
    while curr != start and len(cycle) <= limit:
        if curr in visited:
            raise AssertionError("loop tracing must not revisit non-start cells")
        cycle.append(curr)
        visited.add(curr)
        nbs = [(nr, nc) for nr, nc in _neighbors4(curr[0], curr[1], h, w) if (nr, nc) in coord_set]
        if len(nbs) != 2:
            raise AssertionError("every traced path cell must have degree 2")
        nxt = nbs[0] if nbs[1] == prev else nbs[1]
        prev, curr = curr, nxt

    if curr != start or len(cycle) != len(coords):
        raise AssertionError("tracing must return to the start after one full cycle")

    left_turns = 0
    right_turns = 0
    n = len(cycle)
    for i in range(n):
        pr, pc = cycle[i - 1]
        cr, cc = cycle[i]
        nr, nc = cycle[(i + 1) % n]
        v1r, v1c = cr - pr, cc - pc
        v2r, v2c = nr - cr, nc - cc
        cross = v1r * v2c - v1c * v2r
        if cross > 0:
            left_turns += 1
        elif cross < 0:
            right_turns += 1
    return "ccw" if left_turns > right_turns else "cw"


def _solve_full(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    if g.shape[0] < 3 or g.shape[1] < 2:
        raise AssertionError("grid is too small for this task")

    legend_top, legend_bottom = _legend_positions(g.shape)
    cw_fill = int(g[legend_top])
    ccw_fill = int(g[legend_bottom])

    repaired = _repair_breaks(g)
    repaired[legend_top] = AZURE
    repaired[legend_bottom] = AZURE

    loop_components = _find_loop_components(repaired)
    loop_id_grid = np.full(g.shape, -1, dtype=int)
    fill_by_loop_id: dict[int, int] = {}

    for loop_id, comp in enumerate(loop_components):
        coords = comp["coords"]
        anchors = [(r, c) for r, c in coords if int(g[r, c]) == GREY]
        orientation = _component_orientation(coords, anchors, g.shape)
        fill_by_loop_id[loop_id] = cw_fill if orientation == "cw" else ccw_fill
        for r, c in coords:
            loop_id_grid[r, c] = loop_id

    out = np.array(repaired, copy=True)
    bg_components = _components(out == AZURE)
    h, w = out.shape
    for comp in bg_components:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adjacent_loops = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                loop_id = int(loop_id_grid[nr, nc])
                if loop_id >= 0:
                    adjacent_loops.add(loop_id)
        if len(adjacent_loops) != 1:
            raise AssertionError("each enclosed region must belong to exactly one loop")
        fill_color = fill_by_loop_id[int(next(iter(adjacent_loops)))]
        for r, c in comp:
            out[r, c] = fill_color
    return out


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != AZURE)
    if coords.size == 0:
        return np.full((1, 1), AZURE, dtype=int)
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], copy=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair broken ring paths, fill their interiors by traced rotation direction, and crop to the active area.

    Input structure:
    - The background is azure.
    - Several 1-cell-wide 4-connected colored ring paths appear on the grid; each ring has 1 or 2 grey break cells replacing original path cells.
    - The bottom-left corner contains a 2x1 legend: the upper cell gives the clockwise fill color and the lower cell gives the counterclockwise fill color.

    Rule:
    - Restore every non-legend grey break cell whose two same-colored 4-neighbors identify its original ring color.
    - For each repaired ring, trace the cycle from its lexicographically first repaired break cell, taking the lexicographically smaller same-colored neighbor first, and compare left turns with right turns.
    - Fill the ring interior with the legend's top color when right turns dominate (clockwise) and with the legend's bottom color when left turns dominate (counterclockwise), keeping the ring color itself unchanged.
    - Remove the legend from consideration and crop the result to the minimum bounding box containing all repaired rings and filled interiors.

    Key insight:
    - The grey breaks are local repair clues, and once they are restored they also provide a canonical starting point that makes the loop's rotation direction deterministically traceable.
    """
    return _crop_non_bg(_solve_full(np.asarray(grid, dtype=int)))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _cells_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        seg = _segment_cells(a, b)
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    if cells and cells[-1] == cells[0]:
        cells.pop()
    return cells


def _is_simple_cycle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.shape[0] < 8:
        return False
    comps = _components(mask)
    if len(comps) != 1:
        return False
    coord_set = {tuple(map(int, rc)) for rc in coords}
    h, w = mask.shape
    for r, c in coord_set:
        deg = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c, h, w))
        if deg != 2:
            return False
    return True


def _rectangle_cells(h: int, w: int) -> list[tuple[int, int]]:
    return _cells_from_vertices([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])


def _dent_cells(kind: str, h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    depth = rng.randint(1, max(1, min(3, h - 4 if kind in ("top", "bottom") else w - 4)))
    if kind == "top":
        left = rng.randint(1, w - 4)
        right = rng.randint(left + 2, w - 2)
        verts = [(0, 0), (0, left), (depth, left), (depth, right), (0, right), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    elif kind == "bottom":
        left = rng.randint(1, w - 4)
        right = rng.randint(left + 2, w - 2)
        base = h - 1 - depth
        verts = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, right), (base, right), (base, left), (h - 1, left), (h - 1, 0)]
    elif kind == "left":
        top = rng.randint(1, h - 4)
        bottom = rng.randint(top + 2, h - 2)
        verts = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0), (bottom, 0), (bottom, depth), (top, depth), (top, 0)]
    elif kind == "right":
        top = rng.randint(1, h - 4)
        bottom = rng.randint(top + 2, h - 2)
        base = w - 1 - depth
        verts = [(0, 0), (0, w - 1), (top, w - 1), (top, base), (bottom, base), (bottom, w - 1), (h - 1, w - 1), (h - 1, 0)]
    else:
        raise AssertionError("unknown dent kind")
    return _cells_from_vertices(verts)


def _random_loop_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        shape_kind = rng.choice(["rect", "top", "bottom", "left", "right"])
        if shape_kind == "rect":
            h = rng.randint(5, 10)
            w = rng.randint(5, 11)
            cells = _rectangle_cells(h, w)
        elif shape_kind in ("top", "bottom"):
            h = rng.randint(6, 11)
            w = rng.randint(7, 12)
            cells = _dent_cells(shape_kind, h, w, rng)
        else:
            h = rng.randint(7, 12)
            w = rng.randint(6, 11)
            cells = _dent_cells(shape_kind, h, w, rng)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _is_simple_cycle(mask):
            return mask
    raise RuntimeError("failed to sample a simple loop mask")


def _loop_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(nr, nc) for nr, nc in _neighbors4(cell[0], cell[1], h, w) if bool(mask[nr, nc])]


def _repairable_gap_set(mask: np.ndarray, gaps: tuple[tuple[int, int], ...]) -> bool:
    gap_set = set(gaps)
    for cell in gaps:
        neighbors = _loop_neighbors(mask, cell)
        if len(neighbors) != 2:
            return False
        if any(nb in gap_set for nb in neighbors):
            return False
    return True


def _orientation_for_gap_set(mask: np.ndarray, gaps: tuple[tuple[int, int], ...]) -> str:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    return _component_orientation(coords, list(gaps), mask.shape)


def _select_gaps(mask: np.ndarray, desired_orientation: str, preferred_gap_count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    counts_to_try = [preferred_gap_count, 3 - preferred_gap_count]
    for gap_count in counts_to_try:
        candidates = []
        if gap_count == 1:
            iter_sets = ((cell,) for cell in cells)
        else:
            iter_sets = combinations(cells, 2)
        for gaps in iter_sets:
            if not _repairable_gap_set(mask, gaps):
                continue
            candidates.append(gaps)
        if candidates:
            return rng.choice(candidates)
    raise RuntimeError("failed to find matching break set")


def _can_place_bbox(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    padding: int,
) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + height >= h or left + width >= w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    occupied[r0:r1, c0:c1] = True


def _render_expected(scene: np.ndarray, loops: list[dict[str, object]]) -> np.ndarray:
    out = np.array(scene, copy=True)
    legend_top, legend_bottom = _legend_positions(out.shape)
    out[legend_top] = AZURE
    out[legend_bottom] = AZURE

    loop_id_grid = np.full(out.shape, -1, dtype=int)
    fill_by_id: dict[int, int] = {}
    for loop_id, loop in enumerate(loops):
        fill_by_id[loop_id] = RED if loop["orientation"] == "cw" else GREEN
        for r, c in loop["global_cells"]:
            loop_id_grid[r, c] = loop_id

    for comp in _components(out == AZURE):
        if any(r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1 for r, c in comp):
            continue
        adjacent_loops = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, *out.shape):
                loop_id = int(loop_id_grid[nr, nc])
                if loop_id >= 0:
                    adjacent_loops.add(loop_id)
        if len(adjacent_loops) != 1:
            raise AssertionError("generated loops must enclose disjoint interiors")
        fill = fill_by_id[int(next(iter(adjacent_loops)))]
        for r, c in comp:
            out[r, c] = fill
    return _crop_non_bg(out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), AZURE, dtype=int)
    grid[h - 2, 0] = RED
    grid[h - 1, 0] = GREEN

    occupied = np.zeros((h, w), dtype=bool)
    _mark_bbox(occupied, h - 2, 0, 2, 1, padding=2)

    n_loops = rng.randint(*LOOP_COUNT_RANGE)
    colors = rng.sample(PATH_COLOR_POOL, k=n_loops)
    loops: list[dict[str, object]] = []

    for idx in range(n_loops):
        mask = _random_loop_mask(rng)
        ph, pw = mask.shape
        preferred_gap_count = rng.choice([1, 2])
        try:
            gaps = _select_gaps(mask, "", preferred_gap_count, rng)
            desired_orientation = _orientation_for_gap_set(mask, gaps)
        except RuntimeError:
            return None

        placements = []
        for top in range(1, h - ph):
            for left in range(1, w - pw):
                if _can_place_bbox(occupied, top, left, ph, pw, padding=PLACEMENT_PADDING):
                    placements.append((top, left))
        if not placements:
            return None
        top, left = rng.choice(placements)
        _mark_bbox(occupied, top, left, ph, pw, padding=PLACEMENT_PADDING)

        global_cells = []
        for r, c in np.argwhere(mask):
            rr, cc = int(r) + top, int(c) + left
            global_cells.append((rr, cc))
            grid[rr, cc] = colors[idx]
        for gr, gc in gaps:
            grid[top + gr, left + gc] = GREY

        loops.append(
            {
                "orientation": desired_orientation,
                "global_cells": global_cells,
            }
        )

    expected = _render_expected(_repair_breaks(grid), loops)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30))
        grid = _gen_attempt(rng)
        if grid is not None and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
