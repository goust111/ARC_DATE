from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
INDICATOR = 4
DECORATION = 7
INNER_COLORS = (1, 2, 3, 6, 8, 9)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _flood_from_seeds(passable: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    seen = np.zeros(passable.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        if _in_bounds(passable.shape, r, c) and passable[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _component_count(mask: np.ndarray) -> int:
    remaining = mask.copy()
    total = 0
    for r, c in np.argwhere(remaining):
        if not remaining[r, c]:
            continue
        total += 1
        q: deque[tuple[int, int]] = deque([(int(r), int(c))])
        remaining[r, c] = False
        while q:
            rr, cc = q.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and remaining[nr, nc]:
                    remaining[nr, nc] = False
                    q.append((nr, nc))
    return total


def _inside_mask(grid: np.ndarray) -> np.ndarray:
    indicators = np.argwhere(grid == INDICATOR)
    assert len(indicators) == 1, "expected exactly one yellow indicator"
    start = tuple(map(int, indicators[0]))
    wall = grid == WALL
    outside = _flood_from_seeds(~wall, [start])
    inside = (~wall) & (~outside)
    assert np.any(inside), "expected a non-empty enclosed interior"
    return inside


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    inside = _inside_mask(g)
    out = g.copy()
    for r in range(g.shape[0]):
        cols = np.flatnonzero(inside[r] & (g[r] != BG))
        if len(cols) > 1:
            out[r, cols] = g[r, cols][::-1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each row's colored interior sequence inside a wall-defined flooded partition.

    Input structure:
    - The grid has black background, one yellow indicator pixel, and grey wall pixels forming a closed 4-connected loop that does not touch the canvas edge.
    - The yellow pixel lies outside the wall. The wall partitions the canvas into an outside region and an enclosed inside region.
    - Colored pattern pixels appear mainly inside the wall, with optional decorative colored pixels outside it.

    Rule:
    - Flood-fill 4-connectedly from the yellow indicator, treating grey wall pixels as blocking cells; the reached cells are the outside region.
    - Every non-wall cell not reached by that flood-fill is inside the wall.
    - For each row independently, collect the non-background pixels that lie inside the wall and reverse only their left-to-right color order in place.
    - Leave wall pixels, the indicator pixel, background, and every outside-region pixel unchanged.

    Key insight:
    - The wall does not mark the part to edit directly; it must first be used to separate outside from inside, and only the colored pixels on each interior row are locally reversed.
    """
    return _transform(grid)


def _add_segment(cells: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        a, b = sorted((c0, c1))
        for c in range(a, b + 1):
            cells.add((r0, c))
    else:
        a, b = sorted((r0, r1))
        for r in range(a, b + 1):
            cells.add((r, c0))


def _rect_perimeter(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _apply_notch(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    side: str,
    start: int,
    span: int,
    depth: int,
) -> None:
    end = start + span
    if side == "top":
        for c in range(start + 1, end):
            cells.discard((0, c))
        _add_segment(cells, 0, start, depth, start)
        _add_segment(cells, 0, end, depth, end)
        _add_segment(cells, depth, start, depth, end)
    elif side == "bottom":
        base = h - 1
        inner = base - depth
        for c in range(start + 1, end):
            cells.discard((base, c))
        _add_segment(cells, base, start, inner, start)
        _add_segment(cells, base, end, inner, end)
        _add_segment(cells, inner, start, inner, end)
    elif side == "left":
        for r in range(start + 1, end):
            cells.discard((r, 0))
        _add_segment(cells, start, 0, start, depth)
        _add_segment(cells, end, 0, end, depth)
        _add_segment(cells, start, depth, end, depth)
    else:
        base = w - 1
        inner = base - depth
        for r in range(start + 1, end):
            cells.discard((r, base))
        _add_segment(cells, start, base, start, inner)
        _add_segment(cells, end, base, end, inner)
        _add_segment(cells, start, inner, end, inner)


def _make_wall_patch(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(7, 11)
    w = rng.randint(7, 11)
    cells = _rect_perimeter(h, w)
    notch_count = rng.choice((1, 1, 2, 2, 3))
    sides = rng.sample(["top", "bottom", "left", "right"], notch_count)

    for side in sides:
        if side in ("top", "bottom"):
            if w < 6:
                return None
            span = rng.randint(2, min(4, w - 3))
            start = rng.randint(1, w - span - 2)
            max_depth = min(2, h - 4)
        else:
            if h < 6:
                return None
            span = rng.randint(2, min(4, h - 3))
            start = rng.randint(1, h - span - 2)
            max_depth = min(2, w - 4)
        if max_depth < 1:
            return None
        depth = rng.randint(1, max_depth)
        _apply_notch(cells, h, w, side, start, span, depth)

    wall = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        wall[r, c] = True

    if _component_count(wall) != 1:
        return None

    for r, c in np.argwhere(wall):
        deg = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(wall.shape, nr, nc) and wall[nr, nc]:
                deg += 1
        if deg != 2:
            return None

    outside = _flood_from_seeds(
        ~wall,
        [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)],
    )
    inside = (~wall) & (~outside)
    if not np.any(inside) or _component_count(inside) != 1:
        return None

    changed_rows = [r for r in range(h) if np.count_nonzero(inside[r]) >= 4]
    if len(changed_rows) < 3:
        return None
    return wall, inside


def _sample_row_pattern(rng: random.Random, cols: list[int]) -> tuple[list[int], list[int]]:
    count = rng.randint(2, min(5, len(cols)))
    picks = sorted(rng.sample(cols, count))
    while True:
        colors = [rng.choice(INNER_COLORS) for _ in range(count)]
        if len(set(colors)) >= 2 and colors != colors[::-1]:
            return picks, colors


def _reverse_nonbg_in_rows(mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int)
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(mask[r] & (grid[r] != BG))
        if len(cols) > 1:
            out[r, cols] = grid[r, cols][::-1]
    return out


def _bbox_row_reverse(mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return _reverse_nonbg_in_rows(bbox_mask, grid)


def _generate_case(rng: random.Random) -> np.ndarray | None:
    wall_patch = _make_wall_patch(rng)
    if wall_patch is None:
        return None
    wall_local, inside_local = wall_patch
    ph, pw = wall_local.shape

    margin_top = rng.randint(1, 3)
    margin_bottom = rng.randint(1, 3)
    margin_left = rng.randint(1, 3)
    margin_right = rng.randint(1, 3)
    h = ph + margin_top + margin_bottom
    w = pw + margin_left + margin_right
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    top, left = margin_top, margin_left
    grid[top : top + ph, left : left + pw][wall_local] = WALL

    wall_global = np.zeros_like(grid, dtype=bool)
    wall_global[top : top + ph, left : left + pw] = wall_local
    inside_global = np.zeros_like(grid, dtype=bool)
    inside_global[top : top + ph, left : left + pw] = inside_local
    outside_global = (~wall_global) & (~inside_global)

    edge_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and outside_global[r, c]
    ]
    if not edge_cells:
        return None
    ind_r, ind_c = rng.choice(edge_cells)
    grid[ind_r, ind_c] = INDICATOR

    candidate_rows = [
        r
        for r in range(h)
        if np.count_nonzero(inside_global[r]) >= 4
        and np.count_nonzero(outside_global[r]) >= 2
    ]
    if len(candidate_rows) < 3:
        return None
    rng.shuffle(candidate_rows)
    active_rows = sorted(candidate_rows[: rng.randint(3, min(6, len(candidate_rows)))])

    changed_rows = 0
    for r in active_rows:
        cols = list(map(int, np.flatnonzero(inside_global[r])))
        picks, colors = _sample_row_pattern(rng, cols)
        before = grid[r, picks].copy()
        grid[r, picks] = np.array(colors, dtype=int)
        if not np.array_equal(before, grid[r, picks][::-1]):
            changed_rows += 1

    for r in range(h):
        if r in active_rows or rng.random() >= 0.35:
            continue
        cols = list(map(int, np.flatnonzero(inside_global[r])))
        if len(cols) < 2:
            continue
        extra_count = rng.randint(1, min(3, len(cols)))
        extra_cols = sorted(rng.sample(cols, extra_count))
        for c in extra_cols:
            grid[r, c] = rng.choice(INNER_COLORS)

    if changed_rows < 3:
        return None

    decoration_positions: set[tuple[int, int]] = set()
    deco_row = rng.choice(active_rows)
    deco_candidates = [
        c
        for c in range(w)
        if outside_global[deco_row, c] and grid[deco_row, c] == BG and (deco_row, c) != (ind_r, ind_c)
    ]
    if not deco_candidates:
        return None
    decoration_positions.add((deco_row, rng.choice(deco_candidates)))

    all_deco_candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if outside_global[r, c] and grid[r, c] == BG and (r, c) != (ind_r, ind_c)
    ]
    rng.shuffle(all_deco_candidates)
    extra_decos = rng.randint(0, 3)
    for cell in all_deco_candidates:
        if len(decoration_positions) >= 1 + extra_decos:
            break
        decoration_positions.add(cell)
    for r, c in decoration_positions:
        grid[r, c] = DECORATION

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[wall_global], grid[wall_global]):
        return None
    if out[ind_r, ind_c] != INDICATOR:
        return None
    if not np.array_equal(out[outside_global], grid[outside_global]):
        return None

    changed_row_count = 0
    for r in range(h):
        cols = np.flatnonzero(inside_global[r] & (grid[r] != BG))
        if len(cols) > 1 and not np.array_equal(grid[r, cols], out[r, cols]):
            if np.array_equal(out[r, cols], grid[r, cols][::-1]):
                changed_row_count += 1
            else:
                return None
    if changed_row_count < 3:
        return None

    if np.array_equal(out, _reverse_nonbg_in_rows(~wall_global, grid)):
        return None
    if np.array_equal(out, _reverse_nonbg_in_rows(np.ones_like(wall_global, dtype=bool), grid)):
        return None
    if np.array_equal(out, _bbox_row_reverse(inside_global, grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_ATTEMPTS} attempts")
