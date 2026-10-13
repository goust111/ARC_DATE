from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _compute_active_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_cells = np.argwhere(g == RED)
    assert len(marker_cells) == 1, "expected exactly one red marker"

    marker_col = int(marker_cells[0, 1])
    wall_rows = np.flatnonzero(np.any(g == GRAY, axis=1))
    assert wall_rows.size > 0, "expected at least one wall row"

    top = int(wall_rows.min())
    bottom = int(wall_rows.max())
    height, width = g.shape
    active = np.zeros((height, width), dtype=bool)

    for r in range(top, bottom + 1):
        wall_cols = np.flatnonzero(g[r] == GRAY)
        if wall_cols.size == 0:
            active[r, :] = True
            continue

        left_cols = wall_cols[wall_cols <= marker_col]
        right_cols = wall_cols[wall_cols >= marker_col]
        start = int(left_cols.max()) + 1 if left_cols.size else 0
        stop = int(right_cols.min()) if right_cols.size else width
        if start < stop:
            active[r, start:stop] = True

    return active


def _find_object_components(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = np.isin(g, (GREEN, YELLOW))
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue

        queue: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []

        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

        components.append(np.array(coords, dtype=int))

    return components


def _analyze_objects(grid: np.ndarray) -> list[tuple[np.ndarray, bool, int]]:
    active = _compute_active_mask(grid)
    g = np.asarray(grid, dtype=int)
    analyzed: list[tuple[np.ndarray, bool, int]] = []

    for comp in _find_object_components(g):
        rows = comp[:, 0]
        cols = comp[:, 1]
        fully_inside = bool(np.all(active[rows, cols]))
        marker_count = int(np.count_nonzero(g[rows, cols] == YELLOW))
        analyzed.append((comp, fully_inside, marker_count))

    return analyzed


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp, fully_inside, marker_count in _analyze_objects(g):
        if fully_inside and marker_count >= 2:
            rows = comp[:, 0]
            cols = comp[:, 1]
            out[rows, cols] = BLACK

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase only the foreground objects that are both fully inside a scanline-defined region and carry multiple markers.

    Input structure:
    - The grid has black background, one red singleton marker, and gray wall pixels
      on several rows that define scanline bounds relative to the marker column.
    - Foreground objects are 4-connected multicolor components made from green body
      pixels and embedded yellow marker pixels.

    Rule:
    - Build a logical active region between the topmost and bottommost wall rows.
      On each wall row, use the nearest gray pixel on each side of the red marker
      column as the row bounds, extending to the grid edge when one side is absent.
      Rows between those extrema that contain no gray wall pixels are fully active.
    - For each green/yellow 4-connected object, test whether every object pixel lies
      in the active region and count how many yellow marker pixels it contains.
    - Erase the whole object to black only when both conditions hold: it is fully
      inside the active region and it contains at least two yellow markers.

    Key insight:
    - The erase decision is gated by the conjunction of a spatial condition and a
      marker-count condition, so neither the scanline region nor the marker count
      alone is enough to predict which objects disappear.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _dilate_mask(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    out = mask.copy()
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        neighbors = _neighbors8 if connectivity == 8 else _neighbors4
        for nr, nc in neighbors(r, c):
            if _in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _normalize_shape(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform_shape(
    cells: tuple[tuple[int, int], ...],
    rotations: int,
    flip: bool,
) -> tuple[tuple[int, int], ...]:
    transformed: list[tuple[int, int]] = []
    for r, c in cells:
        x, y = r, c
        for _ in range(rotations % 4):
            x, y = y, -x
        if flip:
            y = -y
        transformed.append((x, y))
    return _normalize_shape(tuple(transformed))


def _marker_candidates(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    cell_set = set(shape)
    return [
        cell
        for cell in shape
        if sum((nr, nc) in cell_set for nr, nc in _neighbors4(*cell)) >= 2
    ]


def _random_shape(size: int, min_marker_sites: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(120):
        cells = {(3, 3)}
        frontier = {(2, 3), (4, 3), (3, 2), (3, 4)}

        while len(cells) < size and frontier:
            options = list(frontier)
            weights = []
            for cell in options:
                degree = sum(neighbor in cells for neighbor in _neighbors4(*cell))
                weights.append((degree + 0.3) ** 3)
            chosen = random.choices(options, weights=weights, k=1)[0]
            frontier.remove(chosen)
            cells.add(chosen)
            for nr, nc in _neighbors4(*chosen):
                if (nr, nc) not in cells and 0 <= nr <= 7 and 0 <= nc <= 7:
                    frontier.add((nr, nc))

        shape = _transform_shape(_normalize_shape(cells), random.randrange(4), bool(random.randrange(2)))
        if len(shape) != size:
            continue
        if len(_marker_candidates(shape)) < min_marker_sites:
            continue
        if max(r for r, _c in shape) >= 5 or max(c for _r, c in shape) >= 5:
            continue
        return shape

    return None


def _choose_marker_sites(
    shape: tuple[tuple[int, int], ...],
    count: int,
) -> tuple[tuple[int, int], ...] | None:
    if count == 0:
        return ()

    candidates = _marker_candidates(shape)
    combos = list(itertools.combinations(candidates, count))
    if not combos:
        return None

    spaced = [
        combo
        for combo in combos
        if all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2
            for a, b in itertools.combinations(combo, 2)
        )
    ]
    chosen = random.choice(spaced or combos)
    return tuple(sorted(chosen))


def _absolute_cells(
    rel_cells: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in rel_cells]


def _paint_object(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    marker_cells: list[tuple[int, int]],
) -> None:
    for r, c in cells:
        grid[r, c] = GREEN
    for r, c in marker_cells:
        grid[r, c] = YELLOW


def _place_object(
    grid: np.ndarray,
    active: np.ndarray,
    blocked: np.ndarray,
    *,
    mode: str,
    marker_count: int,
) -> bool:
    for _ in range(120):
        size = random.randint(5, 9)
        shape = _random_shape(size, marker_count)
        if shape is None:
            continue

        marker_shape = _choose_marker_sites(shape, marker_count)
        if marker_shape is None:
            continue

        shape_h = max(r for r, _c in shape) + 1
        shape_w = max(c for _r, c in shape) + 1
        candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []

        for r0 in range(grid.shape[0] - shape_h + 1):
            for c0 in range(grid.shape[1] - shape_w + 1):
                cells = _absolute_cells(shape, r0, c0)
                if any(blocked[r, c] for r, c in cells):
                    continue

                inside_count = sum(active[r, c] for r, c in cells)
                if mode == "inside" and inside_count != len(cells):
                    continue
                if mode == "straddle" and not (0 < inside_count < len(cells)):
                    continue

                markers = _absolute_cells(marker_shape, r0, c0)
                candidates.append((cells, markers))

        if not candidates:
            continue

        cells, markers = random.choice(candidates)
        _paint_object(grid, cells, markers)

        obj_mask = np.zeros_like(blocked)
        for r, c in cells:
            obj_mask[r, c] = True
        blocked |= _dilate_mask(obj_mask, connectivity=8)
        return True

    return False


def _gen_scanline_layout() -> np.ndarray | None:
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    marker_col = random.randint(4, width - 5)

    top = random.randint(2, 4)
    bottom = random.randint(max(top + 6, height - 7), height - 3)
    span = bottom - top + 1
    wall_count = random.randint(4, min(6, span - 2))

    chosen_rows = {top, bottom}
    middle_rows = [r for r in range(top + 1, bottom)]
    chosen_rows.update(random.sample(middle_rows, wall_count - 2))
    wall_rows = sorted(chosen_rows)
    blank_rows = [r for r in range(top, bottom + 1) if r not in chosen_rows]
    if not blank_rows:
        return None

    marker_row = random.choice(blank_rows)
    base_left = random.randint(1, marker_col - 3)
    base_right = random.randint(marker_col + 3, width - 2)

    grid = np.full((height, width), BLACK, dtype=int)
    for idx, r in enumerate(wall_rows):
        left = max(0, min(marker_col - 2, base_left + random.randint(-1, 1)))
        right = min(width - 1, max(marker_col + 2, base_right + random.randint(-1, 1)))

        if idx in (0, len(wall_rows) - 1) or random.random() < 0.6:
            walls = [left, right]
            if random.random() < 0.35:
                side = random.choice(("L", "R"))
                if side == "L" and left > 0:
                    walls.append(random.randint(0, left - 1))
                elif side == "R" and right < width - 1:
                    walls.append(random.randint(right + 1, width - 1))
        elif random.random() < 0.5:
            walls = [left]
            if left > 0 and random.random() < 0.5:
                walls.append(random.randint(0, left - 1))
        else:
            walls = [right]
            if right < width - 1 and random.random() < 0.5:
                walls.append(random.randint(right + 1, width - 1))

        for c in sorted(set(walls))[:3]:
            grid[r, c] = GRAY

    grid[marker_row, marker_col] = RED
    active = _compute_active_mask(grid)
    if int(active.sum()) < 40:
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _gen_scanline_layout()
    if grid is None:
        return None

    active = _compute_active_mask(grid)
    reserved = grid != BLACK
    blocked = _dilate_mask(reserved, connectivity=4)

    required = [
        ("straddle", 2),
        ("inside", 2),
        ("inside", random.choice((0, 1))),
    ]
    random.shuffle(required)
    for mode, marker_count in required:
        if not _place_object(grid, active, blocked, mode=mode, marker_count=marker_count):
            return None

    extra_count = random.randint(0, 2)
    extra_specs = [
        ("inside", 2),
        ("inside", 1),
        ("inside", 0),
        ("straddle", 2),
        ("straddle", 1),
    ]
    for _ in range(extra_count):
        mode, marker_count = random.choice(extra_specs)
        _place_object(grid, active, blocked, mode=mode, marker_count=marker_count)

    analysis = _analyze_objects(grid)
    if not any(inside and markers >= 2 for _comp, inside, markers in analysis):
        return None
    if not any((not inside) and markers >= 2 for _comp, inside, markers in analysis):
        return None
    if not any(inside and markers <= 1 for _comp, inside, markers in analysis):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scanline-gated erase input")
