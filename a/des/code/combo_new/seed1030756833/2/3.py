from __future__ import annotations

import random

import numpy as np


GREY = 5
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(cell: tuple[int, int], shape: tuple[int, int]):
    r, c = cell
    h, w = shape
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nr, nc in _neighbors4(cur, mask.shape):
                    if not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(g: np.ndarray, colors: set[int]) -> list[dict]:
    comps = []
    for color in colors:
        for coords in _components(g == color):
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _is_solid_3x3(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return comp["area"] == 9 and r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3


def _trace_path(path_cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    shape_cells = path_cells
    degrees = {
        cell: sum((nr, nc) in shape_cells for nr, nc in _neighbors4(cell, _trace_path.shape))
        for cell in shape_cells
    }
    assert all(deg <= 2 for deg in degrees.values()), "grey path must be single-width with no branches"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "grey path must have exactly two endpoints"
    assert start in endpoints, "start path cell must be a path endpoint"

    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        next_cells = [
            nb for nb in _neighbors4(cur, _trace_path.shape)
            if nb in path_cells and nb != prev
        ]
        if prev is not None and len(next_cells) == 0:
            break
        assert len(next_cells) == 1, "path order must be unambiguous"
        nxt = next_cells[0]
        order.append(nxt)
        prev, cur = cur, nxt

    assert len(order) == len(path_cells), "path trace did not visit every grey cell"
    return order


def _marker_square_cells(
    endpoint: tuple[int, int],
    outward: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = endpoint
    dr, dc = outward
    if (dr, dc) == (-1, 0):
        r0, c0 = r - 3, c - 1
    elif (dr, dc) == (1, 0):
        r0, c0 = r + 1, c - 1
    elif (dr, dc) == (0, -1):
        r0, c0 = r - 1, c - 3
    elif (dr, dc) == (0, 1):
        r0, c0 = r - 1, c + 1
    else:
        raise AssertionError("endpoint direction must be orthogonal")

    cells = [(rr, cc) for rr in range(r0, r0 + 3) for cc in range(c0, c0 + 3)]
    h, w = shape
    assert all(0 <= rr < h and 0 <= cc < w for rr, cc in cells), "3x3 marker must fit in the grid"
    return cells


def _find_start_marker(g: np.ndarray, bg: int) -> dict:
    colors = set(map(int, g.ravel())) - {bg, GREY}
    squares = [comp for comp in _monochrome_components(g, colors) if _is_solid_3x3(comp)]
    assert len(squares) == 1, "expected exactly one solid 3x3 start marker"
    return squares[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Carry a color signal along the grey path and stamp its final color at the far endpoint.

    Input structure:
    - The dominant color is background, and a single-pixel-wide grey orthogonal
      path forms one non-branching connected component with two endpoints.
    - One path endpoint touches a unique solid 3x3 start marker whose color is
      the initial signal color.
    - Additional colored single-pixel markers sit orthogonally beside path cells;
      they are not grey, not background, and not part of the start marker.

    Rule:
    - Start from the path endpoint touching the 3x3 marker and walk through the
      grey path to the other endpoint.
    - The signal begins as the start marker color; whenever the current path
      cell has an adjacent side marker, replace the signal with that marker's color.
    - At the far endpoint, draw a new solid 3x3 square beyond the endpoint in
      the outgoing path direction, using the final signal color.
    - Preserve the grey path, all side markers, and the original start marker.

    Key insight:
    - The color written at the empty endpoint is determined by the last side
      marker encountered in path order, so the hidden ordering of the grey path
      must be traced rather than inferred from position or from the start color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    _trace_path.shape = (h, w)

    bg = _background(g)
    path_components = _components(g == GREY)
    assert len(path_components) == 1, "expected one connected grey path"
    path_cells = set(path_components[0])
    assert len(path_cells) >= 2, "path must have at least two cells"

    start_marker = _find_start_marker(g, bg)
    start_color = int(start_marker["color"])
    start_cells = set(start_marker["coords"])

    degrees = {
        cell: sum(nb in path_cells for nb in _neighbors4(cell, g.shape))
        for cell in path_cells
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "grey path must have exactly two endpoints"
    start_candidates = [
        endpoint for endpoint in endpoints
        if any(nb in start_cells for nb in _neighbors4(endpoint, g.shape))
    ]
    assert len(start_candidates) == 1, "exactly one path endpoint must touch the start marker"
    start_endpoint = start_candidates[0]

    colored_other = {
        tuple(map(int, coord))
        for coord in np.argwhere((g != bg) & (g != GREY))
    } - start_cells
    for cell in colored_other:
        assert sum(nb in colored_other for nb in _neighbors4(cell, g.shape)) == 0, "side markers must be single pixels"
        assert sum(nb in path_cells for nb in _neighbors4(cell, g.shape)) == 1, "each side marker must touch exactly one path cell"

    order = _trace_path(path_cells, start_endpoint)
    signal = start_color
    seen_markers: set[tuple[int, int]] = set()
    for cell in order:
        adjacent = [
            nb for nb in _neighbors4(cell, g.shape)
            if nb in colored_other and nb not in seen_markers
        ]
        assert len(adjacent) <= 1, "at most one side marker may belong to each path position"
        if adjacent:
            marker_cell = adjacent[0]
            seen_markers.add(marker_cell)
            signal = int(g[marker_cell])
    assert seen_markers == colored_other, "all colored single-pixel markers must be beside the path"

    prev = order[-2]
    end = order[-1]
    outward = (end[0] - prev[0], end[1] - prev[1])
    end_square = _marker_square_cells(end, outward, g.shape)
    assert all(int(g[r, c]) == bg for r, c in end_square), "endpoint marker area must be empty background"

    out = g.copy()
    for r, c in end_square:
        out[r, c] = signal
    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
PATH_LEN_RANGE = (9, 34)
SIDE_MARKER_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 500
PATH_ATTEMPTS = 220
WALK_ATTEMPTS = 80


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return -direction[0], -direction[1]


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _touches_any(cell: tuple[int, int], cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    return any(nb in cells for nb in _neighbors4(cell, shape))


def _touches_any8(cell: tuple[int, int], cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = (r + dr, c + dc)
            if _in_bounds(nb, shape) and nb in cells:
                return True
    return False


def _valid_new_path_cell(
    cell: tuple[int, int],
    cur: tuple[int, int],
    path: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    if not _in_bounds(cell, shape):
        return False
    if cell in path or cell in forbidden:
        return False
    if _touches_any(cell, forbidden, shape):
        return False
    path_neighbors = [nb for nb in _neighbors4(cell, shape) if nb in path]
    return path_neighbors == [cur]


def _sample_start(shape: tuple[int, int], first_dir: tuple[int, int]):
    h, w = shape
    outward = _opposite(first_dir)
    valid = []
    for r in range(h):
        for c in range(w):
            endpoint = (r, c)
            next_cell = (r + first_dir[0], c + first_dir[1])
            if not _in_bounds(next_cell, shape):
                continue
            try:
                _marker_square_cells(endpoint, outward, shape)
            except AssertionError:
                continue
            valid.append(endpoint)
    return random.choice(valid) if valid else None


def _path_endpoint_is_clear(
    path: list[tuple[int, int]],
    start_area: set[tuple[int, int]],
    end_area: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    path_set = set(path)
    if start_area & path_set or end_area & path_set or start_area & end_area:
        return False
    start = path[0]
    end = path[-1]
    for cell in path[1:]:
        if _touches_any(cell, start_area, shape):
            return False
    for cell in path[:-1]:
        if _touches_any(cell, end_area, shape):
            return False
    return _touches_any(start, start_area, shape) and _touches_any(end, end_area, shape)


def _sample_path(shape: tuple[int, int]):
    for _ in range(PATH_ATTEMPTS):
        first_dir = random.choice(DIRS)
        start = _sample_start(shape, first_dir)
        if start is None:
            continue
        start_area = set(_marker_square_cells(start, _opposite(first_dir), shape))
        second = (start[0] + first_dir[0], start[1] + first_dir[1])
        if second in start_area:
            continue

        target_len = random.randint(*PATH_LEN_RANGE)
        path = [start, second]
        path_set = set(path)
        cur_dir = first_dir
        success = True
        for _step in range(target_len - 2):
            weighted_dirs = [cur_dir, cur_dir] + list(DIRS)
            random.shuffle(weighted_dirs)
            placed = False
            for direction in weighted_dirs:
                if direction == _opposite(cur_dir):
                    continue
                cur = path[-1]
                nxt = (cur[0] + direction[0], cur[1] + direction[1])
                if _valid_new_path_cell(nxt, cur, path_set, start_area, shape):
                    path.append(nxt)
                    path_set.add(nxt)
                    cur_dir = direction
                    placed = True
                    break
            if not placed:
                success = False
                break
        if not success:
            continue

        end = path[-1]
        prev = path[-2]
        outward = (end[0] - prev[0], end[1] - prev[1])
        try:
            end_area = set(_marker_square_cells(end, outward, shape))
        except AssertionError:
            continue
        if _path_endpoint_is_clear(path, start_area, end_area, shape):
            return path, start_area, end_area
    return None


def _path_neighbor_count(cell: tuple[int, int], path_set: set[tuple[int, int]], shape: tuple[int, int]) -> int:
    return sum(nb in path_set for nb in _neighbors4(cell, shape))


def _side_marker_candidates(
    path: list[tuple[int, int]],
    start_area: set[tuple[int, int]],
    end_area: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict[int, list[tuple[int, int]]]:
    path_set = set(path)
    reserved = path_set | start_area | end_area
    candidates: dict[int, list[tuple[int, int]]] = {}
    for idx in range(1, len(path) - 1):
        cells = []
        for nb in _neighbors4(path[idx], shape):
            if nb in reserved:
                continue
            if _path_neighbor_count(nb, path_set, shape) != 1:
                continue
            if _touches_any8(nb, start_area, shape) or _touches_any8(nb, end_area, shape):
                continue
            cells.append(nb)
        if cells:
            candidates[idx] = cells
    return candidates


def _sample_side_markers(
    path: list[tuple[int, int]],
    start_area: set[tuple[int, int]],
    end_area: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, tuple[int, int]] | tuple[int, tuple[int, int]]]:
    candidates = _side_marker_candidates(path, start_area, end_area, shape)
    available = sorted(candidates)
    if len(available) < SIDE_MARKER_RANGE[0]:
        return []
    count = random.randint(SIDE_MARKER_RANGE[0], min(SIDE_MARKER_RANGE[1], len(available)))
    for _ in range(WALK_ATTEMPTS):
        chosen_indices = sorted(random.sample(available, count))
        occupied = set(start_area) | set(end_area) | set(path)
        markers = []
        ok = True
        for idx in chosen_indices:
            options = candidates[idx][:]
            random.shuffle(options)
            picked = None
            for cell in options:
                if cell in occupied:
                    continue
                if _touches_any8(cell, occupied - set(path), shape):
                    continue
                picked = cell
                break
            if picked is None:
                ok = False
                break
            markers.append((idx, picked))
            occupied.add(picked)
        if ok:
            return markers
    return []


def _build_expected(inp: np.ndarray, end_area: set[tuple[int, int]], final_color: int) -> np.ndarray:
    out = inp.copy()
    for r, c in end_area:
        out[r, c] = final_color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    shape = (height, width)
    bg = random.choice([c for c in range(10) if c != GREY])
    palette = [c for c in range(10) if c not in (bg, GREY)]

    sampled = _sample_path(shape)
    if sampled is None:
        return None
    path, start_area, end_area = sampled
    side_markers = _sample_side_markers(path, start_area, end_area, shape)
    if len(side_markers) < SIDE_MARKER_RANGE[0]:
        return None

    start_color = random.choice(palette)
    marker_colors = []
    signal = start_color
    for i in range(len(side_markers)):
        choices = [c for c in palette if c != signal]
        if i == len(side_markers) - 1:
            without_start = [c for c in choices if c != start_color]
            if without_start:
                choices = without_start
        color = random.choice(choices)
        marker_colors.append(color)
        signal = color
    final_color = marker_colors[-1]

    grid = np.full(shape, bg, dtype=int)
    for r, c in path:
        grid[r, c] = GREY
    for r, c in start_area:
        grid[r, c] = start_color
    for (_idx, cell), color in zip(side_markers, marker_colors):
        grid[cell] = color

    expected = _build_expected(grid, end_area, final_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
