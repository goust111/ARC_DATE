from __future__ import annotations

import numpy as np


BG = 0
WALL = 8
PALETTE = [2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _bbox_top_left(coords: np.ndarray) -> tuple[int, int]:
    mins = coords.min(axis=0)
    return int(mins[0]), int(mins[1])


def _row_major_colors(grid: np.ndarray, exclude: set[int]) -> list[int]:
    seen = set(exclude)
    order: list[int] = []
    for value in grid.ravel():
        color = int(value)
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _path_cells(start: Coord, stop: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    cells: list[Coord] = []
    if r0 == r1 or c0 == c1:
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                cells.append((r0, c))
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                cells.append((r, c0))
        return cells

    if first_axis == "h":
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
        step = 1 if r1 >= r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c1))
        return cells

    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        cells.append((r, c0))
    step = 1 if c1 >= c0 else -1
    for c in range(c0 + step, c1 + step, step):
        cells.append((r1, c))
    return cells


def _path_clear(canvas: np.ndarray, path: list[Coord], color: int) -> bool:
    return all(int(canvas[r, c]) in (BG, color) for r, c in path)


def _best_l_path(comp_a: np.ndarray, comp_b: np.ndarray, canvas: np.ndarray, color: int) -> list[Coord] | None:
    candidates: list[tuple[tuple, list[Coord]]] = []
    a_cells = [tuple(map(int, cell)) for cell in comp_a]
    b_cells = [tuple(map(int, cell)) for cell in comp_b]
    a_cells.sort()
    b_cells.sort()

    for start in a_cells:
        for stop in b_cells:
            if start == stop:
                continue
            dist = abs(start[0] - stop[0]) + abs(start[1] - stop[1])
            axes = ["h"] if start[0] == stop[0] else ["v"] if start[1] == stop[1] else ["h", "v"]
            for axis in axes:
                path = _path_cells(start, stop, axis)
                if not _path_clear(canvas, path, color):
                    continue
                # Deterministic tie-break:
                # 1. shortest source-target distance
                # 2. horizontal-first when both one-bend routes are available
                # 3. earliest source pixel, then earliest target pixel
                key = (
                    dist,
                    0 if axis == "h" else 1,
                    start[0],
                    start[1],
                    stop[0],
                    stop[1],
                    tuple(path),
                )
                candidates.append((key, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _solve(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    foreground = _row_major_colors(g, {BG, WALL})
    assert len(foreground) >= 2, "expected at least two non-wall foreground colors"

    component_counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for idx, value in enumerate(g.ravel()):
        color = int(value)
        if color not in (BG, WALL) and color not in first_seen:
            first_seen[color] = idx
    for color in foreground:
        component_counts[color] = len(_components_of_color(g, color))

    noise = min(
        foreground,
        key=lambda color: (-component_counts[color], first_seen[color]),
    )

    out = g.copy()
    out[out == noise] = BG

    remaining = [color for color in foreground if color != noise]
    traces: dict[int, list[list[Coord]]] = {}
    sorted_components: dict[int, list[np.ndarray]] = {}
    for color in remaining:
        comps = _components_of_color(out, color)
        comps.sort(key=_bbox_top_left)
        sorted_components[color] = comps
        traces[color] = []
        for comp_a, comp_b in zip(comps, comps[1:]):
            path = _best_l_path(comp_a, comp_b, out, color)
            assert path is not None, f"expected a clear L-path for color {color}"
            traces[color].append(path)
            for r, c in path:
                out[r, c] = color

    if with_trace:
        return out, {
            "noise": noise,
            "component_counts": component_counts,
            "components": sorted_components,
            "paths": traces,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the color with the most components, then use the freed cells to connect the remaining colors.

    Input structure:
    - The grid has a black background, 3-5 teal wall segments, and 3-4 other foreground colors.
    - One non-black non-teal color appears in strictly more 4-connected components than the others.
    - Every remaining non-wall color has at least two separated components that may need the erased space to connect.

    Rule:
    - Count 4-connected components for each non-black, non-teal color and mark the most frequent one as noise.
    - Erase that noise color to black; if there is a tie, use the color first encountered in row-major scan order.
    - For every remaining color, sort its components by the row-major order of each component's bbox top-left corner.
    - Connect each consecutive component pair with the shortest clear orthogonal path using at most one bend, passing only through black or that same color; when both bends are equally valid, go along the current row first and then turn.
    - Keep teal wall cells unchanged.

    Key insight:
    - The highest-frequency color is not merely ignored: removing it creates the only usable corridor for at least one later connector.
    """
    return _solve(grid)


def _paint(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _all_empty(grid: np.ndarray, cells: list[Coord]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
    return True


def _component_shape(anchor: Coord, primary_away: Coord, secondary_away: Coord | None = None) -> list[Coord]:
    cells = [anchor, (anchor[0] + primary_away[0], anchor[1] + primary_away[1])]
    if secondary_away is not None:
        cells.append((anchor[0] + secondary_away[0], anchor[1] + secondary_away[1]))
    return cells


def _route_only(path: list[Coord], component_cells: set[Coord]) -> list[Coord]:
    return [cell for cell in path if cell not in component_cells]


def _place_special_chain(grid: np.ndarray, color: int):
    h, w = grid.shape
    horizontal_first = bool(np.random.randint(2))

    if horizontal_first:
        r1 = int(np.random.randint(2, h // 2 - 1))
        gap_r = int(np.random.randint(3, 6))
        r2 = r1 + gap_r
        c1 = int(np.random.randint(3, max(4, w // 2 - 1)))
        gap_c = int(np.random.randint(4, 7))
        c2 = min(w - 3, c1 + gap_c)
        if c2 - c1 < 4 or r2 >= h - 2:
            return None

        comp1 = [(r1, c1 - 1), (r1, c1)]
        comp2 = [(r2, c2), (r2, c2 + 1)]
        blocker = [(rr, c1) for rr in range(r1 + 1, r2)]
        intended = _path_cells((r1, c1), (r2, c2), "h")
    else:
        c1 = int(np.random.randint(2, w // 2 - 1))
        gap_c = int(np.random.randint(3, 6))
        c2 = c1 + gap_c
        r1 = int(np.random.randint(3, max(4, h // 2 - 1)))
        gap_r = int(np.random.randint(4, 7))
        r2 = min(h - 3, r1 + gap_r)
        if r2 - r1 < 4 or c2 >= w - 2:
            return None

        comp1 = [(r1 - 1, c1), (r1, c1)]
        comp2 = [(r2, c2), (r2 + 1, c2)]
        blocker = [(r1, cc) for cc in range(c1 + 1, c2)]
        intended = _path_cells((r1, c1), (r2, c2), "v")

    if len(blocker) < 2 or len(blocker) > 4:
        return None

    all_cells = comp1 + comp2 + blocker
    if not _all_empty(grid, all_cells):
        return None

    component_cells = set(comp1 + comp2)
    route_cells = _route_only(intended, component_cells)
    if len(route_cells) < 4:
        return None
    if not _all_empty(grid, route_cells):
        return None

    _paint(grid, blocker, WALL)
    _paint(grid, comp1 + comp2 + route_cells, color)
    return {
        "color": color,
        "components": [np.array(sorted(comp1), dtype=int), np.array(sorted(comp2), dtype=int)],
        "route_cells": route_cells,
        "blocker": blocker,
    }


def _zone_templates(n_extra: int, h: int, w: int) -> list[tuple[int, int, int, int]]:
    zones = [
        (1, w // 2 + 1, h // 2, w - 2),
        (h // 2, 1, h - 2, w // 2 - 1),
        (h // 2, w // 2 + 1, h - 2, w - 2),
    ]
    np.random.shuffle(zones)
    return zones[:n_extra]


def _valid_zone(zone: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = zone
    return r1 - r0 >= 5 and c1 - c0 >= 5


def _place_two_component_chain(grid: np.ndarray, color: int, zone: tuple[int, int, int, int]):
    r0, c0, r1, c1 = zone
    style = str(np.random.choice(["row", "col", "diag"]))
    if style == "row":
        r = int(np.random.randint(r0 + 1, r1))
        a1 = (r, c0 + 1)
        a2 = (r, c1 - 1)
        comp1 = _component_shape(a1, (0, -1), (-1, 0) if r > r0 + 1 and np.random.random() < 0.5 else None)
        comp2 = _component_shape(a2, (0, 1), (1, 0) if r < r1 - 1 and np.random.random() < 0.5 else None)
        path = _path_cells(a1, a2, "h")
    elif style == "col":
        c = int(np.random.randint(c0 + 1, c1))
        a1 = (r0 + 1, c)
        a2 = (r1 - 1, c)
        comp1 = _component_shape(a1, (-1, 0), (0, -1) if c > c0 + 1 and np.random.random() < 0.5 else None)
        comp2 = _component_shape(a2, (1, 0), (0, 1) if c < c1 - 1 and np.random.random() < 0.5 else None)
        path = _path_cells(a1, a2, "v")
    else:
        a1 = (r0 + 1, c0 + 1)
        a2 = (r1 - 1, c1 - 1)
        comp1 = _component_shape(a1, (0, -1), (-1, 0) if np.random.random() < 0.5 else None)
        comp2 = _component_shape(a2, (0, 1), (1, 0) if np.random.random() < 0.5 else None)
        path = _path_cells(a1, a2, "h")

    all_cells = comp1 + comp2
    if not _all_empty(grid, all_cells):
        return None
    component_cells = set(all_cells)
    route_cells = _route_only(path, component_cells)
    if not route_cells or not _all_empty(grid, route_cells):
        return None

    _paint(grid, all_cells + route_cells, color)
    return {
        "color": color,
        "route_cells": route_cells,
    }


def _place_three_component_chain(grid: np.ndarray, color: int, zone: tuple[int, int, int, int]):
    r0, c0, r1, c1 = zone
    combo = str(np.random.choice(["row_then_col", "col_then_row"]))
    if combo == "row_then_col":
        a1 = (r0 + 1, c0 + 1)
        a2 = (r0 + 1, c1 - 2)
        a3 = (r1 - 1, c1 - 2)
        comp1 = _component_shape(a1, (0, -1), (-1, 0) if r0 + 1 > r0 else None)
        comp2 = [a2, (a2[0] - 1, a2[1]), (a2[0], a2[1] + 1)]
        comp3 = _component_shape(a3, (1, 0), (0, 1) if c1 - 2 < c1 - 1 else None)
        path1 = _path_cells(a1, a2, "h")
        path2 = _path_cells(a2, a3, "v")
    else:
        a1 = (r0 + 1, c0 + 1)
        a2 = (r1 - 2, c0 + 1)
        a3 = (r1 - 2, c1 - 1)
        comp1 = _component_shape(a1, (-1, 0), (0, -1) if c0 + 1 > c0 else None)
        comp2 = [a2, (a2[0] + 1, a2[1]), (a2[0], a2[1] - 1)]
        comp3 = _component_shape(a3, (0, 1), (1, 0) if r1 - 2 < r1 - 1 else None)
        path1 = _path_cells(a1, a2, "v")
        path2 = _path_cells(a2, a3, "h")

    all_cells = comp1 + comp2 + comp3
    if any(not _in_bounds(grid.shape, r, c) for r, c in all_cells):
        return None
    if not _all_empty(grid, all_cells):
        return None
    component_cells = set(all_cells)
    route_cells = _route_only(path1, component_cells) + _route_only(path2, component_cells)
    if not route_cells or len(set(route_cells)) != len(route_cells):
        return None
    if not _all_empty(grid, route_cells):
        return None

    _paint(grid, all_cells + route_cells, color)
    return {
        "color": color,
        "route_cells": route_cells,
    }


def _wall_neighborhood_free(grid: np.ndarray, cells: list[Coord]) -> bool:
    for r, c in cells:
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL:
                return False
    return True


def _place_extra_walls(grid: np.ndarray, count: int) -> bool:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _ in range(120):
            horizontal = bool(np.random.randint(2))
            length = int(np.random.randint(2, 5))
            if horizontal:
                r = int(np.random.randint(1, h - 1))
                c = int(np.random.randint(1, w - length))
                cells = [(r, cc) for cc in range(c, c + length)]
            else:
                r = int(np.random.randint(1, h - length))
                c = int(np.random.randint(1, w - 1))
                cells = [(rr, c) for rr in range(r, r + length)]
            if _wall_neighborhood_free(grid, cells):
                _paint(grid, cells, WALL)
                placed = True
                break
        if not placed:
            return False
    return True


def _erase_paths(output: np.ndarray, route_map: dict[int, list[Coord]]) -> np.ndarray:
    inp = output.copy()
    for color, cells in route_map.items():
        for r, c in cells:
            if int(inp[r, c]) == color:
                inp[r, c] = BG
    return inp


def _random_noise_shape(grid: np.ndarray, start: Coord, size: int) -> list[Coord] | None:
    cells = [start]
    used = {start}
    for _ in range(size - 1):
        frontier: list[Coord] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in used:
                    continue
                if int(grid[nr, nc]) != BG:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return None
        next_cell = frontier[int(np.random.randint(len(frontier)))]
        cells.append(next_cell)
        used.add(next_cell)
    return cells


def _noise_separated(grid: np.ndarray, cells: list[Coord], color: int) -> bool:
    chosen = set(cells)
    for r, c in cells:
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in chosen and int(grid[nr, nc]) == color:
                return False
    return True


def _place_noise_components(inp: np.ndarray, noise_color: int, special_route: list[Coord], noise_count: int) -> bool:
    route_candidates = special_route[1:-1]
    if len(route_candidates) < 3:
        return False

    blocker_len = int(np.random.randint(2, min(3, len(route_candidates)) + 1))
    blocker_start = int(np.random.randint(0, len(route_candidates) - blocker_len + 1))
    first_blocker = route_candidates[blocker_start : blocker_start + blocker_len]
    if not _noise_separated(inp, first_blocker, noise_color):
        return False
    _paint(inp, first_blocker, noise_color)
    placed = 1

    blocker_two_ok = len(route_candidates) >= 6 and np.random.random() < 0.5 and noise_count >= 5
    if blocker_two_ok:
        left = route_candidates[: max(0, blocker_start - 1)]
        right = route_candidates[blocker_start + blocker_len + 1 :]
        pools = [segment for segment in (left, right) if len(segment) >= 2]
        if pools:
            segment = pools[int(np.random.randint(len(pools)))]
            extra_len = int(np.random.randint(2, min(3, len(segment)) + 1))
            extra = segment[:extra_len] if np.random.random() < 0.5 else segment[-extra_len:]
            if _noise_separated(inp, extra, noise_color):
                _paint(inp, extra, noise_color)
                placed += 1

    h, w = inp.shape
    while placed < noise_count:
        success = False
        for _ in range(160):
            start = (int(np.random.randint(1, h - 1)), int(np.random.randint(1, w - 1)))
            if int(inp[start]) != BG:
                continue
            size = 2 if np.random.random() < 0.75 else 3
            cells = _random_noise_shape(inp, start, size)
            if cells is None or not _noise_separated(inp, cells, noise_color):
                continue
            _paint(inp, cells, noise_color)
            placed += 1
            success = True
            break
        if not success:
            return False
    return True


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, special_color: int) -> bool:
    try:
        actual, trace = _solve(inp, with_trace=True)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    noise = int(trace["noise"])
    counts = trace["component_counts"]
    others = [count for color, count in counts.items() if color != noise]
    if not others or counts[noise] <= max(others):
        return False

    erased = inp.copy()
    erased[erased == noise] = BG

    comps_before = _components_of_color(inp, special_color)
    comps_before.sort(key=_bbox_top_left)
    if len(comps_before) != 2:
        return False
    path_before = _best_l_path(comps_before[0], comps_before[1], inp, special_color)
    path_after = _best_l_path(comps_before[0], comps_before[1], erased, special_color)
    if path_before is not None or path_after is None:
        return False

    noise_cells = set(map(tuple, np.argwhere(inp == noise)))
    if not any(cell in noise_cells for cell in path_after):
        return False
    return True


def _generate_one() -> np.ndarray | None:
    h = int(np.random.randint(16, 19))
    w = int(np.random.randint(16, 19))
    output = np.full((h, w), BG, dtype=int)

    color_count = int(np.random.randint(3, 5))
    colors = list(np.random.choice(PALETTE, size=color_count, replace=False))
    noise_color = colors[0]
    keep_colors = colors[1:]
    special_color = keep_colors[0]

    special = _place_special_chain(output, special_color)
    if special is None:
        return None
    route_map: dict[int, list[Coord]] = {special_color: list(special["route_cells"])}

    extra_colors = keep_colors[1:]
    zones = _zone_templates(len(extra_colors), h, w)
    if len(zones) != len(extra_colors) or any(not _valid_zone(zone) for zone in zones):
        return None

    for idx, color in enumerate(extra_colors):
        zone = zones[idx]
        wants_three = len(extra_colors) >= 2 and idx == len(extra_colors) - 1 and np.random.random() < 0.6
        placement = _place_three_component_chain(output, color, zone) if wants_three else _place_two_component_chain(output, color, zone)
        if placement is None:
            return None
        route_map[color] = list(placement["route_cells"])

    wall_target = int(np.random.randint(3, 6))
    extra_walls = wall_target - 1
    if extra_walls < 2:
        extra_walls = 2
    if not _place_extra_walls(output, extra_walls):
        return None

    expected = output.copy()
    inp = _erase_paths(expected, route_map)

    low_counts = [len(_components_of_color(inp, color)) for color in keep_colors]
    if not low_counts:
        return None
    noise_count = max(low_counts) + int(np.random.randint(1, 3))
    noise_count = max(4, min(5, noise_count))
    if noise_count <= max(low_counts):
        return None

    if not _place_noise_components(inp, noise_color, special["route_cells"], noise_count):
        return None
    if not _semantic_ok(inp, expected, special_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
