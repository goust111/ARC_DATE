from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    assert flat.size > 0, "expected at least one color sample"
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors(coord: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = coord
    h, w = shape
    out: list[Coord] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in _neighbors(cur, mask.shape):
                    rr, cc = nxt
                    if bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append(nxt)
            comps.append(comp)
    return comps


def _trace_path(component: list[Coord], shape: tuple[int, int]) -> tuple[list[Coord], dict[Coord, list[Coord]]]:
    coord_set = set(component)
    neighbor_map: dict[Coord, list[Coord]] = {}
    endpoints: list[Coord] = []

    if len(component) == 1:
        cell = component[0]
        return [cell], {cell: []}

    for coord in component:
        neigh = [nxt for nxt in _neighbors(coord, shape) if nxt in coord_set]
        degree = len(neigh)
        assert degree in (1, 2), "path components must be simple non-branching paths"
        if degree == 1:
            endpoints.append(coord)
        neighbor_map[coord] = neigh

    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    start = min(endpoints)
    ordered = [start]
    prev: Coord | None = None
    cur = start

    while True:
        choices = [nxt for nxt in neighbor_map[cur] if nxt != prev]
        if not choices:
            break
        assert len(choices) == 1, "simple path traversal should be unique"
        prev, cur = cur, choices[0]
        ordered.append(cur)

    assert len(ordered) == len(component), "traversal must visit every path pixel exactly once"
    return ordered, neighbor_map


def _is_straight(coord: Coord, neighbors: list[Coord]) -> bool:
    if len(neighbors) != 2:
        return False
    return neighbors[0][0] + neighbors[1][0] == 2 * coord[0] and neighbors[0][1] + neighbors[1][1] == 2 * coord[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path segment using the most recent colored marker encountered
    while traversing the path from its lexicographically earliest endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    path_mask = g != bg
    assert bool(path_mask.any()), "expected at least one path pixel"

    components = _components(path_mask)
    traced: list[list[Coord]] = []
    straight_colors: list[int] = []

    for comp in components:
        ordered, neighbor_map = _trace_path(comp, g.shape)
        traced.append(ordered)
        for coord, neigh in neighbor_map.items():
            if _is_straight(coord, neigh):
                straight_colors.append(int(g[coord]))

    if straight_colors:
        path_color = _most_common_color(np.array(straight_colors, dtype=int))
    else:
        path_color = _most_common_color(g[path_mask])
    assert path_color != bg, "path color must differ from the background"

    out = g.copy()
    for ordered in traced:
        active_color = path_color
        for coord in ordered:
            original = int(g[coord])
            if original != path_color:
                active_color = original
            out[coord] = active_color

    return out


# --------------- generator ---------------

GREY = 5
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
MARKER_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
SEGMENT_LENGTH_RANGE = (2, 5)
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 160


def _perpendicular_dirs(direction: Coord) -> list[Coord]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _halo(cells: list[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _can_add_cell(cell: Coord, prev: Coord, used: set[Coord], blocked: set[Coord], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in used or cell in blocked:
        return False
    for neigh in _neighbors(cell, shape):
        if neigh in used and neigh != prev:
            return False
    return True


def _build_path(
    shape: tuple[int, int],
    blocked: set[Coord],
    min_segments: int,
    max_segments: int,
) -> list[Coord] | None:
    h, w = shape
    starts = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    if not starts:
        return None

    for _ in range(PATH_ATTEMPTS):
        start = random.choice(starts)
        segment_count = random.randint(min_segments, max_segments)
        first_dir = random.choice(DIR4)
        path = [start]
        used = {start}
        prev_dir = first_dir
        ok = True

        for seg_idx in range(segment_count):
            dir_options = [first_dir] if seg_idx == 0 else _perpendicular_dirs(prev_dir)
            random.shuffle(dir_options)
            placed = False
            for direction in dir_options:
                lengths = list(range(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1))
                random.shuffle(lengths)
                for length in lengths:
                    cur = path[-1]
                    new_cells: list[Coord] = []
                    good = True
                    for _ in range(length):
                        nxt = (cur[0] + direction[0], cur[1] + direction[1])
                        if not _can_add_cell(nxt, cur, used, blocked, shape):
                            good = False
                            break
                        new_cells.append(nxt)
                        cur = nxt
                    if not good:
                        continue
                    path.extend(new_cells)
                    used.update(new_cells)
                    prev_dir = direction
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if ok:
            return path
    return None


def _turn_indices(path: list[Coord]) -> list[int]:
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return [i for i in range(1, len(path) - 1) if dirs[i - 1] != dirs[i]]


def _choose_markers(path: list[Coord], bg: int, ensure_corner_marker: bool) -> dict[int, int] | None:
    turns = _turn_indices(path)
    if ensure_corner_marker and not turns:
        return None

    start_colored = random.choice([False, True])
    end_colored = random.choice([False, True])
    markers: set[int] = set()

    if turns:
        min_corner_count = 1 if ensure_corner_marker else 0
        corner_count = random.randint(min_corner_count, len(turns))
        if corner_count:
            markers.update(random.sample(turns, corner_count))

    if start_colored:
        markers.add(0)
    if end_colored:
        markers.add(len(path) - 1)

    if not markers:
        choices = turns if turns else [0, len(path) - 1]
        markers.add(random.choice(choices))

    if markers == {len(path) - 1}:
        if turns:
            markers.add(random.choice(turns))
        else:
            markers.add(0)

    palette = [color for color in MARKER_CHOICES if color not in (bg, GREY)]
    ordered_markers = sorted(markers)
    marker_colors: dict[int, int] = {}
    previous_color: int | None = None
    for idx in ordered_markers:
        choices = [color for color in palette if color != previous_color]
        color = random.choice(choices if choices else palette)
        marker_colors[idx] = color
        previous_color = color
    return marker_colors


def _paint_path(grid: np.ndarray, path: list[Coord], marker_colors: dict[int, int]) -> None:
    for i, coord in enumerate(path):
        grid[coord] = marker_colors.get(i, GREY)


def _max_paths(shape: tuple[int, int]) -> int:
    h, w = shape
    area = h * w
    if min(h, w) < 12:
        return 1
    if area < 220:
        return 2
    return 3


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    bg = random.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    path_count = random.randint(1, _max_paths((h, w)))
    blocked: set[Coord] = set()
    paths: list[list[Coord]] = []
    marker_specs: list[dict[int, int]] = []

    first_path = _build_path((h, w), blocked, min_segments=2, max_segments=5)
    if first_path is None:
        return None
    first_markers = _choose_markers(first_path, bg, ensure_corner_marker=True)
    if first_markers is None:
        return None
    paths.append(first_path)
    marker_specs.append(first_markers)
    blocked |= _halo(first_path, (h, w))

    for _ in range(path_count - 1):
        min_segments = random.randint(1, 2)
        max_segments = random.randint(max(min_segments, 2), 4)
        path = _build_path((h, w), blocked, min_segments=min_segments, max_segments=max_segments)
        if path is None:
            return None
        markers = _choose_markers(path, bg, ensure_corner_marker=False)
        if markers is None:
            return None
        paths.append(path)
        marker_specs.append(markers)
        blocked |= _halo(path, (h, w))

    for path, markers in zip(paths, marker_specs):
        _paint_path(grid, path, markers)

    if not bool(np.any(grid == GREY)):
        return None

    expected = main(grid)
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid segmented-path input")
