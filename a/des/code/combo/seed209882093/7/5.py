import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 10
GRID_MAX = 18
MAX_PATHS = 4
PATH_LEN_MIN = 6
PATH_LEN_MAX = 22
MAX_STATIONS = 4
GEN_ATTEMPTS = 400
PATH_SHAPE_ATTEMPTS = 60
PATH_PLACE_ATTEMPTS = 60


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _neighbors8(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component(start: tuple[int, int], mask: np.ndarray, seen: np.ndarray) -> set[tuple[int, int]]:
    q = deque([start])
    seen[start] = True
    coords: set[tuple[int, int]] = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, mask.shape):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                coords.add((nr, nc))
                q.append((nr, nc))
    return coords


def _ordered_path(component: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    degree = {
        cell: sum((nr, nc) in component for nr, nc in _neighbors4(cell[0], cell[1], shape))
        for cell in component
    }
    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "each path component must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degree.values()), "path component cannot branch"

    start = min(endpoints)
    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [nbr for nbr in _neighbors4(cur[0], cur[1], shape) if nbr in component and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "ordered traversal requires a single forward neighbor"
        prev, cur = cur, nxt[0]
        path.append(cur)
    assert len(path) == len(component), "path traversal must cover the whole component"
    return path


def _station_map(path: list[tuple[int, int]], grid: np.ndarray) -> dict[int, int]:
    endpoint_color = int(grid[path[0]])
    assert int(grid[path[-1]]) == endpoint_color, "paired endpoints must share a color"

    stations: dict[int, int] = {}
    for idx in range(1, len(path) - 1):
        color = int(grid[path[idx]])
        if color == GREY:
            continue
        assert color != endpoint_color, "interior non-grey pixels must be dye stations"
        stations[idx] = color
    assert stations, "each path must contain at least one dye station"
    return stations


def _recolor_component(path: list[tuple[int, int]], grid: np.ndarray, out: np.ndarray) -> None:
    stations = _station_map(path, grid)

    nearest_left: list[int | None] = [None] * len(path)
    nearest_right: list[int | None] = [None] * len(path)

    last = None
    for idx in range(len(path)):
        if idx in stations:
            last = idx
        nearest_left[idx] = last

    last = None
    for idx in range(len(path) - 1, -1, -1):
        if idx in stations:
            last = idx
        nearest_right[idx] = last

    for idx in range(1, len(path) - 1):
        r, c = path[idx]
        if int(grid[r, c]) != GREY:
            continue

        left_idx = nearest_left[idx]
        right_idx = nearest_right[idx]
        assert left_idx is not None or right_idx is not None, "grey cell must have a station on at least one side"

        if left_idx is None:
            color = stations[right_idx]  # type: ignore[index]
        elif right_idx is None:
            color = stations[left_idx]
        else:
            left_dist = idx - left_idx
            right_dist = right_idx - idx
            color = stations[left_idx] if left_dist <= right_dist else stations[right_idx]
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey path segment by the nearest dye station while keeping endpoints and stations fixed.

    Input structure:
    - Black background.
    - Each foreground component is a single 4-connected one-pixel-wide path.
    - The two endpoints of a path share one color; interior non-grey cells are dye stations.

    Rule:
    - Walk every path from its lexicographically first endpoint to the other endpoint.
    - Grey cells are recolored by the nearest dye station along the path.
    - Ties go to the station on the start-side of the segment.
    - Endpoint markers and station pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    mask = g != BG
    seen = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        comp = _component((rr, cc), mask, seen)
        path = _ordered_path(comp, g.shape)
        _recolor_component(path, g, out)

    return out


def _turn_dirs(prev_dir: tuple[int, int] | None, rng: random.Random) -> list[tuple[int, int]]:
    if prev_dir is None:
        dirs = list(DIRS4)
    else:
        dr, dc = prev_dir
        dirs = [(-dc, dr), (dc, -dr)]
        if rng.random() < 0.25:
            dirs.append(prev_dir)
    rng.shuffle(dirs)
    return dirs


def _extend_segment(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
    box_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < box_shape[0] and 0 <= nc < box_shape[1]):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors8(nr, nc, box_shape)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _sample_composition(total: int, count: int, lower: int, upper: int, rng: random.Random) -> list[int] | None:
    if count * lower > total or count * upper < total:
        return None

    parts: list[int] = []
    remaining = total
    for idx in range(count):
        slots = count - idx - 1
        lo = max(lower, remaining - upper * slots)
        hi = min(upper, remaining - lower * slots)
        if lo > hi:
            return None
        value = rng.randint(lo, hi)
        parts.append(value)
        remaining -= value
    return parts


def _build_path_shape(rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(PATH_SHAPE_ATTEMPTS):
        box_h = rng.randint(4, 9)
        box_w = rng.randint(4, 9)
        max_run = max(1, min(6, max(box_h, box_w) - 1))
        total_steps = rng.randint(PATH_LEN_MIN - 1, PATH_LEN_MAX - 1)
        segment_count = rng.randint(1, min(6, total_steps))
        run_lengths = _sample_composition(total_steps, segment_count, 1, max_run, rng)
        if run_lengths is None:
            continue

        starts = [(r, c) for r in range(box_h) for c in range(box_w)]
        rng.shuffle(starts)

        def search(
            seg_idx: int,
            prev_dir: tuple[int, int] | None,
            path: list[tuple[int, int]],
            used: set[tuple[int, int]],
        ) -> list[tuple[int, int]] | None:
            if seg_idx == len(run_lengths):
                return path
            for direction in _turn_dirs(prev_dir, rng):
                extended = _extend_segment(path, used, direction, run_lengths[seg_idx], (box_h, box_w))
                if extended is None:
                    continue
                found = search(seg_idx + 1, direction, extended[0], extended[1])
                if found is not None:
                    return found
            return None

        for start in starts:
            found = search(0, None, [start], {start})
            if found is None:
                continue
            min_r = min(r for r, _ in found)
            min_c = min(c for _, c in found)
            normalized = [(r - min_r, c - min_c) for r, c in found]
            if normalized[0] > normalized[-1]:
                normalized = list(reversed(normalized))
            return normalized
    return None


def _sample_station_indices(path_len: int, rng: random.Random) -> list[int] | None:
    interior = path_len - 2
    if interior < 2:
        return None

    max_stations = min(MAX_STATIONS, interior - 1, len(PALETTE) - 1)
    if max_stations < 1:
        return None

    if max_stations == 1:
        station_count = 1
    else:
        station_count = rng.choice([1, 1, 2, 2, 3] + ([4] if max_stations >= 4 else []))
        station_count = min(station_count, max_stations)

    grey_total = interior - station_count
    if grey_total < 1:
        return None

    gaps = [0] * (station_count + 1)
    remaining = grey_total

    if station_count >= 2 and remaining >= 1 and rng.random() < 0.45:
        gap_idx = rng.randint(1, station_count - 1)
        odd_choices = [v for v in range(1, remaining + 1, 2)]
        forced = rng.choice(odd_choices)
        gaps[gap_idx] = forced
        remaining -= forced

    for _ in range(remaining):
        gaps[rng.randrange(len(gaps))] += 1

    indices = []
    idx = 1 + gaps[0]
    indices.append(idx)
    for gap in gaps[1:-1]:
        idx += 1 + gap
        indices.append(idx)
    return indices


def _color_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    endpoint_color: int,
    station_indices: list[int],
    station_colors: list[int],
) -> None:
    station_map = dict(zip(station_indices, station_colors))
    for idx, (r, c) in enumerate(path):
        if idx == 0 or idx == len(path) - 1:
            grid[r, c] = endpoint_color
        elif idx in station_map:
            grid[r, c] = station_map[idx]
        else:
            grid[r, c] = GREY


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        path = spec["path"]
        endpoint_color = spec["endpoint_color"]
        station_indices = spec["station_indices"]
        station_colors = spec["station_colors"]
        station_map = dict(zip(station_indices, station_colors))

        for idx, (r, c) in enumerate(path):
            if idx == 0 or idx == len(path) - 1:
                out[r, c] = endpoint_color
            elif idx in station_map:
                out[r, c] = station_map[idx]
            else:
                left = max((s for s in station_indices if s < idx), default=None)
                right = min((s for s in station_indices if s > idx), default=None)
                if left is None:
                    out[r, c] = station_map[right]  # type: ignore[index]
                elif right is None:
                    out[r, c] = station_map[left]
                else:
                    out[r, c] = station_map[left] if idx - left <= right - idx else station_map[right]
    return out


def _placement_cells(path: list[tuple[int, int]], offset: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = offset
    return [(r + r0, c + c0) for r, c in path]


def _fits_path(path: list[tuple[int, int]], offset: tuple[int, int], reserved: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    cells = _placement_cells(path, offset)
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in reserved:
            return False
    return True


def _reserve_buffer(cells: list[tuple[int, int]], reserved: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    for r, c in cells:
        reserved.add((r, c))
        for nr, nc in _neighbors8(r, c, shape):
            reserved.add((nr, nc))


def _sample_path_count(area: int, rng: random.Random) -> int:
    if area < 140:
        max_paths = 2
    elif area < 220:
        max_paths = 3
    else:
        max_paths = MAX_PATHS
    return rng.randint(1, max_paths)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    num_paths = _sample_path_count(h * w, rng)
    endpoint_colors = rng.sample(PALETTE, num_paths)

    reserved: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for endpoint_color in endpoint_colors:
        placed = False
        for _ in range(PATH_PLACE_ATTEMPTS):
            path_shape = _build_path_shape(rng)
            if path_shape is None:
                continue

            path_len = len(path_shape)
            station_indices = _sample_station_indices(path_len, rng)
            if station_indices is None:
                continue

            station_palette = [color for color in PALETTE if color != endpoint_color]
            station_colors = rng.sample(station_palette, len(station_indices))

            max_r = max(r for r, _ in path_shape)
            max_c = max(c for _, c in path_shape)
            offsets = [
                (r0, c0)
                for r0 in range(h - max_r)
                for c0 in range(w - max_c)
                if _fits_path(path_shape, (r0, c0), reserved, (h, w))
            ]
            if not offsets:
                continue

            offset = rng.choice(offsets)
            placed_path = _placement_cells(path_shape, offset)
            if placed_path[0] > placed_path[-1]:
                placed_path = list(reversed(placed_path))

            specs.append(
                {
                    "path": placed_path,
                    "endpoint_color": endpoint_color,
                    "station_indices": station_indices,
                    "station_colors": station_colors,
                }
            )
            _reserve_buffer(placed_path, reserved, (h, w))
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        _color_path(
            grid,
            spec["path"],
            spec["endpoint_color"],
            spec["station_indices"],
            spec["station_colors"],
        )

    expected = _expected_from_specs((h, w), specs)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
