from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
PERP = {
    "N": ("W", "E"),
    "S": ("W", "E"),
    "W": ("N", "S"),
    "E": ("N", "S"),
}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}

GRID_MIN = 15
GRID_MAX = 28
GENERATION_ATTEMPTS = 500
COMPONENT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 100


def _neighbors(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS.values():
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if grid[nr, nc] != BG and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _marker_pixels(grid: np.ndarray, component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    markers: list[tuple[int, int]] = []
    for r, c in component:
        neighbor_colors = [
            int(grid[nr, nc])
            for nr, nc in _neighbors(r, c, h, w)
            if grid[nr, nc] != BG
        ]
        if len(neighbor_colors) >= 3 and all(int(grid[r, c]) != color for color in neighbor_colors):
            markers.append((r, c))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group each connected pipe network and recolor it to the color of its unique marker junction.

    Input structure:
    - Black (0) is the background.
    - Non-black pixels form single-pixel-wide horizontal and vertical pipe segments.
    - Each connected network contains exactly one marked T- or cross-junction pixel whose
      color differs from all of its non-black 4-neighbors.

    Rule:
    - Find every 4-connected non-black component.
    - In each component, find the unique marked junction.
    - Recolor the entire component, including the marker itself, to the marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must stay in [0, 9]"

    out = g.copy()
    for component in _find_components(g):
        markers = _marker_pixels(g, component)
        assert len(markers) == 1, "each network must contain exactly one marker junction"
        mr, mc = markers[0]
        marker_color = int(g[mr, mc])
        rows, cols = zip(*component)
        out[np.array(rows), np.array(cols)] = marker_color
    return out


def _pick_color(rng: np.random.Generator, exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    return int(rng.choice(choices))


def _shuffled(rng: np.random.Generator, items: list) -> list:
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[index] for index in order]


def _same_color_neighbor(cells: dict[tuple[int, int], int], coord: tuple[int, int]) -> bool:
    color = cells[coord]
    r, c = coord
    for dr, dc in DIRS.values():
        nbr = (r + dr, c + dc)
        if cells.get(nbr) == color:
            return True
    return False


def _extend_path(
    cells: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: str,
    length: int,
    color: int,
) -> list[tuple[int, int]] | None:
    dr, dc = DIRS[direction]
    occupied = set(cells)
    path: list[tuple[int, int]] = []
    prev = start
    for step in range(1, length + 1):
        cur = (start[0] + dr * step, start[1] + dc * step)
        if cur in occupied:
            return None
        for nr, nc in ((cur[0] + ddr, cur[1] + ddc) for ddr, ddc in DIRS.values()):
            if (nr, nc) in occupied and (nr, nc) != prev:
                return None
        occupied.add(cur)
        path.append(cur)
        prev = cur
    for coord in path:
        cells[coord] = color
    return path


def _draw_polyline(
    cells: dict[tuple[int, int], int],
    start: tuple[int, int],
    first_dir: str,
    first_len: int,
    first_color: int,
    second_dir: str | None = None,
    second_len: int = 0,
    second_color: int | None = None,
) -> list[tuple[int, int]] | None:
    trial = dict(cells)
    first = _extend_path(trial, start, first_dir, first_len, first_color)
    if first is None:
        return None
    full = list(first)
    if second_dir is not None:
        assert second_color is not None
        second = _extend_path(trial, first[-1], second_dir, second_len, second_color)
        if second is None:
            return None
        full.extend(second)
    cells.clear()
    cells.update(trial)
    return full


def _random_polyline(
    rng: np.random.Generator,
    cells: dict[tuple[int, int], int],
    start: tuple[int, int],
    initial_dir: str,
    marker_color: int,
) -> bool:
    use_bend = bool(rng.integers(0, 2))
    first_len = int(rng.integers(2, 6))
    first_color = _pick_color(rng, {marker_color})

    options: list[tuple[str | None, int, int | None]]
    if use_bend:
        bend_dirs = list(PERP[initial_dir])
        rng.shuffle(bend_dirs)
        options = []
        for bend_dir in bend_dirs:
            second_len = int(rng.integers(2, 5))
            if rng.random() < 0.5:
                second_color = first_color
            else:
                second_color = _pick_color(rng, {marker_color, first_color})
            options.append((bend_dir, second_len, second_color))
        options.append((None, 0, None))
    else:
        options = [(None, 0, None)]

    for second_dir, second_len, second_color in options:
        drawn = _draw_polyline(
            cells,
            start,
            initial_dir,
            first_len,
            first_color,
            second_dir=second_dir,
            second_len=second_len,
            second_color=second_color,
        )
        if drawn is not None:
            return True
    return False


def _branch_anchor_candidates(cells: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    anchors: list[tuple[int, int]] = []
    for coord in cells:
        if coord == (0, 0):
            continue
        if _same_color_neighbor(cells, coord):
            anchors.append(coord)
    return anchors


def _branch_directions(
    rng: np.random.Generator,
    cells: dict[tuple[int, int], int],
    anchor: tuple[int, int],
) -> list[str]:
    occupied_dirs = [
        direction
        for direction, (dr, dc) in DIRS.items()
        if (anchor[0] + dr, anchor[1] + dc) in cells
    ]
    empty_dirs = [
        direction
        for direction, (dr, dc) in DIRS.items()
        if (anchor[0] + dr, anchor[1] + dc) not in cells
    ]
    preferred = [
        direction
        for direction in empty_dirs
        if any(direction in PERP[used] for used in occupied_dirs)
    ]
    fallback = [direction for direction in empty_dirs if direction not in preferred]
    if preferred:
        preferred = _shuffled(rng, preferred)
    if fallback:
        fallback = _shuffled(rng, fallback)
    return preferred + fallback


def _normalize_cells(
    cells: dict[tuple[int, int], int]
) -> tuple[dict[tuple[int, int], int], int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r_min = min(rows)
    c_min = min(cols)
    normalized = {(r - r_min, c - c_min): color for (r, c), color in cells.items()}
    height = max(r for r, _ in normalized) + 1
    width = max(c for _, c in normalized) + 1
    return normalized, height, width


def _component_to_grid(cells: dict[tuple[int, int], int]) -> np.ndarray:
    normalized, height, width = _normalize_cells(cells)
    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in normalized.items():
        grid[r, c] = color
    return grid


def _generate_component(rng: np.random.Generator) -> tuple[dict[tuple[int, int], int], int, int] | None:
    for _ in range(COMPONENT_ATTEMPTS):
        marker_color = _pick_color(rng, set())
        cells: dict[tuple[int, int], int] = {(0, 0): marker_color}

        if rng.random() < 0.5:
            root_dirs = ["N", "S", "W", "E"]
        else:
            missing = str(rng.choice(np.array(["N", "S", "W", "E"], dtype=object)))
            root_dirs = [direction for direction in ("N", "S", "W", "E") if direction != missing]
        root_dirs = _shuffled(rng, root_dirs)

        ok = True
        for direction in root_dirs:
            if not _random_polyline(rng, cells, (0, 0), direction, marker_color):
                ok = False
                break
        if not ok:
            continue

        extra_branches = int(rng.integers(0, 3))
        for _ in range(extra_branches):
            anchors = _branch_anchor_candidates(cells)
            if not anchors:
                break
            anchors = _shuffled(rng, anchors)
            placed = False
            for anchor in anchors:
                for direction in _branch_directions(rng, cells, anchor):
                    if _random_polyline(rng, cells, anchor, direction, marker_color):
                        placed = True
                        break
                if placed:
                    break

        local_grid = _component_to_grid(cells)
        try:
            transformed = main(local_grid)
        except AssertionError:
            continue
        if np.array_equal(local_grid, transformed):
            continue
        normalized, height, width = _normalize_cells(cells)
        return normalized, height, width
    return None


def _can_place(
    occupied: set[tuple[int, int]],
    coords: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
    for r, c in coords:
        for dr, dc in DIRS.values():
            nbr = (r + dr, c + dc)
            if nbr in occupied and nbr not in coords:
                return False
    return True


def _place_component(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    component: tuple[dict[tuple[int, int], int], int, int],
) -> bool:
    cells, height, width = component
    h, w = grid.shape
    if height > h or width > w:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = int(rng.integers(0, h - height + 1))
        left = int(rng.integers(0, w - width + 1))
        coords = {(top + r, left + c) for r, c in cells}
        if not _can_place(occupied, coords, h, w):
            continue
        for (r, c), color in cells.items():
            grid[top + r, left + c] = color
            occupied.add((top + r, left + c))
        return True
    return False


def _max_components_for_size(h: int, w: int) -> int:
    area = h * w
    if area < 260:
        return 2
    if area < 420:
        return 3
    return 4


def _validate_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    return len(_find_components(grid)) >= 1 and not np.array_equal(grid, out)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    component_count = int(rng.integers(1, _max_components_for_size(h, w) + 1))

    components: list[tuple[dict[tuple[int, int], int], int, int]] = []
    for _ in range(component_count):
        component = _generate_component(rng)
        if component is None:
            return None
        components.append(component)
    components.sort(key=lambda item: item[1] * item[2], reverse=True)

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for component in components:
        if not _place_component(rng, grid, occupied, component):
            return None

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
