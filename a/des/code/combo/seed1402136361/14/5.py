from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE: tuple[int, ...] = tuple(range(10))

# --------------- solver ---------------


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bar_dirs(stem: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = stem
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _t_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar_a, bar_b = _bar_dirs(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar_a[0], c + bar_a[1]),
        (r + bar_b[0], c + bar_b[1]),
    }


def _same_color_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append(coords)
    return components


def _parse_t(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cell_set = set(coords)
    if len(cell_set) != 4:
        return None

    neighbor_dirs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cell_set:
        dirs: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in cell_set:
                dirs.append((dr, dc))
        neighbor_dirs[(r, c)] = dirs

    centers = [cell for cell, dirs in neighbor_dirs.items() if len(dirs) == 3]
    if len(centers) != 1:
        return None
    center = centers[0]

    dirs = set(neighbor_dirs[center])
    if any(len(neighbor_dirs[cell]) != 1 for cell in cell_set if cell != center):
        return None

    if (0, -1) in dirs and (0, 1) in dirs:
        if (-1, 0) in dirs:
            return center, (-1, 0)
        if (1, 0) in dirs:
            return center, (1, 0)
        return None
    if (-1, 0) in dirs and (1, 0) in dirs:
        if (0, -1) in dirs:
            return center, (0, -1)
        if (0, 1) in dirs:
            return center, (0, 1)
        return None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Interpret each foreground object as a discrete T: a 3-cell bar plus a 1-cell stem.
    The stem direction is the handle, so the ray is emitted from the opposite side of
    the bar midpoint and extends to the grid edge, only filling background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    out = g.copy()

    components = _same_color_components(g, bg)
    assert components, "expected at least one foreground T object"

    h, w = map(int, g.shape)
    for coords in components:
        parsed = _parse_t(coords)
        assert parsed is not None, "every foreground component must be a T object"
        center, stem = parsed
        color = int(g[coords[0]])

        front = (-stem[0], -stem[1])
        r = center[0] + front[0]
        c = center[1] + front[1]
        while 0 <= r < h and 0 <= c < w:
            if int(out[r, c]) == bg:
                out[r, c] = color
            r += front[0]
            c += front[1]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (9, 18)
GRID_W_RANGE = (9, 18)
OBJECT_COUNT_RANGE = (2, 5)
MIN_RAY_LEN = 2
GENERATE_RETRIES = 400


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _ray_cells(center: tuple[int, int], stem: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    front = (-stem[0], -stem[1])
    r = center[0] + front[0]
    c = center[1] + front[1]
    ray: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        ray.append((r, c))
        r += front[0]
        c += front[1]
    return ray


def _neighbors8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _sample_directions(count: int) -> list[tuple[int, int]]:
    base = list(DIRS4)
    random.shuffle(base)
    if count <= len(base):
        return base[:count]
    extra = [random.choice(base) for _ in range(count - len(base))]
    out = base + extra
    random.shuffle(out)
    return out


def _try_place_objects(
    h: int, w: int, bg: int, colors: list[int], stems: list[tuple[int, int]]
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((h, w), bg, dtype=int)
    object_buffer: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    for color, stem in zip(colors, stems):
        candidates: list[tuple[tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]]] = []
        for r in range(h):
            for c in range(w):
                center = (r, c)
                cells = _t_cells(center, stem)
                if not _in_bounds(cells, h, w):
                    continue
                ray = _ray_cells(center, stem, h, w)
                if len(ray) < MIN_RAY_LEN:
                    continue
                exact = cells | set(ray)
                if any(cell in object_buffer or cell in occupied for cell in cells):
                    continue
                if any(cell in object_buffer or cell in occupied for cell in ray):
                    continue
                candidates.append((center, cells, ray))

        if not candidates:
            return None

        center, cells, ray = random.choice(candidates)
        for r, c in cells:
            grid[r, c] = color
        object_buffer |= _neighbors8(cells, h, w)
        occupied |= cells | set(ray)
        placed.append({"color": color, "center": center, "stem": stem, "ray": tuple(ray)})

    return grid, placed


def _validate_generated(grid: np.ndarray, placed: list[dict[str, object]], bg: int) -> bool:
    if np.count_nonzero(grid != bg) != 4 * len(placed):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    changed = int(np.count_nonzero(out != grid))
    if changed < MIN_RAY_LEN * len(placed):
        return False

    for spec in placed:
        color = int(spec["color"])
        ray = list(spec["ray"])
        if len(ray) < MIN_RAY_LEN:
            return False
        if any(int(grid[r, c]) != bg for r, c in ray):
            return False
        if any(int(out[r, c]) != color for r, c in ray):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(PALETTE)

    max_objects = min(OBJECT_COUNT_RANGE[1], len(PALETTE) - 1)
    count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    colors = random.sample([c for c in PALETTE if c != bg], count)
    stems = _sample_directions(count)

    built = _try_place_objects(h, w, bg, colors, stems)
    if built is None:
        return None
    grid, placed = built

    unique_dirs = {tuple(spec["stem"]) for spec in placed}
    if len(unique_dirs) < min(2, count):
        return None

    return grid if _validate_generated(grid, placed, bg) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid T-ray input")
